"""GRPO — Group Relative Policy Optimization (Phase 13): the RL half of canonical RLHF.

Per iteration: sample P prompts, roll out G completions per prompt from the live policy
(batched KV-cached sampling, one generate call per group), score every completion with a
frozen reward model, standardize rewards WITHIN each group into advantages (the group IS the
baseline — no learned value model, DeepSeekMath's simplification of PPO), then take ONE
clipped token-level policy-gradient step over the P*G sequences with an explicit k3 KL
penalty to the frozen SFT reference added to the LOSS (not folded into the reward).

Design decisions, documented for the curriculum:

- Trained sequences are REBUILT from the truncated response TEXT + "\\n### End" via
  build_sft_example (the SFT/DPO construction), not the raw sampled token ids. The reward
  model only ever sees the truncated text (its training-time construction re-encodes it), so
  training on post-stop sampled tokens would attribute reward to tokens the RM never scored;
  rebuilding also supervises the model on EMITTING the sentinel, matching sft.py/dpo.py's
  definition of a complete response. Re-encoding can occasionally re-segment a BPE boundary,
  which is why logp_old comes from rescoring (below), not from sampling-time logits.

- logp_old is captured by a no-grad RESCORING pass over the rebuilt sequences, split into the
  SAME micro-batches and run through the SAME forward path (autocast policy, fp32
  log_softmax) as the training pass — ``forward_logps`` is the single numeric path for
  logp_old capture, reference log-probs, the training forward, and the post-step shift
  measurement. The sampling pass instead runs the KV-cache incremental path (different kernel
  shapes), whose logits would bias step-0 ratios away from 1. With rescoring, ratio == 1 at
  the first update (asserted in tests), so clipping is a safety net, not a bias source; and
  because each iteration takes exactly ONE optimizer step over its rollouts, the clip
  fraction should sit near zero for the whole run (it guards a future multi-epoch variant).

- lr default 1e-6: RL fine-tuning moves a policy with per-token gradients scaled by O(1)
  advantages on EVERY response token — orders of magnitude more aggressive per step than a
  cross-entropy nudge — and reward-model exploitation compounds over hundreds of iterations,
  so RLHF learning rates run ~10-50x below SFT ones. Warmup is linear over the first
  ``warmup_iters`` iterations, then CONSTANT: RL has no fixed data horizon to cosine toward.

Resume: with an existing out_dir holding ckpt_*.pt, the latest checkpoint's policy weights +
AdamW state are restored and the run continues at its iteration + 1. Restored: weights,
optimizer moments, the iteration counter (and therefore the LR schedule and the prompt/rollout
seeds — both are pure functions of (seed, iteration), so a resumed run draws exactly what an
uninterrupted one would; there is no RNG state to persist). NOT restored: log files are
appended, never rewound — if a crash landed between a log append and the next checkpoint,
grpo_log.jsonl / samples.jsonl can carry lines past the checkpoint; every line has an "iter"
field, so downstream readers keep the last line per iteration.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from microlab.infer.reference.kv_cache import generate_cached
from microlab.model.reference.sft import IGNORE_INDEX, build_sft_example, collate_sft

# Same sentinel sft.py trains, dpo.py optimizes, and train_reward_model.py scores after —
# scripts/train_grpo.py raises at import if the constants ever drift apart.
END_SENTINEL = "\n### End"
PAD_ID = 0  # matches collate_sft's default and the tokenizer convention here
# The SFT stop sentinels; truncate_at_stops must match build_rlaif_candidates.truncate
# (pinned by a test).
STOP_STRINGS = ["### End", "\n### Instruction:"]
# Written at the end so the run dir serves as a chat model, exactly like sft.py/dpo.py.
SERVE_CONFIG = {"mode": "chat", "stop_strings": ["### End", "\n### Instruction:"]}


# ---------------------------------------------------------------- pure pieces


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Standardize rewards WITHIN each group: (r - mean_g) / (std_g + 1e-8), population std.
    Input (n_groups, group_size), output the same shape. A zero-std group (every rollout
    scored identically) carries no ranking signal and contributes EXACTLY zero advantage —
    explicitly zeroed rather than trusting float cancellation, and never NaN."""
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be (n_groups, group_size), got {tuple(rewards.shape)}")
    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, correction=0)
    adv = (rewards - mean) / (std + 1e-8)
    return torch.where(std > 0, adv, torch.zeros_like(adv))


def per_token_logps(logits: torch.Tensor, labels: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of ``labels`` under ``logits`` at supervised positions. Causal
    shift: logits[:, :-1] predict labels[:, 1:]. Returns (logps, mask), both (B, T-1); logps
    are EXACTLY zero at masked (prompt/pad) positions so sums need no re-masking. log_softmax
    runs in fp32 regardless of autocast — the shared numeric path for logp_old capture,
    reference log-probs, and the training forward, so ratios can't pick up a dtype bias."""
    if logits.dim() != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs labels "
                         f"{tuple(labels.shape)}")
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != IGNORE_INDEX
    logp = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return gathered * mask, mask


def clipped_surrogate(logp_new: torch.Tensor, logp_old: torch.Tensor,
                      advantages: torch.Tensor, mask: torch.Tensor,
                      clip_eps: float) -> tuple[torch.Tensor, int]:
    """PPO-style token-level clipped surrogate. ratio = exp(logp_new - logp_old) per token;
    the per-sequence advantage broadcasts over its response tokens. Returns (surrogate_sum,
    n_clipped): the SUM over response tokens of min(ratio*A, clip(ratio, 1-eps, 1+eps)*A) —
    the caller divides by the GLOBAL response-token count, so gradient accumulation
    reproduces the exact global token mean — and the count of response tokens with
    |ratio - 1| > clip_eps (the standard clip-fraction numerator)."""
    if logp_new.shape != logp_old.shape or logp_new.shape != mask.shape:
        raise ValueError(f"shape mismatch: logp_new {tuple(logp_new.shape)}, logp_old "
                         f"{tuple(logp_old.shape)}, mask {tuple(mask.shape)}")
    if advantages.shape != (logp_new.shape[0],):
        raise ValueError(f"advantages must be ({logp_new.shape[0]},), got "
                         f"{tuple(advantages.shape)}")
    ratio = torch.exp(logp_new - logp_old.detach())
    adv = advantages.unsqueeze(1)
    surr = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    surr_sum = (surr * mask).sum()
    n_clipped = int((((ratio - 1.0).abs() > clip_eps) & mask).sum().item())
    return surr_sum, n_clipped


def k3_kl(ref_logp: torch.Tensor, new_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Schulman's k3 KL estimator, SUMMED over response tokens: exp(d) - d - 1 with
    d = ref_logp - new_logp. Pointwise >= 0 (e^x - x - 1 >= 0 for all x) and exactly 0 when
    the models agree, unlike the naive (new - ref) sample estimator which is unbiased only in
    expectation and can go negative per token. The caller divides by the global token count
    and scales by beta."""
    if ref_logp.shape != new_logp.shape or ref_logp.shape != mask.shape:
        raise ValueError(f"shape mismatch: ref {tuple(ref_logp.shape)} vs new "
                         f"{tuple(new_logp.shape)} vs mask {tuple(mask.shape)}")
    d = ref_logp.detach() - new_logp
    return ((torch.exp(d) - d - 1.0) * mask).sum()


def grpo_lr(iteration: int, warmup_iters: int, base_lr: float) -> float:
    """Linear warmup over the first ``warmup_iters`` iterations (1-based), then CONSTANT —
    no cosine: RL has no fixed data horizon to decay toward, and a decaying lr would
    confound reward curves with schedule effects."""
    if warmup_iters > 0 and iteration <= warmup_iters:
        return base_lr * iteration / warmup_iters
    return base_lr


def iteration_prompt_indices(seed: int, iteration: int, pool_size: int, n: int) -> list[int]:
    """The n DISTINCT pool indices for ``iteration`` — a pure function of (seed, iteration,
    pool_size, n), so a restarted run draws exactly the prompts an uninterrupted one would
    (resume determinism) with no RNG state to persist."""
    if n > pool_size:
        raise ValueError(f"prompts_per_iter {n} > pool size {pool_size}")
    gen = torch.Generator().manual_seed(seed * 1_000_003 + iteration)
    return torch.randperm(pool_size, generator=gen)[:n].tolist()


def truncate_at_stops(text: str) -> str:
    """Cut a generation at the earliest SFT stop sentinel, then strip — the same rule as
    build_rlaif_candidates.truncate (pinned by a test)."""
    cut = min((text.find(s) for s in STOP_STRINGS if s in text), default=-1)
    return (text[:cut] if cut >= 0 else text).strip()


def repetition_score(texts: list[str]) -> float:
    """Fraction of repeated 4-grams across completions (0 = all distinct, higher = loopier) —
    the same metric as scripts/track_probes.py repetition_score (pinned by a test)."""
    dup = tot = 0
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[k:k + 4]) for k in range(len(toks) - 3)]
        if not grams:
            continue
        tot += len(grams)
        dup += len(grams) - len(set(grams))
    return round(dup / tot, 3) if tot else 0.0


def distinct_fraction(groups: list[list[str]]) -> float:
    """Mean over groups of (distinct rollouts / rollouts): 1.0 = every rollout unique,
    1/G = the group collapsed to a single completion (diversity death, the RL failure mode
    where advantages go to zero and learning stalls)."""
    if not groups:
        raise ValueError("distinct_fraction got no groups")
    return sum(len(set(g)) / len(g) for g in groups) / len(groups)


# ---------------------------------------------------------------- model-facing pieces


def forward_logps(model, input_ids: torch.Tensor, labels: torch.Tensor,
                  use_amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """One full-sequence forward -> per-token response log-probs. THE single numeric path
    for logp_old capture (caller wraps in no_grad), reference log-probs (no_grad), the
    training forward (grad), and the post-step shift measurement — the caller chooses the
    grad context; this function guarantees the math inside is identical."""
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)
    else:
        logits, _ = model(input_ids)
    return per_token_logps(logits, labels)


def sample_group(model, prompt_ids: list[int], device: str, group_size: int, temp: float,
                 max_new: int, seed: int, use_amp: bool) -> list[list[int]]:
    """G rollouts of ONE prompt in a single batched generate_cached call — the same trick as
    build_rlaif_candidates.sample_candidates: the rows are the prompt tiled (no padding
    needed) and batched multinomial gives each row an independent draw off the shared seeded
    generator. Returns raw response TOKEN IDS per rollout (the caller decodes + truncates).
    Restores the model's training mode afterwards (generate_cached flips it to eval)."""
    was_training = model.training
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device).repeat(group_size, 1)
    gen = torch.Generator(device=device).manual_seed(seed)
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp
           else contextlib.nullcontext())
    with ctx:
        seqs = generate_cached(model, ids, max_new, temperature=temp, generator=gen)
    if was_training:
        model.train()
    return [row[len(prompt_ids):] for row in seqs.tolist()]


# ---------------------------------------------------------------- the training loop


def run_grpo(policy, reference, tok, prompts: list[str], score_texts, out: str | Path,
             tokenizer_path: str | Path, *, iters: int, prompts_per_iter: int = 8,
             group_size: int = 8, lr: float = 1e-6, beta: float = 0.04,
             clip_eps: float = 0.2, temp: float = 0.8, max_new: int = 80,
             micro_batch: int = 8, save_every: int = 25, dump_every: int = 10,
             seed: int = 1337, device: str = "cpu", warmup_iters: int = 10) -> dict:
    """Run GRPO and write a servable chat run dir (ckpt + tokenizer.json + serve_config.json
    at the end, rolling --save-every checkpoints during). ``score_texts(prompt, texts) ->
    list[float]`` is the frozen reward oracle (injected so tests can rig it and the script
    can pin the RM's exact training-time sequence construction). ``policy`` and ``reference``
    must already sit on ``device``; the reference is frozen here. Appends one JSON line per
    iteration to grpo_log.jsonl and 3 sample generations to samples.jsonl at iteration 1 and
    every --dump-every after. Resumable — see the module docstring for exactly what resume
    does and does not restore. Returns {"iters", "ckpt_path", "out_dir", "log_path"}."""
    if group_size < 2:
        raise ValueError(f"group_size must be >= 2 (the group is the baseline), got {group_size}")
    if prompts_per_iter < 1 or micro_batch < 1 or iters < 1:
        raise ValueError(f"prompts_per_iter, micro_batch, iters must be >= 1, got "
                         f"{prompts_per_iter}, {micro_batch}, {iters}")
    if len(prompts) < prompts_per_iter:
        raise ValueError(f"prompt pool ({len(prompts)}) smaller than prompts_per_iter "
                         f"({prompts_per_iter})")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.startswith("cuda")
    block_size = policy.config.block_size

    policy.train()
    reference.eval()
    reference.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)

    start_iter = 1
    existing = sorted(out_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if existing:
        ckpt = torch.load(existing[-1], map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["model"])  # copies into the existing device tensors
        opt.load_state_dict(ckpt["optimizer"])  # state is cast to the params' device
        start_iter = ckpt["step"] + 1
        print(f"resuming from {existing[-1].name}: continuing at iteration {start_iter}",
              flush=True)
    if start_iter > iters:
        raise ValueError(f"{out_dir} already has {start_iter - 1} iterations — nothing to do "
                         f"for iters={iters} (raise --iters to continue the run)")

    log_path = out_dir / "grpo_log.jsonl"
    samples_path = out_dir / "samples.jsonl"

    def append_jsonl(path: Path, record: dict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def save_ckpt(tag: int) -> Path:
        path = out_dir / f"ckpt_{tag}.pt"
        torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict(),
                    "step": tag, "cfg": policy.config}, path)
        return path

    last_rolling: Path | None = None
    n_seq = prompts_per_iter * group_size
    for it in range(start_iter, iters + 1):
        t_it = time.time()
        cur_lr = grpo_lr(it, warmup_iters, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        # --- rollouts: P prompts x G KV-cached samples, decoded + truncated at chat stops
        pidx = iteration_prompt_indices(seed, it, len(prompts), prompts_per_iter)
        groups_texts: list[list[str]] = []
        reward_rows: list[list[float]] = []
        examples: list[tuple[list[int], list[int]]] = []
        for slot, pi in enumerate(pidx):
            prompt = prompts[pi]
            prompt_ids = tok.encode(prompt)
            rollout_seed = seed + (it * prompts_per_iter + slot) * group_size
            resp = sample_group(policy, prompt_ids, device, group_size, temp, max_new,
                                rollout_seed, use_amp)
            texts = [truncate_at_stops(tok.decode(r)) for r in resp]
            scores = score_texts(prompt, texts)
            if len(scores) != group_size:
                raise ValueError(f"score_texts returned {len(scores)} scores for "
                                 f"{group_size} rollouts")
            groups_texts.append(texts)
            reward_rows.append([float(s) for s in scores])
            for text in texts:
                ex = build_sft_example(tok, prompt, text + END_SENTINEL)
                if len(ex[0]) > block_size:
                    raise ValueError(
                        f"rebuilt sequence ({len(ex[0])} tokens) exceeds block_size "
                        f"{block_size} — the prompt-pool guard should have excluded this "
                        f"prompt; refusing to truncate silently")
                examples.append(ex)

        rewards = torch.tensor(reward_rows, dtype=torch.float32)
        adv = group_advantages(rewards).reshape(-1).to(device)

        # --- capture pass (no grad): logp_old under the CURRENT policy + reference logps,
        # micro-batched exactly as the training pass will be, through the same forward path.
        micros: list[dict] = []
        total_tokens = 0
        with torch.no_grad():
            for s in range(0, n_seq, micro_batch):
                idx = list(range(s, min(s + micro_batch, n_seq)))
                batch = collate_sft([examples[i] for i in idx], PAD_ID, block_size)
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                logp_old, mask = forward_logps(policy, x, y, use_amp)
                ref_logp, _ = forward_logps(reference, x, y, use_amp)
                micros.append({"x": x, "y": y, "logp_old": logp_old, "ref_logp": ref_logp,
                               "mask": mask, "adv": adv[s:s + len(idx)]})
                total_tokens += int(mask.sum().item())
        if total_tokens == 0:
            raise ValueError("no response tokens to train on (every rollout empty?)")

        # --- one optimizer step over all P*G sequences, grad-accumulated per micro-batch.
        # Each micro contributes SUMS divided by the GLOBAL token count, so the accumulated
        # gradient is the exact global response-token mean regardless of micro_batch.
        opt.zero_grad(set_to_none=True)
        loss_total = kl_total = 0.0
        clip_total = 0
        for m in micros:
            logp_new, _ = forward_logps(policy, m["x"], m["y"], use_amp)
            surr_sum, n_clip = clipped_surrogate(logp_new, m["logp_old"], m["adv"],
                                                 m["mask"], clip_eps)
            kl_sum = k3_kl(m["ref_logp"], logp_new, m["mask"])
            loss = (-surr_sum + beta * kl_sum) / total_tokens
            loss.backward()
            loss_total += loss.item()
            kl_total += kl_sum.item()
            clip_total += n_clip
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        # --- post-step measurement: did the step move response log-probs WITH the
        # advantages? mean over sequences of A_i * (logp_after_i - logp_old_i), summed
        # response logps — positive = learning in the advantage direction.
        shift_sum = 0.0
        with torch.no_grad():
            for m in micros:
                logp_after, _ = forward_logps(policy, m["x"], m["y"], use_amp)
                d = (logp_after * m["mask"]).sum(dim=1) - (m["logp_old"] * m["mask"]).sum(dim=1)
                shift_sum += (m["adv"] * d).sum().item()
        del micros

        flat_texts = [t for g in groups_texts for t in g]
        record = {
            "iter": it, "lr": cur_lr, "loss": loss_total,
            "reward_mean": rewards.mean().item(), "reward_max": rewards.max().item(),
            "kl_mean": kl_total / total_tokens,
            "adv_logp_shift": shift_sum / n_seq,
            "resp_len_mean": total_tokens / n_seq,
            "distinct_frac": distinct_fraction(groups_texts),
            "repetition": repetition_score(flat_texts),
            "clip_frac": clip_total / total_tokens,
            "grad_norm": float(grad_norm),
            "prompt_indices": pidx,
            "elapsed_s": round(time.time() - t_it, 2),
        }
        if use_amp:
            record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        append_jsonl(log_path, record)
        print(f"iter {it}/{iters} reward {record['reward_mean']:.3f} "
              f"(max {record['reward_max']:.3f}) kl {record['kl_mean']:.5f} "
              f"clip {record['clip_frac']:.3f} shift {record['adv_logp_shift']:+.3f} "
              f"len {record['resp_len_mean']:.1f} distinct {record['distinct_frac']:.2f} "
              f"lr {cur_lr:.2e} {record['elapsed_s']}s", flush=True)

        # Baseline samples at iteration 1, then every dump_every: first rollout of the first
        # 3 prompts, with their RM scores.
        if dump_every > 0 and (it == 1 or it % dump_every == 0):
            for p in range(min(3, prompts_per_iter)):
                append_jsonl(samples_path, {"iter": it, "prompt": prompts[pidx[p]],
                                            "response": groups_texts[p][0],
                                            "reward": rewards[p][0].item()})

        # Rolling checkpoint (keep only the newest; the final save below is separate).
        if save_every > 0 and it % save_every == 0 and it != iters:
            path = save_ckpt(it)
            if last_rolling is not None and last_rolling.exists():
                last_rolling.unlink()
            last_rolling = path

    ckpt_path = save_ckpt(iters)
    if last_rolling is not None and last_rolling.exists() and last_rolling != ckpt_path:
        last_rolling.unlink()
    # Servable, self-contained run dir — exactly like sft.py/dpo.py.
    (out_dir / "tokenizer.json").write_text(
        Path(tokenizer_path).read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "serve_config.json").write_text(
        json.dumps(SERVE_CONFIG, indent=2) + "\n", encoding="utf-8")
    print(f"done: {iters} iterations -> {ckpt_path}", flush=True)
    return {"iters": iters, "ckpt_path": str(ckpt_path), "out_dir": str(out_dir),
            "log_path": str(log_path)}
