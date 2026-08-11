"""GRPO at 1B scale (Phase 13): on-policy RL against the Bradley-Terry reward model,
starting (and KL-anchored) at the SFT chat model.

    python scripts/train_grpo.py --policy runs/1b-sft-mix --rm runs/1b-rm \\
        --prefs data/corpora/sft_mix.jsonl --out runs/1b-grpo

Prompts are the FIRST --prompt-rows rows of --prefs (default sft_mix.jsonl[:5000]) — the
reward model's HOME distribution. That's load-bearing: this RM scores 74.9% pairwise on its
own distribution but chance (52.4%) off-distribution, so rolling out on other prompts would
optimize noise. Prompts whose chat template + --max-new + the END sentinel don't fit the
block are skipped (counted, printed). RM scoring sequences are built by the EXACT
training-time constructor (train_reward_model.build_reward_sequences, loaded from the script
like eval_best_of_n does) — a re-implementation could silently drift from what the RM was
trained on. The GRPO math, rollout loop, logging, and resume live in microlab.train.grpo.

Memory staging (measured on the RTX 6000 Ada, micro-batch 8): the trainable policy is fp32 +
AdamW (~16 GB); the frozen reference and RM keep fp32 weights too and run every forward under
no_grad + bf16 autocast — exactly the numeric path they were trained/evaluated with (bf16
WEIGHT copies would shift RM scores off the path eval_best_of_n validated). Static total
~24 GB; sampling's fp32 KV cache (~2.8 GB) and training activations peak a measured 32.1 GB
allocated at ~7 s/iteration (P=8, G=8, max_new 80), inside the 40 GB budget, so nothing
needs to be staged in and out."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.data.reference.loaders import load_dolly  # noqa: E402
from microlab.model.reference.checkpoint import (  # noqa: E402
    latest_checkpoint,
    load_variant_from_run,
)
from microlab.model.reference.sft import format_chat  # noqa: E402
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402
from microlab.train.grpo import END_SENTINEL, PAD_ID, run_grpo  # noqa: E402
from microlab.train.reward import collate_reward, load_reward_checkpoint  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


trm = _load_script("train_reward_model")  # THE reward-sequence constructor + sentinel

# The library trains the sentinel the RM scores after; if these ever drift the reward would
# silently score sequences the policy isn't being trained toward. Fail at import, loudly.
if trm.END_SENTINEL != END_SENTINEL or trm.PAD_ID != PAD_ID:
    raise RuntimeError(
        f"sentinel/pad drift: train_reward_model has ({trm.END_SENTINEL!r}, {trm.PAD_ID}) "
        f"but microlab.train.grpo has ({END_SENTINEL!r}, {PAD_ID})")


def build_scoring_sequences(tok, prompt: str, texts: list[str],
                            block_size: int) -> list[list[int]]:
    """Token sequences for RM scoring, one per rollout, built by the training-time
    constructor itself (build_reward_sequences with chosen == rejected == text) so the
    construction can never drift from what the RM was trained on — the same reuse as
    eval_best_of_n. Training SKIPS an overlong pair; here that would silently misalign
    rewards with rollouts, so it raises instead (can't happen when max_new << block_size)."""
    rows = [{"prompt": prompt, "chosen": t, "rejected": t} for t in texts]
    pairs, skipped = trm.build_reward_sequences(tok, rows, block_size)
    if skipped:
        raise ValueError(f"{skipped} rollout(s) + sentinel fill block_size {block_size}; "
                         f"rewards would misalign with rollouts")
    return [chosen for chosen, _ in pairs]


def make_score_texts(rm, tok, block_size: int, device: str, use_amp: bool, score_batch: int):
    """The reward oracle run_grpo calls: rollout texts -> RM scores, batched exactly like
    training/eval (collate_reward right-padding, per-row lengths, bf16 autocast, fp32 out)."""

    @torch.no_grad()
    def score_texts(prompt: str, texts: list[str]) -> list[float]:
        seqs = build_scoring_sequences(tok, prompt, texts, block_size)
        scores: list[float] = []
        for start in range(0, len(seqs), score_batch):
            batch = collate_reward(seqs[start:start + score_batch], PAD_ID)
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = rm(input_ids, lengths)
            else:
                out = rm(input_ids, lengths)
            scores.extend(out.float().tolist())
        return scores

    return score_texts


def build_prompt_pool(tok, rows: list[dict], max_new: int,
                      block_size: int) -> tuple[list[str], int]:
    """Chat-templated prompts whose tokens + max_new sampled tokens + the END sentinel fit
    the block (the sentinel is appended to the TRAINED sequence beyond the sampled tokens,
    so the plain prompt+max_new guard of build_rlaif_candidates isn't quite enough here).
    Returns (prompts, n_skipped); raises if nothing usable survives."""
    sentinel_len = len(tok.encode(END_SENTINEL))
    prompts: list[str] = []
    skipped = 0
    for row in rows:
        prompt, _ = format_chat(row["instruction"], row.get("context", ""))
        if len(tok.encode(prompt)) + max_new + sentinel_len > block_size:
            skipped += 1
            continue
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"no usable prompts: all {len(rows)} rows exceed block_size "
                         f"{block_size} with max_new {max_new}")
    return prompts, skipped


def build_executor_oracle(tok, pool_rows: list[dict], max_new: int, block_size: int,
                          timeout_s: float):
    """Executor-reward mode: chat-format each pool row (same block-fit guard as
    build_prompt_pool), key its I/O cases by the exact prompt string, and return
    (prompts, score_texts). Replaces the RM oracle behind the same interface."""
    from microlab.train.exec_reward import make_exec_score_texts
    sentinel_len = len(tok.encode(END_SENTINEL))
    io_by_prompt: dict[str, list[dict]] = {}
    skipped = 0
    for row in pool_rows:
        prompt, _ = format_chat(row["instruction"], "")
        if len(tok.encode(prompt)) + max_new + sentinel_len > block_size:
            skipped += 1
            continue
        io_by_prompt[prompt] = row["io"]
    if not io_by_prompt:
        raise ValueError(f"no usable pool rows: all {len(pool_rows)} exceed block_size")
    print(f"executor oracle: {len(io_by_prompt)} prompts ({skipped} skipped oversize)")
    return list(io_by_prompt), make_exec_score_texts(io_by_prompt, timeout_s=timeout_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/1b-sft-mix",
                    help="SFT run dir: policy init + frozen KL reference (latest ckpt)")
    ap.add_argument("--rm", default="runs/1b-rm", help="reward-model run dir (latest ckpt)")
    ap.add_argument("--reward", default="rm", choices=["rm", "executor"],
                    help="reward oracle: Bradley-Terry RM (--rm) or the code executor "
                         "(--pool)")
    ap.add_argument("--pool", default=None,
                    help="pool jsonl of {instruction, io} rows (required for --reward "
                         "executor)")
    ap.add_argument("--timeout-s", type=float, default=5.0,
                    help="per-case sandbox timeout for --reward executor")
    ap.add_argument("--prefs", default="data/corpora/sft_mix.jsonl",
                    help="instruction JSONL; the first --prompt-rows rows are the prompt pool")
    ap.add_argument("--out", default="runs/1b-grpo")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--prompts-per-iter", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="k3 KL penalty coefficient")
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="sequences per forward during the update (P*G / micro-batch "
                         "grad-accumulation micro-batches per optimizer step)")
    ap.add_argument("--save-every", type=int, default=25,
                    help=">0: rolling checkpoint every N iterations (latest kept)")
    ap.add_argument("--dump-every", type=int, default=10,
                    help=">0: append 3 sample generations + RM scores every N iterations")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--prompt-rows", type=int, default=5000,
                    help="first N rows of --prefs form the prompt pool (the RM's home "
                         "distribution)")
    ap.add_argument("--warmup-iters", type=int, default=10)
    ap.add_argument("--score-batch", type=int, default=8, help="sequences per RM forward")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {args.device!r} requested but CUDA is unavailable")
    torch.manual_seed(args.seed)

    policy_dir = Path(args.policy)
    tok_path = policy_dir / "tokenizer.json"

    if args.reward == "rm":
        rm_dir = Path(args.rm)
        rm_tok_path = rm_dir / "tokenizer.json"
        if tok_path.read_text(encoding="utf-8") != rm_tok_path.read_text(encoding="utf-8"):
            raise RuntimeError(f"{tok_path} != {rm_tok_path}: policy and RM tokenizers differ — "
                               f"RM scores would be garbage")
    elif not args.pool:
        raise ValueError("--reward executor requires --pool (a jsonl of {instruction, io} "
                         "rows)")
    tok = FastTokenizer.load(str(tok_path))

    t0 = time.time()
    policy, policy_step = load_variant_from_run(policy_dir, device=args.device)
    reference, _ = load_variant_from_run(policy_dir, device=args.device)

    if args.reward == "rm":
        rm_ckpt = latest_checkpoint(rm_dir)
        rm, rm_step = load_reward_checkpoint(rm_ckpt, device=args.device)
        rm.requires_grad_(False)
        block_size = policy.config.block_size
        if rm.backbone.config.block_size != block_size:
            raise RuntimeError(f"RM block_size {rm.backbone.config.block_size} != policy "
                               f"block_size {block_size}")

        rows = load_dolly(args.prefs, limit=args.prompt_rows)
        prompts, skipped = build_prompt_pool(tok, rows, args.max_new, block_size)
        use_amp = args.device.startswith("cuda")
        print(f"GRPO: policy {policy_dir.name} step {policy_step}, RM {rm_ckpt.name} step "
              f"{rm_step}, {len(prompts)} usable prompts of {len(rows)} rows ({skipped} skipped "
              f"by the block guard)", flush=True)
        score_texts = make_score_texts(rm, tok, block_size, args.device, use_amp,
                                       args.score_batch)
    else:
        block_size = policy.config.block_size
        use_amp = args.device.startswith("cuda")
        pool_rows = [json.loads(line) for line in
                    Path(args.pool).read_text(encoding="utf-8").splitlines() if line.strip()]
        prompts, score_texts = build_executor_oracle(tok, pool_rows, args.max_new, block_size,
                                                      args.timeout_s)
        print(f"GRPO: policy {policy_dir.name} step {policy_step}, executor reward, "
              f"{len(prompts)} usable prompts of {len(pool_rows)} pool rows", flush=True)

    print(f"  {args.iters} iters x {args.prompts_per_iter} prompts x {args.group_size} "
          f"rollouts (micro-batch {args.micro_batch}), lr {args.lr:g} (warmup "
          f"{args.warmup_iters}), beta {args.beta}, clip {args.clip_eps}, temp {args.temp}, "
          f"max_new {args.max_new} on {args.device}", flush=True)

    result = run_grpo(policy, reference, tok, prompts, score_texts, args.out, tok_path,
                      iters=args.iters, prompts_per_iter=args.prompts_per_iter,
                      group_size=args.group_size, lr=args.lr, beta=args.beta,
                      clip_eps=args.clip_eps, temp=args.temp, max_new=args.max_new,
                      micro_batch=args.micro_batch, save_every=args.save_every,
                      dump_every=args.dump_every, seed=args.seed, device=args.device,
                      warmup_iters=args.warmup_iters)

    if use_amp:
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print(f"total time: {time.time() - t0:.0f}s -> {result['ckpt_path']}")


if __name__ == "__main__":
    main()
