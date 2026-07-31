#!/usr/bin/env python
"""Pre-flight gate for an ablation lane. Run BEFORE spending GPU, not after.

Every rigour rule this lab has was written as a post-mortem: the verdict-audit protocol
after a verdict looked unexamined, the noise-band rule after a significance claim used the
wrong denominator, the Chinchilla-duration rule after a 4500-step lane inverted its own
result, the parity review after the 1B shipped MHA at 1024 context. Each was correct and
each arrived too late to stop the run it was about.

This is the same rules, mechanised, and applied at the only moment that saves anything:
before launch. It exits non-zero on a hard failure so it can gate a queue script.

    python scripts/preflight_lane.py configs/a.py configs/b.py

Checks:
  1. DURATION      tokens >= 1x Chinchilla (20 x params) for the ablation model.
  2. ARM PARITY    two configs given -> they must differ in as few fields as possible,
                   and the differing fields are printed so a confound is visible.
  3. PARITY GAPS   n_kv_head / block_size / rope_base unset -> the 1B's three errors.
                   WARN (legitimate for ablations), never silent.
  4. SEEDS         a paired A/B needs >= 2 seeds to be adoption-grade; 1 is PROVISIONAL.
  5. TRAINABILITY  an architecture must be affordable AT THE CONTEXT ITS BENEFITS REQUIRE.
                   The GDN hybrid was adopted on long-context wins (4x KV, ~10x length
                   generalisation, decode crossover at ~100k) and validated entirely at
                   1024 tokens. Measured cost vs dense, same GPU, fixed tokens/step:
                       ctx    1024   2048   4096   8192   16384
                       ratio  2.29x  2.83x  3.90x  5.77x  9.33x
                   It gets WORSE with context — the opposite of the architecture — because
                   our scan is a Python loop of T/chunk sequential steps against fused
                   FlashAttention. Adopting an architecture you cannot train at its target
                   context wastes the whole run, so this is a hard failure, not a warning.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load(path: str):
    spec = importlib.util.spec_from_file_location("cfg_mod", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.config


def est_params(c) -> int:
    """TOTAL params, embeddings included — the conventional Chinchilla N, and the more
    CONSERVATIVE choice (a larger N means a larger token target). Using non-embedding
    params instead reported the 4500-step lane as 0.43x where docs/gdn-hybrid-verdict.md
    says 0.30x; two Chinchilla numbers in one repo is exactly the sloppiness this file
    exists to stop. Per layer: 4*d^2 attention + ~3*d*(8/3 d) MLP ~= 12 d^2, plus a tied
    embedding of vocab*d."""
    return 12 * c.n_layer * c.n_embd ** 2 + c.vocab_size * c.n_embd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--allow-under-trained", action="store_true",
                    help="proceed below 1x Chinchilla; the verdict is then UNDER-TRAINED "
                         "and explicitly not adoption-grade")
    a = ap.parse_args()

    cfgs = [(p, load(p)) for p in a.configs]
    hard, warn = [], []

    print("=" * 72)
    for path, c in cfgs:
        params = est_params(c)
        tokens = c.batch_size * c.grad_accum * c.block_size * c.max_steps
        ratio = tokens / (20 * params)
        print(f"{Path(path).name}")
        print(f"  ~{params/1e6:.0f}M params (total) | {tokens/1e9:.2f}B tokens "
              f"| {ratio:.2f}x Chinchilla | {c.max_steps} steps")
        if ratio < 1.0:
            msg = (f"{Path(path).name}: {ratio:.2f}x Chinchilla ({tokens/1e9:.2f}B tokens). "
                   f"A 0.30x lane INVERTED its own verdict (docs/gdn-hybrid-verdict.md). "
                   f"Need ~{int(20*params/(c.batch_size*c.grad_accum*c.block_size))} steps.")
            (warn if a.allow_under_trained else hard).append(msg)

        # TRAINABILITY. Measured 2026-07-31 at 124M, fixed 16384 tokens/step, both arms
        # under identical conditions. Ratios are hybrid/dense wall-clock per step.
        # Two tables, because the answer depends entirely on which kernel you run.
        # UNFUSED: our pure-PyTorch scan (fp32, T/chunk sequential triangular solves).
        # FUSED: flash-linear-attention Triton kernels. Both measured token-matched at 1B
        # with an optimizer step, median of 10 iters.
        UNFUSED = {1024: 2.29, 2048: 2.83, 4096: 3.90, 8192: 5.77, 16384: 9.33}
        FUSED   = {4096: 1.37, 8192: 1.18, 16384: 1.06, 32768: 0.90}
        fused = getattr(c, "gdn_fused", True)
        HYBRID_COST = FUSED if fused else UNFUSED
        if getattr(c, "hybrid_every", None) is not None:
            near = min(HYBRID_COST, key=lambda t: abs(t - c.block_size))
            cost = HYBRID_COST[near]
            if cost > 2.0:
                hard.append(
                    f"{Path(path).name}: hybrid at block_size {c.block_size} costs ~{cost:.1f}x "
                    f"dense per step ({'fused' if fused else 'UNFUSED pure-PyTorch scan'}). "
                    f"The hybrid's benefits are long-context benefits; if it cannot be "
                    f"trained there it cannot deliver them."
                    + ("" if fused else "  Set gdn_fused=True — fused measured 0.90x at 32k."))
            elif cost > 1.2:
                warn.append(f"{Path(path).name}: hybrid costs ~{cost:.1f}x dense at "
                            f"block_size {c.block_size}.")
            if not fused and c.block_size >= 4096:
                warn.append(f"{Path(path).name}: gdn_fused=False at block_size "
                            f"{c.block_size} — the reference scan is an ORACLE, not a "
                            f"training kernel (measured 23-31x slower, gap does not shrink "
                            f"with context).")

        # Memory. Measured at 1B: 32768-token steps cost 27.70 GB with a naive loss path
        # and 15.40 GB with fused CE; grad checkpointing is assumed throughout. A long-context
        # config without both OOMs on a 48GB card — which is exactly how the first frontier
        # smoke run died, after the gate had passed it.
        tok_per_micro = c.batch_size * c.block_size
        if tok_per_micro >= 16384:
            if not getattr(c, "grad_checkpoint", False):
                hard.append(f"{Path(path).name}: {tok_per_micro:,} tokens per micro-batch "
                            f"without grad_checkpoint — will OOM on a 48GB card.")
            if not getattr(c, "fused_ce", False):
                warn.append(f"{Path(path).name}: {tok_per_micro:,} tokens per micro-batch "
                            f"without fused_ce; the loss path is ~half of training memory "
                            f"(measured 44% saving at 32k).")

        # The 1B's three documented errors, checked mechanically.
        if getattr(c, "global_attn", "gqa") == "mla":
            lora = getattr(c, "mla_kv_lora", 512)
            print(f"  global attention: MLA, cache {lora} values/token "
                  f"(per-head distinct K/V; NoPE so no decoupled-rope dims)")
        elif getattr(c, "n_kv_head", None) is None:
            warn.append(f"{Path(path).name}: n_kv_head unset -> full MHA. Cohort ships "
                        f"n_kv_head=2 below ~3B (sota-parity-code-specialist.md).")
        if c.block_size <= 2048:
            warn.append(f"{Path(path).name}: block_size {c.block_size}. Fine for an "
                        f"ablation; the cohort floor for a SHIPPED model is 16k.")
        if c.pos != "nope" and getattr(c, "rope_base", 10000.0) == 10000.0 and c.block_size > 4096:
            warn.append(f"{Path(path).name}: rope_base 1e4 with block_size "
                        f"{c.block_size}; 1e6 is the standard pairing above 4k.")

    if len(cfgs) == 2:
        (pa, ca), (pb, cb) = cfgs
        diffs = [k for k in vars(ca)
                 if k != "out_dir" and getattr(ca, k, None) != getattr(cb, k, None)]
        print(f"\narms differ in {len(diffs)} field(s) besides out_dir: {diffs or 'none'}")
        if len(diffs) == 0:
            da, db = getattr(ca, "data_dir", None), getattr(cb, "data_dir", None)
            if da and db and da != db:
                print(f"  intervention is the data dir: {da}  vs  {db}")
            else:
                hard.append("the two arms are identical — nothing is being measured. If the "
                            "intervention is a --data-dir passed by a launcher script, put "
                            "it in the config (data_dir=) so the experiment is legible from "
                            "the repo alone.")
        elif len(diffs) > 1:
            warn.append(f"arms differ in {len(diffs)} fields {diffs} — a multi-field diff "
                        f"means the result cannot be attributed to one intervention")
        if getattr(ca, "seed", None) != getattr(cb, "seed", None):
            hard.append("arms use DIFFERENT seeds — the paired design is what buys the "
                        "power (paired sd 0.0025 vs cross-seed 0.0065); unpaired at this "
                        "scale cannot resolve a typical architecture effect")

    print("\n" + "=" * 72)
    for w in warn:
        print(f"WARN  {w}")
    for h in hard:
        print(f"FAIL  {h}")
    if hard:
        print(f"\n{len(hard)} hard failure(s) — not adoption-grade. "
              f"Re-run with --allow-under-trained only if you will label the verdict so.")
        return 1
    print(f"\nPASS ({len(warn)} warning(s)). Remember: 1 seed pair detects an inversion "
          f"but does not resolve an effect near the 0.0025 band.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
