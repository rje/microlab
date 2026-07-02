"""Behavioral induction test (Phase-5 run-for-real): does the model exploit a repeated
sequence? A working induction circuit predicts the 2nd+ copy of a block far better than
the 1st. Measured on random tokens (tests a GENERAL copy circuit, the Olsson definition)
and on in-distribution text (tests whether any copying is at least domain-specific).

    python scripts/induction_behavioral.py runs/150m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from microlab.model.reference.checkpoint import load_variant_from_run  # noqa: E402


@torch.no_grad()
def block_losses(model, seq: torch.Tensor, period: int, repeats: int) -> list[float]:
    """Mean next-token loss within each repeat block."""
    logits, _ = model(seq[:, :-1])
    losses = F.cross_entropy(logits[0], seq[0, 1:], reduction="none")
    return [losses[max(r * period - 1, 0):(r + 1) * period - 1].mean().item()
            for r in range(repeats)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tokenizer-dir", default="data/shards/tinystories")
    ap.add_argument("--period", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, step = load_variant_from_run(args.run_dir, device=args.device)
    print(f"loaded step {step}\n")
    P, R = args.period, args.repeats

    block = torch.randint(0, model.config.vocab_size, (P,),
                          generator=torch.Generator().manual_seed(0))
    rand = block_losses(model, block.repeat(R).unsqueeze(0).to(args.device), P, R)

    val = np.fromfile(Path(args.tokenizer_dir) / "val-00000.bin",
                      dtype=np.uint16)[1000:1000 + P].astype(np.int64)
    idist = block_losses(model, torch.tensor(val).repeat(R).unsqueeze(0).to(args.device), P, R)

    print(f"  {'repeat':>8} {'random':>10} {'in-distr':>10}")
    for r in range(R):
        print(f"  {r + 1:>8} {rand[r]:>10.3f} {idist[r]:>10.3f}")
    print(f"\nrandom  loss drop repeat1->2: {rand[0] - rand[1]:+.3f} nats "
          f"({100 * (rand[0] - rand[1]) / rand[0]:+.0f}%)")
    print(f"in-dist loss drop repeat1->2: {idist[0] - idist[1]:+.3f} nats "
          f"({100 * (idist[0] - idist[1]) / idist[0]:+.0f}%)")
    print("\nBig random-token drop => a general induction circuit exists.")
    print("Random flat, in-dist worse on repeat => no copy circuit; pure learned statistics.")


if __name__ == "__main__":
    main()
