"""Compare two runs mechanistically (e.g. a base model vs its fine-tuned sibling): the
per-module relative weight change, and a logit-lens readout of where a prompt's answer
emerges layer-by-layer in each. Answers "what did fine-tuning change, and how."

    python scripts/interp_diff_runs.py runs/350m runs/350m-sft --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.interp.reference.lens import collect_residual_stream, logit_lens  # noqa: E402
from microlab.model.reference.checkpoint import (  # noqa: E402
    latest_checkpoint,
    load_variant_from_run,
)
from microlab.tokenizer.fast import FastTokenizer  # noqa: E402


def _module_type(k: str) -> str:
    if "wte" in k or "lm_head" in k:
        return "embedding/lm_head"
    if "attn" in k:
        return "attention"
    if "mlp" in k:
        return "mlp/swiglu"
    if "ln" in k or "norm" in k:
        return "norms"
    return "other"


def weight_delta(run_a: Path, run_b: Path) -> None:
    a = torch.load(latest_checkpoint(run_a), map_location="cpu", weights_only=False)["model"]
    b = torch.load(latest_checkpoint(run_b), map_location="cpu", weights_only=False)["model"]
    tot_d = sum((b[k] - a[k]).norm().item() ** 2 for k in a) ** 0.5
    tot_b = sum(a[k].norm().item() ** 2 for k in a) ** 0.5
    pct = 100 * tot_d / tot_b
    print(f"overall relative weight change {run_a.name} -> {run_b.name}: {pct:.2f}%\n")
    by_type: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for k in a:
        by_type[_module_type(k)][0] += (b[k] - a[k]).norm().item() ** 2
        by_type[_module_type(k)][1] += a[k].norm().item() ** 2
    print("by module type:")
    for t, (d, base) in sorted(by_type.items(), key=lambda x: -(x[1][0] / x[1][1]) ** 0.5):
        print(f"  {t:>18}: {100 * (d / base) ** 0.5:.2f}%")


def lens_compare(run_a: Path, run_b: Path, prompt: str) -> None:
    tok = FastTokenizer.load(str(run_b / "tokenizer.json"))
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    def table(run: Path):
        model, _ = load_variant_from_run(run, device="cpu")
        res = collect_residual_stream(model, ids)
        lens = logit_lens(res, model.transformer.ln_f, model.lm_head)
        out = []
        for layer in range(lens.shape[0]):
            p = lens[layer, 0, -1].softmax(-1)
            top = p.argmax().item()
            out.append((tok.decode([top]).strip(), p[top].item()))
        return out

    ta, tb = table(run_a), table(run_b)
    print(f"\nlogit lens, prompt {prompt!r} (next-token per layer):")
    print(f"{'layer':>5}  {run_a.name:>14} {'p':>5}   {run_b.name:>14} {'p':>5}")
    for layer, ((wa, pa), (wb, pb)) in enumerate(zip(ta, tb, strict=True)):
        print(f"{layer:>5}  {wa:>14} {pa:5.2f}   {wb:>14} {pb:5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()
    weight_delta(args.run_a, args.run_b)
    lens_compare(args.run_a, args.run_b, args.prompt)


if __name__ == "__main__":
    main()
