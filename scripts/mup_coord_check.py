"""muP coordinate check: activation RMS across widths should stay flat under muP scaling
and drift under standard scaling. Run on CPU or GPU; prints a table, saves nothing.

    python scripts/mup_coord_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from microlab.model.reference.scaling import mup_multipliers  # noqa: E402
from microlab.model.reference.variants import VariantConfig, VariantGPT  # noqa: E402

BASE = 64
WIDTHS = [64, 128, 256, 512]
STEPS = 20
LR = 0.01


def final_block_rms(width: int, mup: bool) -> float:
    torch.manual_seed(0)
    cfg = VariantConfig(vocab_size=256, block_size=64, n_layer=2, n_head=max(1, width // 32),
                        n_embd=width, norm="rms", pos="rope", mlp="swiglu")
    model = VariantGPT(cfg)
    mults = mup_multipliers(BASE, width)
    if mup:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.ndim == 2 and "wte" not in name:
                    p.mul_(mults["hidden_init_std_mult"])
    lr_hidden = LR * (mults["hidden_lr_mult"] if mup else 1.0)
    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "wte" not in n]
    vector = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "wte" not in n)]
    opt = torch.optim.Adam([
        {"params": hidden, "lr": lr_hidden},
        {"params": vector, "lr": LR},
    ])
    gen = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        x = torch.randint(0, 256, (8, 64), generator=gen)
        _, loss = model(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = torch.randint(0, 256, (8, 64), generator=gen)
        h = model.transformer.drop(model.transformer.wte(x))
        for block in model.transformer.h:
            h = block(h)
        return h.pow(2).mean().sqrt().item()


def main() -> None:
    print(f"{'width':>6} {'SP rms':>10} {'muP rms':>10}")
    for w in WIDTHS:
        sp = final_block_rms(w, mup=False)
        mup = final_block_rms(w, mup=True)
        print(f"{w:>6} {sp:>10.3f} {mup:>10.3f}")
    print("muP column should stay roughly flat; SP column should drift with width.")


if __name__ == "__main__":
    main()
