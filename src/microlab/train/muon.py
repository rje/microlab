"""Muon: SGD-momentum whose 2-D update matrices are orthogonalized by a quintic
Newton-Schulz iteration before being applied (Jordan et al.; "Muon is Scalable for LLM
Training", arXiv 2502.16982). Orthogonalizing the momentum equalizes the singular values
of the update, letting rare-but-important gradient directions act at full strength.

Muon is matrices-only by construction: embeddings (including the tied wte/lm_head
tensor), norms, and any other non-2-D params stay on AdamW. `MuonAdamW` packages the two
optimizers behind the subset of the torch optimizer interface the Trainer uses, so
checkpoint save/resume keeps working through the ordinary state_dict round-trip.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Quintic iteration coefficients from the reference impl (tuned to maximize convergence
# slope at zero rather than to converge tightly to 1 — see newton_schulz docstring).
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately orthogonalize G via `steps` quintic Newton-Schulz iterations,
    i.e. return roughly U V^T where G = U S V^T. Runs in bfloat16 on CUDA (the iteration
    is stable enough and ~2x faster); fp32 on CPU where bf16 matmuls are slow.

    Approximate by design: the coefficients drive singular values into roughly
    (0.7, 1.3) instead of exactly 1, which the reference impl found not to hurt.
    Internally transposes so rows <= cols (the X X^T gram is then the small side).
    """
    if G.ndim != 2:
        raise ValueError(f"newton_schulz expects a 2-D matrix, got shape {tuple(G.shape)}")
    a, b, c = NS_COEFFS
    x = G.to(torch.bfloat16) if G.is_cuda else G.to(torch.float32)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT
    # Normalize so the largest singular value is <= 1 (the iteration's convergence basin).
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for 2-D parameters ONLY: nesterov momentum, then Newton-Schulz
    orthogonalization, then the reference impl's aspect-ratio compensation
    max(1, rows/cols)**0.5 (tall matrices get a proportionally larger step; without it
    per-row RMS shrinks as sqrt(cols/rows)). Weight decay is decoupled (AdamW-style).

    Defaults follow the reference impl: lr=0.02, momentum=0.95, nesterov, 5 NS steps,
    weight_decay=0.01 — note lr is on a different scale than AdamW's.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only handles 2-D parameters, got shape {tuple(p.shape)}; "
                        "route embeddings/norms/other params to AdamW instead"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - momentum)  # EMA-style momentum (reference impl form)
                update = g.lerp(buf, momentum) if group["nesterov"] else buf
                update = newton_schulz(update, steps=group["ns_steps"])
                if group["weight_decay"] != 0.0:
                    p.mul_(1.0 - lr * group["weight_decay"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


def build_muon_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition `model`'s params into (muon_params, adamw_params).

    Muon gets 2-D matrices EXCEPT embedding weights and the lm_head weight; everything
    else (norm gains, biases, embeddings) goes to AdamW. The tied wte/lm_head tensor is
    one tensor — named_parameters() deduplicates by identity, and it is excluded from
    Muon by both rules, so it lands exactly once, in the AdamW group.
    """
    excluded: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            excluded.add(id(module.weight))
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        excluded.add(id(lm_head.weight))
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for _, p in model.named_parameters():
        if p.ndim == 2 and id(p) not in excluded:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    if not muon_params:
        raise ValueError("no 2-D matrix params found for Muon — wrong model?")
    return muon_params, adamw_params


class MuonAdamW:
    """Muon (matrices) + AdamW (everything else) behind the optimizer interface the
    Trainer uses: param_groups / zero_grad / step / state_dict / load_state_dict.

    Every param group carries `lr_scale`: the Trainer's LR schedule computes the AdamW
    LR and multiplies by lr_scale per group, so the Muon groups run the same warmup +
    cosine shape at muon_lr's scale. lr_scale is config-derived (not learned state), so
    it is re-stamped after load_state_dict — the current config wins over the checkpoint.
    """

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW, muon_lr_scale: float):
        self.muon = muon
        self.adamw = adamw
        self.muon_lr_scale = muon_lr_scale
        self._stamp_lr_scales()

    def _stamp_lr_scales(self) -> None:
        for g in self.muon.param_groups:
            g["lr_scale"] = self.muon_lr_scale
        for g in self.adamw.param_groups:
            g["lr_scale"] = 1.0

    @property
    def param_groups(self):
        return [*self.muon.param_groups, *self.adamw.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if set(state_dict) != {"muon", "adamw"}:
            raise KeyError(
                f"expected a MuonAdamW checkpoint with keys {{'muon', 'adamw'}}, got "
                f"{sorted(state_dict)} — was this run trained with a different optimizer?"
            )
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self._stamp_lr_scales()


def build_muon_optimizer(model: nn.Module, *, lr: float, muon_lr: float,
                         betas: tuple[float, float], weight_decay: float,
                         fused: bool) -> MuonAdamW:
    """The hybrid optimizer for a VariantGPT run: Muon on the transformer-block matrices
    at muon_lr, AdamW on the rest at lr (the repo's usual AdamW settings).

    Decay routing was silently wrong for a config generation and found only by review:
    Muon was constructed WITHOUT weight_decay, so ~92% of parameters ran at the class
    default 0.01 while the config declared 0.1 — and Moonlight's headline Muon-at-scale
    result is precisely that Muon needs real decoupled decay to hold the AdamW trajectory
    over long horizons. Meanwhile AdamW applied the full 0.1 to its whole group, which is
    mostly 1-D tensors: every RMSNorm gain decayed toward zero with nothing pushing back,
    against universal practice (nanoGPT, Llama, OLMo all exempt 1-D params).
    """
    muon_params, adamw_params = build_muon_param_groups(model)
    muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
    decay = [p for p in adamw_params if p.dim() >= 2]      # embeddings / untied head
    no_decay = [p for p in adamw_params if p.dim() < 2]    # norm gains, biases
    groups = [{"params": ps, "weight_decay": wd}
              for ps, wd in ((decay, weight_decay), (no_decay, 0.0)) if ps]
    adamw = torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    return MuonAdamW(muon, adamw, muon_lr_scale=muon_lr / lr)
