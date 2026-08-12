"""Behavioral signatures: run a candidate on one input expression in the sandbox and
return a comparable outcome. The impure counterpart to microlab.infer.selection —
clustering compares these signatures; candidates that behave identically on shared
inputs land in the same cluster (AlphaCode-lineage behavioral equivalence)."""
from __future__ import annotations

from microlab.evals.code.executor import run_python

_HARNESS = "{candidate}\n\n_r = {input_expr}\nprint(repr(_r))\n"


def behavior_signature(candidate: str, entry_point: str, input_expr: str,
                       timeout_s: float = 3.0) -> tuple:
    """One (candidate, input) execution -> ("ok", repr) | ("err",) | ("timeout",).
    `entry_point` is accepted for interface clarity/logging; the input_expr already names
    the callable. Errors collapse to ("err",) deliberately: two candidates failing
    differently should not cluster as 'same behavior' by error-text accident."""
    prog = _HARNESS.format(candidate=candidate, input_expr=input_expr)
    res = run_python(prog, timeout_s=timeout_s)
    if res.timed_out:
        return ("timeout",)
    if res.exit_code != 0:
        return ("err",)
    return ("ok", res.stdout.strip())


def signatures_for(candidate: str, entry_point: str, input_exprs: list[str],
                   timeout_s: float = 3.0) -> tuple:
    """Signature VECTOR over all shared inputs — the clustering key."""
    return tuple(behavior_signature(candidate, entry_point, e, timeout_s=timeout_s)
                 for e in input_exprs)
