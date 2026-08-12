"""Unit 0: can v1 synthesize inputs that SEPARATE right code from wrong code?

Validity criterion is DISCRIMINATION (spec finding I1), not executability: an input
counts iff the reference runs it cleanly AND its output differs from a known-wrong
solution's. Synthesis prompts carry description+signature only — never gold asserts.

    python scripts/probe_input_discrimination.py --n-tasks 100 \\
        --out evals/harness/probe_discrimination.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_input_exprs(reply: str, entry_point: str) -> list[str]:
    """Call-expressions of entry_point found in the reply: deduped, first 3.

    Matches are single-line (no `entry_point(...)` spanning a newline) and PAREN-BALANCED:
    an enclosing call like `print(add(3,4))` yields `add(3,4)`, not `add(3,4))` — a naive
    `entry_point\\s*\\(.*\\)` regex greedily swallows the outer call's closing paren too."""
    out: list[str] = []
    for m in re.finditer(re.escape(entry_point) + r"\s*\(", reply):
        start = m.start()
        depth = 0
        end = None
        for i in range(m.end() - 1, len(reply)):
            ch = reply[i]
            if ch == "\n":
                break
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            continue
        expr = reply[start:end + 1]
        if expr not in out:
            out.append(expr)
        if len(out) == 3:
            break
    return out


_MUTATIONS = [("<=", ">="), ("<", "<="), (">", ">="), ("==", "!="), ("+", "-")]


def make_mutant(code: str) -> str | None:
    """First mechanical mutation that changes the text (labeled fallback wrong-solution
    when no failed bank candidate exists for a task); None if nothing mutates."""
    for old, new in _MUTATIONS:
        if old in code:
            mutated = code.replace(old, new, 1)
            if mutated != code:
                return mutated
    return None


def is_discriminating(sig_ref: tuple, sig_wrong: tuple) -> bool:
    """Reference must RUN on the input; discrimination = the wrong solution behaves
    differently (different output, error, or timeout)."""
    return sig_ref[0] == "ok" and sig_ref != sig_wrong


def main() -> None:  # pragma: no cover - GPU + sandbox operational
    import torch
    from datasets import load_dataset

    from microlab.infer.behavior import behavior_signature
    from microlab.model.reference.checkpoint import load_variant_from_run
    from microlab.model.reference.sft import format_chat
    from microlab.tokenizer.fast import FastTokenizer
    from microlab.train.exec_reward import sample_solutions

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="runs/coder-1b-instruct-compliant")
    ap.add_argument("--out", default="evals/harness/probe_discrimination.jsonl")
    ap.add_argument("--n-tasks", type=int, default=100, help="split evenly HE/MBPP")
    ap.add_argument("--wrong-bank",
                    default="evals/instruct/coder-1b-instruct-v2-humaneval-sampled.jsonl",
                    help="existing bank searched for a FAILED candidate per HumanEval task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=3.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    half = args.n_tasks // 2
    mbpp_rows = list(load_dataset("google-research-datasets/mbpp", "sanitized",
                                  split="test"))[:half]
    he_rows = list(load_dataset("openai/openai_humaneval", split="test"))[:half]

    # wrong-candidate lookup from an existing bank (failed samples), HumanEval only
    wrong_by_task: dict[str, str] = {}
    wb = Path(args.wrong_bank)
    if wb.exists():
        for line in wb.read_text().splitlines():
            d = json.loads(line)
            if "task_id" in d and not d.get("passed") and d.get("solution", "").strip():
                wrong_by_task.setdefault(d["task_id"], d["solution"])

    # probe entries: (task_id, description, signature, entry_point, reference, wrong, src)
    probes = []
    for row in he_rows:
        ref = row["prompt"] + row["canonical_solution"]
        wrong = wrong_by_task.get(row["task_id"]) or make_mutant(ref)
        if wrong is None:
            continue
        src = "bank" if row["task_id"] in wrong_by_task else "mutant"
        sig = f"def {row['entry_point']}(...)"
        probes.append((row["task_id"], row["prompt"], sig, row["entry_point"],
                       ref, wrong, src))
    for row in mbpp_rows:
        code = row.get("code", "")
        sigline = next((ln.strip() for ln in code.splitlines()
                        if ln.strip().startswith("def ") and ln.rstrip().endswith(":")),
                       None)
        if sigline is None:
            continue
        entry = sigline.split("def ", 1)[1].split("(", 1)[0].strip()
        wrong = make_mutant(code)
        if wrong is None:
            continue
        probes.append((f"Mbpp/{row['task_id']}", row["prompt"], sigline, entry,
                       code, wrong, "mutant"))

    model, _ = load_variant_from_run(Path(args.policy), device=args.device)
    tok = FastTokenizer.load(str(Path(args.policy) / "tokenizer.json"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["task_id"] for x in out.read_text().splitlines()} \
        if out.exists() else set()

    with out.open("a", encoding="utf-8") as f:
        for task_id, desc, sig, entry, ref, wrong, src in probes:
            if task_id in done:
                continue
            instr = (f"{desc.strip()}\n\nGive 3 example calls to `{entry}` (one per "
                     f"line, plain expressions like `{entry}(...)` with concrete "
                     f"arguments). Signature: {sig}")
            prompt, _ = format_chat(instr, "")
            reply = sample_solutions(model, tok._tok, prompt, 1, max_new=120,
                                     temp=0.7, top_k=40, seed=args.seed,
                                     device=args.device)[0]
            exprs = parse_input_exprs(reply, entry)
            n_disc = 0
            for e in exprs:
                s_ref = behavior_signature(ref, entry, e, timeout_s=args.timeout_s)
                s_wr = behavior_signature(wrong, entry, e, timeout_s=args.timeout_s)
                if is_discriminating(s_ref, s_wr):
                    n_disc += 1
            f.write(json.dumps({"task_id": task_id, "n_inputs": len(exprs),
                                "n_discriminating": n_disc, "wrong_source": src}) + "\n")
            f.flush()

    rows = [json.loads(x) for x in out.read_text().splitlines()]
    frac = sum(1 for r in rows if r["n_discriminating"] >= 2) / len(rows)
    band = ("SYNTH-OK (>=56%)" if frac >= 0.56 else
            "AMBIGUOUS (44-56%): run both regimes" if frac >= 0.44 else
            "SUBSET-ONLY (24-44%)" if frac >= 0.24 else
            "SYNTHESIS DEAD (<24%)")
    print(f"PROBE: {sum(1 for r in rows if r['n_discriminating'] >= 2)}/{len(rows)} "
          f"tasks with >=2 discriminating inputs = {frac:.2f} -> {band}")


if __name__ == "__main__":
    main()
