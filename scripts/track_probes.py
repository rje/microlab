"""Track how the 1B's capabilities evolve across checkpoints — qualitative probes + a scored
quantitative eval.

Loads each checkpoint on CPU (training owns the GPU) and greedy-decodes (temperature=0 ->
deterministic, so changes reflect the MODEL, not sampling noise).

- QUAL_PROMPTS: read the actual completion (looping/coherence/factual shifts). Original 3 kept
  for trajectory continuity; harder ones added (in-context learning, arithmetic, long-tail fact,
  causal cloze) that don't saturate at 1B tokens.
- EVAL: ~30 held-out items across 6 capability categories, scored by greedy prefix-match ->
  a hard accuracy % that climbs monotonically, complementing val perplexity.

Appends one JSON line per checkpoint to runs/1b/probe_track.jsonl. A step is "done" only once it
has a record WITH an "eval" field, so this re-seeds milestones logged under the old schema.
Default: milestones (step % 2000 == 0) + latest. Pass explicit steps to override.
"""
from __future__ import annotations

import gc
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch

from microlab.model.reference.sample import generate
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

RUN = Path("runs/1b")
LOG = RUN / "probe_track.jsonl"
MILESTONE = 2000
QUAL_TOKENS = 30
EVAL_TOKENS = 6
SCORE_VERSION = 2  # bump when _hit changes; older records are re-scored (not counted "done")

QUAL_PROMPTS = [
    "The capital of France is",                       # factual recall (original)
    "Once upon a time, there was a",                  # narrative fluency (original)
    "Water is made of hydrogen and",                  # science fact (original)
    "apple: fruit\ncarrot: vegetable\nsalmon: fish\nsparrow:",  # in-context learning
    "2 + 3 = 5\n7 + 4 = 11\n8 + 6 =",                 # few-shot arithmetic
    "The chemical symbol for tungsten is",            # long-tail knowledge (W)
    "Tom opened his umbrella because it started to",  # causal commonsense
]

# (category, prompt, [acceptable answers]) — scored by greedy prefix-match, case-insensitive.
EVAL = [
    ("capital", "The capital of Germany is", ["Berlin"]),
    ("capital", "The capital of Japan is", ["Tokyo"]),
    ("capital", "The capital of Italy is", ["Rome"]),
    ("capital", "The capital of Spain is", ["Madrid"]),
    ("capital", "The capital of Russia is", ["Moscow"]),
    ("science", "Water is made of hydrogen and", ["oxygen"]),
    ("science", "The chemical symbol for gold is", ["Au"]),
    ("science", "The largest planet in the solar system is", ["Jupiter"]),
    ("science", "The center of an atom is called the", ["nucleus"]),
    ("science", "Plants make food using sunlight in a process called", ["photosynthesis"]),
    ("commonsense", "If you drop a glass, it will", ["break", "shatter", "fall"]),
    ("commonsense", "She put on a warm coat because it was", ["cold", "freezing", "snowing"]),
    ("commonsense", "He was very tired, so he went to", ["sleep", "bed"]),
    ("commonsense", "The opposite of up is", ["down"]),
    ("commonsense", "Ice is frozen", ["water"]),
    ("sequence", "Monday, Tuesday, Wednesday,", ["Thursday"]),
    ("sequence", "One, two, three,", ["four"]),
    ("sequence", "January, February,", ["March"]),
    ("sequence", "A, B, C,", ["D"]),
    ("sequence", "Spring, summer, fall,", ["winter"]),
    ("arithmetic", "4 + 5 = 9\n1 + 2 = 3\n6 + 3 =", ["9"]),
    ("arithmetic", "7 + 2 = 9\n3 + 3 = 6\n5 + 4 =", ["9"]),
    ("arithmetic", "8 + 1 = 9\n2 + 2 = 4\n7 + 6 =", ["13"]),
    ("arithmetic", "10 + 5 = 15\n3 + 4 = 7\n6 + 8 =", ["14"]),
    ("arithmetic", "9 + 9 = 18\n2 + 3 = 5\n7 + 7 =", ["14"]),
    ("icl", "apple: fruit\ncarrot: vegetable\nsalmon: fish\nsparrow:", ["bird"]),
    ("icl", "hot: cold\nup: down\nbig: small\nhappy:", ["sad", "unhappy"]),
    ("icl", "France: Paris\nJapan: Tokyo\nItaly: Rome\nSpain:", ["Madrid"]),
    ("icl", "one: two\ntwo: three\nthree: four\nfour:", ["five"]),
    ("icl", "walk: walked\njump: jumped\nplay: played\ncook:", ["cooked"]),
]


def _norm(s: str) -> str:
    return s.strip().lower().lstrip(".,:;-–—\"'` \n\t")


def _hit(completion: str, answers: list[str]) -> bool:
    """Robust to verbosity: as the model matures it wraps the answer in a preamble
    ("...is T, and the symbol is W"), which strict startswith under-counts. Look only at the
    FIRST line and accept: it starts with the answer, OR a single-word answer appears as a
    whole word, OR a multi-word answer appears as a substring."""
    line = completion.strip().splitlines()[0] if completion.strip() else ""
    ln = _norm(line)
    words = re.findall(r"[a-z0-9]+", ln)
    for a in answers:
        an = _norm(a)
        aw = re.findall(r"[a-z0-9]+", an)
        # Single-word answer: must appear as a whole word (so "W" does not match "Washington").
        # Multi-word answer: substring of the first line.
        if len(aw) == 1 and aw[0] in words:
            return True
        if len(aw) > 1 and an in ln:
            return True
    return False


def step_of(p: Path) -> int:
    return int(p.stem.split("_")[1])


def load_ckpt(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    m = VariantGPT(VariantConfig(
        vocab_size=cfg.vocab_size, block_size=cfg.block_size, n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=0.0, norm=cfg.norm, pos=cfg.pos,
        mlp=cfg.mlp,
    ))
    m.load_state_dict(ckpt["model"])
    return m.eval(), ckpt["step"]


def complete(model, tok, prompt: str, n_tokens: int) -> str:
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    out = generate(model, idx, n_tokens, temperature=0.0)
    return tok.decode(out[0].tolist()[len(ids):])


def done_steps() -> set[int]:
    """Steps already scored under the CURRENT SCORE_VERSION. Older records are re-run so a
    scoring change re-scores the whole curve consistently."""
    if not LOG.exists():
        return set()
    out = set()
    for line in LOG.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("score_version") == SCORE_VERSION:
            out.add(rec["step"])
    return out


def run_eval(model, tok) -> tuple[dict, list[str]]:
    cats: dict[str, list[int]] = {}
    comps: list[str] = []
    for cat, prompt, answers in EVAL:
        comp = complete(model, tok, prompt, EVAL_TOKENS)
        comps.append(comp)
        cats.setdefault(cat, []).append(1 if _hit(comp, answers) else 0)
    by_cat = {c: round(sum(v) / len(v), 3) for c, v in cats.items()}
    total = [x for v in cats.values() for x in v]
    ev = {"n": len(total), "accuracy": round(sum(total) / len(total), 3), "by_cat": by_cat}
    return ev, comps


def main() -> None:
    tok = FastTokenizer.load(str(RUN / "tokenizer.json"))
    on_disk = sorted(RUN.glob("ckpt_*.pt"), key=step_of)
    if not on_disk:
        print("no checkpoints yet")
        return
    if len(sys.argv) > 1:
        want = {int(s) for s in sys.argv[1:]}
        ckpts = [c for c in on_disk if step_of(c) in want]
    else:
        milestones = [c for c in on_disk if step_of(c) % MILESTONE == 0]
        ckpts = milestones + [on_disk[-1]]
    already = done_steps()
    seen: set[int] = set()
    for ck in ckpts:
        s = step_of(ck)
        if s in already or s in seen:
            continue
        seen.add(s)
        model, step = load_ckpt(ck)
        outputs = {p: complete(model, tok, p, QUAL_TOKENS) for p in QUAL_PROMPTS}
        ev, eval_comps = run_eval(model, tok)
        with LOG.open("a") as f:
            f.write(json.dumps({"step": step, "tokens": step * 524288, "outputs": outputs,
                                "eval": ev, "eval_completions": eval_comps,
                                "score_version": SCORE_VERSION}) + "\n")
        cats = " ".join(f"{c}={a:.0%}" for c, a in ev["by_cat"].items())
        print(f"=== step {step}  ({step * 524288 / 1e9:.2f}B tok) "
              f"EVAL {ev['accuracy']:.0%} ({ev['n']}) | {cats} ===")
        for p in QUAL_PROMPTS[:3] + QUAL_PROMPTS[3:]:
            label = p.replace(chr(10), " / ")
            print(f"  [{label[:38]!r}] -> {outputs[p]!r}")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
