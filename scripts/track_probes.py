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

from microlab.infer.reference.kv_cache import generate_cached
from microlab.model.reference.variants import VariantConfig, VariantGPT
from microlab.tokenizer.fast import FastTokenizer

RUN = Path("runs/1b")
LOG = RUN / "probe_track.jsonl"
MILESTONE = 2000
QUAL_TOKENS = 30
EVAL_TOKENS = 6
SCORE_VERSION = 5  # bump when scoring/eval-set changes; older records re-scored (not "done")

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

# Likelihood-based multiple choice (the standard base-model eval): score each choice by the
# model's length-normalized log-prob of the continuation and pick the argmax. Robust (no string
# matching / format brittleness) and has real headroom for a 1B — the back-half signal once the
# greedy eval saturates. Hand-written (not from public test sets) to avoid contamination.
# (category, context, [choices...], answer_index). Choices begin with a leading space.
MC_EVAL = [
    # easy-medium commonsense / physical (a floor the model should mostly clear)
    ("commonsense", "It started raining, so he opened his",
     [" umbrella", " refrigerator"], 0),
    ("commonsense", "She was exhausted after work, so she went straight to",
     [" bed", " the gym for three hours"], 0),
    ("commonsense", "The plant on the windowsill died because no one remembered to",
     [" water it", " talk to it"], 0),
    ("commonsense", "She returned the shirt to the store because it did not",
     [" fit", " match her car"], 0),
    ("commonsense", "In the library everyone spoke quietly so they would not",
     [" disturb others", " lose their books"], 0),
    ("physical", "If you want to cut a piece of paper, you use",
     [" scissors", " a spoon"], 0),
    ("physical", "To find out what time it is, you look at a",
     [" clock", " mirror"], 0),
    ("physical", "To keep milk from spoiling, you store it in the",
     [" refrigerator", " oven"], 0),
    # medium science / semantics
    ("science", "Plants take in carbon dioxide and release",
     [" oxygen", " nitrogen"], 0),
    ("science", "An object falls to the ground because of",
     [" gravity", " friction"], 0),
    ("science", "Water freezes into ice when it gets very",
     [" cold", " loud"], 0),
    ("science", "A thermometer is a tool used to measure",
     [" temperature", " weight", " distance"], 0),
    ("semantic", "The opposite of 'generous' is",
     [" stingy", " tall", " quick"], 0),
    ("semantic", "A baby dog is called a",
     [" puppy", " kitten", " calf"], 0),
    ("semantic", "The color you get by mixing red and blue is",
     [" purple", " green", " orange"], 0),
    ("semantic", "A group of wolves is called a",
     [" pack", " herd", " flock"], 0),
    # hard science (counterintuitive — the common wrong answer is a distractor)
    ("science", "The gas that makes up most of Earth's atmosphere is",
     [" nitrogen", " oxygen", " carbon dioxide"], 0),
    ("science", "Sound travels fastest through",
     [" steel", " air", " empty space"], 0),
    ("semantic", "Tokyo is to Japan as Paris is to",
     [" France", " London", " Germany"], 0),
    ("semantic", "Finger is to hand as toe is to",
     [" foot", " arm", " knee"], 0),
    # Reasoning — rewritten COPY-TRAP-FREE and BIAS-BALANCED. The old set scored BELOW chance
    # because distractors echoed context tokens (" 60 miles" right after "travels 60 miles"), and
    # likelihood scoring rewards copying — so it measured copy-susceptibility, not reasoning.
    # Now no distractor repeats a salient context token, and surface biases (recency, lexical
    # frequency) are balanced across items, so a NON-reasoning model lands at ~chance, not below.
    ("reasoning", "A car travels 60 miles in one hour. At the same speed, in three hours it goes",
     [" 180 miles", " 120 miles", " 240 miles"], 0),
    ("reasoning", "Ben is older than Sara. Sara is older than Mia. The oldest person is",
     [" Ben", " Mia", " Sara"], 0),         # recency favours a WRONG answer
    ("reasoning", "Ana is shorter than Kim. Kim is shorter than Zoe. The tallest person is",
     [" Zoe", " Ana", " Kim"], 0),          # recency favours the RIGHT one (balances the above)
    ("reasoning", "All fish can swim. A salmon is a fish. Therefore a salmon can",
     [" swim", " fly", " sing"], 0),
    ("reasoning", "A rectangle has four sides and a triangle has three. Together they have",
     [" seven sides", " five sides", " twelve sides"], 0),
    ("reasoning", "The meeting was on Friday, but it was postponed by three days, so it is now on",
     [" Monday", " Tuesday", " Saturday"], 0),
    ("reasoning", "There are twelve eggs in one dozen, so two and a half dozen eggs is",
     [" thirty eggs", " twenty-four eggs", " twenty-five eggs"], 0),
    ("reasoning", "If baking one cake takes two hours, baking three cakes one after another takes",
     [" six hours", " five hours", " eight hours"], 0),
    # Winograd twins: identical but for one word, and the answer flips. A model that always picks
    # the same noun scores exactly chance across the pair — only real coreference beats it.
    ("reasoning", "The trophy did not fit into the suitcase because it was too large. "
     "The thing that was too large was the",
     [" trophy", " suitcase"], 0),
    ("reasoning", "The trophy did not fit into the suitcase because it was too small. "
     "The thing that was too small was the",
     [" trophy", " suitcase"], 1),
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
    # KV-cached greedy: byte-identical to sample.generate but O(n) not O(n^2) — ~10x faster
    # on CPU where these probes run (per the code-review finding).
    out = generate_cached(model, idx, n_tokens, temperature=0.0)
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


@torch.no_grad()
def _choice_avg_logprob(model, tok, context: str, choice: str) -> float:
    """Length-normalized log-prob of `choice` continuing `context` (lm-eval-harness style).
    Uses the common-prefix split so a BPE boundary merge between context and choice can't
    misalign the scored tokens."""
    ctx = tok.encode(context)
    full = tok.encode(context + choice)
    i = 0
    while i < len(ctx) and i < len(full) and ctx[i] == full[i]:
        i += 1
    if i == 0 or i >= len(full):  # need a non-empty context and a non-empty continuation
        return -1e9
    logits, _ = model(torch.tensor([full], dtype=torch.long))
    logp = torch.log_softmax(logits[0], dim=-1)  # token at pos p is predicted by logits[p-1]
    total = sum(logp[p - 1, full[p]].item() for p in range(i, len(full)))
    return total / (len(full) - i)


NEUTRAL_CTX = "Answer:"  # PMI baseline context


def score_mc(model, tok) -> dict:
    """Two scorings per item, so we can see whether surface bias is driving the result:
      - raw: argmax of length-normalized log P(choice | context)   (standard)
      - pmi: argmax of log P(choice | context) - log P(choice | NEUTRAL_CTX), which cancels the
             choice's intrinsic frequency (e.g. "trophy" being a commoner word than "suitcase")
    `chance` is the random baseline for this item set — the number to beat."""
    cats: dict[str, list[int]] = {}
    cats_pmi: dict[str, list[int]] = {}
    chance: list[float] = []
    for cat, context, choices, answer in MC_EVAL:
        raw = [_choice_avg_logprob(model, tok, context, c) for c in choices]
        base = [_choice_avg_logprob(model, tok, NEUTRAL_CTX, c) for c in choices]
        pmi = [r - b for r, b in zip(raw, base, strict=True)]
        cats.setdefault(cat, []).append(
            1 if max(range(len(raw)), key=raw.__getitem__) == answer else 0)
        cats_pmi.setdefault(cat, []).append(
            1 if max(range(len(pmi)), key=pmi.__getitem__) == answer else 0)
        chance.append(1 / len(choices))

    def _summ(c: dict[str, list[int]]):
        by_cat = {k: round(sum(v) / len(v), 3) for k, v in c.items()}
        total = [x for v in c.values() for x in v]
        return round(sum(total) / len(total), 3), by_cat, len(total)

    acc, by_cat, n = _summ(cats)
    acc_pmi, by_cat_pmi, _ = _summ(cats_pmi)
    return {"n": n, "accuracy": acc, "by_cat": by_cat, "accuracy_pmi": acc_pmi,
            "by_cat_pmi": by_cat_pmi, "chance": round(sum(chance) / len(chance), 3)}


def repetition_score(texts: list[str]) -> float:
    """Fraction of repeated 4-grams across free-form completions (0 = all distinct, higher =
    loopier). Quantifies the greedy looping; should fall as the model matures."""
    dup = tot = 0
    for t in texts:
        toks = t.split()
        grams = [tuple(toks[k:k + 4]) for k in range(len(toks) - 3)]
        if not grams:
            continue
        tot += len(grams)
        dup += len(grams) - len(set(grams))
    return round(dup / tot, 3) if tot else 0.0


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
        mc = score_mc(model, tok)
        rep = repetition_score([outputs[p] for p in QUAL_PROMPTS[:3]])  # 3 free-form prompts
        with LOG.open("a") as f:
            f.write(json.dumps({"step": step, "tokens": step * 524288, "outputs": outputs,
                                "eval": ev, "eval_completions": eval_comps, "mc": mc,
                                "repetition": rep, "score_version": SCORE_VERSION}) + "\n")
        gcats = " ".join(f"{c}={a:.0%}" for c, a in ev["by_cat"].items())
        mcats = " ".join(f"{c}={a:.0%}" for c, a in mc["by_cat"].items())
        print(f"=== step {step}  ({step * 524288 / 1e9:.2f}B tok) ===")
        print(f"  GEN-EVAL {ev['accuracy']:.0%} ({ev['n']}) | {gcats}")
        print(f"  MC-EVAL  {mc['accuracy']:.0%} (pmi {mc['accuracy_pmi']:.0%}, chance "
              f"{mc['chance']:.0%}, n={mc['n']}) | {mcats}")
        print(f"           reasoning: raw {mc['by_cat'].get('reasoning', 0):.0%} / "
              f"pmi {mc['by_cat_pmi'].get('reasoning', 0):.0%}  | repetition {rep:.0%}")
        for p in QUAL_PROMPTS:
            label = p.replace(chr(10), " / ")
            print(f"  [{label[:38]!r}] -> {outputs[p]!r}")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
