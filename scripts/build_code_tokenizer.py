"""Train the code-native tokenizer candidates for the coding-specialist study.

Builds two candidates on the sampled corpora (see build_code_tokenizer_corpora.py):

  code-49k : 49,152-vocab byte-level BPE, digit-splitting, code-heavy mix
             (60% code across langs / 30% prose / 10% glue), multi-space merges enabled.
  code-32k : the same recipe at 32k, for a size-matched comparison against the baseline.

The existing 32k FineWeb tokenizer (runs/1b/tokenizer.json) is the third, baseline point of
comparison and is NOT trained here -- it is copied into place by --copy-baseline for the
fertility study so all three sit under data/tokenizers/.

    python scripts/build_code_tokenizer.py --copy-baseline   # train both + stage baseline

Recipe notes:
  * Digit splitting: a `Digits(individual_digits=True)` pre-tokenizer runs before ByteLevel,
    so every digit is a hard token boundary -- "12345" -> 5 tokens, "3.14" -> "3 . 1 4".
    This directly targets the measured arithmetic floor (digit-pair merges make a model
    memorize "42","99",... instead of composing digits).
  * Whitespace / indentation: the byte-level pre-tokenizer keeps runs of spaces and the
    leading indent together as one pre-token (it does NOT split on whitespace), so BPE is
    free to learn multi-space merge tokens ("    ", "        ", "\n    "). Because the mix
    is code-heavy, those indentation merges are learned -- verified by
    `count_multispace_tokens` on the trained vocab. No extra whitespace pre-tokenizer is
    needed (and none is added, which would forbid those merges).
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from microlab.tokenizer.fast import EOT, FastTokenizer

# Byte-level marker chars (GPT-2 mapping) for a leading space and a newline.
BYTE_SPACE = "Ġ"  # 'Ġ'
BYTE_NEWLINE = "Ċ"  # 'Ċ'

# Language buckets for the training mix. "code" = the primary target languages, "glue" =
# the secondary structured/markup languages, "prose" = English.
CODE_LANGS = ("python", "javascript", "typescript")
GLUE_LANGS = ("shell", "sql", "json", "markdown")
PROSE_LANGS = ("prose",)

# Mix weights: 60% code / 30% prose / 10% glue (by sampled bytes).
MIX_WEIGHTS = {"code": 0.60, "prose": 0.30, "glue": 0.10}


def build_code_tokenizer(vocab_size: int) -> Tokenizer:
    """Construct (untrained) the byte-level BPE with digit-splitting for a code corpus."""
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def _bucket_of(lang: str) -> str:
    if lang in CODE_LANGS:
        return "code"
    if lang in GLUE_LANGS:
        return "glue"
    if lang in PROSE_LANGS:
        return "prose"
    raise ValueError(f"unknown language bucket for {lang!r}")


def plan_byte_budget(available: dict[str, int], total_budget: int) -> dict[str, int]:
    """Allocate a per-language byte budget hitting the 60/30/10 bucket weights.

    `available` maps lang -> bytes on disk. Each bucket's share of `total_budget` is split
    evenly across the languages present in it, then clamped to what is actually available
    (raise-don't-fallback: a language with zero bytes is a corpus bug, surfaced by the
    caller, not silently reweighted here). Returns lang -> target bytes to read.
    """
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    buckets: dict[str, list[str]] = {"code": [], "glue": [], "prose": []}
    for lang in available:
        buckets[_bucket_of(lang)].append(lang)
    plan: dict[str, int] = {}
    for bucket, langs in buckets.items():
        if not langs:
            continue
        per_lang = int(total_budget * MIX_WEIGHTS[bucket] / len(langs))
        for lang in langs:
            plan[lang] = min(per_lang, available[lang])
    return plan


def iter_language_text(lang_dir: Path, max_bytes: int, chunk_chars: int = 200_000) -> Iterator[str]:
    """Yield text chunks from a language's `.txt` shards up to `max_bytes` (utf-8)."""
    emitted = 0
    for path in sorted(lang_dir.glob("*.txt")):
        with path.open("r", encoding="utf-8") as fh:
            while True:
                chunk = fh.read(chunk_chars)
                if not chunk:
                    break
                yield chunk
                emitted += len(chunk.encode("utf-8"))
                if emitted >= max_bytes:
                    return


def mixed_corpus(corpora_root: Path, plan: dict[str, int], seed: int = 0) -> Iterator[str]:
    """Interleave the per-language chunk streams (shuffled) into one training stream."""
    rng = random.Random(seed)
    streams = [iter_language_text(corpora_root / lang, budget)
               for lang, budget in plan.items() if budget > 0]
    while streams:
        idx = rng.randrange(len(streams))
        try:
            yield next(streams[idx])
        except StopIteration:
            streams.pop(idx)


def count_multispace_tokens(tok: Tokenizer, min_spaces: int = 2) -> int:
    """Count learned vocab tokens that are runs of >= min_spaces spaces (indent merges)."""
    n = 0
    for token in tok.get_vocab():
        stripped = token.lstrip(BYTE_NEWLINE)
        if stripped and stripped.count(BYTE_SPACE) >= min_spaces and set(stripped) <= {BYTE_SPACE}:
            n += 1
    return n


def train_candidate(corpora_root: Path, out_path: Path, *, vocab_size: int,
                    total_budget: int, seed: int = 0) -> FastTokenizer:
    """Train one candidate on the mixed corpus and save it as a FastTokenizer-loadable json."""
    available = {}
    for lang in (*CODE_LANGS, *GLUE_LANGS, *PROSE_LANGS):
        lang_dir = corpora_root / lang
        if not lang_dir.is_dir():
            raise FileNotFoundError(
                f"missing corpus for {lang!r} at {lang_dir}; run build_code_tokenizer_corpora.py")
        nbytes = sum(p.stat().st_size for p in lang_dir.glob("*.txt"))
        if nbytes == 0:
            raise ValueError(f"empty corpus for {lang!r} at {lang_dir}")
        available[lang] = nbytes

    plan = plan_byte_budget(available, total_budget)
    print(f"  byte budget plan (MB): "
          f"{ {k: round(v/1e6, 1) for k, v in sorted(plan.items())} }", flush=True)

    tok = build_code_tokenizer(vocab_size)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(mixed_corpus(corpora_root, plan, seed=seed), trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))

    # Sanity: digit splitting + round-trip + indentation merges actually happened.
    ft = FastTokenizer.load(str(out_path))
    assert len(ft.encode("12345")) == 5, "digit splitting not active"
    assert ft.decode(ft.encode("def f(x):\n    return x")) == "def f(x):\n    return x"
    n_indent = count_multispace_tokens(tok)
    print(f"  saved {out_path} vocab={ft.vocab_size} multispace_tokens={n_indent}", flush=True)
    if n_indent == 0:
        raise ValueError("no multi-space indentation merges learned -- mix not code-heavy?")
    return ft


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", default="data/corpora/code-samples")
    ap.add_argument("--out-dir", default="data/tokenizers")
    ap.add_argument("--budget-bytes", type=int, default=800_000_000,
                    help="total bytes of training text to sample across the mix")
    ap.add_argument("--baseline", default="runs/1b/tokenizer.json",
                    help="existing 32k FineWeb tokenizer to stage as the baseline")
    ap.add_argument("--copy-baseline", action="store_true",
                    help="copy the baseline tokenizer into --out-dir for the study")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", choices=["code-49k", "code-32k"],
                    help="train only these candidates (default: both)")
    args = ap.parse_args()

    corpora_root = Path(args.corpora)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = {"code-49k": 49_152, "code-32k": 32_000}
    for name in (args.only or candidates):
        print(f"[{name}] training vocab={candidates[name]} ...", flush=True)
        train_candidate(corpora_root, out_dir / f"{name}.json",
                        vocab_size=candidates[name], total_budget=args.budget_bytes,
                        seed=args.seed)

    if args.copy_baseline:
        dst = out_dir / "fineweb-32k-baseline.json"
        shutil.copyfile(args.baseline, dst)
        print(f"staged baseline {args.baseline} -> {dst}", flush=True)


if __name__ == "__main__":
    main()
