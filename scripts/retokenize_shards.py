#!/usr/bin/env python
"""Retokenise an existing shard corpus into a different vocabulary, by decode -> encode.

Needed because our web data (FineWeb) is tokenised with the 32k vocab while the code
specialist uses code-49k, and a mixed corpus must share one tokenizer. Re-downloading and
re-tokenising FineWeb from source would be far slower than decoding what we already have.

Decode/encode is lossless for our purposes: BPE decode reconstructs the original text
(barring unpaired surrogates, which the source filter already removed), and the round-trip
is verified on a sample before any real work is done -- see --verify.

Documents are split on the source tokenizer's EOT and re-emitted with the target's, so
document boundaries survive the change of vocabulary rather than being smeared.

Resumable and progressive (house rule: long jobs write as they go).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def eot_id(tok: Tokenizer) -> int:
    for name in ("<|endoftext|>", "<|end_of_text|>", "<eos>"):
        i = tok.token_to_id(name)
        if i is not None:
            return i
    raise SystemExit("could not find an end-of-text token in the tokenizer")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="TARGET tokenizer json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--docs-per-batch", type=int, default=2000)
    ap.add_argument("--verify", action="store_true",
                    help="round-trip check on a sample, then exit without writing")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    src_tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    dst_tok = Tokenizer.from_file(a.tokenizer)
    src_eot, dst_eot = eot_id(src_tok), eot_id(dst_tok)

    man = json.loads((src / f"{a.split}-manifest.json").read_text())
    arrays = [np.memmap(src / s["file"], dtype=np.uint16, mode="r") for s in man["shards"]]

    if a.verify:
        sample = np.asarray(arrays[0][:200_000])
        pos = np.flatnonzero(sample == src_eot)[:20]
        ok = 0
        for i in range(len(pos) - 1):
            ids = sample[pos[i] + 1:pos[i + 1]].tolist()
            text = src_tok.decode(ids)
            ok += (src_tok.encode(text).ids == ids)
        print(f"round-trip through the SOURCE tokenizer: {ok}/{len(pos)-1} documents exact")
        print("(a decode->encode mismatch here would mean the source text cannot be "
              "recovered, which would make any retokenisation lossy)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / f"retok-state-{a.split}.json"
    done = json.loads(state_path.read_text()) if state_path.exists() else \
        {"shard_i": 0, "arr": 0, "pos": 0, "written": 0}
    manifest = []
    if (out / f"{a.split}-manifest.json").exists():
        manifest = json.loads((out / f"{a.split}-manifest.json").read_text())["shards"]

    buf: list[int] = []
    shard_i, written = done["shard_i"], done["written"]
    target = a.max_tokens or sum(len(x) for x in arrays)

    def flush():
        nonlocal buf, shard_i
        if not buf:
            return
        np.asarray(buf, dtype=np.uint16).tofile(out / f"{a.split}-{shard_i:05d}.bin")
        manifest.append({"file": f"{a.split}-{shard_i:05d}.bin", "tokens": len(buf)})
        (out / f"{a.split}-manifest.json").write_text(json.dumps(
            {"split": a.split, "dtype": "uint16", "shards": manifest,
             "total_tokens": sum(s["tokens"] for s in manifest)}, indent=1))
        shard_i += 1
        buf = []

    for ai in range(done["arr"], len(arrays)):
        arr = np.asarray(arrays[ai])
        pos = np.flatnonzero(arr == src_eot)
        start = done["pos"] if ai == done["arr"] else 0
        texts = []
        for i in range(start, len(pos) - 1):
            texts.append(src_tok.decode(arr[pos[i] + 1:pos[i + 1]].tolist()))
            if len(texts) >= a.docs_per_batch:
                for enc in dst_tok.encode_batch(texts):
                    buf.extend(enc.ids)
                    buf.append(dst_eot)
                    written += len(enc.ids) + 1
                texts = []
                if len(buf) >= a.shard_tokens:
                    flush()
                    state_path.write_text(json.dumps(
                        {"shard_i": shard_i, "arr": ai, "pos": i, "written": written}))
                    print(f"  {written:,}/{target:,} tokens ({written/target*100:.1f}%)",
                          flush=True)
                if written >= target:
                    break
        if texts:
            for enc in dst_tok.encode_batch(texts):
                buf.extend(enc.ids)
                buf.append(dst_eot)
                written += len(enc.ids) + 1
        if written >= target:
            break
    flush()
    (out / "tokenizer.json").write_bytes(Path(a.tokenizer).read_bytes())
    state_path.write_text(json.dumps({"shard_i": shard_i, "arr": len(arrays), "pos": 0,
                                      "written": written}))
    print(f"done: {written:,} tokens in {shard_i} shards -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
