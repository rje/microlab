"""build_ultrafeedback_prefs.py: pure normalization of HF ultrafeedback_binarized rows into
the {prompt, chosen, rejected} JSONL dpo.py consumes, plus the block-size length filter.
No network: mappers are pure; the loader is a thin streaming wrapper (untested here)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from microlab.model.reference.sft import format_chat

_SPEC = importlib.util.spec_from_file_location(
    "uf_script", Path(__file__).resolve().parents[2] / "scripts" / "build_ultrafeedback_prefs.py")
uf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(uf)


def _row(prompt="Explain tides.", chosen="The moon's gravity.", rejected="Magic."):
    return {
        "prompt": prompt,
        "chosen": [{"role": "user", "content": prompt},
                   {"role": "assistant", "content": chosen}],
        "rejected": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": rejected}],
    }


def test_normalize_maps_fields_and_applies_chat_template():
    out = uf.normalize_ultrafeedback(_row())
    expected_prompt, _ = format_chat("Explain tides.", "")
    assert out == {"prompt": expected_prompt, "chosen": "The moon's gravity.",
                   "rejected": "Magic."}


def test_normalize_rejects_empty_and_identical_pairs():
    assert uf.normalize_ultrafeedback(_row(chosen="  ")) is None
    assert uf.normalize_ultrafeedback(_row(rejected="")) is None
    assert uf.normalize_ultrafeedback(_row(chosen="same", rejected="same")) is None
    assert uf.normalize_ultrafeedback({"prompt": "x", "chosen": [], "rejected": []}) is None


class _ByteTok:
    def encode(self, s):
        return list(s.encode("utf-8"))


def test_fits_block_filters_long_pairs():
    tok = _ByteTok()
    short = {"prompt": "p" * 10, "chosen": "c" * 10, "rejected": "r" * 10}
    long_c = {"prompt": "p" * 10, "chosen": "c" * 300, "rejected": "r" * 10}
    assert uf.fits_block(short, tok, block_size=64)
    # the LONGER side must fit: chosen at 310 total > 64 -> filtered
    assert not uf.fits_block(long_c, tok, block_size=64)
