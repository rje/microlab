"""Hand-write exercise (Phase 6): the two SFT primitives — prompt loss masking and
masked cross-entropy.

Fill in the ``NotImplementedError`` bodies so ``tests/model/test_student_sft.py`` passes.
They're graded against ``microlab.model.reference.sft``. See docs/hand-write/phase6-sft.md.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def build_sft_example(tok, prompt_text: str, response_text: str) -> tuple[list[int], list[int]]:
    """Tokenize prompt+response into input_ids, and labels where the PROMPT positions are
    masked to IGNORE_INDEX so only the response is supervised.

    input_ids = encode(prompt) + encode(response)
    labels    = [IGNORE_INDEX]*len(prompt) + encode(response)
    """
    raise NotImplementedError("build input_ids and prompt-masked labels")


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over (B, T, V) logits and (B, T) position-aligned labels, ignoring
    IGNORE_INDEX. Apply the causal-LM shift: logits[:, :-1] predict labels[:, 1:] (position
    t predicts token t+1), then F.cross_entropy with ignore_index=IGNORE_INDEX."""
    raise NotImplementedError("shift by one and cross-entropy with ignore_index")
