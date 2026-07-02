"""muP oracle tests: the zero-shot hyperparameter-transfer scaling table."""

import math

from microlab.model.reference.scaling import mup_attn_scale, mup_multipliers


def test_identity_at_base_width():
    m = mup_multipliers(256, 256)
    assert all(math.isclose(v, 1.0) for v in m.values())


def test_doubling_width():
    m = mup_multipliers(256, 512)
    assert math.isclose(m["width_mult"], 2.0)
    assert math.isclose(m["hidden_lr_mult"], 0.5)
    assert math.isclose(m["hidden_init_std_mult"], 1 / math.sqrt(2))
    assert math.isclose(m["output_logit_mult"], 0.5)
    assert math.isclose(m["embedding_lr_mult"], 1.0)


def test_attn_scale_is_one_over_d():
    assert math.isclose(mup_attn_scale(64), 1 / 64)
