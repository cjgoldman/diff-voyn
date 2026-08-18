"""Task 1.2 acceptance mechanics: preset sizes, bidirectionality, language
conditioning (conditional ≠ unconditional pathway), and the 10% conditioning
dropout verified from the logged rate."""

import torch

from diff_voyn.data.loader import NULL_LANG_INDEX
from diff_voyn.infra.config import ModelConfig, model_preset
from diff_voyn.model.backbone import Backbone, language_dropout_rate
from diff_voyn.vocab import VOCAB_SIZE

TINY = ModelConfig(
    n_layers=2, d_model=64, n_heads=4, d_ffn=128, dropout=0.0, seq_len=128
)


def test_preset_param_counts():
    n85 = Backbone(model_preset("85m")).n_params()
    n25 = Backbone(model_preset("25m")).n_params()
    assert 78e6 < n85 < 92e6, n85  # 12L/d768/12H/ffn2048 ≈ 85M
    assert 15e6 < n25 < 30e6, n25  # 6L/d512/8H/ffn1408 (design's "~25M" sibling)


def test_forward_shape_and_finiteness():
    model = Backbone(TINY).eval()
    z = torch.randint(6, 31, (3, 96))
    logits = model(z, torch.zeros(3, dtype=torch.long))
    assert logits.shape == (3, 96, VOCAB_SIZE)
    keep = torch.ones(VOCAB_SIZE, dtype=torch.bool)
    keep[1] = False  # MASK column is −inf by SUBS
    assert torch.isfinite(logits[..., keep]).all()


def test_bidirectional_context():
    """No causal mask: perturbing the last token must move logits at pos 0."""
    torch.manual_seed(0)
    model = Backbone(TINY).eval()
    z = torch.randint(6, 31, (1, 64))
    lang = torch.zeros(1, dtype=torch.long)
    a = model(z, lang)
    z2 = z.clone()
    z2[0, -1] = (z2[0, -1] - 6 + 1) % 25 + 6
    b = model(z2, lang)
    assert not torch.allclose(a[0, 0], b[0, 0])


def test_conditional_vs_unconditional_pathways_differ():
    torch.manual_seed(0)
    model = Backbone(TINY).eval()
    z = torch.randint(6, 31, (2, 64))
    cond = model(z, torch.zeros(2, dtype=torch.long))
    uncond = model(z, torch.full((2,), NULL_LANG_INDEX, dtype=torch.long))
    assert not torch.allclose(cond, uncond)


def test_eval_mode_deterministic_and_no_conditioning_dropout():
    model = Backbone(TINY).eval()
    z = torch.randint(6, 31, (4, 64))
    lang = torch.ones(4, dtype=torch.long)
    assert torch.equal(model(z, lang), model(z, lang))


def test_conditioning_dropout_rate_near_10pct():
    torch.manual_seed(0)
    model = Backbone(TINY).train()
    total, n_batches = 0.0, 30
    for _ in range(n_batches):
        z = torch.randint(6, 31, (256, 8))
        model(z, torch.zeros(256, dtype=torch.long))
        total += language_dropout_rate(model)
    rate = total / n_batches
    assert abs(rate - 0.1) < 0.02, rate
