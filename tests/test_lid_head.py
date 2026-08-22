"""Task 4.1 acceptance mechanics: head shape/classes, stop-gradient exactness
(backbone grads exactly zero in Phase-B mode, non-zero when released), CRN
determinism of the pooled features, λ schedule, checkpoint round-trip, and
a wiring check — the head trains to >99% on a trivially separable toy
problem in a handful of steps."""

import torch

from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.infra.checkpoint import load_lid_head, save_checkpoint
from diff_voyn.infra.config import ModelConfig
from diff_voyn.infra.ema import EMA
from diff_voyn.model.backbone import Backbone
from diff_voyn.model.lid_head import (
    ABSTAIN_CLASS,
    LID_CLASSES,
    N_LID_CLASSES,
    LIDHead,
    LIDHeadConfig,
    backbone_grad_norm,
    lambda_schedule,
    lid_logits,
    lid_loss,
    pooled_features,
    predict,
)

TINY = ModelConfig(
    n_layers=2, d_model=64, n_heads=4, d_ffn=128, dropout=0.0, seq_len=128
)


def _models():
    torch.manual_seed(0)
    return Backbone(TINY), LIDHead(LIDHeadConfig(d_model=64, hidden=32, dropout=0.0))


def test_classes_are_languages_plus_abstain():
    assert LID_CLASSES == ("latin", "italian", "german", "abstain")
    assert ABSTAIN_CLASS == len(LANG_TO_INDEX) == 3
    assert N_LID_CLASSES == 4


def test_forward_shape():
    bb, head = _models()
    ids = torch.randint(6, 31, (5, 64))
    logits = lid_logits(bb, head, ids, g=torch.Generator().manual_seed(0))
    assert logits.shape == (5, N_LID_CLASSES)
    assert torch.isfinite(logits).all()


def test_stop_gradient_leaves_backbone_grads_exactly_zero():
    bb, head = _models()
    ids = torch.randint(6, 31, (4, 64))
    labels = torch.tensor([0, 1, 2, 3])
    logits = lid_logits(bb, head, ids, stop_gradient=True)
    lid_loss(logits, labels).backward()
    assert all(p.grad is None for p in bb.parameters())
    assert backbone_grad_norm(bb) == 0.0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_released_gradient_reaches_backbone():
    bb, head = _models()
    ids = torch.randint(6, 31, (4, 64))
    labels = torch.tensor([0, 1, 2, 3])
    logits = lid_logits(bb, head, ids, stop_gradient=False)
    lid_loss(logits, labels).backward()
    assert backbone_grad_norm(bb) > 0.0
    assert bb.blocks[0].attn.qkv.weight.grad.abs().sum() > 0


def test_features_are_unconditional_and_crn_deterministic():
    bb, _head = _models()
    bb.eval()
    ids = torch.randint(6, 31, (3, 64))
    f1 = pooled_features(bb, ids, (0.0, 0.3), g=torch.Generator().manual_seed(1))
    f2 = pooled_features(bb, ids, (0.0, 0.3), g=torch.Generator().manual_seed(1))
    f3 = pooled_features(bb, ids, (0.0, 0.3), g=torch.Generator().manual_seed(2))
    assert torch.equal(f1, f2)
    assert not torch.equal(f1, f3)  # masks differ → features differ


def test_lambda_schedule_ramps_then_holds():
    assert lambda_schedule(0, 100, 0.05) == 0.0
    assert abs(lambda_schedule(50, 100, 0.05) - 0.025) < 1e-12
    assert lambda_schedule(100, 100, 0.05) == 0.05
    assert lambda_schedule(1000, 100, 0.05) == 0.05
    assert lambda_schedule(7, 0, 0.05) == 0.05


def test_head_checkpoint_roundtrip_standalone_and_joint(tmp_path):
    bb, head = _models()
    ema = EMA(head, 0.5)
    with torch.no_grad():
        head.fc2.bias.add_(1.0)
    ema.update(head)
    head.log_temperature.fill_(0.3)
    extra = {"lid_head_config": head.cfg.to_dict(), "config": {"phase": "x"}}
    save_checkpoint(tmp_path / "head.pt", model=head, ema=ema, step=3, extra=extra)
    h_ema, meta = load_lid_head(tmp_path / "head.pt")
    assert meta["weights"] == "ema" and not meta["joint_checkpoint"]
    assert torch.allclose(h_ema.fc2.bias, ema.shadow["fc2.bias"])
    h_raw, _ = load_lid_head(tmp_path / "head.pt", ema=False)
    assert torch.allclose(h_raw.fc2.bias, head.fc2.bias)
    assert abs(h_raw.temperature - float(torch.tensor(0.3).exp())) < 1e-6
    save_checkpoint(
        tmp_path / "joint.pt", model=bb, step=4, extra=extra, lid_head=head, lid_ema=ema
    )
    h_j, meta_j = load_lid_head(tmp_path / "joint.pt")
    assert meta_j["joint_checkpoint"] and meta_j["step"] == 4
    assert torch.allclose(h_j.fc2.bias, ema.shadow["fc2.bias"])


def test_head_learns_separable_toy_problem():
    """Wiring check (4.1: 'anything less than >99% = wiring bug'): four
    'languages' that are constant letters must be separated in a few steps
    behind the stop-gradient."""
    bb, head = _models()
    bb.eval()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    ids = torch.stack([torch.full((48,), 6 + 5 * c) for c in range(4)] * 4)
    labels = torch.tensor(list(range(4)) * 4)
    g = torch.Generator().manual_seed(0)
    for _ in range(60):
        logits = lid_logits(bb, head, ids, g=g, stop_gradient=True)
        loss = lid_loss(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    probs = predict(bb, head, ids, seed=0, autocast=False)
    assert (probs.argmax(-1) == labels).float().mean() == 1.0
