"""Task 0.6 acceptance: kill-and-resume reproduces the loss curve; EMA; config."""

import copy

import torch
from torch import nn

from diff_voyn.infra.checkpoint import load_checkpoint, save_checkpoint
from diff_voyn.infra.config import RunConfig, config_hash
from diff_voyn.infra.ema import EMA


def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 8))


def _train_steps(model, opt, ema, n, losses):
    for _ in range(n):
        x = torch.randn(16, 8)
        loss = ((model(x) - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        losses.append(loss.item())


def test_kill_and_resume_reproduces_loss_curve(tmp_path):
    # Uninterrupted reference run: 10 steps.
    model = _toy_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ema = EMA(model, decay=0.99)
    torch.manual_seed(123)
    ref_losses: list[float] = []
    _train_steps(model, opt, ema, 10, ref_losses)

    # Interrupted run: 5 steps, checkpoint, "kill", resume, 5 more.
    model2 = _toy_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
    ema2 = EMA(model2, decay=0.99)
    torch.manual_seed(123)
    losses: list[float] = []
    _train_steps(model2, opt2, ema2, 5, losses)
    save_checkpoint(
        tmp_path / "ckpt.pt", model=model2, optimizer=opt2, ema=ema2, step=5
    )

    model3 = _toy_model()  # fresh process stand-in
    opt3 = torch.optim.AdamW(model3.parameters(), lr=1e-2)
    ema3 = EMA(model3, decay=0.99)
    state = load_checkpoint(
        tmp_path / "ckpt.pt", model=model3, optimizer=opt3, ema=ema3
    )
    assert state["step"] == 5
    _train_steps(model3, opt3, ema3, 5, losses)

    assert len(losses) == len(ref_losses) == 10
    for a, b in zip(losses, ref_losses):
        assert abs(a - b) < 1e-9, (a, b)
    for k, v in ema.shadow.items():
        assert torch.allclose(v, ema3.shadow[k])


def test_ema_tracks_weights():
    model = _toy_model()
    ema = EMA(model, decay=0.5)
    before = copy.deepcopy(ema.shadow)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    for k, v in ema.shadow.items():
        if v.dtype.is_floating_point and v.requires_grad is False:
            expected = before[k] * 0.5 + (before[k] + 1.0) * 0.5
            assert torch.allclose(v, expected)


def test_config_yaml_round_trip_and_hash(tmp_path):
    cfg = RunConfig(run_name="x", seed=7)
    cfg.to_yaml(tmp_path / "c.yaml")
    cfg2 = RunConfig.from_yaml(tmp_path / "c.yaml")
    assert config_hash(cfg) == config_hash(cfg2)
    cfg2.seed = 8
    assert config_hash(cfg) != config_hash(cfg2)
