"""Task 3.4 reference model: the char-AR transformer is causal, anchored, and
scores on exactly the characters the diffusion NELBO averages over."""

import math

import torch

from diff_voyn.infra.config import ModelConfig
from diff_voyn.model.ar_reference import ARConfig, CharARLM, ar_loss
from diff_voyn.model.backbone import Backbone
from diff_voyn.vocab import LETTER_IDS, VOCAB_SIZE

TINY = ARConfig(n_layers=2, d_model=64, n_heads=4, d_ffn=128, dropout=0.0, seq_len=32)


def letters(b: int, l: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    pool = torch.tensor(sorted(LETTER_IDS))
    return pool[torch.randint(len(pool), (b, l), generator=g)]


def test_causality_future_chars_do_not_leak():
    torch.manual_seed(0)
    model = CharARLM(TINY).eval()
    x = letters(2, 16)
    y = x.clone()
    y[:, 10:] = letters(2, 6, seed=1)  # perturb positions 10..15 only
    with torch.no_grad():
        lx, ly = model(x), model(y)
    # logits at position i depend on BOS + x[:i] only ⇒ positions <= 10 agree
    assert torch.allclose(lx[:, :11], ly[:, :11], atol=1e-5)
    assert not torch.allclose(lx[:, 11:], ly[:, 11:])


def test_uniform_anchor_is_five_bits():
    model = CharARLM(TINY).eval()
    with torch.no_grad():
        model.head.weight.zero_()
    bits = model.nll_bits_per_char(letters(3, 32))
    assert torch.allclose(bits, torch.full((3,), math.log2(VOCAB_SIZE)), atol=1e-5)


def test_loss_matches_per_window_nll_and_overfits_a_pattern():
    torch.manual_seed(0)
    model = CharARLM(TINY)
    x = letters(4, 32)
    loss = ar_loss(model, x)
    assert torch.isclose(loss, model.nll_nats(x).sum() / x.numel())
    # A repeating pattern is learnable by a causal model to near-zero loss.
    pattern = letters(1, 8).repeat(1, 4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(200):
        opt.zero_grad()
        ar_loss(model, pattern).backward()
        opt.step()
    model.eval()
    bits = model.nll_bits_per_char(pattern).item()
    assert bits < 1.0, bits


def test_backbone_attention_stays_bidirectional():
    """The causal flag must default off: the instrument is an encoder."""
    cfg = ModelConfig(
        n_layers=1, d_model=32, n_heads=4, d_ffn=64, dropout=0.0, seq_len=16
    )
    torch.manual_seed(0)
    bb = Backbone(cfg).eval()
    assert all(not blk.attn.causal for blk in bb.blocks)
    x = letters(1, 16)
    y = x.clone()
    y[:, -1] = (x[:, -1] + 1) % len(LETTER_IDS) + min(LETTER_IDS)
    lang = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        assert not torch.allclose(bb(x, lang)[:, 0], bb(y, lang)[:, 0])
