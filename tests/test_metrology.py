"""Phase-3 metrology: CRN semantics of the scoring harness, per-document
aggregation, the single calibration application point (task 3.1/3.3/3.4)."""

import json

import numpy as np
import pytest
import torch

from diff_voyn.data.loader import LANG_TO_INDEX
from diff_voyn.infra.nelbo import per_window_nelbo_bits
from diff_voyn.metrology import (
    CalibrationTable,
    ScoreSettings,
    calibrate_bits,
    family_of,
    per_document,
    rank_languages,
    score_conditions,
)
from diff_voyn.metrology.scoring import CONDITION_UNCOND
from diff_voyn.vocab import MASK_ID, VOCAB_SIZE

LANGS = tuple(LANG_TO_INDEX)


class LangBiasedModel(torch.nn.Module):
    """Logits prefer the target if its id parity matches the language index —
    a deterministic, language-dependent model so score differences are a
    model property while the Monte-Carlo noise is shared."""

    def forward(self, z_t, lang):
        b, l = z_t.shape
        logits = torch.zeros(b, l, VOCAB_SIZE)
        parity = (int(lang[0]) % 2) * torch.ones(VOCAB_SIZE)
        ids = torch.arange(VOCAB_SIZE)
        logits[..., :] = torch.where(ids % 2 == parity, 1.5, 0.0)
        logits[..., MASK_ID] = -1e9
        return logits


def _ids(n=6, length=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(6, 31, (n, length), generator=g)


def test_samples_per_stratum_one_matches_legacy_estimator():
    ids = _ids()
    a = per_window_nelbo_bits(LangBiasedModel(), ids, 0, n_strata=8, seed=5)
    b = per_window_nelbo_bits(
        LangBiasedModel(), ids, 0, n_strata=8, samples_per_stratum=1, seed=5
    )
    assert torch.equal(a, b)


def test_more_samples_reduce_variance():
    ids = _ids(n=4, length=128)
    model = LangBiasedModel()

    def spread(k):
        runs = [
            per_window_nelbo_bits(
                model, ids, 0, n_strata=4, samples_per_stratum=k, seed=s
            ).numpy()
            for s in range(12)
        ]
        return np.std(np.stack(runs), axis=0).mean()

    assert spread(8) < spread(1)


def test_crn_makes_condition_differences_deterministic():
    """Under CRN every condition of a chunk sees the same masks, so the
    between-condition difference is identical across repeated calls with the
    same seed; independent masking is not."""
    ids = _ids()
    model = LangBiasedModel()
    st = ScoreSettings(n_strata=8, seed=3, batch=4)
    s1 = score_conditions(model, ids, LANGS, settings=st)
    s2 = score_conditions(model, ids, LANGS, settings=st)
    assert np.array_equal(s1, s2)
    assert s1.shape == (6, 3)
    # same seed ⇒ same masks for latin and italian ⇒ difference exact
    a = per_window_nelbo_bits(model, ids[:4], 0, n_strata=8, seed=3).numpy()
    b = per_window_nelbo_bits(model, ids[:4], 1, n_strata=8, seed=3).numpy()
    assert np.allclose(s1[:4, 0] - s1[:4, 1], a - b, atol=1e-5)
    ind = score_conditions(model, ids, LANGS, settings=st, crn=False)
    assert not np.allclose(ind[:, 0] - ind[:, 1], s1[:, 0] - s1[:, 1])


def test_crn_reduces_difference_variance():
    ids = _ids(n=8, length=96)
    model = LangBiasedModel()
    crn, ind = [], []
    for r in range(10):
        st = ScoreSettings(n_strata=4, seed=100 * r, batch=8)
        c = score_conditions(model, ids, LANGS, settings=st)
        i = score_conditions(model, ids, LANGS, settings=st, crn=False)
        crn.append(c[:, 0] - c[:, 2])
        ind.append(i[:, 0] - i[:, 2])
    v_crn = np.stack(crn).var(axis=0).mean()
    v_ind = np.stack(ind).var(axis=0).mean()
    assert v_ind > 5 * v_crn


def test_per_condition_texts_share_masks():
    """The trial-decipherment input shape: one text per condition, same masks."""
    ids = {c: _ids(seed=k) for k, c in enumerate(LANGS)}
    model = LangBiasedModel()
    st = ScoreSettings(n_strata=6, seed=9, batch=6)
    s = score_conditions(model, ids, LANGS, settings=st)
    for j, c in enumerate(LANGS):
        direct = per_window_nelbo_bits(model, ids[c], j, n_strata=6, seed=9).numpy()
        assert np.allclose(s[:, j], direct, atol=1e-5)
    bad = dict(ids)
    bad["german"] = _ids(n=5)
    with pytest.raises(ValueError):
        score_conditions(model, bad, LANGS, settings=st)


def test_unconditional_condition_and_short_windows():
    ids = _ids(n=3, length=50)
    s = score_conditions(
        LangBiasedModel(),
        ids,
        LANGS + (CONDITION_UNCOND,),
        settings=ScoreSettings(n_strata=4),
    )
    assert s.shape == (3, 4) and np.isfinite(s).all()


def test_per_document_aggregation():
    scores = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 0.0]])
    docs = per_document(scores, np.array([0, 0, 1]), ["a", "b"], ("x", "y"))
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].n_windows == 2 and docs[0].mean == {"x": 2.0, "y": 3.0}
    assert docs[0].std["x"] == pytest.approx(np.sqrt(2.0))
    assert docs[0].sem["x"] == pytest.approx(1.0)
    assert docs[1].n_windows == 1 and docs[1].std == {"x": 0.0, "y": 0.0}


def test_calibration_single_application_point():
    offs = {"latin": -0.1, "german": 0.2}
    assert calibrate_bits(2.0, "latin", offs) == pytest.approx(1.9)
    assert calibrate_bits(2.0, "italian", offs) == 2.0  # uncalibrated language
    ranked = rank_languages({"latin": 2.0, "german": 1.85, "italian": 1.95}, offs)
    assert [l for l, _ in ranked] == ["latin", "italian", "german"]
    # the evaluator hook delegates to the same arithmetic
    from diff_voyn.heads.evaluator import EvaluatorBase

    class Ev(EvaluatorBase):
        def __init__(self):
            self.languages = list(LANGS)
            self.calibration_offsets_bits = offs

        def score_fixed(self, *a, **k): ...

        def score_segmental(self, *a, **k): ...

        def as_embedding_frame(self, *a, **k): ...

    nats = -2.0 * 100 * np.log(2.0)  # 2 bits/char over 100 chars
    assert Ev().calibrated_bits_per_char(nats, 100, "german") == pytest.approx(2.2)


def test_calibration_table_roundtrip(tmp_path):
    cal = {
        "calibration_version": "vtest",
        "phase": "phase_x",
        "backbone": {"path": "/x", "step": 7},
        "reference": "r",
        "languages": {
            l: {
                "offset_bits": o,
                "offset_sem": 0.01,
                "nelbo_bits": 2.0,
                "nll_ar_bits": 2.0 - o,
            }
            for l, o in zip(LANGS, (0.1, -0.05, 0.2))
        },
    }
    (tmp_path / "calibration").mkdir()
    CalibrationTable.file_for("vtest", tmp_path).write_text(json.dumps(cal))
    t = CalibrationTable.load("vtest", tmp_path)
    assert t.additive_offsets() == {"latin": -0.1, "italian": 0.05, "german": -0.2}
    assert t.apply(2.5, "german") == pytest.approx(2.3)
    assert t.spread_bits == pytest.approx(0.25)
    assert t.rank({"latin": 2.0, "italian": 2.0, "german": 2.05})[0][0] == "german"


def test_family_granularity():
    assert family_of("latin") == family_of("italian") == "romance"
    assert family_of("german") == "germanic"


def test_report_only_policy(tmp_path):
    from diff_voyn.metrology import derive_report_only

    cal = {
        "calibration_version": "vsrc",
        "phase": "phase_x",
        "backbone": {"path": "/x", "step": 7},
        "reference": "r",
        "languages": {
            l: {
                "offset_bits": o,
                "offset_sem": 0.01,
                "nelbo_bits": 2.0,
                "nll_ar_bits": 2.0 - o,
            }
            for l, o in zip(LANGS, (0.1, -0.05, 0.2))
        },
    }
    (tmp_path / "calibration").mkdir()
    CalibrationTable.file_for("vsrc", tmp_path).write_text(json.dumps(cal))
    derive_report_only("vsrc", "vnull", tmp_path)
    t = CalibrationTable.load("vnull", tmp_path)
    assert t.policy == "report-only"
    assert t.additive_offsets() == {l: 0.0 for l in LANGS}
    assert t.apply(2.5, "german") == 2.5
    assert t.offsets_bits["german"] == pytest.approx(0.2)  # measured kept
    assert t.margin_uncertainty_bits("latin", "italian") == pytest.approx(0.15)
    src = CalibrationTable.load("vsrc", tmp_path)
    assert src.policy == "apply" and src.apply(2.5, "german") == pytest.approx(2.3)
    assert src.margin_uncertainty_bits("latin", "italian") == pytest.approx(
        0.01 * 2**0.5
    )
    with pytest.raises(FileExistsError):
        derive_report_only("vsrc", "vnull", tmp_path)
