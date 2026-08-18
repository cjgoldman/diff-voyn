"""Task 0.8: IVTFF parsing and Currier split. Skipped if files not fetched."""

import pytest

from diff_voyn.ciphers.external import data_root
from diff_voyn.vms.ingest import build_dialect_streams, parse_ivtff

IT_PATH = data_root() / "raw" / "vms" / "IT2a-n.txt"

pytestmark = pytest.mark.skipif(
    not IT_PATH.exists(), reason="voynich.nu files not fetched"
)


def test_pages_have_currier_tags():
    pages = parse_ivtff(IT_PATH)
    assert len(pages) > 200  # ~225 pages with text
    tagged = [p for p in pages if p["currier"] in ("A", "B")]
    assert len(tagged) > 150


def test_counts_reconcile_with_published_figures():
    result = build_dialect_streams(IT_PATH)
    c = result["counts"]
    # Published: ~37k words / ~230k chars for the full manuscript.
    assert 30_000 < c["total_words"] < 45_000, c["total_words"]
    assert 150_000 < c["total_chars_pre_strip"] < 260_000, c["total_chars_pre_strip"]


def test_dialects_never_pooled_and_stripped():
    result = build_dialect_streams(IT_PATH)
    for dialect in ("A", "B"):
        stream = result["streams"][dialect]
        assert len(stream) > 10_000
        assert "." not in stream and "," not in stream
        assert not any(ch.isspace() for ch in stream)
    # A and B are separate outputs; nothing merges them.
    assert set(result["streams"]) == {"A", "B", "unassigned"}
