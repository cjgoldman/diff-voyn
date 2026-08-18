"""Corpus assembly (tasks 0.2/0.3) and held-out splits (task 0.4)."""

from .assemble import assemble_language, discover_documents  # noqa: F401
from .splits import carve_splits, load_splits  # noqa: F401
