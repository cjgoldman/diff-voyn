"""Training infrastructure — task 0.6.

Config + run manifests, checkpoint/resume with RNG capture, EMA (0.9999),
ClearML task registration, and the per-language held-out NELBO canary metric.
"""

from .checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
from .config import RunConfig, config_hash  # noqa: F401
from .ema import EMA  # noqa: F401
from .manifest import build_run_manifest  # noqa: F401
from .nelbo import (  # noqa: F401
    estimate_nelbo_bits_per_char,
    per_window_nelbo_bits,
)
