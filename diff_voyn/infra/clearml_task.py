"""ClearML integration (task 0.6, cross-cutting X.1).

Server: ``clearml.acet.network`` (credentials from the host environment via
``CLEARML_API_*`` variables — no clearml.conf needed).

Every training/eval run gets one Task capturing config, the run manifest, and
per-language held-out NELBO scalars — the canary metric consulted at every
gate. The scalar layout is fixed here so all dashboards agree:

- title ``"heldout_nelbo_bits_per_char"``, one series per language;
- title ``"language_sampling_weights"``, one series per language (logged once
  per run so τ-balancing is auditable, task 0.5).
"""

from __future__ import annotations

from clearml import Task

from .config import RunConfig
from .manifest import build_run_manifest

PROJECT_NAME = "diff-voyn"
NELBO_TITLE = "heldout_nelbo_bits_per_char"


def init_task(cfg: RunConfig, data_root, tags: list[str] | None = None) -> Task:
    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=cfg.run_name,
        tags=[cfg.phase] + (tags or []),
        auto_connect_frameworks={"pytorch": True},
        reuse_last_task_id=False,
    )
    task.connect_configuration(build_run_manifest(cfg, data_root), name="run_manifest")
    task.connect(cfg, name="run_config")
    return task


def report_language_weights(task: Task, weights: dict[str, float]) -> None:
    logger = task.get_logger()
    for lang, w in weights.items():
        logger.report_scalar("language_sampling_weights", lang, w, iteration=0)


def report_per_language_nelbo(
    task: Task, nelbo_bits: dict[str, float], iteration: int
) -> None:
    """The first-class canary metric (task 0.6): one series per language."""
    logger = task.get_logger()
    for lang, bits in nelbo_bits.items():
        logger.report_scalar(NELBO_TITLE, lang, bits, iteration=iteration)
