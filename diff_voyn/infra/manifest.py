"""Run manifests (task 0.6): everything needed to reproduce or audit a run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..vocab import spec_hash
from .config import RunConfig, config_hash


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — git may be absent in some envs
        return "unknown"


def build_run_manifest(
    cfg: RunConfig,
    data_root: Path,
    repo_root: Path = Path("/workspace"),
    language_weights: dict[str, float] | None = None,
) -> dict:
    corpus_dir = data_root / "corpora" / cfg.data.corpus_version
    splits_path = corpus_dir / f"splits_{cfg.data.splits_version}.json"
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "run_name": cfg.run_name,
        "phase": cfg.phase,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "config_hash": config_hash(cfg),
        "vocab_spec_sha256": spec_hash(),
        "code_git_sha": _git_sha(repo_root),
        "corpus_manifest": str(corpus_dir / "manifest.json"),
        "splits_file": str(splits_path),
        "language_sampling_weights": language_weights or {},
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        manifest["torch"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    return manifest


def write_run_manifest(manifest: dict, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
