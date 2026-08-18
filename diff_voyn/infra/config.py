"""Run configuration — versioned, hashable, YAML-round-trippable (task 0.6)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..corpus.assemble import CORPUS_VERSION
from ..corpus.splits import SPLITS_VERSION
from ..normalize import NORMALIZER_VERSION
from ..vocab import VOCAB_VERSION


@dataclass
class ModelConfig:
    """Backbone dims. Defaults are the 25M sibling (design §3); the 85M main
    model is 12 layers / d_model 768 / 12 heads / ffn 2048."""

    n_layers: int = 6
    d_model: int = 512
    n_heads: int = 8
    d_ffn: int = 1408
    dropout: float = 0.1
    seq_len: int = 1024
    lang_cond_dropout: float = 0.1


def model_preset(name: str) -> ModelConfig:
    """The two frozen backbone sizes of design §3 (task 1.2)."""
    presets = {
        "25m": ModelConfig(),
        "85m": ModelConfig(n_layers=12, d_model=768, n_heads=12, d_ffn=2048),
    }
    return presets[name]


@dataclass
class DataConfig:
    corpus_version: str = CORPUS_VERSION
    splits_version: str = SPLITS_VERSION
    vocab_version: str = VOCAB_VERSION
    normalizer_version: str = NORMALIZER_VERSION
    sampling_temperature: float = 0.7
    seq_len: int = 1024


@dataclass
class OptimConfig:
    """Design §7.5 defaults."""

    lr: float = 3e-4
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    batch_chars: int = 500_000
    ema_decay: float = 0.9999
    precision: str = "bf16-mixed"


@dataclass
class RunConfig:
    run_name: str = "unnamed"
    phase: str = "phase0"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_yaml(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True))

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        d = yaml.safe_load(path.read_text())
        return cls(
            run_name=d["run_name"],
            phase=d["phase"],
            seed=d["seed"],
            model=ModelConfig(**d["model"]),
            data=DataConfig(**d["data"]),
            optim=OptimConfig(**{**d["optim"], "betas": tuple(d["optim"]["betas"])}),
        )


def config_hash(cfg: RunConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True, default=list).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
