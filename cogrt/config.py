"""Canonical configuration.

TOML is the only configuration format. Every section maps to a dataclass;
unknown keys are rejected so that typos do not silently fall back to defaults.
"""

from __future__ import annotations

import copy
import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints


class ConfigError(ValueError):
    pass


@dataclass
class RuntimeConfig:
    data_dir: str = "data"
    log_level: str = "INFO"


@dataclass
class ModelConfig:
    backend: str = "mock"  # mock | llamacpp | transformers | remote
    model_id: str = "mock/deterministic"
    revision: str = ""
    path: str = ""  # GGUF file or HF directory
    quantization: str = ""
    context_length: int = 1024
    n_threads: int = 0  # 0 = auto (physical cores)
    device: str = "cpu"  # cpu | cuda | auto
    n_gpu_layers: int = 0  # llama.cpp offload; -1 = all
    dtype: str = "bfloat16"  # transformers backend
    seed: int = 1234
    remote_url: str = ""
    remote_timeout_s: float = 120.0


@dataclass
class GenerationConfig:
    max_tokens: int = 48
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=lambda: ["\n"])


@dataclass
class SchedulerConfig:
    autonomous: bool = True
    interval_s: float = 60.0
    min_interval_s: float = 5.0
    max_interval_s: float = 3600.0
    backoff_factor: float = 2.0
    max_cycle_s: float = 120.0
    max_tokens_per_operation: int = 64
    max_model_calls_per_cycle: int = 3
    max_cycles_per_hour: int = 30
    token_budget_per_hour: int = 4000
    dormant_after_unproductive: int = 4
    consolidate_every: int = 5
    user_preempts: bool = True
    poll_s: float = 0.5


@dataclass
class CycleConfig:
    policies: dict[str, float] = field(
        default_factory=lambda: {
            "unresolved": 1.0,
            "contradiction": 1.5,
            "memory": 0.7,
            "association": 0.7,
            "random": 0.2,
        }
    )
    n_candidates: int = 2
    context_items: int = 4
    duplicate_threshold: float = 0.8
    novelty_threshold: float = 0.35
    relevance_threshold: float = 0.05
    min_chars: int = 12
    max_chars: int = 300
    max_hypothesis_confidence: float = 0.5
    initial_hypothesis_confidence: float = 0.3
    revisit_cooldown_cycles: int = 3
    # Derivation depth: seed/user items are 0, an item generated from depth-d
    # inputs has depth d+1. Items at max depth are not used as generation
    # focus, which bounds chains of hypotheses built on hypotheses.
    max_inference_depth: int = 2
    max_example_leakage: float = 0.34


@dataclass
class MemoryConfig:
    max_semantic: int = 2000
    max_episodic: int = 5000
    max_unresolved_open: int = 200
    episodic_retention_days: float = 30.0
    max_events: int = 20000


@dataclass
class WorkspaceConfig:
    max_items_per_slot: int = 8
    recent_topics: int = 16


@dataclass
class SecurityConfig:
    allow_network: bool = False
    export_dir: str = "exports"


@dataclass
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    source_path: str = ""

    @property
    def data_path(self) -> Path:
        p = Path(self.runtime.data_dir)
        if not p.is_absolute() and self.source_path:
            p = Path(self.source_path).parent / p
        return p.resolve()

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("source_path", None)
        return d


def _coerce(section: str, key: str, expected: Any, value: Any) -> Any:
    origin = getattr(expected, "__origin__", None)
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected in (int, float, str, bool):
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ConfigError(f"[{section}] {key}: expected {expected.__name__}, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"[{section}] {key}: expected list")
        return list(value)
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"[{section}] {key}: expected table")
        return {str(k): float(v) for k, v in value.items()}
    return value


def _apply(obj: Any, section: str, values: dict[str, Any]) -> None:
    hints = get_type_hints(type(obj))
    for key, value in values.items():
        if key not in hints:
            raise ConfigError(f"[{section}] unknown key {key!r}")
        setattr(obj, key, _coerce(section, key, hints[key], value))


def from_dict(data: dict[str, Any], base: Config | None = None) -> Config:
    cfg = copy.deepcopy(base) if base else Config()
    for section, values in data.items():
        if section == "source_path":
            continue
        if not hasattr(cfg, section) or not isinstance(values, dict):
            raise ConfigError(f"unknown section [{section}]")
        _apply(getattr(cfg, section), section, values)
    validate(cfg)
    return cfg


def load(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if path:
        p = Path(path)
        with p.open("rb") as fh:
            cfg = from_dict(tomllib.load(fh))
        cfg.source_path = str(p.resolve())
    if overrides:
        src = cfg.source_path
        cfg = from_dict(overrides, cfg)
        cfg.source_path = src
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.model.backend not in {"mock", "llamacpp", "transformers", "remote"}:
        raise ConfigError(f"unknown backend {cfg.model.backend!r}")
    if cfg.model.backend == "remote" and not cfg.security.allow_network:
        raise ConfigError("remote backend requires [security] allow_network = true")
    if cfg.model.device not in {"cpu", "cuda", "auto"}:
        raise ConfigError("model.device must be cpu, cuda or auto")
    s = cfg.scheduler
    if s.min_interval_s <= 0 or s.max_interval_s < s.min_interval_s:
        raise ConfigError("scheduler intervals are inconsistent")
    if s.max_tokens_per_operation <= 0 or s.max_model_calls_per_cycle <= 0:
        raise ConfigError("scheduler budgets must be positive")
    c = cfg.cycle
    if not 0 <= c.initial_hypothesis_confidence <= c.max_hypothesis_confidence <= 1:
        raise ConfigError("hypothesis confidences must satisfy 0 <= initial <= max <= 1")
    if not c.policies or any(w < 0 for w in c.policies.values()):
        raise ConfigError("cycle.policies must be non-empty with non-negative weights")
