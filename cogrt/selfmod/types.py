from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

InterventionMethod = Literal["sparse_perturbation"]
ExperimentDecision = Literal["accept", "reject", "inconclusive", "no_effect", "regressed", "failed"]
ExperimentStatus = Literal["designed", "running", "completed", "failed", "aborted"]


@dataclass
class WeightHypothesis:
    """Structured experimental proposal — not a trusted fact."""

    problem: str
    proposed_change: str
    mechanism: str
    expected_effect: str
    expected_collateral: str
    measurements: str
    intervention: InterventionMethod = "sparse_perturbation"
    tensor_key: str = ""
    layer_index: int = 0
    n_params: int = 32
    magnitude: float = 1e-4
    seed: int = 0
    probe_prompts: list[str] = field(default_factory=lambda: [
        "The capital of France is",
        "In winter the clock tower runs",
    ])
    accept_min_logprob_delta: float = 1e-7
    reject_max_repetition: float = 0.85
    max_regression_ratio: float = 0.02

    def to_meta(self) -> dict[str, Any]:
        return {
            "domain": "weight_self_mod",
            "intervention": self.intervention,
            "tensor_key": self.tensor_key,
            "layer_index": self.layer_index,
            "n_params": self.n_params,
            "magnitude": self.magnitude,
            "seed": self.seed,
            "probe_prompts": self.probe_prompts,
            "accept_min_logprob_delta": self.accept_min_logprob_delta,
            "reject_max_repetition": self.reject_max_repetition,
            "max_regression_ratio": self.max_regression_ratio,
            "problem": self.problem,
            "proposed_change": self.proposed_change,
            "mechanism": self.mechanism,
            "expected_effect": self.expected_effect,
            "expected_collateral": self.expected_collateral,
            "measurements": self.measurements,
        }

    @classmethod
    def from_item_meta(cls, meta: dict[str, Any]) -> "WeightHypothesis":
        if meta.get("domain") != "weight_self_mod":
            raise ValueError("not a weight self-mod hypothesis")
        return cls(
            problem=meta.get("problem", ""),
            proposed_change=meta.get("proposed_change", ""),
            mechanism=meta.get("mechanism", ""),
            expected_effect=meta.get("expected_effect", ""),
            expected_collateral=meta.get("expected_collateral", ""),
            measurements=meta.get("measurements", ""),
            tensor_key=meta.get("tensor_key", ""),
            layer_index=int(meta.get("layer_index", 0)),
            n_params=int(meta.get("n_params", 32)),
            magnitude=float(meta.get("magnitude", 1e-4)),
            seed=int(meta.get("seed", 0)),
            probe_prompts=list(meta.get("probe_prompts", [])),
            accept_min_logprob_delta=float(meta.get("accept_min_logprob_delta", 1e-7)),
            reject_max_repetition=float(meta.get("reject_max_repetition", 0.85)),
            max_regression_ratio=float(meta.get("max_regression_ratio", 0.02)),
        )
