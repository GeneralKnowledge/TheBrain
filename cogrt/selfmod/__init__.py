"""Weight-space self-modification experiments (isolated from production inference)."""

from .pipeline import run_experiment, tick_autonomous
from .types import ExperimentDecision, WeightHypothesis

__all__ = ["ExperimentDecision", "WeightHypothesis", "run_experiment", "tick_autonomous"]
