from __future__ import annotations

from .evaluate import ProbeMetrics, compare
from .types import ExperimentDecision, WeightHypothesis


def decide_outcome(hyp: WeightHypothesis, baseline: ProbeMetrics, candidate: ProbeMetrics,
                   weight_verified: dict[str, float]) -> tuple[ExperimentDecision, str]:
    if weight_verified.get("max_abs_diff", 0) <= 0:
        return "failed", "candidate identical to parent (invalid intervention)"
    cmp = compare(baseline, candidate)
    if candidate.repetition >= hyp.reject_max_repetition:
        return "regressed", f"repetition {candidate.repetition:.2f} >= {hyp.reject_max_repetition}"
    if cmp["logprob_ratio_change"] > hyp.max_regression_ratio and cmp["logprob_delta"] < 0:
        return "regressed", f"logprob regressed ratio {cmp['logprob_ratio_change']:.4f}"
    if abs(cmp["logprob_delta"]) < hyp.accept_min_logprob_delta:
        return "no_effect", f"|logprob_delta| {abs(cmp['logprob_delta']):.2e} below threshold"
    # First milestone: any verified weight change with measurable logprob shift and no regression
    if cmp["logprob_delta"] != 0:
        return "accept", f"weight changed (max_diff={weight_verified['max_abs_diff']:.2e}), logprob_delta={cmp['logprob_delta']:.2e}"
    return "inconclusive", "weight changed but probes inconclusive"
