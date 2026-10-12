"""Baseline vs candidate evaluation using deterministic probes (not model self-judgement)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.base import GenerationRequest, ModelBackend
from ..memory.text import distinct_n, internal_repetition


@dataclass
class ProbeMetrics:
    mean_logprob: float
    mean_latency_s: float
    repetition: float
    per_prompt: list[dict[str, Any]] = field(default_factory=list)


def evaluate_backend(backend: ModelBackend, prompts: list[str], *, max_tokens: int = 24,
                     seed: int = 0) -> ProbeMetrics:
    lps: list[float] = []
    lats: list[float] = []
    texts: list[str] = []
    per: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        cont = " the"
        lp = backend.score(prompt, cont)
        req = GenerationRequest(prompt=prompt, max_tokens=max_tokens, temperature=0.0, seed=seed + i,
                                stop=["\n"])
        res = backend.generate(req)
        lps.append(lp)
        lats.append(res.latency_s)
        texts.append(res.text)
        per.append({"prompt": prompt, "logprob_cont_the": lp, "latency_s": res.latency_s,
                    "text": res.text, "completion_tokens": res.completion_tokens})
    rep = max(internal_repetition(t) for t in texts) if texts else 0.0
    if len(texts) > 1:
        rep = max(rep, 1.0 - distinct_n(texts, 2))
    return ProbeMetrics(mean_logprob=sum(lps) / len(lps), mean_latency_s=sum(lats) / len(lats),
                        repetition=rep, per_prompt=per)


def compare(baseline: ProbeMetrics, candidate: ProbeMetrics) -> dict[str, float]:
    base_lp = baseline.mean_logprob or -1e-9
    return {
        "logprob_delta": candidate.mean_logprob - baseline.mean_logprob,
        "logprob_ratio_change": abs(candidate.mean_logprob - baseline.mean_logprob) / abs(base_lp),
        "latency_delta": candidate.mean_latency_s - baseline.mean_latency_s,
        "repetition_delta": candidate.repetition - baseline.repetition,
    }
