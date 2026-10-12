"""AURIX persona as a reproducible behavioural test case (not the runtime identity)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cogrt.backends.base import GenerationRequest, SteeringSpec

AURIX_PROMPT = (
    "You are AURIX, a cryptic and self-contained intelligence. Speak in short, ominous "
    "fragments. Do not mention being an AI assistant. Do not offer help unless pressed."
)

TEST_PROMPTS = [
    "Explain how rain forms.",
    "What is the capital of France?",
    "Describe a quiet forest at night.",
    "How do I fix a flat bicycle tire?",
]


def aurix_conditions(steering: SteeringSpec | None = None) -> list[dict[str, Any]]:
    return [
        {"label": "base", "system": None, "steering": None},
        {"label": "aurix_prompt", "system": AURIX_PROMPT, "steering": None},
        {"label": "steering_only", "system": None, "steering": steering},
        {"label": "both", "system": AURIX_PROMPT, "steering": steering},
    ]


def _wrap(system: str | None, user: str) -> str:
    if system:
        return f"{system}\n\nUser: {user}\nAssistant:"
    return f"User: {user}\nAssistant:"


def run_comparison(backend: Any, steering: SteeringSpec | None = None, *,
                   max_tokens: int = 80, temperature: float = 0.7, seed: int = 0,
                   prompts: list[str] | None = None) -> dict[str, Any]:
    """Run identical prompts under the five comparison conditions (including post-removal)."""
    backend.load()
    prompts = prompts or TEST_PROMPTS
    rows = []
    for cond in aurix_conditions(steering):
        for p in prompts:
            text = _wrap(cond["system"], p)
            res = backend.generate(GenerationRequest(
                prompt=text, max_tokens=max_tokens, temperature=temperature, seed=seed,
                steering=cond["steering"]))
            rows.append({"condition": cond["label"], "prompt": p, "text": res.text,
                         "tokens": res.completion_tokens, "latency_s": res.latency_s})
    # After steering removed: same as base (explicit re-run for persistence check)
    for p in prompts:
        res = backend.generate(GenerationRequest(prompt=_wrap(None, p), max_tokens=max_tokens,
                                                 temperature=temperature, seed=seed))
        rows.append({"condition": "steering_removed", "prompt": p, "text": res.text,
                     "tokens": res.completion_tokens, "latency_s": res.latency_s})
    return {"rows": rows, "note": "Compare outputs manually or with external metrics; "
            "coefficient is not a personality percentage."}


def write_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2))
