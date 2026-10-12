"""Contrastive activation steering (optional, torch + transformers backend)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from cogrt.backends.base import SteeringSpec, SteeringVector


@dataclass
class SteeringDirection:
    name: str
    layer: int
    vector: list[float]
    ref_norm: float
    pos_prompts: list[str]
    neg_prompts: list[str]

    def to_vector(self, coefficient: float) -> SteeringVector:
        return SteeringVector(layer=self.layer, vector=self.vector, coefficient=coefficient,
                              name=self.name, normalize=True, ref_norm=self.ref_norm)


def extract_direction(backend: Any, layer: int, pos: Iterable[str], neg: Iterable[str],
                      name: str = "direction") -> SteeringDirection:
    """Mean(pos activations) - mean(neg activations) at ``layer``, L2-normalised."""
    import torch

    if not hasattr(backend, "activations"):
        raise RuntimeError("backend does not expose activations")
    pos_h = [backend.activations(p, [layer])[layer] for p in pos]
    neg_h = [backend.activations(p, [layer])[layer] for p in neg]
    pm = torch.stack(pos_h).mean(0)
    nm = torch.stack(neg_h).mean(0)
    diff = (pm - nm).float()
    ref_norm = float(pm.norm())
    vec = (diff / (diff.norm() + 1e-8)).tolist()
    return SteeringDirection(name, layer, vec, ref_norm, list(pos), list(neg))


def save_directions(path: Path, directions: list[SteeringDirection]) -> None:
    path.write_text(json.dumps([asdict(d) for d in directions], indent=2))


def load_directions(path: Path) -> list[SteeringDirection]:
    data = json.loads(path.read_text())
    out = []
    for d in data:
        out.append(SteeringDirection(d["name"], d["layer"], d["vector"], d["ref_norm"],
                                     d.get("pos_prompts", []), d.get("neg_prompts", [])))
    return out


def spec_from_file(path: Path, coefficient: float, names: list[str] | None = None) -> SteeringSpec:
    dirs = load_directions(path)
    if names:
        dirs = [d for d in dirs if d.name in names]
    return SteeringSpec(vectors=[d.to_vector(coefficient) for d in dirs])
