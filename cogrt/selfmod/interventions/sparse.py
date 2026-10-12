"""Sparse perturbation — a real weight-space intervention (not steering, not memory).

Modifies a bounded number of elements in one tensor of a safetensors checkpoint
and stores the result as a candidate delta file plus manifest. Parent weights
on disk are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


@dataclass
class SparsePerturbationResult:
    candidate_dir: Path
    manifest_path: Path
    delta_path: Path
    tensor_key: str
    n_changed: int
    parent_sha256: str
    fingerprint: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_tensor_key(layer_index: int) -> str:
    return f"model.layers.{layer_index}.mlp.down_proj.weight"


def apply_sparse_perturbation(
    parent_dir: Path,
    candidate_dir: Path,
    *,
    tensor_key: str,
    n_params: int,
    magnitude: float,
    seed: int,
    experiment_id: str,
) -> SparsePerturbationResult:
    import torch

    parent_dir = parent_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    weights = parent_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"model.safetensors not found under {parent_dir}")
    parent_sha = file_sha256(weights)

    with safe_open(str(weights), framework="pt", device="cpu") as f:
        if tensor_key not in f.keys():
            raise KeyError(f"{tensor_key} not in checkpoint (have {len(f.keys())} tensors)")
        tensor = f.get_tensor(tensor_key).clone()
    orig_dtype = tensor.dtype
    flat = tensor.float().reshape(-1)
    if n_params <= 0 or n_params > flat.numel():
        raise ValueError(f"n_params must be in 1..{flat.numel()}, got {n_params}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    idx = torch.randperm(flat.numel(), generator=gen)[:n_params]
    before = flat[idx].clone()
    flat[idx] = flat[idx] + magnitude
    changed = int((flat[idx] - before).abs().gt(0).sum())
    if changed == 0:
        raise RuntimeError("intervention was a no-op")

    delta_path = candidate_dir / "delta.safetensors"
    out = flat.reshape(tensor.shape).to(orig_dtype)
    save_file({tensor_key: out}, str(delta_path))

    spec = {
        "tensor_key": tensor_key,
        "n_params": n_params,
        "magnitude": magnitude,
        "seed": seed,
        "indices": idx.tolist(),
    }
    from ..fingerprint import intervention_fingerprint

    fp = intervention_fingerprint(parent_sha, "sparse_perturbation", spec)
    manifest = {
        "experiment_id": experiment_id,
        "parent_dir": str(parent_dir),
        "parent_sha256": parent_sha,
        "intervention": "sparse_perturbation",
        "spec": spec,
        "fingerprint": fp,
        "delta_file": delta_path.name,
    }
    manifest_path = candidate_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return SparsePerturbationResult(
        candidate_dir=candidate_dir,
        manifest_path=manifest_path,
        delta_path=delta_path,
        tensor_key=tensor_key,
        n_changed=changed,
        parent_sha256=parent_sha,
        fingerprint=fp,
    )


def verify_weight_change(parent_dir: Path, candidate_dir: Path, tensor_key: str) -> dict[str, Any]:
    """Confirm candidate tensor differs from parent on at least one element."""
    import torch

    parent_w = parent_dir / "model.safetensors"
    manifest = json.loads((candidate_dir / "manifest.json").read_text())
    delta = candidate_dir / manifest["delta_file"]
    with safe_open(str(parent_w), framework="pt", device="cpu") as pf:
        p = pf.get_tensor(tensor_key).float()
    with safe_open(str(delta), framework="pt", device="cpu") as df:
        c = df.get_tensor(tensor_key).float()
    diff = (p - c).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "fraction_changed": float((diff > 0).float().mean()),
        "n_params_targeted": len(manifest["spec"]["indices"]),
    }
