"""Load a candidate by patching one tensor onto a parent Hugging Face directory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def materialize_candidate(parent_dir: Path, candidate_dir: Path, work_dir: Path) -> Path:
    """Copy tokenizer/config from parent; swap one weight tensor from delta. Returns model dir."""
    import torch
    from safetensors.torch import load_file, save_file

    parent_dir = parent_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    work_dir = work_dir.resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    manifest = json.loads((candidate_dir / "manifest.json").read_text())
    key = manifest["spec"]["tensor_key"]
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "LICENSE"):
        src = parent_dir / name
        if src.is_file():
            shutil.copy2(src, work_dir / name)
    parent_weights = parent_dir / "model.safetensors"
    delta = candidate_dir / manifest["delta_file"]
    merged = load_file(str(parent_weights))
    patched = load_file(str(delta))
    if key not in patched:
        raise KeyError(f"delta missing {key}")
    merged[key] = patched[key]
    save_file(merged, str(work_dir / "model.safetensors"))
    return work_dir
