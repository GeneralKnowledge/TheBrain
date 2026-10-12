"""Qwen-Scope TopK SAE loading and feature extraction (one layer at a time).

Official checkpoints: Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 (537 MB per layer).
Do not load the full model and all SAE layers on 8 GB RAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenScopeSAE:
    layer: int
    w_enc: Any
    w_dec: Any
    b_enc: Any
    b_dec: Any
    top_k: int = 50

    def feature_acts(self, residual: Any) -> Any:
        import torch

        pre = residual @ self.w_enc.T + self.b_enc
        topk_vals, topk_idx = pre.topk(self.top_k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts


def load_sae_layer(path: str | Path, layer: int, map_location: str = "cpu") -> QwenScopeSAE:
    import torch

    p = Path(path)
    file = p / f"layer{layer}.sae.pt" if p.is_dir() else p
    data = torch.load(file, map_location=map_location, weights_only=True)
    return QwenScopeSAE(layer, data["W_enc"], data["W_dec"], data["b_enc"], data["b_dec"])
