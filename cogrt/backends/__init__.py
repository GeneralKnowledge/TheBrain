"""Model backends. Heavy dependencies are imported only by the adapter that needs them."""

from __future__ import annotations

from ..config import Config
from .base import (BackendError, CancelToken, Cancelled, Capabilities, GenerationRequest,
                   GenerationResult, ModelBackend, SteeringSpec, SteeringVector)


def create_backend(cfg: Config) -> ModelBackend:
    m = cfg.model
    if m.backend == "mock":
        from .mock import MockBackend
        return MockBackend(model_id=m.model_id, seed=m.seed)
    if m.backend == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(path=_resolve(cfg, m.path), model_id=m.model_id,
                               context_length=m.context_length, n_threads=m.n_threads,
                               n_gpu_layers=m.n_gpu_layers, seed=m.seed,
                               quantization=m.quantization)
    if m.backend == "transformers":
        from .transformers_backend import TransformersBackend
        return TransformersBackend(path=_resolve(cfg, m.path), model_id=m.model_id,
                                   device=m.device, dtype=m.dtype,
                                   context_length=m.context_length, seed=m.seed,
                                   n_threads=m.n_threads)
    if m.backend == "remote":
        if not cfg.security.allow_network:
            raise BackendError("remote backend disabled: set [security] allow_network = true")
        from .remote import RemoteBackend
        return RemoteBackend(url=m.remote_url, model_id=m.model_id, timeout_s=m.remote_timeout_s)
    raise BackendError(f"unknown backend {m.backend}")


def _resolve(cfg: Config, path: str) -> str:
    from pathlib import Path

    p = Path(path)
    if not p.is_absolute() and cfg.source_path:
        p = Path(cfg.source_path).parent / p
    return str(p)


__all__ = ["BackendError", "CancelToken", "Cancelled", "Capabilities", "GenerationRequest",
           "GenerationResult", "ModelBackend", "SteeringSpec", "SteeringVector", "create_backend"]
