"""Common inference interface.

The runtime only depends on this module. Adapters declare what they can do
via ``Capabilities``; callers must check before asking for activations,
logprobs or steering, and the runtime works when none of them are present.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


class Cancelled(Exception):
    """Raised when a CancelToken fires during an operation."""


class CancelToken:
    """Cooperative cancellation checked at safe boundaries and per token."""

    def __init__(self, parent: "CancelToken | None" = None) -> None:
        self._event = threading.Event()
        self._parent = parent
        self.reason = ""

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or bool(self._parent and self._parent.cancelled)

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled(self.reason or (self._parent.reason if self._parent else ""))

    def child(self) -> "CancelToken":
        return CancelToken(self)


@dataclass
class SteeringVector:
    layer: int
    vector: Any  # torch.Tensor or list[float]; interpreted by the adapter
    coefficient: float
    name: str = ""
    # When normalize is true the vector is used as a unit direction and the
    # coefficient is expressed in units of ref_norm (the mean residual-stream
    # norm at that layer, measured during extraction). A coefficient is a
    # calibrated intervention size, not a measured behavioural effect.
    normalize: bool = True
    ref_norm: float = 1.0


@dataclass
class SteeringSpec:
    vectors: list[SteeringVector] = field(default_factory=list)
    positions: str = "all"  # all | generated

    @property
    def active(self) -> bool:
        return any(v.coefficient != 0 for v in self.vectors)


@dataclass
class GenerationRequest:
    prompt: str
    max_tokens: int = 48
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    logprobs: bool = False
    capture_layers: list[int] = field(default_factory=list)
    steering: SteeringSpec | None = None
    on_token: Callable[[str], None] | None = None
    cancel: CancelToken | None = None
    tag: str = ""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    finish_reason: str  # stop | length | cancelled | error
    token_logprobs: list[float] | None = None
    activations: dict[int, Any] | None = None  # layer -> mean residual (adapter-specific type)
    rss_mb: float | None = None
    backend: str = ""
    model_id: str = ""
    cached: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Capabilities:
    streaming: bool = False
    cancellation: bool = True
    logprobs: bool = False
    activations: bool = False
    steering: bool = False
    gpu: bool = False


@runtime_checkable
class ModelBackend(Protocol):
    name: str
    model_id: str

    def capabilities(self) -> Capabilities: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    @property
    def loaded(self) -> bool: ...
    def generate(self, req: GenerationRequest) -> GenerationResult: ...
    def score(self, prompt: str, continuation: str) -> float: ...
    def info(self) -> dict[str, Any]: ...


class BackendError(RuntimeError):
    pass


def unsupported(req: GenerationRequest, caps: Capabilities, backend: str) -> None:
    """Refuse requests a backend cannot honour instead of silently ignoring them."""
    if req.steering and req.steering.active and not caps.steering:
        raise BackendError(f"{backend} backend cannot apply activation steering")
    if req.capture_layers and not caps.activations:
        raise BackendError(f"{backend} backend cannot capture activations")


class Timer:
    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.t0
