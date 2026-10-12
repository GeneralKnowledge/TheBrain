"""Deterministic, dependency-free backend.

Used by unit tests, the accelerated simulation and as the last-resort
fallback. Its output is a seeded recombination of words from the prompt. It
is not a language model, and results produced with it are labelled
``backend=mock`` everywhere they are recorded.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Any

from .base import Capabilities, GenerationRequest, GenerationResult, unsupported

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_STOP = {
    "the", "and", "that", "this", "with", "from", "into", "what", "which", "about",
    "question", "notes", "note", "related", "possible", "connection", "answer",
    "one", "sentence", "are", "for", "was", "has", "have", "not", "between",
}
_LINKS = ["may depend on", "could explain", "is unlike", "resembles", "might limit",
          "seems related to", "does not imply", "precedes"]


class MockBackend:
    name = "mock"

    def __init__(self, model_id: str = "mock/deterministic", seed: int = 0,
                 latency_per_token_s: float = 0.0, script: list[str] | None = None) -> None:
        self.model_id = model_id
        self.seed = seed
        self.latency_per_token_s = latency_per_token_s
        self.script = list(script or [])
        self._loaded = False
        self.calls = 0

    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, cancellation=True, logprobs=True)

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "seed": self.seed}

    def _compose(self, req: GenerationRequest) -> str:
        if self.script:
            return self.script.pop(0)
        h = hashlib.sha256(f"{self.seed}|{req.seed}|{req.prompt}".encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        tail = req.prompt.rsplit("\n\n", 1)[-1][-600:]
        words = [w.lower() for w in _WORD.findall(tail) if w.lower() not in _STOP]
        if len(words) < 2:
            return ""
        uniq = list(dict.fromkeys(words))
        a, b = rng.sample(uniq, 2) if len(uniq) >= 2 else (uniq[0], uniq[0])
        extra = rng.choice(uniq)
        return f"{a.capitalize()} {rng.choice(_LINKS)} {b} through {extra}."

    def generate(self, req: GenerationRequest) -> GenerationResult:
        if not self._loaded:
            self.load()
        unsupported(req, self.capabilities(), self.name)
        self.calls += 1
        t0 = time.perf_counter()
        text = self._compose(req)
        tokens = text.split(" ") if text else []
        out: list[str] = []
        finish = "stop"
        for i, tok in enumerate(tokens):
            if i >= req.max_tokens:
                finish = "length"
                break
            if req.cancel is not None and req.cancel.cancelled:
                finish = "cancelled"
                break
            piece = tok if i == 0 else " " + tok
            out.append(piece)
            if req.on_token:
                req.on_token(piece)
            if self.latency_per_token_s:
                time.sleep(self.latency_per_token_s)
        return GenerationResult(
            text="".join(out),
            prompt_tokens=len(req.prompt.split()),
            completion_tokens=len(out),
            latency_s=time.perf_counter() - t0,
            finish_reason=finish,
            token_logprobs=[-1.0] * len(out) if req.logprobs else None,
            backend=self.name,
            model_id=self.model_id,
        )

    def score(self, prompt: str, continuation: str) -> float:
        h = hashlib.sha256(f"{prompt}|{continuation}".encode()).digest()
        return -1.0 - h[0] / 255.0
