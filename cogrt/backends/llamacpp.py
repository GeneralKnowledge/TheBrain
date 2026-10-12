"""GGUF quantised inference through llama-cpp-python.

Supports CPU inference (default) and partial or full GPU offload when the
installed build supports it. When GPU offload is requested but not supported
the adapter falls back to CPU and records why.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from ..hardware import process_rss_mb
from .base import BackendError, Capabilities, GenerationRequest, GenerationResult, unsupported


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, path: str, model_id: str = "", context_length: int = 1024,
                 n_threads: int = 0, n_gpu_layers: int = 0, seed: int = 0,
                 quantization: str = "") -> None:
        self.path = path
        self.model_id = model_id or Path(path).name
        self.context_length = context_length
        self.n_threads = n_threads or max(1, os.cpu_count() or 1)
        self.requested_gpu_layers = n_gpu_layers
        self.n_gpu_layers = 0
        self.seed = seed
        self.quantization = quantization
        self.notes: list[str] = []
        self._llm: Any = None
        self._lock = threading.Lock()  # one invocation at a time
        self.load_s: float | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, cancellation=True, logprobs=True,
                            gpu=self.n_gpu_layers != 0)

    def load(self) -> None:
        if self._llm is not None:
            return
        try:
            import llama_cpp  # type: ignore
        except ImportError as exc:
            raise BackendError("llama-cpp-python is not installed (pip install '.[llama]')") from exc
        if not Path(self.path).is_file():
            raise BackendError(f"GGUF file not found: {self.path}")
        gpu_layers = self.requested_gpu_layers
        if gpu_layers and not llama_cpp.llama_supports_gpu_offload():
            self.notes.append("GPU offload requested but this llama.cpp build is CPU-only; using CPU")
            gpu_layers = 0
        t0 = time.perf_counter()
        try:
            self._llm = llama_cpp.Llama(
                model_path=self.path, n_ctx=self.context_length, n_threads=self.n_threads,
                n_gpu_layers=gpu_layers, seed=self.seed, verbose=False, logits_all=False,
            )
        except Exception as exc:  # noqa: BLE001
            if gpu_layers:
                self.notes.append(f"GPU load failed ({exc}); retrying on CPU")
                self._llm = llama_cpp.Llama(
                    model_path=self.path, n_ctx=self.context_length, n_threads=self.n_threads,
                    n_gpu_layers=0, seed=self.seed, verbose=False, logits_all=False,
                )
                gpu_layers = 0
            else:
                raise BackendError(f"failed to load {self.path}: {exc}") from exc
        self.n_gpu_layers = gpu_layers
        self.load_s = time.perf_counter() - t0

    def unload(self) -> None:
        with self._lock:
            if self._llm is not None:
                close = getattr(self._llm, "close", None)
                if close:
                    close()
                self._llm = None

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "path": self.path,
                "quantization": self.quantization, "context_length": self.context_length,
                "n_threads": self.n_threads, "n_gpu_layers": self.n_gpu_layers,
                "load_s": self.load_s, "notes": self.notes}

    def _fit_prompt(self, prompt: str, max_tokens: int) -> tuple[str, int]:
        toks = self._llm.tokenize(prompt.encode(), add_bos=False)
        budget = self.context_length - max_tokens - 8
        if len(toks) > budget:
            toks = toks[-budget:]
            prompt = self._llm.detokenize(toks).decode(errors="ignore")
        return prompt, len(toks)

    def generate(self, req: GenerationRequest) -> GenerationResult:
        unsupported(req, self.capabilities(), self.name)
        self.load()
        with self._lock:
            prompt, n_prompt = self._fit_prompt(req.prompt, req.max_tokens)
            t0 = time.perf_counter()
            pieces: list[str] = []
            logprobs: list[float] = []
            finish = "length"
            stream = self._llm(
                prompt, max_tokens=req.max_tokens, temperature=req.temperature,
                top_p=req.top_p, top_k=req.top_k, repeat_penalty=req.repeat_penalty,
                stop=req.stop or None, seed=req.seed if req.seed is not None else self.seed,
                stream=True, logprobs=1 if req.logprobs else None,
            )
            for chunk in stream:
                if req.cancel is not None and req.cancel.cancelled:
                    finish = "cancelled"
                    break
                ch = chunk["choices"][0]
                piece = ch.get("text", "")
                if piece:
                    pieces.append(piece)
                    if req.on_token:
                        req.on_token(piece)
                lp = ch.get("logprobs")
                if lp and lp.get("token_logprobs"):
                    logprobs.extend(x for x in lp["token_logprobs"] if x is not None)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            text = "".join(pieces)
            n_out = len(self._llm.tokenize(text.encode(), add_bos=False)) if text else 0
            return GenerationResult(
                text=text, prompt_tokens=n_prompt, completion_tokens=n_out,
                latency_s=time.perf_counter() - t0, finish_reason=finish,
                token_logprobs=logprobs if req.logprobs else None, rss_mb=process_rss_mb(),
                backend=self.name, model_id=self.model_id,
            )

    def score(self, prompt: str, continuation: str) -> float:
        """Mean log-probability of ``continuation`` given ``prompt``."""
        import numpy as np  # llama-cpp-python depends on numpy

        self.load()
        with self._lock:
            llm = self._llm
            p = llm.tokenize(prompt.encode(), add_bos=False)
            c = llm.tokenize(continuation.encode(), add_bos=False)
            toks = (p + c)[-(self.context_length - 1):]
            n_c = min(len(c), len(toks) - 1)
            start = len(toks) - n_c
            llm.reset()
            llm.eval(toks[:start])
            total = 0.0
            for j in range(start, len(toks)):
                # Without logits_all only the last evaluated row is populated.
                logits = np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64)
                m = logits.max()
                total += logits[toks[j]] - (m + np.log(np.exp(logits - m).sum()))
                llm.eval([toks[j]])
            llm.reset()
            return float(total / max(1, n_c))
