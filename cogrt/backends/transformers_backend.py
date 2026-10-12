"""PyTorch / Transformers backend for activation-level research.

Slower and larger than the GGUF backend, but it exposes the residual stream:
activation capture, additive steering and exact log-probabilities. It uses a
manual sampling loop so cancellation, streaming and seeding behave the same
way as in the other adapters.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from ..hardware import process_rss_mb
from .base import (BackendError, Capabilities, GenerationRequest, GenerationResult,
                   SteeringSpec)


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


class TransformersBackend:
    name = "transformers"

    def __init__(self, path: str, model_id: str = "", device: str = "cpu",
                 dtype: str = "bfloat16", context_length: int = 1024, seed: int = 0,
                 n_threads: int = 0) -> None:
        self.path = path
        self.model_id = model_id or path
        self.requested_device = device
        self.device = "cpu"
        self.dtype_name = dtype
        self.context_length = context_length
        self.seed = seed
        self.n_threads = n_threads
        self.notes: list[str] = []
        self.model: Any = None
        self.tok: Any = None
        self._lock = threading.Lock()
        self.load_s: float | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, cancellation=True, logprobs=True,
                            activations=True, steering=True, gpu=self.device == "cuda")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError("torch/transformers not installed (pip install '.[torch]')") from exc
        if self.n_threads:
            torch.set_num_threads(self.n_threads)
        dev = self.requested_device
        if dev in ("cuda", "auto"):
            if torch.cuda.is_available():
                dev = "cuda"
            else:
                if dev == "cuda":
                    self.notes.append("CUDA requested but unavailable; using CPU")
                dev = "cpu"
        dtype = getattr(torch, self.dtype_name)
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=dtype,
                                                          low_cpu_mem_usage=True)
        self.model.to(dev).eval()
        self.device = dev
        self.load_s = time.perf_counter() - t0

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.tok = None
            import gc
            gc.collect()

    @property
    def layers(self) -> Any:
        return self.model.model.layers

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "path": self.path,
                "device": self.device, "dtype": self.dtype_name, "load_s": self.load_s,
                "context_length": self.context_length, "notes": self.notes}

    # ------------------------------------------------------------------ hooks
    @contextmanager
    def _steering(self, spec: SteeringSpec | None, prompt_len: int) -> Iterator[None]:
        import torch

        handles = []
        if spec and spec.active:
            for sv in spec.vectors:
                if not 0 <= sv.layer < self.n_layers:
                    raise BackendError(f"steering layer {sv.layer} out of range")
                vec = torch.as_tensor(sv.vector, dtype=torch.float32)
                if sv.normalize:
                    vec = vec / (vec.norm() + 1e-8) * sv.ref_norm
                delta = (vec * sv.coefficient).to(self.device, self.model.dtype)
                only_new = spec.positions == "generated"

                def hook(_m: Any, _i: Any, out: Any, delta: Any = delta, only_new: bool = only_new) -> Any:
                    h = _hidden(out)
                    if only_new and h.shape[1] > 1:
                        return out
                    h = h + delta
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

                handles.append(self.layers[sv.layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @contextmanager
    def _capture(self, layers: list[int], store: dict[int, list[Any]]) -> Iterator[None]:
        handles = []
        for layer in layers:
            def hook(_m: Any, _i: Any, out: Any, layer: int = layer) -> None:
                store.setdefault(layer, []).append(_hidden(out).detach().float().cpu())
            handles.append(self.layers[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------ inference
    def _encode(self, text: str, reserve: int) -> Any:
        ids = self.tok(text, return_tensors="pt").input_ids
        budget = self.context_length - reserve
        if ids.shape[1] > budget:
            ids = ids[:, -budget:]
        return ids.to(self.device)

    def activations(self, text: str, layers: list[int], reduce: str = "mean",
                    steering: SteeringSpec | None = None) -> dict[int, Any]:
        """Residual-stream output of each layer for ``text`` (forward pass only)."""
        import torch

        self.load()
        store: dict[int, list[Any]] = {}
        with self._lock, torch.no_grad():
            ids = self._encode(text, 0)
            with self._steering(steering, ids.shape[1]), self._capture(layers, store):
                self.model(ids)
        out = {}
        for layer, chunks in store.items():
            h = chunks[0][0]  # (seq, d)
            out[layer] = h.mean(0) if reduce == "mean" else (h[-1] if reduce == "last" else h)
        return out

    def generate(self, req: GenerationRequest) -> GenerationResult:
        import torch

        self.load()
        with self._lock, torch.no_grad():
            ids = self._encode(req.prompt, req.max_tokens + 4)
            n_prompt = ids.shape[1]
            gen = torch.Generator(device="cpu")
            gen.manual_seed(req.seed if req.seed is not None else self.seed)
            store: dict[int, list[Any]] = {}
            out_ids: list[int] = []
            logprobs: list[float] = []
            text = ""
            finish = "length"
            eos = self.tok.eos_token_id
            t0 = time.perf_counter()
            with self._steering(req.steering, n_prompt), self._capture(req.capture_layers, store):
                past = None
                inp = ids
                for _ in range(req.max_tokens):
                    if req.cancel is not None and req.cancel.cancelled:
                        finish = "cancelled"
                        break
                    out = self.model(inp, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[0, -1].float()
                    nxt, lp = self._sample(logits, ids, out_ids, req, gen)
                    if nxt == eos:
                        finish = "stop"
                        break
                    out_ids.append(nxt)
                    logprobs.append(lp)
                    new_text = self.tok.decode(out_ids, skip_special_tokens=True)
                    piece, text = new_text[len(text):], new_text
                    if req.on_token and piece:
                        req.on_token(piece)
                    hit = next((s for s in req.stop if s and s in text), None)
                    if hit:
                        text = text[: text.index(hit)]
                        finish = "stop"
                        break
                    inp = torch.tensor([[nxt]], device=self.device)
            acts = None
            if req.capture_layers:
                # Mean over prompt and generated positions of the captured residuals.
                acts = {layer: torch.cat([c[0] for c in chunks], 0).mean(0)
                        for layer, chunks in store.items()}
            return GenerationResult(
                text=text, prompt_tokens=n_prompt, completion_tokens=len(out_ids),
                latency_s=time.perf_counter() - t0, finish_reason=finish,
                token_logprobs=logprobs if req.logprobs else None, activations=acts,
                rss_mb=process_rss_mb(), backend=self.name, model_id=self.model_id,
                extra={"steering": bool(req.steering and req.steering.active)},
            )

    def _sample(self, logits: Any, prompt_ids: Any, out_ids: list[int],
                req: GenerationRequest, gen: Any) -> tuple[int, float]:
        import torch

        if req.repeat_penalty and req.repeat_penalty != 1.0:
            seen = torch.unique(torch.cat([prompt_ids[0].cpu(), torch.tensor(out_ids, dtype=torch.long)]))
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / req.repeat_penalty, vals * req.repeat_penalty)
        full_lp = torch.log_softmax(logits, -1)
        if req.temperature <= 0:
            nxt = int(torch.argmax(logits))
            return nxt, float(full_lp[nxt])
        scaled = logits / req.temperature
        if req.top_k > 0:
            kth = torch.topk(scaled, min(req.top_k, scaled.numel())).values[-1]
            scaled = torch.where(scaled < kth, torch.full_like(scaled, -float("inf")), scaled)
        probs = torch.softmax(scaled, -1)
        if req.top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp, -1)
            mask = cum - sp > req.top_p
            sp[mask] = 0
            probs = torch.zeros_like(probs).scatter_(0, si, sp)
            probs = probs / probs.sum()
        nxt = int(torch.multinomial(probs.cpu(), 1, generator=gen))
        return nxt, float(full_lp[nxt])

    def score(self, prompt: str, continuation: str, steering: SteeringSpec | None = None) -> float:
        """Mean log-probability per token of ``continuation`` after ``prompt``."""
        import torch

        self.load()
        with self._lock, torch.no_grad():
            p = self.tok(prompt, return_tensors="pt").input_ids
            c = self.tok(continuation, return_tensors="pt", add_special_tokens=False).input_ids
            ids = torch.cat([p, c], 1)[:, -self.context_length:].to(self.device)
            n_c = min(c.shape[1], ids.shape[1] - 1)
            with self._steering(steering, ids.shape[1]):
                logits = self.model(ids).logits[0].float()
            lp = torch.log_softmax(logits[:-1], -1)
            tgt = ids[0, 1:]
            tok_lp = lp.gather(1, tgt[:, None])[:, 0][-n_c:]
            return float(tok_lp.mean())
