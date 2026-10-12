"""Optional remote inference over HTTP (llama.cpp ``llama-server`` protocol).

Disabled unless the configuration sets ``[security] allow_network = true``.
Only the ``/completion`` endpoint is used, so any server that implements it
can be substituted. Uses the standard library only.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .base import BackendError, Capabilities, GenerationRequest, GenerationResult, unsupported


class RemoteBackend:
    name = "remote"

    def __init__(self, url: str, model_id: str = "remote", timeout_s: float = 120.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise BackendError("remote_url must be an http(s) URL")
        self.url = url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self._ok = False

    def capabilities(self) -> Capabilities:
        return Capabilities(streaming=False, cancellation=False, logprobs=False)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise BackendError(f"remote backend unavailable: {exc}") from exc

    def load(self) -> None:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=10) as resp:  # noqa: S310
                self._ok = resp.status == 200
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendError(f"remote backend unreachable: {exc}") from exc

    def unload(self) -> None:
        self._ok = False

    @property
    def loaded(self) -> bool:
        return self._ok

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "url": self.url}

    def generate(self, req: GenerationRequest) -> GenerationResult:
        unsupported(req, self.capabilities(), self.name)
        if req.cancel is not None and req.cancel.cancelled:
            return GenerationResult("", 0, 0, 0.0, "cancelled", backend=self.name, model_id=self.model_id)
        t0 = time.perf_counter()
        body = {"prompt": req.prompt, "n_predict": req.max_tokens, "temperature": req.temperature,
                "top_p": req.top_p, "top_k": req.top_k, "repeat_penalty": req.repeat_penalty,
                "stop": req.stop, "cache_prompt": True}
        if req.seed is not None:
            body["seed"] = req.seed
        data = self._post("/completion", body)
        text = data.get("content", "")
        finish = "stop" if data.get("stopped_eos") or data.get("stopped_word") else "length"
        return GenerationResult(text=text, prompt_tokens=int(data.get("tokens_evaluated", 0)),
                                completion_tokens=int(data.get("tokens_predicted", 0)),
                                latency_s=time.perf_counter() - t0, finish_reason=finish,
                                backend=self.name, model_id=self.model_id)

    def score(self, prompt: str, continuation: str) -> float:
        raise BackendError("remote backend does not support scoring")
