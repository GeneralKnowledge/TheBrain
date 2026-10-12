"""Integration tests with the real GGUF model. Run: pytest -m model."""

from pathlib import Path

import pytest

from cogrt.backends.llamacpp import LlamaCppBackend
from cogrt.config import load

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "models" / "Qwen3-1.7B-Base.Q4_K_M.gguf"


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(not GGUF.is_file(), reason="GGUF not downloaded")
def test_llamacpp_load_and_generate() -> None:
    backend = LlamaCppBackend(str(GGUF), context_length=512, n_threads=4, n_gpu_layers=0)
    backend.load()
    from cogrt.backends.base import GenerationRequest

    res = backend.generate(GenerationRequest(
        prompt="Research notebook.\nNote: The river was high.\nTentative idea:",
        max_tokens=24, temperature=0.0, stop=["\n"]))
    assert res.completion_tokens > 0
    assert res.finish_reason in ("stop", "length")
    assert res.rss_mb and res.rss_mb > 500
    backend.unload()


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(not GGUF.is_file(), reason="GGUF not downloaded")
def test_runtime_one_cycle_with_llama(tmp_path) -> None:
    cfg = load(ROOT / "configs" / "laptop-cpu.toml",
               overrides={"runtime": {"data_dir": str(tmp_path / "data")},
                          "scheduler": {"interval_s": 1, "max_cycle_s": 120,
                                        "max_model_calls_per_cycle": 1}})
    from cogrt.runtime import Runtime
    from cogrt.seed_data import seed_store

    rt = Runtime(cfg)
    rt.start(control_thread=False)
    seed_store(rt.store)
    oc = rt.run_cycle()
    rt.shutdown()
    assert oc.model_calls >= 1
    assert oc.outcome in ("productive", "unproductive", "cancelled", "failed")
