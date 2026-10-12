from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")

from cogrt.config import Config, SelfModConfig
from cogrt.memory import MemoryStore
from cogrt.selfmod.experiments_db import (
    complete_experiment,
    create_experiment,
    ensure_schema,
    fingerprint_blocked,
    list_experiments,
    pending_weight_hypotheses,
    record_measurement,
)
from cogrt.selfmod.fingerprint import intervention_fingerprint
from cogrt.selfmod.interventions.sparse import apply_sparse_perturbation, verify_weight_change
from cogrt.selfmod.pipeline import propose_default_hypothesis, tick_autonomous
from cogrt.selfmod.types import WeightHypothesis
from cogrt.selfmod.decide import decide_outcome
from cogrt.selfmod.evaluate import ProbeMetrics


def _tiny_checkpoint(tmp_path: Path) -> Path:
    from safetensors.torch import save_file

    d = tmp_path / "parent"
    d.mkdir()
    save_file({"model.layers.0.mlp.down_proj.weight": torch.randn(8, 8)}, str(d / "model.safetensors"))
    (d / "config.json").write_text("{}")
    return d


def test_sparse_perturbation_changes_weights(tmp_path: Path) -> None:
    parent = _tiny_checkpoint(tmp_path)
    cand = tmp_path / "cand"
    res = apply_sparse_perturbation(parent, cand, tensor_key="model.layers.0.mlp.down_proj.weight",
                                    n_params=4, magnitude=0.01, seed=1, experiment_id="t1")
    v = verify_weight_change(parent, cand, res.tensor_key)
    assert v["max_abs_diff"] > 0
    assert v["fraction_changed"] > 0


def test_fingerprint_and_block(store: MemoryStore) -> None:
    ensure_schema(store)
    fp = intervention_fingerprint("abc", "sparse_perturbation", {"tensor_key": "x", "n_params": 1})
    eid = create_experiment(store, hypothesis_item_id=None, parent_checkpoint="/p", parent_sha256="abc",
                            method="sparse_perturbation", intervention={"x": 1}, fingerprint=fp,
                            baseline_path="/p")
    complete_experiment(store, eid, decision="regressed", reason="test", candidate_path="/c", detail={})
    assert fingerprint_blocked(store, fp)


def test_hypothesis_persistence_and_pending(tmp_path: Path) -> None:
    db = tmp_path / "persist.sqlite"
    store = MemoryStore(db)
    hid = propose_default_hypothesis(store)
    assert pending_weight_hypotheses(store)[0].id == hid
    store.close()
    s2 = MemoryStore(db)
    assert pending_weight_hypotheses(s2)[0].id == hid


def test_decide_detects_no_op_verification() -> None:
    hyp = WeightHypothesis(problem="", proposed_change="", mechanism="", expected_effect="",
                           expected_collateral="", measurements="")
    base = ProbeMetrics(0.0, 0.0, 0.0)
    cand = ProbeMetrics(0.0, 0.0, 0.0)
    d, r = decide_outcome(hyp, base, cand, {"max_abs_diff": 0.0})
    assert d == "failed"


def test_autonomous_skips_when_blocked(store: MemoryStore, cfg: Config, tmp_path: Path) -> None:
    cfg.runtime.data_dir = str(tmp_path / "data")
    cfg.selfmod.enabled = True
    cfg.selfmod.baseline_dir = str(_tiny_checkpoint(tmp_path))
    cfg.model.path = str(cfg.selfmod.baseline_dir)
    cfg.model.backend = "transformers"
    hid = propose_default_hypothesis(store, seed=99)
    # Pre-register a blocked fingerprint matching default hyp spec
    parent_sha = __import__("cogrt.selfmod.interventions.sparse", fromlist=["file_sha256"]).file_sha256(
        Path(cfg.selfmod.baseline_dir) / "model.safetensors")
    from cogrt.selfmod.interventions.sparse import default_tensor_key

    spec = {"tensor_key": default_tensor_key(0), "n_params": 32, "magnitude": 1e-4, "seed": 99}
    fp = intervention_fingerprint(parent_sha, "sparse_perturbation", spec)
    eid = create_experiment(store, hypothesis_item_id=hid, parent_checkpoint="p", parent_sha256=parent_sha,
                            method="sparse_perturbation", intervention=spec, fingerprint=fp, baseline_path="p")
    complete_experiment(store, eid, decision="regressed", reason="block", candidate_path="c", detail={})
    out = tick_autonomous(store, cfg)
    assert out and out.get("skipped")


@pytest.mark.model
@pytest.mark.slow
def test_full_weight_experiment_qwen(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    parent = root / "models/hf/Qwen3-1.7B-Base"
    if not (parent / "model.safetensors").is_file():
        pytest.skip("HF weights not present")
    from cogrt.config import load

    cfg = load(overrides={
        "runtime": {"data_dir": str(tmp_path / "data")},
        "model": {"backend": "transformers", "path": str(parent), "device": "cpu", "context_length": 512},
        "selfmod": {"enabled": True, "baseline_dir": str(parent)},
    })
    store = MemoryStore(tmp_path / "data/m.sqlite")
    hid = propose_default_hypothesis(store, seed=42)
    from cogrt.selfmod.pipeline import run_experiment

    hyp = WeightHypothesis.from_item_meta(store.get(hid).meta)
    hyp.n_params = 16
    hyp.probe_prompts = ["The capital of France is"]
    report = run_experiment(store, cfg, hypothesis_item_id=hid, hyp=hyp)
    assert report["decision"] in ("accept", "no_effect", "inconclusive", "regressed")
    assert report["verified"]["max_abs_diff"] > 0
    exps = list_experiments(store)
    assert exps[0]["decision"] == report["decision"]
