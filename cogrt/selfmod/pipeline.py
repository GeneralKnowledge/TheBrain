"""OBSERVE → HYPOTHESISE → DESIGN → CREATE CANDIDATE → EVALUATE → DECIDE → RECORD."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..backends.transformers_backend import TransformersBackend
from ..config import Config
from ..memory import MemoryStore
from . import experiments_db as db
from .decide import decide_outcome
from .evaluate import evaluate_backend
from .fingerprint import intervention_fingerprint
from .interventions.sparse import apply_sparse_perturbation, default_tensor_key, file_sha256, verify_weight_change
from .interventions.sparse import SparsePerturbationResult
from .types import WeightHypothesis


def _baseline_dir(cfg: Config) -> Path:
    p = Path(cfg.selfmod.baseline_dir or cfg.model.path)
    if not p.is_absolute() and cfg.source_path:
        p = Path(cfg.source_path).parent / p
    return p.resolve()


def _candidates_root(cfg: Config) -> Path:
    root = cfg.data_path / cfg.selfmod.candidates_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _apply_delta_in_memory(backend: TransformersBackend, candidate_dir: Path) -> None:
    from safetensors.torch import load_file

    manifest = json.loads((candidate_dir / "manifest.json").read_text())
    key = manifest["spec"]["tensor_key"]
    delta = load_file(str(candidate_dir / manifest["delta_file"]))
    sd = backend.model.state_dict()
    if key not in sd:
        raise KeyError(f"{key} not in model state_dict")
    patched = sd[key].clone()
    patched.copy_(delta[key].to(dtype=patched.dtype, device=patched.device))
    backend.model.load_state_dict({key: patched}, strict=False)


def run_experiment(store: MemoryStore, cfg: Config, hypothesis_item_id: int | None = None,
                   hyp: WeightHypothesis | None = None) -> dict[str, Any]:
    parent = _baseline_dir(cfg)
    if not (parent / "model.safetensors").is_file():
        raise FileNotFoundError(f"baseline weights missing under {parent}")
    parent_sha = file_sha256(parent / "model.safetensors")

    item = store.get(hypothesis_item_id) if hypothesis_item_id else None
    if item:
        hyp = WeightHypothesis.from_item_meta(item.meta)
    if hyp is None:
        raise ValueError("hypothesis required")
    if not hyp.tensor_key:
        hyp.tensor_key = default_tensor_key(hyp.layer_index)

    spec = {
        "tensor_key": hyp.tensor_key,
        "n_params": hyp.n_params,
        "magnitude": hyp.magnitude,
        "seed": hyp.seed,
    }
    fp = intervention_fingerprint(parent_sha, hyp.intervention, spec)
    if db.fingerprint_blocked(store, fp):
        store.event("experiment", "weight_skipped", {"fingerprint": fp, "reason": "prior reject/fail/regress"})
        return {"skipped": True, "fingerprint": fp, "reason": "blocked by prior outcome"}

    exp_id = db.create_experiment(
        store, hypothesis_item_id=hypothesis_item_id, parent_checkpoint=str(parent),
        parent_sha256=parent_sha, method=hyp.intervention, intervention=spec, fingerprint=fp,
        baseline_path=str(parent),
    )
    cand_dir = _candidates_root(cfg) / f"exp_{exp_id}"
    op = store.start_op("weight_experiment", None, {"experiment_id": exp_id, "fingerprint": fp})
    store.event("experiment", "weight_start", {"id": exp_id, "hypothesis": hypothesis_item_id, "fp": fp})

    try:
        pert: SparsePerturbationResult = apply_sparse_perturbation(
            parent, cand_dir, tensor_key=hyp.tensor_key, n_params=hyp.n_params,
            magnitude=hyp.magnitude, seed=hyp.seed, experiment_id=str(exp_id),
        )
        verified = verify_weight_change(parent, cand_dir, hyp.tensor_key)

        baseline_be = TransformersBackend(str(parent), model_id=cfg.model.model_id, device=cfg.model.device,
                                          dtype=cfg.model.dtype, context_length=min(512, cfg.model.context_length),
                                          seed=cfg.model.seed)
        baseline_be.load()
        baseline_m = evaluate_backend(baseline_be, hyp.probe_prompts, seed=hyp.seed)
        baseline_be.unload()

        cand_be = TransformersBackend(str(parent), model_id=cfg.model.model_id, device=cfg.model.device,
                                      dtype=cfg.model.dtype, context_length=min(512, cfg.model.context_length),
                                      seed=cfg.model.seed)
        cand_be.load()
        _apply_delta_in_memory(cand_be, cand_dir)
        candidate_m = evaluate_backend(cand_be, hyp.probe_prompts, seed=hyp.seed)
        cand_be.unload()

        for phase, m in (("baseline", baseline_m), ("candidate", candidate_m)):
            db.record_measurement(store, exp_id, phase, "mean_logprob", m.mean_logprob)
            db.record_measurement(store, exp_id, phase, "repetition", m.repetition)
            db.record_measurement(store, exp_id, phase, "mean_latency_s", m.mean_latency_s)

        decision, reason = decide_outcome(hyp, baseline_m, candidate_m, verified)
        detail = {
            "verified": verified,
            "baseline": baseline_m.__dict__,
            "candidate": candidate_m.__dict__,
            "candidate_dir": str(cand_dir),
        }
        db.complete_experiment(store, exp_id, decision=decision, reason=reason,
                               candidate_path=str(cand_dir), detail=detail)
        outcome_id, _ = store.add("semantic", "experiment_outcome",
                  f"Weight experiment #{exp_id} ({decision}): {reason}", "observed",
                  operation_id=op, meta={"experiment_id": exp_id, "decision": decision, "fingerprint": fp},
                  dedupe=False)
        if hypothesis_item_id:
            status = "resolved" if decision == "accept" else "deferred"
            meta = (item.meta if item else hyp.to_meta()) | {
                "last_experiment_id": exp_id, "last_decision": decision, "outcome_item_id": outcome_id}
            store.revise(hypothesis_item_id, f"weight experiment {exp_id}: {decision}", op,
                         status=status, meta=meta)
            store.link(hypothesis_item_id, outcome_id, "experiment_outcome", operation_id=op)
        store.event("experiment", "weight_complete", {"id": exp_id, "decision": decision, "reason": reason})
        store.end_op(op, "completed", {"decision": decision}, state_changes=1)
        return {"experiment_id": exp_id, "decision": decision, "reason": reason, "verified": verified,
                "candidate_dir": str(cand_dir)}
    except Exception as exc:  # noqa: BLE001
        db.fail_experiment(store, exp_id, repr(exc))
        store.event("experiment", "weight_failed", {"id": exp_id, "error": repr(exc)})
        store.end_op(op, "failed", error=repr(exc))
        if cand_dir.exists():
            shutil.rmtree(cand_dir, ignore_errors=True)
        raise


def tick_autonomous(store: MemoryStore, cfg: Config) -> dict[str, Any] | None:
    """Run at most one pending weight experiment (no user message)."""
    if not cfg.selfmod.enabled:
        return None
    pending = db.pending_weight_hypotheses(store, limit=1)
    if not pending:
        return None
    it = pending[0]
    try:
        hyp = WeightHypothesis.from_item_meta(it.meta)
    except ValueError:
        return None
    return run_experiment(store, cfg, hypothesis_item_id=it.id, hyp=hyp)


def propose_default_hypothesis(store: MemoryStore, seed: int = 0, *, layer_index: int = 0) -> int:
    """Structured hypothesis template for the first vertical slice."""
    hyp = WeightHypothesis(
        problem="Probe log-probability on fixed prompts may shift under tiny weight edits.",
        proposed_change=f"Add sparse perturbation ({32} params) to layer 0 MLP down projection.",
        mechanism="Direct weight-space sensitivity of output logits.",
        expected_effect="Non-zero change in mean probe log-probability.",
        expected_collateral="Possible minor fluency or repetition change on probes.",
        measurements="mean_logprob on probe prompts; repetition; max_abs_diff vs parent.",
        layer_index=layer_index,
        n_params=32,
        magnitude=1e-4,
        seed=seed,
    )
    iid, _ = store.add("unresolved", "weight_hypothesis", hyp.proposed_change, "inferred",
                       priority=0.85, meta=hyp.to_meta(), note="default self-mod vertical slice")
    return iid
