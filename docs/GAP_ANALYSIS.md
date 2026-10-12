# Gap analysis — autonomous self-modification (code-grounded)

Audit date: 2026-10-12. Answers refer to **main branch + this milestone** unless noted.

| # | Question | Answer |
|---|----------|--------|
| 1 | Can the model produce a hypothesis about its own behaviour? | **Partial.** Internal cycles retain text `hypothesis` items about *notes*, not structured self-mod proposals. **Now:** `weight_hypothesis` items with required fields (`cogrt selfmod propose`). |
| 2 | Can the runtime turn that into a structured experiment? | **Yes (weight track).** `run_experiment` creates DB record, candidate dir, manifest, budgets via config. |
| 3 | Can any code modify actual model weights? | **Yes.** `sparse_perturbation` writes a altered tensor to `delta.safetensors`; evaluation patches `state_dict` in memory. Parent checkpoint on disk is untouched. |
| 4 | Baseline vs candidate under same conditions? | **Yes.** Same probe prompts, same backend settings; deterministic logprob + repetition metrics. |
| 5 | Can results influence a later decision? | **Yes.** `intervention_fingerprint` blocks re-running rejected/regressed/failed interventions; outcomes stored as semantic `experiment_outcome`. |
| 6 | Autonomous multi-cycle? | **Partial.** `[selfmod] autorun = true` runs one pending experiment after internal cycles; mock cognitive loop unchanged. |
| 7 | Recovery after interrupt/restart? | **Yes** for runtime ops/cycles (existing). Weight experiments: status `running` → manual `failed` on crash (same as operations); completed records persist. |

## What remains scaffold-only

- ROME/MEMIT, merge, fine-tuning (register only)
- Model-authored hypothesis text → structured JSON (no freeform parser yet)
- GGUF weight edits (llama.cpp path unchanged)
- Accepting a candidate into “production” weights (evaluation only; no auto-promote to baseline)

## Smallest next step after this milestone

1. Model-generated structured hypotheses (constrained JSON schema + validator).
2. Promotion workflow: explicit human or policy gate before `baseline_dir` swap.
3. Port one targeted-edit method (ROME-class) on Qwen3-1.7B with causal tracing.
