# Implementation plan

**North star:** a persistent runtime that can **observe → hypothesise → experiment on
actual model weights → evaluate → record → repeat**, with increasing autonomy over
*what* to try—not a chatbot and not fine-tune-by-default.

Status of each item is tracked in `docs/STATUS.md`. Gap analysis:
`docs/GAP_ANALYSIS.md`. Method register: `docs/RESEARCH_REGISTER.md`.

## Findings that shaped the plan

* The repository contained only a one-line README. There were no conventions
  to follow, so the stack is chosen from the specification: Python 3.12,
  SQLite (stdlib), TOML (stdlib `tomllib`), pytest.
* The development VM is **not** the Dell G3 target. It has 4 vCPUs, ~16 GB RAM
  of which ~5 GB is available, and **no GPU**. Everything that is claimed
  as measured is measured on this VM and labelled as such. GPU offload is
  implemented behind configuration and hardware detection but is untested.
* `Qwen/Qwen3-1.7B-Base` (Apache-2.0, revision `ea980cb0…`) is a
  `Qwen3ForCausalLM`, 28 layers, hidden size 2048, bf16 weights (3.4 GB).
* Quantised GGUF builds exist (`mradermacher/Qwen3-1.7B-Base-GGUF`,
  Apache-2.0, revision `7b0494b4…`). Q4_K_M is 1.1 GB.
* Qwen publishes official residual-stream TopK SAEs for exactly this
  checkpoint (`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`, Apache-2.0 LICENSE
  file, 537 MB **per layer**, fp32). One layer can be loaded on CPU next to the
  bf16 model in ~4 GB; the model plus all SAEs cannot.

## Architecture (single process, no services)

```
cogrt/                 stable runtime (stdlib only)
  config.py            canonical TOML config → typed dataclasses
  hardware.py          RAM / disk / GPU / driver / acceleration detection
  backends/            ModelBackend protocol + adapters
    mock.py            deterministic, dependency-free (tests, simulation)
    llamacpp.py        GGUF quantised inference (CPU or partial GPU offload)
    transformers_backend.py  PyTorch: activations, logprobs, steering hooks
    remote.py          optional HTTP adapter (llama.cpp server protocol)
  memory/store.py      SQLite + FTS5: episodic, semantic, unresolved,
                       process history, provenance, revisions, events
  workspace.py         bounded, structured, inspectable working state
  cycle/               the bounded internal feedback loop and its policies
  scheduler.py         state machine, budgets, pause/stop/interrupt, recovery
  control.py           command queue (DB) so SSH clients talk to the daemon
  cli.py               `cogrt` terminal interface
  experiments/         experiment definitions, metrics, benchmark suite
  sim/                 accelerated simulation, decoupled from inference
research/              optional research prototypes (torch required)
  steering/            contrastive directions, AURIX comparisons, dimensions
  sae/                 Qwen-Scope SAE loading, feature analysis, intervention
```

## Acceptance criteria (mapped to the specification §12)

| # | Criterion | How it is demonstrated |
|---|-----------|------------------------|
| 1 | Runs on target or blockers documented | `cogrt hw` report; VM measurements; target-machine checklist in docs |
| 2 | Real model loads through documented backend | llama.cpp Q4_K_M and Transformers bf16, integration tests marked `model` |
| 3 | CPU fallback | `n_gpu_layers=0` / `device=cpu` default; GPU auto-falls-back |
| 4 | Internal work without a user prompt | scheduler test + real-model run log |
| 5 | Durable change without a user-facing message | cycle writes memory; `messages` channel untouched (test) |
| 6 | Persistence across restarts | restart tests, recovery of interrupted operations |
| 7 | SSH inspection | `cogrt status/events/memory/workspace/top` |
| 8 | Interrupt + pause | control commands, cancel token, persistent pause (tests) |
| 9 | Bounded compute/memory | config budgets, enforced and tested |
| 10 | Text generation ≠ success | evaluator discards duplicates/degenerate output; metrics count retained changes |
| 11 | Tests | pytest suite; model tests separated by marker |
| 12 | Steering without rewriting | `SteeringSpec` in `GenerationRequest`; research package only |
| 13 | Docs | README, docs/STATUS.md, docs/RESULTS.md |

## Order of work (updated)

Done: M0–M1, partial M2, W1 weight vertical slice.

Next:

1. **W2** — Model-proposed structured hypotheses (JSON schema + validator); still constrained interventions.
2. **W3** — Candidate promotion gate (explicit accept → new baseline revision); never silent overwrite.
3. **M3** — Activation steering vs weight edit comparison on shared probes.
4. **W4** — One targeted-edit method (ROME-class) feasibility on Qwen3-1.7B.
5. M4 AURIX as behavioural instrument post-intervention; M5 simulation at scale.

Each milestone is committed separately.
