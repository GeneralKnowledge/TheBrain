# Implementation status

Last updated on the development VM (4 vCPU, ~16 GB RAM, **no GPU**). Measurements
labelled “VM” are from this environment, not from the Dell G3 target laptop.

## Milestones

| Milestone | Status | Notes |
|-----------|--------|-------|
| M0 Environment & model verification | **Done (VM)** | Qwen3-1.7B-Base Apache-2.0; GGUF Q4_K_M; HF weights present; SAE catalogued, not bundled |
| M1 Minimal persistent runtime | **Done** | Backends, scheduler, SQLite memory, one bounded cycle, pause/interrupt, CLI |
| M2 State evolution | **Partial** | Multiple policies, selective retention, contradictions, depth limit, benchmarks |
| M3 Activation research | **Scaffold** | Transformers backend + hooks; `research/steering`, `research/sae` optional |
| M4 AURIX & controls | **Scaffold** | AURIX prompt + comparison helper; steering extraction API |
| M5 Accelerated simulation | **Initial** | `cogrt sim` with mock/cached inference |

## What is implemented

- **Model backends:** `mock`, `llamacpp` (GGUF), `transformers` (activations/steering), `remote` (optional HTTP)
- **Cognitive scheduler:** states, hourly budgets, continuation/backoff/dormancy, recovery after crash
- **Internal workspace:** bounded slots, JSON snapshot in SQLite meta
- **Memory:** episodic / semantic / unresolved, FTS5, provenance, revisions, links, retention
- **Internal cycle:** select → retrieve → generate → evaluate → decide → commit (no user message required)
- **Conversation:** secondary; ingests claims/questions; optional reply on `message` channel only
- **Security:** data dir sandbox, STOP file, persistent pause, no shell/network for the model
- **Tests:** 36 unit tests (default); 2 integration tests (`pytest -m model`)
- **Benchmarks:** `cogrt bench` (scheduling, persistence, recall, contradictions, novelty, pause)

## Measured on this VM (llama.cpp CPU, Q4_K_M, n_ctx=1024, 4 threads)

| Metric | Value |
|--------|-------|
| Load time | ~2.1 s |
| RSS after load | ~1911 MB |
| Generation (greedy, 64 tok) | ~13.5 tok/s |
| One real runtime cycle | passes integration test (`test_runtime_one_cycle_with_llama`) |

GPU offload: **not tested** (CPU-only wheel). Config `configs/laptop-gpu.toml` documents expectations.

## Not verified / speculative

- Behavioural persistence of activation steering without prompts (API only)
- SAE feature intervention at scale (loader for one layer only)
- Remote backend against a live `llama-server`
- Population-level emergent behaviour (simulation uses mock/cached text)

## Target laptop checklist (Dell G3, 8 GB RAM, GTX 1060)

Run on the machine and record in `docs/RESULTS.md`:

1. `cogrt hw`
2. `cogrt -c configs/laptop-cpu.toml seed configs/seed-notes.txt`
3. `cogrt -c configs/laptop-cpu.toml run --max-cycles 3`
4. Optional CUDA build + `configs/laptop-gpu.toml`
