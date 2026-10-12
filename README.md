# Open Cognitive Runtime (`cogrt`)

This project is an experiment in **persistent, inspectable internal activity** around an
open-weight language model, with a long-term research direction toward **autonomous,
evidence-driven self-modification of model weights** (not fine-tune-by-default).

The runtime decides when and why the model runs; the model is not required to answer a
user on every invocation. A terminal interface is provided for observation and optional
conversation, but it is not the primary purpose of the system.

This is **not** a conventional chatbot with memory, a simulated personality, or a
self-prompting text loop pretending to be cognition. Internal work is structured
(select → retrieve → generate → evaluate → commit), bounded by configuration, and
recorded in SQLite with provenance. Generating text does not count as success unless
deterministic checks accept a durable update.

## Quick start (no model download)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cogrt -c configs/mock.toml seed configs/seed-notes.txt
cogrt -c configs/mock.toml run --max-cycles 5
cogrt -c configs/mock.toml status
cogrt -c configs/mock.toml cycles -v
cogrt -c configs/mock.toml events --channel internal
```

## Low-end laptop (Qwen3-1.7B, CPU)

1. Download GGUF (example):

   `models/Qwen3-1.7B-Base.Q4_K_M.gguf` from
   [mradermacher/Qwen3-1.7B-Base-GGUF](https://huggingface.co/mradermacher/Qwen3-1.7B-Base-GGUF)

2. Install inference extra:

   `pip install -e ".[llama,dev]"`

3. Inspect hardware and run:

```bash
cogrt hw
cogrt -c configs/laptop-cpu.toml seed configs/seed-notes.txt
cogrt -c configs/laptop-cpu.toml run --max-cycles 3
```

See `configs/laptop-gpu.toml` for partial/full GPU offload (requires a CUDA build of
llama-cpp-python).

## Architecture (summary)

| Component | Role |
|-----------|------|
| `cogrt/backends/` | Model adapters (mock, GGUF, transformers, remote) |
| `cogrt/scheduler.py` | States, budgets, continuation/dormancy |
| `cogrt/cycle/` | Bounded internal feedback loop and policies |
| `cogrt/memory/` | SQLite + FTS5 durable memory and process history |
| `cogrt/workspace.py` | Temporary structured working state |
| `cogrt/runtime.py` | Main process and control queue |
| `cogrt/cli.py` | SSH-friendly `cogrt` commands |
| `research/` | Optional steering & SAE prototypes (torch) |

Details: `docs/PLAN.md`, status: `docs/STATUS.md`, measurements: `docs/RESULTS.md`.

## Controls

- **Pause / resume:** `cogrt pause` / `cogrt resume` (persistent flag)
- **Interrupt:** `cogrt interrupt` (cancel at safe boundary)
- **Emergency stop:** `touch data/<profile>/STOP` or `cogrt stop --now`
- **Budgets:** `cogrt set interval_s=120 max_cycles_per_hour=20`

## Weight-space experiments (W1)

Requires Transformers + HF weights under `models/hf/Qwen3-1.7B-Base/`:

```bash
pip install -e ".[torch,dev]"
cogrt -c configs/selfmod-research.toml selfmod propose
cogrt -c configs/selfmod-research.toml selfmod run
cogrt -c configs/selfmod-research.toml selfmod list
```

This performs a **real sparse perturbation** on stored weights (candidate delta on disk;
parent checkpoint unchanged), evaluates baseline vs patched model on fixed probes, and
records accept / reject / inconclusive in SQLite. See `docs/GAP_ANALYSIS.md`.

## Tests and benchmarks

```bash
pytest -q                    # unit tests (no model)
pytest -m model -q           # needs GGUF under models/
cogrt bench
cogrt sim configs/sim-population.toml
```

## Licence

Runtime code: Apache-2.0 (see repository). Model weights follow their respective
Hugging Face model cards (Qwen3-1.7B-Base: Apache-2.0).
