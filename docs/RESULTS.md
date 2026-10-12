# Measured results

## Development VM (2026-10-12)

Environment: Ubuntu 24.04, Python 3.12, 4 logical CPUs, ~5 GB RAM available, no NVIDIA driver.

### Qwen3-1.7B-Base via llama.cpp (Q4_K_M)

- File: `models/Qwen3-1.7B-Base.Q4_K_M.gguf` (mradermacher build, Apache-2.0)
- Base model card: `Qwen/Qwen3-1.7B-Base` @ `ea980cb0…` (Apache-2.0)
- `llama_cpp 0.3.36`, `llama_supports_gpu_offload()` → **False** (CPU wheel)
- Load: 2.14 s, RSS +1870 MB (total ~1911 MB)
- Sample continuation (greedy): 10 tokens in 0.74 s (~13.5 tok/s)

### Autonomous runtime (mock backend)

Five cycles without user input (`configs/mock.toml`): durable unresolved items created;
no entries on the `message` channel. See `data/mock/` from local smoke runs.

### Automated tests

```
pytest -q                 → 36 passed, 2 deselected (model marker)
pytest -m model -q        → 2 passed (GGUF load + one live cycle)
cogrt bench               → 6/6 cases passed (after pause metric fix)
```

### Negative / limitation notes

- Mock-backend “hypotheses” are lexical recombinations; they demonstrate **policy and
  retention mechanics**, not model quality.
- Repetition and novelty metrics in `cogrt bench` are deterministic baselines, not
  semantic judgements.

## Target hardware (Dell G3)

*Not yet measured in this repository run.* Fill this section after running the checklist
in `docs/STATUS.md`.
