# Research register — model self-modification

Status key: **implemented** | **tested** | **measured (VM)** | **scaffold** | **not started**

| Track | Question | Method | Evidence / compatibility | Limitations | Status | Next experiment |
|-------|----------|--------|--------------------------|-------------|--------|-----------------|
| Runtime memory | Can persistent state evolve without user input? | SQLite + internal cycle | 36 unit tests; bench 6/6 | Not weight change | **tested** | — |
| Activation steering | Can inference-time hooks shift behaviour? | Additive residual hooks | Transformers backend | Not persistent weights; not evaluated end-to-end | **scaffold** | Compare vs weight edit on same probe |
| SAE features | Can sparse features suggest mechanisms? | Qwen-Scope TopK SAE | Official Qwen3-1.7B SAE card | 537 MB/layer; no intervention loop | **scaffold** | Feature→target tensor mapping |
| Sparse weight perturbation | Can we change **actual** parameters reversibly? | Indexed delta on one tensor | safetensors single-key read/write | Local effect; not semantic editing | **implemented** | Scale n_params; layer choice policy |
| ROME / MEMIT | Targeted factual edits? | Rank-one updates | Qwen3 arch differs from GPT-2/J | Needs port + causal tracing | **not started** | Literature review on Qwen3 layers |
| Weight arithmetic / merge | Combine checkpoints? | Linear merge in weight space | Many public merge tools | Needs baseline lineage | **not started** | Tiny two-parent merge on 0.6B |
| Conventional fine-tuning | Train adapter or LoRA? | SGD on small set | Standard HF | Expensive; blurs intervention type | **not started** | Compare cost vs sparse edit |
| AURIX persona | Behavioural test instrument | Prompt + steering | `research/steering/aurix.py` | Not a target identity | **scaffold** | Post weight-edit AURIX probes |

Intervention taxonomy (do not conflate):

| Type | Changes stored weights? | Persistent across reload? |
|------|-------------------------|---------------------------|
| Memory update | No | N/A (data only) |
| Prompt | No | No |
| Activation steering | No | No |
| Sparse / ROME / merge / fine-tune | Yes | Yes (checkpoint) |
| Runtime policy change | No (software) | Yes (config/DB) |
