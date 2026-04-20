# 14-moe-offload

## Hypothesis

On 128GB UMA, moving some MoE expert layers to CPU (-ncmoe N) could free GPU scheduling for the active path. With 8-of-205 expert sparsity, most of the model is dead weight at any timestep.

## Result

- **2026-04-16** — 294.45 best_pp_t_s — Best ncmoe=0: pp=294.45 tg=24.35

## Raw data

- Mission spec: `benchmarks/missions/14-moe-offload/mission.json`
- Results: `benchmarks/missions/14-moe-offload/results.json`
- Harness: `scripts/missions/14-moe-offload.py`
