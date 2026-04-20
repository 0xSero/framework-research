# 12-batch-sweep

## Hypothesis

Vulkan MUL_MAT_ID is batch-size-sensitive. Default -b 2048 -ub 512 may be leaving throughput on the table for gfx1151 with 205-expert MoE routing.

## Result

- **2026-04-16** — 385.91 best_pp_t_s — Best b=8192 ub=2048: pp=385.91 tg=24.29

## Raw data

- Mission spec: `benchmarks/missions/12-batch-sweep/mission.json`
- Results: `benchmarks/missions/12-batch-sweep/results.json`
- Harness: `scripts/missions/12-batch-sweep.py`
