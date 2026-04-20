# 13-cpu-tuning

## Hypothesis

16C/32T Zen 5 with UMA means CPU involvement in MoE expert routing and kernel launch matters more than on dedicated-VRAM GPUs. The default --threads=16 may not be optimal.

## Result

- **2026-04-16** — 295.25 best_pp_t_s — Best t=24 prio=0: pp=295.25 tg=24.36

## Raw data

- Mission spec: `benchmarks/missions/13-cpu-tuning/mission.json`
- Results: `benchmarks/missions/13-cpu-tuning/results.json`
- Harness: `scripts/missions/13-cpu-tuning.py`
