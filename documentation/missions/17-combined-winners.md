# 17-combined-winners

## Hypothesis

Combining Mission 11 (Q4_K_M), Mission 12 (-ub 2048), and Mission 09 (-np 8) should stack multiplicatively and push aggregate Qwen3.5-122B throughput well past 200 t/s.

## Result

- **2026-04-16** — 60.54 best_aggregate_tg_t_s — Q4_K_M + ub=2048 + parallel: best agg tg=60.54 total=200.69. vs Mission 09 baseline (Q6_K, ub=512, npl=8): s_tg=53.55 s_total=157.73

## Raw data

- Mission spec: `benchmarks/missions/17-combined-winners/mission.json`
- Results: `benchmarks/missions/17-combined-winners/results.json`
- Harness: `scripts/missions/17-combined-winners.py`
