# 09-parallel-throughput

## Hypothesis

Under concurrent requests, aggregate t/s on the 122B MoE should exceed single-stream because MoE sparsity leaves GPU underutilized at batch=1. Target: aggregate tg ≥ 2x single-stream at concurrency=4.

## Result

- **2026-04-16** — 2.213 tg_aggregate_speedup_vs_npl1 — 1-slot tg=24.2; best @ npl=8 tg=53.55 (2.21x)

## Raw data

- Mission spec: `benchmarks/missions/09-parallel-throughput/mission.json`
- Results: `benchmarks/missions/09-parallel-throughput/results.json`
- Harness: `scripts/missions/09-parallel-throughput.py`
