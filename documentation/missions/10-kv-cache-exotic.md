# 10-kv-cache-exotic

## Hypothesis

Beyond f16/q8_0/q4_0, types iq4_nl, q5_0, q5_1, and q4_1 have dramatically different Vulkan kernel characteristics. Mapping them on 122B reveals a hidden sweet spot.

## Result

- **2026-04-16** — 290.15 best_pp_t_s — Fastest exotic KV: iq4_nl @ 4096 pp=290.15 tg=24.29

## Raw data

- Mission spec: `benchmarks/missions/10-kv-cache-exotic/mission.json`
- Results: `benchmarks/missions/10-kv-cache-exotic/results.json`
- Harness: `scripts/missions/10-kv-cache-exotic.py`
