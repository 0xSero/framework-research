# 08-speculative-decoding

## Hypothesis

Using Qwen3.5-0.8B-Q4_K_M on GPU as a draft model can speed up decode (tg) for Qwen3.5-122B-A10B-REAP-20 Q6_K by at least 1.5x, recovering the tg bottleneck without hurting quality.

## Success criteria

- {'tg_speedup_x': 1.5, 'acceptance_rate_floor': 0.6}

## Result

- **2026-04-16** — 0.0 tg_speedup_x — Baseline tg=0.00 t/s. Best spec: draft_len=None tg=None accept=None. Speedup 0.00x.
- **2026-04-16** — 0.0 tg_speedup_x — Baseline tg=0.00 t/s. Best spec: draft_len=None tg=None accept=None. Speedup 0.00x.
- **2026-04-17** — 1.983 tg_speedup_x — Baseline tg=24.3 t/s (from Mission 11). Best spec: draft_len=5 tg=48.17896031534621 accept=0.4641025641025641. Speedup 1.98x.

## Raw data

- Mission spec: `benchmarks/missions/08-speculative-decoding/mission.json`
- Results: `benchmarks/missions/08-speculative-decoding/results.json`
- Harness: `scripts/missions/08-speculative-decoding.py`
