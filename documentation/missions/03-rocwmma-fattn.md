# 03-rocwmma-fattn

## Hypothesis

lhl's rocWMMA FATTN patch, once rebased, delivers +30-100% decode improvement at long context on RDNA 3.5, and the gain increases with KV length.

## Success criteria

- {'decode_improvement_pct': 20, 'prefill_regression_pct': 5}

## Result

- **2026-04-16** — 0 decode_improvement_pct — Baseline captured. ROCm build not available on this machine; patch must be tested on a ROCm-enabled node.

## Raw data

- Mission spec: `benchmarks/missions/03-rocwmma-fattn/mission.json`
- Harness: `scripts/missions/03-rocwmma-fattn.py`
