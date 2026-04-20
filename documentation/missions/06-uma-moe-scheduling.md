# 06-uma-moe-scheduling

## Hypothesis

Because the Halo has unified memory, CPU threads can prefetch the next layer's active experts into GPU-visible GTT while the GPU computes the current layer, eliminating the MUL_MAT_ID dispatch bottleneck.

## Success criteria

- {'prefill_improvement_pct': 15, 'or_profiling_report_identifies_bottleneck': True}

## Result

- **2026-04-16** — 2610.85 pp_t_s — Baseline captured. Vulkan-side expert cache and CPU prefetch thread remain to be implemented.

## Raw data

- Mission spec: `benchmarks/missions/06-uma-moe-scheduling/mission.json`
- Harness: `scripts/missions/06-uma-moe-scheduling.py`
