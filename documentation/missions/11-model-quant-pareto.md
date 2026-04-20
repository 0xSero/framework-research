# 11-model-quant-pareto

## Hypothesis

On 128GB UMA with Vulkan, Q4_K_M should be prefill-fastest, Q6_K the tg-speed/quality sweet spot, and Q8_0 bandwidth-limited. Mapping all three across contexts exposes where each wins.

## Result

- **2026-04-16** — 343.1 best_pp_at_4k_t_s — pp@4K: {'Q4_K_M': 341.65, 'Q6_K': 289.48, 'Q8_0': 343.1} — tg@4K: {'Q4_K_M': 29.48, 'Q6_K': 24.3, 'Q8_0': 19.75}. Fastest prefill: Q8_0.

## Raw data

- Mission spec: `benchmarks/missions/11-model-quant-pareto/mission.json`
- Results: `benchmarks/missions/11-model-quant-pareto/results.json`
- Harness: `scripts/missions/11-model-quant-pareto.py`
