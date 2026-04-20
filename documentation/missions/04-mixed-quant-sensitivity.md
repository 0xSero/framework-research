# 04-mixed-quant-sensitivity

## Hypothesis

A non-uniform quantization (attention at q8_0, FFN at q4_K_M, experts at q5_K_M) can match uniform Q6_K quality while using less memory and decoding faster.

## Success criteria

- {'beat_q6_k_on_quality_and_speed': True, 'or_match_q6_k_with_speedup_pct': 10}

## Result

- **2026-04-16** — 2651.55 pp_t_s — Baseline Q4_K_M benchmarked. Per-layer mixed quant tool (quantize_layers.py) needed for follow-up.

## Raw data

- Mission spec: `benchmarks/missions/04-mixed-quant-sensitivity/mission.json`
- Harness: `scripts/missions/04-mixed-quant-sensitivity.py`
