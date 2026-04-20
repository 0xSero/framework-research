# 15-long-gen

## Hypothesis

tg128 over-represents real-world decode throughput. As KV grows during generation, tg should degrade due to FLASH_ATTN_EXT scaling. Map tg at 128, 512, 2048, 8192 generated tokens.

## Result

- **2026-04-16** — -4.163052905464011 tg_drop_pct_128_to_8192 — tg128=23.06 → tg8192=24.02 (-4.2% drop)

## Raw data

- Mission spec: `benchmarks/missions/15-long-gen/mission.json`
- Results: `benchmarks/missions/15-long-gen/results.json`
- Harness: `scripts/missions/15-long-gen.py`
