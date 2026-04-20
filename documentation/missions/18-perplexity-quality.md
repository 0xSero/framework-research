# 18-perplexity-quality

## Hypothesis

Mission 11 shows Q4_K_M wins on speed. Mission 01 shows q4_0/q4_0 KV wins on memory. But neither measured quality. Wikitext perplexity on Q4_K_M with f16/q8_0/q4_0 KV will quantify what we trade for memory savings.

## Result

- **2026-04-16** — 0 ppl_delta_q4_0_vs_f16_pct — f16 KV PPL=None, q8_0 PPL=None, q4_0 PPL=None. q4_0 vs f16 delta=None%
- **2026-04-16** — 0.337 ppl_delta_q4_0_vs_f16_pct — f16 KV PPL=5.8134, q8_0 PPL=5.8168, q4_0 PPL=5.833. q4_0 vs f16 delta=0.33715209687963166%

## Raw data

- Mission spec: `benchmarks/missions/18-perplexity-quality/mission.json`
- Results: `benchmarks/missions/18-perplexity-quality/results.json`
- Harness: `scripts/missions/18-perplexity-quality.py`
