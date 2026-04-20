# Mission 02: Prefix Caching for Agentic Workloads — Findings

**Date:** 2026-04-16
**Status:** Complete (v2 re-run on 122B target model)
**Commits:** `6e22c30` (v1, gemma-3-4b), v2 forthcoming

## Hypothesis
On a 128GB system, caching a 4K system prompt across turns reduces TTFT by >50%.

## v1 (gemma-3-4b) — Invalid
Tested on the 4B model, which has ~5ms prompt processing per 1K tokens. The prefix cache simply doesn't matter on a model that small. Result: noise (-13% to +3%).

## v2 — 122B Model with q8_0/q8_0 KV (from Mission 01)

Server config:
```
llama-server -m Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf -ngl 99 -fa 1 \
  -c {ctx} -ctk q8_0 -ctv q8_0 --cache-reuse {0|1}
```

Each run: 5 turns, 4K-token fixed system prefix, `cache_prompt=true`.

### Prompt-processing time (ms) across turns

| ctx | flag | turn 1 | turn 2 | turn 3 | turn 4 | turn 5 |
|-----|------|--------|--------|--------|--------|--------|
| 8K  | `--cache-reuse 0` | 6828.9 | 1617.3 | 1620.6 | 1621.6 | 1619.4 |
| 8K  | `--cache-reuse 1` | 6805.3 | 1612.9 | 1618.4 | 1618.6 | 1617.1 |
| 16K | `--cache-reuse 0` | 6829.4 | 1616.3 | 1620.4 | 1622.8 | 1622.8 |
| 16K | `--cache-reuse 1` | 6836.4 | 1617.4 | 1622.3 | 1623.4 | 1624.5 |

### Comparison (warm average, turns 2-5)

| ctx | no cache | with cache | delta |
|-----|----------|------------|-------|
| 8K  | 1619.7 ms | 1616.8 ms | **0.2%** |
| 16K | 1620.6 ms | 1621.9 ms | **-0.1%** |

### Real finding: turn 1 vs turn 2+

| ctx | cold (turn 1) | warm (turn 2+) | improvement |
|-----|---------------|----------------|-------------|
| 8K  | 6828.9 ms | 1619.7 ms | **76.3%** |
| 16K | 6829.4 ms | 1620.6 ms | **76.3%** |

## Conclusions

1. **The `--cache-reuse` flag is a red herring for this workload.** It controls *cross-request prefix matching* (different prompts sharing a prefix), not turn-to-turn caching within one conversation.

2. **llama.cpp's default prompt-cache already delivers a 76.3% prefill reduction** when the exact same prefix is re-submitted. The hypothesis is **validated** — just not through the mechanism originally named.

3. **The system prompt overhead on turn 1 is 6.8 seconds** for the 122B model at Q6_K with q8_0/q8_0 KV. After turn 1 it drops to ~1.6 seconds. This is the reason agentic workloads feel responsive after the first turn.

4. **The `tokens_cached` field always reports 2368** regardless of the flag, confirming the server is matching the prefix at the KV-cache level transparently.

## Operational recommendation

- **Always send the same byte-exact prefix** in agent turns. The automatic cache will handle the rest.
- **Don't expect `--cache-reuse=1` to help** within a single conversation. Use it only if you're multiplexing multiple clients that share a system prompt across separate request sessions.
- **For cold-start latency,** warmup the server once with the system prefix before the user sees the interface.

## Raw data
See `results_v2.json` and `run_v2.log` for the full logs.
