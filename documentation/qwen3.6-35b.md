# Qwen3.6-35B-A3B — quantization & quality study

_Data snapshot: `2026-04-19T18:14:10.823209+00:00` (from `site/data.json`)._

Qwen3.6-35B-A3B is a 35B-parameter MoE (~3B active). We quantized the
same BF16 checkpoint eight different ways and measured speed and
quality on each, to build a Pareto frontier for this hardware.

## Speed (`llama-bench`)

Prefill tok/s on `pp512`, decode tok/s on `tg128`.

| variant | size_bytes | pp_tps | tg_tps |
|---|---|---|---|
| BF16 | 69376637280 | 396.16 | 10.7 |
| Q8_0 | 36903139360 | 974.81 | 52.74 |
| Q6_K | 28514152480 | 830.02 | 62.23 |
| Q5_K_M | 24729131040 | 942.83 | 64.1 |
| Q4_K_M | 21166757920 | 1021.37 | 70.15 |
| Q4_0 | 19715053600 | 1060.75 | 76.46 |
| IQ4_NL | 19947706400 | 890.75 | 73.09 |
| DYNAMIC | 20481456160 | 1099.76 | 63.98 |

## Perplexity

Lower is better. Measured on a WikiText document split.

| variant | ppl | size_bytes |
|---|---|---|
| DYNAMIC | 5.8635 | 20481456160 |
| Q4_K_M | 5.8936 | 21166757920 |
| Q6_K | 5.8479 | 28514152480 |
| Q8_0 | 5.8444 | 36903139360 |
| IQ4_NL | 5.925 | 19947706400 |
| Q4_0 | 5.9982 | 19715053600 |
| Q5_K_M | 5.9509 | 24729131040 |

## HumanEval (multi-quant)

| variant | n | n_pass | pass_at_1 | elapsed_s |
|---|---|---|---|---|
| Q4_0 | 50 | 48 | 0.96 | 78.4 |
| Q4_K_M | 50 | 49 | 0.98 | 89.5 |
| IQ4_NL | 50 | 50 | 1.0 | 83.4 |
| Q5_K_M | 50 | 49 | 0.98 | 92.5 |
| Q6_K | 50 | 49 | 0.98 | 95.1 |
| Q8_0 | 50 | 48 | 0.96 | 105.0 |

## MBPP (multi-quant)

| variant | n | n_pass | pass_at_1 | elapsed_s |
|---|---|---|---|---|
| Q4_0 | 60 | 56 | 0.9333 | 53.9 |
| IQ4_NL | 60 | 55 | 0.9167 | 62.1 |
| DYNAMIC | 60 | 56 | 0.9333 | 78.1 |
| Q4_K_M | 60 | 53 | 0.8833 | 80.1 |
| Q5_K_M | 60 | 54 | 0.9 | 74.9 |
| Q6_K | 60 | 53 | 0.8833 | 77.7 |
| Q8_0 | 60 | 54 | 0.9 | 79.1 |

## GSM8K (multi-quant)

| variant | n | n_pass | pass_at_1 | elapsed_s |
|---|---|---|---|---|
| Q4_0 | 40 | 39 | 0.975 | 204.5 |
| Q4_K_M | 40 | 39 | 0.975 | 227.9 |
| IQ4_NL | 40 | 39 | 0.975 | 137.7 |
| DYNAMIC | 40 | 39 | 0.975 | 147.9 |
| Q5_K_M | 40 | 39 | 0.975 | 216.1 |
| Q6_K | 40 | 39 | 0.975 | 195.9 |
| Q8_0 | 40 | 39 | 0.975 | 191.8 |

## Thinking vs. no-think

| mode | n | n_pass | pass_at_1 | total_elapsed_s | avg_tokens_per_problem | avg_elapsed_per_problem_s |
|---|---|---|---|---|---|---|
| direct | 30 | 30 | 1.0 | 46.3 | 153.9 | 6.03 |
| thinking | 30 | 19 | 0.6333 | 407.7 | 1663.3 | 53.21 |

## Raw data

- `benchmarks/qwen3.6-35b/Qwen3.6-35B-A3B-bench.jsonl` — llama-bench speed
- `benchmarks/qwen3.6-35b/qwen36_ppl.jsonl` — perplexity
- `benchmarks/qwen3.6-35b/qwen36_he_multiquant.jsonl` — HumanEval per quant
- `benchmarks/qwen3.6-35b/qwen36_mbpp_multiquant.jsonl` — MBPP per quant
- `benchmarks/qwen3.6-35b/qwen36_gsm8k_multiquant.jsonl` — GSM8K per quant
- `benchmarks/qwen3.6-35b/qwen36_thinking.jsonl` — thinking-mode impact
