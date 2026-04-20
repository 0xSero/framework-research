# Models used in this research

## Primary targets

### Qwen3.5-122B-A10B (MoE) — REAP-pruned variants

The main subject of Phase 0 and Phases 1 – 2. Base model: `Qwen3.5-122B-A10B`
(122 B total, ~10 B active, 256 experts × 4 B). We used REAP-pruned versions
(Routing-Enhanced Activation Pruning) to cut expert count while preserving
quality.

| Model                                  | BPW   | Size    | Fits on Halo | Notes                                   |
|----------------------------------------|-------|---------|--------------|-----------------------------------------|
| Qwen3.5-122B-A10B-REAP-20 Q4_K_M        | 4.86  | 57 GB   | yes          | Fastest / default                        |
| Qwen3.5-122B-A10B-REAP-20 Q6_K          | 6.57  | 76 GB   | yes          | Higher quality; primary benchmark model  |
| Qwen3.5-122B-A10B-REAP-20 Q8_0          | 8.51  | 99 GB   | yes (tight)  | Near-lossless; ~85 % of UMA              |
| Qwen3.5-122B-A10B-REAP-40 Q4_K_M        | 4.86  | 73 GB   | yes          | Less-aggressively pruned                 |

REAP-20 removes 20 % of experts (256 → 205 per layer) and retains ~97.9 %
average capability on standard benchmarks. REAP-40 removes 40 % and
retains ~94 %. The tradeoff is interesting on this platform because we
care about both speed and memory footprint.

### Qwen3.6-35B-A3B (MoE)

Smaller sibling used for the quantization / quality study. 35 B total,
~3 B active. Because it's smaller, we could produce every common
quantization and compare them head-to-head on the same hardware.

See [`qwen3.6-35b.md`](qwen3.6-35b.md) for the full quant × quality
matrix (8 quantizations × speed, perplexity, HumanEval, MBPP, GSM8K,
needle, SWE, parallel throughput).

### MiniMax-M2.5 Q4_K_M

129 GB Q4_K_M of the 256-expert × 4.9 B MiniMax MoE. This is the
**only model in the set that does not fit on the Halo alone**. Used
in Mission 34 to demonstrate the RPC split unlocking models beyond
the standalone cap.

### Gemma-3-4B-it / Gemma-3-1B-it

Used as a small target/draft pair for quick sanity checks and some of
the speculative-decoding experiments that do not need the full 122 B
model.

## Hardware-first model recommendations

| Your setup                               | Recommendation                               |
|------------------------------------------|----------------------------------------------|
| Framework Desktop, 64 GB GTT default     | Qwen3.5-122B-A10B-REAP-20 **Q4_K_M**          |
| Framework Desktop, 120 GB GTT enabled    | Qwen3.5-122B-A10B-REAP-20 **Q6_K**            |
| Framework Desktop + 128 GB UMA           | Qwen3.5-122B-A10B-REAP-20 **Q8_0** (tight)    |
| Framework Desktop + any RPC peer         | MiniMax-M2.5 Q4_K_M, or any 100 GB+ model     |
| NVIDIA A100 80 GB                        | Q4_K_M or Q6_K                                |
| Apple M-series 128 GB                    | All quants via Metal                          |
| RTX 4090 (24 GB) alone                   | Use a smaller model — MoE won't fit           |

## REAP — why expert pruning works

Large MoE models over-provision experts. REAP
([arxiv:2510.13999](https://arxiv.org/abs/2510.13999)) identifies
experts whose routing frequency is near zero and removes them while
preserving "super-expert" paths the router relies on.

Applied to Qwen3.5-122B:

| REAP level | Experts retained | Size reduction (Q4_K_M) | Avg capability retained |
|------------|------------------|-------------------------|-------------------------|
| 0 % (baseline) | 256            | —                       | 100 %                   |
| 20 %          | 205             | −20 %                   | 97.9 %                  |
| 40 %          | 154             | −40 %                   | 94 %                    |

The 20 % variant is the sweet spot for this hardware — near-baseline
quality, 1.25× the speed, fits in GTT.

## Credits

- **Base models**: Qwen Team, MiniMax-AI, Google Research.
- **REAP pruning**: applied by [0xSero](https://huggingface.co/0xSero)
  following the recipe in arxiv:2510.13999.
- **Quantization**: `llama-quantize` from llama.cpp.
