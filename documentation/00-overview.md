# Overview — framework-research

## Background

The Framework Desktop ships the **AMD Ryzen AI MAX+ 395** ("Strix Halo") —
a 16-core Zen 5 CPU and Radeon 8060S iGPU (gfx1151, RDNA 3.5) sharing
**128 GB of LPDDR5X** at ~215 GB/s over a single unified memory bus. This
is an unusual configuration: a consumer GPU normally competes with CPU
for memory via PCIe; here everything lives in the same pool.

Practically, this means:

- Models up to about **100 GB of quantized weights** can fit in GPU-visible
  memory without paging tricks.
- GPU and CPU can operate on the same tensor with zero copy cost.
- There is no dedicated VRAM bandwidth advantage — the Radeon runs at the
  same 215 GB/s as the rest of the system.

The research program in this repo set out to characterise this platform
for **large-model LLM inference**: which optimizations that the community
has developed for dedicated-VRAM GPUs still pay off here, which new ones
become possible with unified memory, and where the real performance walls
sit.

Thirty-four missions were run over the course of several weeks, grouped
into three phases:

### Phase 1 — Measurement (missions 01 – 18)

Establish the baseline. Sweep KV cache compression, prefix caching,
model quantization, context length, batch size, parallel slots,
quality vs. speed, and speculative decoding. Build a Pareto frontier
and identify which knobs matter.

### Phase 2 — Longer contexts and quality (missions 19 – 33)

Push what we learned. Q4_K_M with speculative decoding. Needle-in-a-
haystack at 16K/32K/64K. HumanEval full runs. Thermal sustain.
Tokens-per-watt. Multi-client load.

### Phase 3 — Heterogeneous inference (mission 34)

Attach a second node (RTX 3090, 24 GB) as a `llama.cpp` RPC worker.
Measure what it takes to split a single model across two machines with
different bandwidth / capacity profiles.

## The models

- **Qwen3.5-122B-A10B-REAP-20** (Q4_K_M / Q6_K / Q8_0) — primary test model
  for Phase 1 and 2. MoE, 122B total / ~10B active.
- **Qwen3.6-35B-A3B** (Q4_0 / Q4_K_M / Q5_K_M / Q6_K / Q8_0 / BF16) — used
  for the later quality/quant missions.
- **MiniMax-M2.5 Q4_K_M** (129 GB, 256 experts × 4.9B) — the one model
  that genuinely does not fit on a single node; unlocked in Mission 34.
- **Gemma-3-4B-it Q4_K_M + Gemma-3-1B-it Q4_K_M** — small
  target/draft pair for sanity checks and draft-model experiments.

## Reading this repo

1. Start with **[findings.md](findings.md)** for the consolidated
   results by theme (KV cache, prefix caching, spec decoding, long ctx,
   thermals, RPC).
2. For methodology and reproducibility see
   **[methodology.md](methodology.md)**.
3. To actually re-run anything see
   **[reproducing.md](reproducing.md)**.
4. Per-mission detail (one writeup per mission) lives under
   `documentation/missions/`.
5. Raw data is at `benchmarks/missions/<id>/results.json` or
   `results.jsonl`. A top-level `benchmarks/summary.csv` gives a
   one-line-per-mission index.

## What this repo is **not**

- A production inference stack. The harnesses are measurement code, not
  serving code.
- A benchmark suite for other hardware. Numbers here reflect this
  specific Framework Desktop + 3090 pair. Your mileage will vary.
- A drop-in recipe. llama.cpp build flags, kernel params, and
  `tensor-split` values shown here worked for our specific topology on
  builds `b8775` / `b8779`. Treat them as starting points.
