# documentation/

Narrative writeups for the research program.

- [`00-overview.md`](00-overview.md) — what this repo is and how it's organised.
- [`findings.md`](findings.md) — consolidated results grouped by theme.
- [`methodology.md`](methodology.md) — how measurements were taken.
- [`reproducing.md`](reproducing.md) — env-var setup and step-by-step replay.
- [`build-notes.md`](build-notes.md) — llama.cpp build flags and workarounds.
- [`phase-0-strix-benchmarks.md`](phase-0-strix-benchmarks.md) — Vulkan vs.
  ROCm comparison, MUL_MAT_ID profiling, upstream contributions and the
  four spec-decode fixes. All the work that predates the 34-mission series.
- [`platform-setup.md`](platform-setup.md) — full Fedora + kernel + llama.cpp
  setup recipe used across every benchmark.
- [`models.md`](models.md) — Qwen3.5-122B REAP variants, Qwen3.6-35B,
  MiniMax-M2.5, Gemma pair. Hardware-first recommendations.
- [`qwen3.6-35b.md`](qwen3.6-35b.md) — quantization × quality matrix for
  Qwen3.6-35B-A3B (8 quants × speed + perplexity + HumanEval + MBPP +
  GSM8K + needle + SWE-rebench).

Per-mission writeups live in [`missions/`](missions/). Each one references
the raw data under `benchmarks/missions/` and the harness under
`scripts/missions/`.
