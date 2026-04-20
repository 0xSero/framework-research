# framework-research

Research artifacts from a multi-month program pushing LLM inference on the
AMD **Strix Halo** platform (Framework Desktop — Ryzen AI MAX+ 395, Radeon
8060S iGPU, 128 GB unified LPDDR5X) and an **RTX 3090** companion node via
`llama.cpp` over RPC.

Thirty-four numbered missions explore the frontier of what this hardware
can do for large-model inference: KV-cache compression, prefix caching,
rocWMMA Flash Attention, mixed-precision quantization, NPU experiments,
UMA-native MoE scheduling, speculative decoding, long-context / needle
benchmarks, and a final heterogeneous RPC split between the Radeon and the
3090.

## Repository layout

```
framework-research/
├── benchmarks/       raw results (JSON/JSONL) per mission, plus summary CSVs
├── scripts/          harnesses that produced the numbers, generic env-var driven
└── documentation/    mission-level writeups, methodology, findings
```

Start with [`documentation/00-overview.md`](documentation/00-overview.md) for
the full story. Individual writeups live under `documentation/missions/`,
corresponding raw data under `benchmarks/missions/`, and runnable harnesses
under `scripts/missions/`.

## Hardware under test

| Node              | Role             | Accelerator                      | Memory         | Notes                |
|-------------------|------------------|----------------------------------|----------------|----------------------|
| Framework Desktop | primary / driver | Radeon 8060S (gfx1151, RDNA 3.5) | 128 GB LPDDR5X | Vulkan backend       |
| Companion box     | RPC worker       | NVIDIA RTX 3090 (GA102)          | 24 GB GDDR6X   | CUDA 12.8 backend    |

All results reported are from `llama.cpp` builds `b8775` / `b8779`, unless a
mission notes otherwise.

## Highlights

- **Mission 01 — KV cache frontier.** Found 14 Pareto-optimal cache
  combinations. Best long-context result: `f16/f16` KV reached 131 K tokens
  at pp=152.76 / tg=24.58 tok/s. Lower KV precisions trade speed for recall.
- **Mission 08 — Speculative decoding.** 1.98× decode speedup on the 122B
  target with a 0.8B draft at `draft_len=5`.
- **Mission 09 — Parallel throughput.** 2.21× aggregate throughput at
  `npl=8` (53.55 tok/s agg) vs. single-slot.
- **Mission 17 — Combined winners.** Stacking Q4_K_M + `ubatch=2048` +
  parallel slots hit 60.54 agg tok/s total / 200.69 prefill tok/s.
- **Mission 24 — Thermal sustain.** −0.08 % throughput drift over 60
  minutes. The platform holds performance under continuous load.
- **Mission 34 — Heterogeneous RPC inference.** Split Qwen3.5-122B Q6_K
  across the Radeon and the 3090 over Wi-Fi; decoded at 23.24 tok/s
  (within 4.3 % of Halo-solo 24.25 tok/s). Also loaded
  **MiniMax-M2.5 Q4_K_M (129 GB)**, which does **not fit on either node
  alone** — 22.1 GB on the 3090, 109.5 GB on the Halo — at 23 tok/s decode.

Full numbers in [`benchmarks/summary.csv`](benchmarks/summary.csv) and
per-mission writeups under [`documentation/missions/`](documentation/missions/).

## Reproducing

Every harness reads its inputs from environment variables so the repo does
not hard-code any particular network topology. See
[`documentation/reproducing.md`](documentation/reproducing.md) for the full
setup: what to export, how to launch an RPC peer, what models were used.

## License

MIT — see [LICENSE](LICENSE). Benchmark data is provided as-is with no
warranty about applicability to any other hardware.
