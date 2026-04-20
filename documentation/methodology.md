# Methodology

## Nodes & roles

Two nodes participated in the study.

| Label        | Role                              | Hardware summary                      |
|--------------|-----------------------------------|---------------------------------------|
| **Driver**   | runs `llama-server`, issues prompts | Framework Desktop (Strix Halo, Radeon 8060S, 128 GB UMA) |
| **Worker**   | RPC peer for layer-parallel split   | Companion box (RTX 3090, 24 GB GDDR6X) |

Harnesses drive the system over SSH; scripts read node addresses and key
paths from environment variables (`DRIVER_HOST`, `DRIVER_KEY`, etc.) so
nothing is hard-coded.

## Inference engine

All experiments use `llama.cpp`:

| Build | Used by missions | Flags of note                                   |
|-------|------------------|-------------------------------------------------|
| b8775 | 01 – 33          | Vulkan, `GGML_CUDA=OFF`                          |
| b8779 | 34               | driver: Vulkan + RPC; worker: CUDA 12.8 + RPC, `GGML_CUDA_FA=OFF` |

The `-DGGML_CUDA_FA=OFF` workaround on the worker avoids a `ptxas
movmatrix` PTX-ISA mismatch on SM 8.6 under CUDA 12.8 + gcc-11. See
[build-notes.md](build-notes.md) for the full build procedure.

## Measurement conventions

- **Prefill tok/s** and **decode tok/s** are reported from
  `llama-server`'s `timings.prompt_per_second` and
  `timings.predicted_per_second` respectively.
- **Wall time** is measured on the client (the mission harness).
- **Network throughput** for RPC runs is sampled from `/proc/net/dev`
  every 500 ms over the entire request and reported as sustained MB/s
  plus total RX/TX bytes.
- **Context length** unless otherwise noted is 4096. Long-context runs
  explicitly state the `-c` value.
- **Seed / temperature**: `temperature=0` where reproducibility matters,
  `temperature=0.1` for parallel-client tests.
- **Prompt**: short (~20 tokens) unless a given mission explicitly
  sweeps prompt size.
- **Decode length**: typically `n_predict=128`.

## Warm-up

Each harness fires a small decode (n_predict=4) before the real run to
avoid measuring one-time kernel compile/cache costs. Not every mission
includes this but the headline decode-tok/s numbers all do.

## Variance

Phase 1 baselines were each run 3× and the median reported. The RPC
mission reports single-run numbers because of the long model-load time
(6 min to load 80 GB over Wi-Fi) — variance observed between back-to-back
runs was under 2 % on decode tok/s.

## Success criteria (per mission)

Each mission defined success criteria in its `mission.json`. Those
criteria are reproduced in the per-mission writeup under
`documentation/missions/`. A mission is considered **complete** when its
criteria are met or when a conclusive negative result is logged with the
primary metric.
