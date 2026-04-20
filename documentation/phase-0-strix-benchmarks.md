# Phase 0 — Strix Benchmarks

The research captured in this repo started before the structured
34-mission program. The first month was a build-and-optimise cycle on
`llama.cpp`'s Vulkan and ROCm backends, profiling bottlenecks, and
contributing upstream. That work is summarised here so it stays
discoverable alongside the mission series.

## Headline numbers

| Metric                | Value                               |
|-----------------------|-------------------------------------|
| Peak prefill          | **406 tok/s** (ROCm + MMQ, pp512)   |
| Peak decode           | **40.1 tok/s** (ROCm full stack)    |
| Best chat speedup     | **+64 %** (ROCm full stack vs. Vulkan stock) |
| Maximum context tested| **131 072 tokens**                  |

## The four configurations we compared

Model throughout: Qwen3.5-122B-A10B-REAP-20 Q6_K (76 GB), `-fa 1`,
`-ngl 99`, `-c` as noted.

| Config            | Backend | Patches                    | pp512 | tg128 | Best for                      |
|-------------------|---------|----------------------------|-------|-------|-------------------------------|
| Vulkan stock      | Vulkan  | none                       | 303   | 24.7  | Minimum latency               |
| Vulkan + spec     | Vulkan  | PR #20075 + 4 fixes        | 302   | 35.3  | Fast decode, easy deploy      |
| ROCm + MMQ        | ROCm 7.2.1 | PR #21344                | **406** | 18.2 | Long prefill, RAG             |
| ROCm full stack   | ROCm 7.2.1 | #21344 + #20075 + fixes  | 404   | **40.1** | Overall chat throughput   |

## Five things we learned

1. **Vulkan wins on memory mapping.** `vkAllocateMemory` with
   `HOST_VISIBLE_BIT` maps the full 120 GiB GTT. ROCm's `hipMalloc` is
   stuck in the BIOS VRAM window unless
   `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is exported explicitly.
2. **ROCm wins on batched compute.** PR #21344's MMQ VGPR tuning
   unlocked **+19 – 35 % prefill** on gfx1151.
3. **Speculative decoding stacks orthogonally with ROCm.** ROCm's
   single-token decode is slower than Vulkan's, but its batched-verify
   is so fast that speculative decoding flips the ranking. ROCm + spec
   became the fastest overall configuration.
4. **Decode doesn't degrade with context.** At 512 tokens or 131 072
   tokens, decode tok/s is essentially identical. Only prefill pays the
   long-context cost.
5. **MUL_MAT_ID is the hidden villain.** On MoE models the routed-
   expert path consumes **42 – 66 %** of prefill on the Vulkan backend.
   Documented upstream as issue #21948 with full `GGML_VK_PERF_LOGGER`
   data.

## Full build history

| Build                        | Backend | pp512 | pp8K+tg | pp32K+tg | pp128K+tg | tg128 | Notes                            |
|------------------------------|---------|-------|---------|----------|-----------|-------|----------------------------------|
| Vulkan b8779 stock           | Vulkan  | 268.89| 249.65  | 234.94   | 145.03    | 23.52 | Baseline                         |
| Vulkan spec build            | Vulkan  | 302.10| —       | —        | —         | 24.56 | PR #20075 + fixes                |
| ROCm stock (toolbox)         | ROCm    | 262.18| 228.66  | —        | 144.44    | 20.83 | Graph splits from VRAM limit     |
| ROCm + PR #21344             | ROCm    | 354.57| 302.44  | 291.03   | 171.66    | 21.00 | Best pure prefill                |
| ROCm stacked                 | ROCm    | 229.70| —       | —        | —         | 17.96 | `-ffast-math`+hipBLASLt regressed |
| ROCm full stack              | ROCm    | 405.99| —       | —        | —         | 18.29 | PR #21344 + PR #20075 + fixes    |
| REAP-20 Q4_K_M baseline      | Vulkan  | 443   | —       | —        | —         | 29.1  | Overnight baseline (smaller model)|
| REAP-40 Q4_K_M               | Vulkan  | 352   | 349     | 291      | 145       | 29.5  | Recommended smaller model        |

**What the "stacked" build taught us.** We combined PR #21344 with
`-ffast-math` and hipBLASLt expecting additive gains; prefill dropped
from 354 to 230 tok/s. Optimizations do not always compose cleanly —
test each in isolation before stacking.

## Real workload timing

`llama-server` with actual prompts; 256-token generation, `no_think`,
`temperature=0.3`.

| Workload                    | Vulkan stock | Vulkan + spec | ROCm + MMQ   | ROCm full stack | Best             |
|-----------------------------|--------------|---------------|--------------|-----------------|------------------|
| Chat (30 in, 1000 out)      | 41.5 s       | 31.9 s +30 %  | 56.9 s −27 % | **28.3 s +47 %**| ROCm full stack  |
| Code gen (2K in, 2K out)    | 118.3 s      | 99.1 s +19 %  | 138.0 s −14 %| **80.9 s +46 %**| ROCm full stack  |
| Summarize (8K in, 256 out)  | 155.9 s      | 153.5 s +2 %  | 114.5 s +36 %| **107.2 s +46 %**| ROCm full stack |

## Speculative decoding — what it took to make it work

Target: Qwen3.5-122B-A10B-REAP-20 Q6_K (76 GB). Draft: Qwen3.5-0.8B
Q4_K_M (508 MB).

```
effective_tps = accepted_per_step / (draft_time + verify_time)
```

### Results after fixes

| Task                       | Mode      | Baseline | Spec decode | Speedup | Acceptance |
|----------------------------|-----------|----------|-------------|---------|------------|
| Photosynthesis explanation | think     | 24.29    | 30.74       | +26.5 % | 76.0 %     |
| Neural nets explanation    | no_think  | 24.36    | **34.05**   | +39.8 % | 90.3 %     |
| BST code generation        | think     | 24.43    | 29.23       | +21.8 % | 80.6 %     |
| Short answer (2+2)         | think     | 24.43    | 29.70       | +23.8 % | 88.9 %     |

`no_think` mode is where spec decode shines (+40 % at 90 % acceptance).
Thinking mode drops acceptance to ~76 % because reasoning tokens are
harder for the draft to anticipate.

### The four bugs we had to fix

Beyond PR #20075, these were required for Qwen3.5 on non-Metal backends:

1. **Hybrid `seq_rm` skipped attention cleanup.** When recurrent
   rollback failed, the function returned early and never called
   `mem_attn->seq_rm()`. Stale positions accumulated, breaking M-RoPE
   invariants.
2. **Soft rollback corrupted positions.** The partial-removal path
   moved cell positions backward (`cells[i].pos = p0 - 1`) instead of
   erasing them, so `seq_pos_max` returned stale higher positions.
3. **Compat check was chicken-and-egg.**
   `common_speculative_is_compat()` tested `seq_rm` on a fresh context
   before any checkpoints existed. It always failed for hybrid models.
4. **Recurrent reserve size was too small.** With `-np 1`,
   `recurrent_rs_size = max(1, n_seq_max) = 1`. Checkpointing needs up
   to 8 cells per sequence; bumped to `max(16, n_seq_max * 16)`.

## Context scaling

Prefill slows with context; decode stays flat. Full sweep 64 → 32K:

| Context | Vk pp | Vk tg | Vk+spec tg | ROCm pp | ROCm tg | ROCm+spec tg | Best decode  |
|---------|-------|-------|------------|---------|---------|--------------|--------------|
| 64      | 98    | 24.3  | 26.6       | 130     | 17.8    | 28.5         | ROCm+spec    |
| 512     | 303   | 24.4  | 28.1       | 376     | 17.7    | 23.3         | Vk+spec      |
| 2K      | 353   | 24.2  | 25.2       | 423     | 17.7    | 26.7         | ROCm+spec    |
| 4K      | 362   | 24.2  | 25.5       | 413     | 17.7    | 25.3         | ROCm+spec    |
| 8K      | 371   | 23.9  | 25.1       | 407     | 17.4    | 27.4         | ROCm+spec    |
| 16K     | 353   | 23.4  | 25.0       | 371     | 17.0    | 23.6         | Vk+spec      |
| 32K     | 314   | 22.6  | 18.6       | 315     | 16.2    | 22.9         | ROCm+spec    |

ROCm's MMQ prefill advantage fades past 16K — flash attention dominates
batched matmul at that depth. Decode stays flat end-to-end.

Combined throughput from `llama-bench` (prefill + decode per-token):

| Context | Vulkan stock | ROCm + MMQ | Winner         |
|---------|--------------|------------|----------------|
| 512     | 93.0         | 77.8       | Vulkan (+20 %) |
| 2K      | 181.8        | 179.5      | Vulkan (+1 %)  |
| 8K      | 258.0        | 287.3      | ROCm (+11 %)   |
| 16K     | 266.0        | 304.4      | ROCm (+14 %)   |
| 32K     | 255.4        | 285.8      | ROCm (+12 %)   |
| 65K     | 217.2        | 236.8      | ROCm (+9 %)    |
| 131K    | 155.7        | 169.2      | ROCm (+9 %)    |

## What broke along the way

Worth keeping as prior art:

| Attempt                                      | What went wrong                                              | Lesson                                                 |
|----------------------------------------------|---------------------------------------------------------------|--------------------------------------------------------|
| Q8_0 full benchmarks                         | 105 GB model exceeded 120 GiB GTT; OOM-killed repeatedly     | Stick to Q6_K (76 GB) or Q4_K_M (57 GB)                |
| ROCm Q6_K without UMA env var                | Pathologically slow (>1 hr) due to 94 graph splits            | `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is mandatory       |
| ROCm with lhl's FATTN patches                | Patch conflicts with current master; wouldn't apply          | Needs rebase on a matching llama.cpp version           |
| Stacked build with `-ffast-math` + hipBLASLt | pp512 dropped from 354 to 230 tok/s                           | Test optimizations in isolation                        |
| Q6_K on stock Vulkan before kernel fix       | Infinite loading hang (>20 min) for any model > 64 GB        | `amd_iommu=off` is required                            |

## MUL_MAT_ID profiling

`GGML_VK_PERF_LOGGER=1` breakdown on Qwen3.5-122B Q6_K:

| Context | MUL_MAT_ID share | FLASH_ATTN_EXT share | Total ms |
|---------|------------------|----------------------|----------|
| 512     | **66.2 %**       | —                    | 1700     |
| 8K      | 57.6 %           | —                    | 1839     |
| 32K     | 41.9 %           | —                    | 2556     |
| 128K    | 19.7 %           | —                    | 5351     |

At short context, MUL_MAT_ID owns two-thirds of prefill. At long
context flash attention dominates. The fix pattern we want to see
on Vulkan is the map → batched-matmul → unmap approach that PR #13388
(already merged for Metal) uses.

## Upstream contributions

| Date    | Contribution                                                                                          | Details                                                                     |
|---------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Apr 15  | [llama.cpp issue #21948](https://github.com/ggml-org/llama.cpp/issues/21948)                           | Filed Vulkan MUL_MAT_ID bottleneck with full `GGML_VK_PERF_LOGGER` data.    |
| Apr 15  | [llama.cpp PR #21344](https://github.com/ggml-org/llama.cpp/pull/21344)                                | Validated MMQ VGPR tuning on gfx1151: +19 – 35 % prefill.                   |
| Apr 15  | [llama.cpp PR #20075](https://github.com/ggml-org/llama.cpp/pull/20075)                                | Documented four additional fixes for hybrid SSM/MoE spec decode off-Metal.  |
| Apr 14  | Kernel config fix                                                                                      | `amd_iommu=off` + `ttm.pages_limit=335544321` — addresses llama.cpp #14854. |

## Prior art we leaned on

- [llama.cpp PR #13388 — Metal MoE](https://github.com/ggml-org/llama.cpp/pull/13388) — map/batched-matmul/unmap pattern for MoE; 1.8 – 4.1× speedup on Metal. The template we'd like to see on Vulkan.
- [llama.cpp PR #15524 — Vulkan subgroup opts](https://github.com/ggml-org/llama.cpp/pull/15524) — already active in our build; still leaves a large gap.
- [llama.cpp issue #14854](https://github.com/ggml-org/llama.cpp/issues/14854) — the original > 64 GB Vulkan loading hang on Strix Halo.
- [Chips and Cheese — RDNA 3 Infinity Cache](https://chipsandcheese.com/p/rdna-3s-infinity-cache-friend-or) — why Infinity Cache mostly doesn't help LLM decode.

## Models used in Phase 0

| Model                               | Size  | Use                    | Source                                                                   |
|-------------------------------------|-------|------------------------|--------------------------------------------------------------------------|
| Qwen3.5-122B-A10B-REAP-20 Q6_K      | 76 GB | Main target            | [HF: 0xSero](https://huggingface.co/0xSero/Qwen3.5-122B-A10B-REAP-20-GGUF)|
| Qwen3.5-122B-A10B-REAP-20 Q4_K_M    | 57 GB | Smaller variant        | same                                                                      |
| Qwen3.5-122B-A10B-REAP-40 Q4_K_M    | 44 GB | Recommended tradeoff   | [HF: 0xSero](https://huggingface.co/0xSero/Qwen3.5-122B-A10B-REAP-40-GGUF)|
| Qwen3.5-0.8B Q4_K_M                 | 508 MB| Draft for spec decode  | Community GGUF                                                           |
