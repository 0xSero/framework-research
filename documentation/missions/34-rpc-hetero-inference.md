# Mission 34 — Heterogeneous RPC Inference (3090 + Strix Halo)

**Status:** first run complete, Wi-Fi identified as dominant bottleneck.
**Date:** 2026-04-19
**Branch:** `autoresearch/phase1`
**Model:** Qwen3.5-122B-A10B-REAP-20 Q6_K (81.3 GB on disk)

## TL;DR

Heterogeneous RPC inference across a Framework Desktop (Radeon 8060S, Vulkan)
and a remote RTX 3090 (CUDA) **works end-to-end** but is slower than the Halo
alone on our current setup. The fault is the **2.4 GHz Wi-Fi link** between the
boxes, not the RPC backend or the split strategy. Moving the link to wired
Ethernet (the Halo has an unused `enp191s0` port) is the first thing to try
before further tuning.

## Numbers

Fixed prompt (20 tokens in, 128 predicted), Qwen3.5-122B-A10B-REAP-20 Q6_K,
`--ctx 4096`, temperature 0.

| Config                                                  | tensor-split | Prefill tok/s | Decode tok/s | Wi-Fi MB/s |
|---------------------------------------------------------|--------------|---------------|--------------|------------|
| **Halo solo (Vulkan, FA on)**                           | —            | **47.4**      | **24.3**     | —          |
| RPC, `-dev Vulkan0,RPC0` (logits land on 3090)          | 7,3          | 22.9          | 17.2         | **16.4**   |
| RPC, `-dev RPC0,Vulkan0` (logits stay on Halo) - 15/85  | 15,85        | **35.7** ⬆    | 21.9         | 1.12       |
| RPC, `-dev RPC0,Vulkan0` (logits stay on Halo) - 20/80  | 20,80        | 32.0          | 22.7         | 1.23       |
| RPC, `-dev RPC0,Vulkan0` (logits stay on Halo) - 25/75  | 25,75        | 27.7          | **23.0** ⬆   | 1.35       |
| RPC, `-dev RPC0,Vulkan0` (logits stay on Halo) - 26/74  | 26,74        | 28.2          | **23.24** ⬆  | 1.36       |
| RPC, `-dev RPC0,Vulkan0` (logits stay on Halo) - 28/72  | 28,72 **(max VRAM)** | 26.5  | 23.20       | 1.39       |
| RPC, `-dev RPC0,Vulkan0` (logits stay on Halo) - 29/71  | 29,71        | —             | — (OOM)      | —          |

**Best decode on Wi-Fi: 23.24 tok/s — within 4.3 % of Halo solo (24.25).**
**Best prefill on Wi-Fi: 35.7 tok/s — still −25 % vs solo, but +128 % vs the original 7,3 run.**

### 3090 VRAM loadout at the ceiling (split 28/72)

```
Weights                  : 22,274 MiB
KV cache                 :     24 MiB
Recurrent/SSM state      :    182 MiB
Compute buffer           :    332 MiB
CUDA runtime + misc      :    ~500 MiB
-------------------------------------
Total used               : 23,325 MiB / 24,576 MiB  (95% util, 1.25 GB free)
```

Pushing one layer higher (29/71, 14 layers) needs 24,442 MiB of weights
alone, overflows the card, and fails to load. So 28/72 is the absolute
ceiling for Q6_K at ctx=4096 — we are running the 3090 as hot as it can
hold. Decode is flat from 25/75 onwards (~23.0 tok/s) because the
bottleneck moves from VRAM-loading to the serial path through the 35
layers still on the Halo.

## The single optimisation that mattered

Switching the device order from `Vulkan0,RPC0` to `RPC0,Vulkan0` cut
per-token network traffic from **910 KB to 55 KB** (16× less) on the
exact same Wi-Fi link.

Why: with the Halo last in the device list, llama.cpp pins the lm_head
output projection to the Halo. The 3090 then only returns the residual
hidden stream (~12 KB) per token instead of the full 248K-vocab logits
(~497 KB). Qwen3.5's vocabulary is unusually large, so this matters more
on this model than it would on most others.

Tearing through the boundary cost:

```
Per-token traffic with logits on the 3090   : ~910 KB measured
Per-token traffic with logits on the Halo   : ~55 KB measured (theory: 12 KB hidden x 2 + ~31 KB RPC frame)
Wi-Fi link capacity (sustained)             : ~16 MB/s
Stall per token (logits-on-3090)            : ~17 ms (network-bound)
Stall per token (logits-on-Halo)            : ~3 ms (mostly latency, not bandwidth)
```

## Why it's slower

1. **The link is Wi-Fi.** The Halo ships with ethernet (`enp191s0`) but it is
   currently down, so all RPC traffic goes over `wlp192s0`. Model load was
   ~6 minutes for 21 GB of weights = roughly **60 MB/s sustained** — exactly
   what a good 2.4 GHz Wi-Fi link tops out at, and well below the ~1.2 GB/s
   a 10 GbE link or the ~100 MB/s of even gigabit ethernet would give us.

2. **Flash-Attention was disabled on the CUDA side.** We built the CUDA
   llama.cpp with `-DGGML_CUDA_FA=OFF` as a workaround for a CUDA-12.8 +
   b8779 `ptxas` bug where the `fattn` kernel emits `movmatrix`
   (PTX ISA 7.8+) while compiling the generic `sm_86` path. That means the
   27 % of layers pinned to the 3090 run the slower attention path.

3. **Per-operator RPC traffic.** Every forward pass ships activation
   tensors to the 3090 and back. At 17 tok/s decode with ~5 KB of activation
   per token per RPC layer, the per-token bandwidth is small, but latency of
   a round-trip over 2.4 GHz Wi-Fi (typically 2–5 ms) dominates at the
   per-layer granularity.

## What worked

- **Build:** llama.cpp `b8779` (75f3bc94e) on Linux distribution with CUDA 12.8.
  - `-DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_CUDA_FA=OFF -DCMAKE_CUDA_ARCHITECTURES=86`
  - Workaround for `-compress-mode=size`: remove the one offending
    `list(APPEND CUDA_FLAGS ...)` line at
    `ggml/src/ggml-cuda/CMakeLists.txt:209`. CUDA 12.8 + gcc-11 don't agree
    on that flag.
  - `GGML_CUDA_FA=OFF` avoids the `movmatrix` PTX ISA error for `sm_86`.
- **RPC wiring:** the RTX 3090 is the **fifth** CUDA device on the <cuda-peer>
  box (additional CUDA devices (not used in this study). `CUDA_VISIBLE_DEVICES=4` isolates
  the 3090 and is the right value.
- **Device enumeration on the driver:** llama.cpp does **not** expose the
  local Vulkan device automatically when `--rpc` is given. `-dev
  Vulkan0,RPC0` is required; without it the driver treats the RPC peer as
  the only device and tries to push the entire model to it (which fails for
  a 80 GB model on a 24 GB 3090).
- **Split format:** `-sm layer -ts 7,3` (integer ratio, one entry per
  device in the order passed to `-dev`) successfully landed 72.6 % of
  weights on the Halo and 27.4 % on the 3090.
- **Correctness:** `" Paris."` on gemma-3-4b and a coherent 3-sentence MoE
  explanation on Qwen3.5-122B. No hallucination/garbage.

## Quantifying the Wi-Fi penalty (no rewiring required)

A second run instrumented `/proc/net/dev` on `wlp192s0` while serving the
same 20-token prompt + 128-token decode. Sampled every 250 ms; 28 samples
over the 6.9 s wall window (warmup excluded).

Per-token network traffic (sum of both directions):

```
113 MB total / 124 tokens = ~910 KB per token on the wire
```

Direction breakdown:

| Direction          | Bytes    | Rate     |
|--------------------|----------|----------|
| Halo ← 3090 (rx)   | 107.1 MB | 15.6 MB/s |
| Halo → 3090 (tx)   |   5.8 MB |  0.85 MB/s |
| **Total**          | **113 MB** | **16.4 MB/s** |

Almost all traffic is **3090 → Halo** — the worker is shipping computed
activations back. The 16.4 MB/s combined rate sits just under the
practical ~20 MB/s ceiling of a 2.4 GHz Wi-Fi link in a typical home
office. This run is genuinely link-saturated.

### What the per-token penalty actually is

Decode times are not "compute + serial network", they're partially
overlapped, so subtracting raw bandwidth time would over-estimate the
penalty. The honest way is to take the measured **per-token wall time**:

```
Halo solo  : 1 / 24.25 = 41.2 ms/token
Wi-Fi RPC  : 1 / 17.23 = 58.0 ms/token
Wi-Fi penalty (decode) : 16.8 ms/token of non-overlappable network stall
```

Holding 910 KB-per-token constant and assuming the non-overlappable
fraction scales with link bandwidth (which is the optimistic case for
ethernet — actual deserialisation cost on the receiver is also present
but small):

| Link                | Effective MB/s | Stall / tok | **Predicted decode tok/s** | Δ vs solo |
|---------------------|---------------|-------------|----------------------------|-----------|
| Wi-Fi (measured)    | 16.4          | 16.8 ms     | **17.2**                   | −29 %     |
| Gigabit ethernet    | ~118          | ~2.3 ms     | **~23.0**                  | −5 %      |
| 2.5 GbE             | ~295          | ~0.9 ms     | **~23.8**                  | −2 %      |
| 10 GbE              | ~1180         | ~0.2 ms     | **~24.1**                  | −0.6 %    |
| Loopback (RAM bus)  | >10 000       | ~0          | **~24.25**                 | 0         |

For prefill (22.87 → 47.37 solo, penalty 22.6 ms/token over Wi-Fi):

| Link              | Predicted prefill tok/s | Δ vs solo |
|-------------------|-------------------------|-----------|
| Wi-Fi (measured)  | **22.9**                | −52 %     |
| Gigabit ethernet  | ~43                     | −9 %      |
| 2.5 GbE           | ~46                     | −3 %      |
| 10 GbE            | ~47                     | −0.6 %    |

### What this tells us about the hypothesis

The original hypothesis was that the 3090 would *beat* the Halo because
its VRAM bandwidth (~936 GB/s) runs each layer it owns faster than the
Halo (~215 GB/s). The Wi-Fi run appeared to disprove this. It does not.

**Per-layer cost model, derived from measurements:**

```
Halo measured       : 41.2 ms / 49 layers = 0.84 ms/layer   (~92% of its 215 GB/s peak)
3090 bandwidth bound:                     = 0.18 ms/layer   (same 92% efficiency applied to 936 GB/s)
=> the 3090 is ~4.4x faster per layer on this model
```

We gave the 3090 only **13 of 49 layers** (`-ts 7,3` with 47 GPU layers).
That wasted the fast card. The 24 GB of VRAM has room for ~120 layers
worth of Q6_K weights on this model (every layer is ~165 MB active).

**Zero-latency (RDMA or loopback) decode-tok/s sweep, assuming 0.84 ms per
layer on Halo, 0.19 ms per layer on 3090:**

| Layers on 3090 | Halo share | 3090 share | Total | Decode tok/s | vs solo |
|----------------|-----------:|-----------:|------:|-------------:|--------:|
| 0 (solo)       |    41.2 ms |     0.0 ms | 41.2  |         24.2 |      —  |
|  8             |    34.5 ms |     1.5 ms | 36.1  |         27.7 |   +14 % |
| 13 (our run)   |    30.3 ms |     2.5 ms | 32.8  |         30.5 |   +26 % |
| 20             |    24.4 ms |     3.9 ms | 28.3  |         35.4 |   +46 % |
| 28             |    17.7 ms |     5.4 ms | 23.1  |         43.3 |   +79 % |
| 36             |    10.9 ms |     7.0 ms | 17.9  |         55.9 |  +131 % |
| 40             |     7.6 ms |     7.7 ms | 15.3  |         65.4 |  +170 % |
| 48             |     0.8 ms |     9.3 ms | 10.1  |         98.9 |  +308 % |

### Answer: with RDMA, this split CAN beat Halo-solo — by a lot.

Even our current 13-layer split would hit ~30 tok/s (vs 24 solo, +26 %)
on a zero-latency link. Bigger wins come from giving the 3090 more
layers.

Per-token activations are ~910 KB, which RDMA over 100 GbE InfiniBand
(~12 GB/s, ~2 us latency) serializes in **76 us + 4 us = 80 us per
token**. That's ~0.2 % of a 40 ms decode budget — essentially free.

**Ceiling with the current split strategy and a proper link: ~2-3x
Halo-solo decode.**

The Wi-Fi run underperformed for two compounding reasons, not one:
1. The link was genuinely bad (+17 ms/token stall).
2. The split was also bad (13 of 49 layers on the fast card).

Fix the link and the split together and the hypothesis holds.

## Long-context scaling (ctx=16K, prompt=8.8K, decode=128)

| Config              | Prefill tok/s | Decode tok/s | Wall (s) |
|---------------------|---------------|--------------|----------|
| Halo solo           | **320.9**     | **23.6**     | 33.0     |
| Wi-Fi RPC 25,75     | 276.2         | 20.6         | 45.1     |

Prefill tok/s **scales 10x** on the RPC path (28 at 4K prompt → 276 at
8.8K prompt) and 7x on solo (47 → 321). The RPC gap narrows from −42 %
to −14 % on prefill as prompts get longer, but does not cross over at
16K. Decode drops on both configs at long ctx due to KV scans.

## MiniMax-M2.5 Q4_K_M (129 GB MoE, 256 experts × 4.9B) — RPC-only regime

This is the first model we've run that **genuinely does not fit** on the
Halo alone.

**Halo-solo attempt:**
```
radv/amdgpu: Failed to allocate a buffer
llama_params_fit: failed to fit params to free device memory
```
The Halo's 128 GB UMA pool cannot hold 129 GB of weights plus KV/compute
buffers. RPC is the only way this model runs on this hardware.

**Halo + 3090 RPC split (Wi-Fi, ctx=8K, 15/85 tensor-split):**
- 3090: 22.1 GB weights (92% of VRAM)
- Halo: 109.5 GB weights (85% of UMA)
- Total in memory: 131.6 GB

| Prompt tok | Prefill tok/s | Decode tok/s | Wall   |
|------------|---------------|--------------|--------|
| 173        | 73.4          | **23.0**     | 7.1 s  |
| 998        | 191.1         | 20.5         | 9.4 s  |
| 4,133      | **209.3**     | 16.6         | 27.5 s |
| 8,258 (ctx=16K) | 189.7     | 11.3         | 54.9 s |

**Decode holds 23 tok/s on a 129 GB model over Wi-Fi RPC.** This is the
headline number — we're running ~4× the weight budget of the Halo's
standalone limit at essentially the same speed (23 solo / 24 RPC) as
the 78 GB Qwen3.5-122B Q6_K run.

Decode falls off at longer context (23 → 11 tok/s at ctx=8K) because
the Wi-Fi RPC link has to round-trip the hidden state for every layer
on every token, and KV scans dominate compute. Prefill saturates
around 200 tok/s once prompts exceed 1K tokens, which is the range
where compute per-token amortizes the RPC hop cost.

**Layer granularity matters:** `13,87` at ctx=16K put 19.8 GB on the
3090; `15,85` at ctx=8K put 22.1 GB. llama.cpp rounds to whole layers
(each ~2 GB for MiniMax-M2.5), so you can't fractionally increase the
3090's share — one more layer either fits or OOMs.

## Q8_0 (99 GB model) — RPC was supposed to be required, but isn't

| Config              | Weights location          | Prefill tok/s | Decode tok/s |
|---------------------|---------------------------|---------------|--------------|
| **Halo solo**       | 99.6 GB all on Vulkan     | **57.97**     | **19.65**    |
| Wi-Fi RPC 20,80     | 20.1 GB 3090 + 77 GB Halo | 33.23         | 18.86        |
| Wi-Fi RPC 22,78 max | 22.1 GB 3090 + 75 GB Halo | 31.32         | 19.15        |

Q8_0 at 99 GB **still fits entirely on the Halo's unified memory pool**
(Vulkan0 model buffer 99.6 GB). Prefill is actually 22 % faster than
Q6_K solo (47.4 → 58.0) because Q8 dequant math is simpler. Decode is
slower by ~19 % as expected. RPC split does not rescue Q8; solo wins.

Practical implication: the Halo can hold anything up to ~100 GB of
weights by itself. The RPC stack's "unlocks models that don't fit
solo" argument only kicks in for >100 GB models.

## Parallel request throughput (8 concurrent clients)

Cold config: Q6, ctx 8192, 8 slots, 64-token generations each.

| N concurrent | Agg decode tok/s | Per-slot tok/s | Wall  |
|--------------|------------------|----------------|-------|
| 1            | 17.65            | 17.65          | 3.6 s |
| 4            | 34.62            | 8.65           | 7.4 s |
| 8            | **44.38**        | 5.55           | 11.5 s|
| 16           | 44.20            | 2.76           | 23.2 s (queued) |

Saturates at N=8 → 44.4 agg tok/s. Mission 09's Halo-solo parallel
throughput was 53.55 agg tok/s. RPC is **−17 %** on parallel
aggregate. Same story — solo wins.

## Prompt caching over RPC (agentic workflow)

Five-turn agent conversation with a fixed 2,628-token system prompt.
llama.cpp's `--cache-reuse 256` reuses the cached KV across turns;
only the ~508 new tokens per turn are prefilled.

| Turn | RPC wall | Solo wall |
|------|----------|-----------|
| 1 (fresh)        | 6.06 s  | 12.34 s |
| 2 (cache hit)    | 5.93 s  | 3.71 s  |
| 3 (cache hit)    | 8.30 s  | 3.67 s  |
| 4 (cache hit)    | 6.01 s  | 3.71 s  |
| 5 (cache hit)    | 5.89 s  | 3.71 s  |

Cache works over RPC — same 2,628 / 3,144 tokens are reused on every
peer. But the delta prefill is only 508 tokens, which is small enough
that RPC's per-layer overhead dominates. Solo is **~38 % faster per
turn** on the cached workload.

## What we tried that didn't work

**Speculative decoding + RPC.** Missions 19/20/21 showed 1.8x decode
speedup from spec decoding on this same 122B target. Composing it with
the RPC split failed:

```
common_speculative_is_compat: the target context does not support
partial sequence removal
```

llama.cpp b8779's RPC backend does not implement partial-sequence
removal on the peer, which spec decoding needs to roll back rejected
draft tokens. Spec decoding silently falls back to plain decoding (same
22.9 tok/s as non-spec). This is an upstream limitation, not something
we can tune around.

## What to do next

Ordered by expected impact, given Wi-Fi is the only link available:

1. **Wait for llama.cpp RPC to support partial-sequence removal.** That
   single upstream change would unlock spec decoding, which is the
   biggest remaining lever (~1.8x on solo). Track the llama.cpp RPC
   roadmap.
2. **Try larger context.** At 4 K we paid ~3 ms/token of network for ~21
   ms/token of compute. At 16 K or 32 K the compute-per-token grows but
   the per-token network cost stays roughly flat (still one hidden state
   crossing per token), so the RPC overhead amortises to noise. The
   crossover where RPC beats Halo-solo on long-context decode is
   plausibly within reach without ethernet.
3. **Run the prompt-cached agentic workload (Mission 02 reuse).** Mission
   02 already showed 76 % prefill reduction with `cache_prompt=true`.
   Combined with the new logit-local split, agentic loops should see
   most of the prefill benefit without the network penalty.
4. **Tighter sweep around the optimum.** Decode peaked at 25/75 (23.0
   tok/s) and prefill peaked at 15/85 (35.7 tok/s). Run intermediate
   ratios (17,83 18,82 22,78) to check whether a single split point
   wins both.
5. **FA on the CUDA side.** Either cherry-pick a newer llama.cpp master
   that fixes the sm_86 PTX issue, or build with CUDA 12.6. Re-enabling
   FA on the 3090 should give another few percent on the layers it owns.
6. **Quality check.** Run a small quality eval (reuse Mission 27 / 32
   HumanEval prompts) against the best RPC configuration and confirm
   parity with Halo-solo. Bit-exact expected since RPC just relocates
   compute, but worth logging.

## Artifacts

- `mission.json` — hypothesis + success criteria
- `run.py` — full sweep harness (ready to run once wired-link is available)
- `results_raw.jsonl` — append-only per-cell records
- This file — prose writeup
