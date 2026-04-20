# Consolidated findings

Grouped by theme. Individual mission writeups (with full tables, build
commands, and raw logs) live under `documentation/missions/`.

---

## KV cache compression

**Mission 01** — Sweep of `cache_type_k` × `cache_type_v` across
`{f16, q8_0, q4_0, q2_K}` at context lengths from 512 to 131 072.

- 44 configurations tested; **14 meet a ≥ 40 % memory reduction target
  with ≤ 2 % quality loss**.
- Best long-context point: `f16/f16` reached 131 K tokens at
  `pp=152.76 / tg=24.58` tok/s.
- Lower precisions trade ~5 – 12 % throughput for 30 – 70 % memory
  reduction. Pareto curve is in
  `benchmarks/missions/01-kv-cache-frontier/results.json`.

**Mission 10** — Exotic KV types (`iq4_nl`, `iq4_xs`).

- `iq4_nl` at 4K ctx: `pp=290.15 / tg=24.29` — competitive with `q4_0`
  on the same model.
- No combination produced a decode win over `f16` below 16K context.

**Mission 18** — Perplexity quality vs. KV precision.

- `f16`: PPL = 5.8134
- `q8_0`: PPL = 5.8168 (Δ < 0.1 %)
- `q4_0`: PPL = 5.833 (Δ = +0.34 %)

Takeaway: `q8_0` KV is free; `q4_0` KV costs roughly a third of a
percent of perplexity. For agent/chat workloads this is within noise.

---

## Prefix caching (agentic workloads)

**Mission 02** — `--cache-reuse` across a 20-turn coding agent at 8K,
16K, 32K total context.

- Best single TTFT improvement: `−13.2 %` at ctx 32 768 on turn 5+.
- Prompt-processing improvement on the 122 B variant: `0.2 %` at
  ctx 8 192. Cache reuse helps much less than expected once the shared
  prefix approaches the slot-local KV limit.

**Mission 34 — agentic-with-RPC** — 5-turn conversation with a 2,628-
token system prompt over the RPC split.

- Cache reuse does work over RPC (`cache_n=2628` reused on every peer).
- Per-turn wall time after cache hit: **~6.0 s RPC vs ~3.7 s Halo-solo**
  — the 508-token delta is too small to amortise RPC overhead, so solo
  wins when the model fits.

---

## Model quantization & combined recipes

**Mission 11** — Q4_K_M / Q6_K / Q8_0 at 4K ctx (prefill tok/s,
decode tok/s):

| Quant   | pp@4K    | tg@4K   |
|---------|----------|---------|
| Q4_K_M  | 341.65   | 29.48   |
| Q6_K    | 289.48   | 24.30   |
| Q8_0    | **343.10** | 19.75   |

Q8_0 has the fastest prefill (simpler dequant math) but slowest decode
(more bytes per weight). Q4_K_M is the throughput sweet spot.

**Mission 17 — combined winners.** Stack Q4_K_M + `ubatch=2048` +
parallel slots:

- **Best aggregate throughput: 60.54 tok/s** (vs Mission 09 baseline
  53.55 tok/s with Q6_K / ub=512).
- Total (prefill + decode combined): **200.69 tok/s** (vs 157.73 baseline).

**Mission 12 — batch sweep.** `b=8192 ub=2048`: `pp=385.91 tg=24.29`.

**Mission 13 — CPU tuning.** `threads=24 priority=0`: `pp=295.25 tg=24.36`.

**Mission 14 — MoE offload.** `ncmoe=0` was optimal at `pp=294.45 tg=24.35`.
Forcing experts to stay in GPU GTT never beat keeping them in system
memory via UMA — the memory is the same physical pool.

---

## Speculative decoding

**Mission 08** — Qwen3.5-122B Q6_K target + 0.8B Q4_K_M draft.

- Baseline `tg=24.3 tok/s`.
- **Best: `draft_len=5` → `tg=48.18 tok/s, accept=0.464` → 1.98× speedup.**

**Mission 19** — Q4_K_M target + 0.8B draft.

- Baseline `tg=29.48 tok/s`.
- Best: `draft_len=7` → `tg=53.50 tok/s` → 1.81× speedup.

**Mission 21** — Draft-model sweep.

- Best draft config: `gpu_draft_min0`, `tg=52.63 tok/s`, accept rate 0.56.

**Mission 20** — Parallel + speculative.

- Aggregate degrades with more clients: `npl=1 → agg_tg=41.13`, and it
  gets worse with more parallel slots because draft acceptance drops.

**Mission 34 — Spec + RPC (blocked).**

- `common_speculative_is_compat: target context does not support
  partial sequence removal`. Upstream limitation in `b8779` RPC
  backend. Cannot compose speculative decoding with RPC.

---

## Long context & retrieval

**Mission 07 — extreme context.** 65K, 131K tested. `f16/f16` maintained
coherent retrieval at 131K at tg ≈ 24.6 tok/s.

**Mission 23 — needle haystack (first pass).** 3 of 5 needles found at
various depths (16K mid ✓, 32K mid ✗, 65K mid ✗, 32K early ✓, 32K late ✓).

**Mission 28 — needle deep.** Deeper 32K/64K sweep showed the failure
mode is position-specific, not uniform over depth.

**Mission 29 — needle by quant.** Per-quant retrieval reliability across
Q4_0 → Q8_0 on Qwen3.6-35B. Results in
`benchmarks/missions/29-needle-by-quant/results.json`.

**Mission 30 — needle cliff, Mission 31 — 28K deadzone.** A reproducible
retrieval dip in Qwen3.6-35B around 24 – 32K context that sits below
its adjacent bands. Model-specific, not a platform artefact.

**Mission 34 — long-ctx RPC.** 8.8K-token prompt, ctx 16K:

| Config               | Prefill | Decode |
|----------------------|---------|--------|
| Halo solo            | 320.9   | 23.6   |
| Wi-Fi RPC, 25/75     | 276.2   | 20.6   |

Gap narrows from −42 % to −14 % on prefill as prompts grow; still no
crossover where RPC beats solo.

---

## Parallel & multi-client throughput

**Mission 09 — parallel.** Single-slot `tg=24.2`; peak at `npl=8` =
**`tg=53.55 tok/s, 2.21× speedup`**.

**Mission 26 — multi-client load.** 20 clients × `npl=8`: `agg=45.15
tok/s, TTFT p50=1.41s p99=1.84s`.

**Mission 34 — parallel + RPC.** N=1 → 17.65 agg, N=4 → 34.62, N=8 →
**44.38** (saturation), N=16 → 44.20 (queued). −17 % vs Halo-solo at
N=8. Same consistent story.

---

## Thermal & power

**Mission 24 — sustain.** 107 iterations over 60 minutes, decode tok/s
drift `−0.08 %` first-avg → last-avg. Platform is thermally stable.

**Mission 25 — tokens-per-watt.** Harness captured but power measurements
did not reach the final results.json; flagged as incomplete.

---

## Quality benchmarks

**Mission 22 — HumanEval (first 30).** Qwen3.5-122B Q4_K_M: **pass@1 =
23.3 %** (7 of 30).

**Mission 27 / 32 — HumanEval full runs.** Full benchmark data in
`benchmarks/missions/27-humaneval-full/` and `32-humaneval-v2/`.

**Mission 33 — HumanEval full v2.** Regression confirmation run.

---

## Heterogeneous RPC (Mission 34)

The headline mission of this repo. Full writeup at
`documentation/missions/34-rpc-hetero-inference.md`; results at
`benchmarks/missions/34-rpc-hetero-inference/`.

Summary of what we learned:

- **Wi-Fi is the binding constraint** with the current setup. A 2.4 GHz
  link sustains 16.4 MB/s, which adds 17 ms/token of stall to every
  decode step over RPC.
- **Device ordering matters.** `-dev RPC0,Vulkan0` keeps the LM head on
  the driver and cuts per-token network traffic from 910 KB to 55 KB
  (16× less) — because Qwen3.5's 248 K vocabulary makes logits
  enormous.
- **Decode over RPC ties Halo-solo** (23.24 vs 24.25 tok/s) for models
  that fit on the Halo. For those, solo wins.
- **RPC unlocks what solo cannot load.** MiniMax-M2.5 Q4_K_M (129 GB)
  fails to allocate on the Halo alone; splits cleanly across driver
  (109.5 GB) and worker (22.1 GB) and decodes at **23 tok/s**.
- **Speculative decoding doesn't compose** with the current RPC
  backend — upstream missing partial-sequence-removal.

Predicted ceilings (if the Wi-Fi link were replaced):

| Link              | Predicted decode tok/s | Δ vs Halo-solo |
|-------------------|-----------------------:|---------------:|
| Wi-Fi (measured)  | 17.2                   | −29 %          |
| Gigabit ethernet  | ~23.0                  | −5 %           |
| 2.5 GbE           | ~23.8                  | −2 %           |
| 10 GbE            | ~24.1                  | −0.6 %         |
| Loopback (RDMA)   | ~24.25                 | 0              |

With a faster link **and** a larger share of layers on the 3090, a
zero-latency / RDMA-style interconnect would push this setup **2–3×
Halo-solo decode** on models the Halo can run standalone.
