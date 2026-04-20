# llama.cpp build notes

## Driver (AMD Radeon 8060S, Vulkan)

Standard Vulkan build. GCC 11.4.0 on a modern Linux distribution. Tested
on kernel 6.17.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_RPC=ON
cmake --build build -j
```

### Kernel parameters for the Strix Halo

Two parameters unlock the full 120 GiB GTT pool for GPU use:

```
amdgpu.gttsize=122880
amd_iommu=off
```

Reboot after adding them. Verify with:

```bash
cat /sys/class/drm/card*/device/mem_info_gtt_total
```

Expected value: `128849018880` (≈ 120 GiB).

## Worker (NVIDIA RTX 3090, CUDA)

CUDA 12.8 + GCC 11. Two workarounds were needed to compile `b8779`
cleanly:

1. **Remove `-compress-mode=size`** — CUDA 12.8 does not understand this
   flag when invoked by the GCC 11 driver. Edit
   `ggml/src/ggml-cuda/CMakeLists.txt` (around line 209) and drop the
   `list(APPEND CUDA_FLAGS -compress-mode=size)` call.

2. **Build with `-DGGML_CUDA_FA=OFF`** — the Flash Attention kernel emits
   `movmatrix` instructions that are PTX ISA 7.8+, which the SM 8.6
   backend in CUDA 12.8 rejects with a `ptxas` error. Disabling FA on
   the worker costs ~5 % on the layers it owns but lets the build
   succeed.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=OFF \
  -DGGML_RPC=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

### Isolating the device

If the worker box has other CUDA devices you do not want to touch,
pin the RPC server with `CUDA_VISIBLE_DEVICES=<idx>` before launching.
For the 3090 in our companion box we used `CUDA_VISIBLE_DEVICES=0`.

## Verifying an RPC connection

From the driver:

```bash
ncat -zv $WORKER_HOST $WORKER_PORT
```

Should return immediately. Once the driver's `llama-server` starts with
`--rpc`, the worker log will show `accepted client connection` and
begin receiving weight chunks.

The first-time weight transfer over a 1 Gbps link takes about 10 s per
GB. Over Wi-Fi (100 Mbps sustained) it is ~80 s per GB, so an 80 GB model
takes ~6 minutes. Subsequent loads from page cache are much faster.
