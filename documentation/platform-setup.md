# Platform setup — AMD Strix Halo

End-to-end setup used for every measurement in this repo. Starting from a
fresh Fedora 43 install on a Framework Desktop.

## Hardware

| Component   | Spec                                               |
|-------------|----------------------------------------------------|
| CPU         | AMD Ryzen AI MAX+ 395 (16 C / 32 T, Zen 5)        |
| iGPU        | Radeon 8060S (gfx1151, RDNA 3.5, 40 CU)           |
| RAM         | 128 GB LPDDR5X-8000 (~215 GB/s unified)           |
| NPU         | XDNA 2 (50 TOPS)                                   |
| Storage     | 1.9 TB NVMe                                        |
| OS          | Fedora 43, kernel 6.17+                            |

## 1. System update

```bash
sudo dnf update -y
```

## 2. Kernel parameters

All three are required for large-model inference on this platform.

```bash
sudo grubby --update-kernel=ALL --args=\
  "amdgpu.gttsize=122880 amd_iommu=off ttm.pages_limit=335544321 \
   transparent_hugepage=never mitigations=off"
sudo reboot
```

| Flag                              | Why                                                  |
|-----------------------------------|------------------------------------------------------|
| `amdgpu.gttsize=122880`           | Bumps GPU-visible memory to 120 GiB (from 62.5 GiB). |
| `amd_iommu=off`                   | Fixes infinite Vulkan load on models > 64 GB.        |
| `ttm.pages_limit=335544321`       | Allows the TTM buffer manager to track all GTT pages.|
| `transparent_hugepage=never`      | Prevents THP defrag latency spikes during inference. |
| `mitigations=off`                 | Restores ~5-15 % CPU performance. Security tradeoff. |

Verify GTT after reboot:

```bash
cat /sys/class/drm/card*/device/mem_info_gtt_total
# Expect 128849018880 (120 GiB)
```

## 3. Tuned performance profile

```bash
sudo dnf install -y tuned
sudo systemctl enable --now tuned
sudo tuned-adm profile accelerator-performance

cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # -> performance
```

## 4. Disable sleep / suspend

The machine must stay awake for overnight runs:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

sudo mkdir -p /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/no-sleep.conf << 'EOF'
[Login]
IdleAction=ignore
IdleActionSec=infinity
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
HandleSuspendKey=ignore
HandleHibernateKey=ignore
EOF
sudo systemctl restart systemd-logind

gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.desktop.session idle-delay 0
```

## 5. Sysctl tuning

```bash
sudo tee /etc/sysctl.d/99-inference.conf << 'EOF'
vm.swappiness=1
vm.dirty_background_ratio=5
vm.dirty_ratio=10
EOF
sudo sysctl --system
```

## 6. llama.cpp (Vulkan)

```bash
sudo dnf install -y gcc gcc-c++ cmake vulkan-loader-devel \
  vulkan-headers glslang shaderc

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build-vulkan -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j$(nproc)
```

Default-use `RADV` ICD on Strix Halo — on our platform it decoded ~29 %
faster than AMDVLK:

```bash
export AMD_VULKAN_ICD=RADV
export LD_LIBRARY_PATH=$HOME/llama.cpp/build-vulkan/bin:$LD_LIBRARY_PATH
```

## 7. llama.cpp (ROCm)

For the ROCm + MMQ configuration we used a custom podman container
because Fedora's HIP headers conflict with ROCm 7.2.1 runtime:

```bash
# Inside the custom ROCm 7.2.1 container
export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

cmake -B build-rocm \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_RPC=ON
cmake --build build-rocm -j
```

**Do not skip `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`.** Without it ROCm sees
only BIOS VRAM and performance collapses.

## 8. Running inference

### Plain server
```bash
llama-server \
  -m path/to/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf \
  -ngl 99 --flash-attn on -c 4096 \
  --host 0.0.0.0 --port 8080
```

### `llama-bench` (matches our Phase 0 measurements)
```bash
llama-bench \
  -m path/to/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf \
  -ngl 99 -fa 1 -c 131072 \
  -p 512,2048,8192,16384,32768,65536,131072 \
  -n 128
```

### Speculative decoding
```bash
llama-server \
  -m path/to/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf \
  -md path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
  -ngl 99 -ngld 99 -fa 1 -c 4096 \
  --draft-max 8 --parallel 1 \
  --host 0.0.0.0 --port 8080
```

For Qwen3.5 and other hybrid SSM/MoE targets, you will need PR #20075
plus the four extra fixes documented in
[`phase-0-strix-benchmarks.md`](phase-0-strix-benchmarks.md).

## Optimisation checklist

- [x] GTT increased to 120 GiB
- [x] Vulkan RADV backend (or ROCm 7.2.1 with UMA env var)
- [x] Flash Attention enabled
- [x] CPU `performance` governor (`accelerator-performance` tuned profile)
- [x] Sleep / suspend disabled
- [x] Transparent hugepages disabled
- [x] CPU mitigations disabled
- [x] IOMMU disabled
- [x] Swappiness minimised
- [ ] ROCm MMQ patch (PR #21344) — +20 – 35 % prefill
- [ ] 2-node RPC cluster (see Mission 34)
- [ ] Speculative decoding (PR #20075 + four fixes)
