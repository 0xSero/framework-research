# 05-npu-gpu-hybrid

## Hypothesis

The XDNA 2 NPU can run the 0.8B draft model fast enough to offload it from the GPU, freeing GPU cycles for batched verification of the 122B target.

## Success criteria

- {'pipeline_runs_without_crash': True, 'hybrid_throughput_exceeds_gpu_only': False}

## Result

- **2026-04-16** — 0  readiness_score — XDNA present but ONNX Runtime missing. Run setup_npu.sh to install dependencies.
- **2026-04-16** — 1  readiness_score — XDNA and ONNX Runtime available. Full pipeline harness needed in follow-up.

## Raw data

- Mission spec: `benchmarks/missions/05-npu-gpu-hybrid/mission.json`
- Harness: `scripts/missions/05-npu-gpu-hybrid.py`
