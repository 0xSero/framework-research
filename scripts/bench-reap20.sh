#!/bin/bash
# REAP-20 Benchmark Script: Q4_K_M, Q6_K, Q8_0
# Runs on Framework Desktop via SSH

set -euo pipefail

MODEL_DIR="/mnt/nvme/home/liveuser/models"
LLAMA_DIR="/mnt/nvme/home/liveuser/workspace/llama.cpp/build-vulkan"
RESULTS_DIR="/mnt/nvme/home/liveuser/models/eval-results"
export AMD_VULKAN_ICD=RADV
export HF_TOKEN="<hf-token>"

mkdir -p "$RESULTS_DIR"

QUANTS=("Q4_K_M" "Q6_K" "Q8_0")
SIZES=("57" "76" "99")  # GB
NGL=("999" "35" "25")    # layers that fit in 64GB GTT

echo "============================================"
echo "REAP-20 Benchmark Suite"
echo "============================================"

for i in "${!QUANTS[@]}"; do
    Q="${QUANTS[$i]}"
    G="${NGL[$i]}"
    GGUF="$MODEL_DIR/Qwen3.5-122B-A10B-REAP-20-${Q}.gguf"
    
    echo ""
    echo "============================================"
    echo "=== $Q (ngl=$G) ==="
    echo "============================================"
    
    # Speed benchmarks
    echo "--- Vulkan RADV speed ---"
    "$LLAMA_DIR/bin/llama-bench" -m "$GGUF" -ngl "$G" --flash-attn on -mmp 0 -r 3 2>&1 || \
    "$LLAMA_DIR/bin/llama-bench" -m "$GGUF" -ngl "$G" --flash-attn on -r 3 2>&1
    
    echo ""
    
    # Start server for eval
    echo "--- Starting server for $Q ---"
    kill $(pgrep -f llama-server) 2>/dev/null || true
    sleep 2
    
    "$LLAMA_DIR/bin/llama-server" \
        -m "$GGUF" -ngl "$G" --flash-attn on -c 4096 \
        --port 8080 --host 0.0.0.0 &
    SERVER_PID=$!
    
    # Wait for server
    for j in $(seq 1 120); do
        if curl -s http://localhost:8080/health | grep -q ok; then
            echo "Server ready ($j s)"
            break
        fi
        sleep 1
    done
    
    # EvalPlus smoke test (2 samples)
    echo "--- EvalPlus HumanEval smoke ($Q) ---"
    evalplus.evaluate \
        --model "reap20-$Q" \
        --backend openai \
        --base-url http://localhost:8080/v1 \
        --dataset humaneval \
        --greedy \
        --n-samples 2 2>&1 | tail -10 || echo "EvalPlus failed, skipping"
    
    # Kill server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    
    echo "--- $Q complete ---"
done

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "============================================"
