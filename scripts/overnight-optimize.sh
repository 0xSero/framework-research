#!/bin/bash
# overnight-optimize.sh — Full overnight optimization pipeline
# Runs on the Framework Desktop (Fedora 43, Strix Halo)
# Usage: nohup bash overnight-optimize.sh > /tmp/overnight.log 2>&1 &
set -uo pipefail

LLAMA=/home/liveuser/workspace/llama.cpp/build-vulkan
export LD_LIBRARY_PATH=$LLAMA/bin:$LLAMA/lib
MODELS=/home/liveuser/models
RESULTS=$MODELS/bench-results
# Set HF_TOKEN env var before running
TOKEN=${HF_TOKEN:-}
export HF_TOKEN=$TOKEN
export AMD_VULKAN_ICD=RADV

mkdir -p $RESULTS

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ──────────────────────────────────────────────────────────────
# PHASE 0: Wait for current benchmark to finish
# ──────────────────────────────────────────────────────────────
log "PHASE 0: Waiting for downloads and current benchmark to finish..."
# Wait for HF downloads to complete
while pgrep -f "hf download" > /dev/null 2>&1; do
    DL30=$(du -sh /home/liveuser/models/reap30-hf/ 2>/dev/null | awk '{print $1}')
    DL40=$(du -sh /home/liveuser/models/reap40-hf/ 2>/dev/null | awk '{print $1}')
    log "  HF downloads in progress: REAP-30=$DL30 REAP-40=$DL40"
    sleep 120
done
log "  Downloads complete."
# Wait for benchmark
while pgrep -f "full-bench.py" > /dev/null 2>&1; do
    log "  Waiting for full-bench.py to finish..."
    sleep 60
done
log "  Benchmark finished."

# Kill any lingering server
pkill -f llama-server || true
sleep 3

# ──────────────────────────────────────────────────────────────
# PHASE 1: Update llama.cpp to latest
# ──────────────────────────────────────────────────────────────
log "PHASE 1: Updating llama.cpp..."
# Ensure build deps
python3 -m pip install --user transformers --break-system-packages 2>/dev/null || true
cd /home/liveuser/workspace/llama.cpp
OLD_REV=$(git rev-parse --short HEAD)
git pull --ff-only 2>&1 | tail -3
NEW_REV=$(git rev-parse --short HEAD)

if [ "$OLD_REV" != "$NEW_REV" ]; then
    log "Updated llama.cpp: $OLD_REV -> $NEW_REV, rebuilding..."
    cd build-vulkan
    /usr/bin/cmake .. -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    /usr/bin/cmake --build . --config Release -j$(nproc) 2>&1 | tail -5
    # Re-export path in case binary names changed
    LLAMA=/home/liveuser/workspace/llama.cpp/build-vulkan
    export LD_LIBRARY_PATH=$LLAMA/bin:$LLAMA/lib:$LD_LIBRARY_PATH
    log "llama.cpp rebuilt: $($LLAMA/bin/llama-server --version 2>&1 | head -1)"
else
    log "llama.cpp already at latest: $OLD_REV"
fi

# ──────────────────────────────────────────────────────────────
# PHASE 2: Convert REAP-30 to GGUF and quantize
# ──────────────────────────────────────────────────────────────
log "PHASE 2: Converting REAP-30..."
if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" ]; then
    SIZE=$(stat -c%s "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000000000 ]; then
        log "  REAP-30 Q4_K_M already exists ($(numfmt --to=iec $SIZE)), skipping."
    else
        log "  REAP-30 Q4_K_M exists but is too small ($SIZE bytes), re-converting."
        rm -f "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf"
    fi
fi
if [ ! -f "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" ]; then
    ST_FILES=$(ls $MODELS/reap30-hf/model-*.safetensors 2>/dev/null | wc -l)
    if [ "$ST_FILES" -lt 3 ]; then
        log "  ERROR: REAP-30 safetensors incomplete ($ST_FILES files). Skipping."
    else
        log "  Converting REAP-30 safetensors -> GGUF BF16..."
        python3 /home/liveuser/workspace/llama.cpp/convert_hf_to_gguf.py \
            $MODELS/reap30-hf --outtype f16 \
            --outfile $MODELS/reap30-bf16.gguf 2>&1 | tail -5
        if [ ! -f "$MODELS/reap30-bf16.gguf" ] || [ "$(stat -c%s $MODELS/reap30-bf16.gguf 2>/dev/null || echo 0)" -lt 1000000000 ]; then
            log "  ERROR: BF16 conversion failed"
            rm -f $MODELS/reap30-bf16.gguf
        else
            log "  Quantizing REAP-30 BF16 -> Q4_K_M..."
            $LLAMA/bin/llama-quantize $MODELS/reap30-bf16.gguf \
                $MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf Q4_K_M 2>&1 | tail -5
            rm -f $MODELS/reap30-bf16.gguf
            ls -lh $MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf 2>/dev/null
        fi
    fi
fi

# ──────────────────────────────────────────────────────────────
# PHASE 3: Convert REAP-40 to GGUF and quantize
# ──────────────────────────────────────────────────────────────
log "PHASE 3: Converting REAP-40..."
if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" ]; then
    SIZE=$(stat -c%s "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1000000000 ]; then
        log "  REAP-40 Q4_K_M already exists ($(numfmt --to=iec $SIZE)), skipping."
    else
        log "  REAP-40 Q4_K_M too small, re-converting."
        rm -f "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf"
    fi
fi
if [ ! -f "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" ]; then
    ST_FILES=$(ls $MODELS/reap40-hf/model-*.safetensors 2>/dev/null | wc -l)
    if [ "$ST_FILES" -lt 3 ]; then
        log "  ERROR: REAP-40 safetensors incomplete ($ST_FILES files). Skipping."
    else
        log "  Converting REAP-40 safetensors -> GGUF BF16..."
        python3 /home/liveuser/workspace/llama.cpp/convert_hf_to_gguf.py \
            $MODELS/reap40-hf --outtype f16 \
            --outfile $MODELS/reap40-bf16.gguf 2>&1 | tail -5
        if [ ! -f "$MODELS/reap40-bf16.gguf" ] || [ "$(stat -c%s $MODELS/reap40-bf16.gguf 2>/dev/null || echo 0)" -lt 1000000000 ]; then
            log "  ERROR: BF16 conversion failed"
            rm -f $MODELS/reap40-bf16.gguf
        else
            log "  Quantizing REAP-40 BF16 -> Q4_K_M..."
            $LLAMA/bin/llama-quantize $MODELS/reap40-bf16.gguf \
                $MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf Q4_K_M 2>&1 | tail -5
            rm -f $MODELS/reap40-bf16.gguf
            ls -lh $MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf 2>/dev/null
        fi
    fi
fi

# ──────────────────────────────────────────────────────────────
# PHASE 4: Clean up safetensors to free disk
# ──────────────────────────────────────────────────────────────
log "PHASE 4: Freeing disk space..."
# Only delete safetensors if conversion succeeded
if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" ]; then
    log "  REAP-30 Q4_K_M exists, removing safetensors..."
    rm -rf $MODELS/reap30-hf
fi
if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" ]; then
    log "  REAP-40 Q4_K_M exists, removing safetensors..."
    rm -rf $MODELS/reap40-hf
fi
# Always clean up BF16 intermediates
rm -f $MODELS/reap30-bf16.gguf $MODELS/reap40-bf16.gguf
log "  Disk:"
df -h / | tail -1

# ──────────────────────────────────────────────────────────────
# PHASE 5: Speed benchmarks — all models, multiple context sizes
# ──────────────────────────────────────────────────────────────
log "PHASE 5: Speed benchmarks..."

speed_bench() {
    local model=$1
    local label=$2
    local ctx=$3
    local ngl=${4:-999}
    local outfile="$RESULTS/speed-${label}-ctx${ctx}.txt"
    if [ -f "$outfile" ]; then
        log "  SKIP $label ctx=$ctx (already done)"
        return
    fi
    log "  BENCH $label ctx=$ctx ngl=$ngl"
    $LLAMA/bin/llama-bench \
        -m "$model" -ngl $ngl -fa 1 \
        -p $ctx -n 128 \
        2>&1 | tee "$outfile" | grep -E "pp${ctx}|tg128|model"
}

# Small models — fast, test full range including 128K
for ctx in 512 1024 2048 4096 8192 16384 32768 65536 131072; do
    speed_bench "$MODELS/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf" "coder30b-q4xl" $ctx
    speed_bench "$MODELS/Qwen3.5-35B-A3B-Q4_K_M.gguf" "qwen35-35b-q4km" $ctx
done

# Medium models
for ctx in 512 1024 2048 4096 8192 16384 32768 65536 131072; do
    speed_bench "$MODELS/Qwen3-Coder-Next-Q4_K_M.gguf" "coder-next-q4km" $ctx
done

# Large models — REAP-20 at all quants (128K may not fit for Q8_0)
for ctx in 512 1024 2048 4096 8192 16384 32768 65536 131072; do
    speed_bench "$MODELS/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf" "reap20-q4km" $ctx
    speed_bench "$MODELS/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf" "reap20-q6k" $ctx
done
# Q8_0 — only up to 32K, 128K won't fit in 120 GiB GTT
for ctx in 512 1024 2048 4096 8192 16384 32768; do
    speed_bench "$MODELS/Qwen3.5-122B-A10B-REAP-20-Q8_0.gguf" "reap20-q80" $ctx
done

# REAP-30 and REAP-40
for ctx in 512 1024 2048 4096 8192 16384 32768 65536 131072; do
    if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" ]; then
        speed_bench "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" "reap30-q4km" $ctx
    fi
    if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" ]; then
        speed_bench "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" "reap40-q4km" $ctx
    fi
done

# ──────────────────────────────────────────────────────────────
# PHASE 6: Context scaling tests (how speed degrades with ctx)
# ──────────────────────────────────────────────────────────────
log "PHASE 6: Context scaling tests..."
for model_label in reap20-q4km reap30-q4km reap40-q4km; do
    case $model_label in
        reap20-q4km) model="$MODELS/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf" ;;
        reap30-q4km) model="$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" ;;
        reap40-q4km) model="$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" ;;
    esac
    if [ ! -f "$model" ]; then continue; fi
    for ctx in 128 512 1024 2048 4096 8192 16384 32768 65536 131072; do
        speed_bench "$model" "$model_label" $ctx
    done
done

# ──────────────────────────────────────────────────────────────
# PHASE 7: Quality benchmarks on REAP-30 and REAP-40
# ──────────────────────────────────────────────────────────────
log "PHASE 7: Quality benchmarks..."

run_quality_bench() {
    local model=$1
    local label=$2

    # Start server
    pkill -f llama-server || true
    sleep 3
    nohup $LLAMA/bin/llama-server \
        -m "$model" -ngl 999 --flash-attn on -c 131072 \
        --port 8080 --host 0.0.0.0 \
        > /tmp/llama-server-$label.log 2>&1 &
    local server_pid=$!

    # Wait for ready
    for i in $(seq 1 60); do
        if curl -s http://localhost:8080/health 2>/dev/null | grep -q "ok"; then
            break
        fi
        sleep 5
    done

    if ! curl -s http://localhost:8080/health 2>/dev/null | grep -q "ok"; then
        log "  FAILED to start server for $label"
        kill $server_pid 2>/dev/null || true
        return
    fi

    log "  Running quality benchmarks for $label..."
    cd $MODELS
    python3 -u $MODELS/full-bench.py "$label" 2>&1 | tee "$RESULTS/quality-${label}.log"

    # Kill server
    kill $server_pid 2>/dev/null || true
    sleep 3
}

# REAP-30 quality
if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" ]; then
    run_quality_bench "$MODELS/Qwen3.5-122B-A10B-REAP-30-Q4_K_M.gguf" "reap30-q4km"
fi

# REAP-40 quality
if [ -f "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" ]; then
    run_quality_bench "$MODELS/Qwen3.5-122B-A10B-REAP-40-Q4_K_M.gguf" "reap40-q4km"
fi

# REAP-20 Q6_K quality
run_quality_bench "$MODELS/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf" "reap20-q6k"

# ──────────────────────────────────────────────────────────────
# PHASE 8: Batch size / parallelism tests
# ──────────────────────────────────────────────────────────────
log "PHASE 8: Parallelism tests on fastest model..."
pkill -f llama-server || true
sleep 3

for parallel in 1 2 4; do
    log "  Testing parallel=$parallel on Coder-30B..."
    nohup $LLAMA/bin/llama-server \
        -m $MODELS/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
        -ngl 999 --flash-attn on -c 131072 \
        --parallel $parallel \
        --port 8080 --host 0.0.0.0 \
        > /tmp/llama-server-parallel.log 2>&1 &

    for i in $(seq 1 30); do
        if curl -s http://localhost:8080/health 2>/dev/null | grep -q "ok"; then break; fi
        sleep 3
    done

    # Fire parallel requests and measure throughput
    python3 -c "
import urllib.request, json, time, threading

def make_request(idx):
    data = json.dumps({
        'model': 'default',
        'messages': [{'role': 'user', 'content': 'Write a Python fibonacci function. Return only the code.'}],
        'max_tokens': 256,
        'temperature': 0
    }).encode()
    req = urllib.request.Request('http://localhost:8080/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=300)
    elapsed = time.perf_counter() - t0
    r = json.loads(resp.read())
    tokens = r.get('usage', {}).get('completion_tokens', 0)
    tps = tokens / elapsed if elapsed > 0 else 0
    print(f'  Request {idx}: {tokens} tokens in {elapsed:.1f}s = {tps:.1f} t/s')

t0 = time.perf_counter()
threads = []
for i in range($parallel):
    t = threading.Thread(target=make_request, args=(i,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
total_time = time.perf_counter() - t0
print(f'  PARALLEL=$parallel total_time={total_time:.1f}s')
" 2>&1 | tee "$RESULTS/parallel-$parallel.txt"

    pkill -f llama-server || true
    sleep 3
done

# ──────────────────────────────────────────────────────────────
# PHASE 9: Final summary
# ──────────────────────────────────────────────────────────────
log "PHASE 9: Generating summary..."

echo "" > $RESULTS/summary.txt
echo "=== FRAMEWORK DESKTOP OVERNIGHT OPTIMIZATION RESULTS ===" >> $RESULTS/summary.txt
echo "Date: $(date)" >> $RESULTS/summary.txt
echo "llama.cpp: $($LLAMA/bin/llama-server --version 2>&1 | head -1)" >> $RESULTS/summary.txt
echo "Kernel: $(uname -r)" >> $RESULTS/summary.txt
echo "" >> $RESULTS/summary.txt

echo "=== SPEED BENCHMARKS ===" >> $RESULTS/summary.txt
for f in $RESULTS/speed-*.txt; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .txt)
    echo "--- $label ---" >> $RESULTS/summary.txt
    grep -E "pp[0-9]|tg128" "$f" >> $RESULTS/summary.txt 2>/dev/null || echo "  no results" >> $RESULTS/summary.txt
done

echo "" >> $RESULTS/summary.txt
echo "=== QUALITY BENCHMARKS ===" >> $RESULTS/summary.txt
for f in $RESULTS/quality-*.log; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .log)
    echo "--- $label ---" >> $RESULTS/summary.txt
    grep "SUMMARY" "$f" >> $RESULTS/summary.txt 2>/dev/null || echo "  no summary" >> $RESULTS/summary.txt
done

echo "" >> $RESULTS/summary.txt
echo "=== MODELS ===" >> $RESULTS/summary.txt
ls -lhS $MODELS/*.gguf >> $RESULTS/summary.txt
echo "" >> $RESULTS/summary.txt
echo "=== DISK ===" >> $RESULTS/summary.txt
df -h / >> $RESULTS/summary.txt

log "DONE. Results in $RESULTS/"
log "Summary:"
cat $RESULTS/summary.txt
