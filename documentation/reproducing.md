# Reproducing these results

All harnesses in `scripts/` are self-contained Python scripts that talk to
a running `llama-server` over HTTP. None of them assume a particular
address, key path, or username — they read everything from the
environment.

## Environment variables

```bash
# Driver node (runs llama-server)
export DRIVER_HOST="user@driver.example"       # ssh target
export DRIVER_KEY="$HOME/.ssh/id_ed25519"      # ssh identity
export DRIVER_BIN="/path/to/llama.cpp/build"   # where llama-server lives
export DRIVER_MODELS="/path/to/models/gguf"    # GGUF directory
export DRIVER_PORT="8080"                      # llama-server port

# RPC worker (only needed for Mission 34)
export WORKER_HOST="user@worker.example"
export WORKER_KEY="$HOME/.ssh/id_ed25519_worker"
export WORKER_BIN="/path/to/llama.cpp/build"
export WORKER_PORT="50052"
export WORKER_CUDA_VISIBLE_DEVICES="0"          # pin CUDA device if needed
```

## Starting a plain server

```bash
ssh -i "$DRIVER_KEY" "$DRIVER_HOST" "
  export LD_LIBRARY_PATH=$DRIVER_BIN
  $DRIVER_BIN/llama-server \\
    -m $DRIVER_MODELS/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf \\
    -ngl 99 -fa on -c 4096 --host 127.0.0.1 --port $DRIVER_PORT
"
```

## Starting an RPC cluster (Mission 34)

```bash
# Worker
ssh -i "$WORKER_KEY" "$WORKER_HOST" "
  export CUDA_VISIBLE_DEVICES=$WORKER_CUDA_VISIBLE_DEVICES
  export LD_LIBRARY_PATH=$WORKER_BIN
  $WORKER_BIN/rpc-server -H 0.0.0.0 -p $WORKER_PORT -c
"

# Driver
ssh -i "$DRIVER_KEY" "$DRIVER_HOST" "
  export LD_LIBRARY_PATH=$DRIVER_BIN
  $DRIVER_BIN/llama-server \\
    -m \$DRIVER_MODELS/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf \\
    --rpc $WORKER_HOST:$WORKER_PORT \\
    -dev RPC0,Vulkan0 \\
    -ngl 99 -sm layer --tensor-split 25,75 \\
    -fa off -c 4096 --host 127.0.0.1 --port $DRIVER_PORT
"
```

The ordering `-dev RPC0,Vulkan0` is important — see the Mission 34
writeup for why (it pins the LM head on the driver and cuts per-token
network traffic 16×).

## Running a mission harness

Each script under `scripts/missions/` takes no required arguments; they
read the environment and post to `http://$DRIVER_HOST:$DRIVER_PORT`
through an ssh tunnel.

```bash
python3 scripts/missions/09-parallel-throughput.py
# writes benchmarks/missions/09-parallel-throughput/results.json
```

## Reproducing the numbers exactly

You won't. Inference tok/s depends on:

- Exact `llama.cpp` commit (we used `b8775` and `b8779`)
- Vulkan / CUDA driver versions
- Kernel and scheduler settings
- Network link speed (for the RPC mission)
- Thermal headroom in your chassis

The repo documents which knobs matter most; use it as a starting point,
not a guarantee.
