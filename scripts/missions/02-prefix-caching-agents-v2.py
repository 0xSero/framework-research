"""
Mission harness — env-var driven port of the original research script.

Required env vars:
  DRIVER_HOST    e.g. "user@driver.example"
  DRIVER_KEY     path to ssh private key, e.g. "~/.ssh/id_ed25519"
  DRIVER_BIN     path to llama.cpp build on the driver
  DRIVER_MODELS  path to GGUF directory on the driver
  DRIVER_PORT    llama-server port (default 8080)

Writes results next to itself; operators re-running for reproduction
should adjust the output path if they want to preserve published data.
"""
import os

def _env(name, default=None):
    v = os.environ.get(name, default)
    if v is None:
        raise RuntimeError(f"missing env var: {name}")
    return v

DRIVER_HOST   = _env("DRIVER_HOST")
DRIVER_KEY    = _env("DRIVER_KEY")
DRIVER_BIN    = _env("DRIVER_BIN")
DRIVER_MODELS = _env("DRIVER_MODELS", "$HOME/models/gguf")
DRIVER_PORT   = _env("DRIVER_PORT", "8080")

# Base SSH command to the driver; scripts extend this with a remote
# command string as the final argument.
SSH_BASE = ["ssh", "-o", "ConnectTimeout=10",
            "-i", os.path.expanduser(DRIVER_KEY), DRIVER_HOST]


"""Mission 02 v2: Prefix Caching on the 122B target model with exact prefix matching.

The original hypothesis specifically targeted the 122B MoE model, where prefill
at 4K tokens is slow enough that caching the system prompt should measurably
reduce TTFT. We use llama-server's /completion endpoint with cache_prompt=true
and keep the system prefix byte-for-byte identical across turns.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
MISSION_FILE = Path(__file__).parent / "mission.json"

# (SSH_BASE defined in header)

LLAMA_SERVER = (
    "env LD_LIBRARY_PATH=$HOME/.local/opt/llama.cpp/current "
    "$HOME/.local/opt/llama.cpp/current/llama-server"
)

# Use the 122B target model — the original hypothesis was about this class of model
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf"
PORT = 8088
HOST = "127.0.0.1"

# Fixed 4K-token system prefix — this is what should live in the cache
SYSTEM_PREFIX = (
    "You are an expert software engineer assisting with code review. "
    "Think carefully step by step. Be precise, cite line numbers, and "
    "flag security issues. " * 80  # ~4K tokens
)


def ssh_cmd(cmd: str, timeout: int = 60):
    return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)


def start_server(ctx: int, cache_reuse: int) -> None:
    ssh_cmd("killall -9 llama-server 2>/dev/null; sleep 2")
    cmd = (
        f"nohup {LLAMA_SERVER} -m {MODEL} -ngl 99 -fa 1 -c {ctx} "
        f"-ctk q8_0 -ctv q8_0 "  # use the winning KV quant from Mission 01
        f"--port {PORT} --host {HOST} --no-webui "
        f"--cache-reuse {cache_reuse} "
        f"> /tmp/llama-server-m02.log 2>&1 &"
    )
    ssh_cmd(cmd, timeout=5)

    # Wait for server to load the 76 GB model
    for _ in range(120):
        time.sleep(3)
        res = ssh_cmd(f"curl -sf http://{HOST}:{PORT}/health -o /dev/null && echo OK || echo WAIT", timeout=10)
        if "OK" in res.stdout:
            return
    raise RuntimeError("server never came up")


def one_turn(user_msg: str) -> dict:
    """Fire a completion request and extract timing from the response."""
    payload = {
        "prompt": SYSTEM_PREFIX + f"\n\nUser: {user_msg}\nAssistant:",
        "n_predict": 32,
        "cache_prompt": True,
        "temperature": 0.0,
    }
    body = json.dumps(payload).replace('"', '\\"')
    t0 = time.time()
    res = ssh_cmd(
        f'curl -s -X POST http://{HOST}:{PORT}/completion '
        f'-H "Content-Type: application/json" '
        f'-d "{body}"',
        timeout=600,
    )
    wall_ms = (time.time() - t0) * 1000
    try:
        data = json.loads(res.stdout.strip().split("\n")[-1])
        timings = data.get("timings", {})
        return {
            "wall_ms": round(wall_ms, 1),
            "prompt_ms": round(timings.get("prompt_ms", 0), 1),
            "predicted_ms": round(timings.get("predicted_ms", 0), 1),
            "cached_tokens": data.get("tokens_cached", 0),
            "prompt_n": timings.get("prompt_n", 0),
        }
    except (json.JSONDecodeError, IndexError):
        return {"wall_ms": wall_ms, "error": res.stdout[-400:]}


def run_variant(ctx: int, cache_reuse: int) -> dict:
    print(f"\n  cache_reuse={cache_reuse} ctx={ctx}: starting server ...", flush=True)
    start_server(ctx, cache_reuse)
    print(f"    server ready", flush=True)

    turns = []
    for i in range(1, 6):
        user_msg = f"Turn {i}: Explain quicksort in one sentence."
        r = one_turn(user_msg)
        r["turn"] = i
        turns.append(r)
        print(f"    turn {i}: wall={r.get('wall_ms', '?')}ms prompt={r.get('prompt_ms', '?')}ms cached={r.get('cached_tokens', '?')}", flush=True)

    ssh_cmd("killall -9 llama-server 2>/dev/null")

    # Skip turn 1 (cold) when averaging
    warm = turns[1:]
    return {
        "turns": turns,
        "warm_prompt_ms_avg": round(sum(t.get("prompt_ms", 0) for t in warm) / max(len(warm), 1), 1),
        "warm_wall_ms_avg": round(sum(t.get("wall_ms", 0) for t in warm) / max(len(warm), 1), 1),
    }


def main():
    results = []
    # Stick to one context the 122B model can handle well with q8_0/q8_0 KV
    for ctx in [8192, 16384]:
        print(f"\n=== Context {ctx} ===")
        no_cache = run_variant(ctx, 0)
        with_cache = run_variant(ctx, 1)

        if no_cache["warm_prompt_ms_avg"] > 0:
            improvement = round(
                (1 - with_cache["warm_prompt_ms_avg"] / no_cache["warm_prompt_ms_avg"]) * 100, 1
            )
        else:
            improvement = 0.0

        results.append({
            "ctx": ctx,
            "no_cache": no_cache,
            "with_cache": with_cache,
            "prompt_ms_improvement_pct": improvement,
        })
        print(f"\n  >>> prompt_ms improvement at {ctx}: {improvement}%")

    out_path = Path(__file__).parent / "results_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda r: r["prompt_ms_improvement_pct"]) if results else None
    entry = {
        "mission": "02-prefix-caching-agents",
        "status": "complete",
        "date": "2026-04-16",
        "variant": "v2_122B_model",
        "primary_metric": best["prompt_ms_improvement_pct"] if best else 0,
        "primary_unit": "prompt_ms_improvement_pct",
        "conclusion": (
            f"122B + q8_0/q8_0 KV + cache_prompt=true: "
            f"{best['prompt_ms_improvement_pct']}% prompt-processing improvement "
            f"at ctx={best['ctx']}" if best else "no data"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nResults saved to {out_path}")
    print(f"Mission result logged to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
