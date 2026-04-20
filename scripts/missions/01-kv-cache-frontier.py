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


"""Mission 01: KV Cache Compression Frontier — speed sweep harness."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
MISSION_FILE = Path(__file__).parent / "mission.json"

# (SSH_BASE defined in header)

LLAMA_BENCH = (
    "env LD_LIBRARY_PATH=$HOME/.local/opt/llama.cpp/current "
    "$HOME/.local/opt/llama.cpp/current/llama-bench"
)

MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf"


def ssh_cmd(cmd: str, timeout: int = 600):
    # Merge stderr into stdout so we capture everything including errors
    try:
        return subprocess.run(
            SSH_BASE + [cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # Kill any lingering llama-bench on the remote so GPU memory is freed
        kill_cmd = SSH_BASE + ["killall -9 llama-bench 2>/dev/null; echo killed"]
        subprocess.run(kill_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"    TIMEOUT after {timeout}s — marking as failed", flush=True)
        # Return a synthetic CompletedProcess so callers stay uniform
        return subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=-1,
            stdout=exc.stdout or "",
            stderr="",
        )


def parse_bench_output(text: str, ctx: int):
    """Extract pp{ctx} and tg128 from llama-bench markdown table output."""
    pp = None
    tg = None
    for line in text.splitlines():
        m = re.search(rf"pp{ctx}\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return pp, tg


def run_combo(ck, cv, ctx):
    cmd = (
        f"timeout 600s bash -c '{LLAMA_BENCH} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk {ck} -ctv {cv} "
        f"-p {ctx} -n 128 -r 1 --no-warmup'"
    )
    print(f"  Running ctk={ck} ctv={cv} ctx={ctx} ...", flush=True)
    res = ssh_cmd(cmd)
    combined = res.stdout
    if res.returncode == -1:
        return {"pp": None, "tg": None, "stdout": combined, "timeout": True}
    if "failed to load model" in combined or "ErrorOutOfDeviceMemory" in combined:
        print(f"    WARNING: model failed to load for {ck}/{cv}@{ctx} — GPU memory busy?")
        print(combined[:800])
        return {"pp": None, "tg": None, "stdout": combined}
    pp, tg = parse_bench_output(combined, ctx)
    if pp is None or tg is None:
        print(f"    WARNING: failed to parse output for {ck}/{cv}@{ctx}")
        print(combined[:800])
    return {"pp": pp, "tg": tg, "stdout": combined}


def estimate_kv_memory_gb(ctx, ck, cv):
    """Rough estimate of KV cache memory for REAP-20 Q6_K."""
    # REAP-20 has 61 layers, hidden_size ~5120, num_key_value_heads ~8
    # head_dim = 5120 // 40 = 128, but let's use ~128
    layers = 61
    hidden = 5120
    head_dim = 128
    kv_heads = 8
    bytes_per_element = {
        "f16": 2, "q8_0": 1, "q4_0": 0.5, "q2_K": 0.25
    }.get(ck, 2)
    bytes_per_element_v = {
        "f16": 2, "q8_0": 1, "q4_0": 0.5, "q2_K": 0.25
    }.get(cv, 2)
    # Per token: layers * kv_heads * head_dim * 2 (K+V) * bytes
    # Actually separate K and V
    k_per_token = layers * kv_heads * head_dim * bytes_per_element
    v_per_token = layers * kv_heads * head_dim * bytes_per_element_v
    total_gb = (k_per_token + v_per_token) * ctx / 1e9
    return total_gb


def main():
    with open(MISSION_FILE) as f:
        mission = json.load(f)

    combos = mission["combos"]
    contexts = mission["contexts"]
    results = []

    print("=" * 60)
    print("Mission 01: KV Cache Compression Frontier")
    print("=" * 60)

    for ck, cv in combos:
        print(f"\nCombo: cache_type_k={ck}, cache_type_v={cv}")
        for ctx in contexts:
            try:
                bench = run_combo(ck, cv, ctx)
                mem_gb = estimate_kv_memory_gb(ctx, ck, cv)
                results.append({
                    "ck": ck,
                    "cv": cv,
                    "ctx": ctx,
                    "pp": bench["pp"],
                    "tg": bench["tg"],
                    "kv_mem_gb": round(mem_gb, 2),
                })
                if bench["pp"] is not None:
                    print(f"    ctx={ctx:>6} pp={bench['pp']:.1f} tg={bench['tg']:.2f} mem≈{mem_gb:.1f}GB")
            except Exception as e:
                print(f"    ERROR at {ck}/{cv}@{ctx}: {e}")

    # Save raw results
    raw_path = Path(__file__).parent / "results.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")

    # Compute memory reduction vs f16/f16 baseline
    baseline_mem = {}
    for r in results:
        if r["ck"] == "f16" and r["cv"] == "f16":
            baseline_mem[r["ctx"]] = r["kv_mem_gb"]

    for r in results:
        if r["ctx"] in baseline_mem and baseline_mem[r["ctx"]]:
            r["mem_reduction_pct"] = round(
                (1 - r["kv_mem_gb"] / baseline_mem[r["ctx"]]) * 100, 1
            )

    # Find sweet spots
    sweet_spots = [
        r for r in results
        if r.get("mem_reduction_pct", 0) >= mission["success_criteria"]["memory_reduction_pct"]
        and r["pp"] is not None
    ]

    print("\n--- Summary ---")
    print(f"Total configurations tested: {len(results)}")
    print(f"Sweet spots (≥{mission['success_criteria']['memory_reduction_pct']}% memory reduction): {len(sweet_spots)}")
    for s in sweet_spots[:5]:
        print(f"  {s['ck']}/{s['cv']} @ {s['ctx']}: pp={s['pp']:.1f} tg={s['tg']:.2f} mem_red={s.get('mem_reduction_pct')}%")

    # Log mission result
    entry = {
        "mission": "01-kv-cache-frontier",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": len(sweet_spots),
        "primary_unit": "sweet_spot_configs",
        "conclusion": f"Tested {len(results)} configs; {len(sweet_spots)} meet memory target. See results.json for full Pareto data.",
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nMission result logged to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
