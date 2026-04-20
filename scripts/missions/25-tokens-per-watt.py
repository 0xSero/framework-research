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


"""Mission 25: Tokens per watt.

Runs bench configs and samples power (via amd_energy or sensors). Computes
tokens/sec / watts → efficiency metric. Compares:
  A: Q4_K_M baseline (default ub)
  B: Q4_K_M + ub=2048
  C: Q4_K_M + q4_0 KV (memory-saving)
  D: Q4_K_M + parallel 8 (via llama-batched-bench)
"""

import json, os, re, subprocess, time, threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
# (SSH_BASE defined in header)
LLAMA = "$HOME/.local/opt/llama.cpp/current"
BENCH = f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-bench"
BATCHED = f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-batched-bench"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"


def ssh(cmd, timeout=120):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr="")


def read_power():
    """Attempt to read system-wide package power. Strix Halo exposes via amdgpu/hwmon."""
    # Try sensors for "Power" field
    out = ssh("sensors 2>/dev/null | grep -iE 'power|amdgpu|PPT' | head -20", 15).stdout or ""
    watts_total = 0.0
    for line in out.splitlines():
        m = re.search(r"([\d.]+)\s*W(?:\s|$)", line)
        if m:
            watts_total += float(m.group(1))
    return watts_total


def sample_power_while(duration_s, samples):
    """Background sampler: record power every 2s."""
    t0 = time.time()
    while time.time() - t0 < duration_s:
        w = read_power()
        samples.append({"t": round(time.time() - t0, 1), "watts": w})
        time.sleep(2)


def run_bench(label, bench_cmd, duration_s=120):
    print(f"  {label} ...", flush=True)
    samples = []
    # Start power sampler in thread
    sampler = threading.Thread(target=sample_power_while, args=(duration_s, samples),
                               daemon=True)
    sampler.start()
    res = ssh(bench_cmd, duration_s + 180)
    sampler.join(timeout=5)
    out = res.stdout or ""
    pp = tg = None
    for line in out.splitlines():
        m = re.search(r"pp\d+\s*\|\s*([\d.]+)", line)
        if m: pp = float(m.group(1))
        m = re.search(r"tg\d+\s*\|\s*([\d.]+)", line)
        if m: tg = float(m.group(1))
    avg_w = (sum(s["watts"] for s in samples) / len(samples)) if samples else None
    peak_w = max((s["watts"] for s in samples), default=None)
    tg_per_watt = (tg / avg_w) if (tg and avg_w) else None
    print(f"    pp={pp} tg={tg} avg_w={avg_w} peak_w={peak_w} tg/w={tg_per_watt}",
          flush=True)
    return {"label": label, "pp": pp, "tg": tg,
            "avg_watts": avg_w, "peak_watts": peak_w,
            "tg_per_watt": tg_per_watt, "n_power_samples": len(samples)}


def main():
    configs = [
        ("A_default_ub512",
         f"{BENCH} -m {MODEL} -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 "
         f"-p 4096 -n 256 -r 1 --no-warmup"),
        ("B_ub2048",
         f"{BENCH} -m {MODEL} -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 "
         f"-p 4096 -n 256 -r 1 --no-warmup"),
        ("C_q4_0_kv",
         f"{BENCH} -m {MODEL} -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -b 2048 "
         f"-p 4096 -n 256 -r 1 --no-warmup"),
        ("D_batched_npl8",
         f"{BATCHED} -m {MODEL} -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 "
         f"-npp 512 -ntg 128 -npl 8 -c 5632"),
    ]
    results = []
    for label, cmd in configs:
        results.append(run_bench(label, cmd))

    Path(__file__).parent.joinpath("results.json").write_text(json.dumps(results, indent=2))
    best = max((r for r in results if r.get("tg_per_watt")),
               key=lambda r: r["tg_per_watt"], default=None)
    entry = {"mission": "25-tokens-per-watt", "status": "complete", "date": "2026-04-17",
             "primary_metric": best["tg_per_watt"] if best else 0,
             "primary_unit": "best_tg_per_watt",
             "conclusion": f"Best efficiency: {best['label'] if best else None} "
                           f"tg/w={best['tg_per_watt'] if best else None} "
                           f"(avg_w={best['avg_watts'] if best else None})"}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
