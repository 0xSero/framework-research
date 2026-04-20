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


"""Mission 13: CPU thread/prio tuning for Qwen3.5-122B on Vulkan/UMA."""

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"

# (SSH_BASE defined in header)

LLAMA = "$HOME/.local/opt/llama.cpp/current"
BENCH = f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-bench"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf"

# Use compact sweep — (threads, prio) pairs
CONFIGS = [
    (8,  0),
    (16, 0),
    (24, 0),
    (32, 0),
    (16, 2),
    (16, 3),
    (24, 3),
]
CTX = 8192


def ssh(cmd: str, timeout: int = 500):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-bench 2>/dev/null"])
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run(threads: int, prio: int):
    cmd = (
        f"timeout 420s bash -c '{BENCH} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk q8_0 -ctv q8_0 -t {threads} --prio {prio} "
        f"-p {CTX} -n 128 -r 1 --no-warmup'"
    )
    print(f"  threads={threads} prio={prio} ...", flush=True)
    res = ssh(cmd, 480)
    out = res.stdout or ""
    pp = tg = None
    for line in out.splitlines():
        m = re.search(rf"pp{CTX}\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return {"threads": threads, "prio": prio, "pp": pp, "tg": tg}


def main():
    results = []
    for t, p in CONFIGS:
        r = run(t, p)
        results.append(r)
        print(f"    pp={r['pp']} tg={r['tg']}", flush=True)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    best = max((r for r in results if r["pp"]), key=lambda r: r["pp"], default=None)
    entry = {
        "mission": "13-cpu-tuning",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": best["pp"] if best else 0,
        "primary_unit": "best_pp_t_s",
        "conclusion": (
            f"Best t={best['threads']} prio={best['prio']}: pp={best['pp']} tg={best['tg']}"
            if best else "all failed"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
