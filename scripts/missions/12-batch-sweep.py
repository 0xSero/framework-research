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


"""Mission 12: Sweep -b and -ub for Qwen3.5-122B prefill on Vulkan gfx1151."""

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

# ub must be <= b. Keep combinations that respect that.
BATCH = [1024, 2048, 4096, 8192]
UBATCH = [256, 512, 1024, 2048]
CTX = 8192


def ssh(cmd: str, timeout: int = 600):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-bench 2>/dev/null"])
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run(b: int, ub: int):
    cmd = (
        f"timeout 500s bash -c '{BENCH} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk q8_0 -ctv q8_0 -b {b} -ub {ub} -p {CTX} -n 128 "
        f"-r 1 --no-warmup'"
    )
    print(f"  b={b} ub={ub} ...", flush=True)
    res = ssh(cmd, 560)
    out = res.stdout or ""
    pp = tg = None
    for line in out.splitlines():
        m = re.search(rf"pp{CTX}\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return {"b": b, "ub": ub, "pp": pp, "tg": tg}


def main():
    results = []
    for b in BATCH:
        for ub in UBATCH:
            if ub > b:
                continue
            r = run(b, ub)
            results.append(r)
            print(f"    pp={r['pp']} tg={r['tg']}", flush=True)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    best = max((r for r in results if r["pp"]), key=lambda r: r["pp"], default=None)
    entry = {
        "mission": "12-batch-sweep",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": best["pp"] if best else 0,
        "primary_unit": "best_pp_t_s",
        "conclusion": (
            f"Best b={best['b']} ub={best['ub']}: pp={best['pp']} tg={best['tg']}"
            if best else "all configs failed"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
