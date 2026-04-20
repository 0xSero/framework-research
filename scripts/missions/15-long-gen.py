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


"""Mission 15: Long-generation tg scaling on Qwen3.5-122B."""

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

N_PREDICT = [128, 512, 2048, 8192]


def ssh(cmd: str, timeout: int = 900):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-bench 2>/dev/null"])
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run(n: int):
    # llama-bench supports -n for tg length. Use -p 0 -n N to get pure tg.
    cmd = (
        f"timeout 800s bash -c '{BENCH} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk q8_0 -ctv q8_0 -p 0 -n {n} -r 1 --no-warmup'"
    )
    print(f"  tg{n} ...", flush=True)
    res = ssh(cmd, 860)
    out = res.stdout or ""
    tg = None
    for line in out.splitlines():
        m = re.search(rf"tg{n}\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return {"n": n, "tg": tg}


def main():
    results = []
    for n in N_PREDICT:
        r = run(n)
        results.append(r)
        print(f"    tg{n}={r['tg']}", flush=True)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    tg128 = next((r["tg"] for r in results if r["n"] == 128), None)
    tg8k = next((r["tg"] for r in results if r["n"] == 8192), None)
    degradation = ((tg128 - tg8k) / tg128 * 100) if (tg128 and tg8k) else None
    entry = {
        "mission": "15-long-gen",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": degradation or 0,
        "primary_unit": "tg_drop_pct_128_to_8192",
        "conclusion": (
            f"tg128={tg128} → tg8192={tg8k} ({degradation:.1f}% drop)"
            if degradation else f"tg results: {results}"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
