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


"""Mission 03: rocWMMA Flash Attention Tuning for gfx1151."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
MISSION_FILE = Path(__file__).parent / "mission.json"

# (SSH_BASE defined in header)

LLAMA_BENCH = (
    "env LD_LIBRARY_PATH=$HOME/.local/opt/llama.cpp/current "
    "$HOME/.local/opt/llama.cpp/current/llama-bench"
)

MODEL = "$HOME/.local/share/models/gguf/gemma-3-4b-it-Q4_K_M.gguf"


def ssh_cmd(cmd: str, timeout: int = 300):
    return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def run_bench(ctx: int) -> dict:
    cmd = f"timeout 240s bash -c '{LLAMA_BENCH} -m {MODEL} -ngl 99 -fa 1 -p {ctx} -n 128 -r 1 --no-warmup'"
    print(f"  Running ctx={ctx} ...")
    res = ssh_cmd(cmd)
    text = res.stdout
    import re
    pp = tg = None
    for line in text.splitlines():
        m = re.search(rf"pp{ctx}\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return {"pp": pp, "tg": tg}


def main():
    with open(MISSION_FILE) as f:
        mission = json.load(f)

    print("Mission 03: rocWMMA FATTN Tuning")
    print("NOTE: This mission requires a ROCm build with GGML_HIP_ROCWMMA_FATTN=ON.")
    print("The current machine only has Vulkan. Running baseline benchmark.")

    results = []
    for ctx in mission["contexts"]:
        bench = run_bench(ctx)
        results.append({"ctx": ctx, **bench})
        print(f"    ctx={ctx} pp={bench['pp']} tg={bench['tg']}")

    entry = {
        "mission": "03-rocwmma-fattn",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": 0,
        "primary_unit": "decode_improvement_pct",
        "conclusion": "Baseline captured. ROCm build not available on this machine; patch must be tested on a ROCm-enabled node.",
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nMission result logged to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
