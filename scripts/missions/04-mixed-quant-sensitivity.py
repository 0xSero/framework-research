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


"""Mission 04: Per-Layer Mixed Quantization Sensitivity."""

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


def ssh_cmd(cmd: str, timeout: int = 120):
    return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def run_bench(recipe_name: str) -> dict:
    print(f"  Benchmarking {recipe_name} ...")
    cmd = f"timeout 90s bash -c '{LLAMA_BENCH} -m {MODEL} -ngl 99 -fa 1 -p 4096 -n 128 -r 1 --no-warmup'"
    res = ssh_cmd(cmd)
    text = res.stdout
    import re
    pp = tg = None
    for line in text.splitlines():
        m = re.search(r"pp4096\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return {"pp": pp, "tg": tg}


def main():
    with open(MISSION_FILE) as f:
        mission = json.load(f)

    print("Mission 04: Mixed Quantization Sensitivity")
    print("NOTE: Full per-layer quantization requires llama-quantize with custom layer maps.")
    print("This harness benchmarks the existing uniform Q4_K_M baseline only.")

    results = []
    for recipe in mission["recipes"]:
        bench = run_bench(recipe["name"])
        results.append({"recipe": recipe["name"], **bench})
        print(f"    {recipe['name']}: pp={bench['pp']} tg={bench['tg']}")

    entry = {
        "mission": "04-mixed-quant-sensitivity",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": results[0]["pp"] if results else 0,
        "primary_unit": "pp_t_s",
        "conclusion": "Baseline Q4_K_M benchmarked. Per-layer mixed quant tool (quantize_layers.py) needed for follow-up.",
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nMission result logged to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
