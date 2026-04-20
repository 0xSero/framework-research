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


"""Mission 11: Q4_K_M vs Q6_K vs Q8_0 on Qwen3.5-122B-A10B-REAP-20.

Sweeps every context from 512 → 131K with q8_0/q8_0 KV (the Mission 01 winner)
so we compare pure quant effects on prefill, decode, and memory pressure.
"""

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
MODELS = {
    "Q4_K_M": "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf",
    "Q6_K":   "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf",
    "Q8_0":   "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q8_0.gguf",
}

# Skip Q6_K 512-32K (already have that data from Mission 01) but re-run 65K/131K
# for apples-to-apples with Q4_K_M & Q8_0. For Q4_K_M & Q8_0, full sweep.
CONTEXTS = [512, 4096, 16384, 32768, 65536, 131072]


def ssh(cmd: str, timeout: int):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-bench 2>/dev/null"])
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run(model_path: str, ctx: int):
    timeout = max(900, int(ctx / 70))
    cmd = (
        f"timeout {timeout}s bash -c '{BENCH} -m {model_path} -ngl 99 -fa 1 "
        f"-ctk q8_0 -ctv q8_0 -p {ctx} -n 128 -r 1 --no-warmup'"
    )
    print(f"  {Path(model_path).name} @ {ctx} ...", flush=True)
    res = ssh(cmd, timeout + 120)
    out = res.stdout or ""
    pp = tg = None
    for line in out.splitlines():
        m = re.search(rf"pp{ctx}\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    errors = [l for l in out.splitlines() if re.search(r"error|oom|failed", l, re.I)][:3]
    return {"pp": pp, "tg": tg, "errors": errors}


def main():
    results = []
    for name, path in MODELS.items():
        for ctx in CONTEXTS:
            r = run(path, ctx)
            entry = {"model": name, "ctx": ctx, **r}
            results.append(entry)
            print(f"    pp={r['pp']} tg={r['tg']}", flush=True)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summarize pp@4K and tg@4K by model
    pp_4k = {r["model"]: r["pp"] for r in results if r["ctx"] == 4096}
    tg_4k = {r["model"]: r["tg"] for r in results if r["ctx"] == 4096}
    fastest_pp = max(pp_4k, key=lambda k: pp_4k[k] or 0)
    entry = {
        "mission": "11-model-quant-pareto",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": pp_4k.get(fastest_pp, 0),
        "primary_unit": "best_pp_at_4k_t_s",
        "conclusion": (
            f"pp@4K: {pp_4k} — tg@4K: {tg_4k}. "
            f"Fastest prefill: {fastest_pp}."
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
