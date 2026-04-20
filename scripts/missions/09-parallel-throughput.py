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


"""Mission 09: Parallel throughput on Qwen3.5-122B via llama-batched-bench."""

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"

# (SSH_BASE defined in header)

LLAMA = "$HOME/.local/opt/llama.cpp/current"
BATCHED = f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-batched-bench"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf"

PARALLEL = [1, 2, 4, 8]
PP = 512
TG = 128


def ssh(cmd: str, timeout: int = 600):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-batched-bench 2>/dev/null"])
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run(n_parallel: int):
    # llama-batched-bench takes -npp (prompt len), -ntg (gen len), -npl (slots)
    cmd = (
        f"timeout 500s bash -c '{BATCHED} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk q8_0 -ctv q8_0 "
        f"-npp {PP} -ntg {TG} -npl {n_parallel} "
        f"-c {max(4096, (PP+TG) * n_parallel + 512)}'"
    )
    print(f"  npl={n_parallel} ...", flush=True)
    res = ssh(cmd, 560)
    out = res.stdout or ""

    # batched-bench output looks like:
    # |  PP |  TG |  B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | ...
    # parse the row where B == n_parallel
    s_pp = s_tg = s_total = None
    for line in out.splitlines():
        # match rows with numeric cells
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 8 and parts[0] == str(PP) and parts[2] == str(n_parallel):
            try:
                s_pp = float(parts[5])
                s_tg = float(parts[7])
                if len(parts) >= 10:
                    s_total = float(parts[9])
            except (ValueError, IndexError):
                pass
    return {
        "n_parallel": n_parallel,
        "s_pp_tps": s_pp,
        "s_tg_tps": s_tg,
        "s_total_tps": s_total,
        "tail": out.splitlines()[-12:],
    }


def main():
    results = []
    for p in PARALLEL:
        r = run(p)
        results.append(r)
        print(f"    s_pp={r['s_pp_tps']} s_tg={r['s_tg_tps']} s_total={r['s_total_tps']}",
              flush=True)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    r1 = next((r for r in results if r["n_parallel"] == 1), {})
    best = max((r for r in results if r.get("s_tg_tps")),
               key=lambda r: r["s_tg_tps"], default=None)
    speedup = (best["s_tg_tps"] / r1["s_tg_tps"]) if (best and r1.get("s_tg_tps")) else 0
    entry = {
        "mission": "09-parallel-throughput",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": round(speedup, 3),
        "primary_unit": "tg_aggregate_speedup_vs_npl1",
        "conclusion": (
            f"1-slot tg={r1.get('s_tg_tps')}; best @ npl={best['n_parallel'] if best else None} "
            f"tg={best['s_tg_tps'] if best else None} ({speedup:.2f}x)"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
