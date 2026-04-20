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


"""Mission 17: Stack the winners — Q4_K_M model + ub=2048 batching + parallel slots.

Uses llama-batched-bench with the Q4_K_M model. Default -ub bumped to 2048 per
Mission 12. Sweeps npl=1/4/8 to show compounded wins.
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
BATCHED = f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-batched-bench"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"

PP = 512
TG = 128
PARALLEL = [1, 4, 8]
UB = 2048


def ssh(cmd: str, timeout: int = 600):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-batched-bench 2>/dev/null"])
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run(n_parallel: int):
    cmd = (
        f"timeout 500s bash -c '{BATCHED} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk q8_0 -ctv q8_0 -ub {UB} -b {max(UB, 2048)} "
        f"-npp {PP} -ntg {TG} -npl {n_parallel} "
        f"-c {max(4096, (PP+TG) * n_parallel + 512)}'"
    )
    print(f"  npl={n_parallel} ub={UB} ...", flush=True)
    res = ssh(cmd, 560)
    out = res.stdout or ""
    s_pp = s_tg = s_total = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 8 and parts[0] == str(PP) and parts[2] == str(n_parallel):
            try:
                s_pp = float(parts[5])
                s_tg = float(parts[7])
                if len(parts) >= 10:
                    s_total = float(parts[9])
            except (ValueError, IndexError):
                pass
    return {"n_parallel": n_parallel, "s_pp_tps": s_pp, "s_tg_tps": s_tg,
            "s_total_tps": s_total}


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

    best_tg = max((r["s_tg_tps"] for r in results if r.get("s_tg_tps")), default=0)
    best_total = max((r["s_total_tps"] for r in results if r.get("s_total_tps")), default=0)
    entry = {
        "mission": "17-combined-winners",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": best_tg,
        "primary_unit": "best_aggregate_tg_t_s",
        "conclusion": (
            f"Q4_K_M + ub={UB} + parallel: best agg tg={best_tg} total={best_total}. "
            f"vs Mission 09 baseline (Q6_K, ub=512, npl=8): s_tg=53.55 s_total=157.73"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
