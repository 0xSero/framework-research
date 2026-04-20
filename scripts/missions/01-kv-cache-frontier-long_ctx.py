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


"""Mission 01 extension: Push q8_0/q8_0 and q4_0/q4_0 to 65K and 131K.

Mission 01 capped at 32K to keep the full 11×6 sweep feasible. The winners
(symmetric q8_0 and q4_0) never timed out at 32K, so the interesting question
is how far they scale. Asymmetric combos are skipped — they already timed out.
"""

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"

# (SSH_BASE defined in header)

LLAMA_BENCH = (
    "env LD_LIBRARY_PATH=$HOME/.local/opt/llama.cpp/current "
    "$HOME/.local/opt/llama.cpp/current/llama-bench"
)
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf"

COMBOS = [("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0")]
CONTEXTS = [65536, 131072]


def ssh_cmd(cmd: str, timeout: int = 1500):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-bench 2>/dev/null"])
        print(f"    TIMEOUT {timeout}s", flush=True)
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run_combo(ck, cv, ctx):
    cmd = (
        f"timeout 1200s bash -c '{LLAMA_BENCH} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk {ck} -ctv {cv} -p {ctx} -n 128 -r 1 --no-warmup'"
    )
    print(f"  {ck}/{cv} @ {ctx} ...", flush=True)
    res = ssh_cmd(cmd)
    out = res.stdout
    pp = tg = None
    for line in out.splitlines():
        m = re.search(rf"pp{ctx}\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))
    return {"pp": pp, "tg": tg}


def main():
    results = []
    for ck, cv in COMBOS:
        for ctx in CONTEXTS:
            r = run_combo(ck, cv, ctx)
            results.append({"ck": ck, "cv": cv, "ctx": ctx, **r})
            print(f"    pp={r['pp']} tg={r['tg']}", flush=True)

    out_path = Path(__file__).parent / "results_long_ctx.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Find best working combo at each long context
    summary_by_ctx = {}
    for r in results:
        if r["pp"] is None:
            continue
        ctx = r["ctx"]
        if ctx not in summary_by_ctx or r["pp"] > summary_by_ctx[ctx]["pp"]:
            summary_by_ctx[ctx] = r

    entry = {
        "mission": "01-kv-cache-frontier",
        "status": "extended",
        "date": "2026-04-16",
        "variant": "long_context_65k_131k",
        "primary_metric": len(summary_by_ctx),
        "primary_unit": "contexts_reaching_131k",
        "conclusion": (
            f"Tested f16/f16, q8_0/q8_0, q4_0/q4_0 at 65K & 131K. "
            f"Best at 131K: {summary_by_ctx.get(131072, {})}"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
