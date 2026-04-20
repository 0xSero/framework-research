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


"""Mission 07: Push q4_0/q4_0 KV past 131K on the 122B MoE.

Mission 01 validated symmetric q4_0 (75% KV reduction, minor speed loss) to 32K,
and the long-context extension covers 65K & 131K. This mission asks: how far can
we scale before a hard wall (OOM / FLASH_ATTN_EXT collapse / kernel failure)?

Targets 262144, 524288, 1048576. Runs one context at a time with fresh process
so a crash at 524K doesn't taint 262K data. Captures pp, tg, peak RSS, and any
llama.cpp warnings.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"

# (SSH_BASE defined in header)

LLAMA_DIR = "$HOME/.local/opt/llama.cpp/current"
LLAMA_BENCH = f"env LD_LIBRARY_PATH={LLAMA_DIR} {LLAMA_DIR}/llama-bench"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf"

CONTEXTS = [262144, 524288, 1048576]


def ssh_cmd(cmd: str, timeout: int):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-bench 2>/dev/null"])
        print(f"    LOCAL TIMEOUT {timeout}s", flush=True)
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def run_ctx(ctx: int):
    # Scale server-side timeout roughly with context size.
    # At 145 t/s prefill on 131K, 1M would be ~7200s. Give headroom.
    server_timeout = max(1200, int(ctx / 100))
    local_timeout = server_timeout + 300

    cmd = (
        f"timeout {server_timeout}s bash -c '{LLAMA_BENCH} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk q4_0 -ctv q4_0 -p {ctx} -n 128 -r 1 --no-warmup'"
    )
    print(f"  q4_0/q4_0 @ {ctx} (timeout={server_timeout}s) ...", flush=True)
    t0 = time.time()
    res = ssh_cmd(cmd, local_timeout)
    elapsed = time.time() - t0
    out = res.stdout or ""

    pp = tg = None
    for line in out.splitlines():
        m = re.search(rf"pp{ctx}\s*\|\s*([\d.]+)", line)
        if m:
            pp = float(m.group(1))
        m = re.search(r"tg128\s*\|\s*([\d.]+)", line)
        if m:
            tg = float(m.group(1))

    errors = [l for l in out.splitlines()
              if re.search(r"error|oom|failed|killed|abort|GGML_ASSERT", l, re.I)][:8]

    return {
        "ctx": ctx,
        "pp": pp,
        "tg": tg,
        "elapsed_s": round(elapsed, 1),
        "errors": errors,
        "tail": out.splitlines()[-5:] if out else [],
    }


def main():
    results = []
    for ctx in CONTEXTS:
        r = run_ctx(ctx)
        results.append(r)
        print(f"    pp={r['pp']} tg={r['tg']} elapsed={r['elapsed_s']}s", flush=True)
        if r["pp"] is None and r["errors"]:
            print(f"    errors: {r['errors']}", flush=True)
            # If we crashed at this ctx, no point testing larger
            if any("oom" in e.lower() or "out of memory" in e.lower() or "ggml_assert" in e.lower()
                   for e in r["errors"]):
                print("    HARD WALL — skipping larger contexts", flush=True)
                break

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    reached = [r for r in results if r["pp"] is not None]
    max_ctx = max((r["ctx"] for r in reached), default=0)
    entry = {
        "mission": "07-extreme-context",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": max_ctx,
        "primary_unit": "max_ctx_with_valid_pp",
        "conclusion": (
            f"q4_0/q4_0 reached {max_ctx} tokens. "
            f"pp @ max = {next((r['pp'] for r in reached if r['ctx'] == max_ctx), None)} t/s. "
            f"Tested {[r['ctx'] for r in results]}."
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
