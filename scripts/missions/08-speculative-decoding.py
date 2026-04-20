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


"""Mission 08 v3: Speculative decoding on Qwen3.5-122B with Qwen3.5-0.8B draft.

Uses llama-speculative directly. Baseline is taken from Mission 11 Q6_K data
(tg=24.3 t/s) to avoid re-running a 3-minute cold baseline.
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

LLAMA_SPEC = "$HOME/.local/opt/llama.cpp/llama-spec"
TARGET = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf"
DRAFT = "$HOME/.local/share/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"

REMOTE_OUT = "/tmp/mission08_output.txt"
PROMPT = (
    "Write a detailed technical specification for a distributed key-value "
    "store with strong consistency: the replication protocol, "
    "failure-detection strategy, compaction algorithm, and recovery procedure."
)
N_PREDICT = 256
DRAFT_LENGTHS = [3, 5, 7, 10, 16]

# Baseline from Mission 11 Q6_K @ 4K ctx q8_0 KV: tg=24.30 t/s.
BASELINE_TG = 24.30


def ssh(cmd: str, timeout: int):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(SSH_BASE + ["killall -9 llama-speculative 2>/dev/null"], timeout=30)
        except Exception:
            pass
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr="")


def parse_spec_output(out: str):
    """Parse llama-speculative output for tg t/s and acceptance rate."""
    tg_tps = None
    accept = None
    n_drafted = None
    n_accept = None
    for line in out.splitlines():
        # "encoded/decoded/drafted/accepted" summary lines (various versions)
        m = re.search(r"n_predict\s*=\s*(\d+)", line)
        if m:
            pass
        m = re.search(r"n_drafted\s*=\s*(\d+)", line)
        if m:
            n_drafted = int(m.group(1))
        m = re.search(r"n_accept\s*=\s*(\d+)", line)
        if m:
            n_accept = int(m.group(1))
        # "total time = XXX ms / YYY tokens" → t/s = YYY / (XXX/1000)
        m = re.search(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
        if m:
            ms = float(m.group(1))
            toks = int(m.group(2))
            if ms > 0:
                tg_tps = toks / (ms / 1000.0)
        # alternative: "X.XX tokens per second" in generate time block
        m = re.search(r"generate:.*?([\d.]+)\s*tokens per second", line, re.I)
        if m and not tg_tps:
            tg_tps = float(m.group(1))
    if n_drafted and n_accept:
        accept = n_accept / n_drafted
    return tg_tps, accept, n_drafted, n_accept


def run_spec(draft_len: int):
    cmd = (
        f"timeout 240 env LD_LIBRARY_PATH={LLAMA_SPEC} {LLAMA_SPEC}/llama-speculative "
        f"-m {TARGET} -md {DRAFT} -ngl 99 -ngld 99 -fa 1 "
        f"-ctk q8_0 -ctv q8_0 "
        f"--draft-max {draft_len} -c 4096 -n {N_PREDICT} "
        f"-p {json.dumps(PROMPT)} "
        f"> {REMOTE_OUT} 2>&1; "
        f"tail -n 80 {REMOTE_OUT}"
    )
    print(f"  spec draft_len={draft_len} ...", flush=True)
    t0 = time.time()
    res = ssh(cmd, 300)
    elapsed = time.time() - t0
    out = res.stdout or ""
    tg_tps, accept, n_drafted, n_accept = parse_spec_output(out)
    print(f"    elapsed={elapsed:.1f}s tg={tg_tps} accept={accept} "
          f"drafted={n_drafted} accepted={n_accept}", flush=True)
    return {"variant": "speculative", "draft_len": draft_len,
            "tg_tps": tg_tps, "accept": accept,
            "n_drafted": n_drafted, "n_accept": n_accept,
            "elapsed_s": round(elapsed, 1)}


def main():
    results = [{"variant": "baseline", "draft_len": 0,
                "tg_tps": BASELINE_TG, "accept": None,
                "source": "Mission 11 Q6_K@4K q8_0 KV"}]

    for dl in DRAFT_LENGTHS:
        r = run_spec(dl)
        results.append(r)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    best = None
    for r in results:
        if r.get("variant") == "speculative" and r.get("tg_tps"):
            if not best or r["tg_tps"] > best["tg_tps"]:
                best = r
    speedup = (best["tg_tps"] / BASELINE_TG) if best else 0.0

    entry = {
        "mission": "08-speculative-decoding",
        "status": "complete",
        "date": "2026-04-17",
        "primary_metric": round(speedup, 3),
        "primary_unit": "tg_speedup_x",
        "conclusion": (
            f"Baseline tg={BASELINE_TG} t/s (from Mission 11). "
            f"Best spec: draft_len={best['draft_len'] if best else None} "
            f"tg={best['tg_tps'] if best else None} "
            f"accept={best['accept'] if best else None}. "
            f"Speedup {speedup:.2f}x."
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
