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


"""Mission 19: Speculative decoding with Q4_K_M target + 0.8B draft.

Mission 08 used Q6_K target (baseline tg=24.3) and hit 48.18 t/s at draft_len=5.
Q4_K_M target has baseline tg=29.5 (+21%). Test if speculative stacks on top.
Hypothesis: 60-70 t/s feasible.
"""

import json, os, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
# (SSH_BASE defined in header)
LLAMA_SPEC = "$HOME/.local/opt/llama.cpp/llama-spec"
TARGET = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"
DRAFT = "$HOME/.local/share/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
REMOTE_OUT = "/tmp/mission19_out.txt"

PROMPT = ("Write a detailed technical specification for a distributed key-value "
          "store with strong consistency: the replication protocol, "
          "failure-detection strategy, compaction algorithm, and recovery procedure.")
N_PREDICT = 256
DRAFT_LENGTHS = [3, 5, 7]
BASELINE_TG = 29.48  # Mission 11 Q4_K_M @ 4K q8_0 KV


def ssh(cmd, timeout):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(SSH_BASE + ["killall -9 llama-speculative 2>/dev/null"], timeout=30)
        except Exception:
            pass
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr="")


def parse(out):
    tg = n_drafted = n_accept = None
    for line in out.splitlines():
        m = re.search(r"n_drafted\s*=\s*(\d+)", line)
        if m: n_drafted = int(m.group(1))
        m = re.search(r"n_accept\s*=\s*(\d+)", line)
        if m: n_accept = int(m.group(1))
        m = re.search(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", line)
        if m:
            ms, toks = float(m.group(1)), int(m.group(2))
            if ms: tg = toks / (ms / 1000.0)
    accept = (n_accept / n_drafted) if (n_drafted and n_accept) else None
    return tg, accept, n_drafted, n_accept


def run_spec(dl):
    cmd = (f"timeout 240 env LD_LIBRARY_PATH={LLAMA_SPEC} {LLAMA_SPEC}/llama-speculative "
           f"-m {TARGET} -md {DRAFT} -ngl 99 -ngld 99 -fa 1 "
           f"-ctk q8_0 -ctv q8_0 --draft-max {dl} -c 4096 -n {N_PREDICT} "
           f"-p {json.dumps(PROMPT)} > {REMOTE_OUT} 2>&1; tail -n 80 {REMOTE_OUT}")
    print(f"  spec dl={dl} ...", flush=True)
    t0 = time.time()
    res = ssh(cmd, 300)
    elapsed = time.time() - t0
    tg, accept, nd, na = parse(res.stdout or "")
    print(f"    elapsed={elapsed:.1f}s tg={tg} accept={accept} drafted={nd} accepted={na}",
          flush=True)
    return {"draft_len": dl, "tg_tps": tg, "accept": accept,
            "n_drafted": nd, "n_accept": na, "elapsed_s": round(elapsed, 1)}


def main():
    results = [{"variant": "baseline", "draft_len": 0, "tg_tps": BASELINE_TG,
                "source": "Mission 11 Q4_K_M@4K q8_0 KV"}]
    for dl in DRAFT_LENGTHS:
        results.append(run_spec(dl))

    Path(__file__).parent.joinpath("results.json").write_text(json.dumps(results, indent=2))
    best = max((r for r in results if r.get("tg_tps") and r.get("draft_len")),
               key=lambda r: r["tg_tps"], default=None)
    speedup = (best["tg_tps"] / BASELINE_TG) if best else 0
    entry = {"mission": "19-spec-q4km", "status": "complete", "date": "2026-04-17",
             "primary_metric": round(speedup, 3), "primary_unit": "tg_speedup_x_vs_q4km_baseline",
             "conclusion": f"Baseline Q4_K_M tg={BASELINE_TG} t/s. Best spec: "
                           f"dl={best['draft_len'] if best else None} "
                           f"tg={best['tg_tps'] if best else None} "
                           f"({speedup:.2f}x)"}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Saved.")


if __name__ == "__main__":
    main()
