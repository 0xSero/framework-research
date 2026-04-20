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


"""Mission 21: Draft model configuration sweep.

For the best Qwen3.5-122B target (Q4_K_M from Mission 11), sweep draft model
offload and `--draft-min` to find the global optimum.

Dimensions:
 1. Draft on GPU (ngld=99) vs CPU (ngld=0)
 2. --draft-min 0, 1, 3
 3. Fixed draft_len=5 (Mission 08 winner)
"""

import json, os, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
# (SSH_BASE defined in header)
LLAMA_SPEC = "$HOME/.local/opt/llama.cpp/llama-spec"
TARGET = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"
DRAFT = "$HOME/.local/share/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"
REMOTE_OUT = "/tmp/mission21_out.txt"

PROMPT = ("Write a detailed technical specification for a distributed key-value "
          "store with strong consistency: the replication protocol, "
          "failure-detection strategy, compaction algorithm, and recovery procedure.")
N_PREDICT = 256
DRAFT_MAX = 5

# (ngld, draft_min, label)
CONFIGS = [
    (99, 0, "gpu_draft_min0"),
    (99, 1, "gpu_draft_min1"),
    (99, 3, "gpu_draft_min3"),
    (0,  0, "cpu_draft_min0"),
]


def ssh(cmd, timeout):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        try: subprocess.run(SSH_BASE + ["killall -9 llama-speculative 2>/dev/null"], timeout=30)
        except Exception: pass
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


def run(ngld, draft_min, label):
    cmd = (f"timeout 240 env LD_LIBRARY_PATH={LLAMA_SPEC} {LLAMA_SPEC}/llama-speculative "
           f"-m {TARGET} -md {DRAFT} -ngl 99 -ngld {ngld} -fa 1 "
           f"-ctk q8_0 -ctv q8_0 --draft-max {DRAFT_MAX} --draft-min {draft_min} "
           f"-c 4096 -n {N_PREDICT} -p {json.dumps(PROMPT)} "
           f"> {REMOTE_OUT} 2>&1; tail -n 80 {REMOTE_OUT}")
    print(f"  {label} (ngld={ngld} draft_min={draft_min}) ...", flush=True)
    t0 = time.time()
    res = ssh(cmd, 300)
    elapsed = time.time() - t0
    tg, accept, nd, na = parse(res.stdout or "")
    print(f"    elapsed={elapsed:.1f}s tg={tg} accept={accept}", flush=True)
    return {"label": label, "ngld": ngld, "draft_min": draft_min,
            "tg_tps": tg, "accept": accept,
            "n_drafted": nd, "n_accept": na, "elapsed_s": round(elapsed, 1)}


def main():
    results = []
    for ngld, dm, label in CONFIGS:
        results.append(run(ngld, dm, label))

    Path(__file__).parent.joinpath("results.json").write_text(json.dumps(results, indent=2))
    best = max((r for r in results if r.get("tg_tps")),
               key=lambda r: r["tg_tps"], default=None)
    entry = {"mission": "21-draft-sweep", "status": "complete", "date": "2026-04-17",
             "primary_metric": best["tg_tps"] if best else 0,
             "primary_unit": "best_tg_tps",
             "conclusion": f"Best draft config: {best['label'] if best else 'none'} "
                           f"tg={best['tg_tps'] if best else None} accept={best['accept'] if best else None}"}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
