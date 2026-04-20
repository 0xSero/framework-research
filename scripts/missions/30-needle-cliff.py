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


"""Mission 30: Locate Qwen3.5-122B's needle-retrieval cliff precisely.

Mission 28 found 16K=all ✓, 32K=random. Sweep 18K, 20K, 24K, 28K at
depth=0.5 (hardest case from "lost in the middle") to find the exact cliff.
3 trials per point (deterministic sampling, but different distractor rotations
to reduce variance).
"""

import json, os, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _http_helper as H

ROOT = HERE.parent
RESULTS_FILE = ROOT / "results.jsonl"

LLAMA = "$HOME/.local/opt/llama.cpp/current"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"

NEEDLE = "The secret validation code is alpha-seven-delta-nine-mirror."
QUESTION = "\n\nQUESTION: What is the secret validation code? Answer in one sentence.\n"
DISTRACTOR_BLOCK = (
    "In distributed systems, the CAP theorem states a tradeoff between "
    "Consistency, Availability, and Partition tolerance; most practical "
    "designs pick two. Replication protocols like Raft and Paxos solve "
    "consensus under asynchronous failures using leader election and "
    "append-only logs. Key-value stores implement compaction through "
    "tombstones, time-to-live sweeps, and background merge strategies. "
    "Storage engines differ in bloom filter usage and cache layer hit "
    "rates. Compression codecs include zstd, lz4, and snappy, each "
    "trading CPU for wire size. Network partitions can trigger split-"
    "brain where two coordinators both believe they are primary; "
    "fencing tokens are the standard mitigation. Membership protocols "
    "like SWIM provide scalable failure detection via gossip. "
)

CTXS = [18000, 20000, 22000, 24000, 26000, 28000, 30000]
DEPTHS = [0.25, 0.5, 0.75]


def build_prompt(total_ctx, depth, seed=0):
    # rotate distractor text by seed chars to vary tokenization slightly
    block = DISTRACTOR_BLOCK
    if seed:
        block = block[seed % len(block):] + block[: seed % len(block)]
    n_before = int(total_ctx * depth)
    n_after = total_ctx - n_before - 60
    reps_before = max(1, int(n_before * 4 / len(block)))
    reps_after = max(1, int(n_after * 4 / len(block)))
    return (block * reps_before) + "\n\n" + NEEDLE + "\n\n" + (block * reps_after) + QUESTION


def hit(prompt, n=80):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0.0,
            "stream": False, "stop": ["<|im_end|>", "\n\n\n"]}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{H.BASE_URL}/completion", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.time() - t0}
    return {"elapsed_s": round(time.time() - t0, 2),
            "content": resp.get("content", ""),
            "tokens": resp.get("tokens_predicted")}


def main():
    max_ctx = max(CTXS) + 512
    launch = (f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-server "
              f"-m {MODEL} -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -b 2048 "
              f"-c {max_ctx} -np 1 "
              f"--host 127.0.0.1 --port {H.REMOTE_PORT}")
    H.start_remote_server(launch)
    tunnel = H.open_tunnel()
    try:
        if not H.wait_ready(max_wait_s=600):
            print("server failed"); return

        results = []
        for ctx in CTXS:
            for depth in DEPTHS:
                prompt = build_prompt(ctx, depth)
                print(f"  ctx={ctx} d={depth}", flush=True)
                r = hit(prompt)
                content = (r.get("content") or "").lower()
                found = "alpha-seven-delta-nine-mirror" in content
                entry = {"ctx": ctx, "depth": depth, "found_needle": found,
                         "elapsed_s": r.get("elapsed_s"),
                         "answer_preview": (r.get("content") or "")[:150]}
                results.append(entry)
                HERE.joinpath("results.json").write_text(json.dumps(results, indent=2))
                print(f"    found={found}", flush=True)

        # Summarise per ctx
        by_ctx = {}
        for r in results:
            by_ctx.setdefault(r["ctx"], [0, 0])
            by_ctx[r["ctx"]][1] += 1
            if r["found_needle"]:
                by_ctx[r["ctx"]][0] += 1

        # Locate cliff: first ctx where hit rate drops below 1.0
        cliff = None
        for ctx in sorted(by_ctx):
            hits, total = by_ctx[ctx]
            if hits < total:
                cliff = ctx
                break

        entry = {"mission": "30-needle-cliff", "status": "complete",
                 "date": "2026-04-17",
                 "primary_metric": cliff or 0,
                 "primary_unit": "first_ctx_below_100pct_retrieval",
                 "conclusion": f"Retrieval cliff at ctx={cliff}. "
                               f"Per ctx: " +
                               ", ".join(f"{c}={v[0]}/{v[1]}" for c, v in sorted(by_ctx.items()))}
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


if __name__ == "__main__":
    main()
