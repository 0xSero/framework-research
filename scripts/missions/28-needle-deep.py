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


"""Mission 28: Deep needle-in-haystack map across ctx × depth.

Maps the retrieval accuracy over ctx ∈ {16K, 32K, 65K, 131K} × depth ∈
{0.1, 0.3, 0.5, 0.7, 0.9}. 20 points to visualize the full retention
landscape.
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
DISTRACTOR = (
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

CTXS = [16384, 32768, 65536, 131072]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_prompt(total_ctx, depth):
    n_before = int(total_ctx * depth)
    n_after = total_ctx - n_before - 60
    chars_per_tok = 4
    block = DISTRACTOR
    reps_before = max(1, int(n_before * chars_per_tok / len(block)))
    reps_after = max(1, int(n_after * chars_per_tok / len(block)))
    return (block * reps_before) + "\n\n" + NEEDLE + "\n\n" + (block * reps_after) + QUESTION


def hit(prompt, n=80):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0.0,
            "stream": False, "stop": ["<|im_end|>", "\n\n\n"]}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{H.BASE_URL}/completion", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
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
            print("ERROR: server start failed", flush=True); return

        results = []
        for ctx in CTXS:
            for depth in DEPTHS:
                prompt = build_prompt(ctx, depth)
                print(f"  ctx={ctx} depth={depth} chars={len(prompt)}", flush=True)
                r = hit(prompt)
                content = (r.get("content") or "").lower()
                found = "alpha-seven-delta-nine-mirror" in content
                entry = {"ctx": ctx, "depth": depth,
                         "found_needle": found,
                         "elapsed_s": r.get("elapsed_s"),
                         "tokens": r.get("tokens"),
                         "answer_preview": (r.get("content") or "")[:160],
                         "error": r.get("error")}
                results.append(entry)
                HERE.joinpath("results.json").write_text(json.dumps(results, indent=2))
                print(f"    found={found} elapsed={r.get('elapsed_s')}", flush=True)

        hits = sum(1 for r in results if r["found_needle"])
        entry = {"mission": "28-needle-deep", "status": "complete",
                 "date": "2026-04-17",
                 "primary_metric": hits,
                 "primary_unit": f"needles_found_of_{len(results)}",
                 "conclusion": (f"{hits}/{len(results)} hits across 4×5 ctx/depth grid. "
                                f"Details: " +
                                ", ".join(f"{r['ctx']}@d{r['depth']}={'Y' if r['found_needle'] else 'N'}"
                                           for r in results))}
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


if __name__ == "__main__":
    main()
