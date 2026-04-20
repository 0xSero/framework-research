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


"""Mission 29: Does needle retrieval collapse at 32K depend on model quant?

Mission 28 showed Q4_K_M fails 0/5 at 32K needle (every depth). Was that
quantization damage or inherent to the REAP-20 pruning? Test Q6_K and Q8_0 at
the same (32K × {0.1, 0.3, 0.5, 0.7, 0.9}) grid.
"""

import json, os, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _http_helper as H

ROOT = HERE.parent
RESULTS_FILE = ROOT / "results.jsonl"

LLAMA = "$HOME/.local/opt/llama.cpp/current"
MODELS = {
    "Q4_K_M": "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf",
    "Q6_K":   "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q6_K.gguf",
    "Q8_0":   "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q8_0.gguf",
}

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
CTX = 32768
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]


def build_prompt(total_ctx, depth):
    n_before = int(total_ctx * depth)
    n_after = total_ctx - n_before - 60
    block = DISTRACTOR
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
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read())
    except Exception as e:
        return {"error": str(e), "elapsed_s": time.time() - t0}
    return {"elapsed_s": round(time.time() - t0, 2),
            "content": resp.get("content", ""),
            "tokens": resp.get("tokens_predicted")}


def run_for_model(label, model_path):
    print(f"\n--- {label} ---", flush=True)
    launch = (f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-server "
              f"-m {model_path} -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -b 2048 "
              f"-c {CTX + 512} -np 1 "
              f"--host 127.0.0.1 --port {H.REMOTE_PORT}")
    H.start_remote_server(launch)
    tunnel = H.open_tunnel()
    try:
        if not H.wait_ready(max_wait_s=600):
            return [{"label": label, "error": "server_start_fail"}]
        out = []
        for depth in DEPTHS:
            prompt = build_prompt(CTX, depth)
            print(f"  {label} d={depth} chars={len(prompt)}", flush=True)
            r = hit(prompt)
            content = (r.get("content") or "").lower()
            found = "alpha-seven-delta-nine-mirror" in content
            entry = {"model": label, "ctx": CTX, "depth": depth,
                     "found_needle": found,
                     "elapsed_s": r.get("elapsed_s"),
                     "answer_preview": (r.get("content") or "")[:150]}
            out.append(entry)
            print(f"    found={found} elapsed={r.get('elapsed_s')}", flush=True)
        return out
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


def main():
    results = []
    for label, path in MODELS.items():
        for entry in run_for_model(label, path):
            results.append(entry)
            HERE.joinpath("results.json").write_text(json.dumps(results, indent=2))

    hits_by_model = {}
    for r in results:
        m = r.get("model")
        if m is None: continue
        hits_by_model.setdefault(m, [0, 0])
        hits_by_model[m][1] += 1
        if r.get("found_needle"):
            hits_by_model[m][0] += 1

    entry = {"mission": "29-needle-by-quant", "status": "complete",
             "date": "2026-04-17",
             "primary_metric": sum(h[0] for h in hits_by_model.values()),
             "primary_unit": "total_needles_found_across_3_quants",
             "conclusion": f"32K needle by quant: " +
                           ", ".join(f"{m}={v[0]}/{v[1]}" for m, v in hits_by_model.items())}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
