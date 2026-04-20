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


"""Mission 31: Dig into the 28K dead zone.

Mission 30 showed Qwen3.5-122B hits 0/3 at exactly 28K while 26K=3/3 and 30K=1/3.
Sweep 27K, 27500, 28K, 28500, 29K at depth=0.5 with 4 seeds (slight prompt
variations) to confirm whether the 28K failure is a specific attention boundary
or statistical noise.
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

NEEDLE_TEMPLATES = [
    "The secret validation code is alpha-seven-delta-nine-mirror.",
    "The hidden access phrase is: bravo-four-tango-six-paper.",
    "Remember: the passphrase equals charlie-two-sierra-eight-glass.",
    "Important fact — the magic token is delta-nine-omega-three-river.",
]
NEEDLE_CHECKS = [
    "alpha-seven-delta-nine-mirror",
    "bravo-four-tango-six-paper",
    "charlie-two-sierra-eight-glass",
    "delta-nine-omega-three-river",
]
QUESTION_TEMPLATES = [
    "\n\nQUESTION: What is the secret validation code? Answer briefly.\n",
    "\n\nQUESTION: What was the hidden access phrase? Answer briefly.\n",
    "\n\nQUESTION: What passphrase was mentioned? Answer briefly.\n",
    "\n\nQUESTION: What magic token was noted? Answer briefly.\n",
]
DISTRACTOR = ("In distributed systems, the CAP theorem states a tradeoff between "
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
              "like SWIM provide scalable failure detection via gossip. ")

CTXS = [27000, 27500, 28000, 28500, 29000]
DEPTH = 0.5


def build_prompt(total_ctx, seed):
    n_before = int(total_ctx * DEPTH)
    n_after = total_ctx - n_before - 60
    reps_before = max(1, int(n_before * 4 / len(DISTRACTOR)))
    reps_after = max(1, int(n_after * 4 / len(DISTRACTOR)))
    needle = NEEDLE_TEMPLATES[seed % len(NEEDLE_TEMPLATES)]
    q = QUESTION_TEMPLATES[seed % len(QUESTION_TEMPLATES)]
    check = NEEDLE_CHECKS[seed % len(NEEDLE_CHECKS)]
    return (DISTRACTOR * reps_before) + "\n\n" + needle + "\n\n" + (DISTRACTOR * reps_after) + q, check


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
            "content": resp.get("content", "")}


def main():
    launch = (f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-server "
              f"-m {MODEL} -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -b 2048 "
              f"-c {max(CTXS) + 512} -np 1 "
              f"--host 127.0.0.1 --port {H.REMOTE_PORT}")
    H.start_remote_server(launch)
    tunnel = H.open_tunnel()
    try:
        if not H.wait_ready(max_wait_s=600): return
        results = []
        for ctx in CTXS:
            for seed in range(4):
                prompt, check = build_prompt(ctx, seed)
                print(f"  ctx={ctx} seed={seed}", flush=True)
                r = hit(prompt)
                content = (r.get("content") or "").lower()
                found = check in content
                entry = {"ctx": ctx, "seed": seed, "needle_key": check,
                         "found_needle": found,
                         "elapsed_s": r.get("elapsed_s"),
                         "answer_preview": (r.get("content") or "")[:120]}
                results.append(entry)
                HERE.joinpath("results.json").write_text(json.dumps(results, indent=2))
                print(f"    found={found}", flush=True)

        by_ctx = {}
        for r in results:
            by_ctx.setdefault(r["ctx"], [0, 0])
            by_ctx[r["ctx"]][1] += 1
            if r["found_needle"]: by_ctx[r["ctx"]][0] += 1
        entry = {"mission": "31-28k-deadzone", "status": "complete",
                 "date": "2026-04-17",
                 "primary_metric": by_ctx.get(28000, [0, 0])[0],
                 "primary_unit": "hits_at_28k_of_4",
                 "conclusion": "28K dead zone check: " + ", ".join(
                     f"{c}={v[0]}/{v[1]}" for c, v in sorted(by_ctx.items()))}
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


if __name__ == "__main__":
    main()
