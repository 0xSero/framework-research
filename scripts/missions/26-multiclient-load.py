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


"""Mission 26: 20-client HTTP load via llama-server (SSH_BASE tunnel)."""

import json, os, random, sys, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _http_helper as H

ROOT = HERE.parent
RESULTS_FILE = ROOT / "results.jsonl"

LLAMA = "$HOME/.local/opt/llama.cpp/current"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"

TOPICS = [
    "distributed consensus protocols", "vector databases for embeddings",
    "real-time stream processing", "columnar storage formats",
    "lock-free concurrent data structures", "gossip-based membership detection",
    "LSM-tree compaction strategies", "protocol buffer schema evolution",
    "CRDTs for eventual consistency", "service mesh sidecar patterns",
]


def hit(req_id, prompt_text, n_predict=100):
    body = {"prompt": prompt_text, "n_predict": n_predict,
            "temperature": 0.2, "stream": False}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{H.BASE_URL}/completion", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
    except Exception as e:
        return {"id": req_id, "error": str(e)}
    elapsed = time.time() - t0
    tokens = resp.get("tokens_predicted", 0)
    timings = resp.get("timings", {})
    ttft = (timings.get("prompt_ms", 0) / 1000.0) if timings.get("prompt_ms") else None
    return {"id": req_id, "elapsed_s": round(elapsed, 2),
            "tokens": tokens, "ttft_s": ttft,
            "tg_tps": timings.get("predicted_per_second")}


def pct(vals, p):
    vs = sorted(v for v in vals if v is not None)
    if not vs: return None
    return vs[min(int(len(vs) * p / 100), len(vs)-1)]


def main():
    N_CLIENTS = 20
    npl = 8
    launch = (f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-server "
              f"-m {MODEL} -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 "
              f"-c {npl * 2048 + 512} -np {npl} "
              f"--host 127.0.0.1 --port {H.REMOTE_PORT}")
    H.start_remote_server(launch)
    tunnel = H.open_tunnel()
    try:
        if not H.wait_ready():
            print("ERROR: server didn't start", flush=True); return
        print(f"  server ready. firing {N_CLIENTS} concurrent...", flush=True)
        hit(-1, f"Write about {random.choice(TOPICS)}.")

        random.seed(42)
        prompts = [(i, f"Explain {random.choice(TOPICS)} in 3 paragraphs, "
                      f"focusing on implementation concerns. Variant {i}.")
                   for i in range(N_CLIENTS)]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=N_CLIENTS) as ex:
            results = list(ex.map(lambda p: hit(*p), prompts))
        wall_s = time.time() - t0

        ok = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        total_tokens = sum(r.get("tokens", 0) for r in ok)
        agg_tg = total_tokens / wall_s if wall_s > 0 else 0

        summary = {
            "n_clients": N_CLIENTS, "n_successful": len(ok), "n_failed": len(failed),
            "wall_s": round(wall_s, 2), "total_tokens": total_tokens,
            "aggregate_tg_tps": round(agg_tg, 2),
            "ttft_p50": pct([r.get("ttft_s") for r in ok], 50),
            "ttft_p95": pct([r.get("ttft_s") for r in ok], 95),
            "ttft_p99": pct([r.get("ttft_s") for r in ok], 99),
            "elapsed_p50": pct([r.get("elapsed_s") for r in ok], 50),
            "elapsed_p95": pct([r.get("elapsed_s") for r in ok], 95),
            "elapsed_p99": pct([r.get("elapsed_s") for r in ok], 99),
            "per_request": results,
        }
        HERE.joinpath("results.json").write_text(json.dumps(summary, indent=2))
        entry = {"mission": "26-multiclient-load", "status": "complete",
                 "date": "2026-04-17",
                 "primary_metric": summary["aggregate_tg_tps"],
                 "primary_unit": "aggregate_tg_20clients_npl8",
                 "conclusion": f"20c/npl8: agg={summary['aggregate_tg_tps']} "
                               f"ttft p50={summary['ttft_p50']} p99={summary['ttft_p99']} "
                               f"elapsed p50={summary['elapsed_p50']} p99={summary['elapsed_p99']}"}
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


if __name__ == "__main__":
    main()
