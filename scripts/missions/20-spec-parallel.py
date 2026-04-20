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


"""Mission 20: Speculative + parallel via llama-server (SSH_BASE tunnel)."""

import json, os, sys, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _http_helper as H

ROOT = HERE.parent
RESULTS_FILE = ROOT / "results.jsonl"

LLAMA_SPEC = "$HOME/.local/opt/llama.cpp/llama-spec"
TARGET = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"
DRAFT = "$HOME/.local/share/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf"

PROMPT = ("Write a detailed technical specification for a distributed key-value "
          "store with strong consistency: the replication protocol, "
          "failure-detection strategy, compaction algorithm, and recovery procedure. "
          "Cover both the theory and concrete implementation choices.")
N_PREDICT = 128


def hit_server(seed):
    body = {"prompt": PROMPT + f"\n\nSeed: {seed}\n",
            "n_predict": N_PREDICT, "temperature": 0.0, "stream": False}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{H.BASE_URL}/completion", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}
    elapsed = time.time() - t0
    tokens = resp.get("tokens_predicted", 0)
    timings = resp.get("timings", {})
    tg_tps = timings.get("predicted_per_second") or (tokens / elapsed if elapsed else None)
    return {"elapsed_s": round(elapsed, 2), "tokens": tokens, "tg_tps": tg_tps}


def run_config(npl, draft_max):
    print(f"  npl={npl} draft_max={draft_max}", flush=True)
    ctx_size = max(4096, (N_PREDICT + 512) * npl + 512)
    launch = (f"env LD_LIBRARY_PATH={LLAMA_SPEC} {LLAMA_SPEC}/llama-server "
              f"-m {TARGET} -md {DRAFT} -ngl 99 -ngld 99 -fa 1 "
              f"-ctk q8_0 -ctv q8_0 --draft-max {draft_max} "
              f"-c {ctx_size} -np {npl} "
              f"--host 127.0.0.1 --port {H.REMOTE_PORT}")
    H.start_remote_server(launch)
    tunnel = H.open_tunnel()
    try:
        if not H.wait_ready():
            return {"npl": npl, "error": "server_start_fail"}
        hit_server("warmup")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=npl) as ex:
            results = list(ex.map(hit_server, range(npl)))
        wall_s = time.time() - t0

        ok = [r for r in results if "error" not in r]
        total_tokens = sum(r.get("tokens", 0) for r in ok)
        agg_tg = total_tokens / wall_s if wall_s > 0 else 0
        print(f"    wall={wall_s:.1f}s agg_tg={agg_tg:.1f} "
              f"per-stream={[r.get('tg_tps') for r in ok]}", flush=True)
        return {"npl": npl, "draft_max": draft_max,
                "wall_s": round(wall_s, 2),
                "total_tokens": total_tokens,
                "aggregate_tg_tps": round(agg_tg, 2),
                "per_stream": ok}
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


def main():
    results = []
    for npl in [1, 2, 4]:
        results.append(run_config(npl, draft_max=5))

    HERE.joinpath("results.json").write_text(json.dumps(results, indent=2))
    best = max((r for r in results if r.get("aggregate_tg_tps")),
               key=lambda r: r["aggregate_tg_tps"], default=None)
    entry = {"mission": "20-spec-parallel", "status": "complete", "date": "2026-04-17",
             "primary_metric": best["aggregate_tg_tps"] if best else 0,
             "primary_unit": "best_aggregate_tg_with_spec_parallel",
             "conclusion": f"Best: npl={best['npl'] if best else None} "
                           f"agg_tg={best['aggregate_tg_tps'] if best else None}"}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
