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


"""Mission 33: Full HumanEval 164 with v2 multi-strategy parser."""
import gzip, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _http_helper as H

# Re-use Mission 32's logic
sys.path.insert(0, str(HERE.parent / "32-humaneval-v2"))
from run import build_prompt, candidates_from_output, try_candidate, hit

ROOT = HERE.parent
RESULTS_FILE = ROOT / "results.jsonl"
LLAMA = "$HOME/.local/opt/llama.cpp/current"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"
PROBLEMS = HERE.parent / "32-humaneval-v2" / "data" / "humaneval.jsonl"


def solve(problem):
    resp = hit(build_prompt(problem))
    if "error" in resp:
        return {"task_id": problem["task_id"], "passed": False, "error": resp["error"]}
    raw = resp.get("content", "")
    won = None
    for label, cand in candidates_from_output(raw, problem):
        ok, err = try_candidate(cand)
        if ok:
            won = label
            break
    return {"task_id": problem["task_id"], "passed": won is not None,
            "won_strategy": won, "raw_preview": raw[:200]}


def main():
    problems = [json.loads(l) for l in open(PROBLEMS)]
    print(f"  loaded {len(problems)} problems", flush=True)
    npl = 4
    launch = (f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-server "
              f"-m {MODEL} -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 "
              f"-c {npl * 3072 + 512} -np {npl} "
              f"--host 127.0.0.1 --port {H.REMOTE_PORT}")
    H.start_remote_server(launch)
    tunnel = H.open_tunnel()
    try:
        if not H.wait_ready(max_wait_s=600):
            print("server fail"); return
        t0 = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=npl) as ex:
            for i, r in enumerate(ex.map(solve, problems)):
                results.append(r)
                if (i+1) % 10 == 0:
                    n = sum(1 for x in results if x["passed"])
                    print(f"    {i+1}/{len(problems)} pass={n} ({n/(i+1):.1%})", flush=True)
        elapsed = time.time() - t0
        n_pass = sum(1 for r in results if r["passed"])
        pass1 = n_pass / len(results)
        summary = {"model": "Qwen3.5-122B-A10B-REAP-20-Q4_K_M",
                   "n": len(results), "n_pass": n_pass, "pass_at_1": round(pass1,3),
                   "elapsed_s": round(elapsed,1), "results": results}
        HERE.joinpath("results.json").write_text(json.dumps(summary, indent=2))
        entry = {"mission": "33-humaneval-full-v2", "status": "complete",
                 "date": "2026-04-17",
                 "primary_metric": round(pass1*100, 1),
                 "primary_unit": "humaneval_pass_at_1_pct_full_164_v2parser",
                 "conclusion": f"Qwen3.5-122B Q4_K_M FULL HumanEval pass@1 = {pass1:.1%} "
                               f"({n_pass}/{len(results)}) in {elapsed:.0f}s with v2 parser"}
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


if __name__ == "__main__":
    main()
