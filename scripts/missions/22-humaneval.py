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


"""Mission 22: HumanEval on Qwen3.5-122B Q4_K_M via llama-server (SSH_BASE tunnel).

First 30 problems at pass@1 with 4 concurrent streams.
"""

import gzip, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _http_helper as H

ROOT = HERE.parent
RESULTS_FILE = ROOT / "results.jsonl"

LLAMA = "$HOME/.local/opt/llama.cpp/current"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"

DATA_DIR = HERE / "data"
PROBLEMS_JSONL = DATA_DIR / "humaneval.jsonl"

MIRROR = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def ensure_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    if PROBLEMS_JSONL.exists():
        return
    print("  downloading HumanEval...", flush=True)
    tmp = DATA_DIR / "humaneval.jsonl.gz"
    urllib.request.urlretrieve(MIRROR, tmp)
    with gzip.open(tmp, "rb") as gz, open(PROBLEMS_JSONL, "wb") as out:
        out.write(gz.read())
    tmp.unlink()


def load_problems(limit=None):
    items = [json.loads(l) for l in open(PROBLEMS_JSONL)]
    return items[:limit] if limit else items


def build_prompt(problem):
    return (
        "<|im_start|>user\nComplete the Python function body. "
        "Output ONLY the function body (indented), no explanation, "
        "no markdown fences, no re-declaration of the signature:\n\n"
        f"{problem['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    )


def extract_code(raw, prompt_sig):
    text = raw
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    if prompt_sig and prompt_sig in text:
        text = text[text.index(prompt_sig) + len(prompt_sig):]
    return text


def hit(prompt_text, n_predict=512):
    body = {"prompt": prompt_text, "n_predict": n_predict,
            "temperature": 0.0, "stream": False,
            "stop": ["<|im_end|>"]}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{H.BASE_URL}/completion", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def run_test(problem, body):
    prompt = problem["prompt"]
    test = problem["test"]
    entry = problem["entry_point"]
    cands = [
        prompt + body + f"\n\n{test}\n\ncheck({entry})\n",
        body + f"\n\n{test}\n\ncheck({entry})\n",
    ]
    for c in cands:
        try:
            p = subprocess.run(["python3", "-c", c], capture_output=True,
                               timeout=15, text=True)
            if p.returncode == 0:
                return True
        except Exception:
            continue
    return False


def solve(problem):
    resp = hit(build_prompt(problem), n_predict=512)
    if "error" in resp:
        return {"task_id": problem["task_id"], "passed": False,
                "error": resp["error"]}
    content = resp.get("content", "")
    sig_m = re.search(r"def\s+\w+\([^)]*\)[^:]*:", problem["prompt"])
    code = extract_code(content, sig_m.group(0) if sig_m else "")
    return {"task_id": problem["task_id"],
            "passed": run_test(problem, code),
            "content_preview": content[:200]}


def main():
    ensure_dataset()
    problems = load_problems(limit=30)
    print(f"  loaded {len(problems)} problems", flush=True)

    npl = 4
    launch = (f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-server "
              f"-m {MODEL} -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 2048 -b 2048 "
              f"-c {npl * 2048 + 512} -np {npl} "
              f"--host 127.0.0.1 --port {H.REMOTE_PORT}")
    H.start_remote_server(launch)
    tunnel = H.open_tunnel()
    try:
        if not H.wait_ready():
            print("ERROR: server start failed", flush=True); return
        print(f"  solving with {npl} concurrent streams...", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=npl) as ex:
            results = list(ex.map(solve, problems))
        elapsed = time.time() - t0

        n_pass = sum(1 for r in results if r.get("passed"))
        pass1 = n_pass / len(results)

        summary = {
            "model": "Qwen3.5-122B-A10B-REAP-20-Q4_K_M",
            "n_problems": len(results), "n_pass": n_pass,
            "pass_at_1": round(pass1, 3),
            "elapsed_s": round(elapsed, 1),
            "results": results,
        }
        HERE.joinpath("results.json").write_text(json.dumps(summary, indent=2))
        entry = {"mission": "22-humaneval", "status": "complete", "date": "2026-04-17",
                 "primary_metric": round(pass1 * 100, 1),
                 "primary_unit": "humaneval_pass_at_1_pct_first30",
                 "conclusion": f"Qwen3.5-122B Q4_K_M pass@1 = {pass1:.1%} on first 30 "
                               f"({n_pass}/{len(results)})."}
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


if __name__ == "__main__":
    main()
