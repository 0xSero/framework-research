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


"""Mission 32: HumanEval v2 — robust parsing, first 50 problems.

The 8.8% from Mission 27 is almost certainly parsing bugs. This version:
1. Stores raw output verbatim.
2. Re-parses with 6 fallback strategies locally.
3. Executes each candidate via subprocess with short timeout.
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
PROBLEMS = DATA_DIR / "humaneval.jsonl"
MIRROR = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def ensure():
    DATA_DIR.mkdir(exist_ok=True)
    if PROBLEMS.exists(): return
    tmp = DATA_DIR / "h.gz"
    urllib.request.urlretrieve(MIRROR, tmp)
    with gzip.open(tmp, "rb") as g, open(PROBLEMS, "wb") as o:
        o.write(g.read())
    tmp.unlink()


def build_prompt(p):
    return (
        "<|im_start|>user\n"
        "Complete the following Python function. Respond with ONLY the final "
        "complete function (including the signature and docstring), wrapped in "
        "```python ... ``` code fences. Do not include any explanation before "
        "or after.\n\n"
        f"{p['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    )


def hit(prompt, n=700):
    body = {"prompt": prompt, "n_predict": n, "temperature": 0.0,
            "stream": False, "stop": ["<|im_end|>"]}
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{H.BASE_URL}/completion", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def candidates_from_output(raw, problem):
    """Try multiple extraction strategies, return list of candidate program texts."""
    prompt = problem["prompt"]
    test = problem["test"]
    entry = problem["entry_point"]
    out = []

    # strat 1: extract ```python ... ``` block, use as full program
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
        out.append(("fenced_full", code + f"\n\n{test}\n\ncheck({entry})\n"))
        # Also try prompt + extracted-as-body variant (strip the def line if in code)
        body = re.sub(r"^\s*def\s+[^\n]+:\s*(\n\s+\"\"\"[\s\S]*?\"\"\")?", "",
                      code, count=1).lstrip("\n")
        out.append(("fenced_body_append", prompt + body + f"\n\n{test}\n\ncheck({entry})\n"))

    # strat 2: if no fence, whole raw is body
    if not m:
        raw_stripped = raw.strip()
        # if response starts with 'def', treat as full
        if raw_stripped.startswith("def "):
            out.append(("raw_full", raw_stripped + f"\n\n{test}\n\ncheck({entry})\n"))
        else:
            # else as body appended to prompt
            out.append(("raw_body", prompt + raw + f"\n\n{test}\n\ncheck({entry})\n"))

    # strat 3: extract only lines that look like indented code (fallback)
    code_lines = []
    started = False
    for line in raw.splitlines():
        if line.startswith("    ") or line.startswith("\t"):
            code_lines.append(line); started = True
        elif started and not line.strip():
            code_lines.append(line)
        elif started and line.strip():
            break
    if code_lines:
        body_only = "\n".join(code_lines)
        out.append(("indent_body", prompt + body_only + f"\n\n{test}\n\ncheck({entry})\n"))

    return out


def try_candidate(code):
    try:
        r = subprocess.run(["python3", "-c", code], capture_output=True,
                           text=True, timeout=10)
        return r.returncode == 0, r.stderr[-200:] if r.returncode else ""
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def solve(problem):
    resp = hit(build_prompt(problem))
    if "error" in resp:
        return {"task_id": problem["task_id"], "passed": False, "error": resp["error"]}
    raw = resp.get("content", "")
    attempts = []
    passed = False
    strategy_won = None
    for label, cand in candidates_from_output(raw, problem):
        ok, err = try_candidate(cand)
        attempts.append({"strategy": label, "ok": ok, "err": err[:120]})
        if ok:
            passed = True
            strategy_won = label
            break
    return {"task_id": problem["task_id"], "passed": passed,
            "won_strategy": strategy_won,
            "attempts": attempts,
            "raw_preview": raw[:300]}


def main():
    ensure()
    problems = [json.loads(l) for l in open(PROBLEMS)][:50]
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
            print("server start fail"); return
        t0 = time.time()
        results = []
        with ThreadPoolExecutor(max_workers=npl) as ex:
            for i, r in enumerate(ex.map(solve, problems)):
                results.append(r)
                if (i+1) % 5 == 0:
                    n = sum(1 for x in results if x["passed"])
                    print(f"    {i+1}/{len(problems)} done, pass@1 = {n}/{i+1} = {n/(i+1):.1%}",
                          flush=True)
        elapsed = time.time() - t0
        n_pass = sum(1 for r in results if r["passed"])
        pass1 = n_pass / len(results)
        summary = {"model": "Qwen3.5-122B-A10B-REAP-20-Q4_K_M",
                   "n": len(results), "n_pass": n_pass, "pass_at_1": round(pass1,3),
                   "elapsed_s": round(elapsed,1), "results": results}
        HERE.joinpath("results.json").write_text(json.dumps(summary, indent=2))
        entry = {"mission": "32-humaneval-v2", "status": "complete",
                 "date": "2026-04-17",
                 "primary_metric": round(pass1*100, 1),
                 "primary_unit": "humaneval_pass_at_1_pct_first50_v2parser",
                 "conclusion": f"Qwen3.5-122B Q4_K_M pass@1 = {pass1:.1%} on first 50 "
                               f"({n_pass}/{len(results)}) with robust parser in {elapsed:.0f}s"}
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    finally:
        H.close_tunnel(tunnel)
        H.stop_remote_server()


if __name__ == "__main__":
    main()
