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


"""Mission 18: Wikitext perplexity across KV cache quants on Qwen3.5-122B Q4_K_M.

Uses llama-perplexity. Runs the first ~50 chunks (keeps time reasonable) on
3 KV configs. Measures actual quality cost of Mission 01's memory wins.
"""

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"

# (SSH_BASE defined in header)

LLAMA = "$HOME/.local/opt/llama.cpp/current"
PPL_BIN = f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-perplexity"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"

# Wikitext-2 on remote — use huggingface-hosted raw
REMOTE_WIKITEXT = "$HOME/.local/share/wikitext-2-raw/wiki.test.raw"

KV_CONFIGS = [("f16", "f16"), ("q8_0", "q8_0"), ("q4_0", "q4_0")]


def ssh(cmd: str, timeout: int = 600):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        subprocess.run(SSH_BASE + ["killall -9 llama-perplexity 2>/dev/null"])
        return subprocess.CompletedProcess(args=e.cmd, returncode=-1,
                                           stdout=e.stdout or "", stderr="")


def ensure_wikitext():
    """Download wikitext-2-raw if not present."""
    check = ssh(f"test -f {REMOTE_WIKITEXT} && echo yes || echo no", 20)
    if "yes" in (check.stdout or ""):
        return True
    print("  downloading wikitext-2-raw...", flush=True)
    cmd = (
        "bash -c 'mkdir -p $HOME/.local/share/wikitext-2-raw && "
        "cd $HOME/.local/share/wikitext-2-raw && "
        "curl -sL -o wikitext-2-raw.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip "
        "&& unzip -o wikitext-2-raw.zip && "
        "mv wikitext-2-raw/wiki.test.raw . 2>/dev/null || true'"
    )
    ssh(cmd, 180)
    check = ssh(f"test -f {REMOTE_WIKITEXT} && echo yes || echo no", 20)
    return "yes" in (check.stdout or "")


def run(ck: str, cv: str):
    # chunks=50 at ctx=2048 is ~100K tokens, about 7-10 min
    remote_log = f"/tmp/mission18_{ck}_{cv}.log"
    cmd = (
        f"timeout 1200s bash -c '{PPL_BIN} -m {MODEL} -ngl 99 -fa 1 "
        f"-ctk {ck} -ctv {cv} -f {REMOTE_WIKITEXT} "
        f"-c 2048 --chunks 50 > {remote_log} 2>&1'; "
        # return just the perplexity-relevant lines plus the tail
        f"grep -E 'PPL|perplex|Final' {remote_log} | tail -n 20; "
        f"echo '---TAIL---'; tail -n 30 {remote_log}"
    )
    print(f"  {ck}/{cv} ...", flush=True)
    res = ssh(cmd, 1260)
    out = res.stdout or ""
    ppl = None
    for line in out.splitlines():
        # "Final estimate: PPL = 9.1234 +/- 0.1234"
        m = re.search(r"PPL\s*=\s*([\d.]+)", line)
        if m:
            ppl = float(m.group(1))
    errors = [l for l in out.splitlines() if re.search(r"error|failed|ggml_assert", l, re.I)][:3]
    return {"ck": ck, "cv": cv, "ppl": ppl, "errors": errors,
            "tail": out.splitlines()[-15:]}


def main():
    if not ensure_wikitext():
        print("ERROR: could not obtain wikitext; aborting")
        return

    results = []
    for ck, cv in KV_CONFIGS:
        r = run(ck, cv)
        results.append(r)
        print(f"    ppl={r['ppl']}", flush=True)

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    base = next((r for r in results if r["ck"] == "f16"), {})
    q4 = next((r for r in results if r["ck"] == "q4_0"), {})
    base_ppl = base.get("ppl")
    q4_ppl = q4.get("ppl")
    delta_pct = ((q4_ppl - base_ppl) / base_ppl * 100) if (base_ppl and q4_ppl) else None

    entry = {
        "mission": "18-perplexity-quality",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": round(delta_pct, 3) if delta_pct is not None else 0,
        "primary_unit": "ppl_delta_q4_0_vs_f16_pct",
        "conclusion": (
            f"f16 KV PPL={base_ppl}, q8_0 PPL={next((r['ppl'] for r in results if r['ck']=='q8_0'), None)}, "
            f"q4_0 PPL={q4_ppl}. q4_0 vs f16 delta={delta_pct}%"
        ),
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
