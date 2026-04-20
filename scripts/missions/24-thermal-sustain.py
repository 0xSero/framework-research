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


"""Mission 24: 1-hour sustained throughput with periodic thermal sampling.

Loops llama-bench at pp512/tg128 for 60 minutes. Logs each iteration's tg and
reads GPU/CPU temp from `sensors` every 30s. Exposes whether sustained load
causes throttling.
"""

import json, os, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
# (SSH_BASE defined in header)
LLAMA = "$HOME/.local/opt/llama.cpp/current"
BENCH = f"env LD_LIBRARY_PATH={LLAMA} {LLAMA}/llama-bench"
MODEL = "$HOME/.local/share/models/gguf/Qwen3.5-122B-A10B-REAP-20-Q4_K_M.gguf"

DURATION_MINS = 60
ITERATION_PP = 4096
ITERATION_TG = 256


def ssh(cmd, timeout=120):
    try:
        return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args=cmd, returncode=-1, stdout="", stderr="")


def read_temps():
    out = ssh("sensors 2>/dev/null", 15).stdout or ""
    temps = {}
    current = None
    for line in out.splitlines():
        if line and not line.startswith(" "):
            current = line.strip().split()[0]
        m = re.search(r"(?:Tctl|Tdie|edge|junction|Package id \d+)[:=]\s*\+?([\d.]+)", line)
        if m and current:
            temps[f"{current}_{line.strip().split(':')[0].strip()}"] = float(m.group(1))
    return temps


def run_iter():
    cmd = (f"{BENCH} -m {MODEL} -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 "
           f"-ub 2048 -b 2048 -p {ITERATION_PP} -n {ITERATION_TG} -r 1 --no-warmup")
    res = ssh(cmd, 300)
    out = res.stdout or ""
    pp = tg = None
    for line in out.splitlines():
        m = re.search(rf"pp{ITERATION_PP}\s*\|\s*([\d.]+)", line)
        if m: pp = float(m.group(1))
        m = re.search(rf"tg{ITERATION_TG}\s*\|\s*([\d.]+)", line)
        if m: tg = float(m.group(1))
    return pp, tg


def main():
    samples = []
    t0 = time.time()
    end = t0 + DURATION_MINS * 60
    iter_n = 0
    while time.time() < end:
        iter_n += 1
        t_iter = time.time()
        temps_before = read_temps()
        pp, tg = run_iter()
        temps_after = read_temps()
        elapsed = time.time() - t_iter
        total_elapsed = time.time() - t0
        sample = {
            "iter": iter_n,
            "t_minutes": round(total_elapsed / 60, 2),
            "pp": pp, "tg": tg,
            "iter_s": round(elapsed, 1),
            "temps_before": temps_before,
            "temps_after": temps_after,
        }
        samples.append(sample)
        print(f"  iter{iter_n} t={total_elapsed/60:.1f}min pp={pp} tg={tg} "
              f"temps={temps_after}", flush=True)
        Path(__file__).parent.joinpath("results.json").write_text(
            json.dumps(samples, indent=2))
        # brief rest so the thermal between iters is visible; ~2s
        time.sleep(2)

    # Compute degradation: first 3 iters avg vs last 3 iters avg
    if len(samples) >= 6:
        first_tg = sum(s["tg"] for s in samples[:3] if s["tg"]) / 3
        last_tg  = sum(s["tg"] for s in samples[-3:] if s["tg"]) / 3
        delta = (last_tg - first_tg) / first_tg * 100 if first_tg else 0
    else:
        delta = 0

    entry = {"mission": "24-thermal-sustain", "status": "complete", "date": "2026-04-17",
             "primary_metric": round(delta, 3),
             "primary_unit": "tg_drift_pct_first3_to_last3",
             "conclusion": f"{len(samples)} iters over {DURATION_MINS} min. "
                           f"tg drift {delta:+.2f}% (first avg→last avg)"}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
