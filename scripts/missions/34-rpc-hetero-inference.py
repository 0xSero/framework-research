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


"""
Mission 34 — Heterogeneous RPC inference harness.

Sweeps llama.cpp RPC splits between the Strix Halo (Vulkan, 128 GB UMA) and
the <cuda-peer> box's RTX 3090 (CUDA, 24 GB), pinned via LAI_CUDA_VISIBLE_DEVICES
so the 4x Blackwell cards on the same host are excluded.

Node access details (hosts, users, SSH_BASE key paths, model paths) come from
environment variables — see AGENTS.md 'Mission 34' section for the list.

Driving model: Qwen3.5-122B-A10B-REAP-20-Q6_K (75.7 GB).
Baseline:     same model running single-node on the Halo (already logged
              in autoresearch/missions/results.jsonl).

This script orchestrates over SSH_BASE from the dev box. It does NOT run on
either node directly. Each sweep cell:

  1. starts rpc-server on the worker node
  2. starts llama-server on the driver node with --rpc <worker:port>
     and a --tensor-split / -ngl combination matching the cell
  3. hits /completion with a fixed prompt to measure prefill + decode
  4. tears both servers down
  5. appends a JSON line to results_raw.jsonl

Usage:
  python3 run.py                  # run the full sweep
  python3 run.py --dry-run        # print the sweep cells, no execution
  python3 run.py --cell 0         # run a single cell by index
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
RESULTS_FILE = MISSION_DIR / "results_raw.jsonl"

# ---------------------------------------------------------------------------
# Node configuration
# ---------------------------------------------------------------------------

# Node access details are read from the environment so this file does not
# need to embed personal IPs / usernames / SSH_BASE key paths. The expected
# variables (set them in your shell before running, or pass them via the
# mission-control wrapper) are documented in AGENTS.md:
#
#   HALO_HOST, HALO_USER, HALO_SSH_KEY,
#       HALO_LLAMA_DIR, HALO_MODEL_PATH
#   LAI_HOST,  LAI_USER,  LAI_SSH_KEY,
#       LAI_LLAMA_DIR,  LAI_MODEL_PATH,  LAI_CUDA_VISIBLE_DEVICES

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    if not v:
        raise SystemExit(
            f"Missing required env var {name}. See AGENTS.md 'Mission 34' "
            f"section for the list of variables this harness expects."
        )
    return v


HALO = {
    "name": "halo",
    "host": _env("HALO_HOST"),
    "user": _env("HALO_USER"),
    "key": _env("HALO_SSH_KEY"),
    "llama_dir": _env("HALO_LLAMA_DIR", "~/.local/opt/llama.cpp/current"),
    "model_path": _env("HALO_MODEL_PATH"),
    "ld_prefix": f"env LD_LIBRARY_PATH={_env('HALO_LLAMA_DIR', '~/.local/opt/llama.cpp/current')}",
    # Halo: Vulkan device, full GPU offload baseline
    "ngl_default": 99,
    "extra_args": "-fa on",
    "rpc_listen_ip": os.environ.get("HALO_RPC_LISTEN_IP", _env("HALO_HOST")),
}

_lai_visible = os.environ.get("LAI_CUDA_VISIBLE_DEVICES", "4")
LINUX_AI = {
    "name": "<cuda-peer>",
    "host": _env("LAI_HOST"),
    "user": _env("LAI_USER"),
    "key": _env("LAI_SSH_KEY"),
    "llama_dir": _env("LAI_LLAMA_DIR", "~/.local/opt/llama.cpp/current"),
    "model_path": os.environ.get("LAI_MODEL_PATH", ""),
    # Pin to the RTX 3090 only. nvcc enumerates the 3090 as index 4 on this
    # host (the four (redacted sibling CUDA device) cards are indices 0-3). Always
    # set LAI_CUDA_VISIBLE_DEVICES=4 when driving inference here or the
    # model will silently spread across the whole cluster.
    "ld_prefix": f"env CUDA_VISIBLE_DEVICES={_lai_visible} LD_LIBRARY_PATH={_env('LAI_LLAMA_DIR', '~/.local/opt/llama.cpp/current')}",
    "ngl_default": 99,
    # Flash-attention CUDA kernels were disabled in this b8779+CUDA-12.8
    # build to work around a ptxas movmatrix/PTX-ISA bug, so we cannot pass
    # -fa on on the CUDA side. The Halo (Vulkan) still uses FA.
    "extra_args": "",
    "rpc_listen_ip": os.environ.get("LAI_RPC_LISTEN_IP", _env("LAI_HOST")),
}

RPC_PORT = 50052
SERVER_PORT = 8080

# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    driver: str            # "halo" or "<cuda-peer>"
    worker: str            # the other one
    layers_on_3090: int    # used to compute --tensor-split
    ctx: int
    prompt_tokens: int     # nominal; actual measured from response

    def label(self) -> str:
        return f"{self.driver[0]}drv_{self.layers_on_3090}on3090_ctx{self.ctx}"


def build_sweep() -> list[Cell]:
    cells: list[Cell] = []
    # Halo drives, 3090 is RPC worker
    for n in (8, 16, 20, 24, 28):
        for ctx in (4096, 16384):
            cells.append(Cell("halo", "<cuda-peer>", n, ctx, prompt_tokens=512))
    # 3090 drives, Halo is RPC worker (model must be staged on <cuda-peer>)
    for n in (8, 16, 20, 24, 28):
        for ctx in (4096, 16384):
            cells.append(Cell("<cuda-peer>", "halo", n, ctx, prompt_tokens=512))
    return cells


# ---------------------------------------------------------------------------
# SSH_BASE helpers
# ---------------------------------------------------------------------------

def ssh_cmd(node: dict, remote_cmd: str, *, background: bool = False) -> list[str]:
    base = ["ssh", "-i", node["key"], "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{node['user']}@{node['host']}"]
    if background:
        wrapped = f"nohup bash -lc {shlex.quote(remote_cmd)} >/tmp/llama_run.log 2>&1 & echo PID=$!"
    else:
        wrapped = f"bash -lc {shlex.quote(remote_cmd)}"
    return base + [wrapped]


def ssh_run(node: dict, remote_cmd: str, *, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(ssh_cmd(node, remote_cmd), capture_output=True, text=True, timeout=timeout)


def ssh_kill(node: dict, pattern: str) -> None:
    ssh_run(node, f"pkill -f {shlex.quote(pattern)} || true", timeout=15)


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------

def start_rpc_server(node: dict) -> str:
    """Launch rpc-server on the worker node. Returns its PID."""
    ld = node["ld_prefix"]
    binary = f"{node['llama_dir']}/rpc-server"
    cmd = (f"{ld} {binary} -H 0.0.0.0 -p {RPC_PORT} -c")  # -c = use compute backend
    cp = ssh_run(node, f"nohup {cmd} >/tmp/rpc_server.log 2>&1 & echo PID=$!", timeout=15)
    pid = cp.stdout.strip().split("=")[-1]
    # Wait for the port to be reachable
    for _ in range(20):
        try:
            with socket.create_connection((node["host"], RPC_PORT), timeout=1):
                return pid
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"rpc-server on {node['name']} did not come up: {cp.stdout} / {cp.stderr}")


def start_llama_server(driver: dict, worker: dict, cell: Cell) -> str:
    binary = f"{driver['llama_dir']}/llama-server"
    # tensor-split: one entry per device in order: [local devices..., RPC peers...]
    # For the simple case "X layers on the 3090, rest on the Halo" we use
    # --override-tensor / --tensor-split with two slots.
    # We approximate by giving the 3090 (24 GB) a fraction proportional to layers_on_3090,
    # and the Halo (the bulk pool) the rest. Fine-grained per-layer pinning is a follow-up.
    halo_share = max(1, 80 - cell.layers_on_3090)
    split = f"{cell.layers_on_3090},{halo_share}" if driver["name"] == "halo" else f"{halo_share},{cell.layers_on_3090}"
    rpc_endpoint = f"{worker['rpc_listen_ip']}:{RPC_PORT}"

    cmd = (
        f"{driver['ld_prefix']} {binary} "
        f"-m {driver['model_path']} "
        f"--rpc {rpc_endpoint} "
        f"-ngl {driver['ngl_default']} "
        f"--tensor-split {split} "
        f"-c {cell.ctx} "
        f"--port {SERVER_PORT} --host 0.0.0.0 "
        f"{driver['extra_args']}"
    )
    cp = ssh_run(driver, f"nohup {cmd} >/tmp/llama_server.log 2>&1 & echo PID=$!", timeout=15)
    pid = cp.stdout.strip().split("=")[-1]
    # Wait for /health
    deadline = time.time() + 600  # huge model → slow load over Wi-Fi RPC
    url = f"http://{driver['host']}:{SERVER_PORT}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return pid
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"llama-server on {driver['name']} did not come up. tail logs from /tmp/llama_server.log")


def hit_completion(driver: dict, prompt_tokens: int) -> dict:
    # Simple long-form prompt to give a stable measurement window.
    prompt = ("You are a benchmark. " + "the quick brown fox jumps over the lazy dog. " * 64)
    body = json.dumps({
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0.0,
        "cache_prompt": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://{driver['host']}:{SERVER_PORT}/completion",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    elapsed = time.time() - t0
    timings = data.get("timings", {})
    return {
        "elapsed_s": elapsed,
        "tokens_predicted": data.get("tokens_predicted"),
        "tokens_evaluated": data.get("tokens_evaluated"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "prompt_per_second": timings.get("prompt_per_second"),
    }


def teardown(driver: dict, worker: dict) -> None:
    ssh_kill(driver, "llama-server")
    ssh_kill(worker, "rpc-server")
    time.sleep(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_cell(cell: Cell) -> dict:
    nodes = {"halo": HALO, "<cuda-peer>": LINUX_AI}
    driver = nodes[cell.driver]
    worker = nodes[cell.worker]
    started = datetime.utcnow().isoformat() + "Z"
    record: dict = {
        "ts": started,
        "cell": asdict(cell),
        "label": cell.label(),
    }
    try:
        teardown(driver, worker)  # clean slate
        worker_pid = start_rpc_server(worker)
        record["worker_pid"] = worker_pid
        driver_pid = start_llama_server(driver, worker, cell)
        record["driver_pid"] = driver_pid
        # Warm-up + measured request
        _warm = hit_completion(driver, cell.prompt_tokens)
        measured = hit_completion(driver, cell.prompt_tokens)
        record["timings"] = measured
        record["status"] = "ok"
    except Exception as e:
        record["status"] = "error"
        record["error"] = repr(e)
    finally:
        teardown(driver, worker)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cell", type=int, default=None, help="run a single cell index")
    args = ap.parse_args()

    cells = build_sweep()
    if args.dry_run:
        for i, c in enumerate(cells):
            print(f"{i:3d}  {c.label()}  driver={c.driver} worker={c.worker} layers_on_3090={c.layers_on_3090} ctx={c.ctx}")
        return 0

    targets = [cells[args.cell]] if args.cell is not None else cells
    for c in targets:
        print(f"\n=== {c.label()} ===", flush=True)
        rec = run_cell(c)
        print(json.dumps(rec, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
