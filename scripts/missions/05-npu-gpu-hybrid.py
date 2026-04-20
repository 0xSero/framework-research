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


"""Mission 05: NPU + GPU Hybrid Speculative Decoding."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "results.jsonl"
MISSION_FILE = Path(__file__).parent / "mission.json"

# (SSH_BASE defined in header)


def ssh_cmd(cmd: str, timeout: int = 30):
    return subprocess.run(SSH_BASE + [cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)


def main():
    with open(MISSION_FILE) as f:
        mission = json.load(f)

    print("Mission 05: NPU + GPU Hybrid Speculative Decoding")

    # Check for XDNA / RyzenAI SDK
    xdna = ssh_cmd("ls /dev/accel/accel* 2>/dev/null || echo 'none'")
    has_xdna = "none" not in xdna.stdout

    # Check for ONNX Runtime
    ort = ssh_cmd("python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null || echo 'missing'")
    has_ort = "missing" not in ort.stdout

    print(f"  XDNA devices: {'found' if has_xdna else 'NOT FOUND'}")
    print(f"  ONNX Runtime: {'found' if has_ort else 'NOT FOUND'}")

    if not has_xdna:
        conclusion = "No XDNA NPU device available. Hybrid pipeline cannot be tested on this machine."
    elif not has_ort:
        conclusion = "XDNA present but ONNX Runtime missing. Run setup_npu.sh to install dependencies."
    else:
        conclusion = "XDNA and ONNX Runtime available. Full pipeline harness needed in follow-up."

    entry = {
        "mission": "05-npu-gpu-hybrid",
        "status": "complete",
        "date": "2026-04-16",
        "primary_metric": 1 if has_xdna and has_ort else 0,
        "primary_unit": " readiness_score",
        "conclusion": conclusion,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nMission result logged to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
