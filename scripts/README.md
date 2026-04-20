# scripts/

Harnesses that produced the benchmark data. Organised one per mission
under `scripts/missions/`.

## Conventions

- Every script is env-var driven. No network addresses, key paths, or
  usernames are hard-coded. See
  [`../documentation/reproducing.md`](../documentation/reproducing.md)
  for the full list of env vars.
- Scripts post to a running `llama-server` on the driver node. They do
  not start or stop the server — that's an operator task.
- Scripts write their results to `benchmarks/missions/<id>/results.json`
  by convention. Operators running reproduction runs should adjust the
  output path if they want to preserve the published data.

## Running

```bash
# Minimum env for most Phase 1/2 missions
export DRIVER_HOST=user@host
export DRIVER_KEY=~/.ssh/id_ed25519
export DRIVER_BIN=/path/to/llama.cpp/build
export DRIVER_MODELS=/path/to/gguf
export DRIVER_PORT=8080

# Example: parallel throughput mission
python3 scripts/missions/09-parallel-throughput.py
```

For the RPC mission (34) you also need `WORKER_HOST`, `WORKER_KEY`,
`WORKER_BIN`, and `WORKER_PORT`.

## Porting notes

Some scripts reference helper logic that was originally in a local
`_http_helper.py` module. The ported scripts inline that logic or use
small subprocess calls to ssh, depending on which mission. Not every
path the originals exercised was ported — scripts labelled
`*-v2.py` are follow-up reruns and may not be needed to regenerate the
headline numbers.
