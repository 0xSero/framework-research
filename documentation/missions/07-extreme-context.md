# 07-extreme-context

## Hypothesis

With q4_0/q4_0 KV on 128GB UMA we can push the 122B MoE past 131K — targeting 262K, 524K, and 1M contexts — and identify whether FLASH_ATTN_EXT scaling or plain memory is the hard wall.

## Success criteria

- {'max_ctx_reached': 524288, 'pp_floor_t_s': 40}

## Raw data

- Mission spec: `benchmarks/missions/07-extreme-context/mission.json`
- Results: `benchmarks/missions/07-extreme-context/results.json`
- Harness: `scripts/missions/07-extreme-context.py`
