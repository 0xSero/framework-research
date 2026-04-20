# benchmarks/

Raw benchmark outputs per mission.

## Layout

```
benchmarks/
├── summary.csv               one row per mission-result entry
├── results.jsonl             append-only log of all primary metrics
└── missions/<id>/
    ├── mission.json          hypothesis + success criteria
    └── results*.json(l)      raw measurements (per-cell)
```

## Using the data

- **`summary.csv`** is the quickest overview. One row per mission with
  primary metric, unit, and a one-line conclusion.
- **`results.jsonl`** is machine-readable history. Easy to filter with
  `jq`:
  ```bash
  jq -r 'select(.mission=="09-parallel-throughput") | [.date,.primary_metric,.conclusion] | @csv' \
    benchmarks/results.jsonl
  ```
- **`missions/<id>/results.json`** (or `results_raw.jsonl`) contains the
  raw per-cell measurements — every point in every sweep — that the
  headline number summarises.

## Units

Where a mission's `primary_unit` is ambiguous, the corresponding
writeup under `documentation/missions/<id>.md` defines it. Common units:

| Unit                          | Meaning                                        |
|-------------------------------|------------------------------------------------|
| `tg_t_s` / `decode_t_s`       | decode tokens per second                        |
| `pp_t_s` / `prefill_t_s`      | prefill tokens per second                       |
| `tg_speedup_x`                | decode speedup factor vs. baseline              |
| `ppl_delta_*_pct`             | perplexity delta in percent                     |
| `pass_at_1_pct`               | HumanEval pass@1 (first-attempt correctness)    |
| `needles_found_of_5`          | retrieval recall over the 5-case needle suite   |
| `tg_drift_pct_first3_to_last3`| thermal drift over a sustained window           |
