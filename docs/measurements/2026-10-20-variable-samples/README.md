# Measurement evidence — the heuristic matrix runner

The runner for the tables in
[`../../HEURISTICS_AND_VERIFICATION.md`](../../HEURISTICS_AND_VERIFICATION.md)
§9, re-measured on whatever tree it is run against. It runs the same four
configurations the 2026-10-19 and 2026-10-20-post-ki028 evidence used, so the
tables can be regenerated against a changed planner:

| mode | command | what it is |
|---|---|---|
| `equal-duration` | `--candidates 5 --uniform-dt` | the five seeds under one shared `dt` |
| `per-segment` | `--candidates 5 --per-segment-dt` | the five seeds under one `dt` per segment |
| `auto__sweep` | `--candidates 5 --per-segment-dt --sweep` | the default set plus the timing x weight grid |
| `full-sweep` | `--candidates 5 --uniform-dt --sweep` | the default set plus the grid, both timing modes |

```bash
python docs/measurements/2026-10-20-variable-samples/run_sweep.py
```

It writes one log per run (the command, the wall time and the exit code, then
the CLI output), `sweep_results.json` (the matrix runs), `full_sweep_results.json`
(the grid runs) and `TREE.md5` (the fingerprint of the modules the numbers
depend on). `docs/research/rq_a/sync_tables.py` then rewrites §9's tables from
the JSON, and `--check` verifies that every number still matches.

## The tree has to be still

The runner takes the tree's fingerprint **before and after** the run and says
so if they differ:

```text
!! THE TREE CHANGED DURING THE RUN: optimizer.py
   The numbers above straddle two trees and cannot be quoted as one
   measurement. Re-run the measurement on a settled tree.
```

That is not a formality. The working tree is edited while a measurement runs,
and a run that straddles two planners produces numbers that belong to neither.
The same rule is why `research/harness.py` records a `tree` token with every
record.

## No numbers here yet

The folder holds the runner only. The runs it made while the tree was being
edited were abandoned, because the guard above caught them: they straddled two
planners and could not be quoted. The tables in §9 still quote the
2026-10-20-post-ki028 evidence, and §9 says which collocation those numbers
describe.
