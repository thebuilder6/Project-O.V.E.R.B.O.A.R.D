# Measurement evidence — 2026-10-19

Raw evidence for the tables in [`docs/HEURISTICS_AND_VERIFICATION.md`](../../HEURISTICS_AND_VERIFICATION.md) §9.

Every number in that document comes from one of the runs in this directory. Nothing here was hand-edited: the `*.log` files are the untouched stdout and stderr of the CLI, and the `*.json` files are the parsed `##OVERBOARD-VALIDATION##` lines.

## What was run

```bash
# The heuristic matrix: five seeds under one timing model, per mission.
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --uniform-dt
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --per-segment-dt

# The timing-mode x accuracy-weight grid on top of the default candidate set.
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --sweep
```

Missions: `example_straight`, `example_s_curve`, `example_mission_first4`,
`example_complex_mission`, `example_with_stops`, `example_with_events`.
Robot config: `examples/fll_choreo.json`. Default `-n 10` samples per segment.

## Files

| File | What it is |
|---|---|
| `example_*__equal-duration.log` | The five seeds under equal-duration timing |
| `example_*__per-segment.log` | The five seeds under per-segment timing |
| `example_*__auto__sweep.log` | The default candidate set plus the grid, forced to per-segment timing |
| `example_*__full-sweep.log` | The default candidate set plus the grid, both timing modes (the `--sweep` default) |
| `sweep_results.json` | The parsed matrix and auto-sweep runs |
| `full_sweep_results.json` | The parsed full-sweep runs |

Each log starts with the exact command, the wall time and the exit code, then the CLI output.

## Tree state

The source was edited while the first sweep was running, so the matrix was re-run afterwards against a settled tree. It was re-run a second time after the wheel force profile was required to be continuous through the waypoints the robot does not stop at (KI-024): the per-segment rows and every run that contains a per-segment candidate were re-measured, and the equal-duration rows were left alone because that branch of the optimizer is unchanged. The tree the numbers describe:

```text
optimizer.py          MD5 B871E00BD3C8C3CE506B716DF8596CB0
main.py               MD5 3F0325ADEAF492D48D0E724FF48482B0
immrax_validator.py   MD5 C7AAFF38D1E3FDEE6ECECDCE198D5873
validator.py          MD5 70950A6FAAB297AEF734FD67A67B8068
robot_model.py        MD5 2B798C532F02BAE7F77DBA0A7CBE9C81
multiverse_optimizer.py MD5 D67E88CEA2A2B072B5B60F3CF52354F2
jax_optimizer.py      MD5 B6F2DD0ABAED18482A26F81F635D1ADF
path_planning.py      MD5 8E9D030AD66453295F7E22F898CD6B4E
```

Environment: Windows 11, Python 3.12.10, with `casadi`, `jax`, `immrax`, `jaxopt`,
`rsplan`, `numpy`, `scipy` and `pandas` installed. The test suite was green at
the time of the run.

## Reading a log

```text
# command: ...
# wall_seconds: ...
# exit_code: ...

  Candidate 1: baseline, equal-duration
  ...
  ##OVERBOARD-VALIDATION## {"selected": 1, "candidates": [ ... ]}
```

The `##OVERBOARD-VALIDATION##` line is the machine-readable summary: one entry per candidate with its style, timing model, duration, envelope radius, tracking error, pass flag and failure reasons, plus a per-segment breakdown. `selected` is the index of the chosen candidate, or `null` when nothing passed.

## Caveat

These are **model predictions**, not hardware measurements. No hardware logs exist in this repository. The uncertainty ranges in `examples/fll_choreo.json` are declared test assumptions, not calibrated values for a real robot.
