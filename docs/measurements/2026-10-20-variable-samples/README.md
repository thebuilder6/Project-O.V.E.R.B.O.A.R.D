# Measurement evidence — the heuristic matrix, re-measured

The evidence for the tables in
[`../../HEURISTICS_AND_VERIFICATION.md`](../../HEURISTICS_AND_VERIFICATION.md)
§9, measured on the tree that carries the arc-length sample layout, the
self-consistent collocation on by default (KI-026, KI-051, KI-052) and the
consolidated firmware (KI-028).

## What is here

| File | What it is |
|---|---|
| `run_sweep.py` | the runner: the four configurations, one log per run, and the tree's fingerprint |
| `example_*__equal-duration.log` | the five seeds under one shared `dt` |
| `example_*__per-segment.log` | the five seeds under one `dt` per segment |
| `example_*__auto__sweep.log` | the default set plus the timing x weight grid, per-segment timing |
| `example_*__full-sweep.log` | the default set plus the grid, both timing modes |
| `sweep_results.json` | the parsed matrix runs |
| `full_sweep_results.json` | the parsed grid runs |
| `TREE.md5` | the fingerprint of the modules the numbers depend on |

```bash
python docs/measurements/2026-10-20-variable-samples/run_sweep.py
python docs/research/rq_a/sync_tables.py           # rewrite the tables
python docs/research/rq_a/sync_tables.py --check   # verify only
```

The runner takes the tree's fingerprint **before and after** the run and
refuses a measurement whose tree changed underneath it:

```text
!! THE TREE CHANGED DURING THE RUN: optimizer.py
   The numbers above straddle two trees and cannot be quoted as one
   measurement. Re-run the measurement on a settled tree.
```

That is not a formality. The working tree is edited while a measurement runs,
and a run that straddles two planners produces numbers that belong to neither.
Three runs were abandoned for exactly that reason before this one, which was
taken on a tree that did not move.

**Two modules changed after this run** (`robot_model.py` and `main.py`; the
fingerprint in `TREE.md5` is the tree the run was taken on). Neither change
touches the numbers: the planner (`optimizer.py`) and both gates
(`immrax_validator.py`, `validator.py`) are byte-identical to the ones the
run measured, the robot model's new entries are configuration defaults that
are inert unless a measured fit is supplied, and the sweep does not pass
`--sysid-fit`. Re-run the measurement when the tree is still if you want the
fingerprint to match the tree exactly.

## What the numbers say

| | |
|---|---|
| matrix | 60 candidates: **26 pass**, 34 fail |
| failure reasons | tracking 28, motor force 11, **forward-integration 0**, **envelope 0** |
| largest envelope | 1.72 cm, against a 2 cm threshold |
| cells with a passing candidate | **7 of 12** (mission, timing model) |
| selected seed | `baseline` in 1 of those 7, a non-baseline seed in 6 |
| full sweep | 24 of 54 candidates pass |

**The forward-integration failures are gone.** They went with the
self-consistent collocation: the plan no longer disagrees with its own
integration, so the CPU audit's check has nothing left to catch. The tracking
gate is still what rejects candidates, and the envelope still never binds.

**The seeds earn more of their keep than they used to.** The `baseline` seed
passes in 3 of the 12 cells and is rescued in 4 of them - first4 under both
timing models, the complex mission under equal-duration, the events mission
under per-segment.

**These numbers are not comparable with the 2026-10-20-post-ki028 evidence**,
which was taken with the trapezoidal collocation and a uniform sample count.
The two changes are described in §5.7 and §5.8 of the heuristics document.

## Caveat

Model predictions, not hardware measurements. No hardware logs exist in this
repository, and the declared uncertainty ranges in `examples/fll_choreo.json`
are test assumptions rather than calibrated values.
