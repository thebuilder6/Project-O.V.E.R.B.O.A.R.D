# Measurement evidence — 2026-10-20, after KI-025 to KI-028

The heuristic matrix and the experiment outputs, re-measured on the tree that
carries all four fixes:

- **KI-025** the gear backlash was subtracted from the commanded wheel speed
- **KI-026** the optimizer's self-consistent collocation (`--self-consistent`)
- **KI-027** the turn rate is the derivative of the interpolated heading, in
  the firmware and the validator together
- **KI-028** the validator models the controller that runs

## What is here

| File | What it is |
|---|---|
| `example_*__equal-duration.log` | the five seeds under equal-duration timing |
| `example_*__per-segment.log` | the five seeds under per-segment timing |
| `example_*__auto__sweep.log` | the default candidate set plus the accuracy-weight grid, per-segment timing |
| `example_*__full-sweep.log` | the default candidate set plus the grid, both timing modes |
| `sweep_results.json` | the parsed matrix runs |
| `full_sweep_results.json` | the parsed grid runs |
| `rq_a/*.txt` | the output of the `docs/research/rq_a/` experiments on this tree |

## What the measurements say

| | |
|---|---|
| matrix | 60 candidates: **31 pass**, 29 fail |
| failure reasons | tracking 25, motor force 6, forward-integration 13, **envelope 0** |
| largest envelope | 1.75 cm, against a 2 cm threshold |
| cells with a passing candidate | **7 of 12** (mission, timing model) |
| selected seed | `baseline` in 3 of those 7, a non-baseline seed in 4 |
| full sweep | 54 of 108 candidates pass |

Compare the same matrix before the fixes: 24 of 60 passed, 6 of the 12 cells
produced a trajectory at all, and a non-baseline seed was selected in 2 of
them.

**The reachability envelope has never bound.** Not once in 29 failures, and
not once in the 60-candidate matrix before the fixes either. The tracking
gate is the binding constraint, and RQ-A shows it was measuring a
discretisation mismatch rather than robustness.

## The tables are generated

Section 9 of
[`../../HEURISTICS_AND_VERIFICATION.md`](../../HEURISTICS_AND_VERIFICATION.md)
quotes these measurements, and its numbers are written from the JSON rather
than typed in:

```bash
python docs/research/rq_a/sync_tables.py           # rewrite the tables
python docs/research/rq_a/sync_tables.py --check   # verify only, exit 1 on drift
python docs/research/rq_a/verify_docs.py           # the rest of the checks
```

## Caveat

Model predictions, not hardware measurements. No hardware logs exist in this
repository, and the declared uncertainty ranges are test assumptions rather
than calibrated values.
