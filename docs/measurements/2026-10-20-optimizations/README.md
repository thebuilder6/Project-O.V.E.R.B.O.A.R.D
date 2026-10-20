# The optimization variants, compared

The optimizer grew a set of switches in one day: the collocation, the
sample layout, the path splitting, the cost metric, the shape of the force
bound, the weight on the force, and the timing model. Each one changes the
plan, and they have not been compared with each other on the same missions
with the same gates.

This folder holds the comparison: the harness, the numbers, and the figures.

## What is compared

One switch at a time from the shipped default. A full cross product is
2**8, so the switches are turned on one at a time and a switch's effect is
then its own. The combinations are a follow-up.

| variant | the switch it turns |
|---|---|
| `default` | the shipped default |
| `trapezoid` | `self_consistent=False` - the old collocation, where the samples are the ends of the chord |
| `uniform-samples` | `proportional_samples=False` - the same number of samples on every segment |
| `joint-nlp` | `split_at_boundaries=False` - one NLP for the whole path |
| `discretization-cost` | `cost_metric="discretization"` - prices the three mismatches the tracking gate measures instead of the wheel jerk |
| `epigraph-force` | `epigraph_constraints=True` - the force bound as four linear inequalities |
| `force-weight-0.05` | `force_weight=0.05` - the wheel force is priced, lightly |
| `force-weight-0.5` | `force_weight=0.5` - the wheel force is priced, strongly |

Each is run under **both** timing models, because the command line's
candidate set runs both and keeps the better one. So the table is
variant x timing, and the scorecard shows the better of the two per cell.

## How it is scored

With the **production** gate chain, not a copy of it:

- `main._immrax_failure_reasons` and `main._cpu_failure_reasons` for the reasons,
- `validator.validate_samples` for the CPU audit,
- `main._segment_limit` for the local envelope threshold, and
  `main._waypoint_boundaries` / `main._waypoint_of` for the sample-to-waypoint
  mapping, which is **not** `i // samples_per_segment` any more now that the
  sample budget is split by arc length.

A number here means what it means in `main.py`.

## The figures

- `scorecard.png` - every mission against every variant: the gate outcome in
  a grid, and the duration, the envelope and the tracking error as bars.
- `<mission>.png` - the six-panel review figure for the default and the three
  best alternatives: the path, the reachability tube, the tracking error, the
  time spent per segment, the wheel force, and the numbers.

## The tree has to be still

The harness takes the tree's fingerprint before and after the run and says
so if they differ:

```text
!! THE TREE CHANGED DURING THE RUN: optimizer.py
   The numbers above straddle two trees and cannot be quoted as one
   measurement. Re-run the measurement on a settled tree.
```

The working tree is edited while a measurement runs, and a run that
straddles two planners produces numbers that belong to neither. The same
rule is why `research/harness.py` records a `tree` token with every record.

## How to reproduce it

```bash
python docs/measurements/2026-10-20-optimizations/compare_optimizations.py
```

`--missions=example_s_curve` and `--variants=default,trapezoid` narrow the
set, so the harness can be exercised on one mission before the whole matrix
is run.
