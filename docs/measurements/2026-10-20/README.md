# Measurement evidence — 2026-10-20

The same heuristic matrix as [`2026-10-19`](../2026-10-19/), re-run after the
**KI-025** fixes (the backlash dimensional error and the missing heading
metric). The purpose is one thing: **prove that the fixes did not change any
gate outcome.**

## The comparison

| | before | after |
|---|---|---|
| measurement folder | [`../2026-10-19/`](../2026-10-19/) | this folder |
| candidates solved | 60 | 60 |
| **pass flags changed** | — | **0 of 60** |
| selections changed | — | 0 of 12 runs |
| tracking error | — | mean −0.0008 cm (range −0.025 to +0.029) |
| envelope radius | — | mean **+0.0070 cm** (range −0.058 to +0.061) |
| tracking + envelope | — | mean +0.0062 cm |
| worst-case total per run | — | moved by at most 0.065 cm |

The tracking error barely moved, because it is dominated by the
discretisation mismatch described in
[`../../research/RQ-A_tracking_error.md`](../../research/RQ-A_tracking_error.md)
and not by the backlash. What changed is *where* the backlash is accounted
for: it is now a position uncertainty in the reachable set rather than a
systematic lag in the tracking metric.

Against a 5 mm tracking threshold and a 2 cm envelope threshold, a total
uncertainty that moved by 0.006 cm on average and by at most 0.065 cm in the
worst case is noise.

## What was run

Identical to the 2026-10-19 matrix:

```bash
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --uniform-dt
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --per-segment-dt
```

Missions: `example_straight`, `example_s_curve`, `example_mission_first4`,
`example_complex_mission`, `example_with_stops`, `example_with_events`.

## Caveat

Model predictions, not hardware measurements. No hardware logs exist in this
repository.
