# Candidate matrix — 2026-10-20

What the candidate set does on the current tree, with the controller that
actually runs: the 20 ms command hold, the coupled reference (KI-027) and the
validator mirroring both (KI-028).

```bash
python docs/measurements/2026-10-20/matrix/run_matrix.py
python docs/measurements/2026-10-20/matrix/analyse_matrix.py
```

One log per run (`<mission>__<mode>.log`, with the command, the wall time and
the exit code, then the CLI output) and the parsed validation reports in
[`matrix_results.json`](matrix_results.json).

| file | what it is |
|---|---|
| `run_matrix.py` | the runner |
| `analyse_matrix.py` | the analysis: Q3 and Q10 from the same file. `--table` prints the table below, `--tree` the fingerprint of the tree |
| `<mission>__default.log` | the default candidate set: two timing models of the baseline seed |
| `<mission>__seeds.log` | the default set plus the other initial-guess seeds (`--candidates 5`), so the multi-start is measured at all |
| `<mission>__sweep.log` | the default set plus the timing x accuracy-weight grid |
| `<mission>__sweep-self-consistent.log` | the same, with the self-consistent collocation (KI-026) |
| `matrix_results.json` | the parsed `##OVERBOARD-VALIDATION##` line of every run |

## The result

`pass` is the number of candidates that clear every gate; `track/env` is the
best candidate's tracking error and envelope radius in centimetres. The table
is generated: `analyse_matrix.py --table`.

| mission | default | seeds | sweep | sweep + self-consistent |
|---|---|---|---|---|
| straight | 2/2, 0.34 / 0.30 cm | 5/5, 0.34 / 0.30 cm | 10/10, 0.32 / 0.31 cm | 10/10, 0.32 / 0.31 cm |
| s_curve | 0/2, 1.47 / 0.73 cm | 0/5, 1.47 / 0.73 cm | 0/10, 1.47 / 0.73 cm | 0/10, **0.84** / 0.67 cm |
| first4 | 2/2, 0.24 / 0.31 cm | 5/5, 0.24 / 0.31 cm | 9/10, 0.21 / 0.34 cm | 9/10, **0.20** / 0.35 cm |
| complex | 1/2, 0.30 / 1.19 cm | 3/5, 0.30 / 1.19 cm | 5/10, 0.30 / 1.19 cm | **7/10**, 0.22 / 1.57 cm |
| with_stops | 0/2, 0.98 / 0.92 cm | 0/5, 0.93 / 0.92 cm | 0/10, 0.54 / 0.97 cm | **1/10**, 0.49 / 1.17 cm |
| with_events | 1/2, 0.46 / 0.82 cm | 2/5, 0.46 / 0.82 cm | 5/10, 0.45 / 0.83 cm | **6/10**, 0.34 / 0.82 cm |

## What it says

### The insurance comes from the timing model, not from a seed (Q3)

**With one candidate** — the baseline seed, equal-duration timing — only
three of the six missions produce a passing trajectory: straight, first4 and
the complex mission. The S-curve, the stops mission and the events mission
do not.

**The second candidate** — the same seed with the per-segment timing model —
rescues **one mission in six** (the events mission, 4.416 s). That is the
whole of the insurance the default set buys, and it comes from comparing two
*timing models*, not from trying a different starting point.

**The seeds** add no passes at all: the `seeds` column has the same four
missions with a passer as the `default` column. What they do buy is
**speed**, and only where the seeds reach a different optimum:

| mission | the baseline's best passer | the seeds' best passer |
|---|---|---|
| complex | 6.068 s (baseline, equal-duration) | **5.395 s** (rolling, per-segment) |
| first4 | 1.833 s (baseline, per-segment) | **1.796 s** (rolling, per-segment) |
| events | 4.416 s | 4.415 s (1 ms) |
| straight | 2.377 s | 2.377 s (identical) |

On the complex mission the seeds find a passing trajectory **11 % faster**
than any baseline candidate. On the straight and the events missions the
seeds land on top of the baseline to the millisecond, and on the S-curve they
are worse (3.759-3.788 s against the baseline's 3.667 s).

**The cost.** The default set takes 5.8-30.8 s per mission, the seeds
9.2-41.8 s, the sweep 15.2-66.3 s. So: one extra candidate buys a passing
trajectory on 17 % of missions, at about 50 % more wall time; three extra
candidates buy 11 % of the run time on one mission, at about 40 % more wall
time; and the sweep's eight extra weight-grid candidates buy **no** extra
passes on these six missions at all.

### "Smallest envelope" is the wrong selection objective (Q10)

The envelope is almost a proxy for the duration: over the 162 candidates,
`corr(duration, envelope) = +0.97`, while `corr(envelope, tracking) = +0.33`.
The tube grows with the clock — the backlash term widens the reachable set by
`backlash/2 · t` — so "smallest envelope" is, in practice, "shortest run"
with a safety margin that is really elapsed time.

That is not the same thing as accuracy, and on the complex mission it costs
real time. With the seeds in the set:

| rule | what it picks on the complex mission |
|---|---|
| **A** smallest envelope (the rule today) | baseline, equal-duration, **6.068 s**, envelope 1.19 cm |
| B smallest tracking error | baseline, equal-duration, 6.068 s |
| C smallest excess over the local limit | baseline, equal-duration, 6.068 s |
| **D** smallest duration | rolling, per-segment, **5.395 s**, envelope 1.25 cm |

The envelope rule gives up **0.67 s (11 %)** of the run to hold a 0.6 mm
difference in a number that is 8 mm inside its 20 mm limit. The rule that
tracks the duration picks the faster trajectory.

The alternative Q10 proposed, the tracking error, is worse: it differs from
the current rule in 8 of the 17 groups with a passer, and it is **slower in
7 of them and faster in 1** — because the tracking error is nearly
independent of the duration, so preferring it means preferring a longer run.

**The envelope never rejects anything here.** Of the 162 candidates, 83 pass
and the failures are 71 tracking, 52 integration, 8 motor force and 6
constraint. Not one candidate failed on the envelope threshold, and every
segment limit in the matrix is the default 2 cm, so the absolute envelope and
the per-sample excess over the local limit are the same ordering. A mission
with a tight `error_threshold` is needed for them to disagree — and a fitted
model, which blows the envelope up far more than the tracking error
(RQ-A F5), is needed for the envelope to bind at all.

### The self-consistent collocation is the only thing that solves the stops mission

With `--self-consistent`, the stops mission gets a passing candidate (6.437 s
at accuracy weight 0.01) and the complex mission flips from the
equal-duration model to the per-segment one. The S-curve still fails, at
0.84 cm against a 5 mm gate.

### The sweep's weight grid does not pay here

The `sweep` column adds eight candidates and buys no extra passes on any of
the six missions, at two to three times the wall time. It paid once, under
`--self-consistent`, on the stops mission.

## The tree these numbers describe

```text
{generated by: python docs/measurements/2026-10-20/matrix/analyse_matrix.py --tree}
```

Environment: Windows 11, Python 3.12.10, with `casadi`, `jax`, `immrax`, `jaxopt`,
`rsplan`, `numpy`, `scipy` and `pandas` installed.

The 2026-10-19 evidence is **not comparable**: it was measured before the
coupled reference and before the self-consistent collocation, and against a
model that did not mirror the controller in use. The earlier records in this
folder were empty runs - the runner resolved the repository root one level too
high, so every command failed with exit code 2 and no trajectory was written.
The logs here are the rerun on the tree above.
