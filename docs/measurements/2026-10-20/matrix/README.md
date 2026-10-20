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
| `run_matrix.py` | the runner. It records the tree's fingerprint with the results, so the evidence says which tree it was taken on |
| `analyse_matrix.py` | the analysis: Q3 and Q10 from the same file. `--table` prints the table below, `--tree` the fingerprint |
| `<mission>__default.log` | the default candidate set: two timing models of the baseline seed |
| `<mission>__seeds.log` | the default set plus the other initial-guess seeds (`--candidates 5`), so the multi-start is measured at all |
| `<mission>__refine.log` | the default set plus the Multi-Verse pipeline's refined trajectory as a candidate (`--refine`) |
| `<mission>__sweep.log` | the default set plus the timing x accuracy-weight grid |
| `<mission>__sweep-self-consistent.log` | the same, with the self-consistent collocation (KI-026) |
| `matrix_results.json` | the parsed `##OVERBOARD-VALIDATION##` line of every run, and the tree's fingerprint |

## The result

`pass` is the number of candidates that clear every gate; `track/env` is the
best candidate's tracking error and envelope radius in centimetres. The table
is generated: `analyse_matrix.py --table`.

| mission | default | seeds | refine | sweep | sweep + self-consistent |
|---|---|---|---|---|---|
| straight | 2/2, 0.34 / 0.30 cm | 5/5, 0.34 / 0.30 cm | 7/7, 0.32 / 0.31 cm | 10/10, 0.32 / 0.31 cm | 10/10, 0.32 / 0.31 cm |
| s_curve | 0/2, 1.47 / 0.73 cm | 0/5, 1.47 / 0.73 cm | 0/7, 1.33 / 0.68 cm | 0/10, 1.47 / 0.73 cm | 0/10, **0.84** / 0.67 cm |
| first4 | 2/2, 0.24 / 0.31 cm | 5/5, 0.24 / 0.31 cm | 6/7, 0.24 / 0.31 cm | 9/10, 0.21 / 0.34 cm | 9/10, **0.20** / 0.35 cm |
| complex | 1/2, 0.30 / 1.19 cm | 3/5, 0.30 / 1.19 cm | 3/7, 0.30 / 1.19 cm | 5/10, 0.30 / 1.19 cm | **7/10**, 0.22 / 1.57 cm |
| with_stops | 0/2, 0.98 / 0.92 cm | 0/5, 0.93 / 0.92 cm | 0/7, 0.98 / 0.92 cm | 0/10, 0.54 / 0.97 cm | **1/10**, 0.49 / 1.17 cm |
| with_events | 1/2, 0.46 / 0.82 cm | 2/5, 0.46 / 0.82 cm | 6/7, 0.45 / 0.83 cm | 5/10, 0.45 / 0.83 cm | **6/10**, 0.34 / 0.82 cm |

## What it says

### The insurance is one mission in six, and it comes from the timing model (Q3)

**With one candidate** — the baseline seed, equal-duration timing — only
three of the six missions produce a passing trajectory: straight, first4 and
the complex mission. The S-curve, the stops mission and the events mission
do not.

**The second candidate** — the same seed with the per-segment timing model —
rescues **one mission in six** (the events mission, 4.416 s). That is the
whole of the insurance the default set buys, and it comes from comparing two
*timing models*, not from trying a different starting point.

**Nothing else adds a pass.** The seeds (`--candidates 5`) and the refined
trajectory (`--refine`) both leave the count at four of six. What they buy
is **speed**, and only where they reach a different optimum:

| mission | the baseline's best passer | the seeds' | the refined guess's |
|---|---|---|---|
| complex | 6.068 s | 5.395 s (rolling, per-segment) | **4.936 s** (refined, weight 0.001) |
| first4 | 1.833 s | 1.796 s (rolling, per-segment) | **1.700 s** (refined) |
| events | 4.416 s | 4.415 s | 4.416 s |
| straight | 2.377 s | 2.377 s (identical) | 2.377 s |

On the complex mission the refined trajectory finds a passing trajectory
**19 % faster** than any baseline candidate, and the structured seeds 11 %.
On the straight and the events missions the seeds land on top of the baseline
to the millisecond, and on the S-curve they are worse (3.759-3.788 s against
the baseline's 3.667 s).

**The cost.** The default set takes 6.1-34.8 s per mission, the seeds
10.4-50.3 s, the refined 15.9-54.7 s, the sweep 17.7-67.1 s. So: one extra
candidate buys a passing trajectory on 17 % of missions, at about 50 % more
wall time; the extra seeds and the refined trajectory buy 11-19 % of the run
time on one or two missions, at 60-80 % more wall time; and the sweep's
eight extra weight-grid candidates buy **no** extra passes on these six
missions at all.

### "Smallest envelope" is the wrong selection objective (Q10)

The envelope is almost a proxy for the duration: over the 204 candidates,
`corr(duration, envelope) = +0.97`, while `corr(envelope, tracking) = +0.35`.
The tube grows with the clock - the backlash term widens the reachable set by
`backlash/2 * t` - so "smallest envelope" is, in practice, "shortest run"
with a safety margin that is really elapsed time.

Five rules were run offline over the same measurements (only the comparison
changes, nothing is re-solved):

| rule | what it ranks by |
|---|---|
| **A** | the envelope, then the duration (the rule today) |
| B | the tracking error, then the duration |
| C | the excess over the local limit, then the duration |
| D | the duration |
| E | the envelope, then the tracking error |

| rule | differs from A | slower | faster |
|---|---:|---:|---:|
| B tracking error | 12 of 21 groups | **11** | 1 |
| C excess over the limit | 3 of 21 | 2 | 1 |
| **D duration** | 6 of 21 | **0** | **6** |
| E envelope, then tracking | 0 of 21 | 0 | 0 |

**The tracking error is the wrong objective**: it is nearly independent of
the duration, so preferring it is slower in 11 of the 21 groups and faster in
one. On the complex mission under `--sweep --self-consistent` it picks a
7.106 s trajectory where the current rule picks 5.200 s.

**The duration is the only rule that is never worse.** It differs from the
current rule in 6 of the 21 groups and is faster in every one of them. The
sharpest case is the complex mission with the seeds in the set: the current
rule picks the 6.068 s candidate (envelope 1.19 cm) over the 5.395 s one
(envelope 1.25 cm), giving up **0.67 s - 11 % of the run** to hold 0.6 mm of
envelope, with 8 mm of margin on both. Where the refined candidate is in the
set the two rules agree, because that candidate is both faster *and* narrower
(4.936 s, envelope 1.14 cm).

**The per-sample excess is the same choice as the envelope here.** It differs
only where the two candidates have the same duration, because every segment
limit in the matrix is the default 2 cm, so the absolute envelope and the
excess over the local limit are the same ordering. A mission with a tight
`error_threshold` is needed for them to disagree.

**The envelope never rejects anything here.** Of the 204 candidates, 105
pass, and the failures are 90 tracking, 75 integration, 9 motor force and 7
constraint (a candidate can carry more than one reason). Not one candidate
failed on the envelope threshold. A fitted model, which blows the envelope up
far more than the tracking error (RQ-A F5), is needed for the envelope to
bind at all - and that is when the choice of objective starts to matter.

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

The runner records the fingerprint with the results, so the evidence carries
the tree it was taken on. The current tree is printed by:

```bash
python docs/measurements/2026-10-20/matrix/analyse_matrix.py --tree
```

The list is generated, because a hand-kept one goes stale the moment
anything is edited.

Environment: Windows 11, Python 3.12.10, with `casadi`, `jax`, `immrax`, `jaxopt`,
`rsplan`, `numpy`, `scipy` and `pandas` installed.

The 2026-10-19 evidence is **not comparable**: it was measured before the
coupled reference and before the self-consistent collocation, and against a
model that did not mirror the controller in use. The earlier records in this
folder were empty runs - the runner resolved the repository root one level too
high, so every command failed with exit code 2 and no trajectory was written.
The logs here are the rerun on the tree above.
