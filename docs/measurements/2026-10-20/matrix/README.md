# Candidate matrix — 2026-10-20

What the candidate set does on the current tree, with the controller that
actually runs: the 20 ms command hold, the coupled reference (KI-027) and the
validator mirroring both (KI-028).

```bash
python docs/measurements/2026-10-20/matrix/run_matrix.py
```

One log per run (`<mission>__<mode>.log`, with the command, the wall time and
the exit code, then the CLI output) and the parsed validation reports in
[`matrix_results.json`](matrix_results.json).

| file | what it is |
|---|---|
| `run_matrix.py` | the runner |
| `<mission>__default.log` | the default candidate set |
| `<mission>__sweep.log` | the default set plus the timing x accuracy-weight grid |
| `<mission>__sweep-self-consistent.log` | the same, with the self-consistent collocation (KI-026) |
| `matrix_results.json` | the parsed `##OVERBOARD-VALIDATION##` line of every run |

## The result

`pass` is the number of candidates that clear every gate; `track/env` is the
best candidate's tracking error and envelope radius in centimetres.

| mission | default | sweep | sweep + self-consistent |
|---|---|---|---|
| straight | 2/2, 0.34 / 0.30 cm | 10/10, 0.32 / 0.31 cm | 10/10, 0.32 / 0.31 cm |
| s_curve | 0/2, 1.47 / 0.73 cm | 0/10, 1.47 / 0.73 cm | 0/10, **0.84** / 0.67 cm |
| first4 | 2/2, 0.24 / 0.31 cm | 9/10, 0.21 / 0.34 cm | 9/10, **0.20** / 0.35 cm |
| complex | 1/2, 0.30 / 1.19 cm | 5/10, 0.30 / 1.19 cm | **7/10**, 0.22 / 1.57 cm |
| with_stops | 0/2, 0.98 / 0.92 cm | 0/10, 0.54 / 0.97 cm | **1/10**, 0.49 / 1.17 cm |
| with_events | 1/2, 0.46 / 0.82 cm | 5/10, 0.45 / 0.83 cm | **6/10**, 0.34 / 0.82 cm |

## What it says

**The self-consistent collocation buys passing candidates, not just smaller
errors.** Against the sweep alone it turns the complex mission from 5 to 7
passing candidates, the events mission from 5 to 6, and the stops mission from
0 to 1. Its effect on the *error* is a halving on the curved paths
(S-curve 1.47 to 0.84 cm, events 0.45 to 0.34 cm).

**The S-curve is the one mission the combination does not fix.** Its best
candidate is 0.84 cm against a 5 mm gate, so no candidate passes. The stops
mission is the same story one step better: 0.49 cm, still short. Both are
turn-heavy paths, where the plan's own discretisation is the largest term.

**The cost of the sweep is wall time.** The default set solves in 8-42 s per
mission; the sweep in 22-92 s. It is ten candidates and ten validation
passes, so the studio's switch is the right place for it.

**The envelope, not the tracking, is what the complex mission runs into.** Its
best candidate tracks to 0.22 cm (a fifth of the gate) and still has a
1.57 cm envelope against a 2 cm limit. The envelope is built from the declared
uncertainty bands in `examples/fll_choreo.json`, which are test assumptions
rather than measurements, so the band is where the robot data goes (KI-020).

## The tree these numbers describe

```text
optimizer.py            the self-consistent collocation (KI-026)
immrax_validator.py     the coupled reference and the 20 ms hold (KI-027, KI-028)
pybricks_code/          the controllers that run, all at 20 ms
```

```text
optimizer.py                        MD5 92322B2722CD99C168CC1E0527ABD9AA
immrax_validator.py                 MD5 EC0CB778103FB42092606418145F80C2
validator.py                         MD5 70950A6FAAB297AEF734FD67A67B8068
robot_model.py                       MD5 D1AF67DDA7DABDA02F116C38EF368C8B
main.py                              MD5 CEAA7FE3FED40C9A426C20A7E4517165
pybricks_code/ramsete_fast.py       MD5 329CD65E7DB84A91A4E186493B84A729
pybricks_code/ramsete.py            MD5 57FCF86BA07EAD93851FDB3DEDE327F2
pybricks_code/ramsete_compact.py    MD5 75B1C9EA82BFF37072DE43F143181254
pybricks_code/ramsete_compact_events.py MD5 4E3DA03F2494F53F2D095D264582008F
```

Environment: Windows 11, Python 3.12.10, with `casadi`, `jax`, `immrax`, `jaxopt`,
`rsplan`, `numpy`, `scipy` and `pandas` installed. The test suite was green at
the time of the run: 358 tests, 358 successes, 0 failures, 0 skipped, exit code 0.

The 2026-10-19 evidence is **not comparable**: it was measured before the
coupled reference and before the self-consistent collocation, and against a
model that did not mirror the controller in use. The earlier records in this
folder were empty runs - the runner resolved the repository root one level too
high, so every command failed with exit code 2 and no trajectory was written.
The logs here are the rerun on the tree above.
