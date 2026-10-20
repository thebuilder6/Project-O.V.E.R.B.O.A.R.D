# Measurements: matrix.jsonl

Source: `research/results/matrix.jsonl`

- config: `examples/fll_choreo.json`
- spec set: `matrix`, samples: 10, base weight: 0.0
- started: 2026-10-20 00:35:16
- tree hashes:
  - `optimizer.py`: `B871E00BD3C8C3CE506B716DF8596CB0`
  - `main.py`: `E97A773F495158A633D6FC4A7AFCC795`
  - `immrax_validator.py`: `C7AAFF38D1E3FDEE6ECECDCE198D5873`
  - `validator.py`: `70950A6FAAB297AEF734FD67A67B8068`
  - `robot_model.py`: `2B798C532F02BAE7F77DBA0A7CBE9C81`
  - `multiverse_optimizer.py`: `D67E88CEA2A2B072B5B60F3CF52354F2`

Every number below is a model prediction. The declared uncertainty ranges are test assumptions, not measurements, and no hardware logs exist in the repository.
records: 672  missions: 32
config: examples/fll_choreo.json  samples: 10  specs: matrix  started: 2026-10-20 00:35:16
  optimizer.py             B871E00BD3C8C3CE506B716DF8596CB0
  main.py                  E97A773F495158A633D6FC4A7AFCC795
  immrax_validator.py      C7AAFF38D1E3FDEE6ECECDCE198D5873
  validator.py             70950A6FAAB297AEF734FD67A67B8068
  robot_model.py           2B798C532F02BAE7F77DBA0A7CBE9C81
  multiverse_optimizer.py  D67E88CEA2A2B072B5B60F3CF52354F2
converged: 668/672 (99.4%)
passing:   152/672 (22.6%)
duration:  median 5.592 s  mean 6.877 s
envelope:  median 1.06 cm
tracking:  median 0.72 cm
solve wall: median 0.51 s  total 13.6 min

## Per seed and timing model

(stock seeds only; the extra guess families are compared separately)
| seed | timing | n | pass | pass rate (95% CI) | median dur | median env cm | median trk cm |
|---|---|---:|---:|---|---:|---:|---:|
| baseline | equal-duration | 32 | 5 | 16% (7-32%) | 6.127 | 1.10 | 0.74 |
| baseline | per-segment | 32 | 6 | 19% (9-35%) | 5.004 | 0.97 | 0.77 |
| bow_left | equal-duration | 32 | 8 | 25% (13-42%) | 6.208 | 1.09 | 0.60 |
| bow_left | per-segment | 32 | 8 | 25% (13-42%) | 5.015 | 0.93 | 0.74 |
| bow_right | equal-duration | 32 | 8 | 25% (13-42%) | 6.208 | 1.08 | 0.59 |
| bow_right | per-segment | 32 | 6 | 19% (9-35%) | 4.868 | 0.93 | 0.82 |
| rolling | equal-duration | 32 | 9 | 28% (16-45%) | 6.208 | 1.11 | 0.54 |
| rolling | per-segment | 32 | 6 | 19% (9-35%) | 5.001 | 0.93 | 0.83 |
| turn_first | equal-duration | 32 | 6 | 19% (9-35%) | 6.230 | 1.21 | 0.73 |
| turn_first | per-segment | 32 | 4 | 12% (5-28%) | 5.427 | 1.25 | 0.92 |

Fastest candidate per mission (n=32): baseline/equal-duration: 2, baseline/per-segment: 9, bow_left/per-segment: 4, bow_right/per-segment: 6, rolling/equal-duration: 1, rolling/per-segment: 7, turn_first/per-segment: 3

Baseline alone would have produced a passing trajectory on 8/32 missions (25%).
No candidate passed on 19/32 missions.
A non-baseline candidate rescued 5/32 missions: baseline/equal-duration: 1, bow_left/per-segment: 3, bow_right/per-segment: 1

## Failure reasons

| forward integration | 465 | 69.2% |
| tracking error | 458 | 68.2% |
| envelope radius | 308 | 45.8% |
| constraint violations | 113 | 16.8% |
| motor force | 81 | 12.1% |

Envelope radius: median 1.06 cm, max 5.26 cm (gate 2.00 cm)
Tracking error:  median 0.72 cm, max 11.02 cm (gate 0.50 cm)
Candidates with no envelope breach: 411/672

## Accuracy weight vs tracking error (Q6)

Pooled Spearman(weight, tracking) = -0.103
Pooled Pearson (weight, tracking) = -0.055
Per-mission Spearman: median -0.420, 42/64 missions negative (tracking got worse as the weight grew)
Non-monotone steps (higher weight, slower AND less accurate): 34/320

### Tracking error by weight (pooled, baseline seed)

| weight | n | median tracking cm | median duration s | pass rate |
|---:|---:|---:|---:|---:|
| 0 | 128 | 0.80 | 5.634 | 20% |
| 0.001 | 64 | 0.67 | 5.648 | 17% |
| 0.002 | 64 | 0.68 | 5.679 | 22% |
| 0.005 | 64 | 0.67 | 5.739 | 31% |
| 0.01 | 64 | 0.68 | 5.805 | 31% |

## Selection rules (Q10)

| rule | missions with a passing pick | mean duration s | mean envelope cm | mean tracking cm |
|---|---:|---:|---:|---:|
| current | 13/32 | 3.361 | 0.56 | 0.42 |
| fastest | 13/32 | 3.361 | 0.56 | 0.42 |
| tracking_margin | 13/32 | 3.361 | 0.56 | 0.42 |
| envelope_margin | 13/32 | 3.361 | 0.56 | 0.42 |
| fewest_gates | 13/32 | 3.361 | 0.56 | 0.42 |

The current rule (smallest envelope) gives up -0.1 ms on average against the fastest passing candidate (median 0.0 ms, max 0.0 ms).

## Multi-Verse critic vs the gates (Q5)

passing candidates flagged by the critic: 152/152
failing candidates flagged by the critic: 520/520
precision (of the flagged, how many fail): 77%
recall    (of the failures, how many are flagged): 100%

A critic that flags passing trajectories would make a feedback loop harmful: it would send good windows back into the solver.

## Envelope growth vs cheap predictors (Q4)

envelope vs duration: Pearson 0.898, Spearman 0.873 (n=152)
envelope vs path length: Pearson 0.736, Spearman 0.768 (n=152)
envelope vs sample count: Pearson -, Spearman - (n=152)

## Uncertainty sensitivity (spread of the declared ranges)

| spread | n | pass rate | median envelope cm | median tracking cm |
|---:|---:|---:|---:|---:|
| 1 | 672 | 23% | 1.06 | 0.72 |

spread 1.0 is the range as declared in the robot config; 0 collapses every range to its nominal value; 2 doubles its width.

## Timing model (baseline seed, base weight)

| example_straight.json | 2.377 s ok | 2.377 s ok | +0.0 % |
| example_s_curve.json | 3.696 s FAIL | 3.667 s FAIL | +0.8 % |
| example_mission_first4.json | 2.176 s ok | 1.833 s ok | +15.7 % |
| example_complex_mission.json | 6.068 s ok | 4.650 s FAIL | +23.4 % |
| example_complex_segment.json | 3.837 s FAIL | 2.241 s FAIL | +41.6 % |
| example_with_stops.json | 6.246 s FAIL | 5.138 s FAIL | +17.7 % |
| example_with_events.json | 4.637 s FAIL | 4.416 s ok | +4.8 % |
| example_complete.json | 14.612 s FAIL | 10.444 s FAIL | +28.5 % |
| m000 | 5.525 s FAIL | 5.169 s FAIL | +6.4 % |
| m001 | 18.714 s FAIL | 13.217 s FAIL | +29.4 % |
| m002 | 13.065 s FAIL | 9.412 s FAIL | +28.0 % |
| m003 | 15.899 s FAIL | 9.750 s FAIL | +38.7 % |
| m004 | 9.507 s FAIL | 8.602 s FAIL | +9.5 % |
| m005 | 6.186 s FAIL | 4.268 s FAIL | +31.0 % |
| m006 | 15.182 s FAIL | 9.836 s FAIL | +35.2 % |
| m007 | 15.234 s FAIL | 7.988 s FAIL | +47.6 % |
| m008 | 5.731 s FAIL | 5.566 s FAIL | +2.9 % |
| m009 | 12.997 s FAIL | 7.174 s FAIL | +44.8 % |
| m010 | 2.517 s FAIL | 2.010 s ok | +20.1 % |
| m011 | 6.229 s FAIL | 4.869 s FAIL | +21.8 % |
| m012 | 2.304 s FAIL | 2.319 s FAIL | -0.6 % |
| m013 | 10.180 s FAIL | 5.702 s FAIL | +44.0 % |
| m014 | 11.673 s FAIL | 9.374 s FAIL | +19.7 % |
| m015 | 4.728 s ok | 2.964 s FAIL | +37.3 % |
| m016 | 3.088 s FAIL | 3.098 s FAIL | -0.3 % |
| m017 | 1.920 s FAIL | 1.779 s ok | +7.3 % |
| m018 | 3.850 s FAIL | 2.881 s FAIL | +25.2 % |
| m019 | 20.028 s FAIL | 15.388 s FAIL | +23.2 % |
| m020 | 5.156 s FAIL | 3.632 s FAIL | +29.6 % |
| m021 | 1.140 s ok | 0.998 s ok | +12.4 % |
| m022 | 9.501 s FAIL | 7.861 s FAIL | +17.3 % |
| m023 | 9.367 s FAIL | 7.579 s FAIL | +19.1 % |

per-segment is faster on 29, slower on 2, tied on 1 of 32 missions
passing: equal-duration 5/32, per-segment 6/32

## Tracking error vs the heading change per sample (RQ-A)

candidates: 672
Spearman(mean dtheta, tracking) = 0.615
Pearson (mean dtheta, tracking) = 0.326

| mission | mean dtheta (rad) | tracking (cm) | gate |
|---|---:|---:|---|
| example_complete.json | 0.2148 | 0.50 | FAIL |
| example_complex_mission.json | 0.2719 | 0.40 | PASS |
| example_complex_segment.json | 0.3274 | 0.32 | PASS |
| example_mission_first4.json | 0.1721 | 0.27 | PASS |
| example_s_curve.json | 0.3637 | 1.12 | FAIL |
| example_straight.json | 0.0001 | 0.33 | PASS |
| example_with_events.json | 0.1580 | 0.37 | PASS |
| example_with_stops.json | 0.2500 | 0.52 | FAIL |
| m000 | 0.0882 | 0.38 | PASS |
| m001 | 0.2486 | 3.61 | FAIL |
| m002 | 0.2591 | 0.57 | FAIL |
| m003 | 0.2290 | 1.61 | FAIL |
| m004 | 0.1000 | 0.41 | FAIL |
| m005 | 0.3005 | 0.45 | PASS |
| m006 | 0.3067 | 0.64 | FAIL |
| m007 | 0.1764 | 0.56 | FAIL |
| m008 | 0.1297 | 0.37 | FAIL |
| m009 | 0.1029 | 0.64 | FAIL |
| m010 | 0.0294 | 0.34 | PASS |
| m011 | 0.1582 | 0.53 | FAIL |
| m012 | 0.1527 | 0.42 | FAIL |
| m013 | 0.1470 | 0.56 | FAIL |
| m014 | 0.1759 | 1.00 | FAIL |
| m015 | 0.0845 | 0.35 | PASS |
| m016 | 0.2530 | 0.44 | FAIL |
| m017 | 0.1380 | 0.38 | PASS |
| m018 | 0.1842 | 0.59 | FAIL |
| m019 | 0.2496 | 5.01 | FAIL |
| m020 | 0.1062 | 0.42 | FAIL |
| m021 | 0.0067 | 0.39 | PASS |
| m022 | 0.1745 | 0.40 | FAIL |
| m023 | 0.2370 | 1.17 | FAIL |

| mean dtheta bucket | n | median tracking cm | pass rate |
|---|---:|---:|---:|
| 0.00-0.02 | 42 | 0.37 | 83% |
| 0.02-0.05 | 20 | 0.41 | 80% |
| 0.05-0.10 | 50 | 0.53 | 38% |
| 0.10-0.20 | 278 | 0.56 | 24% |
| 0.20-10.00 | 282 | 1.45 | 5% |

## The marginal value of a candidate (Q3)

| candidates | missions with a passing candidate | mean best duration s | solves |
|---|---:|---:|---:|
| 1 (last: baseline / equal-duration) | 5/32 | 3.297 | 1 |
| 2 (last: baseline / per-segment) | 8/32 | 3.026 | 2 |
| 3 (last: rolling / equal-duration) | 11/32 | 3.391 | 3 |
| 4 (last: rolling / per-segment) | 11/32 | 3.328 | 4 |
| 5 (last: bow_left / equal-duration) | 11/32 | 3.328 | 5 |
| 6 (last: bow_left / per-segment) | 12/32 | 3.128 | 6 |
| 7 (last: bow_right / equal-duration) | 12/32 | 3.128 | 7 |
| 8 (last: bow_right / per-segment) | 12/32 | 3.125 | 8 |
| 9 (last: turn_first / equal-duration) | 12/32 | 3.125 | 9 |
| 10 (last: turn_first / per-segment) | 12/32 | 3.125 | 10 |
| 11 (last: baseline / equal-duration / a=0.001) | 12/32 | 3.125 | 11 |
| 12 (last: baseline / per-segment / a=0.001) | 12/32 | 3.125 | 12 |
| 13 (last: baseline / equal-duration / a=0.002) | 12/32 | 3.125 | 13 |
| 14 (last: baseline / per-segment / a=0.002) | 12/32 | 3.125 | 14 |
| 15 (last: baseline / equal-duration / a=0.005) | 12/32 | 3.125 | 15 |
| 16 (last: baseline / per-segment / a=0.005) | 12/32 | 3.125 | 16 |
| 17 (last: baseline / equal-duration / a=0.01) | 13/32 | 3.364 | 17 |
| 18 (last: baseline / per-segment / a=0.01) | 13/32 | 3.364 | 18 |

## Peak wheel acceleration vs the DriveBase limit (7.8 m/s^2, KI-021)

candidates: 672
peak acceleration: median 3.66 m/s^2, max 10.70 m/s^2
above the DriveBase limit: 49/672 (7%)
among the passing candidates: 12/152 (8%) ask for more, median 3.39 m/s^2, max 10.26 m/s^2
