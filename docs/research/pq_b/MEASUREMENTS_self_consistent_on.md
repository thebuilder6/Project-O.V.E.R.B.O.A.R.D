# Measurements: sc_on.jsonl

Source: `research/results/sc_on.jsonl`

- config: `examples/fll_choreo.json`
- spec set: `seeds`, samples: 10, base weight: 0.0
- started: 2026-10-20 02:56:12
- tree hashes:

Every number below is a model prediction. The declared uncertainty ranges are test assumptions, not measurements, and no hardware logs exist in the repository.
records: 160  missions: 16
config: examples/fll_choreo.json  samples: 10  specs: seeds  started: 2026-10-20 02:56:12
converged: 158/160 (98.8%)
passing:   50/160 (31.2%)
duration:  median 6.235 s  mean 7.621 s
envelope:  median 1.14 cm
tracking:  median 0.68 cm
solve wall: median 0.82 s  total 5.5 min

## Per seed and timing model

(stock seeds only; the extra guess families are compared separately)
| seed | timing | n | pass | pass rate (95% CI) | median dur | median env cm | median trk cm |
|---|---|---:|---:|---|---:|---:|---:|
| baseline | equal-duration | 16 | 4 | 25% (10-49%) | 8.265 | 1.55 | 0.80 |
| baseline | per-segment | 16 | 5 | 31% (14-56%) | 5.229 | 1.01 | 0.94 |
| bow_left | equal-duration | 16 | 6 | 38% (18-61%) | 7.496 | 1.35 | 0.59 |
| bow_left | per-segment | 16 | 5 | 31% (14-56%) | 5.079 | 0.98 | 0.70 |
| bow_right | equal-duration | 16 | 5 | 31% (14-56%) | 7.519 | 1.35 | 0.58 |
| bow_right | per-segment | 16 | 5 | 31% (14-56%) | 4.995 | 0.97 | 0.71 |
| rolling | equal-duration | 16 | 6 | 38% (18-61%) | 7.359 | 1.33 | 0.61 |
| rolling | per-segment | 16 | 5 | 31% (14-56%) | 5.029 | 0.98 | 0.69 |
| turn_first | equal-duration | 16 | 5 | 31% (14-56%) | 7.518 | 1.38 | 0.83 |
| turn_first | per-segment | 16 | 4 | 25% (10-49%) | 5.493 | 1.33 | 0.87 |

Fastest candidate per mission (n=16): baseline/per-segment: 3, bow_left/per-segment: 2, bow_right/equal-duration: 1, bow_right/per-segment: 5, rolling/per-segment: 4, turn_first/per-segment: 1

Baseline alone would have produced a passing trajectory on 6/16 missions (38%).
No candidate passed on 10/16 missions.
A non-baseline candidate rescued 0/16 missions: none

## Failure reasons

| tracking error | 95 | 59.4% |
| envelope radius | 70 | 43.8% |
| constraint violations | 24 | 15.0% |
| motor force | 23 | 14.4% |

Envelope radius: median 1.14 cm, max 5.07 cm (gate 2.00 cm)
Tracking error:  median 0.68 cm, max 2.84 cm (gate 0.50 cm)
Candidates with no envelope breach: 94/160

## Accuracy weight vs tracking error (Q6)

Pooled Spearman(weight, tracking) = -
Pooled Pearson (weight, tracking) = -
Per-mission Spearman: median -, 0/0 missions negative (tracking got worse as the weight grew)
Non-monotone steps (higher weight, slower AND less accurate): 0/0

### Tracking error by weight (pooled, baseline seed)

| weight | n | median tracking cm | median duration s | pass rate |
|---:|---:|---:|---:|---:|
| 0 | 32 | 0.83 | 6.946 | 28% |

## Selection rules (Q10)

| rule | missions with a passing pick | mean duration s | mean envelope cm | mean tracking cm |
|---|---:|---:|---:|---:|
| current | 6/16 | 3.512 | 0.59 | 0.16 |
| fastest | 6/16 | 3.449 | 0.60 | 0.17 |
| tracking_margin | 6/16 | 3.449 | 0.60 | 0.17 |
| envelope_margin | 6/16 | 3.449 | 0.60 | 0.17 |
| fewest_gates | 6/16 | 3.449 | 0.60 | 0.17 |

The current rule (smallest envelope) gives up -62.9 ms on average against the fastest passing candidate (median -1.0 ms, max 0.0 ms).

## Multi-Verse critic vs the gates (Q5)

passing candidates flagged by the critic: 50/50
failing candidates flagged by the critic: 110/110
precision (of the flagged, how many fail): 69%
recall    (of the failures, how many are flagged): 100%

A critic that flags passing trajectories would make a feedback loop harmful: it would send good windows back into the solver.

## Envelope growth vs cheap predictors (Q4)

envelope vs duration: Pearson 0.937, Spearman 0.873 (n=50)
envelope vs path length: Pearson 0.528, Spearman 0.615 (n=50)
envelope vs sample count: Pearson -, Spearman - (n=50)

## Uncertainty sensitivity (spread of the declared ranges)

| spread | n | pass rate | median envelope cm | median tracking cm |
|---:|---:|---:|---:|---:|
| 1 | 160 | 31% | 1.14 | 0.68 |

spread 1.0 is the range as declared in the robot config; 0 collapses every range to its nominal value; 2 doubles its width.

## Timing model (baseline seed, base weight)

| example_straight.json | 2.377 s ok | 2.377 s ok | +0.0 % |
| example_s_curve.json | 3.725 s FAIL | 3.636 s FAIL | +2.4 % |
| example_mission_first4.json | 2.184 s ok | 1.680 s ok | +23.1 % |
| example_complex_mission.json | 6.079 s FAIL | 5.200 s ok | +14.5 % |
| example_complex_segment.json | 3.853 s ok | 2.241 s ok | +41.8 % |
| example_with_stops.json | 7.813 s FAIL | 5.189 s FAIL | +33.6 % |
| example_with_events.json | 4.654 s ok | 4.991 s FAIL | -7.2 % |
| example_complete.json | 14.646 s FAIL | 10.412 s FAIL | +28.9 % |
| m000 | 5.559 s FAIL | 5.259 s ok | +5.4 % |
| m001 | 17.814 s FAIL | 13.252 s FAIL | +25.6 % |
| m002 | 11.764 s FAIL | 10.473 s FAIL | +11.0 % |
| m003 | 14.009 s FAIL | 9.768 s FAIL | +30.3 % |
| m004 | 9.507 s FAIL | 8.598 s FAIL | +9.6 % |
| m005 | 8.718 s FAIL | 4.244 s FAIL | +51.3 % |
| m006 | 11.005 s FAIL | 9.845 s FAIL | +10.5 % |
| m007 | 15.381 s FAIL | 7.928 s FAIL | +48.5 % |

per-segment is faster on 14, slower on 1, tied on 1 of 16 missions
passing: equal-duration 4/16, per-segment 5/16

## Tracking error vs the heading change per sample (RQ-A)

candidates: 160
Spearman(mean dtheta, tracking) = 0.573
Pearson (mean dtheta, tracking) = 0.471

| mission | mean dtheta (rad) | tracking (cm) | gate |
|---|---:|---:|---|
| example_complete.json | 0.2134 | 0.63 | FAIL |
| example_complex_mission.json | 0.1317 | 0.14 | PASS |
| example_complex_segment.json | 0.2156 | 0.11 | PASS |
| example_mission_first4.json | 0.1301 | 0.08 | FAIL |
| example_s_curve.json | 0.4308 | 1.01 | FAIL |
| example_straight.json | 0.0000 | 0.01 | PASS |
| example_with_events.json | 0.1331 | 0.32 | PASS |
| example_with_stops.json | 0.2515 | 0.61 | FAIL |
| m000 | 0.0841 | 0.24 | PASS |
| m001 | 0.1074 | 1.28 | FAIL |
| m002 | 0.2953 | 0.75 | FAIL |
| m003 | 0.2212 | 1.13 | FAIL |
| m004 | 0.0949 | 0.31 | FAIL |
| m005 | 0.2128 | 0.55 | FAIL |
| m006 | 0.2762 | 0.67 | FAIL |
| m007 | 0.1565 | 0.59 | FAIL |

| mean dtheta bucket | n | median tracking cm | pass rate |
|---|---:|---:|---:|
| 0.00-0.02 | 10 | 0.01 | 100% |
| 0.05-0.10 | 17 | 0.32 | 41% |
| 0.10-0.20 | 58 | 0.44 | 43% |
| 0.20-10.00 | 75 | 1.02 | 11% |

## The marginal value of a candidate (Q3)

| candidates | missions with a passing candidate | mean best duration s | solves |
|---|---:|---:|---:|
| 1 (last: baseline / equal-duration) | 4/16 | 3.267 | 1 |
| 2 (last: baseline / per-segment) | 6/16 | 3.568 | 2 |
| 3 (last: rolling / equal-duration) | 6/16 | 3.568 | 3 |
| 4 (last: rolling / per-segment) | 6/16 | 3.461 | 4 |
| 5 (last: bow_left / equal-duration) | 6/16 | 3.461 | 5 |
| 6 (last: bow_left / per-segment) | 6/16 | 3.460 | 6 |
| 7 (last: bow_right / equal-duration) | 6/16 | 3.460 | 7 |
| 8 (last: bow_right / per-segment) | 6/16 | 3.449 | 8 |
| 9 (last: turn_first / equal-duration) | 6/16 | 3.449 | 9 |
| 10 (last: turn_first / per-segment) | 6/16 | 3.449 | 10 |
| 11 (last: baseline / equal-duration / a=0.001) | 6/16 | 3.449 | 11 |
| 12 (last: baseline / per-segment / a=0.001) | 6/16 | 3.449 | 12 |
| 13 (last: baseline / equal-duration / a=0.002) | 6/16 | 3.449 | 13 |
| 14 (last: baseline / per-segment / a=0.002) | 6/16 | 3.449 | 14 |
| 15 (last: baseline / equal-duration / a=0.005) | 6/16 | 3.449 | 15 |
| 16 (last: baseline / per-segment / a=0.005) | 6/16 | 3.449 | 16 |
| 17 (last: baseline / equal-duration / a=0.01) | 6/16 | 3.449 | 17 |
| 18 (last: baseline / per-segment / a=0.01) | 6/16 | 3.449 | 18 |

## Peak wheel acceleration vs the DriveBase limit (7.8 m/s^2, KI-021)

candidates: 160
peak acceleration: median 4.39 m/s^2, max 10.60 m/s^2
above the DriveBase limit: 17/160 (11%)
among the passing candidates: 10/50 (20%) ask for more, median 4.94 m/s^2, max 10.60 m/s^2
