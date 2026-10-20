# Measurements: sc_on.jsonl

Source: `research/results/sc_on.jsonl`

- config: `examples/fll_choreo.json`
- spec set: `seeds`, samples: 10, base weight: 0.0
- started: 2026-10-20 02:56:12
- tree hashes:

Every number below is a model prediction. The declared uncertainty ranges are test assumptions, not measurements, and no hardware logs exist in the repository.
records: 110  missions: 11
config: examples/fll_choreo.json  samples: 10  specs: seeds  started: 2026-10-20 02:56:12
converged: 108/110 (98.2%)
passing:   50/110 (45.5%)
duration:  median 4.987 s  mean 6.670 s
envelope:  median 0.86 cm
tracking:  median 0.44 cm
solve wall: median 0.80 s  total 4.7 min

## Per seed and timing model

(stock seeds only; the extra guess families are compared separately)
| seed | timing | n | pass | pass rate (95% CI) | median dur | median env cm | median trk cm |
|---|---|---:|---:|---|---:|---:|---:|
| baseline | equal-duration | 11 | 4 | 36% (15-65%) | 5.559 | 0.89 | 0.48 |
| baseline | per-segment | 11 | 5 | 45% (21-72%) | 5.189 | 0.94 | 0.75 |
| bow_left | equal-duration | 11 | 6 | 55% (28-79%) | 5.345 | 0.88 | 0.41 |
| bow_left | per-segment | 11 | 5 | 45% (21-72%) | 4.801 | 0.81 | 0.32 |
| bow_right | equal-duration | 11 | 5 | 45% (21-72%) | 5.345 | 0.88 | 0.41 |
| bow_right | per-segment | 11 | 5 | 45% (21-72%) | 4.785 | 0.81 | 0.33 |
| rolling | equal-duration | 11 | 6 | 55% (28-79%) | 5.345 | 0.89 | 0.43 |
| rolling | per-segment | 11 | 5 | 45% (21-72%) | 4.793 | 0.81 | 0.33 |
| turn_first | equal-duration | 11 | 5 | 45% (21-72%) | 5.400 | 0.95 | 0.51 |
| turn_first | per-segment | 11 | 4 | 36% (15-65%) | 5.340 | 0.99 | 0.64 |

Fastest candidate per mission (n=11): baseline/per-segment: 2, bow_right/equal-duration: 1, bow_right/per-segment: 3, rolling/per-segment: 4, turn_first/per-segment: 1

Baseline alone would have produced a passing trajectory on 6/11 missions (55%).
No candidate passed on 5/11 missions.
A non-baseline candidate rescued 0/11 missions: none

## Failure reasons

| tracking error | 53 | 48.2% |
| envelope radius | 30 | 27.3% |
| motor force | 19 | 17.3% |
| constraint violations | 16 | 14.5% |

Envelope radius: median 0.86 cm, max 5.07 cm (gate 2.00 cm)
Tracking error:  median 0.44 cm, max 2.84 cm (gate 0.50 cm)
Candidates with no envelope breach: 80/110

## Accuracy weight vs tracking error (Q6)

Pooled Spearman(weight, tracking) = -
Pooled Pearson (weight, tracking) = -
Per-mission Spearman: median -, 0/0 missions negative (tracking got worse as the weight grew)
Non-monotone steps (higher weight, slower AND less accurate): 0/0

### Tracking error by weight (pooled, baseline seed)

| weight | n | median tracking cm | median duration s | pass rate |
|---:|---:|---:|---:|---:|
| 0 | 22 | 0.61 | 5.194 | 41% |

## Selection rules (Q10)

| rule | missions with a passing pick | mean duration s | mean envelope cm | mean tracking cm |
|---|---:|---:|---:|---:|
| current | 6/11 | 3.512 | 0.59 | 0.16 |
| fastest | 6/11 | 3.449 | 0.60 | 0.17 |
| tracking_margin | 6/11 | 3.449 | 0.60 | 0.17 |
| envelope_margin | 6/11 | 3.449 | 0.60 | 0.17 |
| fewest_gates | 6/11 | 3.449 | 0.60 | 0.17 |

The current rule (smallest envelope) gives up -62.9 ms on average against the fastest passing candidate (median -1.0 ms, max 0.0 ms).

## Multi-Verse critic vs the gates (Q5)

passing candidates flagged by the critic: 50/50
failing candidates flagged by the critic: 60/60
precision (of the flagged, how many fail): 55%
recall    (of the failures, how many are flagged): 100%

A critic that flags passing trajectories would make a feedback loop harmful: it would send good windows back into the solver.

## Envelope growth vs cheap predictors (Q4)

envelope vs duration: Pearson 0.937, Spearman 0.873 (n=50)
envelope vs path length: Pearson 0.528, Spearman 0.615 (n=50)
envelope vs sample count: Pearson -, Spearman - (n=50)

## Uncertainty sensitivity (spread of the declared ranges)

| spread | n | pass rate | median envelope cm | median tracking cm |
|---:|---:|---:|---:|---:|
| 1 | 110 | 45% | 0.86 | 0.44 |

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

per-segment is faster on 9, slower on 1, tied on 1 of 11 missions
passing: equal-duration 4/11, per-segment 5/11

## Tracking error vs the heading change per sample (RQ-A)

candidates: 110
Spearman(mean dtheta, tracking) = 0.632
Pearson (mean dtheta, tracking) = 0.469

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

| mean dtheta bucket | n | median tracking cm | pass rate |
|---|---:|---:|---:|
| 0.00-0.02 | 10 | 0.01 | 100% |
| 0.05-0.10 | 10 | 0.28 | 70% |
| 0.10-0.20 | 43 | 0.32 | 58% |
| 0.20-10.00 | 47 | 0.96 | 17% |

## The marginal value of a candidate (Q3)

| candidates | missions with a passing candidate | mean best duration s | solves |
|---|---:|---:|---:|
| 1 (last: baseline / equal-duration) | 4/11 | 3.267 | 1 |
| 2 (last: baseline / per-segment) | 6/11 | 3.568 | 2 |
| 3 (last: rolling / equal-duration) | 6/11 | 3.568 | 3 |
| 4 (last: rolling / per-segment) | 6/11 | 3.461 | 4 |
| 5 (last: bow_left / equal-duration) | 6/11 | 3.461 | 5 |
| 6 (last: bow_left / per-segment) | 6/11 | 3.460 | 6 |
| 7 (last: bow_right / equal-duration) | 6/11 | 3.460 | 7 |
| 8 (last: bow_right / per-segment) | 6/11 | 3.449 | 8 |
| 9 (last: turn_first / equal-duration) | 6/11 | 3.449 | 9 |
| 10 (last: turn_first / per-segment) | 6/11 | 3.449 | 10 |
| 11 (last: baseline / equal-duration / a=0.001) | 6/11 | 3.449 | 11 |
| 12 (last: baseline / per-segment / a=0.001) | 6/11 | 3.449 | 12 |
| 13 (last: baseline / equal-duration / a=0.002) | 6/11 | 3.449 | 13 |
| 14 (last: baseline / per-segment / a=0.002) | 6/11 | 3.449 | 14 |
| 15 (last: baseline / equal-duration / a=0.005) | 6/11 | 3.449 | 15 |
| 16 (last: baseline / per-segment / a=0.005) | 6/11 | 3.449 | 16 |
| 17 (last: baseline / equal-duration / a=0.01) | 6/11 | 3.449 | 17 |
| 18 (last: baseline / per-segment / a=0.01) | 6/11 | 3.449 | 18 |

## Peak wheel acceleration vs the DriveBase limit (7.8 m/s^2, KI-021)

candidates: 110
peak acceleration: median 4.65 m/s^2, max 10.60 m/s^2
above the DriveBase limit: 17/110 (15%)
among the passing candidates: 10/50 (20%) ask for more, median 4.94 m/s^2, max 10.60 m/s^2
