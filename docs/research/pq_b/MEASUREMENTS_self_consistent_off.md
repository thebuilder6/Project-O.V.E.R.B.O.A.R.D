# Measurements: sc_off.jsonl

Source: `research/results/sc_off.jsonl`

- config: `examples/fll_choreo.json`
- spec set: `seeds`, samples: 10, base weight: 0.0
- started: 2026-10-20 02:40:32
- tree hashes:
  - `optimizer.py`: `92322B2722CD99C168CC1E0527ABD9AA`
  - `main.py`: `2B626E971344F968A033AB211FC6D51D`
  - `immrax_validator.py`: `857CF573CB7146E095BFB78AC3AB16CB`
  - `validator.py`: `70950A6FAAB297AEF734FD67A67B8068`
  - `robot_model.py`: `D1AF67DDA7DABDA02F116C38EF368C8B`
  - `multiverse_optimizer.py`: `D67E88CEA2A2B072B5B60F3CF52354F2`

Every number below is a model prediction. The declared uncertainty ranges are test assumptions, not measurements, and no hardware logs exist in the repository.
records: 160  missions: 16
config: examples/fll_choreo.json  samples: 10  specs: seeds  started: 2026-10-20 02:40:32
  optimizer.py             92322B2722CD99C168CC1E0527ABD9AA
  main.py                  2B626E971344F968A033AB211FC6D51D
  immrax_validator.py      857CF573CB7146E095BFB78AC3AB16CB
  validator.py             70950A6FAAB297AEF734FD67A67B8068
  robot_model.py           D1AF67DDA7DABDA02F116C38EF368C8B
  multiverse_optimizer.py  D67E88CEA2A2B072B5B60F3CF52354F2
WARNING: the run spans several tree states:
  039450626CAA: 130 records
  D41D8CD98F00: 30 records
converged: 157/160 (98.1%)
passing:   33/160 (20.6%)
duration:  median 6.186 s  mean 7.654 s
envelope:  median 1.15 cm
tracking:  median 0.80 cm
solve wall: median 0.62 s  total 7.1 min

## Per seed and timing model

(stock seeds only; the extra guess families are compared separately)
| seed | timing | n | pass | pass rate (95% CI) | median dur | median env cm | median trk cm |
|---|---|---:|---:|---|---:|---:|---:|
| baseline | equal-duration | 16 | 3 | 19% (7-43%) | 6.216 | 1.16 | 0.74 |
| baseline | per-segment | 16 | 3 | 19% (7-43%) | 5.154 | 0.98 | 0.99 |
| bow_left | equal-duration | 16 | 4 | 25% (10-49%) | 7.440 | 1.40 | 0.64 |
| bow_left | per-segment | 16 | 4 | 25% (10-49%) | 5.521 | 1.09 | 0.98 |
| bow_right | equal-duration | 16 | 4 | 25% (10-49%) | 7.454 | 1.38 | 0.64 |
| bow_right | per-segment | 16 | 4 | 25% (10-49%) | 5.021 | 0.98 | 1.04 |
| rolling | equal-duration | 16 | 4 | 25% (10-49%) | 7.312 | 1.39 | 0.53 |
| rolling | per-segment | 16 | 3 | 19% (7-43%) | 5.278 | 1.05 | 1.02 |
| turn_first | equal-duration | 16 | 2 | 12% (3-36%) | 7.196 | 1.37 | 0.73 |
| turn_first | per-segment | 16 | 2 | 12% (3-36%) | 5.560 | 1.38 | 0.91 |

Fastest candidate per mission (n=16): baseline/equal-duration: 1, baseline/per-segment: 6, bow_left/per-segment: 2, bow_right/per-segment: 3, rolling/per-segment: 3, turn_first/per-segment: 1

Baseline alone would have produced a passing trajectory on 4/16 missions (25%).
No candidate passed on 10/16 missions.
A non-baseline candidate rescued 2/16 missions: bow_left/per-segment: 1, bow_right/per-segment: 1

## Failure reasons

| tracking error | 120 | 75.0% |
| forward integration | 118 | 73.8% |
| envelope radius | 70 | 43.8% |
| motor force | 26 | 16.2% |
| constraint violations | 23 | 14.4% |

Envelope radius: median 1.15 cm, max 5.07 cm (gate 2.00 cm)
Tracking error:  median 0.80 cm, max 7.39 cm (gate 0.50 cm)
Candidates with no envelope breach: 94/160

## Accuracy weight vs tracking error (Q6)

Pooled Spearman(weight, tracking) = -
Pooled Pearson (weight, tracking) = -
Per-mission Spearman: median -, 0/0 missions negative (tracking got worse as the weight grew)
Non-monotone steps (higher weight, slower AND less accurate): 0/0

### Tracking error by weight (pooled, baseline seed)

| weight | n | median tracking cm | median duration s | pass rate |
|---:|---:|---:|---:|---:|
| 0 | 32 | 0.82 | 6.127 | 19% |

## Selection rules (Q10)

| rule | missions with a passing pick | mean duration s | mean envelope cm | mean tracking cm |
|---|---:|---:|---:|---:|
| current | 6/16 | 3.672 | 0.62 | 0.44 |
| fastest | 6/16 | 3.672 | 0.62 | 0.43 |
| tracking_margin | 6/16 | 3.672 | 0.62 | 0.43 |
| envelope_margin | 6/16 | 3.672 | 0.62 | 0.43 |
| fewest_gates | 6/16 | 3.672 | 0.62 | 0.43 |

The current rule (smallest envelope) gives up -0.1 ms on average against the fastest passing candidate (median -0.0 ms, max 0.0 ms).

## Multi-Verse critic vs the gates (Q5)

passing candidates flagged by the critic: 33/33
failing candidates flagged by the critic: 127/127
precision (of the flagged, how many fail): 79%
recall    (of the failures, how many are flagged): 100%

A critic that flags passing trajectories would make a feedback loop harmful: it would send good windows back into the solver.

## Envelope growth vs cheap predictors (Q4)

envelope vs duration: Pearson 0.906, Spearman 0.834 (n=33)
envelope vs path length: Pearson 0.654, Spearman 0.489 (n=33)
envelope vs sample count: Pearson -, Spearman - (n=33)

## Uncertainty sensitivity (spread of the declared ranges)

| spread | n | pass rate | median envelope cm | median tracking cm |
|---:|---:|---:|---:|---:|
| 1 | 160 | 21% | 1.15 | 0.80 |

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

per-segment is faster on 15, slower on 0, tied on 1 of 16 missions
passing: equal-duration 3/16, per-segment 3/16

## Tracking error vs the heading change per sample (RQ-A)

candidates: 160
Spearman(mean dtheta, tracking) = 0.560
Pearson (mean dtheta, tracking) = 0.202

| mission | mean dtheta (rad) | tracking (cm) | gate |
|---|---:|---:|---|
| example_complete.json | 0.2148 | 0.49 | FAIL |
| example_complex_mission.json | 0.1446 | 0.49 | PASS |
| example_complex_segment.json | 0.3068 | 0.41 | FAIL |
| example_mission_first4.json | 0.1727 | 0.27 | PASS |
| example_s_curve.json | 0.3277 | 1.25 | FAIL |
| example_straight.json | 0.0000 | 0.34 | PASS |
| example_with_events.json | 0.1333 | 0.44 | PASS |
| example_with_stops.json | 0.2500 | 0.52 | FAIL |
| m000 | 0.0798 | 0.41 | PASS |
| m001 | 0.2486 | 3.60 | FAIL |
| m002 | 0.3288 | 0.69 | FAIL |
| m003 | 0.2290 | 1.60 | FAIL |
| m004 | 0.1000 | 0.42 | FAIL |
| m005 | 0.2134 | 1.00 | FAIL |
| m006 | 0.2985 | 0.86 | FAIL |
| m007 | 0.1595 | 0.89 | FAIL |

| mean dtheta bucket | n | median tracking cm | pass rate |
|---|---:|---:|---:|
| 0.00-0.02 | 10 | 0.34 | 100% |
| 0.05-0.10 | 17 | 0.52 | 29% |
| 0.10-0.20 | 55 | 0.54 | 27% |
| 0.20-10.00 | 78 | 1.27 | 4% |

## The marginal value of a candidate (Q3)

| candidates | missions with a passing candidate | mean best duration s | solves |
|---|---:|---:|---:|
| 1 (last: baseline / equal-duration) | 3/16 | 3.540 | 1 |
| 2 (last: baseline / per-segment) | 4/16 | 3.673 | 2 |
| 3 (last: rolling / equal-duration) | 6/16 | 3.960 | 3 |
| 4 (last: rolling / per-segment) | 6/16 | 3.954 | 4 |
| 5 (last: bow_left / equal-duration) | 6/16 | 3.954 | 5 |
| 6 (last: bow_left / per-segment) | 6/16 | 3.694 | 6 |
| 7 (last: bow_right / equal-duration) | 6/16 | 3.694 | 7 |
| 8 (last: bow_right / per-segment) | 6/16 | 3.672 | 8 |
| 9 (last: turn_first / equal-duration) | 6/16 | 3.672 | 9 |
| 10 (last: turn_first / per-segment) | 6/16 | 3.672 | 10 |
| 11 (last: baseline / equal-duration / a=0.001) | 6/16 | 3.672 | 11 |
| 12 (last: baseline / per-segment / a=0.001) | 6/16 | 3.672 | 12 |
| 13 (last: baseline / equal-duration / a=0.002) | 6/16 | 3.672 | 13 |
| 14 (last: baseline / per-segment / a=0.002) | 6/16 | 3.672 | 14 |
| 15 (last: baseline / equal-duration / a=0.005) | 6/16 | 3.672 | 15 |
| 16 (last: baseline / per-segment / a=0.005) | 6/16 | 3.672 | 16 |
| 17 (last: baseline / equal-duration / a=0.01) | 6/16 | 3.672 | 17 |
| 18 (last: baseline / per-segment / a=0.01) | 6/16 | 3.672 | 18 |

## Peak wheel acceleration vs the DriveBase limit (7.8 m/s^2, KI-021)

candidates: 160
peak acceleration: median 4.39 m/s^2, max 10.49 m/s^2
above the DriveBase limit: 13/160 (8%)
among the passing candidates: 2/33 (6%) ask for more, median 3.54 m/s^2, max 9.38 m/s^2
