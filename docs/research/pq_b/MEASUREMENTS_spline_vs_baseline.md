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

## spline vs baseline

matched (mission, timing, weight) pairs: 64
spline faster on 26, slower on 37, tied on 1
spline passes on 15, baseline passes on 11
IPOPT iterations: spline median 33, baseline median 48
solve wall time:  spline median 0.05 s, baseline median 0.07 s
duration delta: median 0.0 %, mean 3.0 %, stdev 14.1 %
  example_complete.json / equal-duration / a=0.0: spline 14.634s ok=False | baseline 14.612s ok=False
  example_complete.json / per-segment / a=0.0: spline 10.519s ok=False | baseline 10.444s ok=False
  example_complex_mission.json / equal-duration / a=0.0: spline 8.685s ok=False | baseline 6.068s ok=True
  example_complex_mission.json / per-segment / a=0.0: spline 4.890s ok=False | baseline 4.650s ok=False
  example_complex_segment.json / equal-duration / a=0.0: spline 3.921s ok=True | baseline 3.837s ok=False
  example_complex_segment.json / per-segment / a=0.0: spline 2.310s ok=True | baseline 2.241s ok=False
  example_mission_first4.json / equal-duration / a=0.0: spline 2.567s ok=True | baseline 2.176s ok=True
  example_mission_first4.json / per-segment / a=0.0: spline 1.856s ok=True | baseline 1.833s ok=True
  example_s_curve.json / equal-duration / a=0.0: spline 3.778s ok=False | baseline 3.696s ok=False
  example_s_curve.json / per-segment / a=0.0: spline 3.770s ok=False | baseline 3.667s ok=False
  example_straight.json / equal-duration / a=0.0: spline 2.377s ok=True | baseline 2.377s ok=True
  example_straight.json / per-segment / a=0.0: spline 2.377s ok=True | baseline 2.377s ok=True
  example_with_events.json / equal-duration / a=0.0: spline 4.638s ok=False | baseline 4.637s ok=False
  example_with_events.json / per-segment / a=0.0: spline 4.416s ok=True | baseline 4.416s ok=True
  example_with_stops.json / equal-duration / a=0.0: spline 6.276s ok=False | baseline 6.246s ok=False
  example_with_stops.json / per-segment / a=0.0: spline 5.207s ok=False | baseline 5.138s ok=False
  m000 / equal-duration / a=0.0: spline 5.391s ok=False | baseline 5.525s ok=False
  m000 / per-segment / a=0.0: spline 6.626s ok=False | baseline 5.169s ok=False
  m001 / equal-duration / a=0.0: spline 25.234s ok=False | baseline 18.714s ok=False
  m001 / per-segment / a=0.0: spline 15.347s ok=False | baseline 13.217s ok=False
  m002 / equal-duration / a=0.0: spline 11.095s ok=False | baseline 13.065s ok=False
  m002 / per-segment / a=0.0: spline 9.373s ok=False | baseline 9.412s ok=False
  m003 / equal-duration / a=0.0: spline 13.960s ok=False | baseline 15.899s ok=False
  m003 / per-segment / a=0.0: spline 9.681s ok=False | baseline 9.750s ok=False
  m004 / equal-duration / a=0.0: spline 9.507s ok=False | baseline 9.507s ok=False
  m004 / per-segment / a=0.0: spline 8.385s ok=False | baseline 8.602s ok=False
  m005 / equal-duration / a=0.0: spline 6.196s ok=False | baseline 6.186s ok=False
  m005 / per-segment / a=0.0: spline 4.261s ok=False | baseline 4.268s ok=False
  m006 / equal-duration / a=0.0: spline 10.130s ok=False | baseline 15.182s ok=False
  m006 / per-segment / a=0.0: spline 10.310s ok=False | baseline 9.836s ok=False
  m007 / equal-duration / a=0.0: spline 13.145s ok=False | baseline 15.234s ok=False
  m007 / per-segment / a=0.0: spline 7.951s ok=False | baseline 7.988s ok=False
  m008 / equal-duration / a=0.0: spline 8.333s ok=False | baseline 5.731s ok=False
  m008 / per-segment / a=0.0: spline 6.224s ok=False | baseline 5.566s ok=False
  m009 / equal-duration / a=0.0: spline 13.106s ok=False | baseline 12.997s ok=False
  m009 / per-segment / a=0.0: spline 7.185s ok=False | baseline 7.174s ok=False
  m010 / equal-duration / a=0.0: spline 2.517s ok=True | baseline 2.517s ok=False
  m010 / per-segment / a=0.0: spline 2.011s ok=True | baseline 2.010s ok=True
  m011 / equal-duration / a=0.0: spline 6.229s ok=False | baseline 6.229s ok=False
  m011 / per-segment / a=0.0: spline 4.840s ok=False | baseline 4.869s ok=False
  m012 / equal-duration / a=0.0: spline 2.440s ok=False | baseline 2.304s ok=False
  m012 / per-segment / a=0.0: spline 3.753s ok=False | baseline 2.319s ok=False
  m013 / equal-duration / a=0.0: spline 10.180s ok=False | baseline 10.180s ok=False
  m013 / per-segment / a=0.0: spline 6.494s ok=False | baseline 5.702s ok=False
  m014 / equal-duration / a=0.0: spline 10.745s ok=False | baseline 11.673s ok=False
  m014 / per-segment / a=0.0: spline 9.162s ok=False | baseline 9.374s ok=False
  m015 / equal-duration / a=0.0: spline 4.727s ok=True | baseline 4.728s ok=True
  m015 / per-segment / a=0.0: spline 2.961s ok=False | baseline 2.964s ok=False
  m016 / equal-duration / a=0.0: spline 3.087s ok=False | baseline 3.088s ok=False
  m016 / per-segment / a=0.0: spline 3.100s ok=False | baseline 3.098s ok=False
  m017 / equal-duration / a=0.0: spline 1.938s ok=True | baseline 1.920s ok=False
  m017 / per-segment / a=0.0: spline 1.780s ok=True | baseline 1.779s ok=True
  m018 / equal-duration / a=0.0: spline 3.826s ok=False | baseline 3.850s ok=False
  m018 / per-segment / a=0.0: spline 2.846s ok=False | baseline 2.881s ok=False
  m019 / equal-duration / a=0.0: spline 19.263s ok=False | baseline 20.028s ok=False
  m019 / per-segment / a=0.0: spline 15.864s ok=False | baseline 15.388s ok=False
  m020 / equal-duration / a=0.0: spline 4.028s ok=True | baseline 5.156s ok=False
  m020 / per-segment / a=0.0: spline 3.682s ok=True | baseline 3.632s ok=False
  m021 / equal-duration / a=0.0: spline 1.140s ok=True | baseline 1.140s ok=True
  m021 / per-segment / a=0.0: spline 0.995s ok=False | baseline 0.998s ok=True
  m022 / equal-duration / a=0.0: spline 9.501s ok=False | baseline 9.501s ok=False
  m022 / per-segment / a=0.0: spline 8.100s ok=False | baseline 7.861s ok=False
  m023 / equal-duration / a=0.0: spline 9.458s ok=False | baseline 9.367s ok=False
  m023 / per-segment / a=0.0: spline 7.270s ok=False | baseline 7.579s ok=False
