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

## proportional vs rolling

matched (mission, timing, weight) pairs: 32
proportional faster on 18, slower on 11, tied on 3
proportional passes on 6, rolling passes on 6
IPOPT iterations: proportional median 40, rolling median 25
solve wall time:  proportional median 0.13 s, rolling median 0.05 s
duration delta: median -0.0 %, mean -1.8 %, stdev 4.0 %
  example_complete.json / per-segment / a=0.0: proportional 10.313s ok=False | rolling 10.296s ok=False
  example_complex_mission.json / per-segment / a=0.0: proportional 4.933s ok=False | rolling 5.395s ok=False
  example_complex_segment.json / per-segment / a=0.0: proportional 2.228s ok=False | rolling 2.243s ok=False
  example_mission_first4.json / per-segment / a=0.0: proportional 1.701s ok=True | rolling 1.796s ok=True
  example_s_curve.json / per-segment / a=0.0: proportional 3.759s ok=False | rolling 3.759s ok=False
  example_straight.json / per-segment / a=0.0: proportional 2.377s ok=True | rolling 2.377s ok=True
  example_with_events.json / per-segment / a=0.0: proportional 4.415s ok=True | rolling 4.415s ok=True
  example_with_stops.json / per-segment / a=0.0: proportional 4.771s ok=False | rolling 4.788s ok=False
  m000 / per-segment / a=0.0: proportional 5.183s ok=True | rolling 5.162s ok=False
  m001 / per-segment / a=0.0: proportional 13.265s ok=False | rolling 13.204s ok=False
  m002 / per-segment / a=0.0: proportional 10.139s ok=False | rolling 10.217s ok=False
  m003 / per-segment / a=0.0: proportional 9.680s ok=False | rolling 9.754s ok=False
  m004 / per-segment / a=0.0: proportional 8.320s ok=False | rolling 9.989s ok=False
  m005 / per-segment / a=0.0: proportional 4.262s ok=False | rolling 4.261s ok=False
  m006 / per-segment / a=0.0: proportional 9.826s ok=False | rolling 9.859s ok=False
  m007 / per-segment / a=0.0: proportional 7.937s ok=False | rolling 7.937s ok=False
  m008 / per-segment / a=0.0: proportional 5.557s ok=False | rolling 5.555s ok=False
  m009 / per-segment / a=0.0: proportional 6.545s ok=False | rolling 7.174s ok=False
  m010 / per-segment / a=0.0: proportional 2.011s ok=True | rolling 2.011s ok=True
  m011 / per-segment / a=0.0: proportional 4.859s ok=False | rolling 4.841s ok=False
  m012 / per-segment / a=0.0: proportional 2.320s ok=False | rolling 2.319s ok=False
  m013 / per-segment / a=0.0: proportional 5.695s ok=False | rolling 5.695s ok=False
  m014 / per-segment / a=0.0: proportional 9.334s ok=False | rolling 9.856s ok=False
  m015 / per-segment / a=0.0: proportional 2.972s ok=False | rolling 2.959s ok=False
  m016 / per-segment / a=0.0: proportional 3.045s ok=False | rolling 3.102s ok=False
  m017 / per-segment / a=0.0: proportional 1.778s ok=False | rolling 1.778s ok=True
  m018 / per-segment / a=0.0: proportional 2.821s ok=False | rolling 2.881s ok=False
  m019 / per-segment / a=0.0: proportional 16.000s ok=False | rolling 15.408s ok=False
  m020 / per-segment / a=0.0: proportional 3.634s ok=True | rolling 3.733s ok=True
  m021 / per-segment / a=0.0: proportional 0.995s ok=False | rolling 0.997s ok=False
  m022 / per-segment / a=0.0: proportional 7.839s ok=False | rolling 8.636s ok=False
  m023 / per-segment / a=0.0: proportional 7.397s ok=False | rolling 7.399s ok=False
