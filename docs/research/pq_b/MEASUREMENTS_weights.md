# Measurements: weights.jsonl

Source: `research/results/weights.jsonl`

- config: `examples/fll_choreo.json`
- spec set: `wide-weights`, samples: 10, base weight: 0.0
- started: 2026-10-20 01:29:03
- tree hashes:
  - `optimizer.py`: `B871E00BD3C8C3CE506B716DF8596CB0`
  - `main.py`: `E97A773F495158A633D6FC4A7AFCC795`
  - `immrax_validator.py`: `C7AAFF38D1E3FDEE6ECECDCE198D5873`
  - `validator.py`: `70950A6FAAB297AEF734FD67A67B8068`
  - `robot_model.py`: `2B798C532F02BAE7F77DBA0A7CBE9C81`
  - `multiverse_optimizer.py`: `D67E88CEA2A2B072B5B60F3CF52354F2`

Every number below is a model prediction. The declared uncertainty ranges are test assumptions, not measurements, and no hardware logs exist in the repository.

## Accuracy weight vs tracking error (Q6)

Pooled Spearman(weight, tracking) = -0.174
Pooled Pearson (weight, tracking) = -0.013
Per-mission Spearman: median -0.683, 20/24 missions negative (tracking got worse as the weight grew)
Non-monotone steps (higher weight, slower AND less accurate): 21/192

### Tracking error by weight (pooled, baseline seed)

| weight | n | median tracking cm | median duration s | pass rate |
|---:|---:|---:|---:|---:|
| 0 | 24 | 0.77 | 5.347 | 21% |
| 0.00025 | 24 | 0.81 | 5.406 | 21% |
| 0.0005 | 24 | 0.79 | 5.574 | 17% |
| 0.001 | 24 | 0.67 | 5.767 | 25% |
| 0.002 | 24 | 0.67 | 5.767 | 33% |
| 0.005 | 24 | 0.65 | 5.840 | 38% |
| 0.01 | 24 | 0.61 | 5.947 | 38% |
| 0.02 | 24 | 0.60 | 6.049 | 42% |
| 0.05 | 24 | 0.53 | 6.610 | 38% |
