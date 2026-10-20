# Measurements: density.jsonl

Source: `research/results/density.jsonl`

- config: `examples/fll_choreo.json`
- spec set: `density`, samples: 10, base weight: 0.0
- started: 2026-10-20 01:49:16
- tree hashes:

Every number below is a model prediction. The declared uncertainty ranges are test assumptions, not measurements, and no hardware logs exist in the repository.

## Sample density (Q7)

| samples | timing | n | pass | median duration s | median envelope cm | median tracking cm | median solve s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | equal-duration | 8 | 1 | 7.498 | 1.28 | 4.99 | 0.28 |
| 5 | per-segment | 8 | 1 | 6.069 | 1.14 | 4.77 | 0.37 |
| 10 | equal-duration | 8 | 3 | 5.796 | 1.00 | 1.01 | 0.59 |
| 10 | per-segment | 8 | 2 | 4.909 | 0.87 | 0.99 | 0.78 |
| 20 | equal-duration | 8 | 0 | 5.736 | 1.04 | 0.81 | 2.09 |
| 20 | per-segment | 8 | 0 | 5.289 | 0.96 | 0.94 | 2.47 |
| 40 | equal-duration | 8 | 1 | 8.985 | 1.51 | 1.00 | 3.55 |
| 40 | per-segment | 8 | 1 | 5.673 | 1.06 | 0.87 | 5.59 |
