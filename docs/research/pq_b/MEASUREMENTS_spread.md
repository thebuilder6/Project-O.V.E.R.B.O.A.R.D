# Measurements: spread.jsonl

Source: `research/results/spread.jsonl`

- config: `examples/fll_choreo.json`
- spec set: `seeds`, samples: 10, base weight: 0.0
- started: 2026-10-20 02:00:55
- tree hashes:

Every number below is a model prediction. The declared uncertainty ranges are test assumptions, not measurements, and no hardware logs exist in the repository.

## Uncertainty sensitivity (spread of the declared ranges)

| spread | n | pass rate | median envelope cm | median tracking cm |
|---:|---:|---:|---:|---:|
| 0 | 90 | 32% | 0.43 | 0.59 |
| 0.5 | 90 | 32% | 0.60 | 0.60 |
| 1 | 90 | 31% | 0.79 | 0.60 |
| 1.5 | 90 | 2% | 0.99 | 0.60 |
| 2 | 90 | 0% | 1.18 | 0.60 |

spread 1.0 is the range as declared in the robot config; 0 collapses every range to its nominal value; 2 doubles its width.
