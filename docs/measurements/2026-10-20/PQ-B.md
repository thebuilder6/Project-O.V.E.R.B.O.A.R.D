# PQ-B measurement data — 2026-10-20

The raw records of the PQ-B campaign: the research agenda measured on **32
missions** instead of six. The analysis is in
[`../../research/pq_b/FINDINGS.md`](../../research/pq_b/FINDINGS.md); the
generated tables are in [`../../research/pq_b/`](../../research/pq_b/).

The files in this folder (`*.jsonl`, `missions/`) are the PQ-B data. The
`*.log` files and `*_results.json` in this folder belong to the separate
before/after KI-025 comparison documented in [`README.md`](README.md).

## The files

| file | what it contains | records |
|---|---|---:|
| `matrix.jsonl` | 32 missions x 21 candidates through the full gate chain | 672 |
| `weights.jsonl` | a 9-point accuracy-weight ladder on 12 missions, both timing models | 216 |
| `density.jsonl` | samples per segment in {5, 10, 20, 40} on 6 missions, both timing models | 48 |
| `multiverse.jsonl` | the Multi-Verse pipeline against the simple solver on 12 missions | 24 |
| `spread.jsonl` | the same candidates validated at five uncertainty widths | 450 |
| `sc_off.jsonl`, `sc_on.jsonl` | the seed matrix with and without `--self-consistent` (KI-026), on one pinned tree | 160 + 160 |
| `missions/` | the 24 generated missions and their index | 24 |

## How to read a record

One JSON object per line. The first line of every file is a `meta` record with
the config, the spec set and the MD5 of every module that decides what the
measurement means. Every other line is one candidate:

```json
{
  "mission": "m007", "profile": "mixed", "tree": "E2A46941D0D2",
  "label": "baseline/per-segment/a=0", "style": "baseline", "guess": "stock",
  "timing": "per-segment", "weight": 0.0, "samples": 10,
  "converged": true, "duration_s": 5.391, "cost": 5.391, "iter_count": 61,
  "ok": false, "tube_m": 0.0287, "tracking_m": 0.0091,
  "max_dtheta_rad": 0.31, "mean_dtheta_rad": 0.18, "max_accel_m_s2": 4.9,
  "motor_violation_N": 0.0, "integrate_max_m": 0.0121,
  "reasons": ["tracking error 0.91 cm > 0.50 cm"]
}
```

The fields that are not obvious:

| field | meaning |
|---|---|
| `tree` | the tree state the record was produced under; every record in one file has the same one |
| `guess` | `stock` (one of the five production seeds), `spline` (the Bezier path family), `proportional_dt` (the length-proportional time allocation) |
| `mean_dtheta_rad` | the mean heading change between consecutive samples — the variable RQ-A identifies as the driver of the tracking error |
| `max_accel_m_s2` | the peak wheel acceleration, against the hub's DriveBase limit of about 7.8 m/s^2 |
| `tube_m` | the maximum reachability envelope radius over the path (gate: 2 cm) |
| `tracking_m` | the maximum closed-loop tracking error (gate: 0.005 m) |

## Reproducing

```powershell
python research/suite.py --count 24 --seed 7 --out research/missions
$env:OVERBOARD_TREE = "<pinned copy of the tree>"
powershell -File research/run_studies.ps1
python research/analyze.py all research/results/matrix.jsonl
```

The pinned tree these records were produced against is
[`../../research/pq_b_PINNED_TREE.md5`](../../research/pq_b_PINNED_TREE.md5).

## Caveat

Model predictions, not hardware measurements. No hardware logs exist in this
repository, and the declared uncertainty ranges are test assumptions. The
tracking numbers additionally measure a discretisation artefact (RQ-A), not a
physical tracking error.
