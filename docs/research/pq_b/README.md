# PQ-B — the research agenda, measured

**Status:** the measurement campaign is complete. **Read
[`FINDINGS.md`](FINDINGS.md)** for the answers. The generated tables and raw
records were archived in the `archive/research-2026-10` tag; retrieve a file
with `git show archive/research-2026-10:docs/measurements/...`.

**Tree under test:** pinned; the hashes are in the archive tag alongside the
raw records.

**Companions:** [`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md)
(the agenda this answers) and [`../RQ-A_tracking_error.md`](../RQ-A_tracking_error.md)
(why the tracking gate fails).

---

## The problem this solves

Every claim in the agenda rests on six example missions. The random benchmark
that was supposed to supply a mission suite was broken (KI-015, repaired in the
meantime), so the "baseline seed always wins" statement, the "the tracking gate
binds" statement and the "the weight grid is the cheapest lever" statement are
all statements about six paths.

The campaign used the measurement tooling under `research/`:

- a stratified random mission suite (24 missions over four path profiles),
- a (mission x candidate) matrix run through the **production** gate chain,
- one JSON record per candidate, tagged with the tree state it was produced
  under,
- an offline analyzer that answers the questions from the records.

Nothing in the production tree is changed by any of it. The raw records and
one-off scripts are in the archive tag, not in `main`.

## The pinned tree

The working tree is edited while a measurement runs (KI-024 landed in the
middle of the first attempt at this work). Every run therefore happens
against a **pinned copy** of the planner, the same convention `rq_a` uses:

```powershell
$env:OVERBOARD_TREE = "C:\path\to\pinned\copy"
powershell -File research/run_studies.ps1
```

The hashes of the pinned tree are in the archive tag, and every record in every
result file carries a `tree` token, so a mixed run is detectable rather than
silently wrong.

## The runs

| result file (archive) | what it measures | questions |
|---|---|---|
| `matrix.jsonl` | 32 missions x 21 candidates through the full gate chain | Q3, Q5, Q8, Q10, and the seed/timing baseline |
| `weights.jsonl` | a 9-point accuracy-weight ladder, both timing models | Q6 |
| `density.jsonl` | samples per segment in {5, 10, 20, 40} | Q7 |
| `spread.jsonl` | the same candidates validated at 5 uncertainty widths | KI-020 |
| `sc_off.jsonl`, `sc_on.jsonl` | the same seed matrix with and without `--self-consistent` (KI-026), on one tree | RQ-A's OQ6 |

## Reproducing

The tooling still lives under `research/`; the recorded runs are in
`archive/research-2026-10`.

```powershell
python research/suite.py --count 24 --seed 7 --out research/missions
$env:OVERBOARD_TREE = "<pinned copy>"
powershell -File research/run_studies.ps1
python research/analyze.py all research/results/matrix.jsonl
```

## The honesty clause

Every number in this directory is a **model prediction**. The declared
uncertainty ranges in `examples/fll_choreo.json` are test assumptions, not
measurements, and no hardware logs exist in the repository. A `PASS` means
"the trajectory passes the gates of an uncalibrated model", not "the robot
can drive it".

One caveat is inherited from RQ-A and applies to every tracking number here:
the closed-loop tracking error is a **discretisation artefact** (the
optimizer's collocation, the controller's linear reference and the plant's
integration are three different discretisations of the same trajectory). It is
not a measure of the robot's ability to follow the plan. The `mean_dtheta_rad`
field in every record is the variable RQ-A identifies as the driver, so the
artefact can be seen in the data rather than assumed.
