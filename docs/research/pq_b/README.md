# PQ-B — the research agenda, measured

**Status:** data collection. This directory holds the tooling and the results
of a measurement campaign designed to answer the twelve questions in
[`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md)
Part IV.

**Why a second research directory:** [`../rq_a/`](../rq_a/) answers *why* the
tracking gate fails (RQ-A). This one answers *what the heuristics do* when
they are measured on more than six hand-written paths.

---

## The problem this solves

Every claim in the agenda rests on six example missions. The random benchmark
that was supposed to supply a mission suite is broken (KI-015), so the
"baseline seed always wins" statement, the "the tracking gate binds" statement
and the "the weight grid is the cheapest lever" statement are all statements
about six paths.

The tooling in [`../../research/`](../../research/) produces:

- a stratified random mission suite (24 missions over four path profiles),
- a (mission x candidate) matrix run through the **production** gate chain,
- one JSON record per candidate, tagged with the tree state it was produced
  under,
- an offline analyzer that answers the questions from the records.

Nothing in the production tree is changed by any of it.

## The pinned tree

The working tree is edited while a measurement runs (KI-024 landed in the
middle of the first attempt at this work). Every run therefore happens
against a **pinned copy** of the planner, the same convention `rq_a` uses:

```powershell
$env:OVERBOARD_TREE = "C:\path\to\pinned\copy"
powershell -File research/run_studies.ps1
```

The hashes of the pinned tree are in [`pq_b_PINNED_TREE.md5`](pq_b_PINNED_TREE.md5),
and every record in every result file carries a `tree` token, so a mixed run
is detectable rather than silently wrong.

## The runs

| file | what it measures | questions |
|---|---|---|
| `research/results/matrix.jsonl` | 32 missions x 21 candidates through the full gate chain | Q3, Q5, Q8, Q10, and the seed/timing baseline |
| `research/results/weights.jsonl` | a 9-point accuracy-weight ladder, both timing models | Q6 |
| `research/results/density.jsonl` | samples per segment in {5, 10, 20, 40} | Q7 |
| `research/results/multiverse.jsonl` | the Multi-Verse pipeline against the simple solver | Q12 |
| `research/results/spread.jsonl` | the same candidates validated at 5 uncertainty widths | KI-020 |

## Reproducing

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
