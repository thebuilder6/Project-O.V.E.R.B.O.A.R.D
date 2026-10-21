# Research tooling

Measurement code for the research agenda in
`docs/HEURISTICS_AND_VERIFICATION.md` Part IV. **Nothing here is imported by
the production code.** It measures the planner; it does not change it.

The point of this directory is that every claim in the agenda currently rests
on six hand-written example paths. These tools produce a mission suite, run a
(mission x candidate) matrix through the production gate chain, and write one
JSON record per candidate so the questions can be answered from data.

## The pieces

| File | What it does |
|---|---|
| `suite.py` | Generates the random mission suite (Q8). Stratified over four path profiles: `even`, `uneven`, `turny`, `mixed`. Deterministic from a seed. |
| `robots.py` | Generates random robots: the same missions planned for machines whose constants differ (mass, inertia, wheel radius, track width, the motor's torque and top speed, the friction, the gear backlash). A declared range moves with its constant, so a drawn robot is still a robot the ranges describe. |
| `seeds.py` | Two extra initial guesses, built outside the production tree and handed to the solver through the existing `initial_guess` argument: a cubic Bezier path family (Q1) and a length-proportional time allocation (Q2). |
| `harness.py` | Runs the matrix. Reuses the production solver, the production Immrax validator and the production CPU audit, and writes one JSON record per candidate. |
| `analyze.py` | Reads the JSONL and prints the answers. |
| `check_instrument.py` | Compares what the harness records with what the CLI's own validation report says, on the same mission and the same tree. Run this before believing anything the harness produces. |
| `peek.py` | Prints a measurement file as a compact table (quick inspection). |
| `weight_breakdown.py` | The accuracy-weight study broken down per timing model and per mission (Q6). |
| `run_studies.ps1` | The full campaign: the matrix, the weight ladder, the density study and the uncertainty sensitivity. |
| `run_self_consistent.ps1` | The same matrix with and without `--self-consistent` (KI-026), for RQ-A's OQ6. The flag is two-way (`--self-consistent/--no-self-consistent`), so the comparison runs on one tree; note that the harness follows the **library** default, which is the self-consistent rule (KI-051, KI-052). |

## Measuring against a pinned tree

The working tree is edited while a measurement runs (KI-024 landed mid-flight
once, and KI-025 after a pin). Every run therefore happens against a pinned
copy of the planner, the same convention `docs/research/rq_a` uses:

```powershell
robocopy . <pinned copy> /E /XD .git __pycache__ research
$env:OVERBOARD_TREE = "<pinned copy>"
powershell -File research/run_studies.ps1
```

`OVERBOARD_TREE` decides which copy of the planner is imported; the tooling
itself always runs from the working tree. Every record carries a `tree` token
(the hash of the measured modules), so a run that straddled a code change is
detectable rather than silently wrong.

## Running it

```powershell
# 1. the mission suite (24 missions, deterministic)
python research/suite.py --count 24 --seed 7 --out research/missions

# 2. the main matrix: every mission x 18 candidates
python research/harness.py --missions examples/example_straight.json research/missions `
    --out research/results/matrix.jsonl --specs matrix --samples 10 --workers 8

# 3. the answers
python research/analyze.py all research/results/matrix.jsonl
```

The campaign that produced the published numbers used 8 example missions and
24 generated ones (32 in total) and 21 candidates per mission; see
[`../docs/research/pq_b/FINDINGS.md`](../docs/research/pq_b/FINDINGS.md) and
[`../docs/measurements/2026-10-20/PQ-B.md`](../docs/measurements/2026-10-20/PQ-B.md).

### Spec sets

| `--specs` | Candidates per mission |
|---|---|
| `default` | the CLI's own default pair: the baseline seed on both timing models (2) |
| `seeds` | 5 stock seeds x 2 timing models, at the base weight (10) |
| `weights` | baseline x 2 timing models x the CLI sweep weights (10) |
| `wide-weights` | baseline x 2 timing models x a 9-point weight ladder (18) |
| `spline` | the Bezier path family x 2 timing models (2) |
| `proportional` | the length-proportional time allocation, per-segment (1) |
| `density` | baseline x 2 timing models x samples in {5, 10, 20, 40} (8) |
| `matrix` | `seeds` + `weights` + `spline` + `proportional`, de-duplicated (19) |

### Uncertainty sensitivity

`--spread` scales every declared uncertainty range around its nominal value:
`1` is as declared, `0` collapses the ranges to the nominal value, `2` doubles
their width. The declared ranges are test assumptions (KI-020), so this
measures how much of a PASS depends on them.

### The stress run: many paths, many robots

`suite.py` varies the path and `robots.py` varies the machine, so the
harness can plan a whole suite of random paths for random robots:

```powershell
python research/suite.py --count 200 --seed 20261020 --out $env:TEMP\overboard_stress\missions
python research/harness.py --missions $env:TEMP\overboard_stress\missions `
    --out research/results/stress.jsonl --specs default --robot-jitter 1 --robots 2
```

`--robot-jitter 1` moves every constant over the full range `robots.py`
lists; `--robot-jitter 0` keeps the declared robot. `--robots N` draws N
robots per mission, and the robot of a (mission, draw) pair is fixed by
`--robot-seed`, so a run is reproducible. Every record carries the factors
that were applied (`robot_factors`), so `analyze.py robots` can say whether
the robot's own size decides whether a path is planned.

`--specs default` is the CLI's own default pair, so a stress run over many
missions and many robots stays affordable. `analyze.py errors` lists the
solver's exceptions, which is what a stress run is for.

## The records

One JSON object per line. The first line of a file is a `meta` record with the
hashes of the package modules that decide what a measurement means, so a
measurement can always be tied to the tree state that produced it.

Each candidate record carries the spec that produced it (`label`, `style`,
`guess`, `timing`, `weight`, `samples`), the solver outcome (`converged`,
`duration_s`, `cost`, `solve_s`, `iter_count`, `segment_dt`), the gate
outcome (`ok`, `tube_m`, `tracking_m`, `motor_violation_N`, `slip_l_N`,
`slip_r_N`, `speed_violation_m_s`, `integrate_max_m`, `integrate_final_m`,
`final_wp_pos_m`, `reasons`) and the critic's opinion (`critic_windows`).
A record also carries the robot it was planned for (`robot`, `robot_factors`)
and the solver's exception text (`error`), when there was one.

## A note on honesty

Every number this directory produces is a **model prediction**. The declared
uncertainty ranges are assumptions, not measurements, and no hardware logs
exist in the repository. A `PASS` here means "the trajectory passes the gates
of an uncalibrated model", not "the robot can drive it".
