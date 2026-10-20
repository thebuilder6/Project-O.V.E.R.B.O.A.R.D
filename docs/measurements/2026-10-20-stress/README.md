# Stress run: many random paths, many random robots — 2026-10-20

The planner was measured against **random paths on random machines**, on the
tree the harness and the CLI measure on. Before this, every heuristic claim
in the repository rested on six hand-written paths, and the mission suite
under `research/` had never been run on the current tree.

Two things were varied:

- **the path** — 200 missions from `research/suite.py`, stratified over four
  profiles (`even`, `uneven`, `turny`, `mixed`), deterministic from a seed;
- **the robot** — `research/robots.py` moves the constants the planner reads
  (mass, inertia, wheel radius, track width, the motor's stall torque and top
  speed, the friction, the gear backlash) and moves each declared
  uncertainty range with its constant, so a drawn robot is still a robot the
  ranges describe.

Every mission was planned for **two different robots**, with the CLI's own
default candidate pair (the baseline seed on both timing models), and scored
with the production gate chain.

## How to reproduce it

```powershell
# 1. the missions (200, deterministic from the seed)
python research/suite.py --count 200 --seed 20261020 --out $env:TEMP\overboard_stress\missions

# 2. a pinned copy of the planner, because the working tree was being edited
#    while the measurement ran
robocopy . $env:TEMP\overboard_pin /E /XD .git __pycache__ research docs .venv
$env:OVERBOARD_TREE = "$env:TEMP\overboard_pin"

# 3. the stress run: 200 missions x 2 robots x 2 candidates
python research/harness.py --missions $env:TEMP\overboard_stress\missions `
    --out stress.jsonl --specs default --samples 10 --workers 14 --no-critic `
    --robot-jitter 1 --robots 2 --robot-seed 20261020 --chunk 30

# 4. the paired comparison: the same 12 missions and the same 24 robots,
#    once with the default collocation and once with --self-consistent
python research/harness.py --missions <the 12 files in missions/> `
    --out slice.jsonl --specs default --samples 10 --workers 4 --no-critic `
    --robot-jitter 1 --robots 2 --robot-seed 20261020 --chunk 12
python research/harness.py --missions <the 12 files in missions/> `
    --out slice_self_consistent.jsonl --specs default --samples 10 --workers 4 `
    --no-critic --robot-jitter 1 --robots 2 --robot-seed 20261020 --chunk 12 `
    --self-consistent
```

`research/analyze.py errors|robots` reads the files back.

| file | what it is |
|---|---|
| `stress.jsonl` | the main run: **180 candidates**, 45 missions, 90 robots. The 800-candidate run was stopped here on purpose; the numbers below are these 180 |
| `slice.jsonl` | the paired comparison, default collocation: 48 candidates, 12 missions, 24 robots |
| `slice_self_consistent.jsonl` | the same 48 candidates with `--self-consistent` |
| `slice_stock_seeds.jsonl` | the same 12 missions and 24 robots with the full stock set (5 seeds x 2 timing models, 240 candidates) |
| `missions/` | the 12 missions of the paired comparison, and their index |

## The result

| | stress run (180 candidates) | slice (48) | slice, self-consistent (48) |
|---|---:|---:|---:|
| converged | 179/180 (99.4 %) | 47/48 | 47/48 |
| **passing every gate** | **26/180 (14.4 %)** | **8/48 (17 %)** | **11/48 (23 %)** |
| duration, median | 7.34 s | 7.06 s | 6.95 s |
| envelope radius, median | 1.46 cm | 1.58 cm | 1.46 cm |
| tracking error, median | 1.21 cm | 0.93 cm | 0.59 cm |

**What rejects them** (a candidate can carry more than one reason):

| reason | stress run | slice | slice, self-consistent |
|---|---:|---:|---:|
| forward integration | 165 (92 %) | 33 | **0** |
| tracking error | 143 (79 %) | 33 | 29 |
| envelope radius | 91 (51 %) | 23 | 23 |
| constraint violations | 32 (18 %) | 6 | 3 |
| motor force | 25 (14 %) | 6 | 2 |

**By path profile** (stress run): `even` 10/48 · `uneven` 9/44 · `turny` 7/44 ·
**`mixed` 0/44**. A mission that mixes segment lengths and heading changes is
the hard case, and nothing in the default candidate set solves one.

**74 of the 90 mission-robot pairs produced no passing candidate at all.**

### What would rescue them: more seeds, on the per-segment model

The same 12 missions and the same 24 robots, solved with the full stock set
(five seeds x two timing models, 240 candidates) instead of the default pair:

| candidate set | pairs with a passing candidate |
|---|---:|
| the default pair (baseline on both timing models) | 6/24 |
| the full stock set (5 seeds x 2 timing models) | **8/24** |

**Two of the eighteen failures are rescued** — `m001#0` by the *rolling* seed
and `m005#1` by *bow_right*, and **both with the per-segment timing model**.
For the pairs both sets solve, the bigger set's best passer is the same
speed (within 0.02 s). So the other seeds buy passes, not speed, and they buy
them only on the per-segment model.

**The CLI now does this by itself (KI-048):** when a run produces nothing
usable — no candidate passes the gates — the seeds the set did not use are
tried as a rescue. On the pinned tree of this run, both rescued pairs are
now trajectories: `m001#0` is written at 7.746 s with a 1.69 cm envelope and
`m005#1` at 3.702 s with a 0.46 cm envelope, the same numbers the harness
measured. The rescue costs nothing when the plan works and one parallel
round of solves when it does not.

**16 of the 24 pairs have no passer at all**, even with ten candidates: on
those paths the gates reject every plan, whichever seed it comes from.

## What it says

### The gates, not the planner, are the binding constraint

The failures are dominated by a **discretisation mismatch**, not by physics.
The plan's samples, the controller's reference and the plant's integration are
three different discretisations of the same path, and they disagree by roughly
the heading change per sample. On the 48 candidates of the slice:

| correlated with the tracking error | Spearman |
|---|---:|
| mean heading change per sample | **0.803** |
| duration | 0.723 |
| number of segments | 0.519 |

The forward-integration gate is the same mismatch seen from the other side:
Spearman(tracking error, forward-integration error) = **0.935**. The CPU audit
integrates the plan's own (v, ω) profile with the trapezoidal rule and finds
that the samples are not exactly on the path that profile traces (KI-026).

**`--self-consistent` removes every forward-integration failure in the set**:
better on 48 of 48 candidates, worse on none, median 0.829 cm → 0.043 cm, and
three more candidates pass. The tracking failures shrink by a third and stay;
that residual is the reference-versus-plant mismatch RQ-A describes.

### The envelope is the clock again

Spearman(duration, envelope radius) = **0.923**. The reachable set widens with
elapsed time (the backlash term, `backlash/2 · t`) and with the gain band, so
a long path is punished twice — once by the tracking error and once by the
envelope. The failing envelopes have a median of 2.3 cm against a 2.00 cm
gate, so this gate is near its limit rather than absurdly over it.

### The robot decides the verdict, not only the margin

**2 of 45 missions pass on one robot and fail on the other**, and the reason
changes with it. On the slice:

```
m000   m000#0: PASS                 m000#1: tracking
m004   m004#0: integration+tracking  m004#1: constraint+motor+tracking
m008   m008#0: envelope+tracking     m008#1: envelope
m010   m010#0: PASS                 m010#1: constraint+motor
```

`m010` passes on one robot and fails the **motor** gate on the other. The
robot's numbers are guesses today (KI-020), and they decide which paths are
plannable — so calibrating the robot comes before tuning more heuristics.

### One new failure mode: the solver stops at its iteration budget

`m004#1`, per-segment timing: IPOPT exhausted its iteration budget (5 000 is
the configured limit, `optimizer.py`) after **36.5 s** and raised
`Maximum_Iterations_Exceeded`. The optimizer catches that and returns
best-effort samples — 41 of them, the same as the successful candidates — but
the candidate is not converged, and a non-converged candidate is never
eligible for export (KI-001), so it is discarded.

The same path on the **other robot** takes 0.26 s and 0.30 s to solve, and
the equal-duration candidate on the same robot needed 3 898 iterations to
converge (27.6 s). So the path is not the problem: the robot's constants
pushed the solver into a hard region where the baseline seed grinds.

**One candidate in 228 hit the limit, and in all 114 mission-robot pairs at
least one candidate converged — no mission was lost to it.** The cost in the
measured case was the 36.5 s and the loss of the faster per-segment
candidate. The default candidate set shares one seed between its two
candidates, so when *both* stop at their budget the other seeds are never
tried; the CLI now tries one more seed before giving up (KI-048).

## The tree these numbers describe

The run was measured against a **pinned copy** of the planner
(`OVERBOARD_TREE`), because the working tree was being edited while the
measurement ran — `optimizer.py` was rewritten underneath it. Every record
carries the tree token, and all 180 records carry the same one, so the run
did not straddle a change:

```
optimizer.py          62773AAD54A46EEE8D8E4F885BC8ACD0
main.py               A6DBAC62F4DC427B7680E348FDC049A2
immrax_validator.py   1D1C35F0CFF95A6BE807D28366EE91FB
validator.py          D7374FAE647300C6A3ACD85CAAD4D38D
robot_model.py        6748938B0E26034B3641FE9386ACA9ED
multiverse_optimizer.py 5D71943AD4EF313D8F9D83B9069E95F7
tree token            50F622B95CD6
```

Environment: Windows 11, Python 3.12.10, with `casadi` 3.7.2, `jax` 0.6.2,
`immrax` and `numpy` 2.3.5 installed.

## A note on honesty

Every number here is a **model prediction**. The declared uncertainty ranges
are test assumptions, not measurements, and no hardware logs exist in the
repository. A `PASS` means "the trajectory passes the gates of an uncalibrated
model", not "the robot can drive it".

The suite's paths are also longer and more turny than a real FLL mission
(up to 5.3 m and 7 segments on a 2.0 × 1.0 m field), so the 14 % pass rate is
**not** the pass rate on a match-day mission. It is the pass rate of the
planner on paths nobody designed for it, which is the number the six
hand-written missions could not give.
