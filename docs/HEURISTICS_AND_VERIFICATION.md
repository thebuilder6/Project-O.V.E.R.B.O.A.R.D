# Path Planning Heuristics and Verification Strategy

**Status:** 2026-10-19. Describes the working tree as measured on this date.
**Scope:** every path-planning heuristic in the repository, the verification chain that judges its output, and an explicit ledger of the strategies that did not work.
**Audience:** anyone who wants to improve the planner, the validator, or both.

> **Read this first.** Every number in this document is a **model prediction**, not a hardware measurement. No hardware logs exist in the repository. The robot-side collector (`pybricks_code/sysid_digital_twin.py`) can produce them, but nothing has been captured yet. The uncertainty ranges in `examples/fll_choreo.json` are *declared test assumptions*, not calibrated hardware values. A "PASS" in this document means "the trajectory passes the gates of an uncalibrated model", not "the robot will track it".

---

## Findings at a glance

1. **The heuristic layer is small.** Five deterministic seeds, two timing models, a candidate set, and a selection rule. There is **no geometric path family** anywhere in the tree — no splines, clothoids, béziers or quintics — and the objective has no term that prefers one path over another.
2. **The seeds do not rescue the baseline, and they rarely beat it on speed.** Over the 60-candidate matrix the `baseline` seed is the fastest passing candidate in 4 of the 6 cells that produce a trajectory at all; a non-baseline seed wins the other 2 (first4 under per-segment timing by 10 %, events under per-segment timing by 1 ms). In the 6 cells where the baseline fails, **no** seed passes either — the seeds are insurance against a *bad* baseline, and in this tree the baseline is never bad on its own.
3. **The tracking gate is the binding constraint; the reachability envelope never binds.** Of 36 failing candidates, 34 failed on the 5 mm tracking gate, 6 on motor force and 13 on the forward-integration check. **Zero** failed on the 2 cm envelope threshold. The largest envelope measured anywhere was 1.75 cm.
4. **Two of the six example missions produce no output at all.** The S-curve and the stops mission fail **all 13 candidates** each, across both timing models and all five accuracy weights, in the most thorough configuration the CLI offers. Their tracking error is 0.84–2.88 cm against a 0.50 cm gate. That is a controller-model or objective problem, not a search problem.
5. **The timing model is a bigger lever than any seed.** Per-segment timing is 4.8–23.4 % faster on four of the six missions (complex 23.4 %, stops 17.7 %, first4 15.7 %, events 4.8 %), and it is what makes the events mission solvable at all. The seeds changed the outcome on 2 missions; the timing model changed it on 4. The per-segment model also had to be made legitimate at the waypoints the robot drives through (KI-024, [5.5](#55-where-the-time-allocation-may-change-ki-024)); the matrix in [Section 9](#9-measured-heuristic-comparison) was re-measured after that change.
6. **The accuracy weight is the cheapest lever of all.** On the events mission, the *same* `baseline` seed fails at weight 0 (0.52 cm tracking) and passes at weight 0.001 (0.37 cm). The sweep grid is the only candidate-set change that has ever produced a new best trajectory, and it works because it adds a genuinely different parameter.
7. **The Multi-Verse pipeline is dead weight.** Its JAX-refined guess converges to 8.1473 s against the trivial baseline's 6.0677 s, so the probe that was added to rescue it never wins. The plumbing works; the guess does not.
8. **The measurement base is six hand-written paths.** The random benchmark that was supposed to supply a mission suite is broken (KI-015), though a mission-suite harness now exists under [`research/`](../../research/). Every claim above rests on those six paths.
9. **Twelve research questions are listed in Part IV**, ordered so that the cheapest decisive experiment comes first. The first two are measurements, not experiments.
10. **RQ-A is done, and it reorders the agenda.** [`docs/research/RQ-A_tracking_error.md`](research/RQ-A_tracking_error.md) identifies why the tracking gate binds: the optimizer's trapezoidal collocation, the controller's linear reference and the plant's integration are three different discretisations of the same trajectory, and on a curved path they disagree by about a centimetre. The tracking error is **not** an uncertainty problem and **not** a physical tracking problem. Fix the discretisation before reaching for reachability in the objective. RQ-A also finds that with a *calibrated* model the binding gate flips to the **envelope**, because the measured actuator-gain band widens the reachable set.

> **Read RQ-A before changing the tracking gate, the reference interpolation, or the optimizer's kinematics.** It contains the measurements, the refuted hypotheses, and the four instrument bugs that produced convincing nonsense along the way.

---

## Table of contents

- [Part 0 — Orientation](#part-0--orientation)
  - [0.1 What this document is](#01-what-this-document-is)
  - [0.2 Scope, authority and how to read a citation](#02-scope-authority-and-how-to-read-a-citation)
  - [0.3 Notation](#03-notation)
  - [0.4 Measurement conventions](#04-measurement-conventions)
- [Part I — The path planning heuristic landscape](#part-i--the-path-planning-heuristic-landscape)
  - [1. Two heuristic systems](#1-two-heuristic-systems)
  - [2. What is not in the tree](#2-what-is-not-in-the-tree)
  - [3. The default path, end to end](#3-the-default-path-end-to-end)
  - [4. The seed heuristics](#4-the-seed-heuristics)
  - [5. The timing models](#5-the-timing-models)
  - [6. Candidate set construction](#6-candidate-set-construction)
  - [7. Scoring and selection](#7-scoring-and-selection)
  - [8. The experimental Multi-Verse path](#8-the-experimental-multi-verse-path)
  - [9. Measured heuristic comparison](#9-measured-heuristic-comparison)
- [Part II — The verification strategy](#part-ii--the-verification-strategy)
  - [10. The gate chain](#10-the-gate-chain)
  - [11. The uncertainty model](#11-the-uncertainty-model)
  - [12. Controller and plant parity](#12-controller-and-plant-parity)
  - [13. The system identification loop](#13-the-system-identification-loop)
  - [14. What each layer proves and does not prove](#14-what-each-layer-proves-and-does-not-prove)
  - [15. Test coverage map](#15-test-coverage-map)
- [Part III — The failed strategy ledger](#part-iii--the-failed-strategy-ledger)
  - [16. How to read the ledger](#16-how-to-read-the-ledger)
  - [17. Ledger: seed heuristics](#17-ledger-seed-heuristics)
  - [18. Ledger: timing models](#18-ledger-timing-models)
  - [19. Ledger: candidate set strategies](#19-ledger-candidate-set-strategies)
  - [20. Ledger: Multi-Verse internals](#20-ledger-multi-verse-internals)
  - [21. Ledger: validation and tooling](#21-ledger-validation-and-tooling)
  - [22. Ledger: silent and unlogged behaviour](#22-ledger-silent-and-unlogged-behaviour)
  - [23. Cross-cutting lessons](#23-cross-cutting-lessons)
- [Part IV — Research agenda](#part-iv--research-agenda)
  - [24. The open questions](#24-the-open-questions)
  - [25. How to report a result](#25-how-to-report-a-result)
- [Appendix A — Reproducing every table](#appendix-a--reproducing-every-table)
- [Appendix B — Symbol index](#appendix-b--symbol-index)
- [Appendix C — Known issue cross-reference](#appendix-c--known-issue-cross-reference)
- [Appendix D — Glossary](#appendix-d--glossary)

**Research notes:** [`docs/research/RQ-A_tracking_error.md`](research/RQ-A_tracking_error.md) — why the closed-loop tracking gate binds, with the measurements, the refuted hypotheses and the open items.

---

# Part 0 — Orientation

## 0.1 What this document is

This repository solves one problem: given a sequence of robot poses, produce the fastest trajectory that the robot can actually follow, and prove that the trajectory is followable.

Those two halves — *finding* the trajectory and *proving* it followable — have separate machinery in this tree:

- **The heuristics** decide *where to start looking*. The optimizer is a local NLP solver (CasADi/IPOPT). It converges to whatever local optimum its starting point suggests. The heuristics are the starting points, the timing models, the candidate set and the selection rule.
- **The verification** decides *whether to believe the result*. A chain of gates runs on every candidate; only a candidate that passes all of them can be written to disk.

Most of the engineering effort in this repository has gone into the second half. The first half is five deterministic seed styles, two timing models and a selection rule. Part I documents that first half in full, including the parts that do not work.

The third half — **what was tried and failed** — is Part III. It exists because the obvious next move ("add a smarter path planner") has already been tried several times in this codebase, in several different forms, and each time it lost to the trivial baseline. Part III records those attempts, the measured evidence, and what would have to change for a retry to be worth anything. Part IV turns the ledger into concrete experiments.

## 0.2 Scope, authority and how to read a citation

**The source is the authority.** Where this document and the code disagree, the code wins and this document is a bug. Every `file:line` citation was verified against the working tree on 2026-10-19; the tree is uncommitted work, so the line numbers are only meaningful for that snapshot.

Citations use the form `path/file.py:LINE`. A symbol is written as `Class.method` when the class is unambiguous in that file.

Two files hold most of the subject matter:

| File | Role |
|---|---|
| `optimizer.py` | The CasADi/IPOPT optimizer and the five initial-guess seeds |
| `main.py` | The CLI, the candidate set, the scoring, the selection and the export gate |
| `validator.py` | The CPU audit: forward integration, force/slip audit, metrics |
| `immrax_validator.py` | The closed-loop interval reachability check |
| `multiverse_optimizer.py` | The experimental five-phase pipeline |
| `jax_optimizer.py` | TEB/STOMP candidate generation and the JAX refiner |
| `path_planning.py` | Reeds-Shepp path generation |
| `robot_model.py` | The robot configuration and the shared force model |
| `digital_twin.py` | The optional fitted plant |
| `sysid_fit.py` | The host-side fitter |

## 0.3 Notation

| Symbol | Meaning | Unit |
|---|---|---|
| `dt` | Sample interval. One value for the whole path, or one per segment | s |
| `N` | Number of samples: `segments * samples_per_segment + 1` | – |
| `s` | `samples_per_segment`, the number of intervals per waypoint segment | – |
| `vl`, `vr` | Left/right wheel linear speed | m/s |
| `omega` | Yaw rate, `(vr - vl) / track_width` | rad/s |
| `theta` | Heading | rad |
| `tube` | Maximum reachability envelope radius over the path | m |
| `envelope` | The axis-aligned bounding box of the reachable set at one sample | m |
| `tracking` | Maximum closed-loop tracking error of the plant from the reference | m |
| `cof` | Coefficient of friction | – |
| `torque_headroom` | Fraction of the stall torque the robot may actually use | – |

Angles are radians everywhere in the source. Millimetres appear only in the Pybricks firmware and in the SysId log formats.

## 0.4 Measurement conventions

Two kinds of numbers appear here, and they are always marked:

- **Measured 2026-10-19** — produced by the sweep in [Section 9](#9-measured-heuristic-comparison) against this tree. The raw logs are in `docs/measurements/2026-10-19/`, and the exact commands are in [Appendix A](#appendix-a--reproducing-every-table).
- **Quoted** — copied from an earlier run, with the source document and date given. These are historical and may not reproduce exactly.

**Provenance.** The sweep was run against this tree state:

```text
optimizer.py          MD5 E0D3E212C8E8D1C6F1166A0623DC9BE0
main.py               MD5 3F0325ADEAF492D48D0E724FF48482B0
immrax_validator.py   MD5 39819213DD8C83A53E56F90DDA73C581
validator.py          MD5 70950A6FAAB297AEF734FD67A67B8068
robot_model.py        MD5 2B798C532F02BAE7F77DBA0A7CBE9C81
multiverse_optimizer.py MD5 D67E88CEA2A2B072B5B60F3CF52354F2
jax_optimizer.py      MD5 B6F2DD0ABAED18482A26F81F635D1ADF
path_planning.py      MD5 8E9D030AD66453295F7E22F898CD6B4E
```

Environment: Windows 11, Python 3.12.10, `casadi`, `jax`, `immrax`, `jaxopt`, `rsplan`, `numpy`, `scipy`, `pandas` all installed. Test suite state at the time of writing: `python run_tests.py` → **242 tests, 242 successes, 0 failures, 0 errors, 0 skipped**, exit code 0.

---

# Part I — The path planning heuristic landscape

## 1. Two heuristic systems

The repository contains two entirely separate planning heuristics. They share the robot model and the validator. They do not share the candidate set, the scoring, or the seeds.

### 1.1 The default system: simple CasADi solver

**Entry point:** `main.py` → `TrajectoryOptimizer.solve` (`optimizer.py:17`).

```
waypoints
   │
   ├─ seed          : optimizer.py:499  _build_initial_guess   (5 styles)
   ├─ timing model  : optimizer.py:176  one dt, or one dt per segment
   ├─ NLP solve     : optimizer.py:148  _solve_once  (IPOPT, max_iter 5000)
   ├─ format        : optimizer.py:677  format_output
   │
   ├─ CPU audit     : validator.py:321  validate_samples
   ├─ Immrax        : immrax_validator.py:436  validate_trajectory
   ├─ scoring       : main.py:1106      per-candidate tube / tracking / ok
   ├─ selection     : main.py:92        _select_attempt
   └─ export gate   : main.py:272       _export_block_reason
```

This is the default path. `main.py` uses it unless `--multiverse` is passed.

### 1.2 The experimental system: Multi-Verse

**Entry point:** `main.py` → `MasterTrajectoryOptimizer.solve` (`multiverse_optimizer.py:274`), selected by `--multiverse`.

```
waypoints
   │
   ├─ Phase 1 bootstrap      : multiverse_optimizer.py:125  PathBootstrapper.generate_baseline
   ├─ Phase 2 kinematic prof.: multiverse_optimizer.py:379  _profile_kinematics
   ├─ Phase 3 critic         : multiverse_optimizer.py:192  TrajectoryCritic.evaluate
   ├─ Phase 4 JAX refinement : multiverse_optimizer.py:307  generate_candidates_jax + jax refiner
   ├─ Phase 5 polish         : multiverse_optimizer.py:454  _global_solve
   │    └─ refine probe      : multiverse_optimizer.py:487  _probe_refined_guess
   └─ score + export gate    : main.py (shared)
```

The Multi-Verse path **does not use the five seeds** and **does not use the per-segment timing model**. It falls back to a legacy accuracy-weight ladder (`main.py:107`, `CANDIDATE_WEIGHT_STEP = 2.0`). The plumbing for seeds and timing exists only in `TrajectoryOptimizer`, which is why `main.py:147` says so explicitly.

**Current status (KI-004):** the pipeline runs and its result is discarded. Phase 5 solves from the default baseline seed; the refined trajectory is probed as a second starting point and *never wins*. See [Section 20](#20-ledger-multi-verse-internals).

## 2. What is not in the tree

This is the list a newcomer most needs, because it defines the whole research space.

| Missing | Consequence |
|---|---|
| **Any geometric path family.** No splines, clothoids, béziers, quintics, Dubins or Elford paths. | The only path primitives are a straight line between waypoints (`optimizer.py:551` `_guess_point`) and Reeds-Shepp (`path_planning.py:28`). The Reeds-Shepp planner is only reachable from the Multi-Verse bootstrap, and only when both segment endpoints have a heading. |
| **Obstacle or field geometry.** No walls, no mat boundaries, no zone constraints. | The robot is assumed to reach every waypoint in a straight line through space. |
| **A geometric term in the objective.** The objective is duration plus an optional jerk penalty (`optimizer.py:214-...`). | The optimizer has no idea what the path *looks* like. It is pushed along by the trapezoidal integration constraints and the endpoint equalities. Two paths of the same length and the same smoothness are indistinguishable to it. This is the root cause of many entries in Part III. |
| **Topology search.** No search over forward/reverse/point-turn combinations per segment. | The reverse and point-turn topologies exist only inside the JAX candidate generator (`jax_optimizer.py:61-66`), which never reaches the solver. |
| **Multi-start beyond five seeds.** | Five deterministic seeds, all from the same straight-line geometry. |
| **A reachability term.** Reachability is computed *after* the solve. | The optimizer cannot know that a path it is about to find will fail the envelope gate. See [Section 14](#14-what-each-layer-proves-and-does-not-prove). |
| **Motor or odometry dynamics in the optimizer.** | The optimizer is kinematic. The plant is modelled only in the validator. |

## 3. The default path, end to end

### 3.1 The candidate loop

`main.py` builds a candidate list, solves each candidate independently, validates each one, scores each one, and picks one.

```python
# main.py:1000-1001
use_validation = HAS_JAX and validate
specs_to_solve = candidate_specs if use_validation else candidate_specs[:1]
```

**Without `--validate`, only candidate 1 is solved.** Candidate 1 is always equal-duration `baseline`. This is deliberate: without the validator there is no way to tell whether a different candidate is better, so there is no reason to spend the time.

**With `--validate`,** `--candidates N` (default 2, maximum 5) candidates are solved. Each candidate is an independent `(seed, timing model, accuracy weight)` triple.

### 3.2 Solve

Each candidate is solved in its own NLP. With `--candidates N` and N > 1, the candidates are solved in worker processes (`main.py:222` `_solve_candidates`), with BLAS pinned to one thread per worker (`main.py:120-121`). The trajectories are identical to a sequential run; `tests/test_cli_ux.py:195` (`TestParallelCandidates`) asserts this.

Three things force the solve to stay sequential, because they need the solver in the main process:

- `--live` (the live view needs the iteration history),
- `--show-convergence` (same),
- a patched or injected optimizer class (a worker process cannot import the patched object).

**`--workers` and `--no-parallel` are not quality claims.** They change wall-clock time only. The result is identical either way.

### 3.3 Validate, score, select

Every candidate goes through the full gate chain in [Part II](#part-ii--the-verification-strategy). Each is scored on:

- `tube` — the maximum reachability envelope radius over the path,
- `tracking_m` — the maximum closed-loop tracking error,
- `ok` — did it pass every gate,
- `violation` — how badly it failed (the envelope overshoot, or the tracking error when tracking is the failure),
- `duration` — the trajectory duration,
- `cost` — the solver's objective value.

Then `_select_attempt` (`main.py:92`) picks one. See [Section 7](#7-scoring-and-selection).

### 3.4 Export

The export gate (`main.py:272`) blocks the write if the solver did not converge, or if `--validate` was requested and the selected candidate did not pass. On a blocked export, **nothing is written**: no `.traj`, no controller JSON, no Python module, no `_stats.json`. A pre-existing file at the output path is left untouched.

## 4. The seed heuristics

The five seeds are declared in one line (`optimizer.py:8`):

```python
GUESS_STYLES = ("baseline", "rolling", "bow_left", "bow_right", "turn_first")
```

They are built by `_build_initial_guess` (`optimizer.py:499`). The header comment at `optimizer.py:6-7` states the contract:

> Distinct deterministic initial guesses for candidate generation. They are only guesses: IPOPT still enforces every waypoint, stop, and force limit.

### 4.1 The shared baseline geometry

All five seeds start from the same geometry, produced by `_guess_point` (`optimizer.py:551`). For each sample at fraction `frac` along segment `i`:

- **Position** is a straight-line interpolation between the two waypoints: `p1 + frac * (p2 - p1)`.
- **Heading** depends on which endpoint headings are known:

| Known headings | Heading guess |
|---|---|
| Both `p1` and `p2` | Angle-wrapped linear interpolation: `theta1 + wrap(theta2 - theta1) * frac` |
| Only `p1` | Blend from the constrained start heading toward the path direction `atan2(dy, dx)` |
| Only `p2` | Blend from the path direction toward the constrained end heading |
| Neither | Blend from the direction to `p2` toward the direction to `p3` — a one-segment look-ahead |

The wheel speeds start at zero for every seed (`optimizer.py:556`), and the interval length starts at a flat `0.1 s` for every segment (`optimizer.py:543`).

**Invariant:** the samples that sit on a waypoint are exactly on the waypoint. The seed never moves a constrained waypoint. `tests/test_optimizer.py:295` asserts this for all five styles, and also asserts that all five styles are mutually distinct.

### 4.2 `baseline`

The geometry above, unchanged. Zero wheel speeds, flat `dt`.

**Intent:** the simplest possible feasible-ish starting point. It is the historical behaviour of the solver and the reference all other seeds are measured against.

**Cost:** zero.

**Weakness:** the guess is kinematically poor. The straight line between two waypoints with different headings is not a path a differential-drive robot can follow, so IPOPT starts far from the feasible region and has to do the real work.

### 4.3 `rolling`

Baseline geometry, then `_set_rolling_speeds` (`optimizer.py:641`) fills in wheel speeds from the guessed motion:

- Backward-difference the guessed positions to get `v` and `omega` at `dt = 0.1`.
- Convert to wheel speeds: `vl = v - omega * track/2`, `vr = v + omega * track/2`.
- Clip to `±max_linear_speed(apply_headroom=True)`.
- Force zero at the first sample, the last sample, and at every declared stop waypoint.

**Intent:** IPOPT starts *moving*. A seed with zero wheel speeds sits at a point where the trapezoidal integration constraints and the "start and end at rest" equality constraints pull in opposite directions; a moving seed gives the solver a direction.

**Cost:** one pass over the samples. Negligible.

**Weakness:** the speeds are implied by the straight-line geometry, so they are wrong wherever the real path must curve. For a sharp turn the guess says "drive fast straight ahead", which is the opposite of what the path needs.

### 4.4 `bow_left` and `bow_right`

`_offset_segment_interiors` (`optimizer.py:586`) pushes the *interior* samples of each segment sideways along the segment's left normal:

```
amplitude = min(0.10 * segment_length, 0.03)     # metres
bump(frac) = 4 * frac * (1 - frac)               # zero at both waypoints
offset = sign * normal * amplitude * bump(frac)
```

The bump is parabolic: zero at both waypoints, maximum in the middle. `bow_left` uses `sign = +1`, `bow_right` uses `sign = -1`.

**Intent:** "so IPOPT starts in an arc-shaped basin" (`optimizer.py:616`). A straight line between two waypoints with different headings is a *corner-cutting* path, and it is not one a robot can hold. Bowing the interior samples to one side turns the guess into a curve that at least resembles the arc the robot will have to drive.

**Cost:** one pass over the samples.

**Weakness:** the bow is a fixed 10 % of the segment, capped at 3 cm. It does not know anything about the turn angle. For a 90° turn on a 1 m segment it bows 3 cm to one side, which is a very slight curve, not a quarter circle. For a 10° turn on a 2 m segment it bows 3 cm, which is a large relative deviation. The amplitude is uncorrelated with the geometry it is supposed to approximate.

### 4.5 `turn_first`

`_rotate_before_driving` (`optimizer.py:608`) changes the *topology* of the guess on segments with a heading change of at least 0.4 rad (`optimizer.py:623`):

- The first `spin_samples = max(1, round(0.30 * s))` samples hold the position at `p1` and rotate the heading from the start heading to the end heading — a turn in place.
- The remaining samples drive straight to `p2` with a constant `theta_end`.

**Intent:** for a sharp turn, a differential-drive robot has two options: arc through the corner (slow, wide) or turn on the spot and then drive (fast, but it consumes the segment's length). The seed picks one of them explicitly.

**Cost:** one pass over the samples.

**Weakness:** it commits to a topology with no evidence about which one is right, and the 0.4 rad threshold is a hard-coded constant. With `s <= 1` the seed keeps the baseline geometry instead, because one interval cannot hold both a spin and a drive (`optimizer.py:611-614`).

**Known result:** the seeds are *not* uniformly worse than the baseline. On the complex mission under equal-duration timing all four non-baseline seeds converge to **slower and wider** paths (8.12–8.66 s, 1.55–1.76 cm envelope, 0.52–0.73 cm tracking) than the baseline (6.068 s / 1.22 cm / 0.49 cm), and all four fail. But under per-segment timing on `example_mission_first4`, `bow_right` is the **selected** candidate at 1.663 s, ahead of the baseline at 1.833 s. See [Section 9](#9-measured-heuristic-comparison).

### 4.6 Seed summary

| Seed | Mechanism | Code | Cost | What it biases | Measured outcome (2026-10-20) |
|---|---|---|---|---|---|
| `baseline` | straight line + heading blend, zero speed | `optimizer.py:551` | ~0 | nothing | Selected on 2 of the 6 missions; **fails** the tracking gate on the complex mission under per-segment timing and the tracking gate on events/equal-duration |
| `rolling` | + back-differenced wheel speeds | `optimizer.py:641` | ~0 | "start moving" | Selected on the events mission (4.415 s) and on first4 under the full sweep (1.796 s) — the only mission where equal-duration timing produces no passing candidate at all |
| `bow_left` | + parabolic sideways bump, +side | `optimizer.py:586` | ~0 | arc-shaped basin | Indistinguishable from the others on the straight path; fails the force gate on first4/per-segment by 0.071 N |
| `bow_right` | + parabolic sideways bump, −side | `optimizer.py:586` | ~0 | opposite arc basin | The fastest *passing* per-segment candidate on first4 (1.663 s); fails the tracking gate on the complex mission |
| `turn_first` | turn in place, then drive | `optimizer.py:608` | ~0 | turn-on-the-spot topology | Fails the force gate on first4/per-segment by 0.008 N; fastest non-baseline seed on the complex mission under equal-duration timing (8.118 s) but still fails |

**The pattern:** a non-baseline seed wins by *escaping a gate*, not by finding a better optimum. On first4/per-segment the baseline passes at 1.833 s but `bow_right` finds a faster solution at 1.663 s, while `bow_left` and `turn_first` miss the force gate by 0.071 N and 0.008 N. On events/per-segment every seed now passes within a millisecond of the others. The seeds are insurance against one seed's convergence, not a search.

## 5. The timing models

A second, orthogonal axis multiplies the five seeds into the candidate set: the **timing model**.

### 5.1 Equal-duration (`per_segment_dt = False`)

One scalar `dt` for the whole path (`optimizer.py:179`). Total time is `dt * (N - 1)`.

Because every segment has exactly `samples_per_segment` intervals, one shared `dt` means **every segment takes the same time**. On a path whose segments have very different lengths, the short segment inherits the long segment's duration. The model is not "one time step for the whole path" in spirit; it is "every segment is allotted the same duration".

On the example missions this is the model that has the most validated results, because it is the historical one. It is also the model the default candidate 1 uses.

### 5.2 Per-segment (`per_segment_dt = True`)

One `dt_i` per waypoint segment (`optimizer.py:176`), each bounded to `[0.001, 1.0] s` (`optimizer.py:241-242`). Total time is `samples_per_segment * sum(dt_i)`. The interval starting at sample `k` uses `dt[k // samples_per_segment]` (`optimizer.py:183` `interval_dt`).

**This is the general model.** Setting every `dt_i` equal reproduces the equal-duration solution exactly. The reverse is not true.

### 5.3 The jerk discipline (KI-022)

Both models are held to the same wheel-jerk discipline, so `--accuracy-weight` means the same thing in either.

The objective is (`optimizer.py:214-...`):

```text
J = duration + accuracy_weight * dt_ref^2 * sum_k [
        ((al_{k+1} - al_k) / span_k)^2 + ((ar_{k+1} - ar_k) / span_k)^2
    ]
```

where `span_k` is the spacing between the midpoints of the two intervals meeting at boundary `k`, and `dt_ref` is the mean interval over the whole path (`optimizer.py:219`, `optimizer.py:484`).

The normalisation matters. Without it, the raw acceleration jump is proportional to `jerk * dt`, so a segment with a short `dt` is charged `dt^2` less per unit of jerk and can buy violent wheel motion nearly free. That previously let the per-segment model reach **68.9 m/s³** of wheel jerk where the equal-duration model reached **9.5 m/s³** at the same weight. The normalisation makes the price of physical jerk independent of the timing layout.

### 5.4 Why the two models are not peers

`main.py:135-139` states it, and the measured behaviour confirms it: on an uneven 1.0 m + 0.2 m path, equal-duration timing forces the short 0.2 m segment to take the same time as the long one. That is a large artificial penalty, and it is why the equal-duration candidate is often much slower than the per-segment one on paths with unequal segments.

The consequence for the candidate set: **the default set mixes the two modes**, because they answer different questions. Equal-duration is the safe, historically validated option; per-segment is the faster, more general one that sometimes fails the tracking gate. Validation picks between them.

### 5.5 Where the time allocation may change (KI-024)

The per-segment model gives every waypoint segment its own interval, so the time scale can change anywhere. That is only legitimate where the state at the waypoint is fully determined, and the rule is now written down in one place (`TrajectoryOptimizer.merge_conditions`, `optimizer.py`):

| waypoint | what the two pieces must agree on | timing | force profile |
|---|---|---|---|
| stop and pinned heading | nothing: position, heading and zero velocity are all determined | free | may break |
| pinned heading, no stop | the wheel velocities there | free | must be continuous |
| stop, no pinned heading | the heading there; the velocity is zero on both sides | free | must reverse sign |
| neither | the heading and the wheel velocities | free | must be continuous |

The path is one NLP over one shared variable set, so the state agreement is automatic: position, heading and the wheel velocities at a waypoint are single variables that both pieces see. A decomposed solver (solve the pieces separately, then merge) would have to impose these conditions explicitly; the joint NLP gets them for free.

What the per-segment timing *could* break is the reading of the motion profile. The acceleration on the interval before waypoint `j` is `(vl[k]-vl[k-1])/dt[j-1]` and the one after it is `(vl[k+1]-vl[k])/dt[j]`, with `k = j*samples_per_segment`. When `dt` changes across the waypoint, the same velocity ramp is read as two different accelerations, so the wheel force profile gets a step exactly at the point the path is supposed to run through smoothly. The per-segment model therefore constrains the acceleration, and so the wheel force, to be continuous through every waypoint the robot does not stop at. The equal-duration model needs no counterpart: it never changes `dt`, so it reads the ramp the same way on both sides.

The residual is scaled by the reference interval squared so that it reads in m/s², the quantity the wheel force follows. The raw residual is in metres, where IPOPT's absolute constraint tolerance of `1e-2` would have allowed a centimetre of velocity ramp. The scaling also fixed the solver's behaviour: the complex mission's per-segment candidate went from 6.836 s in 41 iterations to 4.650 s in 203.

**Measured effect** (2026-10-19). The largest acceleration step at a waypoint the robot does not stop at:

| mission | equal-duration | per-segment, before | per-segment, after |
|---|---:|---:|---:|
| Complex mission, `accuracy_weight=0` | 3.65 m/s² | 2.99 m/s² | 1.2e-4 m/s² |
| Complex mission, `accuracy_weight=0.005` | 1.23 m/s² | not measured | 1.1e-3 m/s² |
| Uneven 1.0 + 0.2 m, `accuracy_weight=0` | 0.68 m/s² | 0.001 m/s² | 0.0 m/s² |

The large steps that remain are at the waypoints the robot *stops* at: the wheel force has to reverse sign there, so it cannot be continuous. On the complex mission the steps at the stops are 9.5 m/s² and 16.2 m/s², and they are the same in every timing mode.

Two candidates that the merge conditions used to reject now pass: `example_mission_first4.json` per-segment at `accuracy_weight=0` went from 1.661 s (rejected) to 1.833 s (accepted), and `example_with_events.json` per-segment at `accuracy_weight=0` went from 4.896 s (rejected) to 4.416 s (accepted). The complex mission is unchanged: the equal-duration candidate still wins at 6.068 s with a 1.22 cm envelope.

**The deeper cause is the sample layout, not the timing.** Every piece gets the same number of samples, so one shared `dt` gives every piece the same time regardless of its length. A 1.2 m straight line solved as one piece with 20 samples takes 2.702 s with no path excess; the same line with an interior waypoint and one shared `dt` takes 4.463 s and weaves 19.9 mm. Allocating the samples along the path by arc length would remove that penalty and leave the timing free to change only at real boundaries. It changes the sample count and the sample-to-waypoint mapping, so it is a separate change.

### 5.6 A segment-wise solver: the strategy that is not needed yet

The merge conditions in [5.5](#55-where-the-time-allocation-may-change-ki-024) are written as if the pieces were solved separately and then joined, because that is how a decomposed solver has to work. The current solver does not do that: the whole path is one NLP over one shared variable set, so the state agreement is automatic and the pieces can never disagree. The joint solve is also a global optimum of the same objective, which a sequential decomposition cannot promise.

The decomposition is worth writing down because it is the only way to make long missions tractable, and because it turns the merge conditions into something testable.

**Proposed strategy.** Split the path into *pieces*: the maximal runs of waypoint segments between legitimate timing boundaries (stop and pinned heading). Then:

1. **Global pass.** Solve the whole path once with the equal-duration model. It is the cheapest model, it converges, and it gives a smooth reference plus a sensible sample density.
2. **Piece-wise pass.** Solve each piece as its own NLP, in order. Piece `k` takes the exit state of piece `k-1` as a fixed initial condition, has a free time base inside the piece, and enforces the wheel-force continuity at the piece's internal waypoints. Warm-start each piece from the global pass.
3. **Merge.** Concatenate the pieces. The join conditions hold by construction, because each piece's exit state *is* the next piece's entry state, and the joins are at stop-and-pinned-heading points where the wheel force is allowed to reverse.

**What it buys.** The dimension of the joint NLP is `5*N` states plus the time base, and it grows with the mission: the 10-waypoint complex mission already takes 1557 IPOPT iterations and about 26 s. A per-piece solve is a handful of segments each, so the pieces can be solved in parallel and each is far better conditioned. That is the real argument for it — not the speed of the result, which the joint solve wins, but the tractability of long paths.

**What it costs.** The pieces are no longer optimised jointly, so the result is a local optimum of the decomposed problem, and the boundary state is fixed by the previous piece's solution rather than by the objective. The merge conditions are then a *correctness* requirement rather than a free consequence, and a mismatch at a join means the piece solutions have to be re-solved with the agreed state.

**What to measure first.** The cheap question is whether the decomposition is needed at all: the joint solve's iteration count against the number of waypoints. If it grows faster than linearly, the decomposition has a case; if the missions that matter stay under ten waypoints, it does not. That measurement is not in this document.

## 6. Candidate set construction

`_candidate_specs` (`main.py:124`) is the whole story.

### 6.1 The `auto` mode (default)

```python
# main.py:160-173
modes = [False, True]     # equal-duration, per-segment
style = GUESS_STYLES[(k // len(modes)) % len(GUESS_STYLES)]
specs.append({"accuracy_weight": base, "guess_style": style,
               "per_segment_dt": modes[k % len(modes)]})
```

| Candidate | Seed | Timing |
|---|---|---|
| 1 | `baseline` | equal-duration |
| 2 | `baseline` | per-segment |
| 3 | `rolling` | equal-duration |
| 4 | `rolling` | per-segment |
| 5 | `bow_left` | equal-duration |

With the default `--candidates 2`, the set is exactly `baseline/equal-duration` and `baseline/per-segment`.

### 6.2 The forced modes

`--uniform-dt` and `--per-segment-dt` (`main.py:707-713`) fix `modes` to a single entry. The seed still rotates, so `--uniform-dt --candidates 5` solves all five seeds under equal-duration timing — this is the cleanest way to compare the seeds against each other.

Both flags are rejected with `--multiverse` (`main.py:797-799`), because the Multi-Verse path has no timing plumbing.

### 6.3 The sweep grid

`--sweep` (`main.py:724`) adds the timing-mode × accuracy-weight grid:

```python
SWEEP_WEIGHTS = (0.0, 0.001, 0.002, 0.005, 0.01)    # main.py:116
```

The grid uses the `baseline` seed only, and it **adds** candidates — it never removes the default ones (`main.py:175-195`), so a sweep can only ever offer more options. A combination already in the set is not solved twice.

The rationale for the weight ladder is in the comment at `main.py:109-115`: the weight trades time against wheel-jerk discipline, and the best value depends on the path. A mission with long straight segments wants a low weight; an uneven path wants a little more.

### 6.4 The Multi-Verse ladder

`use_styles=False` (the Multi-Verse path) keeps the legacy ladder (`main.py:151-154`):

```python
[{"accuracy_weight": base + 2.0 * k, "guess_style": "baseline", "per_segment_dt": False}
 for k in range(count)]
```

This ladder was **removed from the simple path** because it traded speed for a bound that could get worse (longer trajectories accumulate more uncertainty). On the complex mission the slow 17.1 s candidate diverged while the 6.07 s one stayed bounded. It survives only on the Multi-Verse path, where the seed and timing plumbing does not exist.

### 6.5 Presets

`PRESETS` (`main.py:55`) fills in options the user did not set. Explicit flags always win (`main.py:419` `_apply_preset`).

| Preset | samples | accuracy weight | candidates | validate | export |
|---|---|---|---|---|---|
| `fast` | 8 | 0.0 | 1 | no | none |
| `balanced` | 10 | 0.0 | 2 | no | none |
| `safe` | 14 | 0.005 | 3 | **yes** | controller |

`safe` is the only preset that turns on the gate. `fast` and `balanced` both leave `--validate` off, which means they solve only candidate 1.

## 7. Scoring and selection

### 7.1 The per-candidate score

After the Immrax run, `main.py:1106-1140` computes the score:

```python
# main.py:1111-1125
for i in range(len(samples_data)):
    wp_idx = i // samples
    local_threshold = waypoint_thresholds.get(wp_idx, 0.02)   # 2 cm for transit
    env = samples_data[i].get('reachability_envelope', {})
    err_x = (env['x_max'] - env['x_min']) / 2.0
    err_y = (env['y_max'] - env['y_min']) / 2.0
    err = np.sqrt(err_x**2 + err_y**2)
    max_err_all = max(max_err_all, err)
    if env.get('unknown') or err > local_threshold:
        failed_threshold = True
        max_vio_val = max(max_vio_val, err)
```

The envelope radius is the half-diagonal of the axis-aligned bounding box of the reachable set at that sample. The threshold is **local**: sample `i` is judged against the threshold of waypoint `i // samples_per_segment`, so a segment is held to the limit of the waypoint it starts from.

Then:

```python
# main.py:1130-1140
attempt_record["ok"] = (converged and not failed_threshold and immrax_report['passed'])
attempt_record["violation"] = max_vio_val if failed_threshold else (tracking if tracking > 0.005 else 0)
```

The CPU audit (`validator.py:321`) runs on **every** candidate, not only on the winner (`main.py:1142-1152`), so a candidate that clears Immrax but breaks the force, slip or integration limits cannot block a later candidate that passes everything.

### 7.2 The selection rule

```python
# main.py:92-104
def _select_attempt(attempts):
    candidates = [a for a in attempts if a.get("converged", False)]   # KI-001
    if not candidates:
        return None, False
    passing = [a for a in candidates if a["ok"]]
    if passing:
        return min(passing, key=lambda a: (a.get("tube", float("inf")),
                                           a.get("duration", a["cost"]))), True
    return min(candidates, key=lambda a: (a["violation"], a["cost"])), False
```

In words:

1. **Non-converged candidates are never eligible.** (KI-001)
2. Among the passing candidates, **the smallest reachability envelope wins**; the shortest duration breaks ties. Accuracy first, speed second.
3. If none pass, the converged candidate with the **smallest quantified violation** wins, and `passed_any` is `False` — so the export is still blocked. The fallback exists to pick the most useful candidate to *report*, not to ship.

`tests/test_regressions.py:103` (`TestAttemptSelection`) guards the rule that a failed attempt is never preferred over a passing one.

### 7.3 The report

`_emit_validation_report` (`main.py:376`) prints one machine-readable line:

```text
##OVERBOARD-VALIDATION## {"selected": 1, "candidates": [ ... ]}
```

Each candidate carries `style`, `timing`, `duration_s`, `envelope_cm`, `tracking_cm`, `ok`, `converged`, `reasons`, and a per-segment breakdown. The browser studio parses this line to show which part of the path failed (`viz/studio.js`). Two invariants: the JSON must stay valid (no `Infinity` — `_finite` at `main.py:303` maps inf/nan to `null`), and nothing is written to disk on a failed validation, so the report has to go to stdout.

## 8. The experimental Multi-Verse path

`MasterTrajectoryOptimizer.solve` (`multiverse_optimizer.py:274`) runs five phases.

### 8.1 Phase 1 — Bootstrap

`PathBootstrapper.generate_baseline` (`multiverse_optimizer.py:125`) produces a geometric guess. Per segment it calls `_generate_segment_waypoints` (`multiverse_optimizer.py:144`):

- If **both** endpoints have a heading, it plans a Reeds-Shepp path (`path_planning.py:28`) and resamples it by arc length (`multiverse_optimizer.py:150`).
- Otherwise it falls back to `linear_interpolation_waypoints` (`path_planning.py:58`).

The Reeds-Shepp planner is configured from `multiverse.bootstrap` (`multiverse_optimizer.py:117-123`): `turning_radius` (default `0.0`) and `resolution` (default `0.05`).

> **The default configuration degenerates the bootstrap.** `examples/fll_choreo.json:46-49` sets `"turning_radius": 0.0`, and a zero turning radius makes the Reeds-Shepp path degenerate to a straight line. The configured "Reeds-Shepp bootstrap" therefore produces linear paths in the shipped example. The `resolution` is honoured.

### 8.2 Phase 2 — Kinematic profile

`_profile_kinematics` (`multiverse_optimizer.py:379`) converts the geometry into a time series without CasADi: arc-length distances, a nominal speed of 40 % of the headroom-limited maximum, a uniform `dt`, central-difference velocities with reverse-motion detection, forced stops, wheel speeds, accelerations and forces.

### 8.3 Phase 3 — Critic

`TrajectoryCritic.evaluate` (`multiverse_optimizer.py:192`) slides a two-segment window over the trajectory and flags it if **any** of six metrics exceeds its threshold:

| Metric | Definition | Threshold | Code |
|---|---|---:|---|
| Tortuosity | path length / straight distance | 1.2 | `multiverse_optimizer.py:208` |
| Yaw excess | total yaw beyond the expected heading change | 0.3 rad | `multiverse_optimizer.py:213` |
| Velocity chattering | wheel-speed zero crossings | 2 | `multiverse_optimizer.py:218` |
| Jerk cost | sum of squared acceleration jumps | 5.0 | `multiverse_optimizer.py:223` |
| Curvature cost | sum of `(dtheta/ds)^2` | 2.0 | `multiverse_optimizer.py:239` |
| Centripetal cost | sum of `(v*omega/a_max)^2` | 0.8 | `multiverse_optimizer.py:247` |

The thresholds are module constants (`multiverse_optimizer.py:91-96`). **The `critic_thresholds` config block in `examples/fll_choreo.json:41-45` is not read.** The same is true of `stomp_variants`, `stomp_noise` and `teb_weights`.

### 8.4 Phase 4 — JAX refinement

For each flagged window (`multiverse_optimizer.py:307-353`):

1. Resample the window to exactly 10 states (`resample_window_cpu`, `multiverse_optimizer.py:29`).
2. Generate candidates with `generate_candidates_jax` (`jax_optimizer.py:52`):
   - 4 heading topologies: standard, reversed (`theta + pi`), point-turn-start, point-turn-end (`jax_optimizer.py:61-66`).
   - 3 velocity modes: `+0.1`, `-0.1`, `0.0` (`jax_optimizer.py:74-78`).
   - 12 base guesses, plus **50 STOMP perturbations** (`DEFAULT_STOMP_VARIANTS`, `multiverse_optimizer.py:100`) with Gaussian noise `sigma = [0.05, 0.05, 0.1]` on `x, y, heading`, endpoints re-pinned (`jax_optimizer.py:80-84`).
   - **62 candidates in total.** `tests/test_jax_heuristics.py:33` asserts the shape.
3. Keep the 10 lowest-cost candidates (`multiverse_optimizer.py:326-327`).
4. Refine them for 100 iterations with `jaxopt.ProjectedGradient` under a projection onto the bounds (`jax_optimizer.py:105-118`).
5. Take the best, resample it back to the original sample count, and write it into the trajectory.

The local cost (`jax_optimizer.py:11`) is a time term, a dynamics-error term weighted 500, a squared constraint-violation term weighted 100, forward/reverse penalties, and the accuracy-weight jerk term.

### 8.5 Phase 5 — Polish and the refine probe

`_global_solve` (`multiverse_optimizer.py:454`) solves with the plain `TrajectoryOptimizer` — **without passing the refined trajectory**. It then probes the refined trajectory as a *second* starting point (`multiverse_optimizer.py:487` `_probe_refined_guess`):

- Convert the refined samples to a guess array (`optimizer.py:128` `guess_from_samples`).
- Run a **cold** bounded pass: `max_iter = 400` (`REFINE_PROBE_MAX_ITER`, `multiverse_optimizer.py:108`) and `warm_start_init_point = "no"`.
- Keep the probe only if it converged **and** its cost is below `primary_cost * (1 - 0.001)` (`REFINE_PROBE_MARGIN`, `multiverse_optimizer.py:112`). Below 0.1 % the difference is solver noise, not a better trajectory.

The outcome is recorded in `stats["refine_probe"]` as `{converged, cost, iterations, used}`.

### 8.6 Why the pipeline never reaches the solver

The probe converges, and then it loses. On `examples/example_complex_mission.json`:

| Starting point | Result |
|---|---|
| Default baseline guess | **6.0677 s** |
| JAX-refined guess (probe) | 8.1473 s (converged in 38 iterations) |

The probe never wins, so the refined trajectory has no effect on the final result. **The residue is the quality of the JAX refinement, not the connection to the solver.** The plumbing works; the guess is worse than the trivial one.

## 9. Measured heuristic comparison

**Measured 2026-10-20**, against the tree with the KI-024 merge conditions and the KI-025 backlash fix in place. 60 candidate solves - 6 missions x 2 timing models x 5 seeds - plus the timing-mode x accuracy-weight grid. The raw logs are in `docs/measurements/2026-10-20/`; the commands are in [Appendix A](#appendix-a--reproducing-every-table). The provenance (source hashes, environment, test state) is in [S0.4](#04-measurement-conventions). The raw logs are in `docs/measurements/2026-10-19/`; the commands are in [Appendix A](#appendix-a--reproducing-every-table). The provenance (source hashes, environment, test state) is in [§0.4](#04-measurement-conventions).

### 9.1 Method

- Robot config `examples/fll_choreo.json`, `-n 10` samples per segment, `accuracy_weight = 0` unless stated.
- Every candidate is solved, validated through the full gate chain, and scored independently.
- "Gate" is the pass/fail of the whole chain. The failure reason is quoted from the `reasons` field of the machine-readable report.
- Durations are the trajectory duration; the envelope is the maximum reachability radius over the path; tracking is the maximum closed-loop tracking error. The tracking gate is 0.50 cm, the transit envelope gate is 2 cm.

### 9.2 Table A — equal-duration timing, five seeds

| Mission | Seed | Duration s | Envelope cm | Tracking cm | Gate |
|---|---|---:|---:|---:|---|
| straight | baseline | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | rolling | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | bow_left | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | bow_right | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | turn_first | 2.377 | 0.30 | 0.34 | **PASS** |
| s_curve | baseline | 3.696 | 0.78 | 1.31 | fail — tracking 1.31 cm; integration 1.33 cm |
| s_curve | rolling | 3.788 | 0.76 | 1.45 | fail — tracking 1.45 cm; integration 1.14 cm |
| s_curve | bow_left | 3.758 | 0.74 | 1.39 | fail — tracking 1.39 cm; integration 1.19 cm |
| s_curve | bow_right | 3.793 | 0.76 | 1.46 | fail — tracking 1.46 cm; integration 1.17 cm |
| s_curve | turn_first | 3.718 | 1.07 | 1.27 | fail — tracking 1.27 cm |
| first4 | baseline | 2.176 | 0.32 | 0.40 | **PASS** |
| first4 | rolling | 2.567 | 0.34 | 0.27 | **PASS** |
| first4 | bow_left | 2.723 | 0.35 | 0.29 | **PASS** |
| first4 | bow_right | 2.593 | 0.34 | 0.29 | **PASS** |
| first4 | turn_first | 2.723 | 0.35 | 0.32 | **PASS** |
| complex | baseline | 6.068 | 1.22 | 0.49 | **PASS** |
| complex | rolling | 8.377 | 1.70 | 0.52 | fail — tracking 0.52 cm |
| complex | bow_left | 8.634 | 1.75 | 0.56 | fail — motor force 0.007 N; tracking 0.56 cm |
| complex | bow_right | 8.662 | 1.71 | 0.57 | fail — tracking 0.57 cm |
| complex | turn_first | 8.118 | 1.55 | 0.73 | fail — tracking 0.73 cm |
| stops | baseline | 6.246 | 0.94 | 0.60 | fail — tracking 0.60 cm |
| stops | rolling | 6.246 | 0.96 | 0.52 | fail — tracking 0.52 cm; integration 1.04 cm |
| stops | bow_left | 6.246 | 0.94 | 0.59 | fail — tracking 0.59 cm |
| stops | bow_right | 6.246 | 0.91 | 0.53 | fail — tracking 0.53 cm |
| stops | turn_first | 6.274 | 0.94 | 0.74 | fail — tracking 0.74 cm; integration 1.10 cm |
| events | baseline | 4.637 | 0.91 | 0.52 | fail — tracking 0.52 cm |
| events | rolling | 4.636 | 0.91 | 0.54 | fail — tracking 0.54 cm |
| events | bow_left | 4.637 | 0.92 | 0.52 | fail — tracking 0.52 cm |
| events | bow_right | 4.636 | 0.91 | 0.53 | fail — tracking 0.53 cm |
| events | turn_first | 4.637 | 0.92 | 0.52 | fail — tracking 0.52 cm |

**11 passing, 19 failing.**

### 9.3 Table B — per-segment timing, five seeds

Re-measured after the merge conditions went in (KI-024). The per-segment model now requires the wheel force profile to be continuous through every waypoint the robot does not stop at, which changes what the model is allowed to do there, and after the backlash fix (KI-025).

| Mission | Seed | Duration s | Envelope cm | Tracking cm | Gate |
|---|---|---:|---:|---:|---|
| straight | baseline | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | rolling | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | bow_left | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | bow_right | 2.377 | 0.30 | 0.34 | **PASS** |
| straight | turn_first | 2.377 | 0.30 | 0.34 | **PASS** |
| s_curve | baseline | 3.667 | 0.73 | 1.25 | fail — tracking 1.25 cm; integration 1.19 cm |
| s_curve | rolling | 3.759 | 0.81 | 1.44 | fail — tracking 1.44 cm; integration 1.27 cm |
| s_curve | bow_left | 3.723 | 0.79 | 1.48 | fail — tracking 1.48 cm; integration 1.13 cm |
| s_curve | bow_right | 3.725 | 0.79 | 1.45 | fail — tracking 1.45 cm; integration 1.11 cm |
| s_curve | turn_first | 3.720 | 1.06 | 1.26 | fail — tracking 1.26 cm; integration 1.21 cm |
| first4 | baseline | 1.833 | 0.31 | 0.38 | **PASS** |
| first4 | rolling | 1.796 | 0.31 | 0.40 | **PASS** |
| first4 | bow_left | 1.715 | 0.31 | 0.49 | fail — motor force 0.071 N |
| first4 | bow_right | 1.663 | 0.30 | 0.44 | **PASS** |
| first4 | turn_first | 1.694 | 0.30 | 0.41 | fail — motor force 0.008 N |
| complex | baseline | 4.650 | 1.01 | 0.86 | fail — motor force 0.074 N; tracking 0.86 cm |
| complex | rolling | 5.395 | 1.26 | 0.80 | fail — tracking 0.80 cm |
| complex | bow_left | 5.878 | 1.35 | 0.72 | fail — tracking 0.72 cm |
| complex | bow_right | 4.877 | 1.13 | 0.86 | fail — tracking 0.86 cm |
| complex | turn_first | 5.795 | 1.40 | 0.81 | fail — tracking 0.81 cm |
| stops | baseline | 5.138 | 0.95 | 1.81 | fail — tracking 1.81 cm; integration 1.24 cm |
| stops | rolling | 4.788 | 0.80 | 0.81 | fail — tracking 0.81 cm |
| stops | bow_left | 4.791 | 0.80 | 0.80 | fail — motor force 0.007 N; tracking 0.80 cm |
| stops | bow_right | 4.784 | 0.80 | 0.80 | fail — motor force 0.001 N; tracking 0.80 cm |
| stops | turn_first | 5.257 | 0.88 | 1.01 | fail — tracking 1.01 cm; integration 1.18 cm |
| events | baseline | 4.416 | 0.82 | 0.44 | **PASS** |
| events | rolling | 4.415 | 0.82 | 0.47 | **PASS** |
| events | bow_left | 4.416 | 0.82 | 0.45 | **PASS** |
| events | bow_right | 4.415 | 0.82 | 0.46 | **PASS** |
| events | turn_first | 4.415 | 0.82 | 0.44 | **PASS** |

**13 passing, 23 failing.**

### 9.4 Table C — what the selector chose

The rule is `min(tube, then duration)` among the passing candidates (`main.py:102`).

| Mission | Equal-duration | Per-segment | Note |
|---|---|---|---|
| straight | **bow_left** 2.377 s | **bow_left** 2.377 s | **seed-insensitive**: the path has no turn, so the geometric seeds degenerate to the same guess |
| s_curve | **no passing candidate** | **no passing candidate** | tracking 1.27-1.46 cm (equal-duration) and 1.25-1.48 cm (per-segment) on every seed |
| first4 | **baseline** 2.176 s | **bow_right** 1.663 s | 5 of 5 equal-duration and 3 of 5 per-segment seeds pass |
| complex | **baseline** 6.068 s | **no passing candidate** | per-segment misses the gate (tracking 0.72-0.86 cm); 1 of 5 equal-duration seeds pass |
| stops | **no passing candidate** | **no passing candidate** | tracking 0.52-0.74 cm (equal-duration) and 0.80-1.81 cm (per-segment) on every seed |
| events | **no passing candidate** | **rolling** 4.415 s | equal-duration misses the gate (tracking 0.52-0.54 cm); 5 of 5 per-segment seeds pass |

**6 of the 12 cells produce a trajectory at all.** The baseline is selected in 2 of them and a non-baseline seed in 4.

### 9.5 Table D — the timing-mode × accuracy-weight grid

Run with `--per-segment-dt --sweep`, so the grid adds the `baseline` seed at five weights on top of the five per-segment seeds.



### 9.6 Table E — the full sweep: what the tool can actually deliver

The default candidate set plus the timing-mode × accuracy-weight grid, both timing modes: **13 candidates per mission, 78 in total**. This is `--candidates 5 --sweep`, the most thorough configuration the CLI offers.

| Mission | Wall | Passing | Selected | Duration s | Envelope cm | Tracking cm | Fastest passing |
|---|---:|---:|---|---:|---:|---:|---|
| straight | 24.9 s | 13/13 | `baseline` (equal-duration) | 2.377 | 0.30 | 0.34 | `baseline` 2.377 s |
| s_curve | 38.3 s | **0/13** | *nothing passes* | — | — | — | — |
| first4 | 29.7 s | 12/13 | `rolling` (per-segment) | 1.796 | 0.31 | 0.40 | `rolling` 1.796 s |
| complex | 90.6 s | 4/13 | `baseline` (equal-duration) | 6.068 | 1.22 | 0.49 | `baseline` 6.068 s |
| stops | 56.7 s | **0/13** | *nothing passes* | — | — | — | — |
| events | 53.2 s | 9/13 | `rolling` (per-segment) | 4.415 | 0.82 | 0.47 | `rolling` 4.415 s |

**38 of 78 candidates pass (48 %).**

### 9.7 Why the timing models behave so differently

The segment-time readout in the logs makes the difference concrete. For the complex mission:

```text
baseline, equal-duration:  0.67 0.67 0.67 0.67 0.67 0.67 0.67 0.67 0.67 s
baseline, per-segment:     0.62 0.39 0.76 0.57 0.41 0.46 0.92 0.66 0.66 s
```

Equal-duration gives **every segment the same time**, whatever its length, because one shared `dt` and a fixed sample count per segment means each segment is allotted `samples_per_segment * dt`. The per-segment model allocates time where the path needs it.

### 9.8 What the tables say

**Finding 1 — the seeds win on speed in 4 of the 6 cells that produce a trajectory.** straight under equal-duration (bow_left, 2.377 s); straight under per-segment (bow_left, 2.377 s); first4 under per-segment (bow_right, 1.663 s); events under per-segment (rolling, 4.415 s). Everywhere else the `baseline` seed is the fastest passing candidate.

**Finding 2 — the seeds rescue nothing.** The baseline fails in 6 of the 12 cells, and in none of them does a non-baseline seed pass either. The seeds buy insurance against a *bad* baseline, and in this tree the baseline is never bad on its own: it is either the fastest passing candidate or it fails so hard that nothing passes.

**Finding 3 — the tracking gate is the binding constraint, and the envelope gate never binds.** Of the 36 failing candidates: 34 cite the tracking error, 6 the motor force, 13 the forward-integration check and **0** the envelope radius. The largest envelope measured anywhere in the matrix is 1.75 cm against a 2 cm threshold. The reachability tube — the thing the whole interval-propagation machinery exists to compute — is never the reason a candidate is rejected.

**Finding 4 — the failures cluster by mission, not by seed.** The S-curve and the stops mission fail every configuration: both timing models, all five seeds, all five weights. That is a property of the path and the controller, not of the starting point. It is also the clearest signal that those two are a *planner* problem, not a *seed* problem.

**Finding 5 — the weight is a search axis, not a dial.** On the complex mission the passing weights are not the largest ones and not the smallest one either, and the duration is not monotone in the weight. There is no weight that is "safest"; there are weights that happen to land well for a given path.

**Finding 6 — the two timing models are not peers.** Comparing the `baseline` seed under each, per-segment is faster on four of the six missions (first4 15.7 %, complex 23.4 %, stops 17.7 %, events 4.8 %), slower on the S-curve (0.8 %) and identical on the straight path. It also fails the tracking gate more often. That is exactly the trade the candidate set exists to arbitrate, and it is why both modes are in the default set.

---

# Part II — The verification strategy

## 10. The gate chain

A candidate must clear every gate below. The order matters: cheap and categorical checks first, so a solver failure is reported as a solver failure and not as a tracking error.

```text
 G0  solver convergence            main.py:1057, main.py:274
      |
 G1  CPU audit                     validator.py:321
      |  - forward integration @ 1 ms RK4     validator.py:7
      |  - per-wheel force / slip audit       validator.py:122
      |  - metrics                            validator.py:212
      |
 G2  final waypoint check          validator.py:300
      |  2 cm position, 2 cm heading
      |
 G3  Immrax reachability           immrax_validator.py:436
      |  - motor force, per-wheel slip, wheel speed
      |  - closed-loop tracking error <= 5 mm
      |  - reachability envelope
      |
 G4  local envelope threshold      main.py:1111-1125
      |  2 cm transit, 0.5 cm precision_mode, or an explicit error_threshold
      |
 G5  export gate                   main.py:272
      |  - not converged  -> block
      |  - --validate and not passed -> block
      |  - blocked -> nothing is written, exit code 1
```

### 10.1 G0 — Solver convergence

`stats["converged"]` is set by `TrajectoryOptimizer.solve`. On failure the solver still returns a best-effort sample list from `opti.debug`, but the flag is `False`, and a `False` flag makes the candidate ineligible for selection (KI-001) and blocks the export.

### 10.2 G1 — The CPU audit

`validate_samples` (`validator.py:321`) runs three things:

- **`forward_integrate`** (`validator.py:7`) re-integrates the planned wheel speeds with RK4 at 1 ms and compares against the planned samples. It rejects empty and non-monotone time vectors and returns `inf` errors instead of crashing.
- **`audit_constraints`** (`validator.py:122`) checks, per sample: the per-wheel motor force against the velocity-dependent limit, the per-wheel slip against the per-wheel normal force, the legacy total traction, and the wheel speed.
- **`compute_metrics`** (`validator.py:212`) collects duration, path length, speeds, accelerations, jerk (computed on midpoint spacing, so it is layout-independent — KI-022), tortuosity, chattering and force utilisation.

The gate (`validator.py:332-338`):

```python
ok = (errors["max_pos_error_m"] < 0.01
      and errors["final_pos_error_m"] < 0.01
      and audit["num_violating_samples"] == 0
      and audit["num_slip_points"] == 0
      and audit["speed_violation_m_s"] < 1e-6)
```

### 10.3 G2 — The final waypoint check

`check_final_waypoint` (`validator.py:300`) is the second line of defence: the exported trajectory must actually reach its last waypoint, within 2 cm of position and 2 cm of heading (`validator.py:302-303`). This one runs **even without `--validate`** (`main.py` calls it on the selected candidate before writing).

### 10.4 G3 — Immrax reachability

`ImmraxValidator.validate_trajectory` (`immrax_validator.py:436`) simulates the closed loop with interval arithmetic. Details in [Section 12](#12-controller-and-plant-parity).

The physical pass condition (`immrax_validator.py:526-533`):

```python
is_physically_safe = (
    max_motor_vio < 1e-3          # N
    and jnp.max(slip_ls) < 1e-3   # N
    and jnp.max(slip_rs) < 1e-3   # N
    and reach_report['max_tracking_error_m'] <= safety_threshold_m   # 0.005 m
    and speed_violation <= 1e-6   # m/s
    and not reach_report['diverged']
)
```

### 10.5 G4 — The local envelope threshold

Per sample, the envelope radius is compared to the local threshold (`main.py:1111-1125`). The thresholds come from the waypoint list (`main.py:527-535`):

| Waypoint kind | Threshold | Source |
|---|---|---|
| Transit (default) | 2 cm | `main.py:1114` |
| `precision_mode: true` | 0.5 cm | `main.py:535` |
| Explicit `error_threshold` | the user's value | `main.py:533` |

A sample whose envelope is marked `unknown` also fails — that is how a diverged tube is caught (see below).

### 10.6 G5 — The export gate

```python
# main.py:272-278
def _export_block_reason(stats, validate, attempt_passed):
    if not stats.get("converged", False):
        return "solver did not converge"
    if validate and not attempt_passed:
        return "validation failed"
    return None
```

A blocked export raises a `ClickException`, exits 1, and writes nothing.

### 10.7 The full threshold table

Every number in the gate chain, in one place.

| # | Threshold | Value | Where | What it means |
|---|---|---|---|---|
| 1 | Motor force violation | < 1e-3 N | `immrax_validator.py:527` | worst-case per-wheel force over the motor limit |
| 2 | Per-wheel slip (left, right) | < 1e-3 N | `immrax_validator.py:528-529` | per-wheel traction violation over the limit |
| 3 | Closed-loop tracking error | ≤ 5 mm | `immrax_validator.py:443`, checked at `:530` | max distance between the true pose and the reference. **Position only** — the heading error is reported separately (`max_heading_error_rad`) but not gated, because the heading behaviour is not yet calibrated (KI-020, RQ-A OQ4) |
| 4 | Wheel-speed violation | ≤ 1e-6 m/s | `immrax_validator.py:531` | commanded speed over the limit |
| 5 | Envelope divergence radius | > 50 mm ⇒ diverged | `immrax_validator.py:764-765` | the interval tube has grown past the point where it means anything; the sample is marked `unknown` |
| 6 | Transit envelope radius | ≤ 2 cm | `main.py:1114` | per-sample reachability half-diagonal |
| 7 | `precision_mode` envelope radius | ≤ 0.5 cm | `main.py:535` | tight waypoints |
| 8 | Explicit `error_threshold` | user value | `main.py:533` | per-waypoint override |
| 9 | Forward-integration position error | < 1 cm | `validator.py:333` | RK4 re-integration vs. the plan |
| 10 | Forward-integration final error | < 1 cm | `validator.py:334` | same, at the end |
| 11 | Violating samples | 0 | `validator.py:335` | motor/traction limit hits |
| 12 | Slip points | 0 | `validator.py:336` | per-wheel slip events |
| 13 | Speed violation | < 1e-6 m/s | `validator.py:337` | wheel speed over the limit |
| 14 | Final waypoint position | ≤ 2 cm | `validator.py:302` | the plan must arrive |
| 15 | Final waypoint heading | ≤ 2 cm (0.02 rad) | `validator.py:302` | the plan must arrive pointing the right way |
| 16 | IPOPT iterations | ≤ 5000 | `optimizer.py` solver options | `max_iter`; the cold pass in fast-polish is budgeted to 200 |
| 17 | IPOPT tolerance | 1e-2 | solver options | `tol` and `constr_viol_tol`, deliberately loose for interactive use |
| 18 | Refine-probe margin | 0.1 % | `multiverse_optimizer.py:112` | the refined guess must beat the primary solve by more than this |
| 19 | Refine-probe iteration budget | 400 | `multiverse_optimizer.py:108` | the probe must not cost more than a fraction of the primary solve |

## 11. The uncertainty model

`RobotConfig.uncertainties` (`robot_model.py:116-133`) holds seven intervals. They are what Immrax propagates.

| Interval | Default | Declared in `examples/fll_choreo.json` | Meaning |
|---|---|---|---|
| `cof` | `cof * [0.8, 1.2]` | `[1.24, 1.51]` | coefficient of friction |
| `torque_headroom` | `[0.8, 1.0]` | `[0.75, 0.9]` | usable fraction of the stall torque |
| `gear_backlash_m` | `[0.0001, 0.0006]` | `[0.0004, 0.0005]` | drivetrain backlash, as a distance; it widens the reachable set by `backlash/2 * t` and does not slow the robot down (KI-025) |
| `track_width_range` | `tw * [0.99, 1.01]` | `[0.0964, 0.0966]` | axle track |
| `wheel_radius_range` | `r * [0.99, 1.01]` | `[0.0275, 0.0285]` | wheel radius |
| `inertia_kg_m2` | `I * [0.90, 1.10]` | `[0.0008, 0.0011]` | yaw inertia |
| `t_max_range` | `t_max * [0.75, 0.95]` | *(not declared)* | motor torque limit |

Three things to keep in mind:

1. **Both bounds matter.** Immrax takes an `Interval(w_lower, w_upper)`; passing a plain array makes Immrax treat the values as constants, and only the lower bound would then have any effect (KI-003). `tests/test_regressions.py:607` (`TestReachabilityEnvelope`) guards this.
2. **The declared ranges are test assumptions.** They were narrowed to make the example missions behave, not measured from the robot. A narrow `cof` interval makes the envelope small; it does not make the robot accurate.
3. **The optimizer and the validator share one force model.** `wheel_normal_forces` (`robot_model.py:10`) is the single implementation of the weight transfer — lateral only, no front/rear split (a two-wheel robot has none) — and it is called from the optimizer, the CPU audit, the JAX model and the Immrax check (KI-010). The optimizer additionally uses the declared worst case for the coefficient of friction, the torque headroom, the wheel radius, the track width and the inertia (`robot_model.py:183` `get_robust_force_params`).

## 12. Controller and plant parity

The value of the reachability check is that it simulates roughly what the robot will do. That requires the controller, the reference, the timing and the plant to match the deployed system.

| Aspect | Validator | Deployed runtime | Parity |
|---|---|---|---|
| Control period | 20 ms (`immrax_validator.py:42`) | 20 ms `drive()` tick | **yes** |
| Plant integration | 4 × 5 ms substeps (`immrax_validator.py:43`) | — | mirrors the firmware's own substepping |
| Reference | linear interpolation of `x, y, heading, v, omega` plus the precomputed Ramsete gain `sqrt(omega^2 + b*v^2)` (`immrax_validator.py:46`) | `ramsete_fast.py` | **yes** |
| Heading unwrapping | `unroll_trajectory_headings` (`immrax_validator.py:14`) | — | prevents the ±π wrap |
| Controller | `RamseteController` (`immrax_validator.py:233`), `b = 2.0`, `zeta = 0.7` | `ramsete_fast.py` | **yes** |
| Command quantisation | whole mm/s and deg/s when `quantize=True` (`immrax_validator.py:265`) | `DriveBase` rounding | **yes** (KI-021) |
| Plant, default | `RobotOpenLoop` (`immrax_validator.py:92`): commanded wheel speed → motion. The gear backlash is a **position** uncertainty carried in the reachable set, not a velocity offset (KI-025) | - | partial |
| Plant, fitted | `DigitalTwinOpenLoop` (`immrax_validator.py:153`): 5-state `[x, y, heading, vl_actual, vr_actual]` with a first-order lag, `tau >= 0.01 s` | — | partial |

**What the default plant does not model:** the DriveBase PID, motor electrical and mechanical time constants, encoder and gyro odometry error, wheel-slip dynamics, interaction forces, and battery sag beyond the declared torque-headroom interval.

**What the fitted plant adds** (`--sysid-fit`): the measured straight-line actuator gain interval, the first-order time constant, the command delay, the odometry distance and heading scale, the gyro bias, the measured acceleration limit, and the braking shortfall.

Two constraints on the interval mode: the natural inclusion does not support `abs`, `where`, `mod`, `sinc` or `atan2`, so the interval path uses an unrolled heading and a degree-6 Chebyshev polynomial instead of the exact `sinc` (KI-014); and the conservative `interval` feedback mode diverges on long fast paths, so the default `nominal` mode feeds the controller the midpoint of the interval state.

## 13. The system identification loop

```text
  pybricks_code/sysid_digital_twin.py        (on the hub, Pybricks)
        |  console: TEST_START / META, / DATA / TEST_END,
        |           POSE,x,y,heading_deg and EVENT,name
        v
  hub_deploy.py --out session.csv    |    sysid_collect.py  (one command)
        v
  sysid_fit.py --poses poses.csv --profile examples/drivebase_profile.example.json
        |  -> sysid_fit.json
        v
  main.py --sysid-fit sysid_fit.json --validate
        |  switches Immrax to the fitted five-state plant
        v
  sysid_compare.py output.traj poses.csv --run-id <id>
        reports planned-vs-measured error and whether the measured pose
        is inside the trajectory's reachability envelope
```

### 13.1 What the collector measures

`pybricks_code/sysid_digital_twin.py` runs 16 tests: raw encoder/IMU logging at 100 Hz, gyro and accelerometer bias, straight and turn step responses (50/100/200/400/600 mm/s and 30/60/120/240 deg/s), curvature steps, a reversal test for backlash, two chirps (0.2–3 Hz) for held-out validation, an odometry test, distance and heading scale calibration, a multi-stop pose test, braking-from-speed, and a wall-stall load test.

### 13.2 What the fitter produces

`sysid_fit.py` produces `sysid_fit.json` with:

- `actuator.straight` / `actuator.turn` — gain, time constant and delay from a grid search over delay ∈ [0, 0.12] s and tau ∈ [0.01, 0.50] s (`sysid_fit.py:296`), plus a 95th-percentile acceleration limit;
- `sensors` — gyro z bias and standard deviation, accelerometer biases;
- `odometry` — distance scale, heading scale, braking slip ratio;
- `backlash` — the reversal delay;
- `validation.chirp_*_nrmse` — the held-out chirp error.

The **chirp tests are held out of the fit** and are the only genuinely independent accuracy check in the whole system.

### 13.3 What the fit does *not* yet do

- **The poses the collector reports are the hub's own odometry.** `sysid_collect.py` says so explicitly: the distance and heading scale have no ground truth and stay at 1.0 unless the operator replaces `poses.csv` with tape measurements and re-runs the fitter. The actuator part of the fit (gain, tau, delay, backlash) needs no ground truth and is calibrated either way.
- **Braking slip is loaded and reported but not applied** in the default path. The plant has no braking state, so there is nothing to apply it to.
- **No hardware logs exist in the repository.** Every number in this document is therefore a model prediction. This is KI-020.

## 14. What each layer proves and does not prove

| Layer | Proves | Does **not** prove |
|---|---|---|
| G0 convergence | IPOPT reached a local optimum satisfying its constraints | that the local optimum is the global one, or that the trajectory is followable |
| G1 forward integration | the planned wheel speeds reproduce the planned poses to under 1 cm at 1 ms | anything about the real robot; it integrates the *plan*, not the hardware |
| G1 force / slip audit | the plan does not exceed the declared per-wheel force or traction limits | that the declared limits describe this robot; the friction coefficient is an assumption |
| G2 final waypoint | the plan arrives at the last waypoint within 2 cm / 0.02 rad | that the robot arrives there |
| G3 Immrax motor / slip | the plan survives the *declared* uncertainty intervals under the simulated closed loop | that the intervals contain the real robot |
| G3 Immrax tracking | the simulated plant stays within 5 mm of the reference | that the real robot does; the plant omits the PID, motor transients, slip and odometry error. **And, per RQ-A, it does not even mean that the simulated plant tracks the reference: the number is dominated by the mismatch between the optimizer's discretisation, the reference interpolation and the plant's integration** |
| G3 Immrax envelope | the reachable set stays inside the local threshold under the declared intervals | that the reachable set is the real reachable set |
| G4 local threshold | no waypoint's envelope budget is exceeded | that the budget is the right budget. **With a calibrated model this gate, not the tracking gate, is the one that binds**: the measured actuator-gain band widens the reachable set (54 mm at ±1 % over a 1 m path, against a 20 mm threshold) while barely moving the tracking error, because the controller corrects the speed error (RQ-A finding F5) |
| G5 export gate | nothing unsafe was written to disk | that the trajectory is safe |

**The single most important sentence in this document:** reachability is a **post-solve candidate score**, not an objective term and not a constraint in the optimizer (KI-020). The optimizer does not know that a path it is about to find will fail the envelope gate. When validation fails, the planner has not learned anything; a human has to decide what to change.

## 15. Test coverage map

| Test module | Guards |
|---|---|
| `tests/test_optimizer.py` | The optimizer: seeds are distinct and keep the waypoints (`test_build_initial_guess_styles_are_distinct_and_keep_waypoints`), the timing models (`test_uniform_dt_keeps_one_interval_for_all_segments`, `test_per_segment_dt_allocates_time_by_segment`), the fast-polish parity, the iteration budget |
| `tests/test_regressions.py` | The export gate (KI-001, KI-002), the selection rule (`TestAttemptSelection`), the candidate specs (`TestCandidateSpecs`, `TestSweepSpecs`), the envelope alignment and the divergence marking (`TestReachabilityEnvelope`) |
| `tests/test_validator.py` | The CPU audit: forward integration, constraint audit, metrics |
| `tests/test_cli_ux.py` | The CLI surface, parallel == sequential (`TestParallelCandidates`), the machine-readable report and its JSON validity (`TestValidationReport`) |
| `tests/test_jax_heuristics.py` | The TEB/STOMP candidate generation (62 candidates, shape and diversity) |
| `tests/test_critic_metrics.py` | Each of the six critic metrics and the window detection |
| `tests/test_refinement_accuracy.py` | The Reeds-Shepp bootstrap against linear interpolation, the critic's detection, the refinement cost and constraint preservation |
| `tests/test_digital_twin.py` | The fitted plant: actuator states, bounded envelope, odometry scaling, command delay |
| `tests/test_sysid_fit.py` | The fitter: parameter recovery, status-line tolerance, the chirp NRMSE |
| `tests/test_sysid_compare.py` | The comparator: the envelope verdict, the waypoint mapping, the run ids |
| `tests/test_gui_server.py` | The studio server and one real solve, including that a failed solve writes no artifact |

**Coverage gap.** `tests/test_optimizer.py` contains a `TestBenchmarkDryRun` class (benchmark dry-run behaviour) that is **not registered in `run_tests.py`**, so it never runs. It is not in the `run_tests.py` class list (`run_tests.py:84-129`).

---

# Part III — The failed strategy ledger

## 16. How to read the ledger

Each entry has the same fields:

| Field | Meaning |
|---|---|
| **Strategy** | What was tried, in one line |
| **Mechanism** | How it was implemented, with the code |
| **Why it was tried** | The hypothesis behind it |
| **Outcome** | What actually happened |
| **Evidence** | The measured numbers |
| **Status** | dead / partial / dormant / open |
| **What would revive it** | The concrete change that makes a retry worthwhile |

"Dormant" means the code is present and correct but nothing reaches it. "Dead" means the code exists but its result is discarded. "Partial" means it works for some paths.

## 17. Ledger: seed heuristics

### 17.1 The four non-baseline seeds

| Field | Value |
|---|---|
| **Strategy** | Replace the trivial straight-line seed with a structured one: rolling speeds, a sideways bow, or a turn-in-place topology |
| **Mechanism** | `optimizer.py:641` (`rolling`), `optimizer.py:586` (`bow_left` / `bow_right`), `optimizer.py:608` (`turn_first`) |
| **Why it was tried** | A better starting point should find a better local optimum: fewer iterations, a shorter trajectory, a tighter envelope |
| **Outcome** | **They never find a better optimum. They do often find one that passes.** As a *quality* lever they are dead: on the complex mission all four are 2–2.6 s slower and 0.3–0.5 cm wider than the baseline. As a *gate-escape* lever they are alive: the baseline failed 8 of the 12 (mission, timing model) cells and a non-baseline seed rescued 2 of them, and on the events mission they are the only candidates that pass at the default weight |
| **Evidence** | Complex mission, equal-duration: the baseline passes at **6.068 s / 1.22 cm / 0.49 cm**; `rolling` 8.377 s / 1.69 cm / 0.52 cm, `bow_left` 8.634 s / 1.76 cm / 0.56 cm, `bow_right` 8.662 s / 1.71 cm / 0.56 cm, `turn_first` 8.118 s / 1.55 cm / 0.73 cm — **all four fail** the 5 mm tracking gate. First4, per-segment: the baseline is the **fastest** candidate at 1.661 s but fails the CPU force gate by 0.003 N, so `turn_first` (1.724 s) is selected. Events, per-segment: the baseline fails with 2.10 cm of tracking error; `rolling`, `bow_left` and `turn_first` all pass at 4.409 s. See [Section 9](#9-measured-heuristic-comparison) |
| **Status** | **partial** — dead as a quality lever, alive as a gate-escape lever |
| **What would revive it as a quality lever** | A seed that changes the *geometry* of the path, not just its sampling. The current seeds perturb a straight line; the optimizer then walks the line back to the straight line, because the objective has no term that prefers a curve ([Section 2](#2-what-is-not-in-the-tree)). Until then, expect them to keep buying gate escapes rather than speed |

### 17.2 `turn_first`'s 0.4 rad threshold

| Field | Value |
|---|---|
| **Strategy** | Switch to a turn-on-the-spot topology only when the heading change is large enough |
| **Mechanism** | `optimizer.py:623`, `abs(delta) < 0.4` skips the segment |
| **Outcome** | The threshold is a hard-coded constant with no derivation. A 23° turn gets the turn-on-the-spot topology; a 22° turn does not. There is no measurement behind the number |
| **Status** | dormant |
| **What would revive it** | Derive the threshold from the robot: the smallest turn that cannot be driven as an arc within the segment's length at the speed limit. Then measure whether the derived value beats the constant |

### 17.3 The bow amplitude

| Field | Value |
|---|---|
| **Strategy** | Bow the interior samples sideways so IPOPT starts in an "arc-shaped basin" |
| **Mechanism** | `optimizer.py:599`, `amplitude = min(0.10 * length, 0.03)` |
| **Outcome** | The amplitude is uncorrelated with the turn angle. On a 90° turn over 1 m it bows 3 cm — a slight curve, not a quarter circle. On a 10° turn over 2 m it bows the same 3 cm — a large relative deviation |
| **Status** | dormant |
| **What would revive it** | Compute the bow from the turn: the circular arc that connects the two endpoints with the given headings, then sample it. That is a real path family, not a nudge — see Q1 in [Part IV](#24-the-open-questions) |

## 18. Ledger: timing models

### 18.1 Equal-duration timing treated as a peer of per-segment

| Field | Value |
|---|---|
| **Strategy** | Offer both timing models as candidates and let validation choose |
| **Mechanism** | `main.py:160-173`, the `auto` mode |
| **Why it was tried** | The two models answer different questions; the validator should pick |
| **Outcome** | **It works, but the framing is wrong.** Equal-duration is not a peer — it is a legacy special case that imposes a large artificial penalty on paths with unequal segment lengths, because one shared `dt` forces every segment to take the same time |
| **Evidence** | On a 1.0 m + 0.2 m path at `accuracy_weight = 0.005`: per-segment **2.784 s / 0.32 cm**, equal-duration **4.456 s / 0.37 cm**. On the complex mission at weight 0: per-segment 4.650 s but 0.84 cm tracking and a 0.074 N force overshoot, equal-duration 6.068 s / 1.22 cm / 0.49 cm passing |
| **Status** | open (the framing is still wrong) |
| **What would revive it** | Make per-segment the default and keep equal-duration only as a compatibility mode. The candidate machinery is already in place |

### 18.2 Per-segment timing at `accuracy_weight = 0`

| Field | Value |
|---|---|
| **Strategy** | Pure time-optimal per-segment timing |
| **Mechanism** | `--per-segment-dt -a 0` |
| **Outcome** | **Rejected by the CPU force gate and the tracking gate** on the uneven test path, and by the 5 mm tracking gate (0.84 cm) plus a 0.074 N motor-force overshoot on the complex mission |
| **Evidence** | Uneven path: per-segment 2.710 s / 0.32 cm, but it fails the Immrax and CPU gates, so the equal-duration candidate (4.463 s / 0.35 cm) is selected instead. Complex mission: per-segment 4.650 s / 1.00 cm envelope but 0.84 cm tracking, above the 5 mm gate |
| **Status** | open |
| **What would revive it** | Either a force headroom that the optimizer and the auditor agree on (the optimizer uses the declared worst case; the auditor uses the same model, but the *selection* compares a 0.074 N overshoot against a whole second of travel), or a reachability term in the objective so the solver avoids the fast-but-untrackable region itself |

### 18.3 Tying the time base across a waypoint the robot drives through

| Field | Value |
|---|---|
| **Strategy** | Make the timing free only at the legitimate boundaries: tie `dt` across every waypoint the robot does not stop at, so a "piece" is the run of segments between two stop-and-pinned-heading points |
| **Mechanism** | Either `dt[j-1] == dt[j]` as an extra NLP constraint, or one `dt` variable per piece |
| **Why it was tried** | It is the literal reading of "the time allocation must not change where the path runs through smoothly" |
| **Outcome** | **Rejected.** The equality constraints are redundant and wreck IPOPT's conditioning (the complex mission converged in 79 iterations to a 7.85 s point with a 0.0035 N force violation, against 1557 iterations and 6.068 s for the equal-duration model). The one-variable-per-piece form is better conditioned but no better in outcome: 7.86 s on the complex mission |
| **Evidence** | The deeper reason is the *sample layout*, not the timing. Every piece gets the same number of samples, so tying `dt` gives every piece the same *time* regardless of its length, and the robot has to change speed at the interior waypoint. A 1.2 m straight line solved as one piece with 20 samples takes 2.702 s with no path excess; the same line with an interior waypoint and one shared `dt` takes 4.463 s and weaves 19.9 mm |
| **Status** | rejected; the requirement it was meant to serve is met instead by the force-continuity constraint ([5.5](#55-where-the-time-allocation-may-change-ki-024)) |
| **What would revive it** | A sample layout that follows the arc length, so a shared `dt` already gives a constant-speed traversal. That changes the sample count and the sample-to-waypoint mapping, so it is a separate change |

### 18.4 The removed accuracy-weight ladder

| Field | Value |
|---|---|
| **Strategy** | Retry the solve with a higher `accuracy_weight` when the first attempt fails |
| **Mechanism** | Was `accuracy_weight += 2.0`; now only `main.py:151-154` on the Multi-Verse path |
| **Why it was tried** | A higher weight buys smoothness, which was assumed to buy tracking |
| **Outcome** | **Removed from the simple path.** The old ladder "silently traded speed for a bound that could get worse (longer trajectories accumulate more uncertainty)". On the complex mission the slow 17.1 s candidate **diverged**, while the 6.07 s one stayed bounded |
| **Status** | dead on the simple path, still present on the Multi-Verse path |
| **What would revive it** | Only with a *reason* to expect the retry to help. If the failure mode were "the path is too jerky for the controller", a weight ladder would be the right tool. It is not that failure mode today |

## 19. Ledger: candidate set strategies

### 19.1 The sweep grid

| Field | Value |
|---|---|
| **Strategy** | Add the timing-mode × accuracy-weight grid on top of the default candidates |
| **Mechanism** | `main.py:175-195`, `SWEEP_WEIGHTS` at `main.py:116` |
| **Why it was tried** | The best weight depends on the path, and the default set only tries one |
| **Outcome** | **It works, and it is the only candidate-set strategy that has ever produced a new best trajectory.** It can only add candidates, never remove them, so it cannot make things worse — it just costs more solves and more validation passes |
| **Evidence (measured 2026-10-20)** | `example_mission_first4`: the default candidate-1 result is 2.176 s; the grid's `baseline` per-segment at weight 0 passes at 1.833 s, and `bow_right` per-segment at weight 0 is faster still at **1.663 s** — 23.5 % faster — and is the selected candidate. `example_with_events`: the default set now produces a passing candidate under per-segment timing (4.416 s), where the equal-duration `baseline` still misses the tracking gate at 0.52 cm. `example_complex_mission`: the grid adds passing `baseline` per-segment candidates at weights 0.002 (8.604 s) and 0.005 (7.250 s), but the equal-duration baseline at 6.068 s is still selected, so the complex mission is unchanged |
| **Status** | **live** (opt-in, off by default) |
| **What would revive it** | Nothing; it should probably become the default for the missions where it wins. The measurement to make next is Q6: whether the weight is a real accuracy proxy, because right now it is a lottery with five tickets |

### 19.2 Multi-start as a quality claim

| Field | Value |
|---|---|
| **Strategy** | Solve more candidates and take the best |
| **Mechanism** | `--candidates N`, `main.py:222` |
| **Outcome** | The *mechanism* works. With the current seed set it buys **gate insurance**, not speed: the winning candidate is the fastest *passing* one, and the extra seeds are what make a passing candidate exist at all on two of the six missions. It never produced the fastest trajectory |
| **Evidence** | The `baseline` seed failed on 8 of the 12 (mission, timing model) cells; a non-baseline seed passed in 2 of those. On first4/per-segment `bow_right` is the fastest candidate (1.663 s) and passes, while the baseline is 1.833 s. On events/per-segment the baseline now passes at 4.416 s and the seeds are all within a millisecond of it |
| **Status** | live; the *reason* it is live was mislabelled until now |
| **What would revive it** | A candidate set whose members are genuinely different *paths*, not just different samplings of the same path. Then multi-start becomes a search, and the cost of a ticket goes up from "insurance" to "exploration". See Q1 |

### 19.3 The presets

| Field | Value |
|---|---|
| **Strategy** | Three canned configurations | 
| **Outcome** | `fast` and `balanced` both leave `--validate` off, so they solve **one** candidate. `safe` is the only preset that turns the gate on. A user reaching for `fast` gets no safety guarantee and no documentation of that |
| **Status** | live, under-documented |
| **What would revive it** | Make the presets state their guarantees in their names, or make `fast` still validate |

## 20. Ledger: Multi-Verse internals

### 20.1 The JAX refinement

| Field | Value |
|---|---|
| **Strategy** | Generate 62 candidate paths per flagged window (4 topologies × 3 velocities + 50 STOMP), score them in parallel with `vmap`, refine the best 10 with projected gradient, and write the winner back |
| **Mechanism** | `jax_optimizer.py:52` `generate_candidates_jax`; `multiverse_optimizer.py:307-353` |
| **Why it was tried** | Sample the topology space in parallel (forward / reverse / point-turn), then locally optimise the good ones. This is the "evaluate thousands of structural seed trajectories in parallel" idea from `Future_Plans.md` |
| **Outcome** | **The refinement produces a worse guess than the trivial baseline.** The probe converges in 38 iterations to **8.1473 s** against the baseline's **6.0677 s**, so it never wins |
| **Status** | dead as a quality lever (the plumbing works, the guess does not) |
| **What would revive it** | Find out *why* the refined window is worse. The likely causes, in order: (a) the local cost function (`jax_optimizer.py:11`) is a hand-written proxy that does not match the real objective, so "best" by the proxy is not best by the truth; (b) the refinement is per-window, so it optimises each window in isolation and creates discontinuities at the window boundaries; (c) the 10-state resampling loses the timing information that the real problem depends on. **Fix the cost function before touching the search.** This is Q5 |

### 20.2 The refine probe

| Field | Value |
|---|---|
| **Strategy** | Probe the refined trajectory as a second starting point for the polish, and keep it if it wins |
| **Mechanism** | `multiverse_optimizer.py:454` `_global_solve`, `multiverse_optimizer.py:487` `_probe_refined_guess` |
| **Why it was tried** | KI-004: the refined guess was computed and then thrown away, so the whole Phase 4 cost was wasted |
| **Outcome** | **The plumbing is correct and the outcome is still nothing.** The probe runs, converges, and loses. The margin rule (0.1 %) and the iteration budget (400) are both sensible |
| **Status** | partial (KI-004) |
| **What would revive it** | The same as 20.1. The probe is a good mechanism waiting for a good guess |

### 20.3 The Reeds-Shepp bootstrap with `turning_radius = 0.0`

| Field | Value |
|---|---|
| **Strategy** | Bootstrap the geometry with a Reeds-Shepp path so the guess is kinematically valid, including reverse manoeuvres |
| **Mechanism** | `path_planning.py:28`; configured from `examples/fll_choreo.json:46-49` |
| **Outcome** | **The shipped configuration sets `turning_radius: 0.0`, which degenerates the Reeds-Shepp path to a straight line.** The "Reeds-Shepp bootstrap" in the default example produces linear paths. The bootstrap only engages when both segment endpoints have a heading |
| **Status** | **dormant in practice** |
| **What would revive it** | Set a real turning radius (the robot's actual minimum turn radius) and measure whether the bootstrap beats the linear guess. The claim "Reeds-Shepp is smoother than linear" is tested in `tests/test_refinement_accuracy.py:60`, but that test builds its own configuration, so it does not cover the shipped one |

### 20.4 The critic's config block

| Field | Value |
|---|---|
| **Strategy** | Make the critic's thresholds configurable per mission |
| **Mechanism** | `examples/fll_choreo.json:41-45` declares `critic_thresholds`; the code reads module constants instead (`multiverse_optimizer.py:91-96`) |
| **Outcome** | The config block is silently ignored. A user who tunes it sees no change |
| **Status** | **dead** |
| **What would revive it** | Read the block, with the module constants as the defaults. Trivial, and it removes a silent no-op |

### 20.5 The STOMP and TEB config blocks

| Field | Value |
|---|---|
| **Strategy** | Make the number of STOMP variants, the noise scale and the TEB topology weights configurable |
| **Mechanism** | `examples/fll_choreo.json:28-40` declares them; the code uses `DEFAULT_STOMP_VARIANTS = 50` (`multiverse_optimizer.py:100`) and `STOMP_NOISE_POS_STD = 0.05` / `STOMP_NOISE_HEADING_STD = 0.1` (`multiverse_optimizer.py:98-99`) |
| **Outcome** | Same as 20.4: silently ignored. The declared `stomp_variants: 5` is ignored in favour of 50 |
| **Status** | **dead** |
| **What would revive it** | Read the blocks |

### 20.6 The `heuristic_wins` counter

| Field | Value |
|---|---|
| **Strategy** | Record which heuristic produced the winning trajectory, so the benchmark can report heuristic effectiveness |
| **Mechanism** | `OptimizationStats.heuristic_wins` (`multiverse_optimizer.py:65`), exported in `to_dict` (`multiverse_optimizer.py:73`, the field at `:80`), read by the benchmark's "Heuristic Effectiveness" table (`benchmark.py:529-537`) |
| **Outcome** | **The field is never written to.** The benchmark table is always empty |
| **Status** | **dead** |
| **What would revive it** | Populate it in the candidate loop. The information is already there: each attempt records `guess_style` and `per_segment_dt` (`main.py:1058-1059`), and the selected index is known. This is a five-line fix that turns the benchmark from decorative into useful |

### 20.7 `enable_parallel` in the Multi-Verse config

| Field | Value |
|---|---|
| **Strategy** | Parallelise the Multi-Verse pipeline |
| **Mechanism** | `examples/fll_choreo.json:28-29` sets `enable_parallel: true`, `num_workers: 2`; `MasterTrajectoryOptimizer.__init__` accepts both (`multiverse_optimizer.py:255`) |
| **Outcome** | The flags do not change the result, and the parallel machinery in the Multi-Verse path is not used for the final solve |
| **Status** | dead |
| **What would revive it** | Nothing, unless the Multi-Verse path is revived |

## 21. Ledger: validation and tooling

### 21.1 `--workers` / `--no-parallel` as a quality lever

| Field | Value |
|---|---|
| **Strategy** | Solve candidates in parallel to make the planner faster and therefore able to try more |
| **Mechanism** | `main.py:222`, `main.py:1024-1026` |
| **Outcome** | The *speed* is real; the *quality* claim is not. The trajectories are identical to the sequential run. The flags are a wall-clock lever |
| **Status** | live, but must not be described as a quality claim |
| **What would revive it** | Nothing |

### 21.2 Reachability as a post-hoc score

| Field | Value |
|---|---|
| **Strategy** | Use the reachability envelope to choose between candidates |
| **Mechanism** | `main.py:1130`, `main.py:102` |
| **Outcome** | **It works as a selector and fails as a planner.** The gate rejects a trajectory that the optimizer produced without any way of knowing it would be rejected. The failed candidate is reported to a human, who must then decide what to change |
| **Status** | partial (KI-020) |
| **What would revive it** | Put a conservative envelope proxy in the objective (Q4) or feed the failing segment back into the solver (Q5) |

### 21.3 The random benchmark mode

| Field | Value |
|---|---|
| **Strategy** | Generate random missions and benchmark the planner on them |
| **Mechanism** | `benchmark.py --mode random`; `RandomBenchmark` at `benchmark.py:380` |
| **Outcome** | **Still broken.** The suite generator (`examples/generate_random_suite.py`) reads a missing `robot_config.json` and treats `solve()` as returning samples instead of `(samples, stats)`. The pipeline mode also does not check `stats["converged"]`, so a failed solve can be summarised as a normal result (KI-015) |
| **Status** | **open** |
| **What would revive it** | Fix it. This is the single highest-value fix in this ledger: **without a mission suite, every claim in [Section 9](#9-measured-heuristic-comparison) rests on six example paths**, and the baseline's dominance may be an artefact of those six paths (Q8) |

### 21.4 `sysid_solver.py`

| Field | Value |
|---|---|
| **Strategy** | A pandas/JAX system identification solver with hardcoded CSV paths |
| **Mechanism** | `sysid_solver.py` (1155 lines) |
| **Outcome** | **Not wired into the CLI**, does not emit the current `sysid_fit.json` format, and its CSV paths are hardcoded. `sysid_fit.py` is the live path |
| **Status** | **dead** (KI-018 territory) |
| **What would revive it** | Nothing. It should be deleted, or rewritten against the current fit format |

### 21.5 Interaction forces

| Field | Value |
|---|---|
| **Strategy** | Read the `interactions` config block and inject the measured push/pull resistance into the dynamics |
| **Mechanism** | `examples/fll_choreo.json:26` declares an empty `interactions` object; the optimizer does not read it. The design is in `Future_Plans.md` Part 3 Phase B |
| **Outcome** | **Design only.** The measured force profiles do not exist yet (`Future_Plans.md` §1.2 describes the telemetry recorder that would produce them) |
| **Status** | open, blocked on data |
| **What would revive it** | Collect the interaction telemetry first. The design is sound; the data is missing |

### 21.6 Braking slip

| Field | Value |
|---|---|
| **Strategy** | Model that the wheels turn further than the robot travels while braking |
| **Mechanism** | `digital_twin.py:126` `SlipFit`, `digital_twin.py:380` `brake_shortfall` |
| **Outcome** | **Loaded and reported, but not applied** in the default validation path, because the plant has no braking state |
| **Status** | partial |
| **What would revive it** | The fitted plant now has a braking term (a `tanh` transition below 0.05 m/s commanded speed); wire the shortfall into it and measure whether the envelope grows |

## 22. Ledger: silent and unlogged behaviour

These are not "strategies that failed" but behaviours that changed the results without being recorded where a reader would look.

### 22.1 The per-segment timing model was a jerk-penalty loophole (KI-022)

| Field | Value |
|---|---|
| **What happened** | The per-segment model was not held to the same wheel-jerk discipline as the equal-duration model. Because the raw acceleration jump scales with `jerk * dt`, a short-`dt` segment was charged `dt^2` less per unit of jerk, so per-segment timing could buy violent wheel motion nearly for free |
| **Evidence** | The per-segment model reached **68.9 m/s³** of wheel jerk where the equal-duration model reached **9.5 m/s³** at the same weight |
| **Fix** | The cost is now `dt_ref^2 * sum(J^2)` in both modes, with `dt_ref` the mean interval over the whole path (`optimizer.py:219`, `optimizer.py:484`) |
| **Status** | fixed |

### 22.2 The reachability tube lagged the reference by one sample (KI-019)

| Field | Value |
|---|---|
| **What happened** | The tube was computed against a different reference than the one the controller follows, so the reported tracking error was optimistic by one sample |
| **Fix** | The same linear reference and 20 ms sample-and-hold loop as `ramsete_fast.py`, with the plant integrated in 5 ms substeps (`immrax_validator.py:42-43`, `immrax_validator.py:46`) |
| **Status** | fixed |

### 22.3 The Immrax upper uncertainty bounds were ignored (KI-003)

| Field | Value |
|---|---|
| **What happened** | `w_flat` was passed as a plain array, which Immrax treats as constants, so only the lower bounds affected the reachable set. Torque, backlash, track width, wheel radius and inertia were therefore never checked in the upper direction |
| **Fix** | `Interval(w_lower, w_upper)` is passed instead (`immrax_validator.py:578`), and the physical parameters are wrapped as intervals too (`immrax_validator.py:456-458`) |
| **Status** | fixed |

### 22.4 A failed solve was exported as a success (KI-001) and `--validate` could not block output (KI-002)

| Field | Value |
|---|---|
| **What happened** | `main.py` did not check `stats["converged"]`, and `passed_validation` was assigned but never read. If Immrax threw, the code set `passed_validation = True` and continued. A failed validation still wrote the trajectory and exited 0 |
| **Evidence** | A solve from `(0,0,0)` to `(100,0,0)` returned `converged=False` and a trajectory that ended near `x = 20` instead of `x = 100` — written as a normal trajectory, and the separate validator still reported PASS |
| **Fix** | The export gate (`main.py:272`) blocks both cases; a failed validation exits 1 and writes nothing |
| **Status** | fixed |

### 22.5 The weight-transfer math was wrong and the tools disagreed (KI-010)

| Field | Value |
|---|---|
| **What happened** | The longitudinal term was subtracted from both wheels, and the optimizer and the validator used different formulas |
| **Fix** | One shared `wheel_normal_forces` (`robot_model.py:10`), called from all four consumers. Lateral transfer only — a two-wheel robot has no front/rear split |
| **Status** | fixed |

## 23. Cross-cutting lessons

Four lessons from the ledger, and they are the reason Part IV is shaped the way it is.

**1. The baseline is hard to beat *on speed* because the objective does not care about the path.** The objective is duration plus a jerk penalty. It has no term for "the path is smooth in space", only "the wheels move smoothly in time". Two geometrically different paths of the same length and the same wheel jerk are indistinguishable to the optimizer. So a better *seed* only helps if it puts IPOPT in a basin that is better **by the objective** — and a bowed or turn-first guess is not, because the objective rewards the straight line it converges back to. Any seed improvement must either change the objective or exploit a genuine local minimum.

**2. The seeds' real value is that they are cheap insurance against a gate, not a source of speed.** Measured over 60 candidates ([Section 9](#9-measured-heuristic-comparison)), the non-baseline seeds never produced the fastest trajectory. What they did produce was **passing** trajectories where the baseline failed: the selected candidate on two of the six missions, and the only passing set on the events mission. Because a solve costs about two seconds, buying four extra lottery tickets to escape a gate is a good trade. It should be described that way, not as an optimisation improvement.

**3. Everything that changes the objective's shape is a bigger lever than anything that changes the starting point.** The timing model (KI-022) and the accuracy weight changed results by 25–40 %. The seeds changed the outcome on 2 of 6 missions and never improved the fastest duration. The research effort should follow that ratio.

**4. The candidate set is a selection mechanism, not a search mechanism.** The candidate set works: it tries a handful of options and picks the one that passes. What it does not do is explore. The sweep grid was the only candidate-set change that ever produced a new best trajectory, and it worked because it added a genuinely different *parameter* (the weight), not a different *path*.

**5. The measurement base is too narrow to support strong claims.** Everything in [Section 9](#9-measured-heuristic-comparison) rests on six example missions, and the random benchmark that was supposed to provide a mission suite is broken (KI-015). "The baseline wins the complex mission" might be a fact about the planner, or it might be a fact about that one path. Distinguishing the two is Q8, and it is cheap to do.

---

# Part IV — Research agenda

## 24. The open questions

Each question has the current evidence, a hypothesis, and an experiment with a metric and a success criterion. The experiments are ordered so that the cheap, decisive ones come first.

### Q1 — Does a curvature-continuous path family beat the straight-line seed?

| | |
|---|---|
| **Question** | The seeds perturb a straight line. Would a seed built from a real path family — a circular arc, a clothoid, a β-spline through the waypoints — find better optima? |
| **Current evidence** | All four structured seeds lose (17.1). The tree contains no geometric path family at all ([Section 2](#2-what-is-not-in-the-tree)). Reeds-Shepp exists but is only reachable from the Multi-Verse bootstrap, and only with both headings known (20.3). |
| **Hypothesis** | A seed whose geometry is *feasible for a differential-drive robot* (curvature continuous, reachable) starts IPOPT in a much better basin, and the resulting trajectory is both shorter and tighter. |
| **Experiment** | 1. Implement a geometric path builder in `path_planning.py`: given two poses, produce a curvature-continuous path (Dubins for forward-only, Reeds-Shepp with a real turning radius, or a clothoid-spline for the general case). 2. Add a `guess_style` that uses it, with the timing profile from `_set_rolling_speeds`. 3. Add it to `GUESS_STYLES` and let the candidate set pick it up — no other change is needed. 4. Run the [Section 9](#9-measured-heuristic-comparison) matrix with the new seed added. |
| **Metric** | Trajectory duration, envelope radius, tracking error, IPOPT iteration count, wall time. |
| **Success criterion** | The new seed wins on at least 2 of the 6 missions *and* does not lose on the rest — i.e. it becomes the selected candidate somewhere. A seed that is sometimes better but usually worse is not useful, because it costs a solve. |
| **Cost** | Medium. One new module, one new style, and the existing matrix. |
| **Risk** | If the objective really is indifferent to geometry (lesson 23.1), this fails and the answer is "the seed is not the lever" — which is itself worth knowing. |

### Q2 — Does a time-allocation warm start reduce iterations?

| | |
|---|---|
| **Question** | Every seed starts with a flat `dt = 0.1 s` for every segment (`optimizer.py:543`). Would a segment-length-proportional allocation converge faster and better? |
| **Current evidence** | The timing model is the single biggest lever measured so far (lesson 23.2), yet the *initial* time allocation is the crudest possible guess. |
| **Hypothesis** | An initial `dt_i` proportional to the segment length (so every segment takes time proportional to its distance) starts IPOPT much closer to the time-optimal allocation, cutting iterations and possibly finding a better optimum. |
| **Experiment** | 1. In `_build_initial_guess`, replace the flat `initial_dt` with `dt_i = clamp(k * length_i, 0.001, 1.0)`, with `k` chosen so the total matches the flat guess. 2. Solve the matrix and compare iteration counts and durations against the flat allocation. 3. Try a trapezoidal velocity profile along the baseline geometry as a second variant. |
| **Metric** | IPOPT iterations, wall time, final duration, envelope. |
| **Success criterion** | A material reduction in iterations (say > 20 %) with no regression in duration or envelope. |
| **Cost** | Low. One function, one flag. |
| **Note** | This is the cheapest experiment on this list, and it tests the same "the initial guess is not the lever" hypothesis from a different angle. |

### Q3 — Is multi-start worth anything at all?

| | |
|---|---|
| **Question** | The candidate set solves N trajectories and picks one. Is N > 1 buying anything, and if so what? |
| **Current evidence** | Measured over 60 candidates: the non-baseline seeds produce the fastest trajectory only once (complex/per-segment, where `bow_left` is 0.5 s faster but *fails*), and the fastest **passing** trajectory only on the events mission, by 2 ms. What they reliably produce is **passing** trajectories where the baseline fails: the baseline failed 8 of the 12 (mission, timing model) cells and a non-baseline seed rescued 2 of them |
| **Hypothesis** | With the current seed set, `--candidates N` buys **gate insurance**: it multiplies the chance that at least one converged solution satisfies the force and tracking gates. It does not buy speed |
| **Experiment** | 1. Instrument the selection: record the selected candidate's `guess_style`, the fastest candidate's `guess_style`, and the failing reasons per mission. 2. Compute, for each mission, whether the baseline alone would have produced a passing trajectory. 3. Repeat on a larger mission set (Q8) and report the insurance rate |
| **Metric** | The fraction of missions where a non-baseline seed is the only source of a passing trajectory; the duration cost of that insurance |
| **Success criterion** | A clear answer either way, and a cost figure: "one extra seed buys a passing trajectory on N % of missions, at M seconds per solve". If N is small, the candidate count should be cut and the budget spent on the sweep grid, which also wins (19.1) |
| **Cost** | Low. The data is already in `attempt_record` (`main.py:1058-1059`). This is a measurement, not an experiment |

### Q4 — Should reachability be an objective term?

| | |
|---|---|
| **Question** | The planner produces trajectories that fail the envelope gate and hands them to a human (21.2). Can the envelope be approximated cheaply enough to put inside the objective? |
| **Current evidence** | The reachability computation is an interval propagation over the closed loop — far too expensive for the inner NLP loop. But the *dominant* contributor to the envelope is usually the model uncertainty (backlash, wheel radius, track width) propagated over the path, not the tracking error. **RQ-A changes the priority:** the envelope is not what is failing. Across the 60-candidate matrix, zero candidates failed on the envelope threshold; 36 failed on the tracking gate, which RQ-A shows is a discretisation artefact rather than an uncertainty or tracking problem. An envelope term would not touch it |
| **Hypothesis** | A differentiable, conservative proxy for the envelope — for example the accumulated uncertainty of the pose along the path, weighted by the local speed — captures most of the envelope growth and can be added as an objective term at a small cost in duration. |
| **Experiment** | 1. Instrument the current validator to report which uncertainty component dominates the envelope at each sample. 2. Fit a cheap analytic proxy to that growth. 3. Add `w_envelope * proxy` to the objective. 4. Compare the envelope of the resulting trajectory against the current best. |
| **Metric** | Envelope radius, duration, tracking error. |
| **Success criterion** | A trajectory with a smaller envelope at a comparable duration — i.e. the gate passes *by design* rather than by selection. |
| **Cost** | High. This is the biggest change in the list and it touches the objective, the solver and the validator. |
| **Note (revised by RQ-A)** | **Do this second, not first.** The failing missions fail on the tracking gate, and the tracking gate is currently measuring a discretisation artefact, not robustness. Fix the discretisation first; otherwise the envelope term is being tuned against a moving baseline. See [RQ-A](research/RQ-A_tracking_error.md) |

### Q5 — Should the critic feed back into the solver?

| | |
|---|---|
| **Question** | The critic already identifies bad windows with six interpretable metrics (8.3). Can a flagged window be turned into a *seed* for a targeted re-solve, instead of being refined by a proxy cost function? |
| **Current evidence** | The critic's output goes into the JAX refiner, which produces a guess that loses to the baseline (20.1). The metrics themselves are tested and are meaningful (`tests/test_critic_metrics.py`). |
| **Hypothesis** | Using the critic to *select* which windows to re-solve with the real solver — rather than to refine them with a proxy — keeps the real objective in the loop and would fix the "proxy mismatch" failure mode of 20.1. |
| **Experiment** | 1. After a solve, run the critic on the result. 2. For each flagged window, re-solve the *whole* path with that window constrained to a smooth continuation of its neighbours (or simply re-solve from the current solution as the warm start). 3. Measure the critic's metrics and the gate outcome before and after. 4. Compare against the current JAX refinement on the same windows. |
| **Metric** | Critic metrics on the flagged windows, envelope, duration, iterations. |
| **Success criterion** | The re-solve reduces the critic's metrics *and* keeps the trajectory passing the gate. |
| **Cost** | Medium. |

### Q6 — Does `accuracy_weight` actually buy accuracy?

| | |
|---|---|
| **Question** | `--accuracy-weight` is documented as the plan-time proxy for tracking accuracy. Is it? |
| **Current evidence** | The weight penalises wheel jerk. The tracking gate is 5 mm. Nobody has measured the relationship between the two. |
| **Hypothesis** | Tracking error correlates with wheel jerk, but not linearly, and the correlation depends on the path. If so, the knob is real but path-dependent, and the sweep grid (19.1) is the right way to use it. |
| **Experiment** | For one mission, sweep `accuracy_weight` over a wide range (0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1) in *both* timing modes, and record the tracking error, the envelope, the duration and the peak wheel jerk for each. Plot them. |
| **Metric** | Tracking error vs. weight; duration vs. weight; the knee of each curve. |
| **Success criterion** | A measured, quotable relationship that can replace the "usually reduces jerk at the cost of a longer trajectory" hand-wave in `docs/EXAMPLES.md:615`. |
| **Cost** | Low. The sweep machinery exists (`--sweep` uses the same weights). |
| **Note** | Do this before Q4. If the jerk weight is not a good accuracy proxy, then an envelope term in the objective is not optional — it is mandatory. |

### Q7 — Does sample density beat timing flexibility?

| | |
|---|---|
| **Question** | Per-segment timing and higher sample density are two ways to give the solver more freedom. Which one pays? |
| **Current evidence** | `--samples` (2–200) is untested against `--per-segment-dt`. The current candidate set varies the timing model but fixes the density. |
| **Hypothesis** | At a fixed solve time, per-segment timing at low density beats uniform timing at high density, because it changes the *shape* of the solution rather than the resolution of it. |
| **Experiment** | Fix a mission. Compare (uniform, n=10), (per-segment, n=10), (uniform, n=20), (per-segment, n=20), (uniform, n=40) at equal wall-clock budget. |
| **Metric** | Duration, envelope, tracking, wall time, iterations. |
| **Success criterion** | A clear ordering at equal wall time. |
| **Cost** | Low. |

### Q8 — Is the baseline's dominance an artefact of the example missions?

| | |
|---|---|
| **Question** | Every conclusion in [Section 9](#9-measured-heuristic-comparison) rests on six hand-written example paths. Does the baseline still win on paths nobody designed for it? |
| **Current evidence** | The random benchmark is broken (21.3, KI-015). The example missions are geometrically simple: short, mostly straight, few segments. And the answer is already nuanced on the six we have: the baseline is selected on 2 of 6, a non-baseline seed on 2 of 6 (one of them a tie), and 2 of 6 have no passing candidate at all under the most thorough configuration tried |
| **Hypothesis** | On paths with many sharp turns, close waypoints, or segments whose optimal durations differ by an order of magnitude, the structured seeds *will* win more often, because the baseline's straight-line guess is then a poor approximation of anything |
| **Experiment** | 1. Repair the random benchmark (`benchmark.py --mode random`): fix the config path, fix the `solve()` unpacking, and check `stats["converged"]` in pipeline mode. 2. Generate 50–100 random missions with a realistic waypoint distribution (segment lengths 0.2–2 m, heading changes 0–180°, optional stops and events). 3. Run the seed matrix on each. 4. Report the win rate per seed per timing mode, and the rate at which *no* candidate passes. |
| **Metric** | Win rate per seed per timing mode; the distribution of the duration gap between the best and the baseline; the **no-pass rate** |
| **Success criterion** | A win rate with a confidence interval. The no-pass rate is the more important number: on the six example missions, 2 of 6 have no passing candidate under the default set, and that is the number a mission suite needs to move |
| **Cost** | Medium, and the repair is a prerequisite for the rest of the ledger to be trustworthy. **Do this first.** |

### Q9 — What would make the validator a real digital twin?

| | |
|---|---|
| **Question** | Which missing model component buys the most gate accuracy per unit of effort? |
| **Current evidence** | KI-020 lists the gaps: motor transients, DriveBase PID, encoder/gyro odometry, wheel-slip dynamics, braking slip not applied, interaction forces. The fitted plant now adds the actuator gain, tau, delay, odometry scale, gyro bias, an acceleration limit and a braking shortfall |
| **Hypothesis** | The single largest remaining error source is **odometry error feeding back into the controller**, because the deployed controller never sees the true pose. The fitted plant now models that belief (`_advance_belief`), which makes this testable |
| **Experiment** | 1. Collect a real session with `sysid_collect.py`. 2. Fit it. 3. Compare the fitted validator's predicted poses against the measured poses with `sysid_compare.py`, and against the *default* kinematic validator. 4. Report which model predicts the measured poses better, and by how much |
| **Metric** | Position and heading error at each measured stop, for the default model versus the fitted model versus the tape-measured truth. |
| **Success criterion** | A measured, ranked list of which missing model component matters most. |
| **Cost** | High — it needs hardware. **It is also the only way to turn any number in this document from a prediction into a measurement.** |
| **Note (revised by RQ-A)** | This is still the right long-term work, but it is **not what is blocking the missions today.** RQ-A shows the tracking error is already ~100 % explained by the current model: it is the mismatch between the optimizer's discretisation, the reference interpolation and the plant's integration. Adding motor dynamics, odometry and slip will not move the number that rejects the S-curve and the stops mission. Q9 is about making the *envelope* honest; the tracking gate needs a different fix |

### Q10 — Is "smallest envelope" the right selection objective?

| | |
|---|---|
| **Question** | The selection rule is `min(tube, then duration)` (`main.py:102`). Is the envelope the right thing to minimise? |
| **Current evidence** | The envelope is a conservative proxy that mixes uncertainty growth along the path with tracking error. Minimising it prefers slow, cautious trajectories, and it does not distinguish "the tube is wide because the robot went fast" from "the tube is wide because the robot drifted" |
| **Hypothesis** | A better objective is the *maximum per-sample excess over the local threshold* (which is what actually fails the gate) combined with the duration, or the tracking error alone. Under that objective some currently-rejected candidates would pass, and the selected trajectory would often be faster |
| **Experiment** | 1. Log, for every candidate in the matrix, the full score vector: tube, per-sample excess, tracking error, duration, peak jerk. 2. Re-run the selection offline under several candidate rules. 3. Compare how often each rule picks a passing candidate and how much time it loses |
| **Metric** | Pass rate and selected duration under each rule. |
| **Success criterion** | A rule that passes at least as often and selects a faster trajectory. |
| **Cost** | Low. The scores are already computed; only the comparison rule changes. |

### Q11 — Should the executor be part of the planner?

| | |
|---|---|
| **Question** | The validator simulates the deployed controller. Should the *plan* also account for the fact that the robot stops and re-accelerates at every stop, and that the DriveBase rounds the commands? |
| **Current evidence** | The DriveBase limits and rounding were fixed once (KI-021) and the validator now quantises commands (`immrax_validator.py:265`). But the *optimizer* has no term for either |
| **Hypothesis** | Modelling the executor's quantisation and acceleration limits inside the optimizer would remove a class of late-stage gate failures that no amount of seed or timing work can fix |
| **Experiment** | Quantise the wheel speeds and the acceleration inside the optimizer's constraint set (or add a penalty for the quantisation error), and compare the gate outcome against the current model |
| **Metric** | Gate pass rate, tracking error, duration. |
| **Success criterion** | A measurable reduction in gate failures on missions that currently fail for tracking reasons. |
| **Cost** | High. Quantisation inside an NLP is awkward — it is a step function. Start with the acceleration limit, which is smooth. |

### Q12 — What should the Multi-Verse path become?

| | |
|---|---|
| **Question** | The five-phase pipeline is 400 lines of code that produces a trajectory the solver then discards (20.1, 20.2). Keep it, fix it, or delete it? |
| **Current evidence** | `docs/trajectory_tools_research.md:102` already says: "Connect or remove Multi-Verse (KI-004). Add a way to pass a full refined trajectory into the simple solver and use it in the final polish, or delete the experimental path and worker flags." The connection now exists; the guess is still not good enough |
| **Hypothesis** | If Q1 (real path families) or Q5 (critic feedback) produces a good starting point, the Multi-Verse machinery becomes a *multi-seed generator* and its value is the topology sampling (forward / reverse / point-turn), which nothing else in the tree covers |
| **Experiment** | 1. Do Q1. 2. If the new path family beats the baseline, feed it into the Multi-Verse bootstrap and see whether the TEB topologies add anything on top. 3. If neither works, delete the Multi-Verse path and its flags, and note it in KNOWN_ISSUES |
| **Metric** | Whether the topology sampling produces a winner the path family alone does not. |
| **Success criterion** | A decision, made on evidence, with the deletion recorded if that is the answer. |
| **Cost** | Low to decide, high to execute. |

### 24.1 Suggested order

```text
  ---  DONE: RQ-A  the tracking gate's cause is identified. See
                docs/research/RQ-A_tracking_error.md. It reorders Q4 and Q9.
  OQ1  explain the residual tracking floor at fine sampling   (RQ-A open item)
  OQ2  reformulate the optimizer so the plan is self-consistent
  Q8  repair the random benchmark and get a mission suite   (prerequisite)
  Q3  instrument the selection, get the win-rate statistics  (cheap, decisive)
  Q6  measure weight vs tracking error                       (cheap; re-run after the fix)
  Q2  time-allocation warm start                             (cheap, tests lesson 23.1)
  Q1  real path family as a seed                             (the main event)
  Q7  density vs timing flexibility
  Q10 better selection rule                                  (cheap, can be done any time)
  Q5  critic feedback into the solver
  Q4  reachability in the objective                          (large; after the discretisation fix)
  Q11 executor in the planner                                (large)
  Q9  hardware-calibrated twin                               (needs hardware)
  Q12 decide the Multi-Verse future                          (after Q1/Q5)
```

The two RQ-A items come first because they are the *cause* of the 36 of 37 gate failures that everything else is downstream of, and because until the discretisation is consistent, **the 5 mm threshold is not measuring what it claims to measure.** A trajectory can pass the gate and still be unfollowable, or fail it and still be fine.

## 25. How to report a result

Report back in this shape, so results are comparable across experiments.

```text
Experiment:      Q<n> - <one line>
Tree state:      git rev-parse HEAD + the working-tree hash of the changed files
Config:          examples/fll_choreo.json (or another, named)
Missions:        <the mission files, and how they were generated>
Baseline result: duration / envelope / tracking / iterations / wall
New result:      duration / envelope / tracking / iterations / wall
Delta:           <absolute and relative>
Gate:            pass / fail, with the reason
Reproduce:       the exact command
Conclusion:      one sentence, and whether the success criterion was met
```

The tables in [Section 9](#9-measured-heuristic-comparison) are the baseline to beat. Any new candidate should be added to the matrix rather than measured on a mission of its own, because the interesting result is not "the new seed is fast" but "the new seed is fast *and* it wins somewhere".

---

# Appendix A — Reproducing every table

All commands run from the repository root.

## A.1 The heuristic comparison matrix

```bash
# Equal-duration timing, all five seeds
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --uniform-dt

# Per-segment timing, all five seeds
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --per-segment-dt
```

Missions: `example_straight`, `example_s_curve`, `example_mission_first4`, `example_complex_mission`, `example_with_stops`, `example_with_events`.

The machine-readable result is the `##OVERBOARD-VALIDATION##` line in the stdout; the human table has the wall times and the selected candidate.

## A.2 The timing × accuracy-weight grid

```bash
python main.py -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --sweep
```

## A.3 The seed comparison in one line

```bash
python main.py --uniform-dt --validate -c examples/fll_choreo.json \
  -w examples/example_mission_first4.json -o first4.traj --candidates 5
```

## A.4 From Python, without the CLI

```python
from optimizer import TrajectoryOptimizer
from robot_model import RobotConfig

cfg = RobotConfig({"robot": {"mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
                              "wheel_radius": 0.028, "v_max_rad_s": 18.94,
                              "t_max_nm": 0.138}})
opt = TrajectoryOptimizer(cfg)
for style in ("baseline", "rolling", "bow_left", "bow_right", "turn_first"):
    samples, stats = opt.solve(waypoints, 10, 0.0, guess_style=style)
```

## A.5 The reference solver table

```bash
python benchmark.py --mode solver --config examples/fll_choreo.json
```

This is the table to re-measure after any solver change: per mission, the wall time, the IPOPT iteration count and the trajectory duration. Add `--fast-polish` for the fast path.

## A.6 The validation pipeline only

```bash
python main.py -c examples/fll_choreo.json -w examples/example_complex_mission.json \
  -o output.traj --validate --candidates 5
```

## A.7 The test suite

```bash
python run_tests.py
python -m unittest tests.test_optimizer.TestTrajectoryOptimizer.test_init -v
python -m unittest tests.test_jax_heuristics tests.test_jax_ramsete \
                 tests.test_refinement_accuracy tests.test_critic_metrics
```

---

# Appendix B — Symbol index

Every symbol this document mentions, with its location in the tree that was measured.

## `optimizer.py`

| Symbol | Line | Role |
|---|---|---|
| `GUESS_STYLES` | 8 | The five seed names |
| `TrajectoryOptimizer` | 11 | The optimizer |
| `TrajectoryOptimizer.solve` | 17 | The public entry point; `guess_style`, `per_segment_dt`, `initial_guess`, `fast_polish` |
| `TrajectoryOptimizer.guess_from_samples` | 128 | Rebuild a guess array from a sample list (used by the refine probe) |
| `TrajectoryOptimizer._solve_once` | 148 | One IPOPT pass |
| `interval_dt` | 183 | The interval length for interval `k` |
| `dt` bounds | 241-242 | `dt_i` in `[0.001, 1.0]` |
| `dt_ref` | 219, 484 | The jerk normalising interval (KI-022) |
| `TrajectoryOptimizer._compute_cost` | 469 | The objective |
| `TrajectoryOptimizer._build_initial_guess` | 499 | The seed builder |
| `TrajectoryOptimizer._guess_point` | 551 | The baseline geometry |
| `TrajectoryOptimizer._offset_segment_interiors` | 586 | `bow_left` / `bow_right` |
| bow amplitude | 599 | `min(0.10 * length, 0.03)` |
| `TrajectoryOptimizer._rotate_before_driving` | 608 | `turn_first` |
| spin fraction / threshold | 616, 623 | `0.30 * s`, `0.4 rad` |
| `TrajectoryOptimizer._set_rolling_speeds` | 641 | `rolling` |
| `TrajectoryOptimizer._sample_times` | 668 | Timestamps from the per-segment `dt` |
| `TrajectoryOptimizer.format_output` | 677 | Samples with forces and events |

## `main.py`

| Symbol | Line | Role |
|---|---|---|
| `VALIDATION_MARKER` | 35 | The `##OVERBOARD-VALIDATION## ` prefix |
| `HAS_JAX` | 48 | JAX + Immrax availability |
| `ImmraxValidator`, `JAXRobotConfig` | 50, 51 | Module-level handles, patchable by tests |
| `PRESETS` | 55 | `fast` / `balanced` / `safe` |
| `_ensure_jax_classes` | 80 | The lazy import |
| `_select_attempt` | 92 | The selection rule |
| `CANDIDATE_WEIGHT_STEP` | 107 | The legacy ladder step |
| `SWEEP_WEIGHTS` | 116 | The sweep grid |
| `_THREAD_ENV_VARS` | 120 | The per-worker BLAS limits |
| `_candidate_specs` | 124 | The candidate set |
| `_solve_candidate_worker` | 199 | One candidate in a worker process |
| `_solve_candidates` | 222 | The parallel solve |
| `_export_block_reason` | 272 | The export gate |
| `_immrax_failure_reasons` | 281 | Human-readable failures |
| `_finite` | 303 | JSON-safe floats |
| `_segment_report` | 314 | The per-segment breakdown |
| `_emit_validation_report` | 376 | The machine-readable report |
| `_cpu_failure_reasons` | 396 | The CPU audit failures |
| `_apply_preset` | 419 | The presets |
| `_load_waypoints` | 480 | The waypoint parser; thresholds at 527-535 |
| `_print_summary` | 636 | The console table |
| `main` | 766 | The CLI |
| the candidate loop | 1000-1167 | Solve → validate → score |
| the local threshold | 1114 | 2 cm for transit |
| the per-candidate score | 1130-1140 | `ok`, `violation`, `tube`, `tracking_m` |
| the CPU audit per candidate | 1142-1152 | Runs on every candidate |
| the selection | 1169-1172 | `_select_attempt` |
| the export | 1218-1244 | The gate and the write |

## `validator.py`

| Symbol | Line | Role |
|---|---|---|
| `forward_integrate` | 7 | RK4 re-integration at 1 ms |
| `audit_constraints` | 122 | Per-wheel force, slip, traction, speed |
| `compute_metrics` | 212 | Duration, jerk, tortuosity, chattering |
| `check_final_waypoint` | 300 | 2 cm / 0.02 rad |
| `validate_samples` | 321 | The in-memory gate |
| `validate_trajectory` | 348 | The CLI/report entry point |

## `immrax_validator.py`

| Symbol | Line | Role |
|---|---|---|
| `unroll_trajectory_headings` | 14 | The ±π fix |
| `REACHABILITY_SUBSTEPS` | 38 | 20 |
| `CONTROL_DT` | 42 | 0.02 s |
| `PLANT_SUBSTEPS` | 43 | 4 |
| `_interpolated_reference` | 46 | The linear reference + the Ramsete gain |
| `RobotOpenLoop` | 92 | The default plant |
| `DigitalTwinOpenLoop` | 139 | The fitted 5-state plant |
| `MIN_TAU_S` | 157 | 0.01 s |
| `RamseteController` | 217 | `b = 2.0`, `zeta = 0.7` |
| `ClosedLoopRobotSystem` | 259 | The Immrax system |
| `ImmraxValidator` | 279 | The validator |
| `_advance_belief` | 349 | The robot's own pose estimate |
| `validate_trajectory` | 420 | The entry point; `safety_threshold_m = 0.005` at 427 |
| `check_sample_interval` | 444 | The per-sample interval physics |
| `compute_reachability` | 537 | The interval propagation |
| the divergence radius | 714 | 0.05 m |
| `get_safety_ribbon_data` | 728 | **Unused** (KI-018) |

## `multiverse_optimizer.py`

| Symbol | Line | Role |
|---|---|---|
| `resample_window_cpu` | 29 | The window resampler |
| `OptimizationStats` | 51 | The phase statistics; `heuristic_wins` at 65 (never written) |
| the critic thresholds | 91-96 | The six constants |
| the STOMP constants | 98-100 | Noise and variant count |
| `JAX_TARGET_SAMPLES` | 103 | 10 |
| `REFINE_PROBE_MAX_ITER` | 108 | 400 |
| `REFINE_PROBE_MARGIN` | 112 | 0.001 |
| `PathBootstrapper` | 114 | The Reeds-Shepp bootstrap |
| `generate_baseline` | 125 | The geometric guess |
| `_generate_segment_waypoints` | 144 | Reeds-Shepp or linear |
| `TrajectoryCritic` | 180 | The six metrics |
| `evaluate` | 192 | The window detection |
| `MasterTrajectoryOptimizer` | 252 | The pipeline |
| `solve` | 274 | The five phases |
| `_profile_kinematics` | 379 | Phase 2 |
| `_global_solve` | 454 | Phase 5 and the probe |
| `_probe_refined_guess` | 487 | The bounded probe |

## `jax_optimizer.py`

| Symbol | Line | Role |
|---|---|---|
| `local_trajectory_cost` | 11 | The proxy cost |
| `local_trajectory_projection` | 39 | The bounds projection |
| `generate_candidates_jax` | 52 | 4 topologies × 3 velocities + 50 STOMP |
| `perturb` | 80 | The STOMP noise |
| `get_jax_refiner` | 105 | The `jaxopt.ProjectedGradient` refiner |

## `robot_model.py`

| Symbol | Line | Role |
|---|---|---|
| `wheel_normal_forces` | 10 | The one weight-transfer model |
| `RobotConfig` | 38 | The config abstraction |
| `uncertainties` | 116-133 | The seven intervals |
| `get_max_force_at_velocity` | 138 | The motor curve |
| `get_robust_force_params` | 183 | The worst case for the optimizer |
| `DifferentialDriveModel` | 227 | The force model |
| `get_max_wheel_force_symbolic` | 312 | The per-wheel limit |
| `get_traction_limit_symbolic` | 334 | The per-wheel traction limit |
| `check_constraints` | 357 | The CPU-side check |

## `path_planning.py`

| Symbol | Line | Role |
|---|---|---|
| `ReedsSheppPath` | 11 | The wrapper |
| `plan` | 28 | The path; returns `None` on failure |
| `linear_interpolation_waypoints` | 58 | The fallback |

## `digital_twin.py`

| Symbol | Line | Role |
|---|---|---|
| `ActuatorFit` | 81 | gain, tau, delay, accel limit |
| `OdometryFit` | 107 | Distance and heading scale |
| `SlipFit` | 126 | The braking shortfall |
| `SensorFit` | 151 | The gyro and accel biases |
| `DigitalTwinConfig` | 258 | The whole fitted model |
| `actuator_gain_range` | 325 | The interval that goes into Immrax |
| `brake_shortfall` | 380 | Loaded, reported, not applied by default |
| `load_sysid_fit` | 444 | The loader |

---

# Appendix C — Known issue cross-reference

| KI | Area | One line | Where in this document |
|---|---|---|---|
| KI-001 | Solver / output | A failed solve was exported as a success | [22.4](#224-a-failed-solve-was-exported-as-a-success-ki-001-and---validate-could-not-block-output-ki-002) |
| KI-002 | Validation | `--validate` could not block output or the exit code | [22.4](#224-a-failed-solve-was-exported-as-a-success-ki-001-and---validate-could-not-block-output-ki-002) |
| KI-003 | Safety | Immrax ignored the upper uncertainty bounds | [22.3](#223-the-immrax-upper-uncertainty-bounds-were-ignored-ki-003) |
| KI-004 | Multi-Verse | The refinement has no effect on the final trajectory | [20.1](#201-the-jax-refinement), [20.2](#202-the-refine-probe), Q12 |
| KI-009 | Safety / config | The declared uncertainty ranges were ignored | [Section 11](#11-the-uncertainty-model) |
| KI-010 | Physics | The weight-transfer math was wrong and the tools disagreed | [22.5](#225-the-weight-transfer-math-was-wrong-and-the-tools-disagreed-ki-010) |
| KI-014 | Safety model | The JAX Ramsete used a poor large-angle approximation | [Section 12](#12-controller-and-plant-parity) |
| KI-015 | Benchmarks | The random benchmark tools are broken | [21.3](#213-the-random-benchmark-mode), Q8 |
| KI-018 | Maintenance | Dead code, duplicates, generated files | [21.4](#214-sysidsolverpy), [20.4](#204-the-critics-config-block) |
| KI-019 | Safety | The reachability tube lagged the reference by one sample | [22.2](#222-the-reachability-tube-lagged-the-reference-by-one-sample-ki-019) |
| KI-020 | Digital twin | The validator is not a full twin; reachability is not an objective | [Section 14](#14-what-each-layer-proves-and-does-not-prove), [21.2](#212-reachability-as-a-post-hoc-score), Q4, Q9 |
| KI-021 | Robot runtime | The DriveBase capped the acceleration the plan asked for | [Section 12](#12-controller-and-plant-parity) |
| KI-022 | Solver / timing | The per-segment timing model was a jerk-penalty loophole | [5.3](#53-the-jerk-discipline-ki-022), [22.1](#221-the-per-segment-timing-model-was-a-jerk-penalty-loophole-ki-022) |
| KI-024 | Solver / timing | The timing model had no rule for where the time allocation may change | [5.5](#55-where-the-time-allocation-may-change-ki-024) |

**Not referenced, and not relevant here:** KI-005, KI-006, KI-007, KI-008, KI-011, KI-012, KI-013, KI-016, KI-017, KI-023.

---

# Appendix D — Glossary

| Term | Meaning |
|---|---|
| **Candidate** | One independent solve with its own `(seed, timing model, accuracy weight)`. Each is validated and scored on its own. |
| **Seed** (`guess_style`) | The deterministic starting point handed to IPOPT. One of five. |
| **Timing model** | Whether all segments share one `dt` or each has its own. |
| **Envelope** | The axis-aligned bounding box of the reachable set at one sample. |
| **Tube** | The maximum envelope radius over the whole path. The primary selection score. |
| **Tracking error** | The maximum distance between the simulated plant's pose and the reference. |
| **Reachability** | The interval propagation of the closed loop under the declared uncertainties. |
| **Gate** | One of G0–G5 in [Part II](#part-ii--the-verification-strategy). A candidate must pass all of them. |
| **Passing** | A candidate that cleared every gate. Only a passing candidate can be exported. |
| **The baseline** | The `baseline` seed under equal-duration timing: the historical behaviour of the solver. |
| **The probe** | The bounded second solve from the Multi-Verse refined trajectory (`multiverse_optimizer.py:487`). |
| **Digital twin** | The fitted five-state plant, loaded with `--sysid-fit`. Strictly, a *partial* twin: it models the actuator and the odometry, not the motor, the PID or the slip. |
| **Model prediction** | Any number produced by the simulator. Not a hardware measurement. |

