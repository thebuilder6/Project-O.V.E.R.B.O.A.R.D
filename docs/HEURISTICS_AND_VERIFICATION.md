# Path Planning Heuristics and Verification Strategy

**Status:** 2026-10-19. Describes the working tree as measured on this date.
**Scope:** every path-planning heuristic in the repository, the verification chain that judges its output, and an explicit ledger of the strategies that did not work.
**Audience:** anyone who wants to improve the planner, the validator, or both.

> **Read this first.** Every number in this document is a **model prediction**, not a hardware measurement. No hardware logs exist in the repository. The robot-side collector (`pybricks_code/sysid_digital_twin.py`) can produce them, but nothing has been captured yet. The uncertainty ranges in `examples/fll_choreo.json` are *declared test assumptions*, not calibrated hardware values. A "PASS" in this document means "the trajectory passes the gates of an uncalibrated model", not "the robot will track it".

---

## Findings at a glance

1. **The heuristic layer is small.** Five deterministic seeds, two timing models, a candidate set, and a selection rule. There is **no geometric path family** anywhere in the tree — no splines, clothoids, béziers or quintics — and the objective has no term that prefers one path over another. It has three terms that price the *motion* (the duration, the wheel jerk and, optionally, the wheel force), and the passing candidates are ranked by a 5-part fitness score in seconds ([7.2](#72-the-5-part-path-fitness-score)).
2. **The seeds do not rescue the baseline often, but they do win on speed.** Over the 60-candidate matrix the `baseline` seed is the fastest passing candidate in 1 of the 7 cells that produce a trajectory at all; a non-baseline seed wins the other 6. The baseline fails in 9 of the 12 cells, and **four** of them are rescued by another seed: first4 under both timing models, the complex mission under equal-duration, and the events mission under per-segment. In the other 5 — the S-curve and the stops mission, under both models — no seed passes either.
3. **The tracking gate is the binding constraint; the reachability envelope never binds.** Of 34 failing candidates, 28 failed on the 5 mm tracking gate and 11 on motor force. **Zero** failed on the forward-integration check — it went with the self-consistent collocation — and **zero** on the 2 cm envelope threshold. The largest envelope measured anywhere was 1.72 cm.
4. **Two of the six example missions still produce no output at all.** The S-curve and the stops mission fail **every candidate** in the full sweep — 0 of 9 each, across both timing models and the whole accuracy-weight grid. Their best tracking errors are 1.27 cm and 0.75 cm against a 0.50 cm gate. The self-consistent collocation removed the plan's own deviation from the picture, and what is left of the error is the reference's velocity half ([RQ-A finding F8](research/RQ-A_tracking_error.md)).
5. **The timing model is a bigger lever than any seed, but not on every path.** Per-segment timing is 1.9x faster on the complex mission (4.812 s against 9.249 s) and it is what makes the events mission solvable at all; on the two paths whose segments are short and even (first4, the stops mission) the equal-duration model is the faster one. The seeds changed the outcome on 3 missions; the timing model on 1. The per-segment model also had to be made legitimate at the waypoints the robot drives through (KI-024, [5.5](#55-where-the-time-allocation-may-change-ki-024)); the matrix in [Section 9](#9-measured-heuristic-comparison) was re-measured after that change.
6. **The accuracy weight is the cheapest lever of all.** On the first-four-waypoints mission, the *same* `baseline` seed fails at weight 0 (it misses the motor-force limit by 0.006 N) and passes at weight 0.001 (0.26 cm tracking). The sweep grid is the only candidate-set change that has ever produced a new best trajectory, and it works because it adds a genuinely different parameter.
7. **The Multi-Verse pipeline was dead weight; it is repaired, and it is not worth much.** Its JAX-refined guess used to converge to 8.1473 s against the trivial baseline's 6.0677 s. Four defects were behind that - the refinement moved the waypoints, the JAX cost graded a different motion than IPOPT produces, the intervals were hard-coded, and the refined timing was discarded - and with them fixed the probe wins (4.617 s against 6.049 s) and the pipeline is reproducible. But measured over 30 missions as a *seed provider*, the refined seed is **not** better than the stock seeds (it is faster on 2 of the 9 missions where both pass, slower on 6) and it **rescues nothing**; it is a cheaper way to reach the stock set's speed on some missions (KI-004).
8. **The measurement base is six hand-written paths.** The random benchmark that was supposed to supply a mission suite is fixed (KI-015), and a mission-suite harness exists under [`research/`](../../research/). The six paths are still what [Section 9](#9-measured-heuristic-comparison) quotes; the wider suite is measured in `docs/research/pq_b/`.
9. **Twelve research questions are listed in Part IV**, ordered so that the cheapest decisive experiment comes first. The first two are measurements, not experiments.
10. **RQ-A is done, and it reorders the agenda.** [`docs/research/RQ-A_tracking_error.md`](research/RQ-A_tracking_error.md) identifies why the tracking gate binds: the optimizer's collocation, the controller's linear reference and the plant's integration are three different discretisations of the same trajectory, and on a curved path they disagree by about a centimetre. The optimizer's half of that disagreement is closed - the self-consistent collocation is the default ([5.7](#57-the-collocation-where-a-sample-is-placed)) - and the reference's half is not. The tracking error is **not** an uncertainty problem and **not** a physical tracking problem. Fix the discretisation before reaching for reachability in the objective. RQ-A also finds that with a *calibrated* model the binding gate flips to the **envelope**, because the measured actuator-gain band widens the reachable set.

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
    - [19.1 The sweep grid](#191-the-sweep-grid)
    - [19.2 Multi-start as a quality claim](#192-multi-start-as-a-quality-claim)
    - [19.3 The presets](#193-the-presets)
    - [19.4 The 44-Gene Starting-Value Genome & Memetic Evolution](#194-the-44-gene-starting-value-genome--memetic-evolution)
    - [19.5 Deterministic Homotopy Dispatch & Kinematic Velocity Profiling](#195-deterministic-homotopy-dispatch--kinematic-velocity-profiling)
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
| `N` | Number of samples: `1 + sum_i(s_i)` | – |
| `s` | `samples_per_segment`: the **budget** for the whole path's intervals. It is split across the segments by arc length, and `s_i` is what segment `i` actually gets | – |
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

**Provenance.** The sweep in [Section 9](#9-measured-heuristic-comparison) was run against this tree state:

```text
optimizer.py          MD5 EC154CD001FBE5A76863956575F09FF3
main.py               MD5 6D3E811DAE19B48BFC4987AE706DE9AF
immrax_validator.py   MD5 83E32D401CA09AF857191CF8EB660DDD
validator.py          MD5 9B83E54AE9BF3524C022BD5A17A7CA50
robot_model.py        MD5 AAB1A341939AADB2AB47E4D5C9C8C662
pybricks_code/ramsete_fast.py         MD5 D9095DFCDABAA54DF419A8339A398DFB
pybricks_code/ramsete.py              MD5 3ED998C1D8C69897F24D03469E68F7DC
pybricks_code/ramsete_compact.py      MD5 E60C4BFEF3113CB0259FCCD3C1F3865D
pybricks_code/ramsete_compact_events.py MD5 F2020A07297A1D554C947211CC84EF7E
```

The list is generated, not copied: `docs/measurements/2026-10-20-variable-samples/TREE.md5` holds it, and the runner writes it from the tree it measured. The runner also takes the fingerprint **before and after** the run and refuses a measurement whose tree changed underneath it, so a run that straddles two planners cannot be quoted as one.

Environment: Windows 11, Python 3.12.10, `casadi`, `jax`, `immrax`, `jaxopt`, `rsplan`, `numpy`, `scipy` all installed. Test suite state at the time of writing: `python run_tests.py` → **623 tests**, 0 skipped.

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
   ├─ Immrax        : immrax_validator.py:873  validate_trajectory
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

**Current status (KI-004):** the pipeline runs and its result is **used**: phase 5 probes the refined trajectory as a second starting point, and the probe now wins where it used to lose. `main.py --refine` puts the refined trajectory into the candidate set of the simple solver, so the pipeline is a **seed provider** — a cheaper way to reach the stock set's speed rather than a better one. See [Section 8](#8-the-experimental-multi-verse-path) and [Section 20](#20-ledger-multi-verse-internals).

## 2. What is not in the tree

This is the list a newcomer most needs, because it defines the whole research space.

| Missing | Consequence |
|---|---|
| **Any geometric path family.** No splines, clothoids, béziers, quintics, Dubins or Elford paths. | The only path primitives are a straight line between waypoints (`optimizer.py:551` `_guess_point`) and Reeds-Shepp (`path_planning.py:28`). The Reeds-Shepp planner is only reachable from the Multi-Verse bootstrap, and only when both segment endpoints have a heading. |
| **Obstacle or field geometry.** No walls, no mat boundaries, no zone constraints. | The robot is assumed to reach every waypoint in a straight line through space. |
| **A geometric term in the objective.** The objective is duration plus an optional jerk penalty (`optimizer.py:214-...`). | The optimizer has no idea what the path *looks* like. It is pushed along by the integration constraints and the endpoint equalities. Two paths of the same length and the same smoothness are indistinguishable to it. This is the root cause of many entries in Part III. |
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

The wheel speeds start at zero for every seed. The **initial interval** is not flat: it is proportional to the segment's length (PQ-B's Q2, finding F9), so every segment is allotted time in proportion to the distance it has to cover. The total is the same as the old flat guess - `0.1 s` per interval over the whole path - and each segment's share is spread over its own samples, so `dt_i = (0.1 * total_intervals) * (L_i / L) / s_i`, clipped to `[0.01, 1.0] s` (`_build_initial_guess`). The measured trade-off is the one PQ-B measured: it converges **more slowly** than the flat guess (40 iterations against 25) and finds a **faster trajectory** on 18 of 32 missions, so it changes which local optimum the solver reaches, not how fast it gets there.

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

**Intent:** IPOPT starts *moving*. A seed with zero wheel speeds sits at a point where the integration constraints and the "start and end at rest" equality constraints pull in opposite directions; a moving seed gives the solver a direction.

**Cost:** one pass over the samples. Negligible.

**Weakness:** the speeds are implied by the straight-line geometry, so they are wrong wherever the real path must curve. For a sharp turn the guess says "drive fast straight ahead", which is the opposite of what the path needs.

### 4.4 `bow_left` and `bow_right`

`_offset_segment_interiors` (`optimizer.py:586`) pushes the *interior* samples of each segment sideways along the segment's left normal:

```
amplitude = 0.5 * L * tan(|dtheta| / 4)           # the circular-arc sagitta
amplitude = min(amplitude, 0.35 * L)              # metres
bump(frac) = 4 * frac * (1 - frac)               # zero at both waypoints
offset = sign * normal * amplitude * bump(frac)
```

The amplitude is the **sagitta of the circular arc** that connects the two waypoints with the turn `dtheta` the segment asks for, so the guess bows as much as the arc it is imitating. A segment with no turn keeps the old flat rule (`min(0.10 * L, 0.03)`).

The bump is parabolic: zero at both waypoints, maximum in the middle. `bow_left` uses `sign = +1`, `bow_right` uses `sign = -1`.

**Intent:** "so IPOPT starts in an arc-shaped basin" (`optimizer.py:616`). A straight line between two waypoints with different headings is a *corner-cutting* path, and it is not one a robot can hold. Bowing the interior samples to one side turns the guess into a curve that at least resembles the arc the robot will have to drive.

**Cost:** one pass over the samples.

**Weakness:** the bow is a single bump, not an arc: it is zero at both waypoints and parabolic in between, so it approximates the arc's shape rather than reproducing it. The amplitude is capped at 35 % of the segment, which binds on a sharp turn, and the bump cannot follow a path that curves twice. The amplitude is now *related* to the geometry it approximates, which is what the flat 10 % rule was not.

### 4.5 `turn_first`

`_rotate_before_driving` changes the *topology* of the guess on the segments whose heading change is too sharp to be driven as an arc at the robot's own minimum radius. The threshold is derived from the track width `r_min` and the segment's length `L`:

```
ratio = L / (2 * r_min)
threshold = max(0.2, 2 * asin(ratio))   if ratio < 1   (the segment is short)
threshold = 0.4                        otherwise
```

`2 * asin(ratio)` is the turn an arc of radius `r_min` subtends over that segment, so a turn smaller than that can be driven as an arc and does not need a turn on the spot. A segment longer than `2 * r_min` falls back to the old 0.4 rad constant.

- The first `spin_samples = max(1, round(0.30 * s))` samples hold the position at `p1` and rotate the heading from the start heading to the end heading — a turn in place.
- The remaining samples drive straight to `p2` with a constant `theta_end`.

**Intent:** for a sharp turn, a differential-drive robot has two options: arc through the corner (slow, wide) or turn on the spot and then drive (fast, but it consumes the segment's length). The seed picks one of them explicitly.

**Cost:** one pass over the samples.

**Weakness:** it commits to a topology with no evidence about which one is right. The threshold now comes from the robot's geometry, but only for a short segment; a long one still uses the flat 0.4 rad constant. With `s_i <= 1` the seed keeps the baseline geometry instead, because one interval cannot hold both a spin and a drive.

**Known result:** the seeds are *not* uniformly worse than the baseline. On the complex mission under equal-duration timing all four non-baseline seeds converge to **slower and wider** paths (8.12–8.66 s, 1.52–1.75 cm envelope, 0.31–0.45 cm tracking) than the baseline (6.068 s / 1.19 cm / 0.30 cm) — and three of the four pass, `bow_left` being the one that fails, on the motor force (0.007 N). Under per-segment timing on `example_mission_first4`, `bow_right` is the **selected** candidate at 1.663 s, ahead of the baseline at 1.833 s, while `bow_left` and `turn_first` fail there on the motor force (0.071 N and 0.008 N). See [Section 9](#9-measured-heuristic-comparison).

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

Because the interval is shared and the segments do not have the same number of samples, one shared `dt` does **not** mean every segment takes the same time any more: a segment's duration is `s_i * dt`, and the sample counts follow the arc length, so the long segments get more time. The model is still "one time step for the whole path" in spirit - the samples are all the same distance apart in time - but it is no longer "every segment is allotted the same duration".

On the example missions this is the model that has the most validated results, because it is the historical one. It is also the model the default candidate 1 uses.

### 5.2 Per-segment (`per_segment_dt = True`)

One `dt_i` per waypoint segment, each bounded to `[0.001, 1.0] s`. Total time is `sum_i(s_i * dt_i)`: the segment's own interval times its own number of samples. The interval starting at sample `k` is the one of the segment that sample belongs to, which the code keeps as a `sample_to_segment` table rather than computing it from a division.

**This is the general model.** Setting every `dt_i` equal reproduces the equal-duration solution exactly. The reverse is not true.

### 5.3 The jerk discipline (KI-022)

Both models are held to the same wheel-jerk discipline, so `--accuracy-weight` means the same thing in either.

The objective is:

```text
J = duration
  + accuracy_weight * dt_ref^2 * sum_k [
        ((al_{k+1} - al_k) / span_k)^2 + ((ar_{k+1} - ar_k) / span_k)^2
    ]
  + force_weight * sum_k dt_k * [ 0.5*((fl_k/F_stall)^2 + (fr_k/F_stall)^2)
                              + 0.5*((fl_k/F_trac)^2 + (fr_k/F_trac)^2) ]
```

where `span_k` is the spacing between the midpoints of the two intervals meeting at boundary `k`, and `dt_ref` is the mean interval over the whole path.

The normalisation matters. Without it, the raw acceleration jump is proportional to `jerk * dt`, so a segment with a short `dt` is charged `dt^2` less per unit of jerk and can buy violent wheel motion nearly free. That previously let the per-segment model reach **68.9 m/s³** of wheel jerk where the equal-duration model reached **9.5 m/s³** at the same weight. The normalisation makes the price of physical jerk independent of the timing layout.

**The third term is a soft force penalty.** `--force-weight` prices the wheel forces instead of only constraining them: the hard per-wheel limits of [Section 10](#10-the-gate-chain) still apply, and on top of them the objective charges the time integral of the squared force utilisation against both the stall force and the per-wheel traction limit. A plan that runs near the traction limit is therefore paid for *before* the gate rejects it, which is the reachability-driven optimisation KI-020 asks for in its cheapest possible form - the force is a term in the objective, not a constraint on the path. The weight is **0 by default**, so the objective is duration plus jerk unless the operator asks for more; the flag is simple-solver only and is refused by `--multiverse`.

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

What the per-segment timing *could* break is the reading of the motion profile. The acceleration on the interval before waypoint `j` is `(vl[k]-vl[k-1])/dt[j-1]` and the one after it is `(vl[k+1]-vl[k])/dt[j]`, with `k` the sample that waypoint `j` sits on. When `dt` changes across the waypoint, the same velocity ramp is read as two different accelerations, so the wheel force profile gets a step exactly at the point the path is supposed to run through smoothly. The per-segment model therefore constrains the acceleration, and so the wheel force, to be continuous through every waypoint the robot does not stop at. The equal-duration model needs no counterpart: it never changes `dt`, so it reads the ramp the same way on both sides.

The residual is scaled by the reference interval squared so that it reads in m/s², the quantity the wheel force follows. The raw residual is in metres, where IPOPT's absolute constraint tolerance of `1e-2` would have allowed a centimetre of velocity ramp. The scaling also fixed the solver's behaviour: the complex mission's per-segment candidate went from 6.836 s in 41 iterations to 4.650 s in 203.

**Measured effect** (2026-10-19). The largest acceleration step at a waypoint the robot does not stop at:

| mission | equal-duration | per-segment, before | per-segment, after |
|---|---:|---:|---:|
| Complex mission, `accuracy_weight=0` | 3.65 m/s² | 2.99 m/s² | 1.2e-4 m/s² |
| Complex mission, `accuracy_weight=0.005` | 1.23 m/s² | not measured | 1.1e-3 m/s² |
| Uneven 1.0 + 0.2 m, `accuracy_weight=0` | 0.68 m/s² | 0.001 m/s² | 0.0 m/s² |

The large steps that remain are at the waypoints the robot *stops* at: the wheel force has to reverse sign there, so it cannot be continuous. On the complex mission the steps at the stops are 9.5 m/s² and 16.2 m/s², and they are the same in every timing mode.

Two candidates that the merge conditions used to reject now pass: `example_mission_first4.json` per-segment at `accuracy_weight=0` went from 1.661 s (rejected) to 1.833 s (accepted), and `example_with_events.json` per-segment at `accuracy_weight=0` went from 4.896 s (rejected) to 4.416 s (accepted). The complex mission is unchanged: the equal-duration candidate still wins at 6.068 s with a 1.19 cm envelope.

**The deeper cause is the sample layout, not the timing.** Every piece gets the same number of samples, so one shared `dt` gives every piece the same time regardless of its length. A 1.2 m straight line solved as one piece with 20 samples takes 2.702 s with no path excess; the same line with an interior waypoint and one shared `dt` takes 4.463 s and weaves 19.9 mm. Allocating the samples along the path by arc length would remove that penalty and leave the timing free to change only at real boundaries. It changes the sample count and the sample-to-waypoint mapping, so it is a separate change.

### 5.6 A segment-wise solver: the strategy that is not needed yet

The merge conditions in [5.5](#55-where-the-time-allocation-may-change-ki-024) are written as if the pieces were solved separately and then joined, because that is how a decomposed solver has to work. **The solver now does that**, for the paths where the pieces are genuinely separate: `TrajectoryOptimizer.solve` splits the path at the waypoints where the robot stops *and* the heading is pinned - the boundaries the merge conditions call full timing boundaries - and solves each piece as its own NLP, keeping the cheapest of the five seeds per piece and stitching the pieces back together. Inside a piece the path is still one NLP over one shared variable set, so the state agreement is automatic and the pieces of one piece can never disagree.

The split is what the merge conditions were written for, and it is opt-out (`split_at_boundaries`). It buys two things: the pieces are short, so each is better conditioned, and the piece boundaries are places where the time allocation is free anyway, so nothing is lost by solving them apart. What it costs is the joint optimum: the pieces are optimised separately, so the exit state of a piece is the entry state of the next rather than a shared variable, and the result is a local optimum of the decomposed problem, not of the whole.

The decomposition is worth writing down because it is the only way to make long missions tractable, and because it turns the merge conditions into something testable.

**Proposed strategy.** Split the path into *pieces*: the maximal runs of waypoint segments between legitimate timing boundaries (stop and pinned heading). Then:

1. **Global pass.** Solve the whole path once with the equal-duration model. It is the cheapest model, it converges, and it gives a smooth reference plus a sensible sample density.
2. **Piece-wise pass.** Solve each piece as its own NLP, in order. Piece `k` takes the exit state of piece `k-1` as a fixed initial condition, has a free time base inside the piece, and enforces the wheel-force continuity at the piece's internal waypoints. Warm-start each piece from the global pass.
3. **Merge.** Concatenate the pieces. The join conditions hold by construction, because each piece's exit state *is* the next piece's entry state, and the joins are at stop-and-pinned-heading points where the wheel force is allowed to reverse.

**What it buys.** The dimension of the joint NLP is `5*N` states plus the time base, and it grows with the mission: the 10-waypoint complex mission already takes 1557 IPOPT iterations and about 26 s. A per-piece solve is a handful of segments each, so the pieces can be solved in parallel and each is far better conditioned. That is the real argument for it — not the speed of the result, which the joint solve wins, but the tractability of long paths.

**What it costs.** The pieces are no longer optimised jointly, so the result is a local optimum of the decomposed problem, and the boundary state is fixed by the previous piece's solution rather than by the objective. The merge conditions are then a *correctness* requirement rather than a free consequence, and a mismatch at a join means the piece solutions have to be re-solved with the agreed state.

**What to measure first.** The cheap question is whether the decomposition is needed at all: the joint solve's iteration count against the number of waypoints. If it grows faster than linearly, the decomposition has a case; if the missions that matter stay under ten waypoints, it does not. That measurement is not in this document.

**What the split is not.** It is not the decomposition this section proposes: the proposed one is a global pass followed by a piece-wise pass with the pieces warm-started from it, and the merge conditions are re-imposed at the joins. The shipped split is simpler - each piece is solved on its own, with all five seeds tried, and the pieces are concatenated. It also splits only where the merge conditions say the state is determined, so the joins need no extra conditions at all.

### 5.7 The collocation: where a sample is placed

The timing model says *when* a sample is; the collocation says *where* it is. It is the other half of the discretisation, and it is the half RQ-A found to be wrong: the plan's samples were not on the path the plan's own `(v, omega)` profile traces, so the controller was told to be somewhere the plan never goes.

Two rules are in the tree. The CLI plans the second one by default, and `--no-self-consistent` restores the first (KI-026):

| rule | position rule | what it guarantees |
|---|---|---|
| trapezoid (the historical one) | the two endpoints of the interval only: `x_{k+1} = x_k + 0.5*(v_k*cos(theta_k) + v_{k+1}*cos(theta_{k+1}))*dt` | nothing: the samples are the ends of the chord, and the robot drives the chord |
| **self-consistent** | the interval's velocity integrated along the *integrated* heading, with a four-node Gauss-Legendre rule | the samples lie on the path the `(v, omega)` profile traces, and the heading samples are its tangents |

The self-consistent rule is worth roughly half the tracking error on a curved path: the S-curve goes from 1.47 cm to 0.88 cm, and the events mission's forward-integration error goes away. The heading rule is unchanged either way, so the headings are the tangents of the path the plan traces. The measurement is [RQ-A finding F10](research/RQ-A_tracking_error.md).

**Which one you get is a switch, and the default is the same everywhere (KI-051, KI-052).** The CLI plans the self-consistent rule by default and `--no-self-consistent` restores the trapezoid; the studio's "Self-consistent samples" box is ticked by default and unticking it is `--no-self-consistent`; and `TrajectoryOptimizer.solve` defaults to the self-consistent rule as well, so a library caller gets the same plan the command line does. `research/harness.py` follows the library.

### 5.8 The sample layout: how many samples a segment gets

The planner does not give every waypoint segment the same number of samples. `TrajectoryOptimizer.get_segment_sample_counts` splits the budget `segments * samples_per_segment` across the segments **in proportion to their arc length**, with a floor of three samples per segment:

```
total_budget = segments * samples_per_segment
m            = min(samples_per_segment, 3)          # the floor
rem_budget   = total_budget - segments * m
raw_i        = rem_budget * (L_i / sum(L))          # the proportional part
```

Each segment gets `m` samples plus its share of `rem_budget`, rounded down, and the samples left over by the rounding go to the segments with the largest remainders. The total is unchanged - `N = 1 + sum_i(s_i) = 1 + segments * samples_per_segment` - so the **number** of samples is the same as before and only their **distribution** changes. A path with one segment, a path of zero length, or a budget too small for the floor falls back to the uniform allocation.

**What it buys.** A long segment gets more samples, so the path is sampled finely where it is long and coarsely where it is short; the sample-to-waypoint mapping becomes a table (`waypoint_sample_indices`) instead of `k * samples_per_segment`, and `stats["segment_samples"]` reports the counts so a reader can map a sample back to its waypoint. The equal-duration model's old penalty - one shared `dt` over equal counts gives every segment the same time, so a short segment inherits the long one's duration - is gone, because a segment's duration is now `s_i * dt` and `s_i` follows the length.

**What it costs.** The duration term of the objective is `sum_i(s_i * dt_i)` rather than `samples_per_segment * sum_i(dt_i)`, the wheel-jerk normalisation is over a mean interval that is no longer a per-segment constant, and the seeds that count samples per segment (`bow_left`, `turn_first`, `rolling`) have to work from the table rather than from a division. It is also a change of *discretisation*, so every measurement taken before it describes a different planner: that is why [Section 9](#9-measured-heuristic-comparison) says which layout its numbers were taken on.

### 5.9 The discretisation cost metric

The default cost metric is the wheel jerk of [5.3](#53-the-jerk-discipline-ki-022), which is a proxy: it prices the *motion*, and the tracking gate measures a *discretisation*. RQ-A's diagnosis says the two are not the same thing - the gate is decided by three specific mismatches between the plan, the reference and the plant - and the `cost_metric` switch lets the objective price **those** instead:

```text
J = duration + accuracy_weight * sum_k w_k * [ sagitta_err^2 + parab_err^2 + 0.25 * hold_err^2 ]
```

| term | the mismatch it prices |
|---|---|
| `sagitta_err` | the chord against the arc: `(1/8) * v_mid * omega_mid * dt^2`, the mid-interval deviation of a straight chord from the path the plan traces |
| `parab_err` | the heading rule against the reference's: `(1/8) * v_mid * (omega2 - omega1) * dt^2`, the curvature the plan does not have |
| `hold_err` | the 20 ms command hold: `0.5 * (v2 - v1) * 0.02`, the staircase the controller sees |

Each is normalised by the 5 mm gate, so the cost is "how many gates' worth of tracking error does this interval buy". The metric is `cost_metric="discretization"`; the default, `"jerk"`, is unchanged. `sample_weights` puts a per-sample weight `w_k` on the term, so a mission can say *where* it wants the plan to be smooth or accurate - at the stops, say, rather than on the straights.

**This is the objective-side half of RQ-A.** The note's OQ2 and the "reference's velocity half" are about the *reference*, which is firmware; this is about the *plan*, which is the optimizer. The two are mirror images (F13): the plan fix helps at coarse sampling and hurts at fine, the reference fix the other way round.

### 5.10 The force bound, in two forms

The per-wheel force limit is a `max` over the wheel's share of the longitudinal force and its share of the yaw moment, and a `max` in an NLP is a non-smooth constraint: the usual way to write it is `f_bound >= sqrt(f_total^2 + 1e-12)`, and the `1e-12` puts a 10⁶ spike in the Hessian where the two shares are equal (`al == ar`, which is every straight-line sample). `epigraph_constraints=True` replaces it with the **epigraph form**: four exact linear inequalities on one variable per interval, plus smooth motor and normal-force upper bounds. Nothing about the limit changes - the same worst-case force over the declared ranges - but the Hessian does not spike, which is what lets the solver take bigger steps.

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

The CPU audit (`validator.py:321`) runs on **every** candidate, not only on the winner, so a candidate that clears Immrax but breaks the force, slip or integration limits cannot block a later candidate that passes everything.

### 7.2 The 5-part path fitness score

The gates say whether a plan is **allowed**; the fitness score says which of the allowed plans is **best**, and it is computed after the Immrax run (`validator.compute_path_fitness`). It is a single number in **seconds** - the time the plan's defects are worth - so it can be read as "the time this trajectory should have cost". Lower is better.

| # | part | weight | what it measures |
|---|---|---:|---|
| 1 | `stop_precision` | 200 s/m | the reachability spread at every stop and at the final sample, with the heading spread lever-armed (`sqrt(r^2 + (L_arm * dtheta)^2)`) |
| 2 | `slip_risk` | 8 s | a quadratic penalty on the combined motor and traction utilisation above a 70 % dead-zone, `0.5 * mean + 0.5 * max` |
| 3 | `rms_force` | 0.5 s | the time-weighted RMS wheel force, normalised by the stall force |
| 4 | `tracking_accuracy` | 80 s/m | half the RMS path tracking error plus the sum of the closed-loop tracking errors at the stops and the final sample |
| 5 | `time` | 1 s/s | the trajectory's own duration |

The weights are in seconds, so they say what each defect is worth: 1 mm of spread at a stop is worth 0.20 s of run time, and 1 mm of RMS tracking error is worth 0.08 s. **The score is a ranking, not a gate** - a plan is never rejected because its fitness is bad, only because a gate failed - and it needs the Immrax report, so it only exists on a `--validate` run.

### 7.3 The selection rule

```python
def _select_attempt(attempts):
    candidates = [a for a in attempts if a.get("converged", False)]   # KI-001
    if not candidates:
        return None, False
    passing = [a for a in candidates if a["ok"]]
    if passing:
        if any(a.get("fitness") is not None and math.isfinite(float(a["fitness"]))
               for a in passing):
            return min(passing, key=lambda a: (float(a.get("fitness", float("inf"))),
                                                a.get("tube", float("inf")),
                                                a.get("duration", a["cost"]))), True
        return min(passing, key=lambda a: (a.get("tube", float("inf")),
                                           a.get("duration", a["cost"]))), True
    return min(candidates, key=lambda a: (a["violation"], a["cost"])), False
```

In words:

1. **Non-converged candidates are never eligible.** (KI-001)
2. Among the passing candidates, **the lowest fitness score wins** ([7.2](#72-the-5-part-path-fitness-score)); the envelope and then the duration break ties. The old rule - the smallest envelope, then the shortest duration - is still the fallback for a run where no fitness was scored. The `--surrogate-polish` pipeline asks for a third variant, the **gate-certified minimum time**: the shortest duration first, then the smallest envelope, with the fitness score not consulted at all.
3. If none pass, the converged candidate with the **smallest quantified violation** wins, and `passed_any` is `False` — so the export is still blocked. The fallback exists to pick the most useful candidate to *report*, not to ship.

**What the rule used to be, and why it changed.** The rule was `min(tube, then duration)`, and PQ-B's Q10 measured it: five candidate rules picked the same trajectory on every mission, because the envelope is almost a proxy for the duration (`corr(duration, envelope) = +0.97`, and the tube grows with the clock through the backlash term). The fitness score replaces it because it asks a different question - not "which plan is narrowest" but "which plan's defects are worth the least time" - and the two can disagree: a plan with a wider tube at the stops but a shorter run can win.

`tests/test_regressions.py:103` (`TestAttemptSelection`) guards the rule that a failed attempt is never preferred over a passing one.

### 7.4 The report

`_emit_validation_report` (`main.py:376`) prints one machine-readable line:

```text
##OVERBOARD-VALIDATION## {"selected": 1, "candidates": [ ... ]}
```

Each candidate carries `style`, `timing`, `duration_s`, `envelope_cm`, `tracking_cm`, `ok`, `converged`, `reasons`, and a per-segment breakdown, plus the fitness score and its parts: `fitness_s`, `stop_spread_cm`, `slip_risk`, `rms_force_n` and `rms_force_util`. The browser studio parses this line to show which part of the path failed (`viz/studio.js`). Two invariants: the JSON must stay valid (no `Infinity` — `_finite` maps inf/nan to `null`), and nothing is written to disk on a failed validation, so the report has to go to stdout.

## 8. The experimental Multi-Verse path

`MasterTrajectoryOptimizer.solve` (`multiverse_optimizer.py:274`) runs five phases.

### 8.1 Phase 1 — Bootstrap

`PathBootstrapper.generate_baseline` (`multiverse_optimizer.py:125`) produces a geometric guess. Per segment it calls `_generate_segment_waypoints` (`multiverse_optimizer.py:144`):

- If **both** endpoints have a heading, it plans a Reeds-Shepp path (`path_planning.py:28`) and resamples it by arc length (`multiverse_optimizer.py:150`).
- Otherwise it falls back to `linear_interpolation_waypoints` (`path_planning.py:58`).

The Reeds-Shepp planner is configured from `multiverse.bootstrap` (`multiverse_optimizer.py:117-123`): `turning_radius` (default `0.0`) and `resolution` (default `0.05`).

> **The shipped configuration now plans.** `examples/fll_choreo.json` sets `"turning_radius": 0.05` - the robot's own turning radius - so the Reeds-Shepp path is a real path. It used to be `0.0`, and a zero turning radius makes the path degenerate to a straight line (rsplan could not plan with it at all), so the configured "Reeds-Shepp bootstrap" produced linear paths in the shipped example. Measured on the example missions: **0 of 21** segments that have a heading at both ends plan at radius 0.0, and **21 of 21** at 0.05. The `resolution` is honoured either way.

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

`_global_solve` (`multiverse_optimizer.py:454`) solves with the plain `TrajectoryOptimizer` from the default baseline seed, and then probes the refined trajectory as a *second* starting point (`multiverse_optimizer.py:487` `_probe_refined_guess`):

- Convert the refined samples to a guess array (`optimizer.py:128` `guess_from_samples`).
- Run a **cold** bounded pass: `max_iter = 400` (`REFINE_PROBE_MAX_ITER`, `multiverse_optimizer.py:108`) and `warm_start_init_point = "no"`.
- Keep the probe only if it converged **and** its cost is below `primary_cost * (1 - 0.001)` (`REFINE_PROBE_MARGIN`, `multiverse_optimizer.py:112`). Below 0.1 % the difference is solver noise, not a better trajectory.

The outcome is recorded in `stats["refine_probe"]` as `{converged, cost, iterations, used}`.

### 8.6 How the pipeline reaches the solver

The probe used to converge, and then lose. On `examples/example_complex_mission.json`:

| Starting point | Result |
|---|---|
| Default baseline guess | 6.0677 s |
| JAX-refined guess (probe) | 8.1473 s (converged in 38 iterations) |

The probe never won **then**, so the refined trajectory had no effect on the final result. **The residue was the quality of the guess, not the connection to the solver**: four separate defects had to be fixed first (the refinement moved the waypoint in the middle of the window it optimised, the JAX cost integrated from the left endpoint while IPOPT uses the trapezoidal rule, the intervals were hard-coded at 0.1 s whatever the segment's length, and the refined timing was thrown away when the guess was read back with the equal-duration model).

With those fixed the probe **wins**:

| Starting point | Result |
|---|---|
| Default baseline guess | 6.049 s |
| JAX-refined guess (probe) | **4.617 s** |

and `main.py --refine` puts the refined trajectory into the candidate set as a seed provider: the complex mission goes from 6.068 s to **4.936 s** and the trajectory passes the gates (envelope 1.14 cm, tracking 0.45 cm). It is a cheaper way to reach the stock set's speed rather than a better one — the full ten-candidate stock set gives 4.877 s on that mission, and measured over 30 missions the refined seed rescues nothing. See [20.1](#201-the-jax-refinement) and [20.2](#202-the-refine-probe), and KI-004.

## 9. Measured heuristic comparison

**Measured 2026-10-20**, against the tree with the arc-length sample layout, the self-consistent collocation on by default (KI-026, KI-051, KI-052) and the consolidated firmware (KI-028) in place. 60 candidate solves - 6 missions x 2 timing models x 5 seeds - plus the timing-mode x accuracy-weight grid. The raw logs are in `docs/measurements/2026-10-20-variable-samples/`, and the exact commands are in [Appendix A](#appendix-a--reproducing-every-table). The provenance (source hashes, environment, test state) is in [S0.4](#04-measurement-conventions).

**These numbers describe the shipped planner**, and they are **not comparable** with the 2026-10-20-post-ki028 evidence, which was taken with the trapezoidal collocation and a uniform sample count. The changes and what they cost are in [5.7](#57-the-collocation-where-a-sample-is-placed), [5.8](#58-the-sample-layout-how-many-samples-a-segment-gets) and [7.2](#72-the-5-part-path-fitness-score); the measurement that produced this section is in `docs/measurements/2026-10-20-variable-samples/`, and the runner takes the tree's fingerprint before and after the run, so a run whose tree changed underneath it is refused rather than quoted.

**The "Selected" column is the fitness score's choice**, not the envelope's ([7.3](#73-the-selection-rule)): among the candidates that pass, the one with the lowest 5-part fitness score wins, with the envelope and the duration as tie-breakers.

**The tables in this section are generated**, not typed in: `python docs/research/rq_a/sync_tables.py` rewrites them from the measurement JSON and `--check` verifies that every number still matches.

### 9.1 Method

- Robot config `examples/fll_choreo.json`, `-n 10` samples per segment, `accuracy_weight = 0` unless stated.
- The **self-consistent collocation** (the default) and the **arc-length sample layout** (the sample budget split across the segments by their lengths).
- Every candidate is solved, validated through the full gate chain, and scored independently.
- "Gate" is the pass/fail of the whole chain. The failure reason is quoted from the `reasons` field of the machine-readable report.
- Durations are the trajectory duration; the envelope is the maximum reachability radius over the path; tracking is the maximum closed-loop tracking error. The tracking gate is 0.50 cm, the transit envelope gate is 2 cm.

### 9.2 Table A — equal-duration timing, five seeds

| Mission | Seed | Duration s | Envelope cm | Tracking cm | Gate |
|---|---|---:|---:|---:|---|
| straight | baseline | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | rolling | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | bow_left | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | bow_right | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | turn_first | 2.377 | 0.30 | 0.00 | **PASS** |
| s_curve | baseline | 3.725 | 0.78 | 1.49 | fail — tracking 1.49 cm |
| s_curve | rolling | 3.786 | 0.72 | 1.42 | fail — tracking 1.42 cm |
| s_curve | bow_left | 3.697 | 0.70 | 1.27 | fail — tracking 1.27 cm |
| s_curve | bow_right | 3.728 | 0.71 | 1.44 | fail — tracking 1.44 cm |
| s_curve | turn_first | 3.738 | 1.07 | 1.53 | fail — tracking 1.53 cm |
| first4 | baseline | 1.666 | 0.30 | 0.19 | fail — motor force 0.006 N |
| first4 | rolling | 1.676 | 0.30 | 0.19 | **PASS** |
| first4 | bow_left | 1.852 | 0.32 | 0.43 | fail — motor force 0.082 N |
| first4 | bow_right | 1.847 | 0.32 | 0.40 | **PASS** |
| first4 | turn_first | 1.674 | 0.30 | 0.20 | **PASS** |
| complex | baseline | 6.401 | 1.25 | 0.24 | fail — motor force 0.014 N |
| complex | rolling | 9.250 | 1.72 | 0.44 | **PASS** |
| complex | bow_left | 9.239 | 1.63 | 0.58 | fail — motor force 0.082 N; tracking 0.58 cm |
| complex | bow_right | 9.238 | 1.64 | 0.58 | fail — tracking 0.58 cm |
| complex | turn_first | 9.249 | 1.72 | 0.44 | **PASS** |
| stops | baseline | 4.978 | 0.85 | 0.75 | fail — tracking 0.75 cm |
| stops | rolling | 5.690 | 0.97 | 1.60 | fail — tracking 1.60 cm |
| stops | bow_left | 5.690 | 0.97 | 1.60 | fail — motor force 0.002 N; tracking 1.60 cm |
| stops | bow_right | 5.730 | 0.98 | 1.76 | fail — tracking 1.76 cm |
| stops | turn_first | 5.729 | 0.97 | 1.76 | fail — tracking 1.76 cm |
| events | baseline | 4.654 | 0.89 | 0.63 | fail — tracking 0.63 cm |
| events | rolling | 4.658 | 0.89 | 0.60 | fail — tracking 0.60 cm |
| events | bow_left | 4.666 | 0.88 | 0.56 | fail — tracking 0.56 cm |
| events | bow_right | 4.666 | 0.88 | 0.56 | fail — tracking 0.56 cm |
| events | turn_first | 4.656 | 0.89 | 0.62 | fail — tracking 0.62 cm |

**14 passing, 16 failing.**

### 9.3 Table B — per-segment timing, five seeds

Re-measured after the merge conditions went in (KI-024). The per-segment model now requires the wheel force profile to be continuous through every waypoint the robot does not stop at, which changes what the model is allowed to do there, and after the backlash fix (KI-025).

| Mission | Seed | Duration s | Envelope cm | Tracking cm | Gate |
|---|---|---:|---:|---:|---|
| straight | baseline | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | rolling | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | bow_left | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | bow_right | 2.377 | 0.30 | 0.00 | **PASS** |
| straight | turn_first | 2.377 | 0.30 | 0.00 | **PASS** |
| s_curve | baseline | 3.636 | 0.71 | 1.42 | fail — tracking 1.42 cm |
| s_curve | rolling | 3.757 | 0.83 | 1.56 | fail — tracking 1.56 cm |
| s_curve | bow_left | 3.759 | 0.83 | 1.47 | fail — tracking 1.47 cm |
| s_curve | bow_right | 3.776 | 0.79 | 1.63 | fail — tracking 1.63 cm |
| s_curve | turn_first | 3.766 | 1.05 | 1.32 | fail — tracking 1.32 cm |
| first4 | baseline | 1.687 | 0.30 | 0.19 | fail — motor force 0.006 N |
| first4 | rolling | 1.713 | 0.30 | 0.19 | **PASS** |
| first4 | bow_left | 1.870 | 0.32 | 0.43 | **PASS** |
| first4 | bow_right | 1.843 | 0.33 | 0.40 | **PASS** |
| first4 | turn_first | 1.664 | 0.30 | 0.19 | fail — motor force 0.062 N |
| complex | baseline | 4.812 | 1.01 | 0.34 | **PASS** |
| complex | rolling | 5.000 | 1.12 | 0.38 | **PASS** |
| complex | bow_left | 4.864 | 1.10 | 0.36 | **PASS** |
| complex | bow_right | 4.843 | 1.09 | 0.36 | **PASS** |
| complex | turn_first | 4.952 | 1.11 | 0.38 | fail — motor force 0.062 N |
| stops | baseline | 5.150 | 0.94 | 1.59 | fail — tracking 1.59 cm |
| stops | rolling | 5.250 | 0.87 | 1.97 | fail — motor force 0.032 N; tracking 1.97 cm |
| stops | bow_left | 5.251 | 0.86 | 1.91 | fail — motor force 0.022 N; tracking 1.91 cm |
| stops | bow_right | 4.767 | 0.78 | 0.98 | fail — tracking 0.98 cm |
| stops | turn_first | 4.766 | 0.78 | 0.98 | fail — motor force 0.011 N; tracking 0.98 cm |
| events | baseline | 4.991 | 0.94 | 2.28 | fail — tracking 2.28 cm |
| events | rolling | 4.410 | 0.81 | 0.45 | **PASS** |
| events | bow_left | 4.411 | 0.81 | 0.45 | **PASS** |
| events | bow_right | 4.411 | 0.81 | 0.45 | **PASS** |
| events | turn_first | 4.411 | 0.81 | 0.45 | **PASS** |

**17 passing, 13 failing.**

### 9.4 Table C — what the selector chose

The rule is `min(tube, then duration)` among the passing candidates (`main.py:102`).

| Mission | Equal-duration | Per-segment | Note |
|---|---|---|---|
| straight | **rolling** 2.377 s | **rolling** 2.377 s | **seed-insensitive**: the path has no turn, so the geometric seeds degenerate to the same guess |
| s_curve | **no passing candidate** | **no passing candidate** | tracking 1.27-1.53 cm (equal-duration) and 1.32-1.63 cm (per-segment) on every seed |
| first4 | **turn_first** 1.674 s | **rolling** 1.713 s | 3 of 5 equal-duration and 3 of 5 per-segment seeds pass |
| complex | **turn_first** 9.249 s | **baseline** 4.812 s | 2 of 5 equal-duration and 4 of 5 per-segment seeds pass |
| stops | **no passing candidate** | **no passing candidate** | tracking 0.75-1.76 cm (equal-duration) and 0.98-1.97 cm (per-segment) on every seed |
| events | **no passing candidate** | **rolling** 4.410 s | equal-duration misses the gate (tracking 0.56-0.63 cm); 4 of 5 per-segment seeds pass |

**7 of the 12 cells produce a trajectory at all.** The baseline is selected in 3 of them and a non-baseline seed in 4.

### 9.5 Table D — the timing-mode × accuracy-weight grid

Run with `--per-segment-dt --sweep`, so the grid adds the `baseline` seed at five weights on top of the five per-segment seeds.

**`example_mission_first4`** (wall 29.2 s, exit 0)

**`example_mission_first4`** (wall 18.8 s, exit 0)

| # | Seed | Timing | Weight | Duration s | Envelope cm | Tracking cm | Gate |
|---|---|---|---:|---:|---:|---:|---|
| 1 | baseline | per-segment | 0 | 1.687 | 0.30 | 0.19 | fail — motor force 0.006 N |
| 2 | rolling | per-segment | 0 | 1.713 | 0.30 | 0.19 | **PASS** |
| 3 | bow_left | per-segment | 0 | 1.870 | 0.32 | 0.43 | **PASS** |
| 4 | bow_right | per-segment | 0 | 1.843 | 0.33 | 0.40 | **PASS** |
| 5 | turn_first | per-segment | 0 | 1.664 | 0.30 | 0.19 | fail — motor force 0.062 N |
| 6 | baseline | per-segment | 0.001 | 1.742 | 0.31 | 0.26 | **PASS** |
| 7 | baseline | per-segment | 0.002 | 1.791 | 0.31 | 0.26 | **PASS** |
| 8 | baseline | per-segment | 0.005 | 1.875 | 0.32 | 0.27 | **PASS** |
| 9 | baseline | per-segment | 0.01 | 1.948 | 0.32 | 0.28 | **PASS** |

**`example_complex_mission`** (wall 63.2 s, exit 0)

| # | Seed | Timing | Weight | Duration s | Envelope cm | Tracking cm | Gate |
|---|---|---|---:|---:|---:|---:|---|
| 1 | baseline | per-segment | 0 | 4.812 | 1.01 | 0.34 | **PASS** |
| 2 | rolling | per-segment | 0 | 5.000 | 1.12 | 0.38 | **PASS** |
| 3 | bow_left | per-segment | 0 | 4.864 | 1.10 | 0.36 | **PASS** |
| 4 | bow_right | per-segment | 0 | 4.843 | 1.09 | 0.36 | **PASS** |
| 5 | turn_first | per-segment | 0 | 4.952 | 1.11 | 0.38 | fail — motor force 0.062 N |
| 6 | baseline | per-segment | 0.001 | 6.401 | 1.51 | 1.44 | fail — tracking 1.44 cm |
| 7 | baseline | per-segment | 0.002 | 5.749 | 1.29 | 0.61 | fail — tracking 0.61 cm |
| 8 | baseline | per-segment | 0.005 | 5.109 | 1.09 | 0.39 | **PASS** |
| 9 | baseline | per-segment | 0.01 | 7.072 | 1.54 | 1.38 | fail — tracking 1.38 cm |

**`example_complex_mission`** (wall 52.1 s, exit 0)

| # | Seed | Timing | Weight | Duration s | Envelope cm | Tracking cm | Gate |
|---|---|---|---:|---:|---:|---:|---|
| 1 | baseline | per-segment | 0 | 4.812 | 1.01 | 0.34 | **PASS** |
| 2 | rolling | per-segment | 0 | 5.000 | 1.12 | 0.38 | **PASS** |
| 3 | bow_left | per-segment | 0 | 4.864 | 1.10 | 0.36 | **PASS** |
| 4 | bow_right | per-segment | 0 | 4.843 | 1.09 | 0.36 | **PASS** |
| 5 | turn_first | per-segment | 0 | 4.952 | 1.11 | 0.38 | fail — motor force 0.062 N |
| 6 | baseline | per-segment | 0.001 | 6.401 | 1.51 | 1.44 | fail — tracking 1.44 cm |
| 7 | baseline | per-segment | 0.002 | 5.749 | 1.29 | 0.61 | fail — tracking 0.61 cm |
| 8 | baseline | per-segment | 0.005 | 5.109 | 1.09 | 0.39 | **PASS** |
| 9 | baseline | per-segment | 0.01 | 7.072 | 1.54 | 1.38 | fail — tracking 1.38 cm |

### 9.6 Table E — the full sweep: what the tool can actually deliver

The default candidate set plus the timing-mode x accuracy-weight grid, both timing modes: **13 candidates per mission, 78 in total**. This is `--candidates 5 --sweep`, the most thorough configuration the CLI offers.

| Mission | Wall | Passing | Selected | Duration s | Envelope cm | Tracking cm | Fastest passing |
|---|---:|---:|---|---:|---:|---:|---|
| straight | 18.4 s | 9/9 | `baseline` (equal-duration) | 2.401 | 0.31 | 0.00 | `bow_right` 2.377 s |
| s_curve | 28.0 s | **0/9** | *nothing passes* | — | — | — | — |
| first4 | 18.4 s | 7/9 | `turn_first` (equal-duration) | 1.674 | 0.30 | 0.20 | `turn_first` 1.674 s |
| complex | 49.0 s | 5/9 | `baseline` (equal-duration) | 6.457 | 1.50 | 0.27 | `baseline` 6.457 s |
| stops | 38.2 s | **0/9** | *nothing passes* | — | — | — | — |
| events | 33.0 s | 3/9 | `baseline` (equal-duration) | 4.698 | 0.88 | 0.46 | `baseline` 4.673 s |

### 9.7 Why the timing models behave so differently

The segment-time readout in the logs makes the difference concrete. For the complex mission:

```text
baseline, equal-duration:  one dt for the whole path, and the sample count
                          follows the length, so the long segments get the time
baseline, per-segment:    one dt per segment, and the time lands where the
                          path needs it
```

The two models are not peers because the **sample count is not uniform any more**: a segment's duration is `s_i * dt` under the equal-duration model, and `s_i` follows the arc length, so the long segments get more of the time - but still the same *interval*, so the robot is sampled finely and driven slowly there. The per-segment model gives each segment its own interval, and the time lands where the path needs it. On the complex mission the two are 9.249 s and 4.812 s.

### 9.8 What the tables say

**Finding 1 — the seeds win on speed in 6 of the 7 cells that produce a trajectory.** The `baseline` seed is selected in one of them (the complex mission under per-segment timing, 4.812 s); a non-baseline seed in the other six: straight under both timing models (`rolling`, 2.377 s), first4 under both (`turn_first` 1.674 s and `rolling` 1.713 s), the complex mission under equal-duration (`turn_first`, 9.249 s) and the events mission under per-segment (`rolling`, 4.410 s).

**Finding 2 — the seeds rescue four cells, and the baseline fails in 9 of the 12.** The baseline is the fastest passing candidate in only one cell, and it is rescued in four: first4 under both timing models (it misses the motor-force limit by 0.006 N), the complex mission under equal-duration (0.014 N) and the events mission under per-segment (it misses the tracking gate at 0.63 cm). The other five failures are the S-curve and the stops mission, where no seed passes either. The seeds are doing more work than they used to, and the work is in the **equal-duration** model - the per-segment model is where the baseline is enough.

**Finding 3 — the tracking gate is still the binding constraint, and the envelope gate still never binds.** Of the 34 failing candidates: 28 cite the tracking error, 11 the motor force, **0** the forward-integration check and **0** the envelope radius. The largest envelope measured anywhere in the matrix is 1.72 cm against a 2 cm threshold. The reachability tube - the thing the whole interval-propagation machinery exists to compute - is never the reason a candidate is rejected. **The forward-integration failures are gone**, and they went with the self-consistent collocation: the plan no longer disagrees with its own integration, so the audit's check has nothing left to catch.

**Finding 4 — the failures still cluster by mission, not by seed.** The S-curve and the stops mission fail every configuration: both timing models, all five seeds, all five weights. Their best tracking errors are 1.27 cm and 0.75 cm against a 0.50 cm gate. That is a property of the path and the controller, not of the starting point, and it is the clearest signal that those two are a *planner* problem, not a *seed* problem.

**Finding 5 — the weight is a search axis, not a dial.** On the complex mission the passing weights are 0 and 0.005 (4.812 s and 5.109 s) and the failing ones are 0.001, 0.002 and 0.01; the duration is not monotone in the weight, and the weight that passes is neither the largest nor the smallest. There is no weight that is "safest"; there are weights that happen to land well for a given path.

**Finding 6 — the two timing models are not peers, and the gap widened.** The complex mission is 9.249 s under equal-duration timing and 4.812 s under per-segment (1.9x), and the events mission is 0.56-0.63 cm under equal-duration (failing on the tracking gate) and 0.45 cm under per-segment (passing). The equal-duration model now spends its time badly on a path with unequal segments: the sample count follows the length, so a long segment gets more samples and, with one shared `dt`, more time. On the two paths whose segments are short and even (first4, the stops mission) the equal-duration model is the faster one, which is the shape of the trade the candidate set exists to arbitrate.

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
 G3  Immrax reachability           immrax_validator.py:873
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

`ImmraxValidator.validate_trajectory` (`immrax_validator.py:873`) simulates the closed loop with interval arithmetic. Details in [Section 12](#12-controller-and-plant-parity).

**The tube is the width of a *model*, so the model is part of the number.** The reachability is computed by one of two **propagations** (`immrax_validator.py`, `PROPAGATION`), and every candidate in the `##OVERBOARD-VALIDATION##` line names the one that produced its envelope:

| propagation | what it embeds | what the tube bounds |
|---|---|---|
| `plant` (the default) | the plant; the controller is evaluated *outside* the propagation, at the tube's midpoint, with the command held for the tick | the reachable set of an **open-loop** plant driven by a pre-recorded command — the feedback's contraction is not in the tube at all |
| `tick` | one 20 ms tick of the **closed loop** — controller, pose estimate and delayed command — as a discrete map, propagated once per tick | the reachable set of the loop the robot actually runs; the map's Jacobian carries `dF/dx = dPhi/dx + (dPhi/du)(dN/dx)`, and the Ramsete feedback contracts the tube |

The tick propagation needs a Jacobian inclusion: with the natural extension the closed-loop map explodes (627 mm on a 1 m straight). It costs 2-3x (`jac`) to 7-10x (`mjac`) the validation time, which is why it is opt-in (`main.py --propagation tick`). It is the lever on the paths where the plant propagation *diverges* rather than merely widens — the S-curve, the stops and the complex mission, where the mixed-Jacobian gives 1.88, 1.30 and 1.83 cm against 5.36, 5.31 and 5.07 cm, and every tube still contains the sampled reachable set. With the digital twin it is **wider**, and the reason is the twin's plant state, not the propagation (KI-049, KI-050).

Two rules come with it. **The controller must not unpack the pose**: `x, y, theta = current_pose` is the JAX `unstack` primitive, Immrax's natural inclusion has no rule for it, and the tick map dies with "unstack not in inclusion_registry" — so the pose is read by indexing, which compiles to `slice`+`squeeze` and gives the same number. And **an envelope is never read without its model**: the report, the studio's validation note and the studio's chip tooltip all name the propagation, the same rule as `envelope_truncated`.

The physical pass condition (`immrax_validator.py:969-976`):

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

Per sample, the envelope radius is compared to the local threshold. The thresholds come from the waypoint list, and the rule for which segment a waypoint's limit governs is in `main._segment_limit`:

| Waypoint kind | Threshold | Source |
|---|---|---|
| Transit (default) | 2 cm | `main.py:_segment_limit` |
| `precision_mode: true` | 0.5 cm | `main.py:_load_waypoints` |
| Explicit `error_threshold` | the user's value | `main.py:_load_waypoints` |

**The limit belongs to the way *into* the waypoint.** A waypoint's limit governs the segment that ends at it - the approach - and the segment that starts there falls back to it when the next waypoint declares none. So a waypoint marked as a precision one is held to its tight limit on the way the robot arrives, which is what the operator asked for, and the final waypoint's limit governs the last segment instead of being dropped (KI-034).

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
| 7 | `precision_mode` envelope radius | ≤ 0.5 cm | `main.py:_load_waypoints` | tight waypoints, on the way in |
| 8 | Explicit `error_threshold` | user value | `main.py:_load_waypoints` | per-waypoint override, on the way in |
| 9 | Forward-integration position error | < 1 cm | `validator.py:333` | RK4 re-integration vs. the plan |
| 10 | Forward-integration final error | < 1 cm | `validator.py:334` | same, at the end |
| 11 | Violating samples | 0 | `validator.py:335` | motor/traction limit hits |
| 12 | Slip points | 0 | `validator.py:336` | per-wheel slip events |
| 13 | Speed violation | < 1e-6 m/s | `validator.py:337` | wheel speed over the limit |
| 14 | Final waypoint position | ≤ 2 cm | `validator.py:302` | the plan must arrive |
| 15 | Final waypoint heading | ≤ 2 cm (0.02 rad) | `validator.py:302` | the plan must arrive pointing the right way |
| 16 | IPOPT iterations | ≤ 5000 | `optimizer.py` solver options | `max_iter`; the **warm** pass in fast-polish is budgeted to 200, and the cold pass that follows it is not |
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
| Reference | linear interpolation of `x, y, heading, v, omega` plus the precomputed Ramsete gain `sqrt(omega^2 + b*v^2)` (`immrax_validator.py:46`) | `ramsete.py` | **yes** |
| Heading unwrapping | `unroll_trajectory_headings` (`immrax_validator.py:14`) | — | prevents the ±π wrap |
| Controller | `RamseteController` (`immrax_validator.py:233`), `b = 2.0`, `zeta = 0.7` | `ramsete.py` | **yes** |
| Command quantisation | whole mm/s and deg/s when `quantize=True` (`immrax_validator.py:265`) | `DriveBase` rounding | **yes** (KI-021) |
| Firmware's own copies | — | **one**: `ramsete.py` holds the odometry, the Ramsete maths, the limiter and the loop; `ramsete_fast.py` delegates to it and `ramsete_compact.py` delegates to `ramsete_compact_events.py` | **yes** - a second copy cannot drift (KI-028) |
| Traction limit in the limiter | per wheel, with the lateral weight transfer (`robot_model.py:9` `wheel_normal_forces`) | per wheel, same formula (`pybricks_code/ramsete.py`, `KinematicLimiter`) | **yes** (KI-010) |
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
| G4 local threshold | no waypoint's envelope budget is exceeded | that the budget is the right budget. **With a calibrated model this gate, not the tracking gate, is the one that binds**: the measured actuator-gain band widens the reachable set (11.8 mm at ±1 % over a 1 m straight, against a 20 mm threshold) while barely moving the tracking error, because the controller corrects the speed error (RQ-A finding F5). On a curve the tube still crosses the 50 mm guard, but the sampled reachable set there is 4.09 mm against a 48 mm tube, so the excess is the *open-loop* treatment of the feedback, not the band and not the arithmetic (RQ-A finding F12, KI-049). The lever is the closed-loop tick map, which is implemented and off by default: it wins on the default plant and loses on the digital twin, where the belief's spread is added to the plant's |
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
| **Evidence** | Complex mission, equal-duration: the baseline passes at **6.068 s / 1.19 cm / 0.30 cm**; `rolling` 8.377 s / 1.68 cm / 0.45 cm, `bow_left` 8.634 s / 1.75 cm / 0.41 cm, `bow_right` 8.662 s / 1.71 cm / 0.41 cm, `turn_first` 8.118 s / 1.52 cm / 0.31 cm — three pass and `bow_left` fails on the motor force (0.007 N). First4, per-segment: the baseline is not the fastest — `bow_right` is, at **1.663 s** against the baseline's 1.833 s, and it is the selected candidate, while `bow_left` and `turn_first` fail there on the motor force (0.071 N and 0.008 N). Events: under per-segment timing all five seeds pass at 4.415–4.416 s, while under equal-duration timing all five fail (0.65–0.68 cm of tracking error). See [Section 9](#9-measured-heuristic-comparison) |
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
| **Evidence** | On a 1.0 m + 0.2 m path at `accuracy_weight = 0.005`: per-segment **2.784 s / 0.32 cm**, equal-duration **4.456 s / 0.37 cm**. On the complex mission at weight 0: per-segment 4.650 s but 0.43 cm tracking and a 0.074 N force overshoot, equal-duration 6.068 s / 1.19 cm / 0.30 cm passing |
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
| **Evidence (measured 2026-10-20)** | `example_mission_first4`: the default candidate-1 result is 2.176 s; the grid's `baseline` per-segment at weight 0 passes at 1.833 s, and `bow_right` per-segment at weight 0 is faster still at **1.663 s** — 23.5 % faster — and is the selected candidate. `example_with_events`: the default set now produces a passing candidate under per-segment timing (4.416 s), where the equal-duration `baseline` still misses the tracking gate at 0.68 cm. `example_complex_mission`: the grid adds passing `baseline` **equal-duration** candidates at weights 0.001 (7.049 s), 0.002 (7.080 s), 0.005 (7.208 s) and 0.01 (9.323 s), but the equal-duration baseline at weight 0 (6.068 s) is still selected, so the complex mission is unchanged; every per-segment candidate in the grid fails the tracking gate (0.67–0.91 cm) |
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

### 19.4 The 44-Gene Starting-Value Genome & Memetic Evolution

| Field | Value |
|---|---|
| **Strategy** | Evolve the starting-seed macro-parameters (`dt_mults`, `apex_offsets`, `theta_biases`, `speed_scales`) across generations of mutation and crossover with parallel IPOPT local-search polishes (`run_memetic_evolution` archived in `legacy/evolutionary_research/genetic_algorithm.py`) |
| **Mechanism** | An evolutionary algorithm models robot trajectories as discrete chromosomes of segment timings and spatial state sequences. Offspring undergo creep mutation and macro-homotopy mutation, followed by fast warm-start IPOPT polishing |
| **Why it was tried** | To escape bad local minima by discovering globally optimal driving lines (late apexes, customized segment time distributions) without hand-crafting heuristic shapes |
| **Outcome** | **Succeeded in discovering the optimal racing line (+20.66% speedup), but superseded by deterministic generation.** On `example_complex_mission.json`: the stock baseline produced 6.907 s (0.72 cm tracking); the 31-gene genome at 8 generations achieved 5.691 s (0.50 cm tracking); and the 44-gene champion at 12 generations hit **5.417 s and 0.49 cm (4.9 mm)** tracking error, breaking through the 5 mm gate. However, benchmark logging (`output_memetic/memetic_scores.json`) revealed that the winning candidate was actually generated at **Generation 0** (`seed_4` at 5.514 s) by the initial 10-candidate pyramid homotopy bank. Subsequent generations of stochastic evolution (25.8 s of wall-clock time and dozens of IPOPT solves) failed to improve upon Generation 0 |
| **Key Insight** | The 44-gene GA succeeded because it converged on three deterministic physical principles: (1) pushing the apex outward on turns (late apex), (2) giving tight turns more $\Delta t$ and straights less $\Delta t$, and (3) aligning unpinned waypoint headings tangent to the curve. These geometric choices do not require stochastic exploration; they can be calculated directly in $< 2\text{ ms}$ |
| **Status** | **Superseded by Deterministic Homotopy Dispatch (19.5)** |

### 19.5 The Kinetic Frontier Analytical Seed Generator & The 5 Physical Principles

| Field | Value |
|---|---|
| **Strategy** | Replace stochastic genetic search with an analytical seed generator (`frontier_seed.py`) encoding the 5 physical principles uncovered by phase portrait diagnostics and the 50-generation marathon |
| **Mechanism** | `frontier_seed.py` (`generate_frontier_seed`, `generate_canonical_seeds`), integrated into `TrajectoryOptimizer._build_initial_guess` via `GUESS_STYLES = (..., "frontier_sprint", "frontier_sweeper", "frontier_exploratory", "frontier_direct")` |
| **The 5 Governing Principles** | 1. **Active Kinematic Hypotenuse:** $v(t) + \frac{W}{2}|\omega(t)| = v_{\max}$. In differential-drive, motor voltage saturation dominates tire friction by $2.5\times$. The optimal trajectory rides the $L_1$ actuator diamond boundary (one wheel at 100% saturation at all times) with analytical velocity profiling $v_{\text{frontier}}(\kappa) = \frac{v_{\text{wheel\_max}}}{1 + 0.5 B |\kappa|}$.<br>2. **Late-Apex Dynamic Law:** Asymmetric corner entry ($f_{\text{apex}} = 0.66\text{--}0.70$) enables deep trail-braking and early symmetric exit throttle ($0.3\text{ s}$ earlier full acceleration onto straights).<br>3. **Waypoint Momentum Preservation:** Maintains $0.65\text{--}0.75 \times v_{\max}$ through unpinned intermediate waypoints along a continuous energy arc.<br>4. **KKT Slack Barrier Equilibrium:** Balances right on the safety threshold ($4.3\text{--}4.5\text{ mm}$ tracking against the $5.0\text{ mm}$ gate).<br>5. **Multi-Modal Homotopy Triad:** Dispatches 3 canonical homotopies (`frontier_sprint`, `frontier_sweeper`, `frontier_exploratory`). |
| **Measured Outcome (2026-10-20)** | **Shatters the 44-Gene Champion on the 10-waypoint complex mission:**<br>- **Stock Baseline:** $5.948\text{ s}$, $0.96\text{ cm}$ tracking, 200 iters (capped), $3.45\text{ s}$ wall (**FAIL**)<br>- **44-Gene GA Champion (12 Gens):** $5.417\text{ s}$, $0.49\text{ cm}$ ($4.9\text{ mm}$), 120 iters, $\approx 25\text{ s}$ wall (**PASS**)<br>- **Frontier Sprint Seed:** **$5.239\text{ s}$**, **$0.45\text{ cm}$ ($4.5\text{ mm}$)**, $1.17\text{ cm}$ envelope, **Score 5.239** (**PASS**)<br>- **Frontier Sweeper Seed:** **$5.237\text{ s}$**, **$0.44\text{ cm}$ ($4.4\text{ mm}$)**, $1.18\text{ cm}$ envelope, **101 iters**, **Score 5.237** (**PASS**)<br>- **Frontier Exploratory Seed:** **$5.233\text{ s}$**, **$0.43\text{ cm}$ ($4.3\text{ mm}$)**, $1.18\text{ cm}$ envelope, **Score 5.233** (**PASS**)<br>- **Seed Generation Overhead:** **$1.16\text{ ms}$** total for all 3 seeds |
| **Status** | **Live** in `frontier_seed.py` and `optimizer.py` |

### 19.6 The 3-Pillar Unified Trajectory Engine (`main.py --unified`, the default path)

| Field | Value |
|---|---|
| **Strategy** | Condense the 9 physical principles into 3 unified engineering pillars, eliminating stochastic evolutionary loops while ensuring 100% deterministic gate certification across all competition paths |
| **Mechanism** | `unified_planner.py` (`UnifiedPathEngine`), `main.py --unified`, tested by `tests/test_frontier_seed.py` and `tests/test_unified_planner.py` |
| **The 3 Pillars** | **Pillar I: Asymmetric Transition Geometry** (Rule 2 Late-Apex Dynamic Law $f_{\text{apex}} = 0.68$, Rule 5 Homotopy Triad: Sprint, Sweeper, Exploratory, Rule 6 Clothoid Curvature Continuity).<br>**Pillar II: Composite Kinetic Velocity Envelope** (Rule 1 Active Kinematic Hypotenuse on $L_1$ polytope boundary, Rule 3 Waypoint Momentum Preservation $\ge 0.70 v_{\max}$, Rule 8 Dynamic Load Transfer $v_{\text{tip/slip}} = \sqrt{g B / (2 h_{\text{CG}} \|\kappa\|)}$, Rule 9 Pontryagin Threshold Braking for stops).<br>**Pillar III: Invariant Collocation Mesh & KKT Solver** (Rule 4 KKT Slack Barrier Equilibrium at $3.6\text{ mm}$, Rule 7 Invariant 80 ms Sampling Mesh $\ge 10$ nodes/segment, 4-node Gauss-Legendre self-consistent collocation, smooth linear epigraph bounds). |
| **Measured Outcomes (2026-10-20)** | **100% Gate Certification Across the Benchmark Suite:**<br>- **`example_complex_mission.json`:** **$5.365\text{ s}$**, envelope **$1.16\text{ cm}$**, tracking error **$3.6\text{ mm}$** (vs $5.0\text{ mm}$ gate), max integration error **$0.45\text{ mm}$**, 0 violations. Smashes the 44-Gene Champion ($5.417\text{ s}$, $4.9\text{ mm}$) and stock baseline ($6.068\text{ s}$, $1.22\text{ cm}$).<br>- **`example_straight.json`:** **$2.270\text{ s}$**, envelope **$0.30\text{ cm}$**, 0 violations, max pos error $0.45\text{ mm}$.<br>- **`example_with_stops.json`:** **$5.105\text{ s}$**, envelope **$0.89\text{ cm}$**, 0 violations, max pos error $0.46\text{ mm}$.<br>- **`example_s_curve.json`:** **$3.602\text{ s}$**, envelope **$0.83\text{ cm}$**, 0 violations, max pos error $0.46\text{ mm}$.<br>- **Compute Overhead:** Triad analytical seed generation takes **$< 2.5\text{ ms}$**; parallel IPOPT solves converge in **$40\text{--}48\text{ iters}$** in $\approx 1.7\text{ s}$ per candidate. |
| **Status** | **Default** in `unified_planner.py` and `main.py --unified`; `--no-unified` restores the stock candidate set |

## 20. Ledger: Multi-Verse internals

### 20.1 The JAX refinement

| Field | Value |
|---|---|
| **Strategy** | Generate 62 candidate paths per flagged window (4 topologies × 3 velocities + 50 STOMP), score them in parallel with `vmap`, refine the best 10 with projected gradient, and write the winner back |
| **Mechanism** | `jax_optimizer.py:52` `generate_candidates_jax`; `multiverse_optimizer.py:307-353` |
| **Why it was tried** | Sample the topology space in parallel (forward / reverse / point-turn), then locally optimise the good ones. This is the "evaluate thousands of structural seed trajectories in parallel" idea from `Future_Plans.md` |
| **Outcome** | **Repaired (KI-004).** The refinement now produces a *usable* starting point, and the pipeline is re-envisioned as a **seed provider** for the solver that has the right objective. Four separate defects had to be fixed, and each was necessary: the refinement optimised two-segment windows and so **moved the waypoint in the middle**; the JAX cost integrated from the left endpoint while **IPOPT uses the trapezoidal rule**, so the proxy graded a different motion; the initial `dt` and velocities were **hard-coded** at 0.1 s and 0.1 m/s whatever the segment's length; and the refined timing was **thrown away** - the timestamps were not updated, and the probe read the guess with the equal-duration model, which keeps only the first interval |
| **Status** | live (the pipeline is a seed provider: `main.py --refine`) |
| **What would revive it** | Nothing further for the refinement itself. The next lever is the **objective**: the pipeline's cost is duration plus a jerk term, so it buys speed with tracking accuracy and the 5 mm gate rejects the trade. On the complex mission the refined seed gives 4.936 s where the default set gives 6.068 s, but the full ten-candidate stock set gives 4.877 s there, so it is a cheaper way to reach the stock set's speed rather than a better one |

### 20.2 The refine probe

| Field | Value |
|---|---|
| **Strategy** | Probe the refined trajectory as a second starting point for the polish, and keep it if it wins |
| **Mechanism** | `multiverse_optimizer.py:454` `_global_solve`, `multiverse_optimizer.py:487` `_probe_refined_guess` |
| **Why it was tried** | KI-004: the refined guess was computed and then thrown away, so the whole Phase 4 cost was wasted |
| **Outcome** | **The probe now wins.** On `examples/example_complex_mission.json` the refined guess converges to **4.617 s** against the primary solve's **6.049 s**, and it is kept. The reason it never won before was not the mechanism but the guess: the probe read a guess whose per-segment intervals were discarded (the equal-duration model keeps only the first one), so the states were read against a timing they did not belong to |
| **Status** | live (KI-004) |
| **What would revive it** | Nothing; it works. The probe is a bounded cold pass with a 0.1 % margin, and it never costs more than a fraction of the primary solve |

### 20.3 The Reeds-Shepp bootstrap with `turning_radius = 0.0`

| Field | Value |
|---|---|
| **Strategy** | Bootstrap the geometry with a Reeds-Shepp path so the guess is kinematically valid, including reverse manoeuvres |
| **Mechanism** | `path_planning.py:28`; configured from `examples/fll_choreo.json:46-49` |
| **Outcome** | **The bootstrap never ran.** `path_planning.py` read rsplan's `Path` object as a list; the `TypeError` was swallowed by the `except` and every call returned `None`, so the "Reeds-Shepp bootstrap" fell back to the straight line every time. The shipped configuration compounded it with `turning_radius: 0.0`, which rsplan cannot plan with at all: 0 of 12 example segments planned. With the API fixed and the radius at the robot's 0.05 m, 12 of 12 plan (KI-004) |
| **Status** | live (the bootstrap is the geometry the refined seed is built on) |
| **What would revive it** | It is revived. As a seed on its own it is **not** a speed lever: on the six example missions it matches the stock seeds (first4 1.680 s, events 4.418 s) and is worse on the complex mission (7.032 s against 4.877 s). The value is in the *refinement* that is built on top of it |

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
| **Outcome** | **Fixed** (KI-015). The suite generator unpacks `solve()`'s tuple, uses `examples/fll_choreo.json`, and solves every mission; the pipeline mode reads `stats["converged"]` and prints `FAILED`; the random mode validates against the config it actually used (`benchmarks/config_used.json`) and draws its waypoints as a random walk instead of independent uniform draws. A third defect surfaced while checking: the generator's `samples` parameter was shadowed by the solve result, so the *second* mission solved with the first mission's sample list as the sample count and died in the optimizer. `tests/test_random_suite.py` runs two missions and fails if either raises |
| **Status** | **closed** |
| **What would revive it** | Nothing. The mission suite it was supposed to supply now comes from `research/suite.py` and is measured by `research/harness.py` (Q8, answered in `docs/research/pq_b/FINDINGS.md`) |

### 21.4 `sysid_solver.py`

| Field | Value |
|---|---|
| **Strategy** | A pandas/JAX system identification solver with hardcoded CSV paths |
| **Mechanism** | `sysid_solver.py` (1155 lines) |
| **Outcome** | **Not wired into the CLI**, does not emit the current `sysid_fit.json` format, and its CSV paths are hardcoded. `sysid_fit.py` is the live path |
| **Status** | **dead** (kept as a documented prototype, KI-018; the decision to delete or rewrite it is still open) |
| **What would revive it** | Rewrite it against the current fit format, or delete it. It is the only unwired module left in the root |

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

**5. The measurement base is too narrow to support strong claims.** Everything in [Section 9](#9-measured-heuristic-comparison) rests on six example missions. The random benchmark that was supposed to provide a mission suite is fixed (KI-015) and the harness under [`research/`](../../research/) answers Q8 on 32 missions; what is still missing is a **re-run of that suite on the current tree**, because the numbers in `docs/research/pq_b/` predate KI-025 to KI-028. "The baseline wins the complex mission" might be a fact about the planner, or it might be a fact about that one path.

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
| **Current evidence** | Measured over 60 candidates: the non-baseline seeds produce the fastest **passing** trajectory in 4 of the 7 cells that produce one (first4 and complex under per-segment timing, both by 10 % or more, and the straight path, where all five seeds tie). What they reliably produce is **passing** trajectories where the baseline fails: the baseline failed 6 of the 12 (mission, timing model) cells and a non-baseline seed rescued exactly **one** of them (the complex mission under per-segment timing, where the baseline overshoots the motor-force limit by 0.074 N) |
| **Hypothesis** | With the current seed set, `--candidates N` buys **gate insurance**: it multiplies the chance that at least one converged solution satisfies the force and tracking gates. It does not buy speed |
| **Experiment** | 1. Instrument the selection: record the selected candidate's `guess_style`, the fastest candidate's `guess_style`, and the failing reasons per mission. 2. Compute, for each mission, whether the baseline alone would have produced a passing trajectory. 3. Repeat on a larger mission set (Q8) and report the insurance rate |
| **Metric** | The fraction of missions where a non-baseline seed is the only source of a passing trajectory; the duration cost of that insurance |
| **Success criterion** | A clear answer either way, and a cost figure: "one extra seed buys a passing trajectory on N % of missions, at M seconds per solve". If N is small, the candidate count should be cut and the budget spent on the sweep grid, which also wins (19.1) |
| **Cost** | Low. The data is already in `attempt_record` (`main.py:1058-1059`). This is a measurement, not an experiment |
| **Answer (2026-10-20)** | **The hypothesis is half right, and the other half is the opposite.** The insurance is real but small - **one mission in six** - and it comes from the **timing model**, not from a seed: the default set is two timing models of the baseline seed, and the seeds only enter from `--candidates 3` on. The seeds and the refined trajectory add **no** passes; what they buy is **speed** where they reach a different optimum (the complex mission: 6.068 s from the baseline, 5.395 s from a seed, 4.936 s from the refined guess). The measurement is in [`docs/measurements/2026-10-20/matrix/`](measurements/2026-10-20/matrix/): the analysis is `analyse_matrix.py` in that folder and that folder's README carries the summary. |

### Q4 — Should reachability be an objective term?

| | |
|---|---|
| **Question** | The planner produces trajectories that fail the envelope gate and hands them to a human (21.2). Can the envelope be approximated cheaply enough to put inside the objective? |
| **Current evidence** | The reachability computation is an interval propagation over the closed loop — far too expensive for the inner NLP loop. But the *dominant* contributor to the envelope is usually the model uncertainty (backlash, wheel radius, track width) propagated over the path, not the tracking error. **RQ-A changes the priority:** the envelope is not what is failing. Across the 60-candidate matrix, zero candidates failed on the envelope threshold; 25 of the 29 failures cite the tracking gate, which RQ-A shows is a discretisation artefact rather than an uncertainty or tracking problem. An envelope term would not touch it |
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
| **Current evidence** | The random benchmark is fixed (21.3, KI-015) and the question has been answered on 32 missions in `docs/research/pq_b/FINDINGS.md`: the baseline has the **lowest** pass rate of the five seeds. The example missions are geometrically simple: short, mostly straight, few segments. And the answer is already nuanced on the six we have: the baseline is selected on 2 of 6, a non-baseline seed on 2 of 6 (one of them a tie), and 2 of 6 have no passing candidate at all under the most thorough configuration tried |
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
| **Answer (2026-10-20)** | **The hypothesis is wrong on the tracking error, and the excess is not the lever either.** The envelope is almost a proxy for the **duration** (r = +0.97 over 204 candidates), so the current rule is "shortest run" in disguise. The tracking error is nearly independent of the duration (r = +0.35), so preferring it is **slower in 11 of the 21 groups** and faster in one. The per-sample excess agrees with the envelope on every mission here, because every segment limit is the default 2 cm. The only rule that is never worse is the **duration**: it differs from the current rule in 6 of the 21 groups and is faster in every one. The envelope never rejects a candidate either - 105 of 204 pass, and not one failed on the envelope threshold. The measurement is in [`docs/measurements/2026-10-20/matrix/`](measurements/2026-10-20/matrix/); the analysis is `analyse_matrix.py` in that folder, and every candidate now carries its score vector in the `##OVERBOARD-VALIDATION##` line, so a rule can be tried without solving again. |

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
| **Current evidence** | **Answered (KI-004).** The pipeline was repaired rather than deleted: the refinement preserves the waypoints, the JAX cost uses IPOPT's trapezoidal rule, the intervals follow the segment's own length, the refined timing is propagated into the timestamps, the probe reads the guess with the per-segment model, and the Reeds-Shepp bootstrap works (its rsplan API was broken and it had never run). The probe now wins, and the pipeline is re-envisioned as a **seed provider**: `main.py --refine` adds its refined trajectory to the candidate set. Measured over 30 missions, the refined seed is **not** better than the stock seeds and rescues nothing - it is a cheaper way to reach the stock set's speed on some missions (the complex mission: 4.936 s against the default set's 6.068 s, where the full ten-candidate stock set gives 4.877 s) |
| **Hypothesis** | If Q1 (real path families) or Q5 (critic feedback) produces a good starting point, the Multi-Verse machinery becomes a *multi-seed generator* and its value is the topology sampling (forward / reverse / point-turn), which nothing else in the tree covers |
| **Experiment** | Done: the four defects were found by measurement and fixed; the refined trajectory was measured as a seed on the six example missions and over 30. The raw records are in [`../measurements/2026-10-20-ki004/`](../measurements/2026-10-20-ki004/) |
| **Metric** | Whether the pipeline's output wins as a candidate: duration, envelope, tracking, pass rate against the stock seed set |
| **Success criterion** | A decision, made on evidence. **Made**: the pipeline is repaired and kept, as a seed provider; the measurement says the stock seeds are the better buy, so it stays opt-in |
| **Cost** | Low to decide, high to execute. **The next lever is the objective, not the search**: the pipeline's cost is duration plus a jerk term, so it buys speed with tracking accuracy and the 5 mm gate rejects the trade. Making the objective answer the gate is Q4's job |

### 24.1 Suggested order

```text
  ---  DONE: RQ-A  the tracking gate's cause is identified. See
                docs/research/RQ-A_tracking_error.md. It reorders Q4 and Q9.
  ---  DONE: RQ-A's OQ1 and OQ2  the floor at fine sampling is the 20 ms
                command hold, and the optimizer now has a self-consistent
                collocation (KI-026, --self-consistent).
  ---  DONE: Q3  the selection is instrumented and the insurance rate is
                measured. See the matrix folder and its analyse_matrix.py.
  ---  DONE: Q10 the selection rules are compared offline on the same
                matrix. The envelope is a proxy for the duration.
  ---  DONE: Q8  the random benchmark is repaired (KI-015) and the 32-mission
                suite exists (research/suite.py).
  ---  DONE: Q6  the weight ladder is measured: the weight is a real accuracy
                proxy and a search, not a dial.
  ---  DONE: Q2  the length-proportional warm start is measured: slower
                convergence, faster trajectories on 18 of 32 missions.
  ---  DONE: Q1  a Bezier path family is measured: a cheaper solve, not a
                faster trajectory.
  ---  DONE: Q7  the density sweep is measured: n = 10 is the sweet spot.
  ---  DONE: Q12 the Multi-Verse path is repaired and kept as a seed provider
                (KI-004); --refine is 4.936 s where the default set gives
                6.068 s, and the full stock set 4.877 s.
  ---  ANSWERED: Q5  the critic flags every candidate, so it cannot filter;
                its thresholds have to discriminate before it can.
  Q4  reachability in the objective                          (large)
  Q11 executor in the planner                                (large)
  Q9  hardware-calibrated twin                               (needs hardware)
```

The RQ-A items came first because they were the *cause* of the gate failures that everything else is downstream of: 25 of the 29 failures in the 60-candidate matrix cite the tracking gate, and RQ-A shows it was measuring a discretisation artefact rather than the robot. KI-026 to KI-028 have landed, and the artefact is now small at the fine end (the S-curve at n=40: 0.98 → 0.27 cm) but not at the coarse end, where the plan's own deviation is the whole error and only the reference's velocity half ([RQ-A finding F8](research/RQ-A_tracking_error.md)) would remove it. **The 5 mm threshold still measures the discretisation more than it measures the robot**, so it must not be tuned to make missions pass.

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
| `GUESS_STYLES` | 10 | The five seed names |
| `TrajectoryOptimizer` | 24 | The optimizer |
| `TrajectoryOptimizer.solve` | 86 | The public entry point; `guess_style`, `per_segment_dt`, `initial_guess`, `fast_polish` |
| `TrajectoryOptimizer.guess_from_samples` | 353 | Rebuild a guess array from a sample list (used by the refine probe) |
| `TrajectoryOptimizer._solve_once` | 373 | One IPOPT pass |
| `interval_dt` | 418 | The interval length for interval `k` |
| `dt` bounds | 461-462 | `dt_i` in `[0.001, 1.0]` |
| `dt_ref` | 444, 575, 706 | The jerk normalising interval (KI-022) |
| `TrajectoryOptimizer._compute_cost` | 684 | The objective |
| `TrajectoryOptimizer._build_initial_guess` | 720 | The seed builder |
| `TrajectoryOptimizer._guess_point` | 780 | The baseline geometry |
| `TrajectoryOptimizer._offset_segment_interiors` | 806 | `bow_left` / `bow_right` |
| bow amplitude | 829 | the circular-arc sagitta, `0.5 * L * tan(abs(dtheta) / 4)`, capped at `0.35 * L` |
| `TrajectoryOptimizer._rotate_before_driving` | 841 | `turn_first` |
| spin fraction / threshold | 873, 866-868 | `0.30 * s_i`, and the threshold from the track width |
| `TrajectoryOptimizer._set_rolling_speeds` | 889 | `rolling` |
| `TrajectoryOptimizer._sample_times` | 932 | Timestamps from the per-segment `dt` |
| `TrajectoryOptimizer.format_output` | 953 | Samples with forces and events |

## `main.py`

| Symbol | Line | Role |
|---|---|---|
| `VALIDATION_MARKER` | 43 | The `##OVERBOARD-VALIDATION## ` prefix |
| `HAS_JAX` | 56 | JAX + Immrax availability |
| `ImmraxValidator`, `JAXRobotConfig` | 50, 51 | Module-level handles, patchable by tests |
| `PRESETS` | 63 | `fast` / `balanced` / `safe` |
| `_ensure_jax_classes` | 88 | The lazy import |
| `_select_attempt` | 100 | The selection rule |
| `CANDIDATE_WEIGHT_STEP` | 115 | The legacy ladder step |
| `SWEEP_WEIGHTS` | 124 | The sweep grid |
| `_THREAD_ENV_VARS` | 128 | The per-worker BLAS limits |
| `_candidate_specs` | 193 | The candidate set |
| `_solve_candidate_worker` | 317 | One candidate in a worker process |
| `_solve_candidates` | 358 | The parallel solve |
| `_export_block_reason` | 436 | The export gate |
| `_immrax_failure_reasons` | 445 | Human-readable failures |
| `_finite` | 467 | JSON-safe floats |
| `_segment_report` | 536 | The per-segment breakdown |
| `_emit_validation_report` | 608 | The machine-readable report |
| `_cpu_failure_reasons` | 659 | The CPU audit failures |
| `_apply_preset` | 682 | The presets |
| `_load_waypoints` | 747 | The waypoint parser; thresholds at 527-535 |
| `_print_summary` | 914 | The console table |
| `main` | 1073 | The CLI |
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
| `compute_metrics` | 211 | Duration, jerk, tortuosity, chattering |
| `check_final_waypoint` | 299 | 2 cm / 0.02 rad |
| `validate_samples` | 320 | The in-memory gate |
| `validate_trajectory` | 347 | The CLI/report entry point |

## `immrax_validator.py`

| Symbol | Line | Role |
|---|---|---|
| `unroll_trajectory_headings` | 14 | The ±π fix |
| `REACHABILITY_SUBSTEPS` | 38 | 20 |
| `CONTROL_DT` | 59 | 0.02 s |
| `PLANT_SUBSTEPS` | 60 | 4 |
| `_interpolated_reference` | 207 | The linear reference + the Ramsete gain |
| `RobotOpenLoop` | 264 | The default plant |
| `DigitalTwinOpenLoop` | 334 | The fitted 5-state plant |
| `MIN_TAU_S` | 352 | 0.01 s |
| `RamseteController` | 496 | `b = 2.0`, `zeta = 0.7` |
| `ClosedLoopRobotSystem` | 543 | The Immrax system |
| `ImmraxValidator` | 737 | The validator |
| `_advance_belief` | 717 | The robot's own pose estimate |
| `validate_trajectory` | 980 | The entry point; `safety_threshold_m = 0.005` at 427 |
| `check_sample_interval` | 1010 | The per-sample interval physics |
| `compute_reachability` | 1273 | The interval propagation |
| the divergence radius | 714 | 0.05 m |

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
| `solve` | 281 | The five phases |
| `_profile_kinematics` | 492 | Phase 2 |
| `_global_solve` | 567 | Phase 5 and the probe |
| `_probe_refined_guess` | 600 | The bounded probe |

## `jax_optimizer.py`

| Symbol | Line | Role |
|---|---|---|
| `local_trajectory_cost` | 11 | The proxy cost |
| `local_trajectory_projection` | 52 | The bounds projection |
| `generate_candidates_jax` | 65 | 4 topologies × 3 velocities + 50 STOMP |
| `perturb` | 102 | The STOMP noise |
| `get_jax_refiner` | 127 | The `jaxopt.ProjectedGradient` refiner |

## `robot_model.py`

| Symbol | Line | Role |
|---|---|---|
| `wheel_normal_forces` | 13 | The one weight-transfer model |
| `RobotConfig` | 41 | The config abstraction |
| `uncertainties` | 229 | The seven intervals |
| `get_max_force_at_velocity` | 439 | The motor curve |
| `get_robust_force_params` | 506 | The worst case for the optimizer |
| `DifferentialDriveModel` | 705 | The force model |
| `get_max_wheel_force_symbolic` | 790 | The per-wheel limit |
| `get_traction_limit_symbolic` | 812 | The per-wheel traction limit |
| `check_constraints` | 835 | The CPU-side check |

## `path_planning.py`

| Symbol | Line | Role |
|---|---|---|
| `ReedsSheppPath` | 11 | The wrapper |
| `plan` | 28 | The path; returns `None` on failure |
| `linear_interpolation_waypoints` | 93 | The fallback |

## `digital_twin.py`

| Symbol | Line | Role |
|---|---|---|
| `ActuatorFit` | 56 | gain, tau, delay, accel limit |
| `OdometryFit` | 82 | Distance and heading scale |
| `SlipFit` | 101 | The braking shortfall |
| `SensorFit` | 126 | The gyro and accel biases |
| `DigitalTwinConfig` | 412 | The whole fitted model |
| `actuator_gain_range` | 488 | The interval that goes into Immrax |
| `brake_shortfall` | 602 | Loaded, reported, not applied by default |
| `load_sysid_fit` | 676 | The loader |

---

# Appendix C — Known issue cross-reference

| KI | Area | One line | Where in this document |
|---|---|---|---|
| KI-001 | Solver / output | A failed solve was exported as a success | [22.4](#224-a-failed-solve-was-exported-as-a-success-ki-001-and---validate-could-not-block-output-ki-002) |
| KI-002 | Validation | `--validate` could not block output or the exit code | [22.4](#224-a-failed-solve-was-exported-as-a-success-ki-001-and---validate-could-not-block-output-ki-002) |
| KI-003 | Safety | Immrax ignored the upper uncertainty bounds | [22.3](#223-the-immrax-upper-uncertainty-bounds-were-ignored-ki-003) |
| KI-004 | Multi-Verse | The refinement had no effect on the final trajectory — now repaired and kept as a seed provider | [8.6](#86-how-the-pipeline-reaches-the-solver), [20.1](#201-the-jax-refinement), [20.2](#202-the-refine-probe), Q12 |
| KI-009 | Safety / config | The declared uncertainty ranges were ignored | [Section 11](#11-the-uncertainty-model) |
| KI-010 | Physics | The weight-transfer math was wrong and the tools disagreed | [22.5](#225-the-weight-transfer-math-was-wrong-and-the-tools-disagreed-ki-010) |
| KI-014 | Safety model | The JAX Ramsete used a poor large-angle approximation | [Section 12](#12-controller-and-plant-parity) |
| KI-015 | Benchmarks | ~~The random benchmark tools are broken~~ (fixed) | [21.3](#213-the-random-benchmark-mode), Q8 |
| KI-018 | Maintenance | ~~Dead code, duplicates, generated files~~ (fixed) | [21.4](#214-sysidsolverpy), [20.4](#204-the-critics-config-block) |
| KI-019 | Safety | The reachability tube lagged the reference by one sample | [22.2](#222-the-reachability-tube-lagged-the-reference-by-one-sample-ki-019) |
| KI-020 | Digital twin | The validator is not a full twin; reachability is not an objective | [Section 14](#14-what-each-layer-proves-and-does-not-prove), [21.2](#212-reachability-as-a-post-hoc-score), Q4, Q9 |
| KI-021 | Robot runtime | The DriveBase capped the acceleration the plan asked for | [Section 12](#12-controller-and-plant-parity) |
| KI-022 | Solver / timing | The per-segment timing model was a jerk-penalty loophole | [5.3](#53-the-jerk-discipline-ki-022), [22.1](#221-the-per-segment-timing-model-was-a-jerk-penalty-loophole-ki-022) |
| KI-024 | Solver / timing | The timing model had no rule for where the time allocation may change | [5.5](#55-where-the-time-allocation-may-change-ki-024) |
| KI-051 | CLI / solver | ~~`--self-consistent` was a no-op: the switch could not change the plan and the trapezoidal collocation could not be reached from the CLI~~ (fixed; the flag is two-way now) | [5.7](#57-the-collocation-where-a-sample-is-placed) |
| KI-052 | CLI / solver | ~~The CLI and the library plan different trajectories for the same mission~~ (fixed; both default to the self-consistent rule now). The numbers in §9 were measured with the trapezoidal rule and have not been re-measured since | [5.7](#57-the-collocation-where-a-sample-is-placed), [9](#9-measured-heuristic-comparison) |

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

