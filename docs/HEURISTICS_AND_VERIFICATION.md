# Path Planning Heuristics and Verification Strategy

**Status:** 2026-10-19. Describes the working tree as measured on this date.
**Scope:** every path-planning heuristic in the repository and the verification chain that judges its output. The two research records have their own files: [`STRATEGY_LEDGER.md`](STRATEGY_LEDGER.md) (the strategies that did not work) and [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md) (the open questions).
**Audience:** anyone who wants to improve the planner, the validator, or both.

> **Archive note.** Raw logs and experiment scripts referenced below live in the `archive/research-2026-10` tag, not in the product tree. Retrieve a file with `git show archive/research-2026-10:<path>`; see [`ARCHIVE.md`](ARCHIVE.md).
>
> **How to read the module references.** The runtime moved into the `overboard/` package after this
> document was written, so a bare `overboard/core/optimizer.py`, `overboard/main.py`, `overboard/verification/validator.py`, `overboard/verification/immrax_validator.py`,
> `overboard/models/robot_model.py`, `overboard/sysid/digital_twin.py`, `overboard/core/frontier_seed.py` or `overboard/core/unified_planner.py` names the file that
> now lives under `overboard/core/`, `overboard/`, `overboard/verification/`, `overboard/models/`,
> `overboard/sysid/`, `overboard/core/` and `overboard/core/`. Every `file:line` is the line in the
> tree as measured on 2026-10-19, not in today's file. The measurements are the record; the tree is
> the authority on where the code is.
>
> **Read this first.** Every number in this document is a **model prediction**, not a hardware measurement. No hardware logs exist in the repository. The robot-side collector (`pybricks_code/sysid_digital_twin.py`) can produce them, but nothing has been captured yet. The uncertainty ranges in `examples/fll_choreo.json` are *declared test assumptions*, not calibrated hardware values. A "PASS" in this document means "the trajectory passes the gates of an uncalibrated model", not "the robot will track it".

---

## Findings at a glance

1. **The heuristic layer is small.** Three analytical frontier candidates (the default), or five deterministic seeds and two timing models (the stock set), a candidate set, and a selection rule. There is **no geometric path family** anywhere in the tree — no splines, clothoids, béziers or quintics — and the objective has no term that prefers one path over another. It has three terms that price the *motion* (the duration, the wheel jerk and, optionally, the wheel force), and the passing candidates are ranked by a 5-part fitness score in seconds ([7.2](#72-the-5-part-path-fitness-score)) - or, on the default path, by the shortest duration ([7.3](#73-the-selection-rule)).
2. **The seeds do not rescue the baseline often, but they do win on speed.** Over the 60-candidate matrix the `baseline` seed is the fastest passing candidate in 1 of the 7 cells that produce a trajectory at all; a non-baseline seed wins the other 6. The baseline fails in 9 of the 12 cells, and **four** of them are rescued by another seed: first4 under both timing models, the complex mission under equal-duration, and the events mission under per-segment. In the other 5 — the S-curve and the stops mission, under both models — no seed passes either.
3. **The tracking gate is the binding constraint; the reachability envelope never binds.** Of 34 failing candidates, 28 failed on the 5 mm tracking gate and 11 on motor force. **Zero** failed on the forward-integration check — it went with the self-consistent collocation — and **zero** on the 2 cm envelope threshold. The largest envelope measured anywhere was 1.72 cm.
4. **Two of the six example missions still produce no output at all.** The S-curve and the stops mission fail **every candidate** in the full sweep — 0 of 9 each, across both timing models and the whole accuracy-weight grid. Their best tracking errors are 1.27 cm and 0.75 cm against a 0.50 cm gate. The self-consistent collocation removed the plan's own deviation from the picture, and what is left of the error is the reference's velocity half ([RQ-A finding F8](research/RQ-A_tracking_error.md)).
5. **The timing model is a bigger lever than any seed, but not on every path.** Per-segment timing is 1.9x faster on the complex mission (4.812 s against 9.249 s) and it is what makes the events mission solvable at all; on the two paths whose segments are short and even (first4, the stops mission) the equal-duration model is the faster one. The seeds changed the outcome on 3 missions; the timing model on 1. The per-segment model also had to be made legitimate at the waypoints the robot drives through (KI-024, [5.5](#55-where-the-time-allocation-may-change-ki-024)); the matrix in [Section 9](#9-measured-heuristic-comparison) was re-measured after that change.
6. **The accuracy weight is the cheapest lever of all.** On the first-four-waypoints mission, the *same* `baseline` seed fails at weight 0 (it misses the motor-force limit by 0.006 N) and passes at weight 0.001 (0.26 cm tracking). The sweep grid is the only candidate-set change that has ever produced a new best trajectory, and it works because it adds a genuinely different parameter.
7. **The Multi-Verse pipeline was dead weight; it is repaired, and it is archived.** Its JAX-refined guess used to converge to 8.1473 s against the trivial baseline's 6.0677 s. Four defects were behind that - the refinement moved the waypoints, the JAX cost graded a different motion than IPOPT produces, the intervals were hard-coded, and the refined timing was discarded - and with them fixed the probe wins (4.617 s against 6.049 s) and the pipeline is reproducible. But measured over 30 missions as a *seed provider*, the refined seed is **not** better than the stock seeds (it is faster on 2 of the 9 missions where both pass, slower on 6) and it **rescues nothing**; it is a cheaper way to reach the stock set's speed on some missions (KI-004). It has been removed from the tree (git tag `archive/legacy-2026-10`) and has no CLI entry point.
8. **The measurement base is six hand-written paths.** The random benchmark that was supposed to supply a mission suite is fixed (KI-015), and a mission-suite harness exists under [`research/`](../research/). The six paths are still what [Section 9](#9-measured-heuristic-comparison) quotes; the wider suite is measured in `docs/research/pq_b/`.
9. **Twelve research questions are listed in [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md)**, ordered so that the cheapest decisive experiment comes first. The first two are measurements, not experiments.
10. **RQ-A is done, and it reorders the agenda.** [`docs/research/RQ-A_tracking_error.md`](research/RQ-A_tracking_error.md) identifies why the tracking gate binds: the optimizer's collocation, the controller's linear reference and the plant's integration are three different discretisations of the same trajectory, and on a curved path they disagree by about a centimetre. The optimizer's half of that disagreement is closed - the self-consistent collocation is the default ([5.7](#57-the-collocation-where-a-sample-is-placed)) - and the reference's half is not. The tracking error is **not** an uncertainty problem and **not** a physical tracking problem. Fix the discretisation before reaching for reachability in the objective. RQ-A also finds that with a *calibrated* model the binding gate flips to the **envelope**, because the measured actuator-gain band widens the reachable set.

> **Read RQ-A before changing the tracking gate, the reference interpolation, or the optimizer's kinematics.** It contains the measurements, the refuted hypotheses, and the four instrument bugs that produced convincing nonsense along the way.
>
> **Which planner these findings describe.** The numbers below are the **stock** candidate set - the five seeds that are now the rescue round - measured on the tree of 2026-10-20. The stock path itself was retired with the other multi-track paths; the only planner now is the 3-Pillar unified engine with the three frontier candidates ([1.1](#11-the-default-system-the-3-pillar-unified-engine)), and its own measurements are in [19.6](STRATEGY_LEDGER.md#196-the-3-pillar-unified-trajectory-engine---unified-the-default-path) and in `docs/OPTIMAL_CONTROL_DERIVATION.md`.

---

## How to use this document

| If you want to | Read |
|---|---|
| know what the planner does today | [Part I](#part-i--the-path-planning-heuristic-landscape), sections 1-7 |
| know what the gates are and what they mean | [Part II](#part-ii--the-verification-strategy), sections 10-15 |
| know what the measured comparison says | [Section 9](#9-measured-heuristic-comparison) |
| know what was tried and failed, and why | [`STRATEGY_LEDGER.md`](STRATEGY_LEDGER.md) (Part III) |
| know what is still open | [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md) (Part IV) |
| find a symbol | [Appendix B](#appendix-b--where-things-live) |

The ledger and the agenda are the research record: they are not needed to use the planner, and
no strategy in the ledger is to be re-litigated without new evidence.

## Table of contents

  - [Findings at a glance](#findings-at-a-glance)
  - [How to use this document](#how-to-use-this-document)
- [Part 0 — Orientation](#part-0--orientation)
  - [0.1 What this document is](#01-what-this-document-is)
  - [0.2 Scope, authority and how to read a citation](#02-scope-authority-and-how-to-read-a-citation)
  - [0.3 Notation](#03-notation)
  - [0.4 Measurement conventions](#04-measurement-conventions)
- [Part I — The path planning heuristic landscape](#part-i--the-path-planning-heuristic-landscape)
  - [1. Two heuristic systems](#1-two-heuristic-systems)
  - [2. What is not in the tree](#2-what-is-not-in-the-tree)
  - [3. The default path, end to end](#3-the-default-path-end-to-end)
  - [4. The stock seed heuristics](#4-the-stock-seed-heuristics)
  - [5. The timing models](#5-the-timing-models)
  - [6. Candidate set construction](#6-candidate-set-construction)
  - [7. Scoring and selection](#7-scoring-and-selection)
  - [8. The archived Multi-Verse path](#8-the-archived-multi-verse-path)
  - [9. Measured heuristic comparison](#9-measured-heuristic-comparison)
- [Part II — The verification strategy](#part-ii--the-verification-strategy)
  - [10. The gate chain](#10-the-gate-chain)
  - [11. The uncertainty model](#11-the-uncertainty-model)
  - [12. Controller and plant parity](#12-controller-and-plant-parity)
  - [13. The system identification loop](#13-the-system-identification-loop)
  - [14. What each layer proves and does not prove](#14-what-each-layer-proves-and-does-not-prove)
  - [15. Test coverage map](#15-test-coverage-map)
- [Part III–IV — the research record, in their own documents](#part-iiiiv--the-research-record-in-their-own-documents)
  - [STRATEGY_LEDGER.md — the failed strategy ledger](STRATEGY_LEDGER.md) (Part III, §16–§23)
  - [RESEARCH_AGENDA.md — the research agenda](RESEARCH_AGENDA.md) (Part IV, §24–§25)
- [Appendix A — Reproducing every table](#appendix-a--reproducing-every-table)
  - [A.1 The heuristic comparison matrix](#a1-the-heuristic-comparison-matrix)
  - [A.2 The timing × accuracy-weight grid](#a2-the-timing--accuracy-weight-grid)
  - [A.3 The seed comparison in one line](#a3-the-seed-comparison-in-one-line)
  - [A.4 From Python, without the CLI](#a4-from-python-without-the-cli)
  - [A.5 The reference solver table](#a5-the-reference-solver-table)
  - [A.6 The validation pipeline only](#a6-the-validation-pipeline-only)
  - [A.7 The test suite](#a7-the-test-suite)
- [Appendix B — Where things live](#appendix-b--where-things-live)
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

The third half — **what was tried and failed** — is [`STRATEGY_LEDGER.md`](STRATEGY_LEDGER.md). It exists because the obvious next move ("add a smarter path planner") has already been tried several times in this codebase, in several different forms, and each time it lost to the trivial baseline. The ledger records those attempts, the measured evidence, and what would have to change for a retry to be worth anything. [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md) turns the ledger into concrete experiments.

## 0.2 Scope, authority and how to read a citation

**The source is the authority.** Where this document and the code disagree, the code wins and this document is a bug. Every `file:line` citation was verified against the working tree on 2026-10-19; the tree is uncommitted work, so the line numbers are only meaningful for that snapshot.

Citations use the form `path/file.py` (`LINE`). A symbol is written as `Class.method` when the class is unambiguous in that file.

Two files hold most of the subject matter:

| File | Role |
|---|---|
| `overboard/core/optimizer.py` | The CasADi/IPOPT optimizer and the five initial-guess seeds |
| `overboard/main.py` | The CLI, the candidate set, the scoring, the selection and the export gate |
| `overboard/verification/validator.py` | The CPU audit: forward integration, force/slip audit, metrics |
| `overboard/verification/immrax_validator.py` | The closed-loop interval reachability check |
| (removed) | The Multi-Verse pipeline, the Reeds-Shepp bootstrap and the JAX refiner - git tag `archive/legacy-2026-10`, no shipped imports |
| `overboard/models/robot_model.py` | The robot configuration and the shared force model |
| `overboard/sysid/digital_twin.py` | The optional fitted plant |
| `overboard/sysid/fit.py` | The host-side fitter |

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

- **Measured 2026-10-19** — produced by the sweep in [Section 9](#9-measured-heuristic-comparison) against this tree. The raw logs are in `archive/research-2026-10:docs/measurements/2026-10-19/`, and the exact commands are in [Appendix A](#appendix-a--reproducing-every-table).
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

The list is generated, not copied: `archive/research-2026-10:docs/measurements/2026-10-20-variable-samples/TREE.md5` holds it, and the runner writes it from the tree it measured. The runner also takes the fingerprint **before and after** the run and refuses a measurement whose tree changed underneath it, so a run that straddles two planners cannot be quoted as one.

Environment: Windows 11, Python 3.12.10, `casadi`, `jax`, `immrax`, `jaxopt`, `rsplan`, `numpy`, `scipy` all installed. Test suite state at the time of writing: `python run_tests.py` → **623 tests**, 0 skipped.

---

# Part I — The path planning heuristic landscape

## 1. Two heuristic systems

The repository contains two entirely separate planning heuristics. They share the robot model and the validator. They do not share the candidate set, the scoring, or the seeds.

### 1.1 The default system: the 3-Pillar unified engine

**Entry point:** `overboard/main.py` → `UnifiedPathEngine` (`overboard/core/unified_planner.py`) → `TrajectoryOptimizer.solve` (`overboard/core/optimizer.py`).

```
waypoints
   │
   ├─ mesh          : unified_planner.py  compute_invariant_mesh_samples  (10-18 per segment, 50 ms tick span)
   ├─ seed          : frontier_seed.py     generate_frontier_seed          (3 styles: sprint / sweeper / exploratory)
   ├─ timing model  : per segment, always
   ├─ objective     : cost_metric "discretization" + linear epigraph force bounds
   ├─ NLP solve     : optimizer.py         _solve_once  (IPOPT, max_iter 150)
   ├─ format        : optimizer.py         format_output
   │
   ├─ CPU audit     : validator.py         validate_samples
   ├─ Immrax        : immrax_validator.py  validate_trajectory
   ├─ scoring       : main.py              per-candidate tube / tracking / ok / fitness
   ├─ selection     : main.py              _select_attempt   (shortest duration first - see 7.3)
   └─ export gate   : main.py              _export_block_reason
```

This is the default path (`--unified` is on). The three frontier candidates are `frontier_sprint` (accuracy weight 0.001), `frontier_sweeper` (0.002) and `frontier_exploratory` (0.001), all per-segment timing; `--candidates N` truncates that list, and without `--validate` only the first one is solved. The derivation of the seeds is in [19.5](STRATEGY_LEDGER.md#195-the-kinetic-frontier-analytical-seed-generator--the-5-physical-principles) and of the engine in [19.6](STRATEGY_LEDGER.md#196-the-3-pillar-unified-trajectory-engine---unified-the-default-path).

### 1.1b The stock path (retired)

**Entry point:** `overboard/main.py` → `TrajectoryOptimizer.solve` (`overboard/core/optimizer.py`).

```
waypoints
   │
   ├─ seed          : optimizer.py  _build_initial_guess   (5 styles: baseline, rolling, bow_left, bow_right, turn_first)
   ├─ timing model  : optimizer.py  one dt, or one dt per segment
   ├─ NLP solve     : optimizer.py  _solve_once  (IPOPT, max_iter 5000)
   ├─ format        : optimizer.py  format_output
   │
   ├─ CPU audit     : validator.py         validate_samples
   ├─ Immrax        : immrax_validator.py  validate_trajectory
   ├─ scoring       : main.py              per-candidate tube / tracking / ok / fitness
   ├─ selection     : overboard/main.py    _select_attempt   (fastest first - see 7.3)
   └─ export gate   : main.py              _export_block_reason
```

This is the path the measurements in [Section 9](#9-measured-heuristic-comparison) describe. It was selected with `--no-unified` (alias `--simple`); that path is retired, and so are `--uniform-dt`, `--per-segment-dt` and `--sweep` as selectors: the CLI accepts them, warns, and ignores them. The five seeds survive as the rescue round ([1.1c](#11c-the-rescue-round-ki-048)). The Multi-Verse system has been removed (git tag `archive/legacy-2026-10`); no CLI flag selects it.

### 1.1c The rescue round (KI-048)

**Entry point:** `overboard/main.py` -> `_run_candidate_rounds` -> `_rescue_specs`.

When the planner's three candidates produce nothing usable - no candidate
passes the gates, or (without validation) none converged - the CLI tries the
seeds the set did not use before it gives the mission up. The five stock
seeds (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`) are
therefore still in the tree, and only there.

The rescue uses the per-segment model on a validated run, because that is
where the measured rescues came from, and the same model on an unvalidated
one. A rescue candidate is only *selected* when the candidate set itself
produced nothing that passed; it costs nothing when the plan works, and one
parallel round of solves when it does not.

### 1.2 The archived system: Multi-Verse

The five-phase Multi-Verse pipeline was removed from the tree (git tag `archive/legacy-2026-10`) and
has no CLI entry point. It was a seed provider, not a better planner; the
measured conclusion is in [Section 8](#8-the-archived-multi-verse-path) and
[Section 20](STRATEGY_LEDGER.md#20-ledger-archived-multi-verse-internals).

## 2. What is not in the tree

This is the list a newcomer most needs, because it defines the whole research space.

| Missing | Consequence |
|---|---|
| **Any geometric path family.** No splines, clothoids, béziers, quintics, Dubins or Elford paths. | The only path primitive in the shipped planner is a straight line between waypoints (`overboard/core/optimizer.py` `_guess_point`). Reeds-Shepp existed only in the archived Multi-Verse bootstrap (`archive/legacy-2026-10:overboard/legacy/path_planning.py`), and only when both segment endpoints have a heading. |
| **Obstacle or field geometry.** No walls, no mat boundaries, no zone constraints. | The robot is assumed to reach every waypoint in a straight line through space. |
| **A geometric term in the objective.** The objective is duration plus an optional jerk penalty (`overboard/core/optimizer.py`, `_compute_cost`). | The optimizer has no idea what the path *looks* like. It is pushed along by the integration constraints and the endpoint equalities. Two paths of the same length and the same smoothness are indistinguishable to it. This is the root cause of many entries in [`STRATEGY_LEDGER.md`](STRATEGY_LEDGER.md) (Part III). |
| **Topology search.** No search over forward/reverse/point-turn combinations per segment. | The reverse and point-turn topologies existed only inside the archived JAX candidate generator (`archive/legacy-2026-10:overboard/legacy/jax_optimizer.py`), which never reached the shipped solver. |
| **Multi-start beyond the frontier triad.** | The default path solves three frontier candidates (sprint, sweeper, exploratory); the retired stock path solved five deterministic seeds, all from the same straight-line geometry, and those seeds are now the rescue round. |
| **A reachability term.** Reachability is computed *after* the solve. | The optimizer cannot know that a path it is about to find will fail the envelope gate. See [Section 14](#14-what-each-layer-proves-and-does-not-prove). |
| **Motor or odometry dynamics in the optimizer.** | The optimizer is kinematic. The plant is modelled only in the validator. |

## 3. The default path, end to end

### 3.1 The candidate loop

`overboard/main.py` builds a candidate list, solves each candidate independently, validates each one, scores each one, and picks one.

```python
# overboard/main.py
use_validation = HAS_JAX and validate
specs_to_solve = candidate_specs if use_validation else candidate_specs[:1]
```

**Without `--validate`, only candidate 1 is solved.** On the default path that is `frontier_sprint`; on the stock path it is equal-duration `baseline`. This is deliberate: without the validator there is no way to tell whether a different candidate is better, so there is no reason to spend the time.

**With `--validate`,** the candidates are solved and compared. On the default path the set is the three frontier candidates, truncated to `--candidates N` (the flag itself has no upper bound; `N < 3` shortens the triad). On the stock path each candidate is an independent `(seed, timing model, accuracy weight)` triple, and `--candidates N` says how many of them are solved.

### 3.2 Solve

Each candidate is solved in its own NLP. With more than one candidate, they are solved in worker processes (`overboard/main.py`, `_solve_candidates`), with BLAS pinned to one thread per worker. The trajectories are identical to a sequential run; `tests/test_cli_ux.py` (`TestParallelCandidates`) asserts this.

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

Then `_select_attempt` (`overboard/main.py`) picks one - the **shortest passing candidate** on the default path, the **lowest fitness score** on the stock path. See [Section 7](#7-scoring-and-selection) and [7.3](#73-the-selection-rule).

### 3.4 Export

The export gate (`overboard/main.py`) blocks the write if the solver did not converge, or if `--validate` was requested and the selected candidate did not pass. On a blocked export, **nothing is written**: no `.traj`, no controller JSON, no Python module, no `_stats.json`. A pre-existing file at the output path is left untouched.

## 4. The stock seed heuristics

The default path uses the three **frontier** seeds of [1.1](#11-the-default-system-the-3-pillar-unified-engine); the five seeds below are the stock set, retired with the stock path and kept only as the rescue round ([1.1c](#11c-the-rescue-round-ki-048)). They are declared in one line (`overboard/core/optimizer.py`):

```
GUESS_STYLES = ("baseline", "rolling", "bow_left", "bow_right", "turn_first")
```

```python
GUESS_STYLES = ("baseline", "rolling", "bow_left", "bow_right", "turn_first")
```

They are built by `_build_initial_guess` (`overboard/core/optimizer.py`). The header comment there states the contract:

> Distinct deterministic initial guesses for candidate generation. They are only guesses: IPOPT still enforces every waypoint, stop, and force limit.

### 4.1 The shared baseline geometry

All five seeds start from the same geometry, produced by `_guess_point` (`overboard/core/optimizer.py`). For each sample at fraction `frac` along segment `i`:

- **Position** is a straight-line interpolation between the two waypoints: `p1 + frac * (p2 - p1)`.
- **Heading** depends on which endpoint headings are known:

| Known headings | Heading guess |
|---|---|
| Both `p1` and `p2` | Angle-wrapped linear interpolation: `theta1 + wrap(theta2 - theta1) * frac` |
| Only `p1` | Blend from the constrained start heading toward the path direction `atan2(dy, dx)` |
| Only `p2` | Blend from the path direction toward the constrained end heading |
| Neither | Blend from the direction to `p2` toward the direction to `p3` — a one-segment look-ahead |

The wheel speeds start at zero for every seed. The **initial interval** is not flat: it is proportional to the segment's length (PQ-B's Q2, finding F9), so every segment is allotted time in proportion to the distance it has to cover. The total is the same as the old flat guess - `0.1 s` per interval over the whole path - and each segment's share is spread over its own samples, so `dt_i = (0.1 * total_intervals) * (L_i / L) / s_i`, clipped to `[0.01, 1.0] s` (`_build_initial_guess`). The measured trade-off is the one PQ-B measured: it converges **more slowly** than the flat guess (40 iterations against 25) and finds a **faster trajectory** on 18 of 32 missions, so it changes which local optimum the solver reaches, not how fast it gets there.

**Invariant:** the samples that sit on a waypoint are exactly on the waypoint. The seed never moves a constrained waypoint. `tests/test_optimizer.py` asserts this for all five styles, and also asserts that all five styles are mutually distinct.

### 4.2 The five seeds

| Seed | Mechanism | Intent | Weakness |
|---|---|---|---|
| `baseline` | the shared geometry, unchanged: zero wheel speeds | the simplest feasible-ish starting point; the historical behaviour of the solver and the reference the others are measured against | the guess is kinematically poor - a straight line between two waypoints with different headings is not a path a differential-drive robot can follow, so IPOPT starts far from the feasible region |
| `rolling` | `_set_rolling_speeds` back-differences the guessed positions into `v` and `omega`, converts them to wheel speeds, clips them to `±max_linear_speed` and forces zero at the ends and at every stop | IPOPT starts *moving*: a seed with zero wheel speeds sits where the integration constraints and the "start and end at rest" equalities pull in opposite directions | the speeds follow the straight-line geometry, so they are wrong wherever the path must curve - for a sharp turn the guess says "drive fast straight ahead" |
| `bow_left` / `bow_right` | `_offset_segment_interiors` pushes the interior samples sideways along the segment's normal by `0.5 * L * tan(|dtheta| / 4)` (the circular-arc sagitta, capped at `0.35 * L`) times a parabolic bump that is zero at both waypoints | "so IPOPT starts in an arc-shaped basin": a corner-cutting straight line is not a path the robot can hold | the bow is one parabolic bump, not an arc; it cannot follow a path that curves twice, and the cap binds on a sharp turn |
| `turn_first` | `_rotate_before_driving` turns in place on the segments whose heading change cannot be driven as an arc at the robot's minimum radius (the threshold is `2 * asin(L / 2 r_min)` for a short segment, 0.4 rad otherwise), then drives | for a sharp turn a differential-drive robot has two options - arc through the corner, or turn on the spot and then drive - and the seed picks one explicitly | it commits to a topology with no evidence about which one is right, and with a single sample in the segment it keeps the baseline geometry |

The seeds are cheap: every one of them is one pass over the samples, so the whole set costs nothing.

**Known result:** the seeds are *not* uniformly worse than the baseline. On the complex mission under equal-duration timing all four non-baseline seeds converge to **slower and wider** paths (8.12-8.66 s, 1.52-1.75 cm envelope, 0.31-0.45 cm tracking) than the baseline (6.068 s / 1.19 cm / 0.30 cm) - and three of the four pass, `bow_left` being the one that fails, on the motor force (0.007 N). Under per-segment timing on `example_mission_first4`, `bow_right` is the **selected** candidate at 1.663 s, ahead of the baseline at 1.833 s, while `bow_left` and `turn_first` fail there on the motor force (0.071 N and 0.008 N). See [Section 9](#9-measured-heuristic-comparison).

### 4.3 Seed summary

| Seed | Mechanism | Code | Cost | What it biases | Measured outcome (2026-10-20) |
|---|---|---|---|---|---|
| `baseline` | straight line + heading blend, zero speed | `overboard/core/optimizer.py` | ~0 | nothing | Selected on 2 of the 6 missions; **fails** the tracking gate on the complex mission under per-segment timing and the tracking gate on events/equal-duration |
| `rolling` | + back-differenced wheel speeds | `overboard/core/optimizer.py` | ~0 | "start moving" | Selected on the events mission (4.415 s) and on first4 under the full sweep (1.796 s) — the only mission where equal-duration timing produces no passing candidate at all |
| `bow_left` | + parabolic sideways bump, +side | `overboard/core/optimizer.py` | ~0 | arc-shaped basin | Indistinguishable from the others on the straight path; fails the force gate on first4/per-segment by 0.071 N |
| `bow_right` | + parabolic sideways bump, −side | `overboard/core/optimizer.py` | ~0 | opposite arc basin | The fastest *passing* per-segment candidate on first4 (1.663 s); fails the tracking gate on the complex mission |
| `turn_first` | turn in place, then drive | `overboard/core/optimizer.py` | ~0 | turn-on-the-spot topology | Fails the force gate on first4/per-segment by 0.008 N; fastest non-baseline seed on the complex mission under equal-duration timing (8.118 s) but still fails |

**The pattern:** a non-baseline seed wins by *escaping a gate*, not by finding a better optimum. On first4/per-segment the baseline passes at 1.833 s but `bow_right` finds a faster solution at 1.663 s, while `bow_left` and `turn_first` miss the force gate by 0.071 N and 0.008 N. On events/per-segment every seed now passes within a millisecond of the others. The seeds are insurance against one seed's convergence, not a search.

## 5. The timing models

A second, orthogonal axis multiplies the five seeds into the candidate set: the **timing model**.

### 5.1 Equal-duration (`per_segment_dt = False`)

One scalar `dt` for the whole path (`overboard/core/optimizer.py`). Total time is `dt * (N - 1)`.

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

**The third term is a soft force penalty.** `--force-weight` prices the wheel forces instead of only constraining them: the hard per-wheel limits of [Section 10](#10-the-gate-chain) still apply, and on top of them the objective charges the time integral of the squared force utilisation against both the stall force and the per-wheel traction limit. A plan that runs near the traction limit is therefore paid for *before* the gate rejects it, which is the reachability-driven optimisation KI-020 asks for in its cheapest possible form - the force is a term in the objective, not a constraint on the path. The weight is **0 by default**, so the objective is duration plus jerk unless the operator asks for more; the flag is simple-solver only.

### 5.4 Why the two models are not peers

`overboard/main.py` states it, and the measured behaviour confirms it: on an uneven 1.0 m + 0.2 m path, equal-duration timing forces the short 0.2 m segment to take the same time as the long one. That is a large artificial penalty, and it is why the equal-duration candidate is often much slower than the per-segment one on paths with unequal segments.

The consequence for the candidate set: **the stock set mixes the two modes**, because they answer different questions. Equal-duration is the safe, historically validated option; per-segment is the faster, more general one that sometimes fails the tracking gate. Validation picks between them. The default (unified) path is per-segment only - it is the general model, and the equal-duration branch exists there for the measured comparison, not as a candidate.

### 5.5 Where the time allocation may change (KI-024)

The per-segment model gives every waypoint segment its own interval, so the time scale can change anywhere. That is only legitimate where the state at the waypoint is fully determined, and the rule is now written down in one place (`TrajectoryOptimizer.merge_conditions`, `overboard/core/optimizer.py`):

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

There are two candidate sets, and the default is the first one.

### 6.1 The frontier triad (default)

```python
# overboard/main.py, with --unified
specs_to_solve = [
    {"accuracy_weight": 0.001, "guess_style": "frontier_sprint",     "per_segment_dt": True, "cost_metric": "discretization", "epigraph_constraints": True},
    {"accuracy_weight": 0.002, "guess_style": "frontier_sweeper",    "per_segment_dt": True, "cost_metric": "discretization", "epigraph_constraints": True},
    {"accuracy_weight": 0.001, "guess_style": "frontier_exploratory","per_segment_dt": True, "cost_metric": "discretization", "epigraph_constraints": True},
]
```

| Candidate | Seed | Weight | Timing | Metric |
|---|---|---:|---|---|
| 1 | `frontier_sprint` | 0.001 | per-segment | discretization |
| 2 | `frontier_sweeper` | 0.002 | per-segment | discretization |
| 3 | `frontier_exploratory` | 0.001 | per-segment | discretization |

`--candidates N` truncates the list when `N < 3`; an explicit `--accuracy-weight` overrides the three weights. The sample count defaults to the engine's invariant mesh (10-18 samples per segment, a 50 ms tick span) unless `--samples` says otherwise.

### 6.2 The stock set (retired)

`_candidate_specs` (`overboard/main.py`) builds the stock set: a seed and a timing model per candidate, cycling through the two timing models and the five seeds.

```python
# overboard/main.py
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

With the default `--candidates 2`, the set is exactly `baseline/equal-duration` and `baseline/per-segment`. This is the set the measurements in [Section 9](#9-measured-heuristic-comparison) describe.

`--uniform-dt` and `--per-segment-dt` fix `modes` to a single entry. The seed still rotates, so `--uniform-dt --candidates 5` solves all five seeds under equal-duration timing - the cleanest way to compare the seeds against each other. Both flags turn the unified engine off.

### 6.3 The sweep grid (deprecated)

`--sweep` adds the timing-mode × accuracy-weight grid:

```python
SWEEP_WEIGHTS = (0.0, 0.001, 0.002, 0.005, 0.01)    # overboard/main.py
```

The grid uses the `baseline` seed only, and it **adds** candidates - it never removes the default ones, so a sweep can only ever offer more options. A combination already in the set is not solved twice.

The rationale for the weight ladder is in the comment in `overboard/main.py`: the weight trades time against wheel-jerk discipline, and the best value depends on the path. A mission with long straight segments wants a low weight; an uneven path wants a little more. The flag is **deprecated**: it turns the unified engine off, and the invariant mesh in the default path removed the need for a manual grid.

### 6.4 The archived ladder

The Multi-Verse accuracy-weight ladder was removed with the pipeline. It traded
speed for a bound that could get worse (longer trajectories accumulate more
uncertainty); on the complex mission the slow 17.1 s candidate diverged while
the 6.07 s one stayed bounded. Do not reintroduce it.

### 6.5 Presets

`PRESETS` (`overboard/main.py`) fills in options the user did not set. Explicit flags always win (`_apply_preset`).

| Preset | samples | accuracy weight | candidates | validate | export |
|---|---|---|---|---|---|
| `fast` | 8 | 0.0 | 1 | no | none |
| `balanced` | 10 | 0.0 | 2 | no | none |
| `safe` | 14 | 0.005 | 3 | **yes** | controller |

`safe` is the only preset that turns on the gate. `fast` and `balanced` both leave `--validate` off, which means they solve only candidate 1.

## 7. Scoring and selection

### 7.1 The per-candidate score

After the Immrax run, `overboard/main.py` computes the score:

```python
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
attempt_record["ok"] = (converged and not failed_threshold and immrax_report['passed'])
attempt_record["violation"] = max_vio_val if failed_threshold else (tracking if tracking > 0.005 else 0)
```

The CPU audit (`overboard/verification/validator.py`) runs on **every** candidate, not only on the winner, so a candidate that clears Immrax but breaks the force, slip or integration limits cannot block a later candidate that passes everything.

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
def _select_attempt(attempts, prefer_duration=False):
    candidates = [a for a in attempts if a.get("converged", False)]   # KI-001
    if not candidates:
        return None, False
    passing = [a for a in candidates if a["ok"]]
    if passing:
        if prefer_duration:
            return min(passing, key=lambda a: (a.get("duration", a["cost"]),
                                                a.get("tube", float("inf")))), True
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
2. Among the passing candidates, the rule depends on the path:
   - **the only path** sets `prefer_duration`: **the shortest duration wins**, and the envelope breaks ties. This is the *gate-certified minimum time* the unified engine was built for.
   - the **5-part fitness score** ([7.2](#72-the-5-part-path-fitness-score)) is reported for every candidate, so the rule can be re-run offline, but it does not decide.
   - if no fitness was scored at all, the fallback is the smallest envelope, then the shortest duration.
3. If none pass, the converged candidate with the **smallest quantified violation** wins, and `passed_any` is `False` - so the export is still blocked. The fallback exists to pick the most useful candidate to *report*, not to ship.

**What the rule used to be, and why it changed.** The rule was `min(tube, then duration)`, and PQ-B's Q10 measured it: five candidate rules picked the same trajectory on every mission, because the envelope is almost a proxy for the duration (`corr(duration, envelope) = +0.97`, and the tube grows with the clock through the backlash term). The fitness score replaces it because it asks a different question - not "which plan is narrowest" but "which plan's defects are worth the least time" - and the two can disagree: a plan with a wider tube at the stops but a shorter run can win. The unified path then went one step further and asked for the shortest plan that passes at all, because its candidates are all built to be fast.

`tests/test_regressions.py` (`TestAttemptSelection`) guards the rule that a failed attempt is never preferred over a passing one.

### 7.4 The report

`_emit_validation_report` (`overboard/main.py`) prints one machine-readable line:

```text
##OVERBOARD-VALIDATION## {"selected": 1, "candidates": [ ... ]}
```

Each candidate carries `style`, `timing`, `duration_s`, `envelope_cm`, `tracking_cm`, `ok`, `converged`, `reasons`, and a per-segment breakdown, plus the fitness score and its parts: `fitness_s`, `stop_spread_cm`, `slip_risk`, `rms_force_n` and `rms_force_util`. The browser studio parses this line to show which part of the path failed (`viz/studio.js`). Two invariants: the JSON must stay valid (no `Infinity` — `_finite` maps inf/nan to `null`), and nothing is written to disk on a failed validation, so the report has to go to stdout.

## 8. The archived Multi-Verse path

The Multi-Verse pipeline was removed from the tree (git tag `archive/legacy-2026-10`) and has no CLI
entry point or candidate path. Its five phases (bootstrap, kinematic profile,
critic, JAX refinement, polish and refine probe) are preserved there with the
archived `archive/legacy-2026-10:overboard/legacy/path_planning.py` and `archive/legacy-2026-10:overboard/legacy/jax_optimizer.py`; nothing in the shipped
planner imports them. The measured lesson is in [20.1](STRATEGY_LEDGER.md#201-the-jax-refinement)
and [20.2](STRATEGY_LEDGER.md#202-the-refine-probe): the pipeline was a seed provider, not a
better planner.

## 9. Measured heuristic comparison

**Measured 2026-10-20**, against the tree with the arc-length sample layout, the self-consistent collocation on by default (KI-026, KI-051, KI-052) and the consolidated firmware (KI-028) in place. 60 candidate solves - 6 missions x 2 timing models x 5 seeds - plus the timing-mode x accuracy-weight grid. The raw logs are in `archive/research-2026-10:docs/measurements/2026-10-20-variable-samples/`, and the exact commands are in [Appendix A](#appendix-a--reproducing-every-table). The provenance (source hashes, environment, test state) is in [S0.4](#04-measurement-conventions).

**These numbers describe the stock planner** - the retired candidate set, measured before the 3-Pillar unified engine became the default. Re-measuring the matrix on the unified path is queued as [Q13](RESEARCH_AGENDA.md#q13--should-the-measured-matrix-be-re-measured-on-the-unified-path), and it waits for the optimizer to settle first. They are still the right record for the seeds, the timing models and the gates, and the unified engine's own numbers are in [19.6](STRATEGY_LEDGER.md#196-the-3-pillar-unified-trajectory-engine---unified-the-default-path) and in `docs/OPTIMAL_CONTROL_DERIVATION.md`. They are **not comparable** with the 2026-10-20-post-ki028 evidence, which was taken with the trapezoidal collocation and a uniform sample count. The changes and what they cost are in [5.7](#57-the-collocation-where-a-sample-is-placed), [5.8](#58-the-sample-layout-how-many-samples-a-segment-gets) and [7.2](#72-the-5-part-path-fitness-score); the measurement that produced this section is in `archive/research-2026-10:docs/measurements/2026-10-20-variable-samples/`, and the runner takes the tree's fingerprint before and after the run, so a run whose tree changed underneath it is refused rather than quoted.

**The "Selected" column is the fitness score's choice**, not the envelope's ([7.3](#73-the-selection-rule)): among the candidates that pass, the one with the lowest 5-part fitness score wins, with the envelope and the duration as tie-breakers. That is the **stock** set's rule - this matrix is the stock set's measurement (see the note above).

**The tables in this section are generated**, not typed in: `archive/research-2026-10:docs/research/rq_a/sync_tables.py` rewrites them from the measurement JSON and `--check` verifies that every number still matches.

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

The rule is `min(tube, then duration)` among the passing candidates (`overboard/main.py`).

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

The stock candidate set plus the timing-mode x accuracy-weight grid, both timing modes: **13 candidates per mission, 78 in total**. This was `--no-unified --candidates 5 --sweep`, the most thorough configuration the stock path offered; the path and the sweep are retired, and the numbers are the record.

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
 G0  solver convergence            overboard/main.py  (stats["converged"])
      |
 G1  CPU audit                     overboard/verification/validator.py  validate_samples
      |  - forward integration @ 1 ms RK4        forward_integrate
      |  - per-wheel force / slip audit          audit_constraints
      |  - metrics                               compute_metrics
      |
 G2  final waypoint check          overboard/verification/validator.py  check_final_waypoint
      |  2 cm position, 0.02 rad heading
      |
 G3  Immrax reachability           overboard/verification/immrax_validator.py  validate_trajectory
      |  - motor force, per-wheel slip, wheel speed
      |  - closed-loop tracking error <= 5 mm
      |  - reachability envelope
      |
 G4  local envelope threshold      overboard/main.py  _segment_limit
      |  2 cm transit, 0.5 cm precision_mode, or an explicit error_threshold
      |
 G5  export gate                   overboard/main.py  _export_block_reason
      |  - not converged  -> block
      |  - --validate and not passed -> block
      |  - blocked -> nothing is written, exit code 1
```

### 10.1 G0 — Solver convergence

`stats["converged"]` is set by `TrajectoryOptimizer.solve`. On failure the solver still returns a best-effort sample list from `opti.debug`, but the flag is `False`, and a `False` flag makes the candidate ineligible for selection (KI-001) and blocks the export.

### 10.2 G1 — The CPU audit

`validate_samples` (`overboard/verification/validator.py`) runs three things:

- **`forward_integrate`** (`overboard/verification/validator.py`) re-integrates the planned wheel speeds with RK4 at 1 ms and compares against the planned samples. It rejects empty and non-monotone time vectors and returns `inf` errors instead of crashing.
- **`audit_constraints`** (`overboard/verification/validator.py`) checks, per sample: the per-wheel motor force against the velocity-dependent limit, the per-wheel slip against the per-wheel normal force, the legacy total traction, and the wheel speed.
- **`compute_metrics`** (`overboard/verification/validator.py`) collects duration, path length, speeds, accelerations, jerk (computed on midpoint spacing, so it is layout-independent — KI-022), tortuosity, chattering and force utilisation.

The gate (`overboard/verification/validator.py`):

```python
ok = (errors["max_pos_error_m"] < 0.01
      and errors["final_pos_error_m"] < 0.01
      and audit["num_violating_samples"] == 0
      and audit["num_slip_points"] == 0
      and audit["speed_violation_m_s"] < 1e-6)
```

### 10.3 G2 — The final waypoint check

`check_final_waypoint` (`overboard/verification/validator.py`) is the second line of defence: the exported trajectory must actually reach its last waypoint, within 2 cm of position and 2 cm of heading (`overboard/verification/validator.py`). This one runs **even without `--validate`** (`overboard/main.py` calls it on the selected candidate before writing).

### 10.4 G3 — Immrax reachability

`ImmraxValidator.validate_trajectory` (`overboard/verification/immrax_validator.py`) simulates the closed loop with interval arithmetic. Details in [Section 12](#12-controller-and-plant-parity).

**The tube is the width of a *model*, so the model is part of the number.** The reachability is computed by one of two **propagations** (`overboard/verification/immrax_validator.py`, `PROPAGATION`), and every candidate in the `##OVERBOARD-VALIDATION##` line names the one that produced its envelope:

| propagation | what it embeds | what the tube bounds |
|---|---|---|
| `plant` (the default) | the plant; the controller is evaluated *outside* the propagation, at the tube's midpoint, with the command held for the tick | the reachable set of an **open-loop** plant driven by a pre-recorded command — the feedback's contraction is not in the tube at all |
| `tick` | one 20 ms tick of the **closed loop** — controller, pose estimate and delayed command — as a discrete map, propagated once per tick | the reachable set of the loop the robot actually runs; the map's Jacobian carries `dF/dx = dPhi/dx + (dPhi/du)(dN/dx)`, and the Ramsete feedback contracts the tube |

The tick propagation needs a Jacobian inclusion: with the natural extension the closed-loop map explodes (627 mm on a 1 m straight). It costs 2-3x (`jac`) to 7-10x (`mjac`) the validation time, which is why it is opt-in (`main.py --propagation tick`). It is the lever on the paths where the plant propagation *diverges* rather than merely widens — the S-curve, the stops and the complex mission, where the mixed-Jacobian gives 1.88, 1.30 and 1.83 cm against 5.36, 5.31 and 5.07 cm, and every tube still contains the sampled reachable set. With the digital twin it is **wider**, and the reason is the twin's plant state, not the propagation (KI-049, KI-050).

Two rules come with it. **The controller must not unpack the pose**: `x, y, theta = current_pose` is the JAX `unstack` primitive, Immrax's natural inclusion has no rule for it, and the tick map dies with "unstack not in inclusion_registry" — so the pose is read by indexing, which compiles to `slice`+`squeeze` and gives the same number. And **an envelope is never read without its model**: the report, the studio's validation note and the studio's chip tooltip all name the propagation, the same rule as `envelope_truncated`.

The physical pass condition (`overboard/verification/immrax_validator.py`):

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
| Transit (default) | 2 cm | `overboard/main.py` (`_segment_limit`) |
| `precision_mode: true` | 0.5 cm | `overboard/main.py` (`_load_waypoints`) |
| Explicit `error_threshold` | the user's value | `overboard/main.py` (`_load_waypoints`) |

**The limit belongs to the way *into* the waypoint.** A waypoint's limit governs the segment that ends at it - the approach - and the segment that starts there falls back to it when the next waypoint declares none. So a waypoint marked as a precision one is held to its tight limit on the way the robot arrives, which is what the operator asked for, and the final waypoint's limit governs the last segment instead of being dropped (KI-034).

A sample whose envelope is marked `unknown` also fails — that is how a diverged tube is caught (see below).

### 10.6 G5 — The export gate

```python
# overboard/main.py, _export_block_reason
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
| 1 | Motor force violation | < 1e-3 N | `overboard/verification/immrax_validator.py` | worst-case per-wheel force over the motor limit |
| 2 | Per-wheel slip (left, right) | < 1e-3 N | `overboard/verification/immrax_validator.py` | per-wheel traction violation over the limit |
| 3 | Closed-loop tracking error | ≤ 5 mm | `overboard/verification/immrax_validator.py`, checked at `:530` | max distance between the true pose and the reference. **Position only** — the heading error is reported separately (`max_heading_error_rad`) but not gated, because the heading behaviour is not yet calibrated (KI-020, RQ-A OQ4) |
| 4 | Wheel-speed violation | ≤ 1e-6 m/s | `overboard/verification/immrax_validator.py` | commanded speed over the limit |
| 5 | Envelope divergence radius | > 50 mm ⇒ diverged | `overboard/verification/immrax_validator.py` | the interval tube has grown past the point where it means anything; the sample is marked `unknown` |
| 6 | Transit envelope radius | ≤ 2 cm | `overboard/main.py` | per-sample reachability half-diagonal |
| 7 | `precision_mode` envelope radius | ≤ 0.5 cm | `overboard/main.py` (`_load_waypoints`) | tight waypoints, on the way in |
| 8 | Explicit `error_threshold` | user value | `overboard/main.py` (`_load_waypoints`) | per-waypoint override, on the way in |
| 9 | Forward-integration position error | < 1 cm | `overboard/verification/validator.py` | RK4 re-integration vs. the plan |
| 10 | Forward-integration final error | < 1 cm | `overboard/verification/validator.py` | same, at the end |
| 11 | Violating samples | 0 | `overboard/verification/validator.py` | motor/traction limit hits |
| 12 | Slip points | 0 | `overboard/verification/validator.py` | per-wheel slip events |
| 13 | Speed violation | < 1e-6 m/s | `overboard/verification/validator.py` | wheel speed over the limit |
| 14 | Final waypoint position | ≤ 2 cm | `overboard/verification/validator.py` | the plan must arrive |
| 15 | Final waypoint heading | ≤ 2 cm (0.02 rad) | `overboard/verification/validator.py` | the plan must arrive pointing the right way |
| 16 | IPOPT iterations | ≤ 5000 | `overboard/core/optimizer.py` solver options | `max_iter`; the **warm** pass in fast-polish is budgeted to 200, and the cold pass that follows it is not |
| 17 | IPOPT tolerance | 1e-2 | solver options | `tol` and `constr_viol_tol`, deliberately loose for interactive use |

## 11. The uncertainty model

`RobotConfig.uncertainties` (`overboard/models/robot_model.py`) holds seven intervals. They are what Immrax propagates.

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

1. **Both bounds matter.** Immrax takes an `Interval(w_lower, w_upper)`; passing a plain array makes Immrax treat the values as constants, and only the lower bound would then have any effect (KI-003). `tests/test_regressions.py` (`TestReachabilityEnvelope`) guards this.
2. **The declared ranges are test assumptions.** They were narrowed to make the example missions behave, not measured from the robot. A narrow `cof` interval makes the envelope small; it does not make the robot accurate.
3. **The optimizer and the validator share one force model.** `wheel_normal_forces` (`overboard/models/robot_model.py`) is the single implementation of the weight transfer — lateral only, no front/rear split (a two-wheel robot has none) — and it is called from the optimizer, the CPU audit, the JAX model and the Immrax check (KI-010). The optimizer additionally uses the declared worst case for the coefficient of friction, the torque headroom, the wheel radius, the track width and the inertia (`overboard/models/robot_model.py` `get_robust_force_params`).

## 12. Controller and plant parity

The value of the reachability check is that it simulates roughly what the robot will do. That requires the controller, the reference, the timing and the plant to match the deployed system.

| Aspect | Validator | Deployed runtime | Parity |
|---|---|---|---|
| Control period | 20 ms (`overboard/verification/immrax_validator.py`) | 20 ms `drive()` tick | **yes** |
| Plant integration | 4 × 5 ms substeps (`overboard/verification/immrax_validator.py`) | — | mirrors the firmware's own substepping |
| Reference | linear interpolation of `x, y, heading, v, omega` plus the precomputed Ramsete gain `sqrt(omega^2 + b*v^2)` (`overboard/verification/immrax_validator.py`) | `ramsete.py` | **yes** |
| Heading unwrapping | `unroll_trajectory_headings` (`overboard/verification/immrax_validator.py`) | — | prevents the ±π wrap |
| Controller | `RamseteController` (`overboard/verification/immrax_validator.py`), `b = 2.0`, `zeta = 0.7` | `ramsete.py` | **yes** |
| Command quantisation | whole mm/s and deg/s when `quantize=True` (`overboard/verification/immrax_validator.py`) | `DriveBase` rounding | **yes** (KI-021) |
| Firmware's own copies | — | **one**: `ramsete.py` holds the odometry, the Ramsete maths, the limiter and the loop; `ramsete_fast.py` delegates to it and `ramsete_compact.py` delegates to `ramsete_compact_events.py` | **yes** - a second copy cannot drift (KI-028) |
| Traction limit in the limiter | per wheel, with the lateral weight transfer (`overboard/models/robot_model.py` `wheel_normal_forces`) | per wheel, same formula (`pybricks_code/ramsete.py`, `KinematicLimiter`) | **yes** (KI-010) |
| Plant, default | `RobotOpenLoop` (`overboard/verification/immrax_validator.py`): commanded wheel speed → motion. The gear backlash is a **position** uncertainty carried in the reachable set, not a velocity offset (KI-025) | - | partial |
| Plant, fitted | `DigitalTwinOpenLoop` (`overboard/verification/immrax_validator.py`): 5-state `[x, y, heading, vl_actual, vr_actual]` with a first-order lag, `tau >= 0.01 s` | — | partial |

**What the default plant does not model:** the DriveBase PID, motor electrical and mechanical time constants, encoder and gyro odometry error, wheel-slip dynamics, interaction forces, and battery sag beyond the declared torque-headroom interval.

**What the fitted plant adds** (`--sysid-fit`): the measured straight-line actuator gain interval, the first-order time constant, the command delay, the odometry distance and heading scale, the gyro bias, the measured acceleration limit, and the braking shortfall.

Two constraints on the interval mode: the natural inclusion does not support `abs`, `where`, `mod`, `sinc` or `atan2`, so the interval path uses an unrolled heading and a degree-6 Chebyshev polynomial instead of the exact `sinc` (KI-014); and the conservative `interval` feedback mode diverges on long fast paths, so the default `nominal` mode feeds the controller the midpoint of the interval state.

## 13. The system identification loop

```text
  pybricks_code/sysid_digital_twin.py        (on the hub, Pybricks)
        |  console: TEST_START / META, / DATA / TEST_END,
        |           POSE,x,y,heading_deg and EVENT,name
        v
  overboard.deploy.hub_deploy --out session.csv | overboard.sysid.collect  (one command)
        v
  overboard.sysid.fit --poses poses.csv --profile examples/drivebase_profile.example.json
        |  -> sysid_fit.json
        v
  overboard.main --sysid-fit sysid_fit.json --validate
        |  switches Immrax to the fitted five-state plant
        v
  sysid_compare.py output.traj poses.csv --run-id <id>
        reports planned-vs-measured error and whether the measured pose
        is inside the trajectory's reachability envelope
```

### 13.1 What the collector measures

`pybricks_code/sysid_digital_twin.py` runs 16 tests: raw encoder/IMU logging at 100 Hz, gyro and accelerometer bias, straight and turn step responses (50/100/200/400/600 mm/s and 30/60/120/240 deg/s), curvature steps, a reversal test for backlash, two chirps (0.2–3 Hz) for held-out validation, an odometry test, distance and heading scale calibration, a multi-stop pose test, braking-from-speed, and a wall-stall load test.

### 13.2 What the fitter produces

`overboard/sysid/fit.py` produces `sysid_fit.json` with:

- `actuator.straight` / `actuator.turn` — gain, time constant and delay from a grid search over delay ∈ [0, 0.12] s and tau ∈ [0.01, 0.50] s (`overboard/sysid/fit.py`), plus a 95th-percentile acceleration limit;
- `sensors` — gyro z bias and standard deviation, accelerometer biases;
- `odometry` — distance scale, heading scale, braking slip ratio;
- `backlash` — the reversal delay;
- `validation.chirp_*_nrmse` — the held-out chirp error.

The **chirp tests are held out of the fit** and are the only genuinely independent accuracy check in the whole system.

### 13.3 What the fit does *not* yet do

- **The poses the collector reports are the hub's own odometry.** `overboard/sysid/collect.py` says so explicitly: the distance and heading scale have no ground truth and stay at 1.0 unless the operator replaces `poses.csv` with tape measurements and re-runs the fitter. The actuator part of the fit (gain, tau, delay, backlash) needs no ground truth and is calibrated either way.
- **Braking slip is applied by the fitted plant.** While braking, the wheels turn further than the robot travels, so the plant scales the robot's ground truth down with a smooth factor of the commanded speed. The default kinematic plant is unchanged, because it has no fit to read a shortfall from. The grip belongs to the five-state plant, so a fit that carries the slip but no actuator time constant keeps the kinematic plant and the slip is loaded and not applied.
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

**Coverage gap.** `tests/test_optimizer.py` contains a `TestBenchmarkDryRun` class (benchmark dry-run behaviour) that is **not registered in `run_tests.py`**, so it never runs. It is not in the `run_tests.py` class list (`run_tests.py`).

---

# Part III–IV — the research record, in their own documents

The two parts that are not needed to use the planner have their own files now:

- **[`STRATEGY_LEDGER.md`](STRATEGY_LEDGER.md)** — Part III, the failed-strategy ledger: every
  strategy that was tried and did not work, with the measurement that retired it and what would
  revive it. Do not re-litigate one of these without new evidence.
- **[`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md)** — Part IV, the open questions, ordered so the
  cheapest decisive experiment comes first.

The section numbers are unchanged, so a citation like "§19.6" still points at the same text — in
the other file now. The three files share one numbering, so reading them together is reading one
document.

# Appendix A — Reproducing every table

All commands run from the repository root.

## A.1 The heuristic comparison matrix

```bash
# Equal-duration timing, all five seeds
python -m overboard.main -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --uniform-dt

# Per-segment timing, all five seeds
python -m overboard.main -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --per-segment-dt
```

Missions: `example_straight`, `example_s_curve`, `example_mission_first4`, `example_complex_mission`, `example_with_stops`, `example_with_events`.

The machine-readable result is the `##OVERBOARD-VALIDATION##` line in the stdout; the human table has the wall times and the selected candidate.

## A.2 The timing × accuracy-weight grid

```bash
python -m overboard.main -c examples/fll_choreo.json -w examples/<mission>.json \
  -o <tmp>.traj --validate --candidates 5 --sweep
```

## A.3 The seed comparison in one line

```bash
python -m overboard.main --uniform-dt --validate -c examples/fll_choreo.json \
  -w examples/example_mission_first4.json -o first4.traj --candidates 5
```

## A.4 From Python, without the CLI

```python
from overboard.core.optimizer import TrajectoryOptimizer
from overboard.models.robot_model import RobotConfig

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
python -m overboard.main -c examples/fll_choreo.json -w examples/example_complex_mission.json \
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

# Appendix B — Where things live

The modules this document names, and what each one holds. The code is the
authority on where a symbol lives; this map is here so a reader does not have
to go looking.

| Module | Holds | The symbols this document names |
|---|---|---|
| `overboard/main.py` | the CLI, the candidate set, the scoring, the selection, the export gate | `main`, `_candidate_specs`, `_select_attempt`, `_choose_attempt`, `_record_candidate`, `_score_attempt`, `_run_candidate_rounds`, `_export_block_reason`, `_emit_validation_report`, `_immrax_failure_reasons`, `_cpu_failure_reasons`, `_segment_limit`, `_segment_report`, `_load_waypoints`, `_apply_preset`, `_rescue_specs`, `PRESETS`, `SWEEP_WEIGHTS`, `VALIDATION_MARKER`, `HAS_JAX` |
| `overboard/core/optimizer.py` | the CasADi/IPOPT optimizer and the five stock seeds | `TrajectoryOptimizer`, `solve`, `_solve_once`, `_compute_cost`, `_build_initial_guess`, `_guess_point`, `_offset_segment_interiors`, `_rotate_before_driving`, `_set_rolling_speeds`, `_sample_times`, `format_output`, `guess_from_samples`, `get_segment_sample_counts`, `get_adaptive_mesh_sample_counts`, `merge_conditions`, `GUESS_STYLES`, `CASADI_IPOPT_LOCK` |
| `overboard/core/frontier_seed.py` | the analytical frontier seeds | `generate_frontier_seed`, `generate_canonical_seeds`, `compute_tangent_headings` |
| `overboard/core/unified_planner.py` | the 3-Pillar unified engine | `UnifiedPathEngine`, `compute_invariant_mesh_samples`, `generate_seed`, `solve_candidate`, `plan` |
| `overboard/core/unified_planner.py` | the 3-Pillar engine: the candidate set, the seeds and the optional gradient polish | `candidate_specs`, `gradient_polish` |
| `overboard/verification/validator.py` | the CPU audit | `forward_integrate`, `audit_constraints`, `compute_metrics`, `compute_path_fitness`, `check_final_waypoint`, `validate_samples`, `validate_trajectory` |
| `overboard/verification/immrax_validator.py` | the closed-loop interval reachability check | `ImmraxValidator`, `validate_trajectory`, `compute_reachability`, `_interpolated_reference`, `RamseteController`, `RobotOpenLoop`, `DigitalTwinOpenLoop`, `ClosedLoopTickMap`, `PROPAGATION`, `PROPAGATIONS`, `CONTROL_DT`, `PLANT_SUBSTEPS` |
| `overboard/verification/reachability_engine.py` | the compiled `jax.lax.scan` propagation | `CompiledReachabilityEngine`, `unroll_trajectory_headings` |
| `overboard/models/robot_model.py` | the robot configuration and the shared force model | `RobotConfig`, `uncertainties`, `get_max_force_at_velocity`, `get_max_force_symbolic`, `get_robust_force_params`, `get_max_force_at_velocity_robust`, `get_max_force_symbolic_robust`, `max_linear_speed`, `no_load_wheel_speed_worst_case`, `apply_measured_motor`, `apply_sysid_fit`, `DifferentialDriveModel`, `get_dynamics`, `get_dynamics_symbolic`, `get_wheel_normal_forces`, `get_max_wheel_force_symbolic`, `get_traction_limit_symbolic`, `check_constraints` |
| `overboard/models/drivetrain_physics.py` | the one weight-transfer model | `wheel_normal_forces`, `motor_force_at_velocity`, `traction_limits`, `CG_HEIGHT_M` |
| `overboard/models/drivetrain.py` | the motor and drivetrain model | `brake_force`, `brake_deceleration` |
| `overboard/models/jax_robot_model.py` | the JAX-differentiable model | `JAXRobotConfig`, `get_max_force_at_velocity`, `max_linear_speed` |
| `overboard/models/jax_ramsete.py` | the JAX Ramsete controller | `JAXRamseteController`, `quantize_drive_command` |
| `overboard/sysid/digital_twin.py` | the optional fitted plant | `DigitalTwinConfig`, `ActuatorFit`, `OdometryFit`, `SlipFit`, `SensorFit`, `actuator_gain_range`, `brake_shortfall`, `load_sysid_fit` |
| `overboard/sysid/fit.py` | the host-side fitter | `parse_session`, `session_tests`, `write_sheet`, `sheet_to_poses`, `poses_from_hub_session`, `fit_session`, `repeat_statistics` |
| `overboard/sysid/collect.py` | the collector and its one-command loop | `main` |
| `overboard/sysid/compare.py` | plan against measured poses | `compare`, `load_trajectory_document` |
| `overboard/deploy/export.py` | the Choreo and Pybricks writers | `resample_to_fixed_dt`, `write_controller_file`, `write_python_file`, `write_compact_python_file`, `encode_compact_trajectory` |
| `overboard/deploy/hub_deploy.py` | the hub transport | `deploy`, `push_trajectory`, `HUB_CODE_DIR`, `HUB_MISSION_MODULE`, `HUB_RUNNER` |
| `overboard/deploy/hub_ble.py` | BLE discovery | `find_hub`, `list_hubs` |
| `overboard/gui_server.py` | the studio server | `Job`, `start_job`, `build_command`, `start_push`, `_validation_verdict` |
| `overboard/viz/` | the plotting and the studio assets | `plotter`, `fields`, `mat_import`, `overboard/viz/studio.html`, `overboard/viz/studio.js`, `overboard/viz/studio.css`, `overboard/viz/report.js` |
| (removed) | the Multi-Verse pipeline, the Reeds-Shepp bootstrap, the JAX refiner and the evolutionary research - git tag `archive/legacy-2026-10` | `multiverse_optimizer`, `jax_optimizer`, `path_planning` |
| `pybricks_code/` | the firmware: the controller, the runner, the collector, the robot config | `ramsete.py`, `ramsete_fast.py`, `ramsete_compact.py`, `ramsete_compact_events.py`, `hub_runner.py`, `sysid_digital_twin.py`, `robot_config.py` |

---

# Appendix C — Known issue cross-reference

| KI | Area | One line | Where it is documented |
|---|---|---|---|
| KI-001 | Solver / output | A failed solve was exported as a success | [22.4](STRATEGY_LEDGER.md#224-a-failed-solve-was-exported-as-a-success-ki-001-and---validate-could-not-block-output-ki-002) |
| KI-002 | Validation | `--validate` could not block output or the exit code | [22.4](STRATEGY_LEDGER.md#224-a-failed-solve-was-exported-as-a-success-ki-001-and---validate-could-not-block-output-ki-002) |
| KI-003 | Safety | Immrax ignored the upper uncertainty bounds | [22.3](STRATEGY_LEDGER.md#223-the-immrax-upper-uncertainty-bounds-were-ignored-ki-003) |
| KI-004 | Multi-Verse | The refinement had no effect on the final trajectory; measured as a seed provider it rescued nothing, and the pipeline is archived | [Section 8](#8-the-archived-multi-verse-path), [20.1](STRATEGY_LEDGER.md#201-the-jax-refinement), [20.2](STRATEGY_LEDGER.md#202-the-refine-probe), Q12 |
| KI-009 | Safety / config | The declared uncertainty ranges were ignored | [Section 11](#11-the-uncertainty-model) |
| KI-010 | Physics | The weight-transfer math was wrong and the tools disagreed | [22.5](STRATEGY_LEDGER.md#225-the-weight-transfer-math-was-wrong-and-the-tools-disagreed-ki-010) |
| KI-014 | Safety model | The JAX Ramsete used a poor large-angle approximation | [Section 12](#12-controller-and-plant-parity) |
| KI-015 | Benchmarks | ~~The random benchmark tools are broken~~ (fixed) | [21.3](STRATEGY_LEDGER.md#213-the-random-benchmark-mode), Q8 |
| KI-018 | Maintenance | ~~Dead code, duplicates, generated files~~ (fixed) | [21.4](STRATEGY_LEDGER.md#214-sysid_solverpy), [19.2](STRATEGY_LEDGER.md#192-multi-start-as-a-quality-claim) |
| KI-019 | Safety | The reachability tube lagged the reference by one sample | [22.2](STRATEGY_LEDGER.md#222-the-reachability-tube-lagged-the-reference-by-one-sample-ki-019) |
| KI-020 | Digital twin | The validator is not a full twin; reachability is not an objective | [Section 14](#14-what-each-layer-proves-and-does-not-prove), [21.2](STRATEGY_LEDGER.md#212-reachability-as-a-post-hoc-score), Q4, Q9 |
| KI-021 | Robot runtime | The DriveBase capped the acceleration the plan asked for | [Section 12](#12-controller-and-plant-parity) |
| KI-022 | Solver / timing | The per-segment timing model was a jerk-penalty loophole | [5.3](#53-the-jerk-discipline-ki-022), [22.1](STRATEGY_LEDGER.md#221-the-per-segment-timing-model-was-a-jerk-penalty-loophole-ki-022) |
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
| **Digital twin** | The fitted five-state plant, loaded with `--sysid-fit`. Strictly, a *partial* twin: it models the actuator and the odometry, not the motor, the PID or the slip. |
| **Model prediction** | Any number produced by the simulator. Not a hardware measurement. |
