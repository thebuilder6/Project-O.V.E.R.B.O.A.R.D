# The failed strategy ledger

**Status:** 2026-10-27, as measured on that tree. Part III of the heuristics document.
**Scope:** every planning strategy that was tried and did not work - what it was, what it cost, the
measurement that decided it, and what would revive it.
**Audience:** anyone about to try something that is in here.

> **Read this first.** This is the record of what did **not** work. A strategy in this ledger is not
> to be re-litigated without new evidence: the measurement that retired it is quoted with it. The
> current-state reference is [`HEURISTICS_AND_VERIFICATION.md`](HEURISTICS_AND_VERIFICATION.md) -
> what the planner does today, the gate chain, and the measured comparison of the candidate sets.
>
> **The section numbers are the ones the sections had in that document**, so a citation like "§19.6"
> still points at the same text - in this file now.

## Contents

- [16. How to read the ledger](#16-how-to-read-the-ledger)
- [17. Ledger: seed heuristics](#17-ledger-seed-heuristics)
  - [17.1 The four non-baseline seeds](#171-the-four-non-baseline-seeds)
  - [17.2 `turn_first`'s 0.4 rad threshold](#172-turn_firsts-04-rad-threshold)
  - [17.3 The bow amplitude](#173-the-bow-amplitude)
- [18. Ledger: timing models](#18-ledger-timing-models)
  - [18.1 Equal-duration timing treated as a peer of per-segment](#181-equal-duration-timing-treated-as-a-peer-of-per-segment)
  - [18.2 Per-segment timing at `accuracy_weight = 0`](#182-per-segment-timing-at-accuracy_weight--0)
  - [18.3 Tying the time base across a waypoint the robot drives through](#183-tying-the-time-base-across-a-waypoint-the-robot-drives-through)
  - [18.4 The removed accuracy-weight ladder](#184-the-removed-accuracy-weight-ladder)
  - [18.5 Global re-timing (the time floor) as a speed lever](#185-global-re-timing-the-time-floor-as-a-speed-lever)
- [19. Ledger: candidate set strategies](#19-ledger-candidate-set-strategies)
  - [19.1 The sweep grid](#191-the-sweep-grid)
  - [19.2 Multi-start as a quality claim](#192-multi-start-as-a-quality-claim)
  - [19.3 The presets](#193-the-presets)
  - [19.4 The 44-Gene Starting-Value Genome & Memetic Evolution](#194-the-44-gene-starting-value-genome--memetic-evolution)
  - [19.5 The Kinetic Frontier Analytical Seed Generator & The 5 Physical Principles](#195-the-kinetic-frontier-analytical-seed-generator--the-5-physical-principles)
  - [19.6 The 3-Pillar Unified Trajectory Engine (`--unified`, the default path)](#196-the-3-pillar-unified-trajectory-engine---unified-the-default-path)
  - [19.7 The Q17 topology and geometry candidates](#197-the-q17-topology-and-geometry-candidates)
- [20. Ledger: archived Multi-Verse internals](#20-ledger-archived-multi-verse-internals)
  - [20.1 The JAX refinement](#201-the-jax-refinement)
  - [20.2 The refine probe](#202-the-refine-probe)
- [21. Ledger: validation and tooling](#21-ledger-validation-and-tooling)
  - [21.1 `--workers` / `--no-parallel` as a quality lever](#211---workers----no-parallel-as-a-quality-lever)
  - [21.2 Reachability as a post-hoc score](#212-reachability-as-a-post-hoc-score)
  - [21.3 The random benchmark mode](#213-the-random-benchmark-mode)
  - [21.4 `sysid_solver.py`](#214-sysid_solverpy)
  - [21.5 Interaction forces](#215-interaction-forces)
  - [21.6 Braking slip](#216-braking-slip)
- [22. Ledger: silent and unlogged behaviour](#22-ledger-silent-and-unlogged-behaviour)
  - [22.1 The per-segment timing model was a jerk-penalty loophole (KI-022)](#221-the-per-segment-timing-model-was-a-jerk-penalty-loophole-ki-022)
  - [22.2 The reachability tube lagged the reference by one sample (KI-019)](#222-the-reachability-tube-lagged-the-reference-by-one-sample-ki-019)
  - [22.3 The Immrax upper uncertainty bounds were ignored (KI-003)](#223-the-immrax-upper-uncertainty-bounds-were-ignored-ki-003)
  - [22.4 A failed solve was exported as a success (KI-001) and `--validate` could not block output (KI-002)](#224-a-failed-solve-was-exported-as-a-success-ki-001-and---validate-could-not-block-output-ki-002)
  - [22.5 The weight-transfer math was wrong and the tools disagreed (KI-010)](#225-the-weight-transfer-math-was-wrong-and-the-tools-disagreed-ki-010)
- [23. Cross-cutting lessons](#23-cross-cutting-lessons)
- [24. Ledger: objective terms and residual pricing (F2)](#24-ledger-objective-terms-and-residual-pricing-f2)
  - [24.1 The static plan-feature surrogate for the tracking residual](#241-the-static-plan-feature-surrogate-for-the-tracking-residual)
  - [24.2 The gradient polish's guidance gradient (Q16's lens)](#242-the-gradient-polishs-guidance-gradient-q16s-lens)

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
| **Mechanism** | `overboard/core/optimizer.py` (`rolling`), `overboard/core/optimizer.py` (`bow_left` / `bow_right`), `overboard/core/optimizer.py` (`turn_first`) |
| **Why it was tried** | A better starting point should find a better local optimum: fewer iterations, a shorter trajectory, a tighter envelope |
| **Outcome** | **They never find a better optimum. They do often find one that passes.** As a *quality* lever they are dead: on the complex mission all four are 2–2.6 s slower and 0.3–0.5 cm wider than the baseline. As a *gate-escape* lever they are alive: the baseline failed 8 of the 12 (mission, timing model) cells and a non-baseline seed rescued 2 of them, and on the events mission they are the only candidates that pass at the default weight |
| **Evidence** | Complex mission, equal-duration: the baseline passes at **6.068 s / 1.19 cm / 0.30 cm**; `rolling` 8.377 s / 1.68 cm / 0.45 cm, `bow_left` 8.634 s / 1.75 cm / 0.41 cm, `bow_right` 8.662 s / 1.71 cm / 0.41 cm, `turn_first` 8.118 s / 1.52 cm / 0.31 cm — three pass and `bow_left` fails on the motor force (0.007 N). First4, per-segment: the baseline is not the fastest — `bow_right` is, at **1.663 s** against the baseline's 1.833 s, and it is the selected candidate, while `bow_left` and `turn_first` fail there on the motor force (0.071 N and 0.008 N). Events: under per-segment timing all five seeds pass at 4.415–4.416 s, while under equal-duration timing all five fail (0.65–0.68 cm of tracking error). See [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) |
| **Status** | **partial** — dead as a quality lever, alive as a gate-escape lever |
| **What would revive it as a quality lever** | A seed that changes the *geometry* of the path, not just its sampling. The current seeds perturb a straight line; the optimizer then walks the line back to the straight line, because the objective has no term that prefers a curve ([Section 2](HEURISTICS_AND_VERIFICATION.md#2-what-is-not-in-the-tree)). Until then, expect them to keep buying gate escapes rather than speed |

### 17.2 `turn_first`'s 0.4 rad threshold

| Field | Value |
|---|---|
| **Strategy** | Switch to a turn-on-the-spot topology only when the heading change is large enough |
| **Mechanism** | `overboard/core/optimizer.py`, `abs(delta) < 0.4` skips the segment |
| **Outcome** | The threshold is a hard-coded constant with no derivation. A 23° turn gets the turn-on-the-spot topology; a 22° turn does not. There is no measurement behind the number |
| **Status** | dormant |
| **What would revive it** | Derive the threshold from the robot: the smallest turn that cannot be driven as an arc within the segment's length at the speed limit. Then measure whether the derived value beats the constant |

### 17.3 The bow amplitude

| Field | Value |
|---|---|
| **Strategy** | Bow the interior samples sideways so IPOPT starts in an "arc-shaped basin" |
| **Mechanism** | `overboard/core/optimizer.py`, `amplitude = min(0.10 * length, 0.03)` |
| **Outcome** | The amplitude is uncorrelated with the turn angle. On a 90° turn over 1 m it bows 3 cm — a slight curve, not a quarter circle. On a 10° turn over 2 m it bows the same 3 cm — a large relative deviation |
| **Status** | dormant |
| **What would revive it** | Compute the bow from the turn: the circular arc that connects the two endpoints with the given headings, then sample it. That is a real path family, not a nudge — and it was tried: Q1 in [Part IV](RESEARCH_AGENDA.md#240-the-closed-questions---do-not-re-litigate) is closed, because the measured family is cheaper to solve but not faster |

## 18. Ledger: timing models

### 18.1 Equal-duration timing treated as a peer of per-segment

| Field | Value |
|---|---|
| **Strategy** | Offer both timing models as candidates and let validation choose |
| **Mechanism** | `overboard/main.py`, the `auto` mode |
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
| **Status** | rejected; the requirement it was meant to serve is met instead by the force-continuity constraint ([5.5](HEURISTICS_AND_VERIFICATION.md#55-where-the-time-allocation-may-change-ki-024)) |
| **What would revive it** | A sample layout that follows the arc length, so a shared `dt` already gives a constant-speed traversal. That changes the sample count and the sample-to-waypoint mapping, so it is a separate change |

### 18.4 The removed accuracy-weight ladder

| Field | Value |
|---|---|
| **Strategy** | Retry the solve with a higher `accuracy_weight` when the first attempt fails |
| **Mechanism** | Was `accuracy_weight += 2.0`; removed from the shipped path and archived with Multi-Verse |
| **Why it was tried** | A higher weight buys smoothness, which was assumed to buy tracking |
| **Outcome** | **Removed.** The old ladder "silently traded speed for a bound that could get worse (longer trajectories accumulate more uncertainty)". On the complex mission the slow 17.1 s candidate **diverged**, while the 6.07 s one stayed bounded |
| **Status** | dead; no shipped code path uses it |
| **What would revive it** | Only with a *reason* to expect the retry to help. If the failure mode were "the path is too jerky for the controller", a weight ladder would be the right tool. It is not that failure mode today |

### 18.5 Global re-timing (the time floor) as a speed lever

| Field | Value |
|---|---|
| **Strategy** | Re-time each plan's fixed geometry with a global forward-backward (TOPP-style) pass and treat the gap to that time floor as the speed the timing layer leaves on the table |
| **Mechanism** | `research/retime_gap.py`: E = v^2 forward-backward pass under the NLP's own robust-arm bounds (wheel-speed box, motor torque-speed line, traction with lateral transfer, accel ramp), runs split at stops and direction flips |
| **Why it was tried** | The objective prices time + jerk + discretisation, not global time optimality, so a local NLP solve might sit visibly above the floor of its own path |
| **Outcome** | **Retired as a speed lever.** Gap median **2.33 %** of the floor (p25 1.50, p75 3.64; 96/103 within 5 %), and the gap buys a ~4x smoother profile (peak jerk 40 vs 676 m/s^3). The floor is not gate-feasible: the re-timed plans fail the motor gate on **103/103** (median 1.10 N over a 0.001 N threshold) where the NLP plans pass 103/103 at 0.000 N |
| **Evidence** | `research/results/retime_gap.jsonl`, Q21 in [RESEARCH_AGENDA](RESEARCH_AGENDA.md) |
| **Status** | dead as a speed lever; dormant as a convex timing layer for Q20 robustness (no iteration cliff) |
| **What would revive it** | Q20 grinding becoming a shipping problem (solver deaths on the nominal arm), or jerk entering as a hard bound so a convex layer can enforce it - the re-time would then need the same force margin the NLP pricing buys |

## 19. Ledger: candidate set strategies

### 19.1 The sweep grid

| Field | Value |
|---|---|
| **Strategy** | Add the timing-mode × accuracy-weight grid on top of the default candidates |
| **Mechanism** | `overboard/main.py`, `SWEEP_WEIGHTS` at `overboard/main.py` |
| **Why it was tried** | The best weight depends on the path, and the default set only tries one |
| **Outcome** | **It works, and it is the only candidate-set strategy that has ever produced a new best trajectory.** It can only add candidates, never remove them, so it cannot make things worse — it just costs more solves and more validation passes |
| **Evidence (measured 2026-10-20)** | `example_mission_first4`: the default candidate-1 result is 2.176 s; the grid's `baseline` per-segment at weight 0 passes at 1.833 s, and `bow_right` per-segment at weight 0 is faster still at **1.663 s** — 23.5 % faster — and is the selected candidate. `example_with_events`: the default set now produces a passing candidate under per-segment timing (4.416 s), where the equal-duration `baseline` still misses the tracking gate at 0.68 cm. `example_complex_mission`: the grid adds passing `baseline` **equal-duration** candidates at weights 0.001 (7.049 s), 0.002 (7.080 s), 0.005 (7.208 s) and 0.01 (9.323 s), but the equal-duration baseline at weight 0 (6.068 s) is still selected, so the complex mission is unchanged; every per-segment candidate in the grid fails the tracking gate (0.67–0.91 cm) |
| **Status** | **live** (opt-in, off by default) |
| **What would revive it** | Nothing; it should probably become the default for the missions where it wins. The measurement to make next is Q6: whether the weight is a real accuracy proxy, because right now it is a lottery with five tickets |

### 19.2 Multi-start as a quality claim

| Field | Value |
|---|---|
| **Strategy** | Solve more candidates and take the best |
| **Mechanism** | `--candidates N`, `overboard/main.py` |
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
| **Strategy** | Replace stochastic genetic search with an analytical seed generator (`overboard/core/frontier_seed.py`) encoding the 5 physical principles uncovered by phase portrait diagnostics and the 50-generation marathon |
| **Mechanism** | `overboard/core/frontier_seed.py` (`generate_frontier_seed`, `generate_canonical_seeds`), integrated into `TrajectoryOptimizer._build_initial_guess` via `GUESS_STYLES = (..., "frontier_sprint", "frontier_sweeper", "frontier_exploratory", "frontier_direct")` |
| **The 5 Design Heuristics** | 1. **Active Kinematic Hypotenuse:** $v(t) + \frac{W}{2}|\omega(t)| = v_{\max}$. In differential-drive, motor voltage saturation dominates tire friction by $2.5\times$. The optimal trajectory rides the $L_1$ actuator diamond boundary (one wheel at 100% saturation at all times) with analytical velocity profiling $v_{\text{frontier}}(\kappa) = \frac{v_{\text{wheel\_max}}}{1 + 0.5 B |\kappa|}$.<br>2. **Late-Apex Dynamic Heuristic:** Asymmetric corner entry ($f_{\text{apex}} = 0.66\text{--}0.70$) enables deep trail-braking and early symmetric exit throttle ($0.3\text{ s}$ earlier full acceleration onto straights).<br>3. **Waypoint Momentum Preservation:** Maintains $0.65\text{--}0.75 \times v_{\max}$ through unpinned intermediate waypoints along a continuous energy arc.<br>4. **KKT Slack Barrier Equilibrium:** Balances right on the safety threshold ($4.3\text{--}4.5\text{ mm}$ tracking against the $5.0\text{ mm}$ gate).<br>5. **Multi-Modal Homotopy Triad:** Dispatches 3 canonical homotopies (`frontier_sprint`, `frontier_sweeper`, `frontier_exploratory`). |
| **Measured Outcome (2026-10-20)** | **Shatters the 44-Gene Champion on the 10-waypoint complex mission:**<br>- **Stock Baseline:** $5.948\text{ s}$, $0.96\text{ cm}$ tracking, 200 iters (capped), $3.45\text{ s}$ wall (**FAIL**)<br>- **44-Gene GA Champion (12 Gens):** $5.417\text{ s}$, $0.49\text{ cm}$ ($4.9\text{ mm}$), 120 iters, $\approx 25\text{ s}$ wall (**PASS**)<br>- **Frontier Sprint Seed:** **$5.239\text{ s}$**, **$0.45\text{ cm}$ ($4.5\text{ mm}$)**, $1.17\text{ cm}$ envelope, **Score 5.239** (**PASS**)<br>- **Frontier Sweeper Seed:** **$5.237\text{ s}$**, **$0.44\text{ cm}$ ($4.4\text{ mm}$)**, $1.18\text{ cm}$ envelope, **101 iters**, **Score 5.237** (**PASS**)<br>- **Frontier Exploratory Seed:** **$5.233\text{ s}$**, **$0.43\text{ cm}$ ($4.3\text{ mm}$)**, $1.18\text{ cm}$ envelope, **Score 5.233** (**PASS**)<br>- **Seed Generation Overhead:** **$1.16\text{ ms}$** total for all 3 seeds |
| **Status** | **Live** in `overboard/core/frontier_seed.py` and `overboard/core/optimizer.py` |

### 19.6 The 3-Pillar Unified Trajectory Engine (`--unified`, the default path)

| Field | Value |
|---|---|
| **Strategy** | Condense the 9 physical principles into 3 unified engineering pillars, eliminating stochastic evolutionary loops while ensuring 100% deterministic gate certification across all competition paths |
| **Mechanism** | `overboard/core/unified_planner.py` (`UnifiedPathEngine`), `--unified`, tested by `tests/test_frontier_seed.py` and `tests/test_unified_planner.py` |
| **The 3 Pillars** | **Pillar I: Asymmetric Transition Geometry** (Rule 2 Late-Apex Dynamic Law $f_{\text{apex}} = 0.68$, Rule 5 Homotopy Triad: Sprint, Sweeper, Exploratory, Rule 6 Clothoid Curvature Continuity).<br>**Pillar II: Composite Kinetic Velocity Envelope** (Rule 1 Active Kinematic Hypotenuse on $L_1$ polytope boundary, Rule 3 Waypoint Momentum Preservation $\ge 0.70 v_{\max}$, Rule 8 Dynamic Load Transfer $v_{\text{tip/slip}} = \sqrt{g B / (2 h_{\text{CG}} \|\kappa\|)}$, Rule 9 Pontryagin Threshold Braking for stops).<br>**Pillar III: Invariant Collocation Mesh & KKT Solver** (Rule 4 KKT Slack Barrier Equilibrium at $3.6\text{ mm}$, Rule 7 Invariant 50 ms Sampling Mesh $\ge 10$ nodes/segment, 4-node Gauss-Legendre self-consistent collocation, smooth linear epigraph bounds). |
| **Measured Outcomes (2026-10-20)** | **100% gate certification across the benchmark suite** - measured on 2026-10-20 against the then-current mission set and narrowed seed ranges. It is not a standing claim: `example_s_curve.json` fails its envelope gate today (2.77 cm in segment 1->2; KI-020).<br>- **`example_complex_mission.json`:** **$5.365\text{ s}$**, envelope **$1.16\text{ cm}$**, tracking error **$3.6\text{ mm}$** (vs $5.0\text{ mm}$ gate), max integration error **$0.45\text{ mm}$**, 0 violations. Smashes the 44-Gene Champion ($5.417\text{ s}$, $4.9\text{ mm}$) and stock baseline ($6.068\text{ s}$, $1.22\text{ cm}$).<br>- **`example_straight.json`:** **$2.270\text{ s}$**, envelope **$0.30\text{ cm}$**, 0 violations, max pos error $0.45\text{ mm}$.<br>- **`example_with_stops.json`:** **$5.105\text{ s}$**, envelope **$0.89\text{ cm}$**, 0 violations, max pos error $0.46\text{ mm}$.<br>- **`example_s_curve.json`:** **$3.602\text{ s}$**, envelope **$0.83\text{ cm}$**, 0 violations, max pos error $0.46\text{ mm}$.<br>- **Compute Overhead:** Triad analytical seed generation takes **$< 2.5\text{ ms}$**; parallel IPOPT solves converge in **$40\text{--}48\text{ iters}$** in $\approx 1.7\text{ s}$ per candidate. |
| **Status** | **The only planner**, in `overboard/core/unified_planner.py` and in the CLI; the stock candidate set and the other engines are retired |

### 19.7 The Q17 topology and geometry candidates

| Field | Value |
|---|---|
| **Strategy** | Append candidates that differ in *topology* (reverse segments) and *geometry* (mirrored bow, Bezier) to the frontier triad, hoping they land in different NLP basins and pass missions the triad cannot (Q17, `docs/RESEARCH_AGENDA.md` §24) |
| **Mechanism** | `overboard/core/frontier_seed.py` (`StyleParams.apex_sign`, `geometry: straight|bezier`, the reverse axis: on a segment whose pinned heading is more than 120° off the chord, `heading + pi` and `v < 0`), `overboard/core/unified_planner.py:candidate_specs` positions 4-6: `frontier_sprint_mirror`, `frontier_spline`, `frontier_reverse` |
| **Why it was tried** | The triad passes or fails as a block (Q13/Q14): every passing mission has exactly one passing candidate, and the three differ by 0.89 % in duration. Q1's Bezier family had shown a gate-insurance signal (15 vs 11 passes) but was never in the candidate set |
| **Outcome** | **Null: zero gate coverage, and the candidates do not stay distinct.** On the 32-mission suite (m000-m031, seed 7, `--propagation plant`, default limits) the expanded arm (`--candidates 6`) passes exactly the same 6 missions as the triad (`m010, m015, m017, m021, m028, m031`) - 0 flips - at +0.02 % mean selected duration (production rule). The collapse diagnostic fires: the seeds are 3.03 cm apart (mean pairwise, new x triad) but the solved plans sit 0.013 cm apart - IPOPT walks every shaped seed back to one path, exactly F8's mechanism. Every one of the 288 solves converged; the tail candidates gate-ok only where the triad already does. Stage 2 (the geometric anchor `w_anchor * \|p - p_seed\|^2`) was measured as registered and is also a null: at `w_anchor = 1000 s/m^2` the plans stay 0.030 cm apart, coverage is still 0 flips (both anchored arms 5/32, `m031` lost), and the price is severe (mean selected duration 2.32 -> 3.38 s, iterations x4). The term was removed from the optimizer again with the verdict (Q15 discipline) |
| **Evidence (measured 2026-10-26)** | `research/results/q17_base.jsonl` (triad, 6/32), `q17_expanded.jsonl` (6 candidates, 6/32, deviation 0.0126 cm over 288 pairs), `q17_base_anchor.jsonl` / `q17_expanded_anchor.jsonl` (stage 2). The seeds' own pairwise deviation was measured separately: 3.03 cm mean, so the collapse is the solver's, not the seeders' |
| **Status** | **Tail, opt-in only** (the pre-registered null-result policy: additive candidates stay in the tree inert at the shipped default). `--candidates 6` is the gate-coverage mode; the default stays 2 and nothing about the shipped path changes. The anchor term is **removed** |
| **What would revive it** | New evidence, not this experiment again: a geometric term that survives at a duration price the mission budget can pay (the anchor's failure was price, not mechanism - at extreme weights it holds the shape but the plans lose 33-142 % of their duration), or missions whose gates a *deliberately slower* shaped path can pass. The measurement to make first is what the seeds see: why 3 cm of seed separation is worth less than 0.1 mm of basin separation at the objective's indifference |

## 20. Ledger: archived Multi-Verse internals

The Multi-Verse internals were removed from the tree (git tag `archive/legacy-2026-10`) and are kept
as history, not as a supported path. Do not reintroduce `--multiverse`,
`--refine`, the Reeds-Shepp bootstrap, the critic, the JAX refinement or the
refine probe without new evidence; the measured conclusion was that the
pipeline rescued nothing over the stock candidate set.

### 20.1 The JAX refinement

**Hypothesis:** a JAX-refined geometric guess would beat the straight-line seeds.
**Result:** after four defects were fixed, the probe could win on one mission,
but over 30 missions the refined seed rescued nothing. The defect list and the
measurements are in `archive/research-2026-10` (see
[`ARCHIVE.md`](ARCHIVE.md)).
**Mechanism (archived):** `archive/legacy-2026-10:overboard/legacy/jax_optimizer.py`
`generate_candidates_jax`; `archive/legacy-2026-10:overboard/legacy/multiverse_optimizer.py`.

### 20.2 The refine probe

**Hypothesis:** a second cold solve from the refined trajectory would beat the
primary solve by more than solver noise.
**Result:** it did on one mission and not across the suite, so `--refine` was
removed with the pipeline.
**Mechanism (archived):** `archive/legacy-2026-10:overboard/legacy/multiverse_optimizer.py`
`_global_solve`, `:487` `_probe_refined_guess`.

## 21. Ledger: validation and tooling

### 21.1 `--workers` / `--no-parallel` as a quality lever

| Field | Value |
|---|---|
| **Strategy** | Solve candidates in parallel to make the planner faster and therefore able to try more |
| **Mechanism** | `overboard/main.py`, `overboard/main.py` |
| **Outcome** | The *speed* is real; the *quality* claim is not. The trajectories are identical to the sequential run. The flags are a wall-clock lever |
| **Status** | live, but must not be described as a quality claim |
| **What would revive it** | Nothing |

### 21.2 Reachability as a post-hoc score

| Field | Value |
|---|---|
| **Strategy** | Use the reachability envelope to choose between candidates |
| **Mechanism** | `overboard/main.py`, `overboard/main.py` |
| **Outcome** | **It works as a selector and fails as a planner.** The gate rejects a trajectory that the optimizer produced without any way of knowing it would be rejected. The failed candidate is reported to a human, who must then decide what to change |
| **Status** | partial (KI-020) |
| **What would revive it** | Put a conservative envelope proxy in the objective (Q4) or feed the failing segment back into the solver (Q5) |

### 21.3 The random benchmark mode

| Field | Value |
|---|---|
| **Strategy** | Generate random missions and benchmark the planner on them |
| **Mechanism** | `benchmark.py --mode random`; `RandomBenchmark` at `benchmark.py` |
| **Outcome** | **Fixed** (KI-015). The suite generator unpacks `solve()`'s tuple, uses `examples/fll_choreo.json`, and solves every mission; the pipeline mode reads `stats["converged"]` and prints `FAILED`; the random mode validates against the config it actually used (`benchmarks/config_used.json`) and draws its waypoints as a random walk instead of independent uniform draws. A third defect surfaced while checking: the generator's `samples` parameter was shadowed by the solve result, so the *second* mission solved with the first mission's sample list as the sample count and died in the optimizer. `tests/test_random_suite.py` runs two missions and fails if either raises |
| **Status** | **closed** |
| **What would revive it** | Nothing. The mission suite it was supposed to supply now comes from `research/suite.py` and is measured by `research/harness.py` (Q8, answered in `docs/research/pq_b/FINDINGS.md`) |

### 21.4 `sysid_solver.py`

| Field | Value |
|---|---|
| **Strategy** | A pandas/JAX system identification solver with hardcoded CSV paths |
| **Outcome** | **Never wired into the CLI** and never emitting the current `sysid_fit.json` format |
| **Status** | **dead** - the file was deleted and `pandas` dropped from the optional requirements (KI-045). `overboard/sysid/fit.py` is the live path |

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
| **Mechanism** | `overboard/sysid/digital_twin.py` `SlipFit`, `overboard/sysid/digital_twin.py` `brake_shortfall` |
| **Outcome** | **Applied** by the fitted plant as a smooth grip factor (the default kinematic path is unchanged, because it has no fit to read it from) |
| **Status** | partial |
| **What would revive it** | The shortfall is wired into the five-state plant, and the effect is measured: on the complex mission with an illustrative actuator fit (50 ms lag, 2 % gain band) the envelope grows 1.11 cm and the tracking error 0.55 cm (`research/brake_slip_effect.py`). What is left is a fit measured on the robot, and the gates it would have to pass (KI-020) |

### 21.7 Coast-down drag (Q25.2, null)

| Field | Value |
|---|---|
| **Strategy** | Coulomb + viscous load in the wheel-speed dynamics, from coast-down tails |
| **Mechanism** | Was `DragFit` in `overboard/sysid/digital_twin.py`, `_fit_drag` in `overboard/sysid/fit.py`, plant load in `overboard/verification/immrax_validator.py` (`DigitalTwinOpenLoop`); removed again with the verdict |
| **Why it was tried** | RQ-B measured the coast gap (oracle 3.59 vs model 1.51 m/s2, 2.4x); the braking-shortfall fit absorbs part of the drag (RQ-C Finding 3) |
| **Outcome** | **Null, retired.** Fitted coulomb 3.069 m/s2 (viscous 0, 3 repeats, zero-width deterministic intervals) closes the RQ-B gap to 1.17x, but as an always-on wheel load it degrades twin-vs-oracle on the registered 40: mean e_pos 606.6 -> 964.7mm (+358.1, +59%), worst 1824 -> 1972mm, containment 4.12 -> 1.89%, 12/40 beats-or-ties. The DriveBase PID compensates drag while driving, so a motors-off coast number double-counts |
| **Evidence** | `research/results/q25_c2_drag.jsonl` / `q25_c2_drag_summary.json`, `research/results/q25_c2_drag.md`, `research/q25_drag.py` (reuses C1 samples + traces, no new WSL) |
| **Status** | dead, removed |
| **What would revive it** | A regime-gated drag (coast-only, command~0) with evidence that final-stop accuracy moves a gate; today's max-error metric is dominated by driven excursions the PID already masks |

### 21.8 Per-side radii in the truth kinematics alone (Q25 untangle, null)

| Field | Value |
|---|---|
| **Strategy** | Claim C1's twin-vs-oracle gain for the UMBmark pair (`Ed`/`Eb`) as geometry in the truth kinematics |
| **Mechanism** | `DigitalTwinOpenLoop` truth map with `wheel_geometry_tuple()` (kL/kR/b_eff); belief held at scales-only (arm R vs arm A) |
| **Why it was tried** | C1 bundled geometry + belief wiring and landed (mean -233.3mm); the untangle split the bundle to stop radii claiming the win |
| **Outcome** | **Null.** On the registered 40 (4-way intersection, 1962 tubes): R-A -1.1mm mean (-0.2%), -2.7mm worst; F-A -231.4mm (-41.9%); B-F -1.0mm; total B-A -232.4mm (C1 -233.3mm, agree within 1mm). Belief carries ~99.5%; radii + interaction ~1mm each (oracle floor noise). Strict beats-or-ties on intersection: R<=A 34/40, F<=A 39/40, B<=A 39/40; containment tie 5.9-7.2% (shared mm-tube vs dm-excursion limitation) |
| **Evidence** | `research/results/q25_c1_untangle.jsonl` / `q25_c1_untangle_summary.json`, `research/results/q25_c1_untangle.md` (Q25-C1 untangle, 40 missions, intersection; m006 109 vs 48 precedent), `research/q25_untangle.py` |
| **Status** | **dead as a standalone claim** (Q15/Q17 discipline: the term stays only as wired belief + geometry, belief-first; radii alone claim nothing) |
| **What would revive it** | A residual stratified by phase that isolates a truth-kinematics error beyond the repeatability floor after belief wiring; today's metric is dominated by the feedback wiring, not the geometry |

### 21.9 Slip-ratio state beyond the braking shortfall (Q25-4 trigger check, SKIP)

| Field | Value |
|---|---|
| **Strategy** | A slip state (slip-ratio dynamics) beyond the existing `SlipFit` braking shortfall, identified from extended `brake_from_*` tests |
| **Mechanism** | (never built) Q25 component 4 in `docs/research/RQ-C_digital_twin_model.md` rank order, contingent on components 1-3 leaving braking/acceleration-concentrated residual |
| **Why it was tried** | Registered trigger: build ONLY if the post-C1/C2 residual concentrates in braking/acceleration phases beyond the B1 repeatability floor (exactly 0) |
| **Outcome** | **SKIP — contingent-not-triggered.** B-arm residual is diffuse: braking+accel hold 15.5% of squared error on 24.9% of samples (under-represented), braking/cruise contrast 1.06, F-arm identical (15.5%); full-set B agrees (19.1% on 22.8%). The weight sits in turn (35.5%) and cruise (49.0%) with the worst sample (719.8mm) in cruise — a belief/odometry-drift shape, not longitudinal slip. All three pre-registered gate conjuncts fail. The traction cap + `SlipFit` shortfall stand |
| **Evidence** | `research/results/q25_stratify.jsonl` / `q25_stratify_summary.json` / `q25_stratify.md`, `research/q25_stratify.py` (1962-tube intersection + 2964-tube full set, 40 missions, no new WSL runs) |
| **Status** | **closed as contingent-not-triggered** (Q15/Q17 discipline: the term is never added) |
| **What would revive it** | A phase-stratified residual with braking+accel >= 60% of squared error on a minority of samples AND >= 2x braking-vs-cruise contrast, replicated on the F-arm — i.e. the registered gate passing on new evidence, not a re-tune of its bars |

## 22. Ledger: silent and unlogged behaviour

These are not "strategies that failed" but behaviours that changed the results without being recorded where a reader would look.

### 22.1 The per-segment timing model was a jerk-penalty loophole (KI-022)

| Field | Value |
|---|---|
| **What happened** | The per-segment model was not held to the same wheel-jerk discipline as the equal-duration model. Because the raw acceleration jump scales with `jerk * dt`, a short-`dt` segment was charged `dt^2` less per unit of jerk, so per-segment timing could buy violent wheel motion nearly for free |
| **Evidence** | The per-segment model reached **68.9 m/s³** of wheel jerk where the equal-duration model reached **9.5 m/s³** at the same weight |
| **Fix** | The cost is now `dt_ref^2 * sum(J^2)` in both modes, with `dt_ref` the mean interval over the whole path (`overboard/core/optimizer.py`, `overboard/core/optimizer.py`) |
| **Status** | fixed |

### 22.2 The reachability tube lagged the reference by one sample (KI-019)

| Field | Value |
|---|---|
| **What happened** | The tube was computed against a different reference than the one the controller follows, so the reported tracking error was optimistic by one sample |
| **Fix** | The same linear reference and 20 ms sample-and-hold loop as `ramsete_fast.py`, with the plant integrated in 5 ms substeps (`overboard/verification/immrax_validator.py`, `overboard/verification/immrax_validator.py`) |
| **Status** | fixed |

### 22.3 The Immrax upper uncertainty bounds were ignored (KI-003)

| Field | Value |
|---|---|
| **What happened** | `w_flat` was passed as a plain array, which Immrax treats as constants, so only the lower bounds affected the reachable set. Torque, backlash, track width, wheel radius and inertia were therefore never checked in the upper direction |
| **Fix** | `Interval(w_lower, w_upper)` is passed instead (`overboard/verification/immrax_validator.py`), and the physical parameters are wrapped as intervals too (`overboard/verification/immrax_validator.py`) |
| **Status** | fixed |

### 22.4 A failed solve was exported as a success (KI-001) and `--validate` could not block output (KI-002)

| Field | Value |
|---|---|
| **What happened** | `overboard/main.py` did not check `stats["converged"]`, and `passed_validation` was assigned but never read. If Immrax threw, the code set `passed_validation = True` and continued. A failed validation still wrote the trajectory and exited 0 |
| **Evidence** | A solve from `(0,0,0)` to `(100,0,0)` returned `converged=False` and a trajectory that ended near `x = 20` instead of `x = 100` — written as a normal trajectory, and the separate validator still reported PASS |
| **Fix** | The export gate (`overboard/main.py`) blocks both cases; a failed validation exits 1 and writes nothing |
| **Status** | fixed |

### 22.5 The weight-transfer math was wrong and the tools disagreed (KI-010)

| Field | Value |
|---|---|
| **What happened** | The longitudinal term was subtracted from both wheels, and the optimizer and the validator used different formulas |
| **Fix** | One shared `wheel_normal_forces` (`overboard/models/robot_model.py`), called from all four consumers. Lateral transfer only — a two-wheel robot has no front/rear split |
| **Status** | fixed |

## 23. Cross-cutting lessons

Four lessons from the ledger, and they are the reason the research agenda ([`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md)) is shaped the way it is.

**1. The baseline is hard to beat *on speed* because the objective does not care about the path.** The objective is duration plus a jerk penalty. It has no term for "the path is smooth in space", only "the wheels move smoothly in time". Two geometrically different paths of the same length and the same wheel jerk are indistinguishable to the optimizer. So a better *seed* only helps if it puts IPOPT in a basin that is better **by the objective** — and a bowed or turn-first guess is not, because the objective rewards the straight line it converges back to. Any seed improvement must either change the objective or exploit a genuine local minimum.

**2. The seeds' real value is that they are cheap insurance against a gate, not a source of speed.** Measured over 60 candidates ([Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison)), the non-baseline seeds never produced the fastest trajectory. What they did produce was **passing** trajectories where the baseline failed: the selected candidate on two of the six missions, and the only passing set on the events mission. Because a solve costs about two seconds, buying four extra lottery tickets to escape a gate is a good trade. It should be described that way, not as an optimisation improvement.

**3. Everything that changes the objective's shape is a bigger lever than anything that changes the starting point.** The timing model (KI-022) and the accuracy weight changed results by 25–40 %. The seeds changed the outcome on 2 of 6 missions and never improved the fastest duration. The research effort should follow that ratio.

**4. The candidate set is a selection mechanism, not a search mechanism.** The candidate set works: it tries a handful of options and picks the one that passes. What it does not do is explore. The sweep grid was the only candidate-set change that ever produced a new best trajectory, and it worked because it added a genuinely different *parameter* (the weight), not a different *path*.

**5. The measurement base is too narrow to support strong claims.** Everything in [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) rests on six example missions. The random benchmark that was supposed to provide a mission suite is fixed (KI-015) and the harness under [`research/`](../research/) answers Q8 on 32 missions; what is still missing is a **re-run of that suite on the current tree**, because the numbers in `docs/research/pq_b/` predate KI-025 to KI-028. "The baseline wins the complex mission" might be a fact about the planner, or it might be a fact about that one path. **(2026-10-22: the suite has now been re-run on the current tree, but on the shipping planner - Q13, `--candidates 3 --validate`: 5 of 32 missions pass under `plant` and 6 under `tick`, with exactly one passing candidate per mission. The stock sets that produced the 10-of-32 figure remain a different candidate family, so the older numbers are still a record of a planner that no longer ships, not a baseline to compare against.)**

---

## 24. Ledger: objective terms and residual pricing (F2)

The gate measures the closed-loop tracking residual (Q18), the goal order prices
spread first (PROJECT_GOAL), and the objective prices time + jerk +
discretisation. These are the strategies that tried to close that gap.

### 24.1 The static plan-feature surrogate for the tracking residual

| Field | Value |
|---|---|
| **Strategy** | Price the closed-loop tracking residual in the objective through a static map of recorded plan features |
| **Mechanism** | Within-mission (fixed-effects) regression of the residual on duration, jerk, turn-rate, accel, wheel-speed, path-length and force-util features (`research/residual_features.py`); the fitted map would have been the objective term |
| **Why it was tried** | The gate measures the residual, the residual is separable from the clock and has 2.8-4.6 mm of headroom where plans vary (Q22), and a static term is the cheapest way to put it in the loop |
| **Outcome** | **Retired.** Best honest FE R^2 is **0.738** (5 features x 63 rows, density) and **0.654** (weights) against the pre-registered 0.75 bar, and the sign of the best feature (mean turn-rate) flips across missions (0.50 on weights). A term fitted on this map would be mis-signed on about half the missions |
| **Evidence** | `research/results/residual_features.json`, Q23 in [RESEARCH_AGENDA](RESEARCH_AGENDA.md); the high R^2 on the q13/q14 corpora (0.67-0.99) fits 0.07-0.38 mm of within-mission noise (Q22's scale check), not signal |
| **Status** | dead |
| **What would revive it** | A per-mission fitted map could price a trade the operator already has through `accuracy_weight` (Q6); a mission-general static term cannot |

### 24.2 The gradient polish's guidance gradient (Q16's lens)

| Field | Value |
|---|---|
| **Strategy** | Let the differentiated reachability rollout say which samples to fix, then re-solve with those samples weighted (`--polish`; the unified planner's near-gate self-heal) |
| **Mechanism** | `reachability_engine._get_grad_fn` differentiated a 1-tick x 1-substep rollout into per-sample weights (`unified_planner._sensitivity_weights`), and the re-solve re-priced the smoothness/discretisation costs with them |
| **Why it was tried** | The rollout is differentiable, so its gradient should locate the samples that move the reachable set |
| **Outcome** | **Broken lens, fixed (2026-10-27), verdict re-measured and confirmed.** At 1x1 the propagation diverges: it reports 133-351 mm tracking error where the gate reports 0.3-3.0 mm, fires the gate-violation hinge on comfortable plans, and produces gradients 100-1000x too large. The fidelity cliff sits between 2x2 and 5x5; `GRAD_TICKS_PER_INTERVAL` is now 5 with `PLANT_SUBSTEPS` and the gradient matches the gate's loss to 0-6.4 %. Q16 re-run on the fixed lens: 33/33 candidate polishes across the six examples and the five near-miss plans changed nothing (byte-identical on/off). Differentiated faithfully, the gradient descends the gate's loss on every active plan at a right-sized step (one probe plan clears its hinge entirely), but a fixed step overshoots on others - and the polish cannot land the trade regardless: the re-solve raises `accuracy_weight` x3 (smoothness bought with time, the Q6 trade) while the accept rule admits no time increase and demands strict (tracking, envelope) improvement |
| **Evidence** | `research/results/rollout_gradient.json`, `research/rollout_gradient.py`, `research/results/q16_*_v2.jsonl`, Q24 and the Q16 record in [RESEARCH_AGENDA](RESEARCH_AGENDA.md) |
| **Status** | dead as a polish (2026-10-27): guidance repaired, but the accept rule forbids the trade the re-solve makes; the gradient machinery is retained as the M1 vehicle |
| **What would revive it** | Nothing in polish form - the accepted set (not slower, strictly better tube) is empty under the Q6 trade, and a polish that is allowed to cost time is just another candidate the decision score already prices. The machinery revives only as the M1 in-loop linear/proximal term, where time is spent to reach feasibility rather than to polish; gated on KI-020 (the residual it prices is the twin's) |
