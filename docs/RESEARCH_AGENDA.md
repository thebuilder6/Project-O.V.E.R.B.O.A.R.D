# The research agenda

**Status:** 2026-10-19, as measured on that tree. Part IV of the heuristics document.
**Scope:** the open questions, ordered so that the cheapest decisive experiment comes first, each
with the current evidence, the hypothesis, the experiment, the metric and the success criterion.
**Audience:** anyone about to spend an afternoon on the planner.

> **Read this first.** A question here is open because the measurement that would answer it has not
> been made yet. The current-state reference is
> [`HEURISTICS_AND_VERIFICATION.md`](HEURISTICS_AND_VERIFICATION.md), and the strategies that
> already failed are in [`STRATEGY_LEDGER.md`](STRATEGY_LEDGER.md) - read that before spending a
> day on a strategy that is in there.
>
> **The section numbers are the ones the sections had in that document**, so a citation like "§24.1"
> still points at the same text - in this file now.

## Contents

- [24. The open questions](#24-the-open-questions)
  - [Q1 — Does a curvature-continuous path family beat the straight-line seed?](#q1--does-a-curvature-continuous-path-family-beat-the-straight-line-seed)
  - [Q2 — Does a time-allocation warm start reduce iterations?](#q2--does-a-time-allocation-warm-start-reduce-iterations)
  - [Q3 — Is multi-start worth anything at all?](#q3--is-multi-start-worth-anything-at-all)
  - [Q4 — Should reachability be an objective term?](#q4--should-reachability-be-an-objective-term)
  - [Q5 — Should the critic feed back into the solver?](#q5--should-the-critic-feed-back-into-the-solver)
  - [Q6 — Does `accuracy_weight` actually buy accuracy?](#q6--does-accuracy_weight-actually-buy-accuracy)
  - [Q7 — Does sample density beat timing flexibility?](#q7--does-sample-density-beat-timing-flexibility)
  - [Q8 — Is the baseline's dominance an artefact of the example missions?](#q8--is-the-baselines-dominance-an-artefact-of-the-example-missions)
  - [Q9 — What would make the validator a real digital twin?](#q9--what-would-make-the-validator-a-real-digital-twin)
  - [Q10 — Is "smallest envelope" the right selection objective?](#q10--is-smallest-envelope-the-right-selection-objective)
  - [Q11 — Should the executor be part of the planner?](#q11--should-the-executor-be-part-of-the-planner)
  - [Q12 — What should the Multi-Verse path become?](#q12--what-should-the-multi-verse-path-become)
  - [Q13 — Should the measured matrix be re-measured on the unified path?](#q13--should-the-measured-matrix-be-re-measured-on-the-unified-path)
  - [24.1 Suggested order](#241-suggested-order)
- [25. How to report a result](#25-how-to-report-a-result)

- [24. The open questions](#24-the-open-questions)
  - [Q1 — Does a curvature-continuous path family beat the straight-line seed?](#q1--does-a-curvature-continuous-path-family-beat-the-straight-line-seed)
  - [Q2 — Does a time-allocation warm start reduce iterations?](#q2--does-a-time-allocation-warm-start-reduce-iterations)
  - [Q3 — Is multi-start worth anything at all?](#q3--is-multi-start-worth-anything-at-all)
  - [Q4 — Should reachability be an objective term?](#q4--should-reachability-be-an-objective-term)
  - [Q5 — Should the critic feed back into the solver?](#q5--should-the-critic-feed-back-into-the-solver)
  - [Q6 — Does `accuracy_weight` actually buy accuracy?](#q6--does-accuracy_weight-actually-buy-accuracy)
  - [Q7 — Does sample density beat timing flexibility?](#q7--does-sample-density-beat-timing-flexibility)
  - [Q8 — Is the baseline's dominance an artefact of the example missions?](#q8--is-the-baselines-dominance-an-artefact-of-the-example-missions)
  - [Q9 — What would make the validator a real digital twin?](#q9--what-would-make-the-validator-a-real-digital-twin)
  - [Q10 — Is "smallest envelope" the right selection objective?](#q10--is-smallest-envelope-the-right-selection-objective)
  - [Q11 — Should the executor be part of the planner?](#q11--should-the-executor-be-part-of-the-planner)
  - [Q12 — What should the Multi-Verse path become?](#q12--what-should-the-multi-verse-path-become)
  - [24.1 Suggested order](#241-suggested-order)
- [25. How to report a result](#25-how-to-report-a-result)

## 24. The open questions

Each question has the current evidence, a hypothesis, and an experiment with a metric and a success criterion. The experiments are ordered so that the cheap, decisive ones come first.

### Q1 — Does a curvature-continuous path family beat the straight-line seed?

| | |
|---|---|
| **Question** | The seeds perturb a straight line. Would a seed built from a real path family — a circular arc, a clothoid, a β-spline through the waypoints — find better optima? |
| **Current evidence** | All four structured seeds lose (17.1). The shipped tree contains no geometric path family at all ([Section 2](HEURISTICS_AND_VERIFICATION.md#2-what-is-not-in-the-tree)). Reeds-Shepp exists only in the archived Multi-Verse bootstrap, not as a shipped seed. |
| **Hypothesis** | A seed whose geometry is *feasible for a differential-drive robot* (curvature continuous, reachable) starts IPOPT in a much better basin, and the resulting trajectory is both shorter and tighter. |
| **Experiment** | 1. Implement a geometric path builder in the shipped package (`overboard/core/`): given two poses, produce a curvature-continuous path (Dubins for forward-only, Reeds-Shepp with a real turning radius, or a clothoid-spline for the general case). 2. Add a `guess_style` that uses it, with the timing profile from `_set_rolling_speeds`. 3. Add it to `GUESS_STYLES` and let the candidate set pick it up — no other change is needed. 4. Run the [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) matrix with the new seed added. |
| **Metric** | Trajectory duration, envelope radius, tracking error, IPOPT iteration count, wall time. |
| **Success criterion** | The new seed wins on at least 2 of the 6 missions *and* does not lose on the rest — i.e. it becomes the selected candidate somewhere. A seed that is sometimes better but usually worse is not useful, because it costs a solve. |
| **Cost** | Medium. One new module, one new style, and the existing matrix. |
| **Risk** | If the objective really is indifferent to geometry (lesson 23.1), this fails and the answer is "the seed is not the lever" — which is itself worth knowing. |

### Q2 — Does a time-allocation warm start reduce iterations?

| | |
|---|---|
| **Question** | Every seed starts with a flat `dt = 0.1 s` for every segment (`overboard/core/optimizer.py`). Would a segment-length-proportional allocation converge faster and better? |
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
| **Cost** | Low. The data is already in `attempt_record` (`overboard/main.py`). This is a measurement, not an experiment |
| **Answer (2026-10-20)** | **The hypothesis is half right, and the other half is the opposite.** The insurance is real but small - **one mission in six** - and it comes from the **timing model**, not from a seed: the default set is two timing models of the baseline seed, and the seeds only enter from `--candidates 3` on. The seeds and the refined trajectory add **no** passes; what they buy is **speed** where they reach a different optimum (the complex mission: 6.068 s from the baseline, 5.395 s from a seed, 4.936 s from the refined guess). The measurement is in `archive/research-2026-10:docs/measurements/2026-10-20/matrix/`: the analysis is `analyse_matrix.py` in that folder and that folder's README carries the summary. |

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
| **Hypothesis** | Tracking error correlates with wheel jerk, but not linearly, and the correlation depends on the path. If so, the knob is real but path-dependent, and the retired sweep grid (19.1) was the way it was measured. |
| **Experiment** | For one mission, sweep `accuracy_weight` over a wide range (0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1) under the per-segment model, and record the tracking error, the envelope, the duration and the peak wheel jerk for each. Plot them. |
| **Metric** | Tracking error vs. weight; duration vs. weight; the knee of each curve. |
| **Success criterion** | A measured, quotable relationship that can replace the "usually reduces jerk at the cost of a longer trajectory" hand-wave in `docs/EXAMPLES.md:615`. |
| **Cost** | Low. The retired sweep machinery used the same weights; a one-off weight ladder is enough now. |
| **Note** | Do this before Q4. If the jerk weight is not a good accuracy proxy, then an envelope term in the objective is not optional — it is mandatory. |

### Q7 — Does sample density beat timing flexibility?

| | |
|---|---|
| **Question** | Per-segment timing and higher sample density are two ways to give the solver more freedom. Which one pays? |
| **Current evidence** | `--samples` (2–200) is untested against the per-segment model. The per-segment model is now the only one, so density is the remaining freedom. |
| **Hypothesis** | At a fixed solve time, per-segment timing at low density beats uniform timing at high density, because it changes the *shape* of the solution rather than the resolution of it. |
| **Experiment** | Fix a mission. Compare per-segment at n=10, n=20 and n=40 at equal wall-clock budget (the uniform comparison is historical; the equal-duration model was removed). |
| **Metric** | Duration, envelope, tracking, wall time, iterations. |
| **Success criterion** | A clear ordering at equal wall time. |
| **Cost** | Low. |

### Q8 — Is the baseline's dominance an artefact of the example missions?

| | |
|---|---|
| **Question** | Every conclusion in [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) rests on six hand-written example paths. Does the baseline still win on paths nobody designed for it? |
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
| **Current evidence** | KI-020 lists the gaps: motor transients, DriveBase PID, encoder/gyro odometry, wheel-slip dynamics, interaction forces. The fitted plant now adds the actuator gain, tau, delay, odometry scale, gyro bias, an acceleration limit and a braking shortfall |
| **Hypothesis** | The single largest remaining error source is **odometry error feeding back into the controller**, because the deployed controller never sees the true pose. The fitted plant now models that belief (`_advance_belief`), which makes this testable |
| **Experiment** | 1. Collect a real session with `overboard/sysid/collect.py`. 2. Fit it. 3. Compare the fitted validator's predicted poses against the measured poses with `overboard/sysid/compare.py`, and against the *default* kinematic validator. 4. Report which model predicts the measured poses better, and by how much |
| **Metric** | Position and heading error at each measured stop, for the default model versus the fitted model versus the tape-measured truth. |
| **Success criterion** | A measured, ranked list of which missing model component matters most. |
| **Cost** | High — it needs hardware. **It is also the only way to turn any number in this document from a prediction into a measurement.** |
| **Note (revised by RQ-A)** | This is still the right long-term work, but it is **not what is blocking the missions today.** RQ-A shows the tracking error is already ~100 % explained by the current model: it is the mismatch between the optimizer's discretisation, the reference interpolation and the plant's integration. Adding motor dynamics, odometry and slip will not move the number that rejects the S-curve and the stops mission. Q9 is about making the *envelope* honest; the tracking gate needs a different fix |

### Q10 — Is "smallest envelope" the right selection objective?

| | |
|---|---|
| **Question** | The selection rule is `min(tube, then duration)` (`overboard/main.py`). Is the envelope the right thing to minimise? |
| **Current evidence** | The envelope is a conservative proxy that mixes uncertainty growth along the path with tracking error. Minimising it prefers slow, cautious trajectories, and it does not distinguish "the tube is wide because the robot went fast" from "the tube is wide because the robot drifted" |
| **Hypothesis** | A better objective is the *maximum per-sample excess over the local threshold* (which is what actually fails the gate) combined with the duration, or the tracking error alone. Under that objective some currently-rejected candidates would pass, and the selected trajectory would often be faster |
| **Experiment** | 1. Log, for every candidate in the matrix, the full score vector: tube, per-sample excess, tracking error, duration, peak jerk. 2. Re-run the selection offline under several candidate rules. 3. Compare how often each rule picks a passing candidate and how much time it loses |
| **Metric** | Pass rate and selected duration under each rule. |
| **Success criterion** | A rule that passes at least as often and selects a faster trajectory. |
| **Cost** | Low. The scores are already computed; only the comparison rule changes. |
| **Answer (2026-10-20)** | **The hypothesis is wrong on the tracking error, and the excess is not the lever either.** The envelope is almost a proxy for the **duration** (r = +0.97 over 204 candidates), so the current rule is "shortest run" in disguise. The tracking error is nearly independent of the duration (r = +0.35), so preferring it is **slower in 11 of the 21 groups** and faster in one. The per-sample excess agrees with the envelope on every mission here, because every segment limit is the default 2 cm. The only rule that is never worse is the **duration**: it differs from the current rule in 6 of the 21 groups and is faster in every one. The envelope never rejects a candidate either - 105 of 204 pass, and not one failed on the envelope threshold. The measurement is in `archive/research-2026-10:docs/measurements/2026-10-20/matrix/`; the analysis is `analyse_matrix.py` in that folder, and every candidate now carries its score vector in the `##OVERBOARD-VALIDATION##` line, so a rule can be tried without solving again. |

### Q11 — Should the executor be part of the planner?

| | |
|---|---|
| **Question** | The validator simulates the deployed controller. Should the *plan* also account for the fact that the robot stops and re-accelerates at every stop, and that the DriveBase rounds the commands? |
| **Current evidence** | The DriveBase limits and rounding were fixed once (KI-021) and the validator now quantises commands (`overboard/verification/immrax_validator.py`). But the *optimizer* has no term for either |
| **Hypothesis** | Modelling the executor's quantisation and acceleration limits inside the optimizer would remove a class of late-stage gate failures that no amount of seed or timing work can fix |
| **Experiment** | Quantise the wheel speeds and the acceleration inside the optimizer's constraint set (or add a penalty for the quantisation error), and compare the gate outcome against the current model |
| **Metric** | Gate pass rate, tracking error, duration. |
| **Success criterion** | A measurable reduction in gate failures on missions that currently fail for tracking reasons. |
| **Cost** | High. Quantisation inside an NLP is awkward — it is a step function. Start with the acceleration limit, which is smooth. |

### Q12 — What should the Multi-Verse path become?

| | |
|---|---|
| **Question** | Keep, fix or delete the five-phase pipeline? |
| **Current evidence** | **Answered (KI-004).** The pipeline was repaired enough to measure as a seed provider, and the measurement said the stock seeds are the better buy: over 30 missions the refined seed rescued nothing. It was removed from the tree (git tag `archive/legacy-2026-10`) and has no CLI entry point. |
| **Experiment** | Done; the raw records are in `archive/research-2026-10` (see [`ARCHIVE.md`](ARCHIVE.md)). |
| **Metric** | Whether the pipeline's output wins as a candidate: duration, envelope, tracking, pass rate against the stock seed set. |
| **Success criterion** | A decision, made on evidence. **Made**: archived, not shipped. |
| **Cost** | Low. Do not revive it without a new reason; the next lever is the objective, not the search. |

### Q13 — Should the measured matrix be re-measured on the unified path?

| | |
|---|---|
| **Question** | The matrix in [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) is the **stock** candidate set's measurement: five seeds × two timing models, taken before the 3-Pillar unified engine became the default. The default path solves three frontier candidates instead. Which set is the better buy? |
| **Current evidence** | The two paths are not comparable as measured: the frontier triad carries its own accuracy weights (0.001 / 0.002 / 0.001), a per-segment timing model and the discretization cost metric, so a candidate-for-candidate comparison would be comparing different objectives. The single numbers that exist for the unified path are from its own campaign (the complex mission at 5.365 s / 3.6 mm / 1.16 cm) and are not a matrix. |
| **Hypothesis** | The frontier triad wins as a *set*: three candidates that all pass, on a per-segment timing model, against a stock set that leaves two of six missions with no passing candidate at all. If that holds, the stock set is the thing to prune, not the frontier seeds. |
| **Experiment** | 1. Wait until the optimizer is stable — the frontier seeds and the collocation are both in motion, and a matrix taken now is stale on arrival. 2. Re-run the same matrix with the unified path: the same six missions, the same robot config, the same sample layout, the unified candidates instead of the five seeds. 3. Report both sets on the same missions: candidates solved, candidates passing, selected duration / envelope / tracking, and the wall time. 4. Change the selection rule only if the stock set wins somewhere the triad does not. |
| **Metric** | Pass rate and selected duration / envelope / tracking per mission, for both sets, on the same tree and the same config. |
| **Success criterion** | A decision on which candidate set ships as the default, with the numbers behind it. A tie is a decision too: keep the frontier triad and say why. |
| **Cost** | Low to run, one matrix; the wait is the cost. |
| **Note** | This is a measurement, not an experiment: nothing in the tree changes. The rules in [Section 25](#25-how-to-report-a-result) are the report shape, and the tables in Section 9 are the baseline to beat. |

### 24.1 Suggested order

```text
  ---  DONE: RQ-A  the tracking gate's cause is identified. See
                docs/research/RQ-A_tracking_error.md. It reorders Q4 and Q9.
  ---  DONE: RQ-A's OQ1 and OQ2  the floor at fine sampling is the 20 ms
                command hold, and the optimizer now uses the self-consistent
                collocation unconditionally (KI-026).
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
  ---  DONE: Q12 the Multi-Verse path is archived (KI-004); measured as a
                seed provider it rescued nothing against the stock seeds.
  ---  ANSWERED: Q5  the critic flags every candidate, so it cannot filter;
                its thresholds have to discriminate before it can.
  Q4  reachability in the objective                          (large)
  Q11 executor in the planner                                (large)
  Q9  hardware-calibrated twin                               (needs hardware)
  Q13 re-measure the matrix on the unified path              (waits for a
                                                              stable optimizer)
```

The RQ-A items came first because they were the *cause* of the gate failures that everything else is downstream of: 25 of the 29 failures in the 60-candidate matrix cite the tracking gate, and RQ-A shows it was measuring a discretisation artefact rather than the robot. KI-026 to KI-028 have landed, and the artefact is now small at the fine end (the S-curve at n=40: 0.98 → 0.27 cm) but not at the coarse end, where the plan's own deviation is the whole error. The reference's velocity half has landed too, as the tick-level midpoint lead ([RQ-A finding F8](research/RQ-A_tracking_error.md), commit `47633e9`); it is not the next fix. The remaining work is the plan at coarse sampling and the hardware calibration of the plant (KI-020). **The 5 mm threshold still measures the discretisation more than it measures the robot**, so it must not be tuned to make missions pass.

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

The tables in [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) are the baseline to beat. Any new candidate should be added to the matrix rather than measured on a mission of its own, because the interesting result is not "the new seed is fast" but "the new seed is fast *and* it wins somewhere".

---
