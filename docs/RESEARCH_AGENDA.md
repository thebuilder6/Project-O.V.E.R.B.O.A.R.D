# The research agenda

**Status:** 2026-10-21, as measured on that tree. Part IV of the heuristics document.
**Scope:** the questions that are still open, each with the current evidence, the hypothesis, the
experiment, the metric and the success criterion; [24.1](#241-suggested-order) is the order, cheapest
decisive experiment first. The eleven that are answered are closed in
[24.0](#240-the-closed-questions---do-not-re-litigate) with the measurement that closed them, so
nobody runs them again.
**Audience:** anyone about to spend an afternoon on the planner.

> **Read this first.** The current-state reference is
> [`HEURISTICS_AND_VERIFICATION.md`](HEURISTICS_AND_VERIFICATION.md), and the strategies that already
> failed are in [`STRATEGY_LEDGER.md`](STRATEGY_LEDGER.md) - read that before spending a day on a
> strategy that is in there. Every answer below is a measurement with a stated n; the campaign that
> produced most of them is [`research/pq_b/FINDINGS.md`](research/pq_b/FINDINGS.md) (32 missions, 672
> candidate solves, raw records in the `archive/research-2026-10` tag).
>
> **The section numbers are the ones the sections had in that document**, so a citation like "§24.1"
> still points at the same text - in this file now. A closed question keeps its number for the same
> reason.

## Contents

- [24. The open questions](#24-the-open-questions)
  - [24.0 The closed questions - do not re-litigate](#240-the-closed-questions---do-not-re-litigate)
  - [Q9 — What would make the validator a real digital twin?](#q9--what-would-make-the-validator-a-real-digital-twin)
  - [Q11 — Should the executor be part of the planner?](#q11--should-the-executor-be-part-of-the-planner)
  - [Q13 — Should the measured matrix be re-measured on the unified path?](#q13--should-the-measured-matrix-be-re-measured-on-the-unified-path)
  - [Q15 — Should the tracking gate be a constraint instead of a price?](#q15--should-the-tracking-gate-be-a-constraint-instead-of-a-price)
  - [Q16 — Does the gradient polish ever change a verdict?](#q16--does-the-gradient-polish-ever-change-a-verdict)
  - [24.1 Suggested order](#241-suggested-order)
- [25. How to report a result](#25-how-to-report-a-result)

## 24. The open questions

Sixteen questions were asked. Eleven are measured and closed; five are open. Each open one has the
current evidence, a hypothesis, and an experiment with a metric and a success criterion, ordered so
that the cheap, decisive ones come first.

### 24.0 The closed questions - do not re-litigate

| # | question | answer | where |
|---|---|---|---|
| Q1 | curvature-continuous path family? | **No as a speed lever** (mean +3.0 % slower), yes as a cheaper solve (33 vs 48 iterations) and slightly better gate coverage (15 vs 11 passing) | F8 |
| Q2 | time-allocation warm start? | **Refuted as stated** (slower convergence, 40 vs 25 iterations); the length-proportional allocation finds a faster trajectory on 18 of 32 missions (mean -1.8 %) | F9 |
| Q3 | is multi-start worth anything? | **Insurance, and its price is known**: one extra seed is worth 3 missions, candidates 4-16 buy 1. Under the self-consistent collocation the whole seed axis buys nothing | F5, §7 |
| Q4 | reachability as an objective term? | **Nothing left to buy.** The envelope is a function of the duration (r = +0.90; +0.97 on the 204-candidate matrix), so a term would re-price the clock the objective already minimises. It does bind (46 % of candidates), but the mission-blocking failures are tracking, and the absolute-limit problem is KI-054, not an objective | F2, F11, KI-054 |
| Q5 | feed the critic back into the solver? | **No**: the critic flags 100 % of passing and 100 % of failing candidates, so it cannot select. Its thresholds have to discriminate before it can | F7 |
| Q6 | does `accuracy_weight` buy accuracy? | **Yes within a mission** (median Spearman -0.68) **as a trade, not a dial** (21 of 192 steps non-monotone, best weight per mission). Best found: per-segment at 0.005, 42 % pass against 17 % at 0, for 5 % more time | §4 |
| Q7 | sample density against timing flexibility? | **Per-segment timing at low density.** n = 10 is the sweet spot: 0.60 cm tracking at n = 10 against 0.72 at n = 40 for 5x the solve, and n = 20 passed 0 of 6 | archive tag |
| Q8 | is the baseline's dominance an artefact? | **Yes.** The baseline is the speed winner (11 of 32) *and* the gate loser (lowest pass rate); on the six example paths those two roles did not conflict | F1 |
| Q10 | is "smallest envelope" the right selection rule? | **It does not matter**: five rules pick the same trajectory. The rule has since changed twice (the fitness score, then "fastest passing"); see [7.3](HEURISTICS_AND_VERIFICATION.md#73-the-selection-rule) | F6, [7.3](HEURISTICS_AND_VERIFICATION.md#73-the-selection-rule) |
| Q12 | what should the Multi-Verse path become? | **Archived** (KI-004): measured as a seed provider it rescued nothing over 30 missions | §5 |
| Q14 | how much risk should a plan carry? | **Not a choice between candidates.** On the 32-mission suite (8 per profile, `--candidates 3 --validate --propagation plant`) 10 missions pass, and on every one of them all three candidates pass and no rescue candidate ever does: the set passes or fails as a block. Inside a mission the three differ by 0.89 % in duration (max 1.68 %) and 0.005 of normalised margin (max 0.013), so a margin floor never changes the pick - it only drops missions, and the balanced rule moves 2 of 10 for +0.001 s. A passing plan runs at 54 % of its accuracy limit (mean margin 0.46); the 161 failing candidates need the limit multiplied by 2.9 (median), 14.7 (p90) and 34 (worst), and 49 also break the 5 mm tracking gate. Force is not a dial: every candidate sits at 0.84 of the motor limit. **The risk is in the limit, not in the pick**, so there is nothing for a `--risk` flag to choose between. Plant propagation only; `tick` unmeasured | `research/results/q14_plant.jsonl`, `research/risk_margins.py`, `tests/test_risk_margins.py` |

In the last column, `F<n>` and `§<n>` are sections of [FINDINGS](research/pq_b/FINDINGS.md),
`archive tag` means the raw records in `archive/research-2026-10`. The answers, question by question,
are also in [FINDINGS §8](research/pq_b/FINDINGS.md#8-the-answers-question-by-question).
Do not quote a number from this agenda without its n and its tree: three headline claims of the
six-mission era (the baseline dominates, the envelope never binds, the seeds never win) did not
survive the wider suite, which is why the campaign exists.

### Q9 — What would make the validator a real digital twin?

| | |
|---|---|
| **Question** | Which missing model component buys the most gate accuracy per unit of effort? |
| **Current evidence** | KI-020 lists the gaps: motor transients, DriveBase PID, encoder/gyro odometry, wheel-slip dynamics, interaction forces. The fitted plant now adds the actuator gain, tau, delay, odometry scale, gyro bias, an acceleration limit and a braking shortfall (`overboard/sysid/digital_twin.py`), and the tick propagation can carry the robot's **belief** - its odometry error - as a state (`overboard/verification/immrax_validator.py`, `_advance_belief` / `belief_mode`) |
| **Hypothesis** | The single largest remaining error source is **odometry error feeding back into the controller**, because the deployed controller never sees the true pose. The belief state makes that testable |
| **Experiment** | 1. Collect a real session with `overboard/sysid/collect.py`. 2. Fit it. 3. Compare the fitted validator's predicted poses against the measured poses with `overboard/sysid/compare.py`, and against the *default* kinematic validator. 4. Report which model predicts the measured poses better, and by how much |
| **Metric** | Position and heading error at each measured stop, for the default model versus the fitted model versus the tape-measured truth |
| **Success criterion** | A measured, ranked list of which missing model component matters most |
| **Cost** | High — it needs hardware. **It is also the only way to turn any number in this document from a prediction into a measurement** |
| **Do not wait on it** | It is not what blocks the missions: RQ-A shows the tracking error is already explained by the optimizer's discretisation, the reference interpolation and the plant's integration, and the declared uncertainty does not move it (F3, §6). Q9 is about making the *envelope* honest |

### Q11 — Should the executor be part of the planner?

| | |
|---|---|
| **Question** | The plan ignores two things the executor does: it caps wheel acceleration, and it quantises the commanded speeds. Should the planner know? |
| **Current evidence** | The DriveBase's limits and rounding were fixed once (KI-021) and the validator quantises commands (`immrax_validator.py`, `quantize_drive_command`). The *optimizer* has no term for either. The gate is no longer blind: the default `examples/fll_choreo.json` declares the documented 7.8 m/s^2 DriveBase limit in its `digital_twin` section, and a declared limit alone selects the five-state plant that ramps the wheel speeds - no calibration needed (`DigitalTwinConfig.actuator_accel_limit`, `ImmraxValidator` plant selection). A synthetic plan that asks for 0 to 0.6 m/s in 20 ms (30 m/s^2) went from 0.00003 m tracking error under the kinematic plant to 0.016 m under the documented limit, so it now crosses the 5 mm gate. F10's measurement (**7 % of candidates, 8 % of the passing ones, ask for more acceleration than the hub delivers**) was on the retired stock set, so how many of those 49 candidates the shipping planner still produces is unmeasured |
| **Hypothesis** | A plan that respects the executor's acceleration limit passes the hub as it passes the model, and the gate then sees the plans it currently calls safe and cannot execute |
| **Experiment** | 1. ~~Put the DriveBase's documented limit into the default robot config's digital twin and check that the gate flags an over-limit plan.~~ **Landed** (2026-10-21): `examples/fll_choreo.json` carries `digital_twin.actuator.straight.accel_limit = 7.8`; `tests/test_digital_twin.py` pins the plant and the crossing. 2. Bound wheel acceleration in the optimizer's constraint set. 3. Quantisation last: it is a step function, and a penalty for the quantisation error is the only smooth form. The remaining measurement for step 1 is the 32-mission suite with the shipping planner (Q13), counting how many over-limit candidates now fail |
| **Metric** | Gate pass rate, the fraction of passing plans that exceed the hub's limit, tracking error, duration |
| **Success criterion** | The plans the hub cannot execute stop passing, at a stated duration cost |
| **Cost** | Low for step 1, medium for step 2, high for step 3 |

### Q13 — Should the measured matrix be re-measured on the unified path?

| | |
|---|---|
| **Question** | The tables in [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) are the **retired stock** candidate set's measurement: five seeds × two timing models. The 3-Pillar unified engine is the only planner and solves its own frontier candidates, so those tables describe a planner the tree no longer has. What do they still tell us? |
| **Current evidence** | The stock set is gone, so the comparison this question used to ask ("which set ships") is settled by deletion: the unified path ships. What is left is that Section 9, the ledger and Q1-Q8 were measured on the stock set, and the unified path has only its own campaign numbers (the complex mission at 5.365 s / 3.6 mm / 1.16 cm) rather than a matrix. The candidate-for-candidate comparison is not possible anyway: the frontier triad carries its own accuracy weights, timing model and cost metric, so it would compare different objectives |
| **Hypothesis** | The unified path's numbers land in the same place as the stock set's on every claim the agenda still leans on (the timing model is the lever, the tracking gate is the binding one, Q1-Q8's orderings), so Section 9 needs a provenance stamp rather than a replacement |
| **Experiment** | 1. The wait is over: KI-026 has landed and the collocation is unconditional, so the tree is stable enough to measure. 2. Run the 32-mission suite (`research/suite.py`, `research/harness.py`) through the unified path with `--candidates 3 --validate`, both `--propagation plant` and `tick`. 3. Report per mission: candidates solved, candidates passing, selected duration / envelope / tracking, wall time, and where no candidate passes. 4. Mark Section 9 with the set it came from, and correct any claim the unified set contradicts |
| **Metric** | Pass rate and selected duration / envelope / tracking per mission, on one tree and one config |
| **Success criterion** | Every claim in Section 9 is either reproduced on the shipping planner or stamped as a stock-set measurement. A tie is a result too |
| **Cost** | Low to run, one matrix. This is a measurement, not an experiment: nothing in the tree changes |

### Q15 — Should the tracking gate be a constraint instead of a price?

| | |
|---|---|
| **Question** | The shipping planner prices the three discretisation mechanisms in the objective (`cost_metric="discretization"`) and lets the validator reject afterwards. Should the gate be an inequality on the mesh instead of a price? (The refinement half is answered, below.) |
| **Current evidence** | Every candidate runs on `cost_metric="discretization"` (`overboard/core/unified_planner.py`, `solve_candidate` and `candidate_specs`), and `w_disc = max(10 * w_acc, 0.02)` is fixed by construction, so the candidates differ only in the jerk weight. The price is not the guarantee: the gate is the verdict, and a candidate that misses it is not exported. **The refinement half is closed by RQ-A F6.** The mesh does miss its own name - `compute_invariant_mesh_samples` sizes on `v_nom = 0.65 * 0.477 = 0.310 m/s` and clamps to 10-18, which saturates at 18 for any segment over 0.27 m, so `example_s_curve` asks for 38 per segment and gets 18 (a 102 ms interval, not 50 ms) and `example_straight` asks for 64 and gets 18 - but density is not the lever: the tracking error is U-shaped in density with its floor at 4-5 controller ticks per interval, the shipped 102 ms is 5.1 ticks, and on F6's reference 86 and 118 ms tracked 0.245 and 0.489 cm. F6's 0.271 cm at 40 intervals per segment is no better than 86 ms for five times the solve, so raising the clamp is not the experiment. What F6 does point at and what does not ship is the interval-average reference (0.098 cm against 0.271 on the same plan, `_interpolated_reference` ships the tick midpoint instead); RQ-A owns that, not this question. The constraint arm is untested, and the 112 -> 3145 iteration blow-up in `optimizer.py` is the *traction row at the interval's end speed*, not this inequality - a warning about dense rows, not a measurement of this constraint |
| **Hypothesis** | A hard inequality is infeasible wherever the plan misses the gate by more than the margin, because the deviation is the plan's own chord rather than slack the solver can retract, and it costs duration where it does not. The price is the cheaper of the two until measured, and deleting `w_disc` before that would remove the only term that makes the solver slow where the chord lags |
| **Experiment** | 1. An opt-in `disc_constraints=True` writing the three mechanisms as inequalities against the gate with a margin, on the shipping mesh. 2. The six examples, then the 32-mission suite. 3. Report the infeasible solves separately: they are the answer, not a failure |
| **Metric** | Gate pass count, duration, iterations, wall time, and the number of infeasible solves |
| **Success criterion** | More missions pass at a stated duration cost, or the arm is retired with the count of infeasibilities and the price stays |
| **Cost** | Low - the three expressions already exist in the objective |

### Q16 — Does the gradient polish ever change a verdict?

| | |
|---|---|
| **Question** | `--polish`, and the engine's automatic pass near the gate, re-solve the plan with sample weights taken from the reachability gradient. Has that ever turned a failing candidate into a passing one? |
| **Current evidence** | The mechanism is described in five places (`overboard/core/unified_planner.py`, `gradient_polish` / `polish_candidate_sensitivities` / `_sensitivity_weights`; ControlsDoc's adjoint-polish section; `WHITEPAPER.md` 4.5; `AGENT_OPS.md`; `CONFIGURATION.md`) and **no measurement of its effect is recorded in `docs/`** - no before/after duration, tracking error or pass count. The weights are heuristic (`1 + 4 * grad_norm / peak + 6 * excess^2`); the acceptance rule is not (a polished plan is kept only when it is not slower and not worse in the reachability). The engine calls it on its own when the champion rides within 10 % of the gate, so the S-curve champion (0.51 cm against the 0.50 cm gate) is exactly the case it exists for, and the S-curve still fails |
| **Hypothesis** | It buys nothing on the paths that fail, because the error is the plan's own chord ([RQ-A](research/RQ-A_tracking_error.md)) and re-weighting the samples does not change the reference the controller follows; it may buy a little on the paths that already pass |
| **Experiment** | Solve the six examples with and without `--polish`, recording champion duration, tracking, envelope and the number of solves; the 32-mission suite only if the six say something |
| **Metric** | Verdict flips per mission, duration and tracking delta, extra wall time |
| **Success criterion** | At least one verdict flip or a measured improvement; otherwise delete the path and the JAX dependency it puts on a self-healing solve |
| **Cost** | Low - it is a flag |

### 24.1 Suggested order

```text
  CLOSED (§24.0, do not re-run): Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q10 Q12 Q14
  Q11 executor in the planner, step 1: the documented DriveBase
      acceleration limit into the default digital twin            DONE
  Q13 re-measure the matrix on the shipping planner               (measurement)
  Q16 does the gradient polish change any verdict                 (measurement, one flag)
  Q15 constrain the gate instead of pricing it                    (refinement closed by F6)
  Q11 steps 2-3: acceleration bound, then quantisation            (large)
  Q9  hardware-calibrated twin                                    (needs hardware)
```

The RQ-A items came first because they were the *cause* of the gate failures that everything else is
downstream of: 25 of the 29 failures in the 60-candidate matrix cited the tracking gate, and RQ-A
shows it was measuring a discretisation artefact rather than the robot. KI-026 to KI-028 have landed
and the artefact is now small at the fine end (the S-curve at n=40: 0.98 → 0.27 cm) but not at the
coarse end, where the plan's own deviation is the whole error. The reference's velocity half has
landed too, as the tick-level midpoint lead ([RQ-A finding F8](research/RQ-A_tracking_error.md),
commit `47633e9`); it is not the next fix. **The 5 mm threshold still measures the discretisation
more than the robot**, so it must not be tuned to make missions pass, and the envelope's absolute
limit on a cumulative tube is a separate open issue (KI-054), not a threshold to re-tune.

## 25. How to report a result

Report back in this shape, so results are comparable across experiments.

```text
Experiment:      Q<n> - <one line>
Tree state:      git rev-parse HEAD + the working-tree hash of the changed files
Config:          examples/fll_choreo.json (or another, named)
Missions:        <the mission files, and how they were generated, with n>
Baseline result: duration / envelope / tracking / iterations / wall
New result:      duration / envelope / tracking / iterations / wall
Delta:           <absolute and relative>
Gate:            pass / fail, with the reason
Reproduce:       the exact command
Conclusion:      one sentence, and whether the success criterion was met
```

A new candidate is measured on the 32-mission suite, not on a mission of its own, because the
interesting result is not "the new seed is fast" but "the new seed is fast *and* it wins somewhere".
The tables in [Section 9](HEURISTICS_AND_VERIFICATION.md#9-measured-heuristic-comparison) are the
record for the stock set, not a baseline to beat on the shipping planner (Q13).

---
