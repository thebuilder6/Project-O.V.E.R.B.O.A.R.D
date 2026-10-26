# The research agenda

**Status:** 2026-10-21, as measured on that tree. Part IV of the heuristics document.
M3 (2026-10-25) closed Q15 and Q11 step 2; their records are in §24.0 and the
Q11/Q15 entries below. Q17 (topology and geometry candidates) is pre-registered
2026-10-26 and unmeasured; its entry is below Q15.
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
  - [Q15 — Should the tracking gate be a constraint instead of a price?](#q15--should-the-tracking-gate-be-a-constraint-instead-of-a-price)
  - [Q17 — Do topology and geometry candidates buy gate coverage?](#q17--do-topology-and-geometry-candidates-buy-gate-coverage)
  - [24.1 Suggested order](#241-suggested-order)
- [25. How to report a result](#25-how-to-report-a-result)

## 24. The open questions

Seventeen questions are on the record: sixteen were asked and measured (fifteen closed, Q9 open), and
Q17 is pre-registered (2026-10-26), unmeasured. Each open one has the
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
| Q14 | how much risk should a plan carry? | **Not a choice between candidates.** On the 32-mission suite (8 per profile, `--candidates 3 --validate --propagation plant`) 10 missions pass, and on every one of them all three candidates pass and no rescue candidate ever does: the set passes or fails as a block. Inside a mission the three differ by 0.89 % in duration (max 1.68 %) and 0.005 of normalised margin (max 0.013), so a margin floor never changes the pick - it only drops missions, and the balanced rule moves 2 of 10 for +0.001 s. A passing plan runs at 54 % of its accuracy limit (mean margin 0.46); the 161 failing candidates need the limit multiplied by 2.9 (median), 14.7 (p90) and 34 (worst), and 49 also break the 5 mm tracking gate. Force is not a dial: every candidate sits at 0.84 of the motor limit. **The risk is in the limit, not in the pick**, so there is nothing for a `--risk` flag to choose between. Measured on the **stock** candidate sets, `plant` only; on the shipping planner the same suite passes 5 of 32 with **one** passing candidate per mission, never three (Q13) | `research/results/q14_plant.jsonl`, `research/risk_margins.py`, `tests/test_risk_margins.py` |
| Q13 | should the measured matrix be re-measured on the unified path? | **Re-measured, and the stock matrix does not carry over.** The shipping planner at `--candidates 3 --validate` and the default limits passes **5 of 32 missions under `plant` and 6 under `tick`** (the five plus m008), 96 candidates solved per arm, 86 checked / 10 skipped and 84 / 12. The stock sets passed 10 of 32, but with 3-8 candidates per mission - a different candidate family, so this is not a regression measurement, it is the answer to "what does the planner that ships actually do". **Every passing mission has exactly one passing candidate**, never two of three. Of the 27 plant failures, 9 are within 0.5 cm of the local limit, 3 within 1-2 cm and 15 beyond 2 cm; 11 of the selected plans were cut off at the validator's guard, so their envelope is a lower bound (KI-049). The two models disagree on one mission and in the counter-intuitive direction - `tick` passes m008 and `plant` does not - because the closed-loop map contracts the tube instead of widening it (KI-049, KI-050) | `research/results/q13_plant.jsonl`, `q13_tick.jsonl`, `research/cli_matrix.py`, `research/q13_report.py` |
| Q16 | does the gradient polish change a verdict? | **No, on any plan measured.** The six examples and the five nearest-miss missions from Q13 (m012, m020, m023, m026, m030) solved with and without `--polish`: **zero verdict flips and byte-identical candidate reports** in all 11, for 2-7x the wall time (the complex mission 11.4 s -> 86.2 s). A probe on the polish itself shows why: on a converged plan the two rounds are run and the plan comes back unchanged (`in_duration == out_duration`, `changed=False`), and the engine records `polish: {rounds: 2}` whether or not a round was kept. The path's success criterion is not met, so the delete branch is the evidence-backed consequence; it is a separate change (it removes `gradient_polish`, `_replan`, `_sensitivity_weights`, `_polish_key`, the flag and the JAX requirement it puts on a self-healing solve) and has not been taken | `research/results/q16_off.jsonl`, `q16_on.jsonl`, `q16_near_off.jsonl`, `q16_near_on.jsonl` |
| Q15 | should the tracking gate be a constraint instead of a price? | **No - the arm is retired (2026-10-25).** The interval inequality `sagitta^2 + parab^2 + 0.25*hold^2 <= disc_margin^2` (margin 0.9 of the 5 mm gate), written on the shipping mesh next to the priced arm, flips **zero verdicts**: the 8 examples pass 3/8 in both arms and the 32-mission suite (m000-m031, 8 per profile) passes 6/32 in both arms, with the same missions passing (m010, m015, m017, m021, m028, m031). **Zero infeasible solves of 240 candidates** - the inequality never became the binding constraint, because the priced arm already satisfies it: on a 6-mission probe the worst interval term is inside `disc_margin^2` on 5, and on the one mission where it is not (m001, term 1.05 vs 0.81) the constraint merely re-times the plan by +0.5 % and the mission still fails on the 16 cm envelope, which no 5 mm discretisation term can move. Duration cost of the arm: 0.01 % mean (2.3231 s -> 2.3228 s selected), iterations -2 %, wall +1 %. Success criterion "more missions pass at a stated duration cost" is not met; the experimental rows are removed from the optimizer, `w_disc` and the priced arm stay | `research/results/m3_examples_priced_accel.jsonl`, `m3_examples_disc_accel.jsonl`, `m3_suite_priced_accel.jsonl`, `m3_suite_disc_accel.jsonl`, `research/m3_matrix.py` |
| Q11 | (step 2, 2026-10-25) does the wheel-acceleration bound change the plans? | **The bound lands and binds where it must, and the shipping suite never trips it.** With `digital_twin.actuator.straight.accel_limit` the optimizer now holds `-a*dt <= dv <= a` per wheel per interval (see Q11 below). The synthetic 0 -> 0.6 m/s in 20 ms request (30 m/s^2) is refused: the returned plan peaks at <= 7.8 m/s^2. On the 32-mission suite and the 8 examples **no candidate was over the ramp limit before or after** (0/96 and 0/24; worst observed 4.94 m/s^2), so gate pass rate is unchanged (6/32, 3/8) and the duration cost is 0.2 % (mean selected 2.3271 s -> 2.3231 s). The F10 population of over-limit plans does not occur on the shipping candidate set; the bound is the guarantee that it cannot, not a change to today's plans | `research/results/m3_suite_priced_accel.jsonl`, `m3_suite_priced_noaccel.jsonl`, `tests/test_optimizer.py` (the accel-bound tests) |

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
| **Current evidence** | Steps 1 and 2 are landed and measured (2026-10-25). Step 1: `examples/fll_choreo.json` declares `digital_twin.actuator.straight.accel_limit = 7.8`, and a declared limit alone selects the five-state plant (`tests/test_digital_twin.py`). Step 2: the optimizer bounds the wheel acceleration per interval (`-a_limit*dt_k <= vl[k+1]-vl[k] <= a_limit*dt_k`, same for `vr`), reading the one declared limit through `RobotConfig.accel_limit_m_s2` (`overboard/models/robot_model.py`, parsed with `DigitalTwinConfig.from_dict` - no second config path). A config with no declared limit gets no bound and behaves exactly as before. Measured with `research/m3_matrix.py` on the 8 examples and the 32-mission suite (`--candidates 3`, `plant`, `examples/fll_choreo.json`): the synthetic 0 -> 0.6 m/s in 20 ms request (30 m/s^2) is refused (returned plan <= 7.8 m/s^2), and on the real matrix **0 of 96 suite candidates and 0 of 24 example candidates exceed the ramp limit before or after** (worst observed 4.94 m/s^2), so pass rate is unchanged (6/32 suite, 3/8 examples) and mean selected duration moves 2.3271 -> 2.3231 s. The F10 over-limit population does not exist on the shipping candidate set. Iterations +4 %, wall +1 %; `research/results/m3_suite_priced_accel.jsonl` vs `m3_suite_priced_noaccel.jsonl` |
| **Hypothesis** | Step 3 (quantisation) is the remaining gap: the commanded speed is a step function the plant smooths, and a penalty for the quantisation error is the only smooth form |
| **Experiment** | 1. ~~Documented limit in the digital twin.~~ **Landed** (2026-10-21). 2. ~~Bound wheel acceleration in the optimizer.~~ **Landed and measured** (2026-10-25): the bound refuses over-limit requests and costs 0.2 % duration; the shipping suite never trips it, so it is a guarantee, not a change to today's plans. 3. Quantisation: price the step error in the objective, and measure whether the gate sees the difference |
| **Metric** | Gate pass rate, the fraction of passing plans that exceed the hub's limit, tracking error, duration |
| **Success criterion** | (step 2, met) plans the hub cannot execute stop being planned, at a stated duration cost - the synthetic 30 m/s^2 request is refused; (step 3) same shape as step 2 |
| **Cost** | High for step 3 (the step function needs a smooth surrogate) |

### Q15 — Should the tracking gate be a constraint instead of a price? **(closed 2026-10-25: no, arm retired)**

| | |
|---|---|
| **Question** | The shipping planner prices the three discretisation mechanisms in the objective (`cost_metric="discretization"`) and lets the validator reject afterwards. Should the gate be an inequality on the mesh instead of a price? |
| **Answer** | **No. Measured and retired (2026-10-25).** The opt-in arm (`disc_constraints=True`, `disc_margin=0.9`) wrote the approved interval inequality `sagitta_err^2 + parab_err^2 + 0.25*hold_err^2 <= disc_margin^2` per interval on the shipping mesh, next to the untouched priced arm (`w_disc` stayed on). It flips **zero verdicts**: the 8 examples pass 3/8 in both arms (straight, complex_segment, mission_first4) and the 32-mission suite (m000-m031, 8 per profile) passes 6/32 in both arms (m010, m015, m017, m021, m028, m031). **Zero infeasible solves of 240 candidates.** Why: the priced arm already satisfies the inequality. On a 6-mission activity probe the worst interval term sits inside `disc_margin^2 = 0.81` on 5 (0.07-0.33); the one mission where it exceeds (m001, 1.05) is a 16 cm envelope failure, and the constraint only re-times the plan there by +0.5 % (14.125 -> 14.191 s sprint) without changing the verdict. Cost of the arm: mean selected duration 2.3231 -> 2.3228 s (0.01 %), iterations 7117 -> 6979 (-2 %), wall 434.8 -> 438.6 s (+1 %). The success criterion ("more missions pass at a stated duration cost") is not met, so the constraint rows are removed from the optimizer again and `w_disc` stays. The refinement half was already closed by RQ-A F6: density is not the lever |
| **Records** | `research/results/m3_examples_priced_accel.jsonl`, `m3_examples_disc_accel.jsonl`, `m3_suite_priced_accel.jsonl`, `m3_suite_disc_accel.jsonl`; `research/m3_matrix.py` drives the library candidate loop and the production gate chain (`_record_candidate` / `_score_attempt`) since the CLI has no flag for this arm |
| **Reproduce** | `python research/suite.py --count 32 --seed 7 --out research/missions`, then `python research/m3_matrix.py research/missions examples/example_straight.json examples/example_s_curve.json examples/example_complex_segment.json examples/example_with_stops.json examples/example_complex_mission.json examples/example_complete.json examples/example_with_events.json examples/example_mission_first4.json --out research/results/m3_suite_disc_accel.jsonl --disc --accel` (the `--disc` flag existed on the measurement tree only; the arm is retired on this tree) |
| **Do not re-run** | The measurement is a null result on the shipping mesh and the shipping candidate set. A future inequality would need a different margin, a different mesh or a different gate - new evidence, not this experiment again |

### Q17 — Do topology and geometry candidates buy gate coverage? **(pre-registered 2026-10-26, unmeasured)**

| | |
|---|---|
| **Question** | Do candidates that differ in *topology* (reverse segments) and *geometry* (a real path family, a mirrored bow) pass missions the frontier triad cannot? |
| **Current evidence** | The triad is not diverse: on the 32-mission suite every passing mission has exactly one passing candidate (Q13, `--candidates 3 --validate`, default limits: 5/32 `plant`, 6/32 `tick`), and the three differ by 0.89 % in duration (max 1.68 %) and 0.005 of normalised margin, so the set passes or fails as a block (Q14). Q1 (F8, 64 matched pairs on the pq_b tree) measured a cubic Bezier family (`research/seeds.py`, handed in through `initial_guess`): mean +3.0 % slower - **not a speed lever** - but 33 vs 48 IPOPT iterations and 15 vs 11 gate passes, a gate-insurance ticket; the solver walks the curve back toward the straight line because the objective has no term that prefers a curve (HEURISTICS §2, STRATEGY_LEDGER). Q3 closed multi-start *within one straight-line geometry* ("the whole seed axis buys nothing"); it did not measure differently-shaped or reverse seeds. Topology search is not in the tree at all (HEURISTICS §2), but the NLP already admits reverse motion (`overboard/core/optimizer.py`: `vl >= -v_bound`; `tests/test_optimizer.py` exercises the accel bound in reverse) - the gap is the seeds and the candidate set, not the constraints. |
| **Hypothesis** | Candidates that change the *geometry* and the *topology* of the guess land in different NLP basins and break the pass/fail-as-a-block behaviour, buying gate coverage at a stated duration cost. Per F8's mechanism, a shaped seed without a geometric term is walked back toward the straight line; if the new candidates collapse to one path, they need the anchor term of stage 2 to stay distinct. |
| **Experiment** | 1. Extend `overboard/core/frontier_seed.py` styles with `apex_sign: ±1` (mirrored bow), `geometry: straight|bezier` (promote `research/seeds.py:spline_guess` into `overboard/core/` - research code may not be imported by `overboard/`), and reverse seeding (`v < 0`, heading + pi on a segment whose pinned heading is more than 120° off the chord). Append three candidates after the triad (`frontier_sprint_mirror`, `frontier_spline`, `frontier_reverse`); the default solve set stays the first two candidates, so nothing new runs unless `--candidates` asks for it. 2. Measure with `research/m3_matrix.py` (it drives the library candidate loop and the production gate chain): baseline `--candidates 3` against expanded `--candidates 6`, `--validate --propagation plant`, on the same 32 missions. |
| **Metric** | Missions passed (primary); mean selected duration; mean pairwise candidate deviation (the collapse diagnostic); IPOPT iterations and wall time. Report per [§25](#25-how-to-report-a-result). |
| **Success criterion** | At least 2 more missions pass at no more than 5 % mean selected-duration cost, and the new candidates are geometrically distinct from the triad (mean pairwise deviation above 2 cm). If the collapse diagnostic fires, stage 2 - a geometric anchor term `w_anchor * ||p - p_seed||^2` in `_compute_cost`, priced like the discretisation term - is measured the same way before any conclusion is drawn. |
| **Cost** | Medium: seed and candidate code plus one matrix run (32 missions x 2 arms); stage 2 touches the objective. No hardware needed. |
| **Reproduce** | `python research/suite.py --count 32 --seed 7 --out research/missions`, then `python research/m3_matrix.py research/missions --out <arm>.jsonl --candidates 3 --propagation plant` and the same with `--candidates 6` on the change tree. |
| **Status** | **open** - pre-registered 2026-10-26, unmeasured. Deliberately not a re-run of Q1 (which measured the family as a *speed* lever) or Q3 (which measured same-geometry seeds); a retry after a null result needs new evidence, per [§24.0](#240-the-closed-questions---do-not-re-litigate). |

### 24.1 Suggested order

```text
  CLOSED (§24.0, do not re-run): Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q10 Q12 Q13 Q14 Q15 Q16
  Q17 topology/geometry candidate axis (pre-registered)     NEXT - cheap, no hardware
  Q11 executor in the planner, step 1: the documented DriveBase
      acceleration limit into the default digital twin            DONE
  Q11 step 2: the optimizer's wheel-acceleration bound            DONE (2026-10-25)
  Q11 step 3: quantisation (a smooth penalty for the step)        (large)
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
