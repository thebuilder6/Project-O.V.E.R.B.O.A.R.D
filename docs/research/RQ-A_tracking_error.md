# RQ-A — Why does the closed-loop tracking gate fail?

**Status:** the cause is identified; the structural fix is a package of two changes that must land together. **Two direct bugs found along the way are fixed** (see [Fixes applied](#fixes-applied) below).
**Date:** 2026-10-20
**Tree under test:** pinned, see [`rq_a/PINNED_TREE.md5`](rq_a/PINNED_TREE.md5).
**Companion:** [`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md) (the current-state reference and the research agenda).
**Also:** [`../../research/`](../../research/) — the mission-suite harness for the agenda's Q1 (a Bezier path family), Q2 (a length-proportional time allocation) and Q8 (a 24-mission suite). That work measures the planner across many paths; this note diagnoses why the tracking gate binds on a few. Their results were produced before the KI-025 validator fix and should be re-run against the current tree: the fix moves the tracking error by up to 0.3 mm, and about a sixth of their candidates sit within that margin of a gate.

---

## The question

The 5 mm tracking gate is the binding constraint of the whole planner. In the 60-candidate measurement matrix in the companion document, **36 of the 37 failing candidates failed on it** — the reachability envelope, the thing the interval-propagation machinery exists to compute, never bound once.

Every research question downstream (reachability in the objective, a real digital twin, a better selection rule) assumes the tracking gate is failing because the robot cannot follow the plan. **Is that true?**

## The short answer

**No — at least not in the way the gate measures it.** The tracking error is a *deterministic discretisation artefact*, not an uncertainty problem and not a physical tracking problem. The optimizer plans a trajectory in one discretisation, the controller is told to follow it in a second, and the plant integrates it in a third. On a curved path the three disagree by about a centimetre, and that disagreement is what the gate is measuring.

**The concrete cause:** the optimizer's trapezoidal collocation places the samples *off the path that the trajectory's own speed and turn profile actually traces*. The deployed controller follows a straight chord between consecutive samples, so the robot is asked to be somewhere its own profile never goes.

| mission | heading change per sample | plan vs. its own profile | tracking error | gate |
|---|---:|---:|---:|---|
| `example_straight` | 0.0000 rad | 0.094 cm | 0.346 cm | PASS |
| `example_with_events` | 0.1333 rad | 0.262 cm | 0.435 cm | PASS |
| `example_complex_mission` | 0.1446 rad | 0.129 cm | 0.489 cm | PASS |
| `example_with_stops` | 0.2202 rad | 1.170 cm | 1.814 cm | **fail** |
| `example_s_curve` | 0.3277 rad | **1.206 cm** | **1.260 cm** | **fail** |

On the S-curve the plan/profile discrepancy is 96 % of the reported tracking error. The missions that fail are exactly the missions whose paths turn hard between samples.

*(Measured 2026-10-20, `rq_a/exp9_summary.py`.)*

---

## Method

Everything below was measured against a **pinned copy** of the tree, because the working tree was being edited during the session (KI-024 landed mid-session). The scripts live in [`rq_a/`](rq_a/) and default to the repository root; set `OVERBOARD_TREE` to reproduce the recorded numbers against the pinned tree.

```bash
python docs/research/rq_a/exp9_summary.py          # the summary table above
python docs/research/rq_a/exp1_uncertainty.py      # which uncertainty drives the error
python docs/research/rq_a/exp2_reference_artefact.py
python docs/research/rq_a/exp3_sampling.py         # sampling-density sweep
python docs/research/rq_a/exp4_decompose.py        # toggling the modelling choices
python docs/research/rq_a/exp5_attribution.py      # synthetic self-consistent arcs
python docs/research/rq_a/exp6_consistency.py      # is the optimizer's plan consistent?
python docs/research/rq_a/exp7_recipe.py           # exact positions x sampling density
python docs/research/rq_a/exp8_proposal.py         # the proposed fix, in pieces
python docs/research/rq_a/check_hermite.py         # verifies the spline instrument
```

Three controls were used throughout, because two of the intermediate results were wrong before they were controlled:

1. **A synthetic self-consistent arc** built analytically (positions on a circle, heading = the exact tangent) as a ground truth for "what does the validator report on a trajectory that is internally consistent?"
2. **Instrument verification** — `check_hermite.py` checks the spline interpolant against the exact circle before its results are believed.
3. **Per-criterion gate breakdown** — the tracking criterion is reported separately from motor force and slip, so "the tracking gate failed" is never inferred from `passed=False`.

---

## Findings

### F1 — The tracking error does not depend on the uncertainty

Collapsing each declared interval in turn, on the S-curve (all-uncertainty baseline: 1.260 cm):

| interval active | tracking error | change |
|---|---:|---:|
| `cof` only | 1.259 cm | +0.1 % |
| `torque_headroom` only | 1.259 cm | +0.1 % |
| `gear_backlash_m` only | 1.262 cm | −0.2 % |
| `track_width_range` only | 1.261 cm | −0.1 % |
| `wheel_radius_range` only | 1.259 cm | +0.1 % |
| `inertia_kg_m2` only | 1.259 cm | +0.1 % |
| **all collapsed** | **1.259 cm** | **+0.1 %** |

The same holds on the stops mission (+0.2 %) and the complex mission (−2.3 %).

This is structural, not accidental: `cof`, `torque_headroom`, `wheel_radius_range` and `inertia_kg_m2` are consumed only inside `check_sample_interval` (`immrax_validator.py:460-504`), which is `vmap`ed and never touches the propagated state. Only `track_width_range` reaches the plant (`immrax_validator.py:115, 130`). The backlash reached the plant too, as a velocity offset; it is now a position uncertainty in the reachable set ([Fix 1](#fix-1--the-backlash-was-subtracted-from-the-commanded-wheel-speed)).

**So the tracking gate is not a robustness gate.** It is a deterministic check of the model-versus-plan consistency.

### F2 — Backlash was a velocity loss, and only mattered on straights

**This is now fixed** (see [Fix 1](#fix-1--the-backlash-was-subtracted-from-the-commanded-wheel-speed)); the model as it was:

```python
# the old plant, immrax_validator.py:119-120 before the KI-025 fix
vl_act = vl_cmd - (backlash / 2.0) * jnp.tanh(vl_cmd / 1e-3)
```

is not a position dead-band: for any command above 1 mm/s, `tanh → 1`, so the wheel turns at `command − backlash/2` in **both** directions. With the declared `[0.4, 0.5] mm` the robot ran 0.225 mm/s slower than commanded, which was a pure position lag of `0.225 mm/s × t`.

On a 1 m straight that was the whole tracking error (226 µm, and 0.83 µm with the backlash collapsed). On the S-curve it was 1 % of the error. **The backlash model was worth fixing for its own sake — it should be a dead-band, not a constant offset — but it was not what was failing.**

### F3 — A self-consistent trajectory tracks to 0.16 cm

The control: an arc of radius 0.30 m and length 0.60 m, built analytically so that the positions are on the circle and the heading is the exact tangent.

| n | exact arc | trapezoid positions, tangent headings | heading = chord direction |
|---:|---:|---:|---:|
| 10 | 0.156 cm | 0.209 cm | **4.134 cm** |
| 20 | 0.167 cm | 0.178 cm | **1.920 cm** |
| 40 | 0.168 cm | 0.173 cm | **0.877 cm** |
| 80 | 0.169 cm | 0.170 cm | 0.357 cm |

Two conclusions:

- **The validator, the plant and the controller are not the problem.** A trajectory that is internally consistent is tracked to 0.16 cm at any sampling density, far inside the 5 mm gate.
- **The optimizer's trapezoidal rule is nearly harmless** (0.17–0.21 cm). The rule itself is not the defect.
- **A half-sample heading error is catastrophic** (4.13 cm at n=10, falling as ~1/n).

### F4 — The plan is not on the path its own profile traces

Rebuilding each optimizer plan by *exact* integration of its own `(v, ω)` profile — the same speed profile, the same sample times, the samples placed where the profile actually goes:

| mission | tracking, plan as built | tracking, same profile, exact positions |
|---|---:|---:|
| `example_s_curve` | 1.260 cm | **0.495 cm** (PASS) |
| `example_with_stops` | 1.814 cm | 0.980 cm |
| `example_complex_mission` | 0.489 cm | 0.564 cm |
| `example_straight` | 0.346 cm | 0.433 cm |

On the S-curve, putting the samples where the profile takes them removes **61 %** of the tracking error and flips the gate to PASS.

The deviation itself falls as ~1/n², exactly as a second-order discretisation error should:

| n | 10 | 20 | 40 | 80 |
|---|---:|---:|---:|---:|
| plan-vs-profile deviation | 1.145 cm | 0.219 cm | 0.063 cm | 0.018 cm |
| **tracking error** | **1.260 cm** | **0.479 cm** | **0.958 cm** | **1.072 cm** |

**The deviation collapses; the tracking error does not.** More samples is not the fix — the optimizer simply returns a different trajectory at each density, and the new trajectory has the same problem. (F6 explains why: the error is U-shaped in the density, and the default `-n 10` sits on its coarse side.)

### F5 — With a calibrated model, the **envelope** becomes the binding gate, not the tracking error

This is the counterpart to F1/F3, and it matters because the studio now keeps a calibrated model with the robot and passes `--sysid-fit` on every solve.

The digital twin adds a seventh uncertainty: the measured steady-state **actuator gain**, which multiplies the commanded wheel speeds. A gain band of ±1 % is a speed error of ±1 %, so over a 1 m path the robot travels 10 mm less than it was told to — which reads like a 10 mm tracking error, twice the 5 mm gate.

**It is not.** The controller closes the loop. Measured on straight paths (`rq_a/exp10_gain_band.py`):

| path | gain band | tracking | envelope | tracking gate (5 mm) | envelope gate (20 mm) |
|---:|---|---:|---:|---|---|
| 1.0 m | none | 0.001 mm | 2.9 mm | pass | pass |
| 1.0 m | ±0.1 % | 0.041 mm | 13.1 mm | pass | pass |
| 1.0 m | ±0.5 % | 0.573 mm | 55.0 mm | pass | **fail** |
| 1.0 m | ±1 % | 0.878 mm | 54.0 mm | pass | **fail** |
| 2.0 m | ±0.1 % | 0.319 mm | 45.7 mm | pass | **fail** |
| 2.0 m | ±5 % | 4.568 mm | 91.3 mm | pass | **fail** |

The tracking error stays two orders of magnitude below the gate even at ±5 %, because the Ramsete controller sees the position lag and corrects it. What the gain band destroys is the **reachability envelope**: the interval propagation has to carry the whole speed uncertainty, and the tube blows past 20 mm at a gain band of 0.5 % over 1 m.

**So the two gates measure two different things, and which one binds depends on the configuration:**

| configuration | binding gate | why |
|---|---|---|
| default (no `--sysid-fit`) | **tracking** (36 of 37 failures) | a discretisation mismatch, not an uncertainty effect |
| with a fitted model | **envelope** | the measured gain band widens the reachable set |

This also explains the measurement-suite result that the envelope was 27 % of the failures there and 0 % on the six example missions: the example missions are easy, and the suite includes hard ones.

**Practical consequence.** A calibrated model makes most paths fail validation — but on the envelope, not on the tracking error, and the open-loop "10 mm over 1 m" arithmetic is not the reason. The reachable set cannot be shrunk by making the trajectory smoother; it can only be shrunk by a narrower gain band, a shorter path, or by deciding that the envelope gate and the tracking gate are measuring overlapping things and merging them.

### F6 — The tracking error is U-shaped in the sampling density, and the default is on the wrong side of the optimum

`exp10_floor.py` holds the plan fixed and toggles one discretisation axis at a time. Two axes move the number, and they dominate at opposite ends of the density range:

- **coarse** (few control ticks per sample interval): the plan's positions are off the path its own `(v, ω)` profile traces. Replacing them with the exact integral takes the S-curve from 1.248 cm to **0.494 cm — a pass**.
- **fine** (one to three ticks per interval): the 20 ms command hold is comparable to the sample interval, so the command is stale relative to the reference the controller is following. Refining the tick to 2 ms takes the S-curve at n=40 from 0.979 cm to **0.048 cm — a pass**.

Everything else is small: the plant's substep count (0.94–1.01×), the command quantiser (0.98–1.00×), the Ramsete gains (0.57–1.57×, the widest of the rest). And telling the controller the *chord* heading instead of the plan's heading is catastrophic (43–63×), which is the plan-inconsistency term again, from the other side.

Putting the two together maps the whole curve (`exp11_density.py`, S-curve, per-segment timing):

| n | dt | ticks/interval | shipped | exact plan | 2 ms tick | **both** |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 414 ms | 20.7 | 7.414 | 4.893 | 8.276 | 5.744 |
| 10 | 181 ms | 9.1 | 1.248 | **0.494 ✓** | 1.727 | 1.017 |
| 15 | 118 ms | 5.9 | **0.347 ✓** | 0.562 | 0.755 | 0.476 ✓ |
| 20 | 86 ms | 4.3 | 0.491 | 0.676 | 0.285 | 0.126 |
| 30 | 57 ms | 2.8 | 0.799 | 0.901 | **0.107 ✓** | **0.034 ✓** |
| 40 | 44 ms | 2.2 | 0.979 | 1.012 | **0.048 ✓** | **0.074 ✓** |
| 60 | 30 ms | 1.5 | 0.973 | 1.000 | **0.066 ✓** | **0.114 ✓** |

**(✓ = passes the 5 mm tracking gate.)**

Three things follow, and they change the plan:

1. **The shipped error is not monotone in `n`.** It is U-shaped with a minimum near 4–6 control ticks per sample interval. **"Sample more finely" makes the robot track worse**, and the default `-n 10` (9 ticks per interval on the S-curve) is on the coarse side of the optimum. Any earlier conclusion that "more samples would help" was reading the wrong half of the curve.
2. **The two fixes are complementary, not alternatives.** The exact-plan fix helps where the ticks are many (n ≤ 15) and hurts where they are few; the finer-tick fix is the mirror image. **Both together give 0.034 cm at n=30** — the best result in the table, and 37× better than the shipped 1.248 cm.
3. **The single highest-value change at the current default is the optimizer's kinematics.** At n=10 the exact-plan fix alone turns the S-curve from a fail into a pass (1.248 → 0.494 cm). The stops mission improves from 1.807 to 0.973 cm on the same change — it still fails, but by half.

The finer tick is a property of the *model*, not of the robot: the deployed firmware runs a 20 ms loop and the validator mirrors it. The honest reading is that the plan's sampling density has to be **matched to the controller's tick** — roughly 4 to 6 ticks per sample interval — and the plan's positions have to be the exact integral of its own profile. A 2 ms tick in the validator is a way of *measuring* how much the density mismatch costs; it is not a fix you can ship without changing the firmware loop.

---

## Hypotheses that were tested and refuted

These are recorded so they are not re-run. Each was a reasonable reading of the evidence at the time.

| Hypothesis | Refuted by | What the truth turned out to be |
|---|---|---|
| "The tracking gate fails because the uncertainty is too wide." | F1: collapsing all seven intervals moves the error by 0.1–0.2 % | It is a deterministic model-versus-plan mismatch |
| "Backlash is the driver." | F1: collapsing it moves the S-curve error by 1 % | Backlash is a straight-path lag only (F2) |
| "The deg/s command quantisation is the driver." | `exp4`: removing quantisation moves the error by ≤ 0.5 % | It is a real dead-band, but a small contributor here |
| "More samples fix it." | F4: the deviation falls 1/n² but the tracking error does not | The optimizer returns a different trajectory at each density |
| "The validator's reference interpolation is the bug." | `exp8`: a spline reference on the plan as built gives 26–38 cm and a 125–190 cm envelope | The reference is faithful to the firmware; the *plan* is what is inconsistent |
| "The optimizer's trapezoidal rule is the bug." | F3: the trapezoid tracks to 0.17–0.21 cm | The rule is fine; the plan's *headings* are not the tangents of its path |
| "The tracking metric is measuring robot tracking error." | F3: a self-consistent arc tracks to 0.16 cm | It measures how far the plan is from a path the controller can follow |

### Instrument bugs hit along the way

Recorded because two of them produced convincing nonsense that briefly looked like a finding:

| Bug | Symptom | Fix |
|---|---|---|
| Hermite ω used the arc-length rate instead of the parameter rate | ω 16× too small; the spline "failed" with 20–47 cm | `dθ/dt = (x'y'' − y'x'')/(x'²+y'²) / dt` |
| `atan2` wraps the heading to (−π, π] while the plan's headings are continuous | multi-radian heading error at the wrap | anchor the branch to the sample's heading |
| Curvature divided by the curve speed, which is **zero** at the start/end samples (the robot is at rest) | infinite ω, envelope divergence | fall back to the plan's profile on a degenerate interval |
| The Ramsete gain `kr = sqrt(ω² + b·v²)` taken from the plan's linear interpolation while `v, ω` came from the spline | wrong drive time constant | recompute `kr` from the reported `v, ω` |

**Verify the instrument against ground truth before believing a large effect.** `check_hermite.py` now does this for the spline: it reproduces the exact circle to 0.0012 mm (the chord's sagitta is 1.5 mm) and reports ω = 1.3378 rad/s against a true 1.3333.

---

## Fixes applied

Two of the findings were **direct bugs in the code**, not design gaps, and are now fixed (KI-025). The structural finding — the three-way discretisation mismatch — is **not** fixed; it is the open item below.

### Fix 1 — the backlash was subtracted from the commanded wheel *speed*

```python
# before
vl_act = vl_cmd - (backlash / 2.0) * jnp.tanh(vl_cmd / epsilon)
```

`gear_backlash_m` is a distance in metres; `vl_cmd` is a velocity in m/s. Subtracting one from the other is a **dimensional error**. Two consequences: the robot ran permanently `backlash/2` slower than commanded in both directions (the `tanh` saturates above 1 mm/s), and that deficit grew into a position lag of `backlash/2 × t` which the validator reported as a *tracking error*.

A gear backlash is a **position dead-band**: taken up once at a direction reversal, after which the wheel turns at exactly the commanded speed. The model had the speed wrong and the position wrong.

**The fix:** the plant turns the wheel at the commanded speed, and the backlash is carried in the reachable set, where a position uncertainty belongs:

```python
# after (immrax_validator.py, compute_reachability)
spread = backlash_half * (s_next['t'] - samples[0]['t'])
env = {"x_min": env["x_min"] - spread, "x_max": env["x_max"] + spread,
       "y_min": env["y_min"] - spread, "y_max": env["y_max"] + spread}
```

**The gate is not made more permissive.** The worst-case total uncertainty at each sample is unchanged: the same magnitude the plant used to bias the tracking error by, moved into the envelope and made symmetric instead of one-sided.

Measured on a synthetic straight, backlash on versus off:

| path length | tracking difference | envelope difference |
|---|---:|---:|
| 0.5 m | **0.0000 mm** | +0.348 mm |
| 1.0 m | **0.0000 mm** | +0.679 mm |
| 2.0 m | **0.0000 mm** | +1.310 mm |

Measured over the **whole 60-candidate matrix** (before → after the fix):

| quantity | mean change | range |
|---|---:|---|
| tracking error | −0.0008 cm | −0.025 to +0.029 cm |
| envelope radius | **+0.0070 cm** | −0.058 to +0.061 cm |
| tracking + envelope | +0.0062 cm | −0.066 to +0.063 cm |
| **pass flags changed** | **0 of 60** | — |

The tracking error barely moved (it is dominated by the discretisation mismatch, not the backlash), the envelope grew by the backlash that used to be in the tracking number, and **no candidate's gate outcome changed**. The worst-case total per run moved by at most 0.065 cm, against a 5 mm tracking threshold and a 2 cm envelope threshold.

Reproduce with `rq_a/verify_backlash_fix.py`; the behaviour is pinned by `tests/test_validator_model.py`.

### Fix 2 — the tracking metric ignored the heading

The metric is position-only (`immrax_validator.py:740-741`), so a trajectory could follow the path while arriving at the wrong heading, and nothing measured it. `compute_reachability` now reports `max_heading_error_rad`, and the machine-readable validation report carries `heading_error_deg` per candidate.

It is **reported, not gated**: the heading behaviour is not calibrated against hardware, so it must not become a pass/fail criterion on its own. It is the measurement OQ4 needs.

### Not fixed — the structural mismatch

The three-way discretisation inconsistency is untouched. It needs the package fix described above, and the open item (OQ1) has to be closed first.

---

## The proposal, and why it is a package

The fix has two halves and **neither works alone**.

**Half 1 — make the plan self-consistent.** The optimizer's samples should lie on the path the trajectory's own `(v, ω)` profile traces, and the heading samples should be the tangents of that path. Today the trapezoidal collocation does not enforce this, and nothing else in the objective does either. Effect on its own: 61 % less tracking error at n=10 on the S-curve, and *worse* at n=20–80.

**Half 2 — interpolate the reference as a curve through the samples.** The deployed controller (`pybricks_code/ramsete_fast.py`) and the validator (`immrax_validator.py:46`) both interpolate `x, y, θ, v, ω` **linearly**, i.e. they follow the chord. A cubic Hermite through the samples reproduces the true path to 0.0012 mm. Effect on its own: catastrophic — 26 cm on the S-curve, with the interval tube diverging to 125 cm.

**Together:** the reason half 2 fails is precisely that half 1 is missing. A Hermite through samples whose headings are not their tangents cannot be both through the points and tangent to them, so it develops curvature spikes, and the closed loop diverges.

The measurement in `exp8` that isolates this:

| variant | S-curve | stops | complex | straight |
|---|---:|---:|---:|---:|
| 1. as shipped | 1.260 cm | 1.814 cm | 0.489 cm | 0.346 cm |
| 2. self-consistent plan | **0.495 cm** | 0.980 cm | 0.564 cm | 0.433 cm |
| 3. spline ref on plan as built | 26.0 cm | 38.3 cm | 6.0 cm | 0.439 cm |
| 4. self-consistent + spline ref | 12.9 cm | 1.148 cm | 21.5 cm | 0.441 cm |

Variant 4 is *worse* than variant 2, so "self-consistent plan + spline" is **not** sufficient either. The self-consistency produced by rebuilding the positions does not make the plan's *heading samples* tangents — the rebuilt heading is the integral of ω, but the residual heading inconsistency at the stops (where the chord degenerates and the direction reverses) survives, and the spline amplifies it.

**This is the open part of RQ-A:** find a plan formulation in which the samples, the headings and the profile are mutually consistent *by construction*, including through stops and direction reversals. That is a change to the optimizer's kinematics and constraints, not to the validator.

---

## Open questions

| # | Question | Why it matters | Next step |
|---|---|---|---|
| **OQ1** | What sets the floor of ~0.5–1.1 cm at n=40–80 on the S-curve? The plan/profile deviation is 0.063 cm there, so 95 % of the error is unexplained. | Until this is answered, no fix can be shown to work. | **CLOSED (F6).** It is the 20 ms command hold: refining the tick to 2 ms takes the error from 0.979 cm to 0.048 cm at n=40. At coarse sampling the floor is the plan/profile mismatch instead, and the error is U-shaped in the density |
| **OQ2** | Can the optimizer be reformulated so that consistency is structural? | It is the only fix that addresses the root cause. | Replace the trapezoidal collocation with a higher-order quadrature over the *interpolated* heading, so `x, y` are the exact integral of `v·cos θ(t)`. Then the samples lie on the path by construction and the headings are the tangents. |
| **OQ3** | Should the reference become a spline in the validator **and** the firmware? | The firmware and the validator must agree, or the parity is lost. | `pybricks_code/ramsete_fast.py` and `ramsete_compact*.py` interpolate linearly today. Any change there is a firmware change and needs the plan fix first. |
| **OQ4** | Does the tracking metric need a heading term? | The metric is position-only (`immrax_validator.py:740-741`). A plan can be followed in position while the heading is off by up to Δθ/2 — 9.4° on the S-curve at n=10. | Measure the heading error the closed loop actually shows, and decide whether it is a gate, a reported metric, or both. |
| **OQ5** | What is the stops mission's extra 0.6 cm? (1.17 cm deviation but 1.81 cm tracking.) | Stops are where the discretisations interact worst: the robot reverses, the backlash sign flips, the chord degenerates. | Separate the reversals from the curvature. Build a variant of the stops mission with the stops removed but the same geometry. |
| **OQ6** | Does any of this change the seed conclusions? | The companion document's headline is that the seeds buy gate insurance, not speed. | Probably not, but re-run the 60-candidate matrix once the discretisation is fixed. The win counts may change, because the gate is what the seeds are escaping. |

---

## What this changes in the research agenda

In [`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md) Part IV:

- **Q4 (reachability in the objective) is not the right first lever.** The failing missions fail on a deterministic discretisation mismatch; an envelope term would not touch it. Revise the order: consistent discretisation first, reachability term second.
- **Q9 (a real digital twin) is still needed, but it is not what is failing today.** The tracking error is already ~100 % explained by the current model. Adding motor dynamics, odometry and slip will not move the number that is blocking the missions. Q9 is about *honesty* about the envelope, not about the tracking gate.
- **The 5 mm threshold is not currently meaningful.** It is being compared against a number that is dominated by a modelling artefact. Until the discretisation is consistent, a pass or a fail says more about the discretisation than about the robot. **Do not tune the threshold to make missions pass** — that would hide the defect.
- **Q6 (does the accuracy weight buy accuracy?) should be re-run after the fix**, because the weight's effect on the tracking error may itself be an artefact.
- **Q9 (a real digital twin) is not only about honesty about the envelope — it moves the binding gate.** With a fitted model the gain band widens the reachable set until the *envelope* gate binds (F9), which is a different failure from the one the tracking gate reports today. A calibrated robot is a different problem, not the same problem with better numbers.
- **Q4 (reachability in the objective) therefore has two audiences.** With the default model the envelope is never the binding constraint, so an envelope term in the objective buys nothing on the six example missions. With a calibrated model it is the *only* binding constraint. Whichever configuration the project settles on decides whether Q4 is urgent.

---

## Status summary

| | |
|---|---|
| Question | Answered: the cause is identified, and the open item is closed |
| Root cause | The optimizer's discretisation, the controller's reference and the plant's integration are three different discretisations of the same trajectory; on a curved path they disagree by ~1 cm |
| OQ1 (the floor at fine sampling) | **Answered.** It is the 20 ms command hold: at n=40, refining the tick to 2 ms takes the error from 0.979 cm to 0.048 cm. At coarse sampling the floor is the plan/profile mismatch instead, and the error is U-shaped in the density (F6) |
| Fix | **Two direct bugs fixed** (the backlash dimensional error, and the missing heading metric). The **structural fix is now precise**: (1) the optimizer's positions must be the exact integral of the trajectory's own `(v, ω)` profile, and (2) the sampling density must be matched to the controller's 20 ms tick — roughly 4 to 6 ticks per sample interval. Neither works alone; together they give 0.034 cm on the S-curve, against 1.248 cm shipped |
| Confidence | High on the diagnosis (four independent measurements agree, with synthetic controls). Medium on the fix: the two halves have been shown to work *in the model*, and the second one needs a firmware change or a density policy to ship |
| Safety | The two fixes changed no gate outcome: 0 of 60 candidates changed pass flag on the example matrix, and 0 of 231 on the mission suite, with the worst-case uncertainty per run moving by at most 0.065 cm |
| Which gate binds | **Tracking** in the default configuration (34 of 36 failures). **Envelope** once a calibrated model is fitted, because the measured gain band widens the reachable set (F5) |
| Hardware | **None.** Every number here is a model prediction. No hardware logs exist in this repository, and the uncertainty ranges in `examples/fll_choreo.json` are declared test assumptions. The *relative* findings (which term dominates, how the error scales) are properties of the model and hold regardless; the absolute numbers do not |

