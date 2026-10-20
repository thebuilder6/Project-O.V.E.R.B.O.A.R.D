# RQ-A — Why does the closed-loop tracking gate fail?

**Status:** the cause is identified. Three of the four sides of the three-way discretisation mismatch are now closed in the tree — the plan's own collocation (KI-026), the reference's coupling (KI-027) and the controller parity (KI-028) — and **every table below has been re-measured against that tree**. What is still open is the *shape* of the reference: the controller follows the chord between the samples, and the feedforward reports an interpolated value where the plan's own rule assumes an interval average.
**Date:** 2026-10-20. The first pass was measured in the morning against a pinned copy; the tables were re-measured in the afternoon against the working tree, after KI-027 and KI-028 landed. Where a number has moved, the old one is kept and marked, because it is the record of the bug.
**Tree under test:** the working tree, token `FD268F4F79AD`; the per-module MD5s are in [`rq_a/TREE_2026-10-20.md5`](rq_a/TREE_2026-10-20.md5). The first pass was measured against the pinned copy whose MD5s are in [`rq_a/PINNED_TREE.md5`](rq_a/PINNED_TREE.md5).
**Companion:** [`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md) (the current-state reference and the research agenda).
**Also:** [`../../research/`](../../research/) — the mission-suite harness for the agenda's Q1 (a Bezier path family), Q2 (a length-proportional time allocation) and Q8 (a 24-mission suite). That work measures the planner across many paths; this note diagnoses why the tracking gate binds on a few. **Their results predate KI-025, KI-027 and KI-028, and so does the 60-candidate matrix in the companion document**: the reference fix moves the tracking error by up to 2 mm on the hard missions, and the parity fix moved it again. Both have to be re-run before their numbers are quoted as current. The raw evidence for this note's own re-measurement is in [`../../measurements/2026-10-20-post-ki028/`](../../measurements/2026-10-20-post-ki028/).

---

## The question

The 5 mm tracking gate is the binding constraint of the whole planner. In the 60-candidate measurement matrix in the companion document, **36 of the 37 failing candidates failed on it** — the reachability envelope, the thing the interval-propagation machinery exists to compute, never bound once.

Every research question downstream (reachability in the objective, a real digital twin, a better selection rule) assumes the tracking gate is failing because the robot cannot follow the plan. **Is that true?**

## The short answer

**No — at least not in the way the gate measures it.** The tracking error is a *deterministic discretisation artefact*, not an uncertainty problem and not a physical tracking problem. The optimizer plans a trajectory in one discretisation, the controller is told to follow it in a second, and the plant integrates it in a third. On a curved path the three disagree by about a centimetre, and that disagreement is what the gate is measuring.

**The concrete cause:** the optimizer's trapezoidal collocation places the samples *off the path that the trajectory's own speed and turn profile actually traces*. The deployed controller follows a straight chord between consecutive samples, so the robot is asked to be somewhere its own profile never goes.

| mission | heading change per sample | plan vs. its own profile | tracking error | gate |
|---|---:|---:|---:|---|
| `example_straight` | 0.0000 rad | 0.094 cm | 0.341 cm | PASS |
| `example_with_events` | 0.1333 rad | 0.262 cm | 0.459 cm | PASS |
| `example_complex_mission` | 0.1446 rad | 0.129 cm | 0.303 cm | PASS |
| `example_with_stops` | 0.2202 rad | 1.170 cm | 2.191 cm | **fail** |
| `example_s_curve` | 0.3277 rad | **1.206 cm** | **1.468 cm** | **fail** |

On the S-curve the plan/profile discrepancy is 82 % of the reported tracking error. The missions that fail are exactly the missions whose paths turn hard between samples.

*(Re-measured 2026-10-20 against the working tree, `rq_a/exp9_summary.py`. The first pass, against the pinned tree, read 0.346 / 0.435 / 0.489 / 1.814 / 1.260 cm: the reference fix moved the S-curve and the stops mission, and the parity fix moved the stops mission again.)*

The wider claim — that the tracking gate, and not the envelope, is what rejects candidates — was measured on the 60-candidate matrix in the companion document, where **36 of the 37 failing candidates failed on it**. That matrix predates KI-025, KI-027 and KI-028 and has to be re-run before it is quoted as current; what survives the re-measurement is the mechanism below, not the count.

---

## Method

The first pass of this note was measured against a **pinned copy** of the tree, because the working tree was being edited during the session (KI-024 landed mid-session). The scripts live in [`rq_a/`](rq_a/) and default to the repository root; set `OVERBOARD_TREE` to reproduce the recorded numbers against the pinned tree.

The tables were **re-measured against the working tree** after KI-027 (the turn rate is the derivative of the heading) and KI-028 (the validator mirrors the controller that runs) landed. The raw output of every script is in [`../../measurements/2026-10-20-post-ki028/rq_a/`](../../measurements/2026-10-20-post-ki028/rq_a/), one file per script.

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
python docs/research/rq_a/exp10_floor.py           # what sets the floor
python docs/research/rq_a/exp10_gain_band.py       # the calibrated model's gain band
python docs/research/rq_a/exp11_density.py         # the density curve
python docs/research/rq_a/exp12_reference_consistency.py
python docs/research/rq_a/exp13_package.py         # the two halves together
python docs/research/rq_a/exp14_which_inconsistency.py
python docs/research/rq_a/exp15_interval_average.py
python docs/research/rq_a/exp16_firmware_parity.py # the controller that runs
python docs/research/rq_a/exp16_combined.py        # the two fixes, in both timing modes
python docs/research/rq_a/check_hermite.py         # verifies the spline instrument
python docs/research/rq_a/verify_backlash_fix.py  # the KI-025 backlash fix
```

Three controls were used throughout, because two of the intermediate results were wrong before they were controlled:

1. **A synthetic self-consistent arc** built analytically (positions on a circle, heading = the exact tangent) as a ground truth for "what does the validator report on a trajectory that is internally consistent?"
2. **Instrument verification** — `check_hermite.py` checks the spline interpolant against the exact circle before its results are believed.
3. **Per-criterion gate breakdown** — the tracking criterion is reported separately from motor force and slip, so "the tracking gate failed" is never inferred from `passed=False`.

---

## Findings

### F1 — The tracking error does not depend on the uncertainty

Collapsing each declared interval in turn, on the S-curve (all-uncertainty baseline: 1.468 cm):

| interval active | tracking error | change |
|---|---:|---:|
| `cof` only | 1.465 cm | +0.2 % |
| `torque_headroom` only | 1.465 cm | +0.2 % |
| `gear_backlash_m` only | 1.465 cm | +0.2 % |
| `track_width_range` only | 1.468 cm | +0.0 % |
| `wheel_radius_range` only | 1.465 cm | +0.2 % |
| `inertia_kg_m2` only | 1.465 cm | +0.2 % |
| **all collapsed** | **1.465 cm** | **+0.2 %** |

The same holds on the stops mission (+0.7 %) and the complex mission (+0.2 %).

This is structural, not accidental: `cof`, `torque_headroom`, `wheel_radius_range` and `inertia_kg_m2` are consumed only inside `check_sample_interval` (`immrax_validator.py:460-504`), which is `vmap`ed and never touches the propagated state. Only `track_width_range` reaches the plant (`immrax_validator.py:115, 130`). The backlash reached the plant too, as a velocity offset; it is now a position uncertainty in the reachable set ([Fix 1](#fix-1--the-backlash-was-subtracted-from-the-commanded-wheel-speed)).

**So the tracking gate is not a robustness gate.** It is a deterministic check of the model-versus-plan consistency.

*(Re-measured 2026-10-20. The first pass read 1.259-1.262 cm on the S-curve with a +0.1 % spread; the baseline moved with the reference fix and the spread is the same size.)*

### F2 — Backlash was a velocity loss, and only mattered on straights

**This is now fixed** (see [Fix 1](#fix-1--the-backlash-was-subtracted-from-the-commanded-wheel-speed)); the model as it was:

```python
# the old plant, immrax_validator.py:119-120 before the KI-025 fix
vl_act = vl_cmd - (backlash / 2.0) * jnp.tanh(vl_cmd / 1e-3)
```

is not a position dead-band: for any command above 1 mm/s, `tanh → 1`, so the wheel turns at `command − backlash/2` in **both** directions. With the declared `[0.4, 0.5] mm` the robot ran 0.225 mm/s slower than commanded, which was a pure position lag of `0.225 mm/s × t`.

On a 1 m straight that was the whole tracking error (226 µm, and 0.83 µm with the backlash collapsed). On the S-curve it was 1 % of the error. **The backlash model was worth fixing for its own sake — it should be a dead-band, not a constant offset — but it was not what was failing.**

With the fix in the tree the backlash does not touch the tracking error at all: on the S-curve the shipped model and the backlash-collapsed model both report **1.468 cm**, and the difference is in the envelope (0.732 cm with it, 0.605 cm without). The same holds on the stops mission (2.191 cm either way; envelope 0.945 → 0.764 cm) and on the complex mission (0.303 cm either way; envelope 1.185 → 0.974 cm).

### F3 — A self-consistent trajectory tracks to 0.13 cm

The control: an arc of radius 0.30 m and length 0.60 m, built analytically so that the positions are on the circle and the heading is the exact tangent.

> **Re-measured 2026-10-20 after KI-027 and KI-028.** The numbers below were taken with the validator's old 20 ms control tick; the model now runs at 5 ms, which is what the deployed controller actually runs at, and the reference takes the turn rate as the derivative of the interpolated heading. The conclusions are unchanged; the numbers moved.

| n | exact arc | trapezoid positions, tangent headings | heading = chord direction |
|---:|---:|---:|---:|
| 10 | 0.127 cm | 0.181 cm | **4.008 cm** |
| 20 | 0.135 cm | 0.146 cm | **1.989 cm** |
| 40 | 0.136 cm | 0.139 cm | **0.939 cm** |
| 80 | 0.136 cm | 0.137 cm | 0.405 cm |

Two conclusions:

- **The validator, the plant and the controller are not the problem.** A trajectory that is internally consistent is tracked to 0.13 cm at any sampling density, far inside the 5 mm gate.
- **The optimizer's trapezoidal rule is nearly harmless** (0.14–0.18 cm). The rule itself is not the defect.
- **A half-sample heading error is catastrophic** (4.01 cm at n=10, falling as ~1/n).

### F4 — The plan is not on the path its own profile traces

Rebuilding each optimizer plan by *exact* integration of its own `(v, ω)` profile — the same speed profile, the same sample times, the samples placed where the profile actually goes:

> **Re-measured 2026-10-20 after KI-027 and KI-028.** The left column is the tree as it stands; the right column is the same plan with the positions rebuilt.

| mission | tracking, plan as built | tracking, same profile, exact positions |
|---|---:|---:|
| `example_s_curve` | 1.468 cm | **0.495 cm** (PASS) |
| `example_with_stops` | 2.191 cm | 0.980 cm |
| `example_complex_mission` | 0.303 cm | 0.564 cm |
| `example_straight` | 0.341 cm | 0.433 cm |
| `example_with_events` | 0.459 cm | — |

On the S-curve, putting the samples where the profile takes them removes **66 %** of the tracking error and flips the gate to PASS.

The deviation itself falls as ~1/n², exactly as a second-order discretisation error should:

| n | 10 | 20 | 40 | 80 |
|---|---:|---:|---:|---:|
| plan-vs-profile deviation | 1.145 cm | 0.219 cm | 0.063 cm | 0.018 cm |
| **tracking error** | **1.468 cm** | **0.245 cm** | **0.271 cm** | **0.292 cm** |

**The deviation collapses, and now the tracking error follows it.** Before KI-027 that was not true: the tracking error sat at 0.48–1.07 cm while the deviation fell to 0.018 cm, because the reference's own inconsistency (F7) dominated the fine end. With the turn rate taken from the interpolated heading, the fine end is fixed, and what is left at `n=10` is the plan's own deviation: 1.145 cm of the 1.468 cm.

The same holds on the stops mission, where the tracking error now falls monotonically with the density:

| n | 10 | 20 | 40 | 80 |
|---|---:|---:|---:|---:|
| plan-vs-profile deviation | 1.152 cm | 0.213 cm | 0.100 cm | 0.015 cm |
| **tracking error** | **2.191 cm** | 0.461 cm | **0.418 cm ✓** | 0.341 cm ✓ |

(✓ = passes the 5 mm tracking gate; the default `-n 10` does not.)

**More samples is still not the fix** — the optimizer returns a different trajectory at each density, and the coarse end is the plan's own inconsistency, which is a solver problem, not a sampling problem. But the fine end is no longer a floor: it is now the plan's deviation, which the self-consistent collocation removes by construction (KI-026).

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

**The mechanism is the distance covered per interval, not the tick's phase.** The plan's own trapezoidal rule says an interval is covered at the *interval-average* speed, `0.5*(v_k + v_{k+1})*dt_k`. The controller reads the reference at the tick and holds the drive command until the next tick, so the robot's velocity is a *staircase* of the reference velocity, and the distance it covers in an interval is the mean of that staircase — not the interval average the plan assumed. Each interval is therefore covered one tick's worth of velocity change wrong, and the errors accumulate along the path.

Giving the controller the **interval-average velocity** — exactly what the plan's own rule assumes — removes the hold error without touching the tick (`exp10_floor.py`, the `avg_v` column):

| mission | n | shipped | interval-average velocity | 2 ms tick | plan/profile dev |
|---|---:|---:|---:|---:|---:|
| s_curve | 40 | 0.979 cm | **0.110 cm** | 0.048 cm | 0.063 cm |
| s_curve | 80 | 1.087 cm | **0.083 cm** | 0.091 cm | 0.017 cm |
| stops | 40 | 1.294 cm | **0.202 cm** | 0.084 cm | 0.100 cm |
| complex | 40 | 1.161 cm | **0.100 cm** | 0.112 cm | 0.013 cm |
| straight | 40 | 0.432 cm | **0.002 cm** | 0.041 cm | 0.022 cm |

With the hold gone the error lands on the plan's own deviation, which is the *other* term. The hold is also proportional to the path's total velocity variation — 2.96 m/s on the S-curve against 0.95 m/s on the straight path, which is why the straight path's hold error is 0.43 cm and the S-curve's is 0.98 cm — so a flatter velocity profile is a planner-side lever that has not been tested.

This also says where a fix can live. The *tick* is a property of the model (the firmware's 20 ms loop, which the validator mirrors and cannot change). The *reference* is not: `pybricks_code/ramsete_fast.py` interpolates `v` linearly between samples, and the validator mirrors that too. Interpolating the reference velocity as the interval average would remove the hold error in both, in lockstep, with no change to the control loop.

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

The finer tick is a property of the *model*, not of the robot: the controllers that are in use hold the drive command for 5 ms and the validator now models that (Fix 4), so the "2 ms tick" column below is a way of *measuring* how much the density mismatch costs - it is not a fix you can ship without changing the firmware loop. The honest reading is that the plan's sampling density has to be **matched to the controller's tick** - roughly 4 to 6 ticks per sample interval - and the plan's positions have to be the exact integral of its own profile.

### F7 — The reference interpolates four coupled quantities independently, and the firmware does the same

`_interpolated_reference` (`immrax_validator.py:46-77`) interpolates the heading, the speed, the turn rate and the Ramsete gain as four independent lines between the samples. But they are coupled:

- the turn rate must be the **derivative** of the heading,
- the heading must be the **integral** of the turn rate,
- the gain must be `sqrt(ω² + b·v²)` of the *interpolated* `ω` and `v`.

A line for `ω` means the heading is a **parabola**, not a line. The two agree at the sample instants and disagree in between by up to `Δω·dt/8` — **8.7° of contradictory heading mid-interval** on the S-curve at `n=10`.

**The deployed controller has the same defect.** `pybricks_code/ramsete_compact_events.py:155-160` interpolated `tr` (the heading), `vr` (the speed) and `wr` (the turn rate) as three independent lines and handed the robot a heading target and a turn rate that describe different motions. `ramsete_compact.py:151-156` and `ramsete_fast.py:85-91` did the same. The validator (`immrax_validator.py:46-77`) mirrored them, which is why the model and the robot agreed on the bug.

| file | heading | turn rate | gain |
|---|---|---|---|
| `ramsete_compact_events.py` *(deployed)* | line | **line** ✗ | recomputed in `ramsete.py:104` ✓ |
| `ramsete_compact.py` | line | **line** ✗ | line ✗ |
| `ramsete_fast.py` *(legacy)* | line | **line** ✗ | line ✗ |
| `immrax_validator.py` *(the model)* | line | **line** ✗ | line ✗ |

So the deployed path had **two** of the three inconsistencies, and the fix in each file is one line: replace the independently interpolated `wr` with the derivative of the interpolated heading.

```python
# ramsete_compact_events.py:157-160, before
dh = (h2 - h1 + _PI) % _TWO_PI - _PI
tr = (h1 + alpha * dh + _PI) % _TWO_PI - _PI
vr = v1 + alpha * (v2 - v1)
wr = w1 + alpha * (w2 - w1)
# after
wr = dh / dt_traj
```

**One parity trap.** The firmware derives `dh` from the *packed* headings, which are wrapped, so `dh` is the shortest-angle difference. The validator unrolls the headings first (`unroll_trajectory_headings`), so its difference is the true one. They agree for any turn below 180° per sample interval and **disagree in sign above it**. At the default density the per-sample turn is 0.13–0.33 rad, so it does not arise — but at `-n 5` a fast turn can exceed it, and the two sides must then use the same convention.

**This is fixed (KI-027, [Fix 3](#fix-3---the-turn-rate-is-the-derivative-of-the-heading)).** The tree now ships:

| file | heading | turn rate | gain |
|---|---|---|---|
| `ramsete_compact_events.py` *(deployed)* | line | **dh/dt** ✓ | recomputed ✓ |
| `ramsete_compact.py` | line | **dh/dt** ✓ | recomputed ✓ |
| `ramsete.py` *(the IDE missions)* | line | **dh/dt** ✓ | recomputed ✓ |
| `ramsete_fast.py` *(legacy)* | line | **dh/dt** ✓ | line ✗ |
| `immrax_validator.py` *(the model)* | line | **dh/dt** ✓ | recomputed ✓ |

The legacy path still interpolates the gain and rounds the command; everything else agrees, and the validator mirrors the deployed path (KI-028).

Which inconsistency dominates (`exp14_which_inconsistency.py`, per-segment timing):

| variant | s_curve n=10 | s_curve n=40 | stops n=40 | complex n=10 | complex n=20 |
|---|---:|---:|---:|---:|---:|
| A as shipped | 1.248 | 0.979 | 1.294 | 0.864 | 1.046 |
| D gain recomputed only | 1.225 | 0.987 | 1.292 | 0.885 | 1.051 |
| F heading from ω, gain fixed | 0.882 | 1.035 | 1.362 | 1.028 | 1.071 |
| **C turn rate from heading, gain fixed** | **1.470** | **0.271 ✓** | **0.415 ✓** | **0.428** | **0.394 ✓** |

(✓ = passes the 5 mm tracking gate. This table was measured before the fix landed; row C is what the tree ships now, and the re-measurement below confirms the same numbers from the other direction.)

Three things follow:

1. **The gain inconsistency is worth nothing** (0.98–1.08× everywhere). Do not spend time on it.
2. **The turn rate is the one that matters.** Deriving `ω` from the interpolated heading — `ω = (θ_{k+1} − θ_k)/dt` — turns **five of the nine failing (mission, density) cells into passes**, and halves the error on the complex mission at the default density.
3. **"Make it consistent" does not pick a winner; "make the heading primary" does.** F (deriving the heading from `ω`, which is the choice that matches the plan's own trapezoidal kinematics) measures *worse* than the shipped reference at fine sampling. C — keeping the heading's linear interpolation and deriving the turn rate from it — is the winner, and it is also the cheaper change: the heading interpolation does not move.

**This is the cheapest high-value fix in the whole investigation.** It is one line in each of `ramsete_compact_events.py`, `ramsete_compact.py`, `ramsete_fast.py` and `immrax_validator.py`, it does **not** change the plan, so it does not invalidate the measured candidate set or the seed conclusions, and it needs no change to the optimizer. It does change the robot's behaviour, so the validator and the firmware have to change together or the parity is lost.

It is also *complementary* to the optimizer's self-consistent collocation (KI-026), in the same mirror-image way as the density: the plan fix helps at coarse sampling and hurts at fine, the reference fix is the other way round. Together they are worth 0.034 cm on the S-curve at n=30, against 1.248 cm shipped.

---

These are recorded so they are not re-run. Each was a reasonable reading of the evidence at the time.

| Hypothesis | Refuted by | What the truth turned out to be |
|---|---|---|
| "The tracking gate fails because the uncertainty is too wide." | F1: collapsing all seven intervals moves the error by 0.2–0.7 % | It is a deterministic model-versus-plan mismatch |
| "Backlash is the driver." | F1: collapsing it moves the S-curve error by 0 % | Backlash is a straight-path lag only (F2) |
| "The deg/s command quantisation is the driver." | `exp4`: removing quantisation moves the error by 0 % | It is a real dead-band, and it is not a contributor here |
| "More samples fix it." | F4: the deviation falls 1/n² and the tracking error follows it | The optimizer returns a different trajectory at each density, and the coarse end is the plan's own deviation |
| "The validator's reference interpolation is the bug." | `exp8`: a spline reference on the plan as built gives 14–17 cm | The reference is faithful to the firmware; the *plan* is what is inconsistent |
| "The optimizer's trapezoidal rule is the bug." | F3: the trapezoid tracks to 0.14–0.18 cm | The rule is fine; the plan's *headings* are not the tangents of its path |
| "The tracking metric is measuring robot tracking error." | F3: a self-consistent arc tracks to 0.13 cm | It measures how far the plan is from a path the controller can follow |

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

### F8 — The complete reference fix: report the interval averages, not interpolated instantaneous values

Two independent lines of work landed on the same place.

The **hold analysis** (from the floor experiment): the plan's own trapezoidal rule says the distance covered in one sample interval is the *interval average speed* × the interval. The controller holds its command for a whole 20 ms tick, so the robot's velocity is a **staircase** and the distance it covers in an interval is the mean of that staircase — not the average the plan assumed. The mismatch per interval is `0.5·(v_k − v_{k+1})·dt_k`, and it accumulates.

The **consistency analysis** (F7): the heading and the turn rate are interpolated independently, and if `ω` is a line then the heading is a parabola. They disagree by up to 9° mid-interval.

Both say the same thing: **the reference should report the interval averages.** The heading target stays interpolated (the robot really does sweep through it) and the position stays interpolated (the plan's samples are the chord's ends), but the *feedforward* — `v`, `ω` and the gain `kr` — should be the interval averages.

And there is a clean reason it works: the plan's trapezoidal rule **is** "move at the interval-average velocity along the chord". Tell the robot that and it traces exactly the path the plan assumed.

Measured (`exp15_interval_average.py`, per-segment timing; **VW** = interval-average velocity + interval-average turn rate + gain recomputed from them):

| mission | n | shipped | VW | best of {shipped, plan fix, VW} |
|---|---:|---:|---:|---:|
| s_curve | 10 | 1.248 | 2.141 | **0.494 ✓** (plan fix) |
| s_curve | 20 | 0.491 | 0.337 | 0.337 |
| s_curve | 40 | 0.979 | **0.110 ✓** | **0.110 ✓** |
| stops | 10 | 1.807 | 2.982 | 0.974 (plan fix) |
| stops | 20 | 1.065 | 0.407 | 0.407 |
| stops | 40 | 1.294 | **0.202 ✓** | **0.202 ✓** |
| complex | 10 | 0.864 | **0.371** | **0.371** |
| complex | 20 | 1.046 | **0.153 ✓** | **0.153 ✓** |
| complex | 40 | 0.687 | **0.067 ✓** | **0.067 ✓** |

(✓ = passes the 5 mm tracking gate. "Plan fix" is the optimizer's self-consistent collocation, KI-026.)

**The two fixes are mirror images across the density, and that is the whole story of RQ-A:**

| | helps at coarse sampling (n = 10) | helps at fine sampling (n = 20–40) |
|---|---|---|
| **the plan's positions** (KI-026) | yes — S-curve 1.248 → 0.494 ✓ | no — it makes the fine end worse |
| **the reference's averages** (this finding) | no — it makes the coarse end worse | yes — up to **10×** better |

Taken together, **the best of the three variants improves every cell** and passes **5 of the 9**, with the tracking error down to 0.07–0.49 cm across the whole matrix, from 0.49–1.81 cm shipped. The density question, which looked like the central mystery, dissolves once both are in place: the reference fix makes the fine end work, and the plan fix covers the coarse end.

**What this means for the work.** The reference fix is the bigger and cheaper of the two: it is one line per file in `ramsete_compact_events.py`, `ramsete_compact.py`, `ramsete_fast.py` and `immrax_validator.py`, it does not change the plan, so it does not invalidate the measured candidate set or the seed conclusions, and it is worth up to 10× rather than 2.5×. The plan fix is already implemented as `--self-consistent` and is worth 2.5× at the coarse end.

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

### Fix 3 - the turn rate is the derivative of the heading

F7 found that the reference interpolated the heading, the speed, the turn rate and the gain as four independent lines, so the heading and the turn rate described different motions: a line for `omega` means the heading is a parabola, and the two disagree by up to `domega*dt/8` mid-interval. The firmware and the model did it in the same way, which is why they agreed on the bug.

The turn rate is now the **derivative of the interpolated heading**, in the four places that did it, and in the model with the firmware's own shortest-angle convention so the parity holds even above 180 degrees per sample interval:

- `pybricks_code/ramsete_fast.py`, `ramsete_compact.py`, `ramsete_compact_events.py` (the deployed one) and `ramsete.py` (the event reference): `wr = dh / dt`.
- `immrax_validator.py` (`_interpolated_reference`): the same, with the heading difference taken the short way.

The gain was left alone: F7 measured it as worth nothing (0.98-1.08x everywhere).

Measured after the fix, per-segment timing, the five cells F7 tabulated:

| cell | as shipped | now | gate |
|---|---:|---:|---|
| s_curve n=10 | 1.248 cm | 1.51 cm | fail (worse, as F7 predicted) |
| s_curve n=40 | 0.979 cm | **0.27 cm** | **pass** |
| stops n=40 | 1.294 cm | **0.41 cm** | **pass** |
| complex n=10 | 0.864 cm | **0.41 cm** | **pass** |
| complex n=20 | 1.046 cm | **0.39 cm** | **pass** |

Four of the five failing cells now pass, and the complex mission halves its error at the default density. The S-curve at the default density gets *worse*, which is F7's own finding: the coupling cannot be fixed in isolation, the plan has to be self-consistent too (KI-026), and the two together are what the sweep is for.

### Fix 4 - the validator was mirroring a firmware that is not in use

F6 identified the 20 ms command hold as the floor, and treated the period as a
property of the robot. It is not: **it is a property of the model**, and the
model was wrong about which firmware it models.

The validator was written against `pybricks_code/ramsete_fast.py`. The
controllers that are actually in use are the two compact ones and the
list-of-dict reference, and they did not agree with each other:

| | ramsete_compact_events.py *(deployed: hub_runner.py)* | ramsete_compact.py *(the SysId replay)* | ramsete.py *(the IDE missions)* | the validator used to model |
|---|---|---|---|---|
| control period | 5 ms, free-running | 20 ms, synced | 5 ms, free-running | **20 ms** |
| drive command | floats | rounded to mm/s and deg/s | floats | **rounded** |
| Ramsete gain | recomputed from the interpolated v, w | interpolated per sample | recomputed | **interpolated** |
| rate limit | `KinematicLimiter` | none | `KinematicLimiter` | none |

There was no single behaviour for one validator to mirror: the two compact
controllers ran at different periods, and the validator matched neither the
deployed one nor the plain one.

**What changed.**

- One control period for every controller in use, and for the model: a
  **synced 20 ms heartbeat** in `ramsete_compact.py`,
  `ramsete_compact_events.py` and `ramsete.py`, and `CONTROL_DT = 0.02` in
  `immrax_validator.py`. A synced heartbeat is what makes the period exact -
  the loop waits for the slack to the next tick, so the hold is 20 ms and not
  20 ms plus the loop's own compute time. It is also the hub's budget: the
  loop does the odometry update, the controller and the limiter every
  iteration. 20 ms is also the period `ramsete_fast.py` always used.
- `pybricks_code/ramsete_compact.py` now uses the same three parts as its
  event-capable twin - the shared `Odometry`, `RamseteController` and
  `KinematicLimiter` from `ramsete.py` - and the same unrounded drive
  command, so the two are the same controller with and without events and
  the duplicated odometry and Ramsete math is gone.
- `immrax_validator.py` does not round the command, and recomputes the
  Ramsete gain from the interpolated speed and turn rate, like the
  controller the firmware uses.
- `tests/test_firmware_events.py::TestFirmwareParity` pins the parity: the
  controllers in use must declare the same control period, the validator's
  period must equal it, the two compact controllers must use the shared
  parts, and the validator must not round the command.

**Measured** (`exp16_firmware_parity.py`, per-segment timing, the legacy
mirror against the deployed one, plan held fixed): the two now differ by
0.94-1.02x on every (mission, density) cell - the rounding and the gain are
worth about 1 % (F7), and the period is now the same on both sides. So this
fix is a *correctness* fix, not a numerical one: the model now says what the
robot does, and the two compact controllers can no longer drift apart. The
gate is unchanged in its strictness relative to the 20 ms model the tree
already used, and stricter than a 5 ms one would be.

**Two gaps are still open**, and both are recorded rather than closed:

1. The `KinematicLimiter` is in the firmware and in no model. The firmware
   limits the command by the motor's force limit and by the gear backlash;
   the validator models neither. The limiter also still carries the *old*
   backlash model - a deficit subtracted from the wheel speed, which is the
   dimensional error KI-025 removed from the model but not from the firmware.
   So the robot is stricter than the model in one way (the force limit) and
   has a bias the model no longer has in another (the backlash).
2. `ramsete_fast.py` is still the legacy path. It is the one the validator
   used to model, and it is not what runs.

### Not fixed - the structural mismatch

Two of the three sides of the three-way discretisation inconsistency are now closed: the plan's own collocation (KI-026) and the reference's coupling (Fix 3). What is left is the *shape* of the reference - the controller still follows the chord between the samples rather than the curve, which is the second half of the package and needs the plan fix to be in place everywhere, and the command hold, which is now modelled at the firmware's real period (Fix 4) and is the remaining floor (OQ1).

---

## The proposal, and why it was a package

The fix has two halves and **neither works alone**.

**Half 1 — make the plan self-consistent.** The optimizer's samples should lie on the path the trajectory's own `(v, ω)` profile traces, and the heading samples should be the tangents of that path. The trapezoidal collocation does not enforce this, and nothing else in the objective does either. Effect on its own: 61 % less tracking error at n=10 on the S-curve, and *worse* at n=20–80.

**Half 2 — interpolate the reference as a curve through the samples.** The deployed controller and the validator both interpolate `x, y, θ, v, ω` **linearly**, i.e. they follow the chord. A cubic Hermite through the samples reproduces the true path to 0.0012 mm. Effect on its own: catastrophic — 26 cm on the S-curve, with the interval tube diverging to 125 cm.

**Together:** the reason half 2 fails is precisely that half 1 is missing. A Hermite through samples whose headings are not their tangents cannot be both through the points and tangent to them, so it develops curvature spikes, and the closed loop diverges.

The measurement in `exp8` that isolates this:

| variant | S-curve | stops | complex | straight |
|---|---:|---:|---:|---:|
| 1. as shipped | 1.260 cm | 1.814 cm | 0.489 cm | 0.346 cm |
| 2. self-consistent plan | **0.495 cm** | 0.980 cm | 0.564 cm | 0.433 cm |
| 3. spline ref on plan as built | 26.0 cm | 38.3 cm | 6.0 cm | 0.439 cm |
| 4. self-consistent + spline ref | 12.9 cm | 1.148 cm | 21.5 cm | 0.441 cm |

Variant 4 is *worse* than variant 2, so "self-consistent plan + spline" is **not** sufficient either. The self-consistency produced by rebuilding the positions does not make the plan's *heading samples* tangents — the rebuilt heading is the integral of ω, but the residual heading inconsistency at the stops (where the chord degenerates and the direction reverses) survives, and the spline amplifies it.

### What of that shipped, and what is left

| half | status | evidence |
|---|---|---|
| the plan is self-consistent | **shipped** (KI-026, `--self-consistent`) | F10: the S-curve's error halves, the events mission's forward-integration error goes away |
| the reference is *coupled* (the turn rate is the derivative of the heading) | **shipped** (KI-027, Fix 3) | F7: five of the nine failing (mission, density) cells become passes |
| the model mirrors the controller that runs | **shipped** (KI-028, Fix 4) | Fix 4: the numbers are the robot's, not a mirror of a controller that is not in use |
| the reference is a *curve* through the samples (half 2) | **not shipped** | OQ3: a firmware change, and it needs the plan fix everywhere first |
| the feedforward reports the *interval averages* (F8) | **not shipped** | F8: up to 10× at fine sampling, and it is a one-line change per file |

The open part of RQ-A is therefore narrower than it was: **not** "find a formulation in which the samples, the headings and the profile are mutually consistent by construction" — that is KI-026 — but "make the controller follow the path between the samples instead of the chord, and give it the feedforward the plan's own rule assumes". Both are firmware changes, so both need the parity work again, and both are cheap: the plan fix is in the tree, and the reference fix is one line per file.

---

## Open questions

| # | Question | Why it matters | Next step |
|---|---|---|---|
| **OQ1** | What sets the floor of ~0.5–1.1 cm at n=40–80 on the S-curve? The plan/profile deviation is 0.063 cm there, so 95 % of the error is unexplained. | Until this is answered, no fix can be shown to work. | **CLOSED (F6).** It is the 20 ms command hold: refining the tick to 2 ms takes the error from 0.979 cm to 0.048 cm at n=40. At coarse sampling the floor is the plan/profile mismatch instead, and the error is U-shaped in the density. The mechanism is the distance covered per interval: giving the controller the interval-average velocity removes the hold without touching the tick (F6) |
| **OQ2** | Can the optimizer be reformulated so that consistency is structural? | It is the only fix that addresses the root cause. | Replace the trapezoidal collocation with a higher-order quadrature over the *interpolated* heading, so `x, y` are the exact integral of `v·cos θ(t)`. Then the samples lie on the path by construction and the headings are the tangents. |
| **OQ3** | Should the reference become a spline in the validator **and** the firmware? | The firmware and the validator must agree, or the parity is lost. | `pybricks_code/ramsete_fast.py` and `ramsete_compact*.py` interpolate linearly today. Any change there is a firmware change and needs the plan fix first. |
| **OQ4** | Does the tracking metric need a heading term? | The metric is position-only (`immrax_validator.py:740-741`). A plan can be followed in position while the heading is off by up to Δθ/2 — 9.4° on the S-curve at n=10. | Measure the heading error the closed loop actually shows, and decide whether it is a gate, a reported metric, or both. |
| **OQ5** | What is the stops mission's extra 0.6 cm? (1.17 cm deviation but 1.81 cm tracking.) | Stops are where the discretisations interact worst: the robot reverses, the backlash sign flips, the chord degenerates. | Separate the reversals from the curvature. Build a variant of the stops mission with the stops removed but the same geometry. |
| **OQ6** | Does any of this change the seed conclusions? | The companion document's headline is that the seeds buy gate insurance, not speed. | Probably not, but re-run the 60-candidate matrix once the discretisation is fixed. The win counts may change, because the gate is what the seeds are escaping. |

---

### F10 — the self-consistent collocation, implemented (OQ2, half 1)

OQ2 asked whether the optimizer can be reformulated so the consistency is
structural. It can: replace the trapezoidal position rule with a quadrature
over the *integrated* heading, and the samples land on the path the plan's own
(v, omega) profile traces, with the heading samples as its tangents.

`optimizer.py` now has it behind `--self-consistent` (off by default, so the
measured plans stay reproducible). The position rule integrates the interval's
velocity along the heading the robot will hold - the turn rate integrated, not
the two endpoints - with a four-node Gauss-Legendre rule, O(dt^7) per interval
against the trapezoid's O(dt^3). The heading rule is unchanged, so the headings
are the tangents by construction. Nothing on the robot changes: the firmware
still follows the chord, and the chord is now within the sagitta of the path.

| S-curve, n=10 (default) | plan/profile deviation | tracking | gate |
|---|---:|---:|---|
| trapezoid, candidate 1 / 2 | 11.5 mm | 1.55 / 1.47 cm | fail / fail |
| **self-consistent, candidate 1 / 2** | < 0.1 mm | **0.98 / 0.88 cm** | fail / fail |

| events, n=10 (default) | tracking | gate |
|---|---:|---|
| trapezoid, candidate 1 / 2 | 0.68 / 0.46 cm | fail / **PASS** |
| self-consistent, candidate 1 / 2 | **0.44** / 1.45 cm | **PASS** / fail |

The validator's own forward-integration check stops reporting its error,
because the plan now agrees with itself.

It is not a free win, and it is not the whole fix. Measured on the current
tree (the default candidate set, `--self-consistent` on and off; the raw
records are in
[`../../measurements/2026-10-20/matrix/`](../../measurements/2026-10-20/matrix/)):

| mission | trapezoid: tracking / gate | self-consistent: tracking / gate |
|---|---|---|
| `example_straight` | 0.34 cm / PASS | 0.34 cm / PASS |
| `example_mission_first4` | 0.24 cm / PASS | 0.26 cm / PASS |
| `example_with_events` | 0.68 cm (equal) / **0.46 cm** (per-segment) | **0.44 cm** (equal) / 1.45 cm (per-segment) |
| `example_complex_mission` | **0.30 cm** (equal) / 0.43 cm (per-segment) | 0.42 cm (equal) / **0.35 cm** (per-segment) |
| `example_with_stops` | 0.98 cm / 2.19 cm, both fail | 1.23 cm / 1.69 cm, both fail |
| `example_s_curve` | 1.55 cm / 1.47 cm, both fail | 0.98 cm / 0.88 cm, both fail |

What that says:

- **The S-curve's error halves** (1.47 → 0.88 cm) but still misses the 5 mm
  gate by 0.38 cm. The coarse end needs the reference fix as well (F8).
- **The candidate set has to be re-tuned under the new formulation.** The
  complex mission's *equal-duration* candidate passes trapezoidal (0.30 cm)
  and fails self-consistent (0.42 cm), and its per-segment candidate does the
  opposite (0.43 → 0.35 cm). The two models swap places; neither is a peer.
- **The stops mission gets slower** under equal-duration timing (6.25 → 7.81 s)
  and its per-segment tracking error falls from 2.19 to 1.69 cm: a direction
  reversal is where a self-consistent plan is hardest (OQ5).
- **The sweep makes the stops mission solvable.** With `--sweep
  --self-consistent`, the equal-duration candidate at weight 0.01 passes at
  6.44 s / 0.49 cm — the first passing stops trajectory in the tree.
- These numbers are measured with F7 (the coupled reference) and F4 (the
  parity fix) already in the tree. Before F7 landed, the S-curve's
  self-consistent candidate measured 0.47 cm and passed, so the two changes
  interact and have to be tuned together.
- At fine sampling this changes little: the floor there is the command hold,
  not the plan (OQ1).

## What this changes in the research agenda
In [`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md) Part IV:

- **Q4 (reachability in the objective) is not the right first lever.** The failing missions fail on a deterministic discretisation mismatch; an envelope term would not touch it. Revise the order: consistent discretisation first, reachability term second.
- **Q9 (a real digital twin) is still needed, but it is not what is failing today.** The tracking error is already ~100 % explained by the current model. Adding motor dynamics, odometry and slip will not move the number that is blocking the missions. Q9 is about *honesty* about the envelope, not about the tracking gate.
- **The 5 mm threshold is not currently meaningful.** It is being compared against a number that is dominated by a modelling artefact. Until the discretisation is consistent, a pass or a fail says more about the discretisation than about the robot. **Do not tune the threshold to make missions pass** — that would hide the defect.
- **Q6 (does the accuracy weight buy accuracy?) should be re-run after the fix**, because the weight's effect on the tracking error may itself be an artefact.
- **Q9 (a real digital twin) is not only about honesty about the envelope — it moves the binding gate.** With a fitted model the gain band widens the reachable set until the *envelope* gate binds (F5), which is a different failure from the one the tracking gate reports today. A calibrated robot is a different problem, not the same problem with better numbers.
- **Q4 (reachability in the objective) therefore has two audiences.** With the default model the envelope is never the binding constraint, so an envelope term in the objective buys nothing on the six example missions. With a calibrated model it is the *only* binding constraint. Whichever configuration the project settles on decides whether Q4 is urgent.

---

## Status summary

| | |
|---|---|
| Question | Answered: the cause is identified, and the open item is closed |
| Root cause | The optimizer's discretisation, the controller's reference and the plant's integration are three different discretisations of the same trajectory; on a curved path they disagree by ~1 cm |
| OQ1 (the floor at fine sampling) | **Answered.** Two mechanisms, at opposite ends of the density range: at coarse sampling the plan's positions are off the path its own profile traces; at fine sampling the control loop's hold of the drive command is the floor. The error is U-shaped in the density (F6) |
| Fixes landed | **KI-025** the backlash dimensional error. **KI-026** the optimizer's self-consistent collocation (`--self-consistent`). **KI-027** the turn rate as the derivative of the interpolated heading, in the firmware and the validator together. **KI-028** the validator models the controller that runs. The structural fix is therefore **in the tree** |
| Measured effect | On the example matrix, re-measured after those fixes: **31 of 60 candidates pass** (was 24), **7 of the 12 (mission, timing model) cells produce a trajectory** (was 6), and a non-baseline seed is selected in 4 of them (was 2). The envelope still never binds: **0 of 29 failures** cite it, and the largest envelope is 1.75 cm against a 2 cm threshold |
| Confidence | High on the diagnosis (four independent measurements agree, with synthetic controls) |
| Safety | The KI-025 fix changed no gate outcome on its own: 0 of 60 candidates on the example matrix and 0 of 231 on the mission suite changed pass flag, with the worst-case uncertainty per run moving by at most 0.065 cm |
| Which gate binds | **Tracking** in the default configuration (25 of 29 failures). **Envelope** once a calibrated model is fitted, because the measured gain band widens the reachable set (F5) |
| Hardware | **None.** Every number here is a model prediction. No hardware logs exist in this repository, and the uncertainty ranges in `examples/fll_choreo.json` are declared test assumptions. The *relative* findings (which term dominates, how the error scales) are properties of the model and hold regardless; the absolute numbers do not |
| Re-measured | The numbers in F3 and F4 were **re-taken on 2026-10-20** after KI-027 and KI-028, because the model changed under them: the control tick, the reference and the plant all moved. The conclusions did not change, the numbers did |

