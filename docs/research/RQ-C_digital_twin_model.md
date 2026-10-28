# RQ-C — What would a full digital twin need, and in which order?

**Status:** literature pass complete; the ranked component order below is registered as Q25 in [`../RESEARCH_AGENDA.md`](../RESEARCH_AGENDA.md). Component 1 (per-side radii + effective wheelbase) is measured: LAND 2026-10-28 (research/results/q25_c1_component1.md); component 2 (rolling drag) is measured: NULL, retired 2026-10-28 (research/results/q25_c2_drag.md); components 3-4 are unmeasured. KI-064 is the run-to-run loop that would consume a landed component; KI-067 is the asymmetry detector on top of component 1, not a separate arm.
**Date:** 2026-10-27
**Companion:** [`RQ-B_drivetrain_model.md`](RQ-B_drivetrain_model.md) (the motor model), [`../RESEARCH_AGENDA.md`](../RESEARCH_AGENDA.md) (Q9, Q18, Q19, Q25), [`../CALIBRATION_PROCEDURE.md`](../CALIBRATION_PROCEDURE.md), and KI-020/KI-064/KI-067 in [`../../KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md).

---

## The question

KI-020 names what the fitted plant still lacks: motor torque transients, the
DriveBase closed-loop speed loop, integrated encoder/gyro odometry, wheel-slip
dynamics, rolling resistance, and feedback through the robot's own
`DriveBase.state()` estimate. Q18 (closed) showed the 5 mm tracking gate
measures the twin plant's closed-loop residual — so twin fidelity is now the
single biggest lever on gate coverage (6/32). Q19 (closed) ranked the
sensitivities: `track_width` moves verdicts ~30x more than anything else
(implied tolerance ±1.9 %), then inertia, `wheel_radius`, `v_max_rad_s`. Q9
(open, registered) names odometry error feeding back through the controller as
the prime suspect.

This document is a literature pass over four areas, to decide **which of the
missing components are worth building, which are already subsumed by what we
fit, and in which order to build them**:

1. differential-drive odometry calibration (UMBmark and successors),
2. LEGO SPIKE motor modelling and the DriveBase closed loop,
3. wheel-slip and traction modelling for small robots,
4. digital-twin verification & validation methodology.

Third-party numbers below are literature values used as checks — never copied
code (the RQ-B rule). What was read in full and what was taken from metadata
only is listed under [Sources](#sources-and-what-was-actually-read).

---

## What was done

### 1. Differential-drive odometry calibration

The UMBmark line (Borenstein & Feng, 1994/95) identifies **two dominant
systematic errors**: unequal wheel diameters (`Ed`) and the *effective*
wheelbase (`Eb`) — the distance over which the wheels actually grip, which
differs from the tape-measured axle track because axles bend and tyres compress
under load. The prescribed test is a square path driven in **both** directions
(cw and ccw): the two errors interact, so a single-direction test can show them
cancelling and hiding each other; only the cw+ccw pair separates them. From the
two end-pose errors `X_cw`, `X_ccw` over side length `L` the method computes a
per-leg veer `beta` and a turn error `alpha`, and corrects the wheelbase and
the per-side encoder scales (procedure and formulas verified in the DPRG
summary, <https://www.dprg.org/calibrating-your-robot-using-umbmark/>). The
test is run slowly to keep slip out of the measurement. The 1996 report claims
"by changing only the effective wheelbase and the effective wheel-diameter
ratio the vehicle's odometric accuracy (with respect to systematic errors) can
be improved by at least one order of magnitude"
(<http://web.cecs.pdx.edu/~mperkows/CLASS_479/S2006/Paper-correction-odometry-error.pdf>).

Successors keep the two-parameter core and improve the estimation:

- **Linear least-squares:** the odometric parameters enter the error model
  linearly, so one least-squares fit over an arbitrary trajectory identifies
  them without a square path (Antonelli & Chiaverini, *Linear Estimation of the
  Odometric Parameters for Differential-Drive Mobile Robots*, Autonomous Robots
  23(1):59–68, 2007).
- **IPEM:** integrated prediction-error minimization fits dynamic model
  parameters by *integrating* the model over the whole trajectory against
  low-frequency pose observations, instead of fitting one-step responses
  (Seegmiller & Kelly, IJRR 32(8), 2013,
  <https://journals.sagepub.com/doi/10.1177/0278364913488635>).
- **Simultaneous ML calibration:** the odometry parameters and a sensor's
  extrinsics can be calibrated together in closed form; identifiability of the
  *set* matters — calibrating too few parameters hides their interaction
  (Censi & Marchionni, *Simultaneous maximum-likelihood calibration of odometry
  and sensor parameters*, 2007).
- **Modern re-evaluation:** the survey line (e.g. Sousa & Petry, *Evolution of
  Odometry Calibration Methods for Ground Mobile Robots*, IEEE ICRA/CRSC 2020)
  still finds the UMBmark pair — effective wheelbase and wheel-diameter ratio —
  the standard systematic-error model.

**Fit to our collector kit:** the square cw/ccw test maps exactly onto the
existing pattern — a collector test in `pybricks_code/sysid_digital_twin.py`
(like `odometry_straight` / `odometry_turn`), end poses recorded on the
measurement sheet, fitted in `overboard/sysid/fit.py` under the KI-010
three-repeat 95 % t-interval discipline. The LSQ formulation is the same
parameters with a different estimator and can cross-check the closed-form
`alpha`/`beta` result on the same session.

### 2. LEGO SPIKE motor modelling and the DriveBase closed loop

The official spec sheet for the LEGO Technic **Medium Angular Motor** (MPN
45603 — the motor this robot uses; <https://le-www-live-s.legocdn.com/sc/media/files/support/spike-prime/techspecs_technicmediumangularmotor_19684ffc443792280359ef217512a1d1.pdf>,
all at 7.2 V): **no-load 185 RPM ±15 %** (0 Ncm, 110 mA ±15 %), **max
efficiency 3.5 Ncm at 135 RPM ±15 %** (280 mA ±15 %), **stall 18 Ncm**
(0 RPM, 800 mA ±15 %). Rotation sensor: 360 counts/rev, ≤ ±3° accuracy
including gearbox slack, 100 Hz update. These are citable checks for our
measured motor pair (stall torque from the wall test, no-load speed from
`top_speed`, RQ-B); the measured wall/top-speed tests remain authoritative
for the actual robot.

One arithmetic observation on the published numbers: the straight torque-speed
line through (185 RPM, 0 Ncm) and (0 RPM, 18 Ncm) predicts 4.9 Ncm at
135 RPM, but the spec's max-efficiency point is 3.5 Ncm there. The three
official points are **not collinear**, so the two-end line RQ-B uses is an
approximation and the efficiency point is a third, independent check a
measured torque-speed pair can be held against (speed tolerance is ±15 % and
no torque tolerance is published, so treat the shape as approximate).

What Pybricks' `DriveBase` actually does (verified in
<https://pybricks.com/ev3-micropython/robotics.html> and
<https://pybricks.com/ev3-micropython/motors.html>): `drive(v, ω)` is tracked
by **two PID controllers** — `distance_control` and `heading_control` — each a
`Control` with `pid(kp, ki, kd, integral_range, integral_rate, feed_forward)`,
where `feed_forward` adds duty in the direction of the speed reference, and
`limits(speed, acceleration, actuation)` where `actuation` caps the duty in
percent of maximum. `state()` returns (distance, drive speed, angle, turn
rate) from the internal rotation sensors — "wheels may slip … only estimates".
Settings only change while stopped; `stop()` coasts.

So a planner-facing model must carry: the documented acceleration ceilings
(already declared and enforced — Q11 steps 1–2), the actuation/duty cap (which
is our motor force cap), and the **closed-loop step response at the `drive()`
boundary**. Our fitted first-order gain/tau/delay are measured *through*
`drivebase.drive()` at 20 ms with raw encoders logged at 100 Hz
(`pybricks_code/sysid_digital_twin.py`), i.e. they already describe the loop's
net response at exactly the boundary the planner commands. The DC motor's
electrical time constant is much smaller than its mechanical one (e.g. the
Iowa State EE476 workbook,
<http://class.ece.iastate.edu/ee476/motion/Main_manual.pdf>), so current and
torque build-up sit below the 20 ms tick. LEGO's own calibration advice in the
DriveBase docs — `straight(1000)` to fix `wheel_diameter`, `turn(360)` to fix
`axle_track` — is a two-test UMBmark-lite.

### 3. Wheel-slip and traction modelling

The slip literature for small differential-drive robots (tutorial: Naveed,
*A Tutorial on Modeling and Control of Slippage in Wheeled Mobile Robots*,
arXiv:2306.14074, 2023; *Study of wheel slip and traction forces in
differential drive robots and slip avoidance control strategy*) agrees on the
shape: a Coulomb traction cap bounds the usable force, slip-ratio dynamics
describe what happens *near* that limit, and slip-avoidance control keeps the
robot out of the regime. Slip degrades odometry online, so odometry
calibration and slip interact under low traction (De Giorgi et al., *Online
Odometry Calibration for Differential Drive Mobile Robots in Low Traction
Conditions with Slippage*, MDPI Robotics 13(1):7, 2023).

**When does slip *dynamics* matter instead of a static cap?** Only when the
wheel runs near or past the friction limit — hard braking, hard acceleration,
low-µ surfaces. Our operating regime is the opposite: the planner keeps wheel
force under the traction limit so the planned path is slip-free by
construction, braking slip is already a fitted plant term (`SlipFit`, applied
by the plant), and FLL mats are high-µ. The remaining slip gap is the
braking/acceleration corner, and RQ-B already found braking is the corner our
model handles worst (the braking-shortfall fit exists *because* of it).

### 4. Digital-twin verification & validation methodology

The V&V consensus (read in depth: Mertens & Denil, *Reusing Model Validation
Methods for the Continuous Validation of Digital Twins of Cyber-Physical
Systems*, arXiv:2512.04117, 2025, building on Sargent and on Oberkampf &
Roy) is: **validation is comparison with experimental data of the physical
system**, within a stated domain of applicability and for a stated intended
purpose; verification (including code-to-code comparison against an
independent simulator) is a different, weaker claim. Validation metrics are
simple distances between predicted and measured traces (RMSE, normalised
Euclidean distance, average/maximum relative error), and a twin is re-validated
whenever the physical system drifts. Permissible claim from sim-only
comparison: "agrees with the reference simulator" — never "matches the robot".
This is exactly RQ-B's stance ("structure confirmed against an independent
simulator; numbers measured later"), and it is why KI-020 stays open until
hardware exists.

---

## Findings

**Finding 1 — the literature's dominant odometry errors are our #1
sensitivity.** Borenstein & Feng's two systematic errors — unequal wheel
diameters and the effective wheelbase — are precisely the parameters Q19
ranked first (`track_width`, ±1.9 % implied) and third (`wheel_radius`), and
they are exactly the parameters the Q9 belief path is written in. Literature,
sensitivity ranking and hypothesis all point at the same two numbers.

**Finding 2 — the DriveBase PID and the motor torque transient are subsumed as
separate model terms.** Our actuator fit is measured through `drive()`, so the
first-order gain/tau/delay already describe the closed-loop response at the
planner's boundary (Pybricks documents the loop as PID + feed-forward +
actuation limits), and the electrical transient is far below the 20 ms tick
(§2). A separate loop or current model would be unidentifiable from
drive()-level data and buys nothing at the boundary we model.

**Finding 3 — rolling resistance is the one measured, quantified oracle gap.**
RQ-B measured it: coast deceleration is 2.4x the model's in the reference
simulator, and every predicted deceleration is therefore an upper bound. It is
the cheapest honest fix and it un-blurs the braking-shortfall fit, which
currently absorbs part of the drag.

**Finding 4 — sim-only claims are bounded by doctrine.** Twin-vs-oracle
agreement is verification, not validation; the Q25 registration words every
claim accordingly, and KI-020 stays open until a hardware session exists.

---

## What was dropped, and why

- **DriveBase closed-loop speed loop (dropped — subsumed).** The fit's step
  tests command `drivebase.drive()` at 20 ms and log raw encoders at 100 Hz,
  so the fitted first-order gain/tau/delay already carry the loop's response at
  the boundary the planner sees — Pybricks documents the internals (PID +
  feed-forward + actuation limits,
  <https://pybricks.com/ev3-micropython/motors.html>) but nothing observable at
  `drive()` level can identify them separately.
- **Motor torque transient (dropped — subsumed).** The DC motor's electrical
  time constant is much smaller than the mechanical one
  (<http://class.ece.iastate.edu/ee476/motion/Main_manual.pdf>), so current and
  torque build-up sit below the 20 ms tick and are folded into the fitted
  delay; the torque-speed line itself is already structurally validated
  (RQ-B: top speed within 7 % of an independent rigid-body simulator).

Both are re-openable only on new evidence: measured residual structure in the
twin-vs-oracle comparison that a first-order actuator cannot explain.

---

## Sources, and what was actually read

| source | what was taken | how verified |
|---|---|---|
| DPRG, *Calibrating Your Robot Using UMBMark* (Bouchier 2009) <https://www.dprg.org/calibrating-your-robot-using-umbmark/> | full UMBmark procedure, alpha/beta formulas, Ed/Eb corrections, "run slowly to avoid slip" | read in full |
| Borenstein et al. 1996, *Measurement and Correction of Systematic Odometry Errors in Mobile Robots* <http://web.cecs.pdx.edu/~mperkows/CLASS_479/S2006/Paper-correction-odometry-error.pdf> | "one order of magnitude" improvement claim | listing/abstract snippet only — full PDF not fetched |
| Antonelli & Chiaverini 2007, *Linear Estimation of the Odometric Parameters…*, Autonomous Robots 23(1):59–68 | linear least-squares identifiability of the odometric parameters | metadata only |
| Seegmiller & Kelly 2013, IPEM <https://journals.sagepub.com/doi/10.1177/0278364913488635> | integrated prediction-error identification over whole trajectories | abstract only (PDF not fetchable) |
| Censi & Marchionni 2007, *Simultaneous maximum-likelihood calibration of odometry and sensor parameters* | closed-form joint calibration; identifiability of the parameter set | abstract only |
| Sousa & Petry 2020, *Evolution of Odometry Calibration Methods for Ground Mobile Robots* | modern re-evaluation of the calibration methods | metadata only (PDF fetch failed) |
| LEGO Technic Medium Angular Motor spec sheet (MPN 45603) <https://le-www-live-s.legocdn.com/sc/media/files/support/spike-prime/techspecs_technicmediumangularmotor_19684ffc443792280359ef217512a1d1.pdf> | no-load 185 RPM ±15 % (110 mA), max efficiency 3.5 Ncm @ 135 RPM ±15 % (280 mA), stall 18 Ncm (800 mA), 100 Hz sensor | official sheet; PDF not fetchable by the fetcher, text cross-checked across independent reproductions of the same sheet |
| Pybricks docs <https://pybricks.com/ev3-micropython/robotics.html>, <https://pybricks.com/ev3-micropython/motors.html> | DriveBase = distance + heading PID, Control pid/feed-forward/limits, state() contents, stop() coasts | read in full |
| Naveed 2023, arXiv:2306.14074 <https://arxiv.org/abs/2306.14074> | slip/skid modelling and slip-avoidance overview | abstract only |
| De Giorgi et al. 2023, MDPI Robotics 13(1):7; *Study of wheel slip and traction forces in differential drive robots…* | slip–odometry interaction under low traction | metadata only |
| Iowa State EE476 workbook <http://class.ece.iastate.edu/ee476/motion/Main_manual.pdf> | electrical time constant << mechanical time constant | snippet only |
| Mertens & Denil 2025, arXiv:2512.04117 <https://arxiv.org/html/2512.04117> | V&V definitions, validation metrics, sim-only claim limits (via Sargent, Oberkampf & Roy) | read (sections 1–4) |

---

## What applies to us

| gap (KI-020 list) | literature recommendation | our current state |
|---|---|---|
| unequal wheel diameters + effective wheelbase | UMBmark square cw/ccw test, or one LSQ over an arbitrary trajectory; keep Ed and Eb separate — they cancel in single-direction tests | UMBmark pair (kL/kR/b_eff) landed with the belief wiring (Q25-1, belief-first); single scales stay as the no-fit fallback; Q19 requires `track_width` to ±1.9 % |
| rolling resistance / viscous drag | a Coulomb + viscous drag term, identified by coast-down | **absent** — RQ-B measured coast deceleration 2.4x off; every predicted deceleration is an upper bound; the braking-shortfall fit partly absorbs the drag |
| feedback through the robot's own estimate | the controller must consume its own integrated odometry, not the true pose (UMBmark errors propagate into the loop; De Giorgi: slip degrades it online) | belief-first production default (Q25-1): `_advance_belief` / `belief_mode="estimate"` with fitted scales + gyro bias (geometry replaces scales); scaled truth only as the no-fit fallback (KI-020) |
| DriveBase closed-loop speed loop | planner-facing model = step response at `drive()` + documented ceilings + duty cap | **dropped, subsumed** — fitted gain/tau/delay measured through `drive()` already carry it |
| motor torque transient | second-order only where residual structure demands it | **dropped, subsumed** — electrical TC << mechanical, folded into the fitted delay |
| wheel-slip dynamics | slip-ratio states only near the friction limit | **contingent, last** — traction cap keeps the plan below the limit; braking slip already fitted (`SlipFit`) |
| (cross-cutting) validation claims | sim-to-sim comparison is verification; validation needs hardware data | oracle claims will be worded as "agrees with fll-sim"; KI-020 stays open until a hardware session |

## THE RANKED COMPONENT ORDER (pre-registered as Q25)

| # | component (model term) | rationale — literature + Q9 + Q19 |
|---|---|---|
| 1 | **Per-side effective wheel radii + effective wheelbase, identified jointly, wired belief-first** (replaces the single distance/heading scales) | UMBmark: unequal diameters and the effective wheelbase are the two dominant systematic errors and only a cw+ccw pair separates them; the LSQ line keeps the same parameters with a better estimator. Q19: `track_width` is the #1 sensitivity (±1.9 % implied), `wheel_radius` #3. Q9's belief path is parameterised by exactly these numbers, so nothing downstream is measurable before they are tight. **Q25-1 lock (2026-10-28 untangle): belief wiring + geometry, belief-first** — belief carries ~99.5% of C1's gain (F-A -231.4mm), radii in truth alone -1.1mm (-0.2%). Identification: a square cw/ccw collector test + sheet end poses (KI-010 x3), closed-form alpha/beta cross-checked by LSQ. |
| 2 | **Rolling resistance / viscous drag** (Coulomb + viscous term in the wheel-speed dynamics) | The one *measured* oracle discrepancy (RQ-B: coast deceleration 2.4x; predicted decelerations are upper bounds) and the standard missing term in a wheel-force model. Identification: a coast-down test (the collector already logs the braking transients; coast mode gives the decay directly). It also stops the braking-shortfall fit from absorbing drag. Cheapest term with a guaranteed measurable effect. |
| 3 | **Integrated encoder/gyro odometry + feedback through the robot's own `DriveBase.state()` estimate** (the Q9 belief) | Q9's registered prime suspect: the deployed controller never sees the true pose, so odometry error feeds back through the reference it chases — and Q18 showed today's failures are exactly the twin plant's closed-loop residual. The belief machinery is built and inert; enabling it needs component 1's parameters plus the gyro bias the `static` test already fits. This arm is also the synthetic dress rehearsal for the Q9 hardware experiment. |
| 4 | **Wheel-slip / grip dynamics beyond the braking shortfall (contingent)** | Slip-ratio models matter only near the friction limit (slip literature); the traction cap keeps planned force below it and braking slip is already fitted, so build this only if components 1–3 leave a braking/acceleration residual the cap cannot explain. Rank last: contingent on evidence, not scheduled. |
