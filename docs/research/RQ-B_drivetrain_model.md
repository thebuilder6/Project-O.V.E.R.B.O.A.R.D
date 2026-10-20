# RQ-B — What can the robot's motor actually do, and how do we know?

**Status:** the motor model is in place and its structure is confirmed against an independent simulator. The motor's numbers are now measured instead of declared. **One reference simulator defect was found** (see [Finding 3](#finding-3-the-reference-simulators-brake-mode-reverses-the-wheels)).
**Date:** 2026-10-20
**Companion:** [`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md), [`../CALIBRATION_PROCEDURE.md`](../CALIBRATION_PROCEDURE.md), and KI-020 in [`../../KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md).

---

## The question

The planner's motor model is two numbers: the stall torque `t_max_nm` and the
speed `v_max_rad_s` at which the motor's torque reaches zero. Together they
are the whole torque-speed line, because a motor with a back EMF has a
straight one.

Both numbers were **declared** in the robot config. A declared number is a
guess: the robot has to be pushed into a wall to find the stall torque, and
lifted off the ground to find the free-running speed. Meanwhile the
optimizer planned against the declared motor while the validator checked the
*measured* one (KI-020), so a plan could satisfy its own constraint and
still trip the validator.

Three questions:

1. Can the motor model be written down once, in one place, so that the
   optimizer, the CPU audit, the validator and the exported controller cannot
   disagree about it?
2. Can the two numbers be *measured* on this robot?
3. Is the model any good? Is there an independent check?

---

## What was done

### 1. One drivetrain model, in one place

[`../../drivetrain.py`](../../drivetrain.py) holds the motor and drivetrain
model: the back-EMF torque law, the gearbox's Coulomb and viscous friction,
the traction cap, the braking and coasting decelerations, and a small
integrator that runs a wheel-speed command through the motor and the ground.

The *structure* is textbook motor physics. The SPIKE Prime numbers in it
are **literature values** from a third-party simulator
(<https://github.com/martijas/fll-sim>, `packages/sim/src/motor.ts`), which
has no licence file, so nothing was copied: the equations are written out
here and the constants are cited as what they are — numbers somebody else
published, not measurements of this robot.

### 2. The motor's numbers are measured

The calibration kit now measures the free-running speed with the wheels off
the ground (`top_speed` in the collector), which is the other end of the
torque-speed line. The stall torque was already measured by pushing the
robot into a wall.

The fit reports both, with the same 95 % interval over the repeats as every
other fitted number. A measurement is only used when it is valid: if the
robot *reached* the commanded speed, the command was the limit and not the
motor, and the number is rejected with the reason.

`main.py --sysid-fit` now writes the measured motor into the robot config,
so **the optimizer, the CPU audit, the Immrax validator and the exported
controller all plan against the same motor** (KI-010's rule).

The measured numbers also fix a subtlety in the declared ones. The config's
`v_max_rad_s` is 18.94 rad/s (1085 deg/s), which is the SPIKE Prime medium
motor's *regulated* speed (1110 deg/s) rather than its *no-load* speed
(1290 deg/s). The model uses that number as the no-load speed, so as
declared it was slightly pessimistic: the motor still has torque at the
speed the model says it has none. The measurement settles which one it is.

### 3. An independent check

[`../../research/crosscheck/`](../../research/crosscheck/) drives a
wheel-speed step profile in an independent rigid-body simulator and compares
what happened with what OVERBOARD's model predicts. The simulator is run in
WSL2, because it is a Linux/TypeScript project; the harness copies one
script into its checkout, runs it, and copies the trace back.

---

## The evidence

The reference simulator's own drive base (56 mm wheels, 120 mm track, SPIKE
Prime medium motors, 0.598 kg) is used for the comparison, because the
question is about the model and not about a particular robot.

| quantity | reference simulator | OVERBOARD's model | difference |
|---|---:|---:|---:|
| steady speed on a full-speed command | 1005 deg/s (0.491 m/s) | 1085 deg/s (0.530 m/s) | 7 % |
| deceleration after the command is dropped (coast) | 3.59 m/s² | 1.51 m/s² | 2.4× |
| deceleration after the command is dropped (brake) | 2.87 m/s² | 8.83 m/s² | 3.1× |

```
python research/crosscheck/compare_step.py
```

### Finding 1 — the motor model's structure holds

The back-EMF line predicts the top speed to within 7 % of a rigid-body
simulation that knows nothing about our model. The torque-speed law, the
gearbox drag and the traction cap are the right model for this robot.

The 7 % is in the expected direction: the model has no rolling resistance,
so it expects the robot to hold a little more speed than a real one does.

### Finding 2 — the model has no rolling resistance

The coast deceleration is 2.4× larger in the simulator than the model says.
The model only knows the gearbox's friction; a real robot also has the
tyres' rolling resistance and whatever drags on the mat (a caster, a skid).

**Consequence:** any deceleration this model predicts is an **upper bound**.
Do not use `brake_deceleration()` or `coast_deceleration()` to predict where
the robot stops. They are capability numbers — what the motor *could* do —
not what the robot *does*. The planner does not use them today, and should
not until a drag term is measured on this robot.

### Finding 3 — the reference simulator's brake mode reverses the wheels

In the reference simulator, `Stop.BRAKE` applies a torque that accelerates
the wheel in the direction it was already turning: the wheel reverses
instead of slowing. The trace shows the left motor at −1045 deg/s before the
stop and +785 deg/s one sample later, and the robot stops because the
reversed wheel drags on the mat, not because the motor braked.

**Consequence:** its braking distance is not a usable reference, and the
brake row in the table above should be read as "not comparable". The coast
row is the one to compare. Anyone using that simulator as a reference for
braking has to know this.

---

## What was not done, and why

- **No deceleration model in the plant.** The plant would need an
  asymmetric rate limit on the wheel-speed state (accelerating fast,
  decelerating slowly), and Immrax's natural inclusion cannot compare or
  select. The measured `accel_limit` already covers the symmetric case, and
  the plan's own deceleration is a *commanded* deceleration that the motor
  can deliver, so the extra machinery would buy nothing measurable today.
- **No wheel-slip dynamics.** The planner keeps the wheel force under the
  traction limit, so the planned path is slip-free by construction; slip
  only matters for the uncertainty, and the envelope already carries that.
- **No torque-speed curve as data.** The curve is still a line between two
  measured ends. Fitting a curve to the load channel would need the load
  channel's units settled first, and the collector's `l_load_mNm` column
  does not say whether it is milli-newton-metres or a percentage of the
  stall torque.

---

## Consequences for the repository

| change | where |
|---|---|
| the drivetrain model | [`../../drivetrain.py`](../../drivetrain.py) |
| the measured motor reaches the config | [`../../robot_model.py`](../../robot_model.py) (`apply_measured_motor`), [`../../main.py`](../../main.py) |
| the fit extracts the free-running speed | [`../../sysid_fit.py`](../../sysid_fit.py) (`_fit_no_load_speed`) |
| the collector measures it | [`../../pybricks_code/sysid_digital_twin.py`](../../pybricks_code/sysid_digital_twin.py) (`test_top_speed`) |
| the cross-check harness | [`../../research/crosscheck/`](../../research/crosscheck/) |
