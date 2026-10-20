# Digital twin plan (KI-020)

The validator is a model of the robot, and a model is only as good as its
parameters. The rule for this repository: **build the machinery now, let the
SysId fill in the numbers.** Every mechanism below is inert until the fit
provides its parameter, so a plan without data behaves exactly as it did
before, and a plan with data is as faithful as the measurement allows.

## The data contract

`sysid_fit.py` already writes every parameter the plant needs. They were
parsed and ignored; the plant and the belief now consume them.

| Fit block | Field | Consumed by | What it models |
|---|---|---|---|
| `actuator.straight` | `gain`, `gain_min`, `gain_max` | the wheel-speed target | the steady-state speed error of the drivebase |
| `actuator.straight` | `tau_s` | the wheel-speed lag | the actuation time constant |
| `actuator.straight` | `delay_s` | the sampled control loop | the command delay |
| `actuator.straight` | `accel_limit` | the wheel-speed rate | the measured acceleration limit |
| `odometry` | `distance_scale` | the belief integration | the encoder distance scale error |
| `odometry` | `heading_scale` | the belief integration | the gyro rate scale error |
| `sensors` | `gyro_z_bias_deg_s` | the belief integration | the gyro drift over a run |
| `slip` | `brake_ratio_median` | the plant's grip factor | the braking shortfall |

The poses are the one thing the robot cannot measure for itself: the fit
needs a tape measure at the stops of an `odometry_straight`, an
`odometry_turn` or a `brake_from_*` test. The poses are matched to the log by
`run_id`, and a poses file with the wrong run id is silently ignored.

## What is built

1. **The robot's belief.** The deployed controller never sees the true pose:
   it integrates the wheel encoders and the gyro. The validator now
   integrates the same belief alongside the plant and feeds *that* to the
   controller, so the odometry scale error and the gyro drift separate the
   robot's estimate from the truth over a run. The belief is deliberately not
   a plant state: it is an estimate, not a physical quantity, so it does not
   belong in the reachable tube. It is advanced in the same order and with
   the same wheel speeds as the plant, so a perfect odometry keeps it exactly
   on the truth; only the measured errors separate the two.
2. **The braking shortfall.** While braking, the wheels turn further than
   the robot travels, so the truth covers less ground than the encoders
   report. The plant applies the measured fraction, smoothly, because
   Immrax's natural inclusion cannot compare or select.
3. **The acceleration limit.** The wheel speed cannot change faster than the
   measured limit, so the model cannot accelerate faster than the robot did.
4. **The gyro bias.** The belief's heading drifts with the measured bias.

Each mechanism has a host test that pins its semantics, so the model is
checkable before any hardware exists.

## What this does not fix

[`docs/research/RQ-A_tracking_error.md`](research/RQ-A_tracking_error.md) shows that
the 5 mm tracking gate is measuring a discretisation mismatch — the
optimizer's collocation, the controller's linear reference and the plant's
integration are three different discretisations of the same trajectory — and
not a tracking problem. So:

- **This work will not move the number that blocks the missions.** The
  tracking error is already ~100 % explained by the current model. What this
  work makes honest is the *envelope*: the reachability tube now reflects the
  robot's own odometry, its braking shortfall and its measured limits.
- The two are complementary, not alternatives: RQ-A fixes what the gate
  measures, the digital twin fixes what the envelope claims.
- Do not read a change in the tracking number as evidence about the robot.

## What is still open after this build

| Gap | What it needs | Why it is not built yet |
|---|---|---|
| DriveBase PID and the motor current transient | a step response per axis, or the firmware's gains | the first-order lag plus the measured limit covers the first order; the transient needs a current measurement |
| The drivebase's torque limit (wheel slip under load) | the measured traction point at the planned accelerations | the plan is traction-limited, but the inner loop can still command more torque than the wheels can use |
| Slip during acceleration, not only braking | a slip model, not one measured number | the fit measures the braking case; the dynamic case needs a different test |
| The reachability envelope as an optimizer term | a candidate set that cannot find a passing path | the KI defers this; the post-solve score already selects the candidate |
| The interval feedback mode | a working interval Ramsete | it still sees the scaled truth rather than the belief, and it diverges on long fast paths; the nominal mode is the production path |

The backlash is not on this list: KI-025 moved it out of the plant and into
the reachable set as a position uncertainty, so the plant no longer loses
backlash/2 in both directions.

## The order of work

1. **Build the machinery** (done above). It is inert without a fit.
2. **Collect.** `python sysid_collect.py --conn ble --profile examples/drivebase_profile.example.json`
   pushes the collector, captures the session and writes the fit.
3. **Compare.** `python sysid_compare.py output.traj poses.csv --run-id <id>`
   gives the planned-versus-measured residual. That residual is the only
   honest measure of what the model is still missing.
4. **Decide from the residual.** Add the mechanism the residual points at,
   not the one that seems plausible.
5. **Only then** consider the envelope as an optimizer term.

The 5 mm tracking gate and the 2 cm envelope threshold stay as they are.
If the measured residual is larger than the gate, the model is wrong, not the
threshold.
