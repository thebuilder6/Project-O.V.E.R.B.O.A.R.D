# API Documentation: overboard/verification and overboard/deploy

Trajectory validation, safety gating, and controller export utilities.

---

# `overboard/verification/validator.py`

The CPU validator is the first line of defense. `overboard/main.py` runs `validate_samples` before
writing any output (KI-002). Immrax reachability is a separate check in
`overboard/verification/immrax_validator.py` (see the last section).

## `forward_integrate(samples, robot_cfg, fine_dt=0.001)`

Forward-integrates differential-drive kinematics from the optimized wheel-velocity profile using RK4.

**Parameters**

- `samples` (`list[dict]`): Trajectory samples.
- `robot_cfg` (`RobotConfig`): Robot configuration.
- `fine_dt` (`float`): Integration timestep in seconds (default `0.001`).

**Returns** `(integrated, errors)`

- `integrated` (`list[dict]`): States with keys `t, x, y, heading`.
- `errors` (`dict`):
  - `max_pos_error_m`
  - `rms_pos_error_m`
  - `max_heading_error_rad`
  - `final_pos_error_m`
  - `final_heading_error_rad`

If the timestamps are empty, non-finite, non-monotone, or have no positive duration, the function returns `([], errors)` with every error set to `inf`. A failed solve is therefore never mistaken for a good integration.

**Method**

- Linearly interpolates `vl`/`vr` at 1 ms steps.
- Integrates `dx = v*cos(theta)`, `dy = v*sin(theta)`, `dtheta = omega` with RK4.
- Compares the integrated pose at each planned timestamp.
- Wraps heading errors to `[-π, π]`.

## `audit_constraints(samples, robot_cfg, apply_headroom=True)`

Re-evaluates motor, traction, slip, and wheel-speed limits at every sample.

**Returns** a dict with:

| Key | Meaning |
|---|---|
| `left_motor_force` | Max left motor force violation (N) |
| `right_motor_force` | Max right motor force violation (N) |
| `left_wheel_slip` | Max left wheel slip force (N) |
| `right_wheel_slip` | Max right wheel slip force (N) |
| `traction_total` | Max legacy total-traction violation (N) |
| `max_wheel_speed_m_s` | Maximum absolute wheel speed |
| `wheel_speed_limit_m_s` | `RobotConfig.no_load_wheel_speed_worst_case()`: `v_max_rad_s * min(wheel_radius_range) / gearing`, no headroom (KI-035) |
| `speed_violation_m_s` | Max speed over the limit (0 if none) |
| `left_normal_force` | Mean left normal force (N) |
| `right_normal_force` | Mean right normal force (N) |
| `num_violating_samples` | Samples with a motor or total-traction violation |
| `violating_sample_indices` | Indices of those samples |
| `num_slip_points` | Samples with per-wheel slip |
| `slip_points` | Slip details |

Each `slip_points` entry contains:

```text
index, time, x, y, heading,
left_wheel_slip_N, right_wheel_slip_N,
left_normal_force_N, right_normal_force_N,
vl, vr
```

Wheel slip is checked per wheel against `cof * normal_force`, not only against the total `cof*m*g` limit. This matches the optimizer's per-wheel constraint (KI-010).

## `compute_metrics(samples, robot_cfg=None)`

Returns:

- `total_time_s`
- `path_length_m`
- `max_linear_speed_m_s`
- `avg_linear_speed_m_s`
- `max_wheel_speed_m_s`
- `max_accel_m_s2`
- `max_jerk_m_s3`
- `avg_jerk_m_s3`
- `tortuosity`
- `velocity_chattering`
- `avg_force_utilization` and `max_force_utilization` (reported as 0 when `robot_cfg` is omitted)

`compute_metrics` does not validate; it only reports. Force utilization is computed against the nominal motor curve, not the robust limits.

## `compute_path_fitness(samples, robot_cfg=None, stop_sample_indices=None, immrax_report=None, cpu_summary=None, weights=None)`

Returns the 5-part time-equivalent **path fitness score** and its parts. Lower is better; the units are seconds, so the score reads as "the time this trajectory's defects are worth".

| part | weight | what it measures |
|---|---:|---|
| `stop_precision` | 200 s/m | the reachability spread at every stop and at the final sample, with the heading spread lever-armed: `sqrt(r^2 + (L_arm * dtheta)^2)` |
| `slip_risk` | 8 s | a quadratic penalty on the combined motor and traction utilisation above a 70 % dead-zone: `0.5 * mean + 0.5 * max` of `(utilisation - 0.7)^2` |
| `rms_force` | 0.5 s | the time-weighted RMS wheel force, normalised by the stall force |
| `tracking_accuracy` | 80 s/m | half the RMS path tracking error plus the sum of the closed-loop tracking errors at the stops and the final sample |
| `time` | 1 s/s | the trajectory's own duration |

Returns:

- `fitness_score` — the weighted sum, in seconds
- `travel_time_s`
- `stop_precision_spread_m` and `final_precision_spread_m`
- `slip_risk_penalty` and `peak_utilization`
- `rms_wheel_force_n` and `rms_force_utilization`
- `tracking_error_m`, `rms_tracking_error_m` and `stop_tracking_error_m`

**Usage.** The tracking error comes from the Immrax report when one is supplied (the per-sample closed-loop error), and from the CPU audit's forward integration otherwise. The reachability spread comes from the Immrax report's per-sample envelopes, and from the envelopes attached to the samples if there is no report. `weights` overrides the default weights; `stop_sample_indices` are the sample indices of the stops, and the final sample is always a checkpoint.

The score is a **ranking, not a gate**: a plan is never rejected because its fitness is bad. `overboard/main.py` uses it to pick between the candidates that passed the gates **on the stock path** (`--no-unified`); the default unified path ranks the passing candidates by duration first. See `docs/HEURISTICS_AND_VERIFICATION.md` §7.3.

## `check_final_waypoint(samples, waypoint, pos_tol=0.02, heading_tol=0.02)`

Checks that the last sample reaches the requested final pose.

**Returns**

```python
{
    "passed": bool,
    "pos_error_m": float,
    "heading_error_rad": float,
}
```

If `samples` is empty or `waypoint` is `None`, it returns `passed=True` with zero errors. `overboard/main.py` calls this even when `--validate` is not set.

## `validate_samples(samples, robot_cfg, apply_headroom=True)`

Runs the same CPU checks as `validate_trajectory` without printing or writing files.

**Returns**

```python
{
    "passed": bool,
    "metrics": dict,
    "audit": dict,
    "errors": dict,
    "integrated": list,
}
```

**Pass criteria**

- `errors["max_pos_error_m"] < 0.01`
- `errors["final_pos_error_m"] < 0.01`
- `audit["num_violating_samples"] == 0`
- `audit["num_slip_points"] == 0`
- `audit["speed_violation_m_s"] < 1e-6`

This function does **not** check the final waypoint; `overboard/main.py` does that separately with `check_final_waypoint`.

## `validate_trajectory(traj_file, config_file, apply_headroom=True, waypoints=None)`

CLI-style validation entry point. Loads the `.traj` and config, runs `validate_samples`, prints a report, and optionally checks the final waypoint.

**Parameters**

- `traj_file` (`str`): Path to a `.traj` file.
- `config_file` (`str`): Path to a robot config JSON.
- `apply_headroom` (`bool`): Use the robust headroom limits (default `True`).
- `waypoints` (`list[tuple]`, optional): If supplied, checks the final pose too.

**Returns** `(metrics, audit, errors)`.

**Output**

The report includes metrics, constraint audit, wheel-speed limit, slip details, forward-integration
errors, and a final-waypoint check when `waypoints` is supplied. It prints `PASS` or `WARN`, but the
`WARN` text is advisory in this function. The CLI gate is the Immrax candidate check plus
`validate_samples` and `check_final_waypoint` in `overboard/main.py`.

**Usage**

```bash
python -m overboard.verification.validator output.traj examples/fll_choreo.json
```

---

# `overboard/deploy/export.py`

## `resample_to_fixed_dt(samples, target_dt=0.02, track_width=0.0965)`

Linearly resamples a variable-timestep trajectory to a fixed controller timestep.

**Returns** a list of dicts with:

- `t, x, y, heading, vl, vr, v, omega`
- `event` (optional)

Events from the source are attached to the nearest output sample within one timestep. The heading is unwrapped before interpolation, so a path that crosses ±π is interpolated the short way and the output headings are wrapped back to [-π, π] (KI-011). A consumer that computes heading *differences* must use the shortest angle, as the deployed controllers do.

## `export_controller_json(samples, target_dt=0.02, track_width=0.0965)`

Returns a JSON-serializable controller profile:

```json
{
  "format": "controller_profile",
  "version": 1,
  "dt": 0.02,
  "num_samples": 150,
  "samples": [
    {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "vl": 0.0, "vr": 0.0, "v": 0.0, "omega": 0.0}
  ]
}
```

## `write_controller_file(input_traj_file, output_file, target_dt=0.02, track_width=0.0965)`

Loads a `.traj` file, resamples it, and writes the controller profile JSON. Prints the sample count and timestep.

```python
from overboard.deploy.export import write_controller_file

write_controller_file(
    "output.traj",
    "output_controller.json",
    target_dt=0.02,
    track_width=0.0965,
)
```

## `write_python_file(input_traj_file, output_file, compact=False)`

Loads a `.traj` file and writes a Python module containing:

- a `config` dictionary with the canonical robot fields (`mass`, `inertia`, `track_width`, `wheel_radius`, `v_max_rad_s`, `t_max_nm`, `gearing`, `cof`)
- a `samples` list with `t, x, y, heading, vl, vr, omega`, plus `event` when present

The config is parsed through `RobotConfig`, so canonical, legacy, and flat trajectory configs all produce the same physical values. The output is not a Choreo file and does not include the `robot` wrapper.

```python
from overboard.deploy.export import write_python_file

write_python_file("output.traj", "trajectory.py")
```

CLI equivalent:

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o output.traj --export-format python
```

With `compact=True`, `write_python_file` calls `write_compact_python_file` instead. The generated module contains a packed `samples` bytes object, a `config` dictionary, and an `events` tuple. The blob is a 4-byte little-endian header (`<HH` version, count) followed by 24 bytes per sample (`<6f` t, x, y, heading, v, omega). `ramsete_compact.py` and `ramsete_compact_events.py` decode this format directly; the other
controllers expect the list format. The compact module is what `hub_runner.py` deploys.

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o output.traj --export-format python-compact
```

`encode_compact_trajectory(samples)` returns `(blob, events)`. It rejects missing or non-finite fields and values that do not fit in float32. `write_compact_python_file(input_traj_file, output_file)` performs the same load/write path as the CLI.

---

# `overboard/verification/immrax_validator.py`

## `PROPAGATION` / `PROPAGATIONS`

The module constant `PROPAGATION` (default `"plant"`) names the reachability model, and
`PROPAGATIONS` holds the two names `"plant"` and `"tick"`. `--propagation` and
`ImmraxValidator(propagation=...)` select one; every candidate carries the model it used, so an
envelope is never read without it.

- **plant** (default): the plant is embedded and propagated, and the controller is evaluated
  *outside* the tube at its midpoint, with the command held for a tick. The tube is that of an
  open-loop plant driven by a recorded command.
- **tick**: one control tick of the closed loop is embedded as a discrete map, controller
  inside, and the map is propagated once per tick. Its Jacobian carries the feedback term, so
  the tube contracts the way the robot's does. It needs the mixed-Jacobian inclusion (the natural
  extension explodes) and costs 2-3x (`jac`) to 7-10x (`mjac`) the validation time.

`docs/research/RQ-A_tracking_error.md` holds the measurement behind the choice.

## `ImmraxValidator(config, ramsete_b=2.0, ramsete_zeta=0.7, control_dt=0.02, feedback_mode="nominal", digital_twin=None, propagation=None, ...)`

Wraps `JAXRobotConfig`. `feedback_mode="nominal"` feeds the interval midpoint to the controller
(the deployed robot reacts to its best estimate); `"interval"` evaluates the controller inclusion
on the whole interval and diverges on long fast paths. The remaining arguments select the
inclusions and the digital-twin options, and default to module constants so a run is reproducible.

## `validate_trajectory(samples, cof_range, torque_margin_range, backlash_range, track_width_range=None, wheel_radius_range=None, inertia_range=None, safety_threshold_m=0.005)`

Interval audit of every sample: motor force, per-wheel slip and wheel speed, with the stall torque
read from the config (`--sysid-fit` is what fills it with the measured number). **Returns**

```text
max_motor_violation_N, max_traction_violation_N,
max_slip_l_N, max_slip_r_N,
max_tracking_error_m, rms_tracking_error_m,
max_heading_error_rad, rms_heading_error_rad,
max_envelope_radius_m,
max_wheel_speed_m_s, wheel_speed_limit_m_s, speed_violation_m_s,
reachability,          # the tube report, see compute_reachability
inclusion,             # the plant / controller / tick inclusion used
propagation,           # "plant" or "tick"
digital_twin,          # the fit's summary, or None
passed                 # the verdict
```

`passed` is true when the motor force, the per-wheel slip, the tracking error, the wheel speed and
the tube's non-divergence are all inside their thresholds; the envelope size is judged by the
caller against the per-waypoint limit. `overboard/main.py` folds the numbers into the candidate's
record and into the machine-readable validation report.

## `compute_reachability(samples, backlash_range, track_width_range, wheel_radius_range, cof_range, torque_margin_range, inertia_range, legacy_debug_loop=None)`

Propagates the reachable tube sample to sample and returns:

- `envelope` (per sample: `x_min/x_max`, `y_min/y_max`, `theta_min/theta_max`, tracking and heading
  error, and `unknown` when the tube could not be computed)
- `max_error_m`, `max_tracking_error_m`, `rms_tracking_error_m`, `max_heading_error_rad`,
  `rms_heading_error_rad`, `max_envelope_radius_m`
- `diverged` and `envelope_truncated`
- `propagation`, the model that produced it

The compiled `jax.lax.scan` path in `overboard/verification/reachability_engine.py` is the
shipped one; `legacy_debug_loop=True` keeps the older Python while-loop for A/B measurements. With
fewer than two samples, or with non-monotone timestamps, the report is `inf`/`diverged` rather than
empty, so a failed solve never reads as a clean tube.

## The digital twin

`ImmraxValidator` takes an optional `DigitalTwinConfig` (`overboard/sysid/digital_twin.py`,
`load_sysid_fit(path)`, fed by `--sysid-fit`) describing how the *deployed* controller and
DriveBase respond, as opposed to `RobotConfig`'s description of the physical robot. A plan without
a fit behaves exactly as before; every mechanism is inert unless the fit supplies its parameter.
What the plant applies:

- the straight-line steady-state wheel-speed gain range,
- the first-order actuator time constant and the command delay,
- an acceleration limit on the wheel speed,
- the robot's own odometry belief (distance/heading scale and gyro bias) fed to the controller, so
  the controller reacts to its estimate instead of the true pose,
- the braking shortfall: while braking, the wheels turn further than the robot travels, so the
  truth covers less ground than the encoders report. The factor is a smooth function of the
  commanded speed, because Immrax's natural inclusion cannot compare or select.

The optional `drivebase` block (name, wheel radius, track width) is preserved in
`DigitalTwinConfig.drivebase` and `summary()`; `overboard/main.py` warns when it differs from the
robot config, but it does not override the validator geometry. What is still missing is KI-020's
list; `docs/DIGITAL_TWIN_PLAN.md` tracks it.

---

# Safety contract

- A failed solve is never exported (KI-001).
- With `--validate`, a failed CPU audit, Immrax reachability check, or final-waypoint check blocks the `.traj`, controller, Python file, and stats file (KI-002).
- Without `--validate`, output is exploratory and has not passed the Immrax gate.
- `--sysid-fit sysid_fit.json` opts into the measured actuator/odometry model (gain, time
  constant, delay, acceleration limit, odometry belief, braking shortfall). The default kinematic
  model is unchanged when it is omitted (KI-020).
- `--export-format python-compact` writes the packed module used by `ramsete_compact.py` and `ramsete_compact_events.py`, not the legacy list format.
