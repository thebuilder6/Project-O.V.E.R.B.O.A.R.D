# API Documentation: overboard/models/robot_model.py

Robot configuration parsing, motor/traction limits, and differential-drive force calculations shared by the optimizer, CPU validator, JAX model, and Immrax.

## Module constant

- `CG_HEIGHT_M = 0.05` (m): Center-of-mass height used by every weight-transfer calculation.
  Along with `wheel_normal_forces`, it is defined in `overboard/models/drivetrain_physics.py`
  and re-exported by `overboard/models/robot_model.py`, so the optimizer, CPU validator,
  JAX model and Immrax all import the one copy (KI-009).

## `wheel_normal_forces(mass, g, h_cg, track_width, centripetal_accel)`

Returns `(nl, nr)`, the normal force on the left and right wheels:

```text
nl = m*g/2 - m*h_cg*(v*omega)/b
nr = m*g/2 + m*h_cg*(v*omega)/b
```

Only lateral weight transfer is modeled. A two-wheel differential robot has no front/rear axle, so longitudinal acceleration is not split left/right (KI-010). The function accepts floats, CasADi expressions, JAX arrays, or Immrax intervals.

## `RobotConfig`

### Constructor

```python
RobotConfig(config_dict: dict)
```

`config_dict` may be:

- canonical: `{ "robot": { ... } }`
- legacy Choreo: `{ "config": { "mass": { "val": 0.723 }, ... } }`
- flat: `{ "mass": 0.723, ... }`

The canonical and legacy key names are normalized (`track_width`/`differentialTrackWidth`, `wheel_radius`/`radius`, `v_max_rad_s`/`vmax`, `t_max_nm`/`tmax`, `torque_headroom`/`torqueHeadroom`, `speed_headroom`/`speedHeadroom`).

### Fields and defaults

| Attribute | Default | Unit |
|---|---:|---|
| `mass` | 0.723 | kg |
| `inertia` | 0.0024 | kg·m² |
| `track_width` | 0.0965 | m |
| `wheel_radius` | 0.028 | m |
| `v_max_rad_s` | 15.7 | rad/s |
| `t_max_nm` | 0.04 | N·m |
| `gearing` | 1.0 | — |
| `cof` | 0.45 | — |
| `g` | 9.81 | m/s² |
| `torque_headroom` | 0.85 | — |
| `speed_headroom` | 0.90 | — |
| `rolling_resistance` | 0.0 | — |
| `gearbox_coulomb_nm` | 0.0 | N·m |
| `gearbox_viscous_nm_s` | 0.0 | N·m·s/rad |
| `reflected_inertia_kg_m2` | 0.0 | kg·m² |
| `width_m` | 0.16 | m |
| `length_m` | 0.18 | m |
| `front_bumper_m` | 0.10 | m |
| `back_bumper_m` | 0.08 | m |
| `lever_arm_m` | 0.10 | m |
| `color_sensor_offsets` | `{'left': (0.06, -0.03), 'right': (0.06, 0.03)}` | m |
| `hardware` | hub axes & motor/sensor port mapping | — |
| `uncertainties` | see below | — |

### `uncertainties`

`RobotConfig` normalizes the declared ranges once. Missing entries fall back to these defaults:

| Key | Default |
|---|---|
| `cof` | `(0.8 * cof, 1.2 * cof)` |
| `torque_headroom` | `(0.8, 1.0)` |
| `gear_backlash_m` | `(0.0001, 0.0006)` |
| `track_width_range` | `(0.99 * track_width, 1.01 * track_width)` |
| `wheel_radius_range` | `(0.99 * wheel_radius, 1.01 * wheel_radius)` |
| `inertia_kg_m2` | `(0.90 * inertia, 1.10 * inertia)` |
| `t_max_range` | `(0.75 * t_max_nm, 0.95 * t_max_nm)` |
| `initial_x_m` | `(-0.005, 0.005)` |
| `initial_y_m` | `(-0.005, 0.005)` |
| `initial_theta_rad` | `(-1.0 deg, 1.0 deg)` |

`t_max_range` is parsed and stored, and the sysid fit writes it back (`apply_sysid_fit`).
The optimizer, CPU validator and Immrax path bound the motor with `t_max_nm` and
`torque_headroom` instead. The JAX model does read `t_max_range`, but only behind
`use_intervals=True` (`get_max_force_at_velocity`, `check_constraints`), which no shipped
path passes.

### Force-limit methods

#### `get_max_force_at_velocity(v_wheel, apply_headroom=True) -> float`

Nominal linear motor curve:

```text
omega = (v_wheel / wheel_radius) * gearing
torque = t_max_nm * max(0, 1 - |omega| / v_max_rad_s)
force = (torque / wheel_radius) * gearing
if apply_headroom:
    force *= torque_headroom
```

#### `get_max_force_symbolic(v_wheel, apply_headroom=True) -> casadi.MX`

CasADi version. It uses `sqrt(omega**2 + 1e-8)` as a smooth absolute value and `fmax(0, torque)`.

#### `max_linear_speed(apply_headroom=True) -> float`

`v_max_rad_s * wheel_radius / gearing * (speed_headroom if apply_headroom else 1.0)`.
The gearing divide matches the force model, which compares `(v / wheel_radius) * gearing`
with `v_max_rad_s` (KI-035).

#### `get_robust_force_params() -> tuple[float, float, float]`

Returns `(speed_radius, force_radius, torque_headroom)` from the declared uncertainty ranges: smallest radius for the speed term, largest radius for the force term, and smallest headroom.

#### `get_max_force_at_velocity_robust(v_wheel) -> float`

Worst-case per-wheel force limit matching the lower bound used by Immrax.

#### `get_max_force_symbolic_robust(v_wheel) -> casadi.MX`

CasADi version of the robust limit.

### Calibration and Hardware Sync

#### `apply_measured_motor(t_max_nm=None, t_max_range=None, no_load_speed_rad_s=None) -> dict`

Replaces the declared motor numbers with measured ones and returns what changed (empty when
nothing was applied). `no_load_speed_rad_s` is read at the motor; the wheel's no-load speed
divides it by `gearing`. Unless the config pinned `t_max_range` itself, the derived range follows
the measured stall torque. Because the optimizer, CPU audit, Immrax validator and exported
controller all read this one config, a measured motor no longer lets a plan satisfy its own
constraint and still trip the validator (KI-010).

#### `apply_sysid_fit(twin) -> dict`

Performs a full merge of a digital-twin fit (`DigitalTwinConfig` or `dict`) into `RobotConfig`:
- Updates effective `wheel_radius` (scaled by measured odometry `distance_scale`).
- Updates effective `track_width` (scaled by measured odometry `heading_scale`).
- Updates motor curve (`t_max_nm`, `v_max_rad_s`, and derived/measured `t_max_range`) by
  delegating to `apply_measured_motor`.
- Updates `gear_backlash_m` from calibrated backlash tests.
- Updates `rolling_resistance` from braking slip tests.
- Updates 95% confidence intervals for actuator gain and heading drift.

#### `sync_pybricks_robot_config(path=None) -> bool`

Syncs `pybricks_code/robot_config.py` from this `RobotConfig`, ensuring `WHEEL_DIAMETER_MM`, `AXLE_TRACK_MM`, motor ports and directions, attachment ports, sensor ports, and hub IMU axes never drift apart between host and robot firmware.

#### `chassis_radius_m() -> float`

Effective body radius covering front and back bumpers and chassis width.

## `DifferentialDriveModel`

### Constructor

```python
DifferentialDriveModel(config: RobotConfig)
```

### `get_dynamics(vl, vr, al, ar) -> tuple[float, float]`

Returns `(fl, fr)` from:

```text
a = (al + ar) / 2
alpha = (ar - al) / track_width
F_total = mass * a
M_total = inertia * alpha
fr = (F_total + 2*M_total/track_width) / 2
fl = F_total - fr
```

### `get_dynamics_symbolic(vl, vr, al, ar) -> tuple[casadi.MX, casadi.MX]`

CasADi version of `get_dynamics`.

### `get_wheel_normal_forces(vl, vr, al, ar) -> tuple[float, float]`

Computes `v = (vl + vr)/2`, `omega = (vr - vl)/track_width`, and calls `wheel_normal_forces(mass, g, CG_HEIGHT_M, track_width, v*omega)`. The acceleration arguments are accepted for call-site compatibility but are not used (KI-010).

### `get_max_wheel_force_symbolic(al, ar, apply_headroom=True) -> casadi.MX`

Worst-case per-wheel force magnitude over the declared inertia and track-width ranges:

```text
f_total = mass * (al + ar) / 2
split = max(inertia) * (ar - al) / min(track_width)^2
|f| <= |f_total|/2 + |split|
```

The symbolic implementation uses `sqrt(x**2 + eps)` in place of `abs` for IPOPT.

### `get_traction_limit_symbolic(vl, vr, apply_headroom=True) -> casadi.MX`

Per-wheel traction limit matching the Immrax slip check. Both wheels are limited by the smaller normal force over the lateral transfer, using the minimum declared cof and track width when headroom is enabled.

### `check_constraints(vl, vr, al, ar, apply_headroom=True) -> dict`

Returns a dictionary, not a list:

| Key | Meaning |
|---|---|
| `left_motor_violation` | N over the left motor force limit (0 if none) |
| `right_motor_violation` | N over the right motor force limit (0 if none) |
| `left_wheel_slip` | N over the left wheel's friction limit (0 if none) |
| `right_wheel_slip` | N over the right wheel's friction limit (0 if none) |
| `traction_violation` | Legacy total-traction violation in N |
| `left_normal_force` | Left normal force in N |
| `right_normal_force` | Right normal force in N |

With `apply_headroom=True`, the motor limits use the declared worst-case ranges. With `apply_headroom=False`, they use the nominal curve. Slip checks always use the nominal cof and the lateral-transfer normal forces.

## Usage example

```python
from overboard.models.robot_model import RobotConfig, DifferentialDriveModel

config = RobotConfig({
    "robot": {
        "mass": 0.723,
        "inertia": 0.0024,
        "track_width": 0.0965,
        "wheel_radius": 0.028,
        "v_max_rad_s": 18.94,
        "t_max_nm": 0.138,
        "gearing": 1.0,
        "cof": 0.95,
        "torque_headroom": 0.85,
        "speed_headroom": 0.90,
        "uncertainties": {
            "cof": [0.85, 1.05],
            "torque_headroom": [0.75, 0.90],
            "gear_backlash_m": [0.0002, 0.0008],
            "track_width_range": [0.095, 0.098],
            "wheel_radius_range": [0.027, 0.029],
            "inertia_kg_m2": [0.0020, 0.0028]
        }
    }
})

model = DifferentialDriveModel(config)
checks = model.check_constraints(vl=0.5, vr=0.6, al=1.0, ar=1.2)

if all(checks[k] <= 1e-6 for k in (
    "left_motor_violation", "right_motor_violation",
    "left_wheel_slip", "right_wheel_slip", "traction_violation"
)):
    print("State is feasible")
else:
    print(checks)
```
