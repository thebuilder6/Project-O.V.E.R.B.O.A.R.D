# Configuration Guide

This guide describes the JSON configuration and waypoint files used by the FLL Trajectory Optimizer.

## Accepted configuration formats

`RobotConfig` accepts three layouts and normalizes them to the same internal fields:

1. **Canonical** (recommended): `{ "robot": { ... } }`, as in `examples/fll_choreo.json`.
2. **Legacy Choreo**: `{ "config": { "mass": { "val": 0.723 }, ... } }`.
3. **Flat**: the robot fields at the top level.

Canonical and legacy key names are both understood, for example `track_width`/`differentialTrackWidth`, `wheel_radius`/`radius`, `v_max_rad_s`/`vmax`, and `t_max_nm`/`tmax`. New files should use the canonical names.

## Full example

```json
{
  "name": "fll_choreo",
  "version": 4,
  "type": "Differential",
  "robot": {
    "mass": 0.723,
    "inertia": 0.001,
    "track_width": 0.0965,
    "wheel_radius": 0.028,
    "v_max_rad_s": 18.94,
    "t_max_nm": 0.138,
    "gearing": 1.0,
    "cof": 1.37,
    "gravity": 9.81,
    "torque_headroom": 0.85,
    "speed_headroom": 0.90,
    "uncertainties": {
      "cof": [1.24, 1.51],
      "torque_headroom": [0.75, 0.90],
      "gear_backlash_m": [0.0002, 0.0007],
      "inertia_kg_m2": [0.0008, 0.0011],
      "track_width_range": [0.096, 0.097],
      "wheel_radius_range": [0.0275, 0.0285]
    }
  },
  "interactions": {},
  "multiverse": {
    "bootstrap": {
      "turning_radius": 0.0,
      "resolution": 0.05
    }
  }
}
```

## Robot parameters (`robot`)

| Key | Unit | Default | Meaning |
|---|---:|---:|---|
| `mass` | kg | 0.723 | Robot mass. Affects acceleration and normal force. |
| `inertia` | kg·m² | 0.0024 | Rotational inertia. Affects the force split between wheels during turns. |
| `track_width` | m | 0.0965 | Distance between wheel centers. |
| `wheel_radius` | m | 0.028 | Drive-wheel radius. |
| `v_max_rad_s` | rad/s | 15.7 | Motor no-load speed. Convert RPM with `RPM * 2π / 60`. |
| `t_max_nm` | N·m | 0.04 | Motor stall torque. |
| `gearing` | — | 1.0 | Motor-to-wheel ratio. Values above 1 trade speed for torque. |
| `cof` | — | 0.45 | Nominal wheel/field friction coefficient. Measure it; FLL mats are often around 1.0-1.5, but the value depends on wheels and surface. |
| `gravity` | m/s² | 9.81 | Gravitational acceleration. |
| `torque_headroom` | — | 0.85 | Fraction of the motor torque curve available to the planner. |
| `speed_headroom` | — | 0.90 | Fraction of no-load wheel speed available to the planner. |

### Force model

For each wheel, the available force is

```text
omega = (v_wheel / wheel_radius) * gearing
torque = t_max_nm * max(0, 1 - |omega| / v_max_rad_s)
force = (torque / wheel_radius) * gearing * torque_headroom
```

The optimizer and validator use the same per-wheel normal-force model:

```text
nl = m*g/2 - m*h_cg*(v*omega)/track_width
nr = m*g/2 + m*h_cg*(v*omega)/track_width
```

Only lateral (turning) weight transfer is modeled. A two-wheel differential robot has no front/rear axle, so longitudinal acceleration does not shift load between the left and right wheels (KI-010). `h_cg` is a shared constant (`CG_HEIGHT_M = 0.05 m`).

### Uncertainties

The `uncertainties` block is used in two places:

- The optimizer's per-wheel force and traction limits use the declared worst cases (minimum cof, minimum torque headroom, smallest wheel radius for the speed term, largest wheel radius for the force term, maximum inertia and minimum track width for the force split).
- `--validate` passes the same ranges to Immrax.

The optional `--sysid-fit sysid_fit.json` model is separate from this file. It is produced by `sysid_fit.py` from robot logs and is not a `robot` config block. `sysid_fit.py --profile examples/drivebase_profile.example.json` can read the minimal wheel-radius/track-width profile instead of a full robot config. After a run, `sysid_compare.py` compares the planned trajectory with measured `poses.csv` rows.

A minimal drivebase profile is:

```json
{
  "name": "My differential drive",
  "wheel_radius_m": 0.028,
  "track_width_m": 0.0965
}
```

`wheel_radius_m` is required for fitting; `track_width_m` is optional metadata. Pybricks-style `wheel_diameter_mm` and `axle_track_mm` are accepted, as are a full robot config (`robot.wheel_radius`, `robot.track_width`) and an existing `sysid_fit.json`. `sysid_compare.py --profile ...` prints the profile and warns when its geometry differs from the trajectory config. The comparator also accepts a plain predicted-pose CSV (`t,x,y,heading`, or `t_ms`/`heading_deg`) instead of a `.traj`.

Recognized ranges:

| Key | Default if omitted |
|---|---|
| `cof` | `[0.8 * cof, 1.2 * cof]` |
| `torque_headroom` | `[0.8, 1.0]` |
| `gear_backlash_m` | `[0.0001, 0.0006]` |
| `track_width_range` | `[0.99 * track_width, 1.01 * track_width]` |
| `wheel_radius_range` | `[0.99 * wheel_radius, 1.01 * wheel_radius]` |
| `inertia_kg_m2` | `[0.90 * inertia, 1.10 * inertia]` |

`examples/fll_choreo.json` currently narrows `track_width_range` and `gear_backlash_m` for repeatable tests; those values are test assumptions, not measured hardware ranges. The parser also accepts `t_max_range`, but the current optimizer, CPU validator, and Immrax path use `t_max_nm` plus `torque_headroom` rather than this range. Use measured values and keep the ranges honest; a narrow range makes the validator optimistic.

## Multi-Verse parameters (`multiverse`)

> **Current status:** the Multi-Verse pipeline runs and its refined guess now reaches the solver as a **seed** — the polish probes it, and `main.py --refine` adds it to the candidate set (KI-004). `--workers` and `--no-parallel` still do not change the final trajectory. The only `multiverse` block read by the current code is `bootstrap`.

### `bootstrap` (used)

- `turning_radius` (m): Minimum Reeds-Shepp turning radius. `0.0` allows point turns.
- `resolution` (m): Discretization step for the generated geometry.

### Reserved / currently ignored

These keys may appear in older configs but are not read by the current pipeline:

- `enable_parallel`, `num_workers`
- `stomp_variants`, `stomp_noise.position_std`, `stomp_noise.heading_std`
- `teb_weights` (`forward_bias`, `reverse_bias`, `point_turn_bias`, `wide_sweep_bias`)
- `critic_thresholds` (the critic uses module-level constants in `multiverse_optimizer.py`)

The JAX candidate generator currently uses a fixed 50 variants, fixed noise constants, and the critic's hardcoded thresholds. Treat these keys as reserved until the code reads them.

## Waypoint file format

Waypoints can be dictionaries or `[x, y, heading]` arrays:

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0 },
  { "x": 1.0, "y": 0.5, "heading": 0.5 }
]
```

### Fields

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | float, required | Position in meters. |
| `heading` | float or null | Heading in radians. Omit or use `null` for an unconstrained heading. |
| `stop` | bool | Force both wheel speeds to zero at this waypoint. |
| `event` | string | Event name exported with the waypoint sample. |
| `error_threshold` | float | Local reachability-envelope threshold in meters for `--validate`. It governs the way *into* this waypoint: the segment that ends here is held to it. |
| `precision_mode` | bool | Shortcut for a 0.005 m local envelope threshold. Ignored if `error_threshold` is also present. |

`--stop-waypoints` and `--events` override or add to the JSON settings. Indices are zero-based.

### Example with stops and events

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0, "stop": true },
  { "x": 0.5, "y": 0.3, "heading": 0.5, "event": "lower_arm", "error_threshold": 0.01 },
  { "x": 1.0, "y": 0.0, "heading": 0.0, "stop": true }
]
```

> **Runtime caveat:** an event on the final waypoint is exported but not fired by the deployed Pybricks loop (KI-012).

## Command line interface

Run from the repository root:

```bash
python main.py --simple -c examples/fll_choreo.json -w examples/example_straight.json -o output.traj
```

### Input/output

- `-c, --config`: Robot configuration JSON.
- `-w, --waypoints`: Waypoint JSON.
- `-o, --output`: Output `.traj` path (default `output.traj`).

### Solver

- `-n, --samples`: Samples per segment (default 10).
- `-a, --accuracy-weight`: Smoothness/jerk weight (0 = time-optimal).
- `--candidates`: Candidates to solve and compare with `--validate` (default 2, max 5). With the simple solver each candidate combines an initial-guess seed (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`) with a timing mode: by default candidate 1 is equal-duration baseline, candidate 2 is per-segment baseline, then the remaining seeds alternate. The Multi-Verse path keeps the `accuracy_weight` ladder. A passing candidate wins by smallest envelope, then shortest time; `1` disables the comparison.
- `--stop-waypoints "0,2,4"`: Comma-separated stop indices.
- `--events "1:lower_arm,3:release"`: Comma-separated event mappings.
- `--simple`: CasADi-only optimizer (recommended).
- `--per-segment-dt`: Force each waypoint segment to have its own sample interval (simple solver only). This is the general timing model. The interval may only change across a waypoint where the robot stops *and* the heading is pinned; everywhere else the wheel force profile must be continuous through the point (KI-024). By default candidates mix both modes.
- `--uniform-dt`: Force one shared sample interval across all segments (simple solver only). Because every segment has the same sample count, this also forces every segment to take the SAME time, so a path with unequal segment lengths is slow. By default candidates mix both modes. Use it to reproduce the equal-duration-only behavior.
- Timing tradeoff: an uneven 1.0 m + 0.2 m path passes all gates with per-segment at `accuracy_weight=0.005` in 2.784 s / 0.32 cm envelope / 0.36 cm tracking / 9.52 m/s³ peak wheel jerk, versus 4.456 s / 0.37 cm / 0.36 cm / 9.52 m/s³ for the equal-duration mode at the same weight, so the mixed loop selects per-segment. At `accuracy_weight=0` the per-segment candidate is 2.710 s / 0.32 cm but fails the Immrax and CPU gates, so the mixed loop falls back to the equal-duration candidate. The complex mission at `accuracy_weight=0` is faster with per-segment (4.650 s / 1.00 cm) but its tracking is 0.84 cm and its motor force is 0.074 N over the limit, so the equal-duration 6.068 s / 1.22 cm candidate is selected. The CPU audit runs per candidate, so a partial pass cannot block a later fully passing candidate. Both modes are held to the same wheel-jerk discipline, so `accuracy_weight` means the same thing in either (KI-022).
- `--no-parallel`, `--workers`: Multi-Verse options; currently no effect on the result (KI-004).

### Validation and export

- `--validate`: CPU audit + Immrax + final-waypoint gate. Requires JAX/Immrax. With `--candidates N` (default 2, max 5), solves N candidates (seed/timing combinations with `--simple`, accuracy-weight variants otherwise) and exports the passing one with the smallest envelope; on failure, exits 1 and writes nothing.
- `--sysid-fit PATH`: Optional `sysid_fit.json` measured actuator/odometry model; requires `--validate` or `--live`. The default kinematic model is used when omitted. Slip is loaded/reported but not applied yet (KI-020).
- `--export-format controller|python|python-compact`: Write a fixed-timestep controller JSON, a legacy list-of-dicts Python module, or a packed-compact Python module. The compact module is consumed by `pybricks_code/ramsete_compact.py` or `ramsete_compact_events.py`.
- `--controller-dt`: Controller timestep in seconds (default 0.02).
- `--benchmark`: Write `<output>_stats.json` after a successful export gate.

### Visualization

- `--plot`, `--animate`: Blocking Matplotlib views.
- `--live`: Blocking WebSocket live view on port 8765. It kills the process that owns the port (KI-005), and the first client misses config/waypoints (KI-016).
- `--show-convergence`: Works with the simple optimizer.
- `--convergence-mode`: `parallel`, `best`, or `layered`.
- `--convergence-animate`, `--convergence-output`: Parsed but currently unused (KI-017).
- `-q, --quiet`: Suppress progress output.

## Installation note

`pip install -r requirements.txt` covers the simple optimizer. `--validate`, `--multiverse` and `--refine` need `jax`, `jaxopt`, and `immrax`. A plain solve can run without JAX, but live reachability envelopes require it. The optional packages live in `requirements-optional.txt` (KI-006). JAX 0.6.2 was used for the current tests. `install.bat` and `install.sh` install both sets into a `.venv` and check the result, and `python main.py --doctor` reports what is missing and prints the install command.
