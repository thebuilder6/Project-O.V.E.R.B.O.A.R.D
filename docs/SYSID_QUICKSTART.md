# Drivebase Profile and SysId Quick Start

This is the short path for calibrating a differential drive without editing
project constants or using the example FLL robot's full config. The robot-side
collector is still the Pybricks reference implementation; the host fitter,
validator, and comparator are drivebase-agnostic.

## 1. Create your drivebase profile

Copy `examples/drivebase_profile.example.json` to a working file and measure
your robot:

```json
{
  "name": "My differential drive",
  "wheel_radius_m": 0.028,
  "track_width_m": 0.0965
}
```

- `wheel_radius_m`: wheel radius in metres. Required for fitting.
- `track_width_m`: centre-to-centre distance between the left and right wheels,
  in metres. Optional, but enables the geometry check.
- Pybricks-style `wheel_diameter_mm` and `axle_track_mm` are accepted.
- A full robot config or an existing `sysid_fit.json` can also be passed to
  `--profile`.
- `--wheel-radius` overrides the profile for a quick check.

## 2. Collect one profile session

Run `pybricks_code/sysid_digital_twin.py` on the Hub, not with host Python. It
wraps the drivebase from `robot_config.py`, runs the step/turn/odometry/brake
tests, and prints one or more `TEST_START` / `TEST_END` blocks.

Save the complete console capture as `logs/session.csv`.

Measure the final pose of each run and write `poses.csv`:

```csv
run_id,stop,x_m,y_m,heading_deg
1,0,0.998,0.002,0.0
```

`run_id` is the value printed by `META,run_id`. For a `multi_stop` run, add one
row per stop in order (`stop` = 0, 1, 2, ...). If you combine session files,
prefix the printed id with the file stem, for example `session1:1`.

## 3. Fit the model

From the repository root:

```bash
python sysid_fit.py logs/session.csv --poses poses.csv \
  --profile my_drivebase.json --out sysid_fit.json
```

The fitter prints the straight/turn gain, time constant, command delay,
odometry scale, and brake-slip estimate. It writes those parameters plus a
`drivebase` block into `sysid_fit.json`.

Existing full configs still work:

```bash
python sysid_fit.py logs/session.csv --robot-config examples/fll_choreo.json \
  --out sysid_fit.json
```

Do not pass both `--profile` and `--robot-config`.

## 4. Validate with the fitted model

```bash
python main.py --simple -c examples/fll_choreo.json \
  -w examples/example_straight.json -o output.traj \
  --validate --sysid-fit sysid_fit.json
```

`--sysid-fit` requires `--validate` or `--live` and JAX/Immrax. Validation is a
hard gate: a failed solve or failed validation writes no `.traj`, controller,
Python file, or `_stats.json`.

`main.py` prints the profile and warns if the fitted wheel radius or track width
differs from the robot config. The validator still uses the robot config for
geometry; the profile does not override it.

## 5. Compare planned and measured poses

```bash
python sysid_compare.py output.traj poses.csv --run-id 1 \
  --profile my_drivebase.json
```

For a multi-stop run, map the measured rows to the trajectory:

```bash
python sysid_compare.py output.traj poses.csv --run-id 12 \
  --waypoints examples/example_with_stops.json --profile my_drivebase.json
```

Or pass explicit sample indices:

```bash
python sysid_compare.py output.traj poses.csv --run-id 12 \
  --stop-indices 20,40 --profile my_drivebase.json
```

The comparator also accepts any predicted-pose CSV, so it is not tied to the
optimizer's `.traj` format:

```csv
t,x,y,heading
0.0,0.0,0.0,0.0
1.0,0.2,0.0,0.0
```

`t` is in seconds and `heading` is in radians; `t_ms` and `heading_deg` are also
accepted (use `t_ms,x,y,heading_deg`). Pass `--profile` to print the drivebase
and check geometry.

The report is diagnostic only. It prints position and heading errors, says
whether each measured pose is inside the reachability envelope, and does not
change the 5 mm tracking or 2 cm envelope thresholds.

## Common fixes

- `provide --profile with wheel_radius_m`: add the wheel radius to the profile
  or pass `--wheel-radius`.
- `has no drivebase geometry`: the profile has neither a wheel radius nor a
  track width.
- `run_id ... not found`: check the `META,run_id` values and any session-file
  prefixes in `poses.csv`.
- `multiple measured poses need --waypoints or --stop-indices`: add the
  mapping that matches the hardware stops.

## Limits

- The Pybricks collector is a reference adapter. A non-Pybricks drivebase needs
  a logger that emits the same CSV blocks.
- Braking slip is loaded and reported but not applied by the plant yet (KI-020).
- There are no checked-in hardware logs; the current fitted-model numbers are
  synthetic or model-only. Calibrate on your own drivebase before trusting the
  envelope.
- Do not tune `b`, `zeta`, or the validation thresholds to a single path.
