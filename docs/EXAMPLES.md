# Usage Examples

This document provides practical examples for using the FLL Trajectory Optimizer.

> **Optimizer choice:** CasADi/IPOPT is the backend and the 3-Pillar unified planner is the only planner on top of it, so every example below works with a core `requirements.txt` install. The old Multi-Verse pipeline was removed (git tag `archive/legacy-2026-10`) and has no CLI entry point. `--validate` needs JAX/Immrax and a scoring timing budget (the shipped missions carry one; a mission of your own declares it in its `scoring` section or gets it from `--target-time`/`--max-time`). With `--validate`, `--candidates N` (default 2) solves N of the planner's frontier candidates, scores each one, and exports the eligible candidate with the highest decision score (a failed one only under `--accept-score N`); the 5-part fitness score is reported for every candidate as a diagnostic. Use `--candidates 1` for a single solve. Without `--validate`, only the first candidate is solved. The score is a 0-100 decision number (see [CONFIGURATION.md](CONFIGURATION.md)), not a probability of hardware success.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Robot variants](#robot-variants)
- [Advanced Examples](#advanced-examples)
- [Python API Examples](#python-api-examples)
- [Troubleshooting Examples](#troubleshooting-examples)
- [Integration Examples](#integration-examples)
- [Browser Studio](#browser-studio)
- [Solving in the Browser](#solving-in-the-browser)

## Basic Examples

### Example 1: Simple Two-Point Path

Generate a straight-line trajectory from (0, 0) to (1, 0):

**waypoints.json:**

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0 },
  { "x": 1.0, "y": 0.0, "heading": 0.0 }
]
```

**Command:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o straight.traj --plot
```

**Expected output:**

- With the example config, the optimizer takes about 0.06 s of solve wall time
- Robot accelerates from rest, cruises, and decelerates to rest
- Trajectory time is about 2.38 s for the 1 m straight example; it depends on the robot's speed limits

---

### Example 2: Three-Point Turn

Generate a trajectory with a turn:

**waypoints.json:**

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0 },
  { "x": 0.5, "y": 0.5, "heading": 0.785 },
  { "x": 1.0, "y": 0.0, "heading": 0.0 }
]
```

**Command:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o turn.traj --plot
```

**Expected output:**

- Robot follows curved path through middle waypoint
- Heading is constrained at all three waypoints
- Slower than straight line due to turn

---

### Example 3: Unconstrained Heading

Generate a path where only positions matter, not heading:

**waypoints.json:**

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0 },
  { "x": 1.0, "y": 1.0, "heading": null },
  { "x": 2.0, "y": 0.0, "heading": 0.0 }
]
```

**Command:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o arc.traj --plot
```

**Expected output:**

- Start and end headings are fixed
- Middle waypoint heading is free (optimizer chooses optimal heading)
- Results in smooth arc through middle point

---

### Example 4: Using Built-in Example Files

The repository includes several example waypoint files:

```bash
# Straight line
python -m overboard.main -c examples/fll_choreo.json -w examples/example_straight.json -o straight.traj --plot

# S-curve path
python -m overboard.main -c examples/fll_choreo.json -w examples/example_s_curve.json -o s_curve.traj --plot

# Complete mission with stops and events
python -m overboard.main -c examples/fll_choreo.json -w examples/example_complete.json -o complete.traj --plot

# Complex mission
python -m overboard.main -c examples/fll_choreo.json -w examples/example_complex_mission.json -o complex.traj --plot

# First four waypoints of the complex mission (candidate-seed benchmark)
python -m overboard.main --validate -c examples/fll_choreo.json -w examples/example_mission_first4.json -o first4.traj --candidates 5
```

With `--validate`, the open-loop `plant` tube grows with distance, so a curved mission can exceed the 2 cm envelope gate: add `--propagation tick` (the `--preset safe` shortcut sets it for you). The straight line still passes with the default `plant`.

### Measured scoring budgets of the shipped missions

Every shipped mission carries its scoring timing in its `scoring` section, so
a `--validate` run needs no extra flags. The budgets are **measured, not
guessed**: `target_time_s` is `T_ref`, the duration of the fastest converged
candidate of the shipping planner on the clean baseline tree (the 3-Pillar
candidates `frontier_sprint`, `frontier_sweeper`, `frontier_exploratory`,
solved with the CLI defaults: `examples/fll_choreo.json`, automatic mesh,
per-segment dt, no validation), and `max_time_s` is `2 * T_ref`.

| Mission | `T_ref` (s) | `target_time_s` | `max_time_s` |
|---|---|---|---|
| `examples/example_straight.json` | 2.299253443234989 | 2.299253443234989 | 4.598506886469978 |
| `examples/example_s_curve.json` | 3.817753723929007 | 3.817753723929007 | 7.635507447858014 |
| `examples/example_mission_first4.json` | 2.051747299801067 | 2.051747299801067 | 4.103494599602134 |
| `examples/example_complex_segment.json` | 2.1818234038781523 | 2.1818234038781523 | 4.363646807756305 |
| `examples/example_with_events.json` | 4.453385011684768 | 4.453385011684768 | 8.906770023369535 |
| `examples/example_with_stops.json` | 5.431187277994541 | 5.431187277994541 | 10.862374555989081 |
| `examples/example_complex_mission.json` | 5.7744055179583045 | 5.7744055179583045 | 11.548811035916609 |
| `examples/example_complete.json` | 11.674326703354728 | 11.674326703354728 | 23.348653406709456 |
| `missions/realistic_complex_mission.json` | 18.87646664557883 | 18.87646664557883 | 37.75293329115766 |

Re-measure and rewrite the budgets when the planner or the robot config
changes: the duration score only means something against the plan the current
tree produces.

---

## Robot variants

Three ready-made configs in `examples/robots/` for testing a plan against a different build, instead of editing `fll_choreo.json`:

| Config | What it represents | What it tests |
|---|---|---|
| `tuned.json` | A measured, healthy robot: nominal physics, tight ranges | The reference case; a plan that fails here is the planner or the mission |
| `slippery_underpowered.json` | Same geometry, COF 0.55 and 55 % torque headroom | Force and slip gates - the plan should slow down, not export |
| `worn.json` | Heavier, 0.6-1.2 mm backlash, ±1 mm geometry | The uncertainty path: a good plan can still fail the envelope |

```bash
python -m overboard.main -c examples/robots/slippery_underpowered.json \
  -w examples/example_straight.json --preset safe -o slippery.traj
```

A robot file is a normal `-c` config; nothing else changes. The ranges in these files are declared, not measured (KI-020).

## Advanced Examples

### Example 5: Intermediate Stops (JSON)

Generate a trajectory with stops at specific waypoints using JSON:

**waypoints_with_stops.json:**

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0, "stop": true },
  { "x": 0.5, "y": 0.0, "heading": 0.0, "stop": false },
  { "x": 0.5, "y": 0.3, "heading": 1.57, "stop": true },
  { "x": 1.0, "y": 0.3, "heading": 1.57, "stop": false },
  { "x": 1.0, "y": 0.0, "heading": 0.0, "stop": true }
]
```

**Command:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints_with_stops.json -o stops.traj --plot
```

**Expected output:**

- Robot starts at rest at (0, 0)
- Moves to (0.5, 0) without stopping
- Comes to complete stop at (0.5, 0.3)
- Moves to (1.0, 0.3) without stopping
- Comes to complete stop at final waypoint (1.0, 0)

**Use case:** Multi-segment FLL missions where robot needs to pause between maneuvers (e.g., to wait for a mechanism to complete, to align precisely, or to execute an action).

---

### Example 6: Intermediate Stops (CLI)

Alternatively, specify stops via CLI flag:

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o stops.traj \
  --stop-waypoints "0,2,4" --plot
```

This forces the robot to stop at waypoints 0, 2, and 4 (0-indexed).

---

### Example 7: Event Markers (JSON)

Generate a trajectory with mission event markers using JSON:

**waypoints_with_events.json:**

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0 },
  { "x": 0.5, "y": 0.3, "heading": 0.5, "event": "lower_arm" },
  { "x": 1.0, "y": 0.0, "heading": 0.0 },
  { "x": 1.5, "y": 0.3, "heading": 0.0, "event": "release" }
]
```

**Command:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints_with_events.json -o events.traj \
  --export-format controller \
  --controller-dt 0.02
```

**Expected output:**

- Trajectory file contains `"event"` field at waypoint samples
- Controller export preserves events at nearest timestep
- Robot controller reads events and triggers actions (arm, intake, release)

> **Final event:** an event on the final waypoint is fired after the loop, once the robot has arrived and the drivebase has stopped (KI-012, fixed).

---

### Example 8: Event Markers (CLI)

Alternatively, specify events via CLI flag:

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o events.traj \
  --events "1:lower_arm,3:release" \
  --export-format controller \
  --controller-dt 0.02
```

Format: `--events "index:event,index:event,..."` where index is the 0-based waypoint index.

---

### Example 9: Accuracy Weighting

Compare time-optimal vs. smooth trajectories:

**Time-optimal (default):**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o fast.traj -a 0.0 --validate
```

**Balanced smoothness:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o smooth.traj -a 1.0 --validate
```

**Very smooth:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o very_smooth.traj -a 5.0 --validate
```

**Comparison:**

- `a=0.0`: Fastest time, highest jerk, may overshoot on real robot
- `a=1.0`: Smoothness cost, better tracking
- `a=5.0`: Significant time cost, very smooth, minimal overshoot

The exact time/jerk tradeoff depends on the path and robot; measure it on your own trajectory. `--validate` can reject a curved path even when the solve succeeds (KI-020), so check the exit code before using the output.

---

### Example 10: Controller Export

Generate a trajectory and export for on-robot execution:

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o traj.traj \
  --validate \
  --export-format controller \
  --controller-dt 0.02
```

**Output files:**

- `traj.traj`: Full variable-timestep trajectory (for analysis)
- `traj_controller.json`: Fixed 20ms timestep (for robot controller)

**Controller file format:**

```json
{
  "format": "controller_profile",
  "version": 1,
  "dt": 0.02,
  "num_samples": 150,
  "samples": [
    {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "vl": 0.0, "vr": 0.0, "v": 0.0, "omega": 0.0},
    {"t": 0.02, "x": 0.001, "y": 0.0, "heading": 0.0, "vl": 0.1, "vr": 0.1, "v": 0.1, "omega": 0.0},
    ...
  ]
}
```

---

### Example 11: Python Export

Export trajectory as a Python file for direct import:

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o traj.traj \
  --export-format python
```

**Output file:** `traj.py` with a `config` dictionary and a `samples` list containing trajectory data (including events when present).

**Use case:** Direct integration with Python-based robot controllers or analysis scripts.

---

### Example 12: Compact Python Export

Export the packed module used by the compact Pybricks controllers:

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o traj.traj \
  --export-format python-compact
```

**Output file:** `traj.py` with a `config` dictionary, a packed `samples` bytes object, and an `events` tuple. The blob uses a 4-byte header plus 24 bytes per sample.

**Use case:** Import the module with `pybricks_code/ramsete_compact.py` for pathing or `pybricks_code/ramsete_compact_events.py` for events. This is the format a pushed run uses: `hub_runner.py` imports `ramsete_compact_events.follow_trajectory`, and `robot.py` menu option 1 runs `hub_runner.py`, so a pushed compact mission is wired into the menu.

---

### Example 13: Varying Sample Density

Compare different sample densities:

**Low density (fast, less accurate):**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o low_res.traj -n 5
```

**Medium density (default):**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o med_res.traj -n 10
```

**High density (slower, more accurate):**

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o high_res.traj -n 20
```

**Tradeoffs:**

- `-n 5`: Faster optimization, but may miss sharp turns
- `-n 10`: Good balance for most FLL paths
- `-n 20`: Slower optimization, smoother curves, better for complex paths

The three frontier candidates are solved and the decision score picks the one to export: the eligible candidate with the highest score, with an ordinary pass preferred over an accepted one when the scores tie (see [CONFIGURATION.md](CONFIGURATION.md)). A waypoint is a legitimate timing boundary only when the robot stops there *and* the heading is pinned; at every other waypoint the per-segment model keeps the wheel force profile continuous through the point (KI-024).

The planner always uses the per-segment model, so there is no timing flag to choose: the
`--per-segment-dt` and `--uniform-dt` flags are removed and rejected as unknown options. What the
timing model changes is the optimization variables, not the waypoint constraints.

---

### Example 14: Complex FLL Mission Path

Generate a trajectory for a typical FLL mission:

**mission_waypoints.json:**

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0 },
  { "x": 0.3, "y": 0.2, "heading": 0.5 },
  { "x": 0.6, "y": 0.4, "heading": 1.0 },
  { "x": 0.9, "y": 0.3, "heading": 0.8 },
  { "x": 1.2, "y": 0.5, "heading": 0.0 }
]
```

**Command:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w mission_waypoints.json -o mission.traj \
  -n 10 \
  -a 0.5 \
  --validate \
  --export-format controller \
  --controller-dt 0.02 \
  --plot
```

**Expected output:**

- Solves are typically a few tenths of a second on the example missions
- The solve may converge while `--validate` still rejects the path; which example missions pass is a measurement that moves with the config and the planner (see the ledger in `docs/HEURISTICS_AND_VERIFICATION.md` and KI-020), so check the exit code rather than assuming a list
- If validation passes, the controller profile has one sample per 20 ms of trajectory

Run without `--validate` for exploratory output, or use a path that passes the gate before exporting to the robot.

---

## Python API Examples

### Example 15: Using the Optimizer Directly

```python
from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer
import json

# Load configuration
with open('examples/fll_choreo.json', 'r') as f:
    config_data = json.load(f)

robot_cfg = RobotConfig(config_data)
optimizer = TrajectoryOptimizer(robot_cfg)

# Define waypoints
waypoints = [
    (0.0, 0.0, 0.0),
    (1.0, 0.5, 0.5),
    (2.0, 1.0, 1.0)
]

# Solve with accuracy weighting. solve() returns (samples, stats).
samples, stats = optimizer.solve(
    waypoints,
    num_samples_per_segment=10,
    accuracy_weight=1.0
)

# Print results
print(f"Total time: {samples[-1]['t']:.3f}s")
print(f"Solve wall time: {stats['total_time']:.3f}s")
print(f"Max speed: {max(s['vl'] + s['vr'] for s in samples) / 2:.3f} m/s")
```

---

### Example 16: Validating a Trajectory

```python
from overboard.verification.validator import validate_trajectory

# Run validation
metrics, audit, errors = validate_trajectory('output.traj', 'examples/fll_choreo.json')

print("\n=== Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\n=== Constraint Audit ===")
print(f"Violating samples: {audit['num_violating_samples']}")
if audit['num_violating_samples'] > 0:
    print(f"Max left motor violation: {audit['left_motor_force']:.6f} N")
    print(f"Max right motor violation: {audit['right_motor_force']:.6f} N")
    print(f"Max traction violation: {audit['traction_total']:.6f} N")

print("\n=== Forward Integration Errors ===")
print(f"Max position error: {errors['max_pos_error_m']:.6f} m")
print(f"Final position error: {errors['final_pos_error_m']:.6f} m")

# Check if trajectory is safe. This mirrors validate_samples().
is_safe = (
    errors['max_pos_error_m'] < 0.01 and
    errors['final_pos_error_m'] < 0.01 and
    audit['num_violating_samples'] == 0 and
    audit['num_slip_points'] == 0 and
    audit['speed_violation_m_s'] < 1e-6
)
print(f"\nTrajectory is {'SAFE' if is_safe else 'UNSAFE'}")

# The CLI also checks the final waypoint. Pass waypoints=... to include it here.
```

---

### Example 17: Exporting for Controller

```python
from overboard.deploy.export import write_controller_file
import json

# Export with 20ms timestep
write_controller_file(
    'output.traj',
    'controller_profile.json',
    target_dt=0.02,
    track_width=0.0965
)

# Load and inspect
with open('controller_profile.json', 'r') as f:
    ctrl_data = json.load(f)

print(f"Controller profile: {ctrl_data['num_samples']} samples at {ctrl_data['dt']}s dt")
print(f"First sample: {ctrl_data['samples'][0]}")
```

---

### Example 18: Comparing Accuracy Weights

```python
from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer
from overboard.verification.validator import compute_metrics
import json

# Load config
with open('examples/fll_choreo.json', 'r') as f:
    config_data = json.load(f)

robot_cfg = RobotConfig(config_data)
optimizer = TrajectoryOptimizer(robot_cfg)

waypoints = [
    (0.0, 0.0, 0.0),
    (1.0, 0.5, 0.5),
    (2.0, 1.0, 0.0)
]

# Test different accuracy weights
weights = [0.0, 0.5, 1.0, 2.0, 5.0]

print("Accuracy Weight | Time (s) | Max Accel (m/s²) | Max Jerk (m/s³)")
print("-" * 65)

for w in weights:
    samples, stats = optimizer.solve(waypoints, num_samples_per_segment=10, accuracy_weight=w)
    metrics = compute_metrics(samples, robot_cfg)
    print(f"{w:14.1f} | {metrics['total_time_s']:7.3f} | {metrics['max_accel_m_s2']:15.3f} | {metrics['max_jerk_m_s3']:13.3f}")
```

**Typical output:** exact values depend on the robot and path. The table is intentionally not hardcoded; record the output from your own run. Increasing the weight usually reduces jerk at the cost of a longer trajectory.

### Example 19: Comparing Initial-Guess Seeds and Timing Modes

Each candidate combines an initial-guess seed with a timing mode. By default candidate 1 is equal-duration baseline, candidate 2 is per-segment baseline, and the remaining seeds alternate modes. To compare only the five seeds with equal-duration timing, force that mode:

```bash
python -m overboard.main --validate -c examples/fll_choreo.json \
  -w examples/example_mission_first4.json -o first4.traj --candidates 3
```

The five seeds (`baseline`, `rolling`, `frontier_sprint`, `frontier_sweeper`, `frontier_exploratory`) were the rescue round's candidates until 2026-10-21 (KI-048, since removed); they are still declared in `overboard/core/optimizer.py`, and nothing solves them. The measurements below are the record of the stock candidate set, which was retired with the rest of the multi-track paths: on the first four waypoints of the complex mission all five uniform seeds passed, and `baseline` won at 2.176 s / 0.27 cm; on the full mission the baseline also won (6.068 s / 1.22 cm), the other uniform seeds converging to slower, wider paths (8.1-8.7 s), so they were rejected. The per-segment baseline was faster (1.833 s on first4, and 4.650 s on the mission), but on the mission its tracking was 0.84 cm and its motor force 0.074 N over the limit, so it was rejected there.

From Python, pass `guess_style` to `solve`; the default `per_segment_dt=False` reproduces the uniform seed comparison:

```python
for style in ("baseline", "rolling", "frontier_sprint", "frontier_sweeper", "frontier_exploratory"):
    samples, stats = optimizer.solve(waypoints, 10, 0.0, guess_style=style)
```

The seeds only change the initial guess; every waypoint, stop, and force limit is still enforced by the solver.

---

## Troubleshooting Examples

### The program does not start at all

If `python -m overboard.main` dies with `ModuleNotFoundError: No module named 'click'`
(or any other package), nothing is installed yet. Run the install script from
the project folder:

```bash
install.bat            # Windows
sh install.sh          # macOS/Linux
```

It creates a `.venv`, installs everything the solver and the studio need, and
runs the check at the end. `python -m overboard.main --doctor` reports the same thing
without needing any package, and it names the exact command that fixes what is
missing.

---

### Example 20: Debugging Optimization Failure

If optimization fails or times out:

```bash
# First, try with fewer samples
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o test.traj -n 5

# If that works, gradually increase
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o test.traj -n 10
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o test.traj -n 15
```

Check for:

- Infeasible waypoints (too far apart for robot speed)
- Incorrect config parameters (wrong units, unrealistic values)
- Conflicting constraints (e.g., stops too close together)

---

### Example 21: Checking Constraint Violations

If validation shows constraint violations:

```bash
# Run validation to see details
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o output.traj --validate
```

**Output interpretation:**

- `left_motor_force > 0`: Left wheel force exceeds motor capability
- `right_motor_force > 0`: Right wheel force exceeds motor capability
- `left_wheel_slip` / `right_wheel_slip > 0`: Per-wheel friction limit exceeded
- `traction_total > 0`: Legacy total-force limit exceeded
- `max_wheel_speed_m_s > wheel_speed_limit_m_s`: Commanded speed exceeds the no-load bound

**Fixes:**

- Reduce `v_max_rad_s` or `t_max_nm` in the config
- Measure and increase `cof` only if the surface really supports it
- Add accuracy weight to smooth the trajectory
- Increase sample density for better resolution
- Adjust `torque_headroom` or `speed_headroom`
- For curved paths, expect the Immrax gate to reject the current model (KI-020) even when the CPU audit is clean

---

### Example 22: Robot Tracking Poorly

If the robot deviates significantly from the planned path:

**Symptoms:**
- Large position errors (> 10 mm)
- Consistent overshoot at waypoints
- Robot slips during turns

**Solutions:**

1. **Increase accuracy weight:**
```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o smooth.traj -a 2.0
```

2. **Check configuration:**
- Verify wheel radius (measure under load)
- Verify motor specs (vmax, tmax)
- Check friction coefficient for actual surface

3. **Add headroom:**
```json
{
  "robot": {
    "torque_headroom": 0.80,
    "speed_headroom": 0.85
  }
}
```

4. **Increase sample density:**
```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o high_res.traj -n 20
```

---

## Integration Examples

### Example 23: Batch Processing Multiple Paths

```python
import os
import json
from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer

# Load config once
with open('examples/fll_choreo.json', 'r') as f:
    config_data = json.load(f)
robot_cfg = RobotConfig(config_data)
optimizer = TrajectoryOptimizer(robot_cfg)

# Process multiple waypoint files
waypoint_files = [
    'mission1_waypoints.json',
    'mission2_waypoints.json',
    'mission3_waypoints.json'
]

for wp_file in waypoint_files:
    with open(wp_file, 'r') as f:
        wp_data = json.load(f)

    waypoints = [(w['x'], w['y'], w.get('heading')) for w in wp_data]

    samples, stats = optimizer.solve(waypoints, num_samples_per_segment=10, accuracy_weight=0.5)

    output_file = wp_file.replace('waypoints.json', 'traj.traj')
    result = {
        "name": os.path.basename(output_file).split('.')[0],
        "version": 3,
        "trajectory": {
            "config": config_data.get("robot", config_data.get("config", config_data)),
            "samples": samples
        }
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=1)

    print(f"Generated {output_file}")
```

> **Note:** `TrajectoryOptimizer.solve()` returns `(samples, stats)` and is the supported batch path.

---

### Example 24: Generating Comparison Plots

```python
from overboard.viz.plotter import plot_trajectory
import json

# Load trajectory
with open('output.traj', 'r') as f:
    traj_data = json.load(f)

samples = traj_data['trajectory']['samples']

# Load waypoints
with open('waypoints.json', 'r') as f:
    wp_data = json.load(f)

waypoints = [(w['x'], w['y'], w.get('heading')) for w in wp_data]

# Plot with waypoints overlay
plot_trajectory(samples, waypoints=waypoints, title="FLL Mission Trajectory")
```

---

### Example 25: Complete FLL Mission (All Features)

Generate a comprehensive trajectory using all available features: stops, events, accuracy weighting, validation, controller export, and plotting.

**complete_mission.json:**

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0, "stop": true },
  { "x": 0.4, "y": 0.2, "heading": 0.5, "event": "lower_arm" },
  { "x": 0.8, "y": 0.4, "heading": 0.8, "stop": true },
  { "x": 1.2, "y": 0.3, "heading": 0.0, "event": "release" },
  { "x": 1.5, "y": 0.0, "heading": -0.5, "stop": false },
  { "x": 1.8, "y": 0.2, "heading": 0.5, "event": "intake" },
  { "x": 2.0, "y": 0.5, "heading": 1.57, "stop": true },
  { "x": 1.8, "y": 0.8, "heading": 3.14, "event": "deposit" },
  { "x": 1.5, "y": 0.5, "heading": -1.57, "stop": false },
  { "x": 1.2, "y": 0.3, "heading": 0.0, "stop": true }
]
```

**Command:**

```bash
python -m overboard.main -c examples/fll_choreo.json -w complete_mission.json -o examples/complete_mission.traj \
  -a 1.0 \
  -n 15 \
  --validate \
  --export-format controller \
  --controller-dt 0.02 \
  --plot
```

**Features demonstrated:**

- **Stops**: Robot comes to rest at waypoints 0, 2, 6, and 9 (indices with `"stop": true`)
- **Events**: Mission actions triggered at waypoints 1 (lower_arm), 3 (release), 5 (intake), and 7 (deposit)
- **Accuracy weighting**: `-a 1.0` balances time-optimality with smoothness for better tracking
- **Sample density**: `-n 15` provides high resolution for complex path
- **Validation**: `--validate` checks constraint violations and forward integration errors
- **Controller export**: `--export-format controller` resamples to 20ms timestep for robot
- **Plotting**: `--plot` visualizes the trajectory with waypoints overlay

> **Status is a measurement, not a promise:** which example missions pass the Immrax gate moves with the config and the planner. Always check the exit code: failed `--validate` runs exit 1 and write nothing. Run without `--validate` only for exploratory output, or use a path that passes the gate before exporting. This is KI-020, not a solver crash; the latest pass/fail ledger is in `docs/HEURISTICS_AND_VERIFICATION.md`.

> **Final event:** waypoint 7 is not the final waypoint here; an event placed on the final waypoint would fire after the loop, once the robot has arrived and the drivebase has stopped (KI-012, fixed).

**Output files:**

- `examples/complete_mission.traj`: Full variable-timestep trajectory with events
- `examples/complete_mission_controller.json`: Fixed 20ms timestep for robot controller with events

**Use case:** Complete FLL mission with multiple segments, stops for precision, events for mechanism control, and validation for safety.

---

## Browser Studio

### Example 26: Draw a Mission in the Browser

**Command:**

```bash
python -m overboard.main --gui
```

The browser opens at `http://127.0.0.1:8000` (use `--gui-port` for a different port).

**Steps:**

1. Double-click the field to add waypoints, drag them to move them, and use the right-hand panel to set a heading, a stop, an event name or an envelope limit.
2. Pick a preset: **fast** (one candidate with the default settings, no validation) or **safe** (three candidates, validated with the closed-loop tick map, controller export).
3. Press **Solve**. The solver log streams into the panel; the trajectory, the reachability envelope and the numbers appear on the field.
4. Download the trajectory, the controller profile or the Python module from the result panel.
5. **Save** writes the mission to `missions/<name>.json`; the file shows up in the mission list next time. **Download** gives you the same JSON as a file.

**Before you solve:**

- The mission panel shows the path length and the fastest the robot could possibly do it: *"Path 1.36 m | at least 2.85 s at 0.48 m/s top speed"*. A waypoint dragged 5 m away shows up here, not in the solve time.
- **Fast polish** in the Solve tab's Solved group gives the warm-start solve an iteration budget, and solves the mission cold from the warm stage's iterate if the budget runs out. On the ten-waypoint complex mission the plain solve takes about 1257 iterations and 33 s on the current tree, and the fast path 433 iterations and 12.5 s, for a trajectory within 1 % of the plain one (6.096 s against 6.087 s). Leave it off when you want the plain solver.

**On the field:**

- A **stop** is a blue square, an **event** a pink diamond, both with their name next to the waypoint, so you can see where the robot rests and what fires.
- A segment that failed validation is drawn in red, the segment where the path drifts most in amber.
- While the solver runs, the header ticks the elapsed time and names the stage: *"Solving... - 4.2 s - 3 candidates in parallel"*.

**Which candidate won, and why:**

- The line above the candidate chips says it in words: *"equal-duration baseline won (7.21 s). per-segment baseline was 0.48 s faster but failed: tracking error 0.61 cm > 0.50 cm"*. When nothing passes it says *"no candidate passed; equal-duration baseline came closest (3.70 s)"*.
- Each chip carries the candidate's duration and its pass/fail state; hover it for the reasons.

**When a validation fails:**

- The result panel shows a row per path segment: the reachability envelope, the local limit and the drift.
- A segment over its limit is drawn in red on the field, and the segment where the path drifts most is drawn in amber.
- The note under the table says which gate failed and by how much, and the candidates can be compared with the chips above the table.
- With **Keep the best path when validation fails** ticked (the default), the best candidate that failed on accuracy alone stays on the field and can be downloaded, marked *Failed validation*: it is for inspection, not approval. Untick it, or fail on force, slip, integration or convergence, and the export gate stays closed and nothing is downloaded.

**Editing the robot:**

- Open **Edit robot** under the robot facts. Mass, inertia, wheel radius, track width, motor speed and torque, gearing, friction and the two headrooms are editable; lengths are in millimetres.
- The uncertainty ranges live under it. They are what the validator plans for, so keep them honest: a narrow range makes the check optimistic.
- The edited robot is what the solver receives. **Download config** saves a copy as JSON.

**Your work is never lost:**

- Every edit is undoable with Ctrl+Z (redo with Ctrl+Shift+Z), and the Undo/Redo buttons sit next to New.
- The mission is stored in the browser, so a reload or a crash brings it back exactly as you left it.
- The panel under the mission name says whether the mission is saved to `missions/` or changed since the last save.
- Ctrl+S saves to `missions/`, Space plays the trajectory, Delete removes the selected waypoint.

**Notes:**

- The studio runs the same command line in the background, so the export gate, the validation rules and the file formats are identical to the terminal path. A failed validation shows the reason; the best path that failed on accuracy alone is kept on the field and marked *Failed validation*, and is never offered for the hub.
- It needs no extra packages and no internet connection.
- It leaves the backend at the CLI default and forwards no planner or collocation switch; there is no Multi-Verse toggle.

---

## Solving in the Browser

### Example 27: Solve a Mission in the Studio

Open the studio on a mission and solve it there, with the same options the command line would use:

**Command:**

```bash
python -m overboard.main --gui
```

**Steps:**

1. Run the command. The studio opens in your browser with that mission loaded.
2. The solve starts on its own; the header ticks the elapsed time and the result panel shows the trajectory, the envelope and the numbers.
3. Everything the command line can do is there: validate, export the controller, edit the robot, draw more waypoints.

**Notes:**

- `--gui` opens the studio. The mission is loaded in the browser and the solve is started from there; the CLI's own wrappers for it (`--live`, `--show-convergence`, `--convergence-mode`) are gone, because the studio does all three natively.
- The server is blocking: stop it with Ctrl+C.
- The studio runs the command line in a subprocess, so the export gate and the validation rules are the same as on the terminal.

**Use case:** Watching a solve you started from the terminal, or giving a mission to someone who will not read a README.
