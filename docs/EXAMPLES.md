# Usage Examples

This document provides practical examples for using the FLL Trajectory Optimizer.

> **Optimizer choice:** most examples work with either solver. Use `--simple` for the CasADi-only path, especially if you installed only `requirements.txt`. The default Multi-Verse path needs JAX and currently produces the same final trajectory (KI-004). `--validate` always needs JAX/Immrax. With `--validate`, `--candidates N` (default 2, max 5) solves N candidates and exports the passing one with the smallest envelope; with `--simple` the candidates are distinct initial-guess seeds, otherwise accuracy-weight variants. Use `--candidates 1` for a single solve. Without `--validate`, only the first candidate is solved.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Advanced Examples](#advanced-examples)
- [Python API Examples](#python-api-examples)
- [Troubleshooting Examples](#troubleshooting-examples)
- [Integration Examples](#integration-examples)
- [Live Visualization](#live-visualization)

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
python main.py --simple -c examples/fll_choreo.json -w waypoints.json -o straight.traj --plot
```

**Expected output:**

- With the example config, the simple solver takes about 0.06 s of solve wall time
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o turn.traj --plot
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o arc.traj --plot
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
python main.py -c examples/fll_choreo.json -w examples/example_straight.json -o straight.traj --plot

# S-curve path
python main.py -c examples/fll_choreo.json -w examples/example_s_curve.json -o s_curve.traj --plot

# Complete mission with stops and events
python main.py -c examples/fll_choreo.json -w examples/example_complete.json -o complete.traj --plot

# Complex mission
python main.py -c examples/fll_choreo.json -w examples/example_complex_mission.json -o complex.traj --plot

# First four waypoints of the complex mission (candidate-seed benchmark)
python main.py --simple --validate -c examples/fll_choreo.json -w examples/example_mission_first4.json -o first4.traj --candidates 5
```

---

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
python main.py -c examples/fll_choreo.json -w waypoints_with_stops.json -o stops.traj --plot
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o stops.traj \
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
python main.py -c examples/fll_choreo.json -w waypoints_with_events.json -o events.traj \
  --export-format controller \
  --controller-dt 0.02
```

**Expected output:**

- Trajectory file contains `"event"` field at waypoint samples
- Controller export preserves events at nearest timestep
- Robot controller reads events and triggers actions (arm, intake, release)

> **Runtime caveat:** an event on the final waypoint is exported but not fired by the deployed `ramsete_fast.py` loop (KI-012). Trigger the final action after the trajectory finishes.

---

### Example 8: Event Markers (CLI)

Alternatively, specify events via CLI flag:

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o events.traj \
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o fast.traj -a 0.0 --validate
```

**Balanced smoothness:**

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o smooth.traj -a 1.0 --validate
```

**Very smooth:**

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o very_smooth.traj -a 5.0 --validate
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o traj.traj \
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o traj.traj \
  --export-format python
```

**Output file:** `traj.py` with a `config` dictionary and a `samples` list containing trajectory data (including events when present).

**Use case:** Direct integration with Python-based robot controllers or analysis scripts.

---

### Example 11b: Compact Python Export

Export the packed module used by the compact Pybricks controllers:

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o traj.traj \
  --export-format python-compact
```

**Output file:** `traj.py` with a `config` dictionary, a packed `samples` bytes object, and an `events` tuple. The blob uses a 4-byte header plus 24 bytes per sample.

**Use case:** Import the module with `pybricks_code/ramsete_compact.py` for pathing or `pybricks_code/ramsete_compact_events.py` for events. The compact controllers are not wired into the existing `robot.py` menu or missions, and the final-event caveat still applies (KI-012).

---

### Example 12: Varying Sample Density

Compare different sample densities:

**Low density (fast, less accurate):**

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o low_res.traj -n 5
```

**Medium density (default):**

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o med_res.traj -n 10
```

**High density (slower, more accurate):**

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o high_res.traj -n 20
```

**Tradeoffs:**

- `-n 5`: Faster optimization, but may miss sharp turns
- `-n 10`: Good balance for most FLL paths
- `-n 20`: Slower optimization, smoother curves, better for complex paths

With `--simple`, `--per-segment-dt` lets each waypoint segment use its own sample interval instead of forcing one interval across the whole path:

```bash
python main.py --simple --per-segment-dt \
  -c examples/fll_choreo.json -w waypoints.json -o per_segment.traj
```

This changes the optimization variables, not the waypoint constraints. It is not available on the Multi-Verse path.

---

### Example 13: Multi-Verse Refinement (experimental)

The default optimizer is the Multi-Verse pipeline. It requires JAX and runs:

```text
bootstrap -> kinematic profile -> critic -> JAX refinement -> CasADi polish
```

Example:

```bash
python main.py -c examples/fll_choreo.json -w complex_waypoints.json -o complex.traj \
  -n 15 \
  -a 1.0 \
  --plot
```

**Current behaviour:** the refinement phase updates an intermediate trajectory, but the final CasADi polish does not receive that refined guess (`multiverse_optimizer.py:_global_solve`, KI-004). The final trajectory is therefore the same as the simple solve, and `--workers`/`--no-parallel` do not change the result. A measured 3-waypoint pipeline benchmark produced 0.00 % cost improvement.

**Use the simple optimizer for predictable results:**

```bash
python main.py --simple -c examples/fll_choreo.json -w waypoints.json -o output.traj
```

**Flags kept for compatibility:**

```bash
python main.py -c examples/fll_choreo.json -w waypoints.json -o output.traj --no-parallel
python main.py -c examples/fll_choreo.json -w waypoints.json -o output.traj --workers 16
```

Do not expect these to improve quality or speed up the final solve until KI-004 is fixed.

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
python main.py --simple -c examples/fll_choreo.json -w mission_waypoints.json -o mission.traj \
  -n 10 \
  -a 0.5 \
  --validate \
  --export-format controller \
  --controller-dt 0.02 \
  --plot
```

**Expected output:**

- With `--simple`, solves are typically a few tenths of a second; the default Multi-Verse path adds JAX overhead without changing the result (KI-004)
- The solve may converge while `--validate` still rejects the path; under the current test ranges the S-curve, stops, and events examples fail, while the complex mission passes (KI-020)
- If validation passes, the controller profile has one sample per 20 ms of trajectory

Run with `--simple` and without `--validate` for exploratory output, or use a path that passes the gate before exporting to the robot.

---

## Python API Examples

### Example 15: Using the Optimizer Directly

```python
from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
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

### Example 16: Using Multi-Verse Optimizer

```python
from robot_model import RobotConfig
from multiverse_optimizer import MasterTrajectoryOptimizer
import json

# Load configuration
with open('examples/fll_choreo.json', 'r') as f:
    config_data = json.load(f)

robot_cfg = RobotConfig(config_data)
optimizer = MasterTrajectoryOptimizer(
    robot_cfg,
    enable_parallel=True,
    num_workers=8
)

# Define waypoints
waypoints = [
    (0.0, 0.0, 0.0),
    (0.5, 0.3, 0.5),
    (1.0, 0.0, 0.0),
    (1.5, 0.3, -0.5),
    (2.0, 0.0, 0.0)
]

# Solve with stops and events. solve() returns (samples, stats).
stop_waypoint_indices = [0, 2, 4]
waypoint_events = {1: "lower_arm", 3: "release"}

samples, stats = optimizer.solve(
    waypoints,
    num_samples_per_segment=15,
    accuracy_weight=1.0,
    stop_waypoint_indices=stop_waypoint_indices,
    waypoint_events=waypoint_events
)

print(f"Total time: {samples[-1]['t']:.3f}s")
print(f"Number of samples: {len(samples)}")
print(f"Refinement phase: {stats['phase_times']['refinement']:.4f}s")
```

> **Current status:** the Multi-Verse `solve()` returns `(samples, stats)`, but the final polish ignores the refined guess (KI-004). For predictable output, use `TrajectoryOptimizer` and `--simple`.

---

### Example 17: Validating a Trajectory

```python
from validator import validate_trajectory

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

### Example 18: Exporting for Controller

```python
from export import write_controller_file
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

### Example 19: Comparing Accuracy Weights

```python
from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
from validator import compute_metrics
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

### Example 19b: Comparing Initial-Guess Seeds

With `--simple`, `--candidates N` solves the same waypoints from N deterministic initial guesses and validates each one:

```bash
python main.py --simple --validate -c examples/fll_choreo.json \
  -w examples/example_mission_first4.json -o first4.traj --candidates 5
```

The seeds are `baseline`, `rolling`, `bow_left`, `bow_right`, and `turn_first`. The candidate table prints each seed's duration and envelope, and selection keeps the passing one with the smallest envelope (shortest breaks ties). On the first four waypoints of the complex mission all five pass; `baseline` wins at 2.176 s / 0.27 cm. On the full mission the baseline also wins (6.068 s / 1.22 cm); the other seeds converge to slower, wider paths, so they are rejected.

From Python, pass `guess_style` to `solve`:

```python
for style in ("baseline", "rolling", "bow_left", "bow_right", "turn_first"):
    samples, stats = optimizer.solve(waypoints, 10, 0.0, guess_style=style)
```

The seeds only change the initial guess; every waypoint, stop, and force limit is still enforced by the solver.

---

## Troubleshooting Examples

### Example 20: Debugging Optimization Failure

If optimization fails or times out:

```bash
# First, try with fewer samples
python main.py -c examples/fll_choreo.json -w waypoints.json -o test.traj -n 5

# If that works, gradually increase
python main.py -c examples/fll_choreo.json -w waypoints.json -o test.traj -n 10
python main.py -c examples/fll_choreo.json -w waypoints.json -o test.traj -n 15
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o output.traj --validate
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o smooth.traj -a 2.0
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
python main.py -c examples/fll_choreo.json -w waypoints.json -o high_res.traj -n 20
```

---

## Integration Examples

### Example 23: Batch Processing Multiple Paths

```python
import os
import json
from robot_model import RobotConfig
from multiverse_optimizer import MasterTrajectoryOptimizer

# Load config once
with open('examples/fll_choreo.json', 'r') as f:
    config_data = json.load(f)
robot_cfg = RobotConfig(config_data)
optimizer = MasterTrajectoryOptimizer(robot_cfg, enable_parallel=True, num_workers=8)

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

> **Note:** `MasterTrajectoryOptimizer.solve()` returns `(samples, stats)`, and the current refinement does not change the final trajectory (KI-004). For batch work, `TrajectoryOptimizer` with `--simple` is faster and predictable.

---

### Example 24: Generating Comparison Plots

```python
from plotter import plot_trajectory
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
python main.py --simple -c examples/fll_choreo.json -w complete_mission.json -o examples/complete_mission.traj \
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

> **Current status:** under the narrowed test ranges in `examples/fll_choreo.json`, the complex mission passes (about 6.068 s, 1.22 cm envelope, 0.49 cm tracking), while the S-curve, stops, and events examples still fail the Immrax gate. Always check the exit code: failed `--validate` runs exit 1 and write nothing. Run without `--validate` only for exploratory output, or use a path that passes the gate before exporting. This is KI-020, not a solver crash.

> **Final event:** waypoint 7 is not the final waypoint here, but if an event were placed on the final waypoint it would not fire on the deployed controller (KI-012).

**Output files:**

- `examples/complete_mission.traj`: Full variable-timestep trajectory with events
- `examples/complete_mission_controller.json`: Fixed 20ms timestep for robot controller with events

**Use case:** Complete FLL mission with multiple segments, stops for precision, events for mechanism control, and validation for safety.

---

## Live Visualization

### Example 26: Live Visualization in Browser

Monitor the optimization process in real-time using a web browser:

**Command:**

```bash
python main.py --simple -c examples/fll_choreo.json -w waypoints.json --live
```

**Steps:**

1. Run the command with the `--live` flag.
2. The terminal prints: `Live visualizer started. Open viz/index.html in your browser.`
3. Open `viz/index.html` in your browser.
4. The final trajectory is broadcast when the solve completes.

**Current limitations:**

- The server kills whatever owns port 8765 before starting (KI-005). Do not use `--live` while another application needs that port.
- The first browser client does not receive the robot config or waypoints because the initial broadcast happens before it connects (KI-016). The final trajectory still arrives.
- The regenerate controls are dead code; `on_regenerate` is never assigned.
- `--live` is blocking: the command waits for Enter before exiting.

**Use case:** Inspecting the final trajectory and any attached reachability envelopes. It is not currently a reliable live debugging tool for configuration or topology exploration.
