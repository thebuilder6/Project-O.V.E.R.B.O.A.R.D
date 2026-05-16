# **O**ptimal **V**elocity **E**ngine for **R**obotic **B**oundary-constrained **O**rchestration and **A**dvanced **R**amsete **D**ynamics

A high-performance trajectory optimizer for LEGO differential-drive robots in FIRST LEGO League (FLL). Generates time-optimal, physically feasible trajectories using direct collocation with CasADi and IPOPT.

Also contains a Pybricks template with a performance optimized Ramsete controller.
## Features

- **Time-optimal trajectories**: Minimizes travel time while respecting actuator limits
- **Physics-aware modeling**: Accounts for motor torque curves, wheel dynamics, and traction limits
- **Accuracy tuning**: Optional smoothness penalty to improve real-world tracking accuracy
- **Fast optimization**: Sub-second solve times using CasADi + IPOPT
- **Multi-Verse refinement**: Advanced optimization with parallel processing for complex paths
- **Trajectory validation**: Forward-integration verification with constraint checking
- **Controller export**: Resamples trajectories to fixed timesteps for on-robot execution
- **Event markers**: Trigger robot actions at specific waypoints
- **Stop constraints**: Force robot to come to rest at waypoints



## Installation

### Prerequisites

- Python 3.8+
- pip

### Dependencies

```bash
pip install numpy casadi click matplotlib rsplan
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create a robot configuration

Create a `.json` file with your robot's physical parameters:

```json
{
  "name": "my_robot",
  "version": 2,
  "type": "Differential",
  "config": {
    "mass": {"val": 0.8},
    "inertia": {"val": 0.000001},
    "differentialTrackWidth": {"val": 0.0965},
    "radius": {"val": 0.028},
    "vmax": {"val": 15.7},
    "tmax": {"val": 0.04},
    "gearing": {"val": 1.0},
    "cof": {"val": 1.5},
    "torqueHeadroom": {"val": 0.85},
    "speedHeadroom": {"val": 0.90}
  }
}
```

**Key parameters:**
- `mass`: Robot mass in kg (0.5-1.5 kg typical)
- `inertia`: Rotational inertia in kg·m² (1e-6 to 1e-4 typical)
- `differentialTrackWidth`: Distance between wheel centers in meters
- `radius`: Wheel radius in meters
- `vmax`: Motor no-load speed in rad/s (RPM × 2π/60)
- `tmax`: Motor stall torque in N·m
- `gearing`: Gear ratio (1.0 for direct drive)
- `cof`: Coefficient of friction (1.0-1.5 for FLL mats)
- `torqueHeadroom`: Safety margin for motor torque (0.85 recommended)
- `speedHeadroom`: Safety margin for wheel speed (0.90 recommended)

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for detailed parameter explanations and calibration procedures.

### 2. Define waypoints

Create a JSON file with waypoints:

```json
[
  {"x": 0.0, "y": 0.0, "heading": 0.0},
  {"x": 1.0, "y": 0.5, "heading": 0.5},
  {"x": 2.0, "y": 1.0, "heading": 1.0}
]
```

**Waypoint options:**
- `x`, `y`: Position in meters (required)
- `heading`: Heading in radians (optional, use `null` for unconstrained)
- `stop`: Force robot to stop at this waypoint (optional, default: `false`)
- `event`: Trigger an action at this waypoint (optional, e.g., `"lower_arm"`)

### 3. Generate a trajectory

```bash
python main.py -c fll_choreo.chor -w test_waypoints.json -o output.traj --plot
```

### 4. Validate and export for controller

```bash
python main.py -c fll_choreo.chor -w test_waypoints.json -o output.traj \
  --validate --export-format controller --controller-dt 0.02
```

This generates two files:
- `output.traj`: Full trajectory with variable timesteps (for analysis) or generate .py python file for the robot
- `output_controller.json`: Fixed 20ms timesteps (for robot controller)

### 5. Live Visualization (Optional) (Broken) 

To see the optimization progress in real-time in your browser:

1. Run the optimizer with the `--live` flag:
   ```bash
   python main.py -c fll_choreo.chor -w test_waypoints.json --live
   ```
2. Open `viz/index.html` in any modern web browser.
3. The visualizer will connect to the WebSocket server and show trajectory updates as they happen.

## CLI Options

```
Usage: main.py [OPTIONS]

Options:
  -c, --config PATH              Path to the configuration file (.chor or .json). [required]
  -w, --waypoints PATH           Path to waypoints JSON file. [required]
  -o, --output TEXT              Output trajectory file path. [default: output.traj]
  -n, --samples INTEGER          Samples per segment. [default: 10]
  -a, --accuracy-weight FLOAT    Smoothness/accuracy weight (0 = pure time-optimal). [default: 0.0]
  --stop-waypoints TEXT          Comma-separated waypoint indices where robot must stop (e.g., "2,5,7").
  --events TEXT                  Comma-separated waypoint:event pairs (e.g., "2:lower_arm,5:release").
  --validate                     Run validation report on the generated trajectory.
  --export-format [none|controller|python] Export format for controller consumption. [default: none]
  --controller-dt FLOAT          Fixed timestep for controller export (seconds). [default: 0.02]
  --plot                        Plot the resulting trajectory.
  --animate                     Animate the trajectory in real-time.
  --live                        Enable live interactive visualization in browser.
  --show-convergence            Show convergence visualization (Plotly).
  --convergence-mode [parallel|best|layered] Convergence visualization mode. [default: best]
  --convergence-animate         Animate convergence.
  --simple                       Use simple optimizer instead of Multi-Verse refinement.
  --no-parallel                  Disable parallel processing for Multi-Verse refinement.
  --workers INTEGER              Number of parallel workers for Multi-Verse refinement. [default: 8]
```

## Project Structure

```
FLL_Choreo/
├── main.py                    # CLI entry point
├── optimizer.py               # Core trajectory optimization (CasADi + IPOPT)
├── multiverse_optimizer.py    # Advanced Multi-Verse refinement optimizer
├── robot_model.py             # Robot configuration and differential drive dynamics
├── validator.py               # Trajectory validation via forward integration
├── export.py                  # Controller-ready export with fixed timesteps
├── plotter.py                 # Trajectory visualization
├── path_planning.py           # Path planning utilities
├── fll_choreo.chor            # Example robot configuration file
├── example_*.json             # Example waypoint files
├── README.md                  # This file
├── docs/
│   ├── CONFIGURATION.md       # Detailed configuration guide
│   ├── EXAMPLES.md            # Usage examples
│   ├── API_optimizer.md       # Optimizer API documentation
│   ├── API_robot_model.md     # Robot model API documentation
│   └── API_validator_export.md # Validator and export API documentation
└── pybricks_code/             # Pybricks robot controller examples
```

## How It Works

The optimizer uses **direct collocation** to solve an optimal control problem:

1. **Discretization**: The trajectory is divided into N time steps (samples per segment × segments)
2. **Decision variables**: At each step, the optimizer chooses position (x, y, θ) and wheel velocities (vl, vr)
3. **Constraints**:
   - Kinematic constraints (trapezoidal collocation)
   - Motor torque limits (velocity-dependent)
   - Traction limits (wheel slip prevention)
   - Waypoint constraints (position and optional heading)
   - Stop constraints (velocity = 0 at specified waypoints)
   - Start/end at rest
4. **Objective**: Minimize total time + optional smoothness penalty (jerk)

The solver (IPOPT) finds the optimal trajectory that satisfies all constraints in sub-second time.

### Multi-Verse Refinement

For complex paths, the optimizer can use **Multi-Verse refinement**:
- Generates multiple candidate trajectories with different initial conditions
- Refines the best candidates in parallel
- Produces higher-quality trajectories for challenging paths
- Use `--simple` flag to disable and use the basic optimizer

## Common Usage Patterns

### Simple straight line

```bash
python main.py -c fll_choreo.chor -w example_straight.json -o straight.traj --plot
```

### Path with turns

```bash
python main.py -c fll_choreo.chor -w example_s_curve.json -o s_curve.traj --plot
```

### Mission with stops and events

```bash
python main.py -c fll_choreo.chor -w example_complete.json -o mission.traj \
  --validate --export-format controller --controller-dt 0.02 --plot
```

### Smooth trajectory for better tracking

```bash
python main.py -c fll_choreo.chor -w waypoints.json -o smooth.traj \
  -a 1.0 --validate --plot
```

### Power User Optimization (All-in-One)

For complex missions, use the full suite of Multi-Verse features:

```bash
python main.py -c robot_config.json -w waypoints.json -o mission.json \
  -a 1.0 --workers 8 --live --benchmark --validate --export-format controller \
  --show-convergence --convergence-animate
```

- **-a 1.0**: Smoothness priority
- **--workers 8**: Parallel refinement
- **--live**: Interactive visualization
- **--benchmark**: Performance telemetry
- **--validate**: Physical safety check
- **--show-convergence**: Progress analysis

## Accuracy vs. Speed Tradeoff

By default, the optimizer is purely time-optimal (`--accuracy-weight 0.0`). This produces the fastest possible trajectory but may have high jerk (rapid acceleration changes) that can cause tracking errors on real robots.

Use `--accuracy-weight` to add a smoothness penalty:

- `0.0`: Pure time-optimal (fastest, most aggressive)
- `0.5-1.0`: Balanced smoothness (~3-5% time cost, ~35% less jerk)
- `2.0+`: Very smooth (significant time cost, minimal overshoot)

## Output Format

The optimizer outputs a JSON file in Choreo format:

```json
{
  "name": "output",
  "version": 3,
  "trajectory": {
    "config": { ... },
    "samples": [
      {
        "t": 0.0,
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0,
        "vl": 0.0,
        "vr": 0.0,
        "omega": 0.0,
        "al": 0.0,
        "ar": 0.0,
        "fl": 0.0,
        "fr": 0.0
      },
      ...
    ]
  }
}
```

## Validation

Run `--validate` to verify the trajectory:

- Forward-integrates the trajectory using RK4 (1 ms steps)
- Checks constraint violations (motor limits, traction)
- Reports max position error and final error
- Provides pass/fail verdict

## Controller Export

Use `--export-format controller` to generate a fixed-timestep file for on-robot execution:

- Resamples variable-timestep trajectory to fixed `dt` (default 20 ms)
- Outputs: `(t, vl, vr, v, omega, x, y, heading)`
- Ready for direct consumption by your robot's trajectory follower

## Performance

Typical solve times on a modern laptop:

- 2 waypoints (1 segment, 10 samples): ~67 ms
- 5 waypoints (4 segments, 40 samples): ~383 ms
- Complex paths: < 500 ms

## Research & References

See `trajectory_tools_research.md` for:
- Comparison with other tools (Choreo, PathPlanner, CasADi)
- Academic papers on trajectory optimization
- Research directions for accuracy improvements

## Development Status

✅ Phase 1: CasADi + IPOPT solver (complete)
✅ Phase 2: Time vs. accuracy tradeoff (complete)
✅ Phase 3: Validation & controller export (complete)
✅ Phase 4: Multi-Verse refinement optimizer (complete)
✅ Phase 5: Event markers and stop constraints (complete)
⚪ Phase 6: Field geometry/obstacle constraints (optional)

See `project_plan.md` for details.

## Additional Documentation

- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**: Detailed robot configuration guide with calibration procedures
- **[docs/EXAMPLES.md](docs/EXAMPLES.md)**: Comprehensive usage examples and troubleshooting
- **[docs/API_optimizer.md](docs/API_optimizer.md)**: Optimizer API reference
- **[docs/API_robot_model.md](docs/API_robot_model.md)**: Robot model API reference
- **[docs/API_validator_export.md](docs/API_validator_export.md)**: Validator and export API reference

## Pybricks Integration

The `pybricks_code/` directory contains example robot controller code for Pybricks:
- `robot.py`: Base robot class with Ramsete controller
- `ramsete.py`: Ramsete trajectory following algorithm
- `mission_*.py`: Example mission scripts
- `plot_tracking.py`: Trajectory tracking visualization

See `pybricks_code/README.md` for Pybricks setup instructions.

## License

This project is for FLL educational use.
