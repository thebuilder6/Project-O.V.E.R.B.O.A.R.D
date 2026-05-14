# FLL Trajectory Optimizer

A high-performance trajectory optimizer for LEGO differential-drive robots in FIRST LEGO League (FLL). Generates time-optimal, physically feasible trajectories using direct collocation with CasADi and IPOPT.

## Features

- **Time-optimal trajectories**: Minimizes travel time while respecting actuator limits
- **Physics-aware modeling**: Accounts for motor torque curves, wheel dynamics, and traction limits
- **Accuracy tuning**: Optional smoothness penalty to improve real-world tracking accuracy
- **Fast optimization**: Sub-second solve times using CasADi + IPOPT
- **Trajectory validation**: Forward-integration verification with constraint checking
- **Controller export**: Resamples trajectories to fixed timesteps for on-robot execution
- **Choreo-compatible**: Uses Choreo-like configuration format for easy migration

## Installation

### Prerequisites

- Python 3.8+
- pip

### Dependencies

```bash
pip install numpy casadi click matplotlib
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create a robot configuration

Create a `.chor` file with your robot's physical parameters:

```json
{
  "config": {
    "mass": {"val": 0.8},
    "inertia": {"val": 0.001},
    "differentialTrackWidth": {"val": 0.0965},
    "radius": {"val": 0.028},
    "vmax": {"val": 15.7},
    "tmax": {"val": 0.04},
    "gearing": {"val": 1.0},
    "cof": {"val": 1.5}
  }
}
```

### 2. Define waypoints

Create a JSON file with waypoints:

```json
[
  {"x": 0.0, "y": 0.0, "heading": 0.0},
  {"x": 1.0, "y": 0.5, "heading": 0.5},
  {"x": 2.0, "y": 1.0, "heading": 1.0}
]
```

### 3. Generate a trajectory

```bash
python main.py -c fll_choreo.chor -w test_waypoints.json -o output.traj --plot
```

### 4. (Optional) Validate and export for controller

```bash
python main.py -c fll_choreo.chor -w test_waypoints.json -o output.traj \
  --validate --export-format controller --controller-dt 0.02
```

## CLI Options

```
Usage: main.py [OPTIONS]

Options:
  -c, --config PATH           Path to the .chor configuration file. [required]
  -w, --waypoints PATH        Path to waypoints JSON file. [required]
  -o, --output TEXT           Output trajectory file path. [default: output.traj]
  -s, --samples INTEGER       Samples per segment. [default: 10]
  -a, --accuracy-weight FLOAT Smoothness/accuracy weight (0 = pure time-optimal). [default: 0.0]
  --validate                  Run validation report on the generated trajectory.
  --export-format [none|controller] Export format for controller consumption. [default: none]
  --controller-dt FLOAT       Fixed timestep for controller export (seconds). [default: 0.02]
  --plot                      Plot the resulting trajectory.
```

## Project Structure

```
FLL_Paths/
├── main.py              # CLI entry point
├── optimizer.py         # Core trajectory optimization (CasADi + IPOPT)
├── robot_model.py       # Robot configuration and differential drive dynamics
├── validator.py         # Trajectory validation via forward integration
├── export.py            # Controller-ready export with fixed timesteps
├── plotter.py           # Trajectory visualization
├── README.md            # This file
├── project_plan.md      # Development roadmap
└── trajectory_tools_research.md  # Research notes and references
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
   - Start/end at rest
4. **Objective**: Minimize total time + optional smoothness penalty (jerk)

The solver (IPOPT) finds the optimal trajectory that satisfies all constraints in sub-second time.

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
⚪ Phase 4: Field geometry/obstacle constraints (optional)

See `project_plan.md` for details.

## TODO
- rework Robot definition ".chor" file no need to keep choreos format
- move input output into on directory




## License

This project is for FLL educational use.
