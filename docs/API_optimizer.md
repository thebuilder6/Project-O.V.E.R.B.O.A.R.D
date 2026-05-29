# API Documentation: optimizer.py and multiverse_optimizer.py

This documentation covers the trajectory optimization capabilities, including the legacy direct collocation optimizer and the modern Multi-Verse refinement architecture.

---

## `multiverse_optimizer.py`

The new Multi-Verse optimizer pipeline uses a hybrid approach: Bootstrap -> Global Solve -> Critic -> Refinement -> Polish.

### Class `MasterTrajectoryOptimizer`

Orchestrates the Multi-Verse refinement pipeline.

#### Constructor

```python
MasterTrajectoryOptimizer(config: RobotConfig, enable_parallel: bool = True, num_workers: int = 8, verbose: bool = True)
```

**Parameters:**
- `config` (RobotConfig): Robot configuration object.
- `enable_parallel` (bool): Whether to use parallel processing for refinements.
- `num_workers` (int): Number of worker threads for parallel evaluation.
- `verbose` (bool): If True, prints progress messages.

#### `solve(...)`

Solves the trajectory using the Multi-Verse pipeline.

**Parameters:**
- `waypoints` (list[tuple]): List of `(x, y, heading)` tuples.
- `num_samples_per_segment` (int): Number of samples per segment.
- `accuracy_weight` (float): Smoothness weight.
- `stop_waypoint_indices` (list[int]): Indices where robot must stop.
- `waypoint_events` (dict[int, str]): Event markers.
- `apply_headroom` (bool): Apply safety margins.
- `verbose` (bool): Print progress messages.
- `capture_iterations` (bool): Capture intermediate states for convergence visualization.
- `live_viz` (bool): Stream updates to a visualizer.

**Returns:**
- `tuple[list[dict], dict]`: A tuple containing (trajectory_samples, optimization_stats).

---

### Components

#### Class `PathBootstrapper`
Generates kinematically valid initial guesses using Reeds-Shepp paths or linear interpolation.
- `generate_baseline(waypoints, num_samples_per_segment)`

#### Class `TrajectoryCritic`
Evaluates trajectory quality using research-grounded metrics (tortuosity, jerk cost, curvature cost, centripetal cost) to identify problematic segments that need refinement.
- `evaluate(trajectory, num_samples_per_segment) -> List[Tuple[int, int]]`

#### Class `LocalSegmentOptimizer`
Miniature optimizer for solving local trajectory segments with pinned boundary conditions. Uses fast settings in IPOPT.

#### Class `MultiVerseRefiner`
Handles TEB (Timed Elastic Band) and STOMP (Stochastic Trajectory Optimization for Motion Planning) parallel refinement of problematic segments identified by the Critic.
- `refine_segment(...)`

---

## `optimizer.py` (Legacy)

Core trajectory optimization using direct collocation with CasADi and IPOPT solver.

### Class `TrajectoryOptimizer`

Optimizes time-optimal trajectories for differential-drive robots. Now primarily serves as the "fast global solve" and "polish" engine within the Multi-Verse architecture, or standalone as the "simple" optimizer.

#### Constructor

```python
TrajectoryOptimizer(config: RobotConfig)
```

#### `solve(...)`

Solves the trajectory optimization problem using direct collocation.

**Parameters:**
- `waypoints` (list[tuple]): List of waypoints as `(x, y, heading)` tuples
- `num_samples_per_segment` (int): Number of collocation points per segment (default: 10)
- `accuracy_weight` (float): Weight for smoothness/jerk penalty (default: 0.0)
- `stop_waypoint_indices` (list[int]|None): List of waypoint indices where robot must come to rest.
- `waypoint_events` (dict[int, str]|None): Dictionary mapping waypoint indices to event names.
- `apply_headroom` (bool): If True, applies safety margin for real-world tracking.
- `verbose` (bool): If True, prints progress messages.
- `capture_iterations` (bool): If True, captures intermediate solver states.

**Returns:**
- `tuple[list[dict], dict]`: A tuple containing (trajectory_samples, solver_stats).

**Optimization Problem Formulation:**

**Decision Variables:**
- `dt`: Time step between collocation points
- `X`: State matrix of shape (N, 5) containing `[x, y, theta, vl, vr]` at each point

**Constraints:**
1. **Time step bounds:** `0.001 <= dt <= 1.0`
2. **Wheel speed bounds:** `|vl|, |vr| <= 0.99 * max_linear_speed`
3. **Kinematic constraints** (trapezoidal collocation)
4. **Motor force limits:** `|fl| <= max_force_at_velocity(vl)`
5. **Traction limit:** `|fl| + |fr| <= cof * mass * g`

**Usage Example:**

```python
from robot_model import RobotConfig
from multiverse_optimizer import MasterTrajectoryOptimizer
import json

# Load configuration
with open('robot_config.json', 'r') as f:
    config_data = json.load(f)

robot_cfg = RobotConfig(config_data)
optimizer = MasterTrajectoryOptimizer(robot_cfg)

# Define waypoints
waypoints = [
    (0.0, 0.0, 0.0),      # Start at origin, facing East
    (1.0, 0.5, 0.5),      # Waypoint 1
    (2.0, 1.0, 1.0),      # Waypoint 2
    (3.0, 1.5, None)      # End at (3, 1.5), heading unconstrained
]

# Solve using the Multi-Verse pipeline
samples, stats = optimizer.solve(waypoints, num_samples_per_segment=10)
print(f"Total solve time: {stats['total_time']:.3f}s")

# Solve with Live Visualization enabled
# This will start the WebSocket server automatically
samples, stats = optimizer.solve(waypoints, live_viz=True)
```

---

## `live_visualizer.py`

Provides a robust, bi-directional WebSocket server for real-time trajectory visualization and interaction in a web browser.

### Function `get_visualizer()`

Returns the global `LiveVisualizer` instance, starting the server thread if it hasn't been started yet. Reclaims the port (8765) if it is currently in use by another process.

**Returns:**
- `LiveVisualizer`: The global visualizer instance.

### Class `LiveVisualizer`

#### `start()`
Starts the WebSocket server on `localhost:8765` in a separate daemon thread.

#### `send_trajectory(trajectory, phase="solve", iteration=0)`
Broadcasts the current trajectory state including rich telemetry (position, heading, velocity, force).

#### `send_candidates(window, candidates)`
Broadcasts alternative candidate trajectories from the Multi-Verse refinement phases.

#### `send_config(robot_config, waypoints)`
Broadcasts the initial robot configuration and waypoint set to the visualizer.

#### `on_regenerate` (Callback)
A configurable callback property that is triggered when the visualizer sends a `regenerate` request for a specific window. This runs in a separate thread to maintain UI responsiveness.

**Usage with Browser:**
1. Call `get_visualizer()` or set `live_viz=True` in `solve()`.
2. Open `viz/index.html` in a modern web browser.
3. The browser will automatically connect to `ws://localhost:8765`.
4. The new interactive dashboard allows for panning/zooming, segment selection, playback animation, and on-demand refinement.
