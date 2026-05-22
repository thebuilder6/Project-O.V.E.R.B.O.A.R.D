"""
Multi-Verse Segment Refinement Pipeline

A hybrid trajectory generation architecture combining:
- Gradient-based optimization (CasADi)
- Topology exploration (TEB)
- Stochastic perturbation (STOMP)
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import casadi as ca
from concurrent.futures import ThreadPoolExecutor, as_completed
from robot_model import RobotConfig, DifferentialDriveModel
from path_planning import ReedsSheppPath, linear_interpolation_waypoints
from live_visualizer import get_visualizer
import sys
import time


class OptimizationStats:
    """Container for detailed optimization telemetry."""
    def __init__(self):
        self.total_time = 0.0
        self.phase_times = {
            "bootstrap": 0.0,
            "global_solve": 0.0,
            "critic": 0.0,
            "refinement": 0.0,
            "polish": 0.0
        }
        self.bad_segments_found = 0
        self.refinements_attempted = 0
        self.refinements_solved = 0
        self.heuristic_wins = []  # List of dicts with window, heuristic, improvement
        self.initial_cost = 0.0
        self.final_cost = 0.0

    def to_dict(self):
        return {
            "total_time": self.total_time,
            "phase_times": self.phase_times,
            "bad_segments_found": self.bad_segments_found,
            "refinements_attempted": self.refinements_attempted,
            "refinements_solved": self.refinements_solved,
            "heuristic_wins": self.heuristic_wins,
            "initial_cost": self.initial_cost,
            "final_cost": self.final_cost,
            "improvement_pct": (1 - self.final_cost / self.initial_cost) * 100 if self.initial_cost > 0 else 0
        }


# File constants for thresholds and parameters
TORTUOSITY_THRESHOLD = 1.1  # More sensitive to unnecessary path length
YAW_BUFFER_RAD = 0.2  # More sensitive to excess rotation
VELOCITY_CHATTERING_THRESHOLD = 2
JERK_COST_THRESHOLD = 2.0  # Normalized jerk cost threshold
CURVATURE_COST_THRESHOLD = 1.5  # Turn sharpness threshold
CENTRIPETAL_COST_THRESHOLD = 0.6  # Safety margin for friction limits

STOMP_NOISE_POS_STD = 0.05
STOMP_NOISE_HEADING_STD = 0.1
DEFAULT_STOMP_VARIANTS = 5

TEB_FORWARD_BIAS = 10.0
TEB_REVERSE_BIAS = 10.0
TEB_POINT_TURN_BIAS = 5.0
TEB_WIDE_SWEEP_BIAS = 2.0


class PathBootstrapper:
    """Generates kinematically valid initial guesses using Reeds-Shepp paths."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        # Get multiverse config or use defaults
        mv_cfg = config.multiverse_config
        bootstrap_cfg = mv_cfg.get("bootstrap", {})
        self.turning_radius = bootstrap_cfg.get("turning_radius", 0.0)
        self.resolution = bootstrap_cfg.get("resolution", 0.05)
        self.rs_path = ReedsSheppPath(self.turning_radius)

    def generate_baseline(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10) -> np.ndarray:
        """
        Generate kinematically valid baseline trajectory.

        Args:
            waypoints: List of (x, y, heading) tuples
            num_samples_per_segment: Number of samples per segment

        Returns:
            Initial guess array [dt, x, y, theta, vl, vr, ...]
        """
        num_segments = len(waypoints) - 1
        N = num_segments * num_samples_per_segment + 1
        initial_dt = 0.1
        guess = [initial_dt]
        
        for i in range(num_segments):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            count = num_samples_per_segment if i < num_segments - 1 else num_samples_per_segment + 1
            
            # Try Reeds-Shepp first
            segment_waypoints = self._generate_segment_waypoints(p1, p2, count)
            
            for j in range(count):
                if j < len(segment_waypoints):
                    x, y, theta = segment_waypoints[j]
                else:
                    # Fallback to linear interpolation if Reeds-Shepp fails
                    frac = j / num_samples_per_segment
                    x = p1[0] + (p2[0] - p1[0]) * frac
                    y = p1[1] + (p2[1] - p1[1]) * frac
                    theta = self._interpolate_heading(p1, p2, frac)
                
                guess.extend([x, y, theta, 0.0, 0.0])
        
        return np.array(guess)
    
    def _generate_segment_waypoints(self, start: Tuple[float, float, Optional[float]], goal: Tuple[float, float, Optional[float]], num_points: int) -> List[Tuple[float, float, float]]:
        """Generate waypoints for a segment using Reeds-Shepp or fallback."""
        # Try Reeds-Shepp if both headings are known
        if start[2] is not None and goal[2] is not None:
            rs_path = self.rs_path.plan(start, goal, step_size=self.resolution)
            if rs_path is not None and len(rs_path) > 2:
                # Resample rs_path to exact num_points
                return self._resample_path(rs_path, num_points)
        
        # Fallback to linear interpolation
        return linear_interpolation_waypoints(start, goal, num_points)

    def _resample_path(self, path: List[Tuple[float, float, float]], num_points: int) -> List[Tuple[float, float, float]]:
        """Resample a path (list of x, y, theta) to num_points."""
        if len(path) < 2:
            return path
        
        # Calculate cumulative distances
        dists = [0.0]
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            # Include theta in distance to handle point turns better? 
            # For now just x, y distance
            dists.append(dists[-1] + np.sqrt(dx**2 + dy**2))
        
        total_dist = dists[-1]
        resampled = []
        
        # If total_dist is very small (pure point turn), resample by index
        if total_dist < 1e-6:
            for i in range(num_points):
                idx = int(i / (num_points - 1) * (len(path) - 1))
                resampled.append(path[idx])
            return resampled

        for i in range(num_points):
            target_d = (i / (num_points - 1)) * total_dist
            # Find segment
            for j in range(len(dists) - 1):
                if dists[j] <= target_d <= dists[j+1]:
                    frac = (target_d - dists[j]) / (dists[j+1] - dists[j]) if dists[j+1] > dists[j] else 0.0
                    x = path[j][0] + frac * (path[j+1][0] - path[j][0])
                    y = path[j][1] + frac * (path[j+1][1] - path[j][1])
                    # Heading interpolation with angle wrapping
                    dtheta = (path[j+1][2] - path[j][2] + np.pi) % (2 * np.pi) - np.pi
                    theta = path[j][2] + frac * dtheta
                    resampled.append((x, y, theta))
                    break
        
        return resampled
    
    def _interpolate_heading(self, p1: Tuple[float, float, Optional[float]], p2: Tuple[float, float, Optional[float]], frac: float) -> Optional[float]:
        """Interpolate heading between two poses."""
        if p1[2] is not None and p2[2] is not None:
            diff = (p2[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi
            return p1[2] + diff * frac
        elif p1[2] is not None:
            return p1[2]
        elif p2[2] is not None:
            return p2[2]
        else:
            return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


class TrajectoryCritic:
    """Evaluates trajectory quality and identifies problematic segments."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        mv_cfg = config.multiverse_config
        thresholds = mv_cfg.get("critic_thresholds", {})

        self.tortuosity_threshold = thresholds.get("tortuosity", TORTUOSITY_THRESHOLD)
        self.yaw_buffer_rad = thresholds.get("yaw_buffer_rad", YAW_BUFFER_RAD)
        self.velocity_chattering_threshold = thresholds.get("velocity_chattering_threshold", VELOCITY_CHATTERING_THRESHOLD)
        self.jerk_cost_threshold = thresholds.get("jerk_cost", JERK_COST_THRESHOLD)
        self.curvature_cost_threshold = thresholds.get("curvature_cost", CURVATURE_COST_THRESHOLD)
        self.centripetal_cost_threshold = thresholds.get("centripetal_cost", CENTRIPETAL_COST_THRESHOLD)
    
    def evaluate(self, trajectory: List[Dict[str, Any]], num_samples_per_segment: int) -> List[Tuple[int, int]]:
        """
        Evaluate trajectory and identify problematic segments.

        Args:
            trajectory: List of trajectory sample dictionaries
            num_samples_per_segment: Number of samples per segment

        Returns:
            List of (start_idx, end_idx) tuples for problematic windows
        """
        bad_windows = []
        num_segments = (len(trajectory) - 1) // num_samples_per_segment
        
        for i in range(num_segments):
            start_idx = i * num_samples_per_segment
            end_idx = min((i + 2) * num_samples_per_segment, len(trajectory))
            
            segment = trajectory[start_idx:end_idx]
            
            # Calculate metrics
            tortuosity = self._compute_tortuosity(segment)
            yaw_excess = self._compute_yaw_excess(segment)
            chattering = self._compute_velocity_chattering(segment)
            
            # Calculate research-grounded metrics
            jerk_cost = self._calculate_jerk_cost(segment)
            curvature_cost = self._calculate_curvature_cost(segment)
            centripetal_cost = self._calculate_centripetal_cost(segment)
            
            # Check thresholds (including new research-grounded metrics)
            if (tortuosity > self.tortuosity_threshold or
                yaw_excess > self.yaw_buffer_rad or
                chattering > self.velocity_chattering_threshold or
                jerk_cost > self.jerk_cost_threshold or
                curvature_cost > self.curvature_cost_threshold or
                centripetal_cost > self.centripetal_cost_threshold):
                bad_windows.append((i, i + 2))
        
        return bad_windows
    
    def _compute_tortuosity(self, samples: List[Dict[str, Any]]) -> float:
        """Compute path length / straight line distance ratio."""
        path_length = 0.0
        for i in range(len(samples) - 1):
            dx = samples[i + 1]['x'] - samples[i]['x']
            dy = samples[i + 1]['y'] - samples[i]['y']
            path_length += np.sqrt(dx**2 + dy**2)
        
        straight_distance = np.sqrt(
            (samples[-1]['x'] - samples[0]['x'])**2 +
            (samples[-1]['y'] - samples[0]['y'])**2
        )
        
        if straight_distance < 1e-6:
            return 0.0
        return path_length / straight_distance
    
    def _compute_yaw_excess(self, samples: List[Dict[str, Any]]) -> float:
        """Compute excess yaw rate beyond expected turn."""
        total_yaw_change = 0.0
        for i in range(len(samples) - 1):
            dt = samples[i+1]['t'] - samples[i]['t']
            total_yaw_change += abs(samples[i]['omega'] * dt)
        
        expected_yaw = abs(
            (samples[-1]['heading'] - samples[0]['heading'] + np.pi) % (2 * np.pi) - np.pi
        )
        
        return max(0, total_yaw_change - expected_yaw)
    
    def _compute_velocity_chattering(self, samples: List[Dict[str, Any]]) -> int:
        """Count zero-crossings of wheel velocities."""
        vl_crossings = 0
        vr_crossings = 0
        
        for i in range(1, len(samples)):
            if samples[i]['vl'] * samples[i-1]['vl'] < 0:
                vl_crossings += 1
            if samples[i]['vr'] * samples[i-1]['vr'] < 0:
                vr_crossings += 1
        
        return vl_crossings + vr_crossings
    
    def _calculate_jerk_cost(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Calculate jerk cost (rate of acceleration change).

        Research-grounded metric from smoothness optimization literature.
        High jerk indicates rapid acceleration changes leading to tracking errors.

        Args:
            trajectory: List of trajectory sample dictionaries

        Returns:
            Total jerk cost (sum of squared jerk)
        """
        if len(trajectory) < 3:
            return 0.0
        
        jerk_cost = 0.0
        for k in range(len(trajectory) - 2):
            dt_k = trajectory[k+1]['t'] - trajectory[k]['t']
            dt_k1 = trajectory[k+2]['t'] - trajectory[k+1]['t']
            
            # Left wheel jerk
            al_k = (trajectory[k+1]['vl'] - trajectory[k]['vl']) / max(dt_k, 1e-3)
            al_k1 = (trajectory[k+2]['vl'] - trajectory[k+1]['vl']) / max(dt_k1, 1e-3)
            jerk_cost += ((al_k1 - al_k) / max(dt_k1, 1e-3))**2
            
            # Right wheel jerk
            ar_k = (trajectory[k+1]['vr'] - trajectory[k]['vr']) / max(dt_k, 1e-3)
            ar_k1 = (trajectory[k+2]['vr'] - trajectory[k+1]['vr']) / max(dt_k1, 1e-3)
            jerk_cost += ((ar_k1 - ar_k) / max(dt_k1, 1e-3))**2
        
        return jerk_cost
    
    def _calculate_curvature_cost(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Calculate curvature cost (sharpness of turns).

        Research-grounded metric from vehicle trajectory planning.
        High curvature indicates sharp turns that may cause wheel slip.

        Args:
            trajectory: List of trajectory sample dictionaries

        Returns:
            Total curvature cost (sum of squared curvature)
        """
        if len(trajectory) < 2:
            return 0.0
        
        curvature_cost = 0.0
        for k in range(len(trajectory) - 1):
            dx = trajectory[k+1]['x'] - trajectory[k]['x']
            dy = trajectory[k+1]['y'] - trajectory[k]['y']
            ds = np.sqrt(dx**2 + dy**2)
            
            if ds > 1e-6:
                dtheta = abs(trajectory[k+1]['heading'] - trajectory[k]['heading'])
                # Normalize to [-pi, pi]
                dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                curvature = abs(dtheta) / ds
                curvature_cost += curvature**2
        
        return curvature_cost
    
    def _calculate_centripetal_cost(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Calculate centripetal acceleration cost.

        Research-grounded metric from vehicle dynamics.
        Penalizes trajectories approaching friction limits (wheel slip risk).

        Args:
            trajectory: List of trajectory sample dictionaries

        Returns:
            Total centripetal cost (penalty for approaching friction limit)
        """
        cost = 0.0
        a_max = self.config.cof * self.config.mass * self.config.g / self.config.mass  # = cof * g
        
        for state in trajectory:
            v = (state['vl'] + state['vr']) / 2.0
            omega = state['omega']
            a_centripetal = abs(v * omega)
            
            # Penalize approaching friction limit (80% threshold)
            if a_centripetal > 0.8 * a_max:
                cost += (a_centripetal / a_max)**2
        
        return cost


class LocalSegmentOptimizer:
    """Miniature optimizer for solving local trajectory segments."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.model = DifferentialDriveModel(config)

    def solve_window(self, start_state: Tuple[float, float, float, float, float], end_state: Tuple[float, float, float, float, float], num_samples: int, initial_guess: Optional[np.ndarray] = None, apply_headroom: bool = True, direction_constraint: str = 'none', accuracy_weight: float = 0.0, intermediate_waypoints: Optional[List[Tuple[int, float, float, Optional[float]]]] = None) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Solve trajectory for a local window with pinned boundary states.

        Args:
            start_state: (x, y, theta, vl, vr) tuple
            end_state: (x, y, theta, vl, vr) tuple
            num_samples: Number of samples in the window
            initial_guess: Optional initial guess array
            apply_headroom: Whether to apply safety margins
            direction_constraint: 'none', 'forward', or 'reverse'
            accuracy_weight: Weight for smoothness/jerk penalty
            intermediate_waypoints: List of (local_idx, x, y, theta)

        Returns:
            Tuple of (success, cost, trajectory_array)
        """
        N = num_samples
        opti = ca.Opti()
        
        # Decision variables
        dt = opti.variable()
        X = opti.variable(N, 5)  # x, y, theta, vl, vr
        
        x = X[:, 0]
        y = X[:, 1]
        theta = X[:, 2]
        vl = X[:, 3]
        vr = X[:, 4]
        
        # Objective: minimize total time + smoothness penalty
        time_cost = dt * (N - 1)
        smoothness_cost = 0
        if accuracy_weight > 0 and N > 2:
            for k in range(N - 2):
                al_k = (vl[k + 1] - vl[k]) / dt
                al_k1 = (vl[k + 2] - vl[k + 1]) / dt
                ar_k = (vr[k + 1] - vr[k]) / dt
                ar_k1 = (vr[k + 2] - vr[k + 1]) / dt
                smoothness_cost += (al_k1 - al_k)**2 + (ar_k1 - ar_k)**2

        total_cost = time_cost + accuracy_weight * smoothness_cost
        opti.minimize(total_cost)

        # Intermediate waypoint constraints
        if intermediate_waypoints:
            for idx, wx, wy, wtheta in intermediate_waypoints:
                if 0 < idx < N - 1:
                    opti.subject_to(x[idx] == wx)
                    opti.subject_to(y[idx] == wy)
                    if wtheta is not None:
                        opti.subject_to(theta[idx] == wtheta)
        
        # Bounds
        opti.subject_to(dt >= 0.001)
        opti.subject_to(dt <= 1.0)
        opti.subject_to(x >= -20)
        opti.subject_to(x <= 20)
        opti.subject_to(y >= -20)
        opti.subject_to(y <= 20)
        
        v_max = self.config.max_linear_speed(apply_headroom)
        v_bound = 0.99 * v_max
        
        vl_min, vl_max = -v_bound, v_bound
        vr_min, vr_max = -v_bound, v_bound
        
        if direction_constraint == 'forward':
            vl_min, vr_min = 0.0, 0.0
        elif direction_constraint == 'reverse':
            vl_max, vr_max = 0.0, 0.0
            
        opti.subject_to(vl >= vl_min)
        opti.subject_to(vl <= vl_max)
        opti.subject_to(vr >= vr_min)
        opti.subject_to(vr <= vr_max)
        
        # Dynamics constraints
        for k in range(N - 1):
            v1 = (vl[k] + vr[k]) / 2.0
            v2 = (vl[k + 1] + vr[k + 1]) / 2.0
            omega1 = (vr[k] - vl[k]) / self.config.track_width
            omega2 = (vr[k + 1] - vl[k + 1]) / self.config.track_width
            
            opti.subject_to(x[k + 1] == x[k] + 0.5 * (v1 * ca.cos(theta[k]) + v2 * ca.cos(theta[k + 1])) * dt)
            opti.subject_to(y[k + 1] == y[k] + 0.5 * (v1 * ca.sin(theta[k]) + v2 * ca.sin(theta[k + 1])) * dt)
            opti.subject_to(theta[k + 1] == theta[k] + 0.5 * (omega1 + omega2) * dt)
            
            al = (vl[k + 1] - vl[k]) / dt
            ar = (vr[k + 1] - vr[k]) / dt
            
            fl, fr = self.model.get_dynamics_symbolic(vl[k], vr[k], al, ar)
            max_fl = self.config.get_max_force_symbolic(vl[k], apply_headroom)
            max_fr = self.config.get_max_force_symbolic(vr[k], apply_headroom)
            
            opti.subject_to(ca.fabs(fl) <= max_fl)
            opti.subject_to(ca.fabs(fr) <= max_fr)
            
            f_total = ca.fabs(fl) + ca.fabs(fr)
            f_traction_max = self.config.cof * self.config.mass * self.config.g
            opti.subject_to(f_total <= f_traction_max)
        
        # Pin boundary states
        opti.subject_to(x[0] == start_state[0])
        opti.subject_to(y[0] == start_state[1])
        opti.subject_to(theta[0] == start_state[2])
        opti.subject_to(vl[0] == start_state[3])
        opti.subject_to(vr[0] == start_state[4])
        
        opti.subject_to(x[N - 1] == end_state[0])
        opti.subject_to(y[N - 1] == end_state[1])
        opti.subject_to(theta[N - 1] == end_state[2])
        opti.subject_to(vl[N - 1] == end_state[3])
        opti.subject_to(vr[N - 1] == end_state[4])
        
        # Initial guess
        if initial_guess is not None:
            opti.set_initial(dt, float(initial_guess[0]))
            opti.set_initial(X, initial_guess[1:].reshape((N, 5)))
        else:
            opti.set_initial(dt, 0.1)
            opti.set_initial(X, np.zeros((N, 5)))
            for i in range(N):
                frac = i / (N - 1)
                opti.set_initial(x[i], start_state[0] + frac * (end_state[0] - start_state[0]))
                opti.set_initial(y[i], start_state[1] + frac * (end_state[1] - start_state[1]))
                opti.set_initial(theta[i], start_state[2] + frac * (end_state[2] - start_state[2]))
        
        # Solver setup (fast settings for local optimization)
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 2000,
            "print_level": 0,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            "acceptable_tol": 5e-2,
            "acceptable_constr_viol_tol": 5e-2,
            "acceptable_iter": 5,
            "hessian_approximation": "limited-memory",
            "sb": "yes",
        }
        opti.solver("ipopt", p_opts, s_opts)
        
        try:
            sol = opti.solve()
            dt_val = float(sol.value(dt))
            X_val = np.array(sol.value(X))
            cost = float(sol.value(total_cost))
            return True, cost, np.concatenate([[dt_val], X_val.flatten()])
        except Exception:
            return False, float('inf'), None
    


class MultiVerseRefiner:
    """Handles TEB/STOMP parallel refinement of problematic segments."""

    def __init__(self, config: RobotConfig, enable_parallel: Optional[bool] = None, num_workers: Optional[int] = None, verbose: bool = True) -> None:
        self.config = config
        mv_cfg = config.multiverse_config
        self.enable_parallel = enable_parallel if enable_parallel is not None else mv_cfg.get("enable_parallel", True)
        self.num_workers = num_workers if num_workers is not None else mv_cfg.get("num_workers", 8)
        self.verbose = verbose
        self.local_solver = LocalSegmentOptimizer(config)
        self.refinement_history: List[Dict[str, Any]] = []  # Store heuristic results for convergence visualization
        
        # Get multiverse config
        mv_cfg = config.multiverse_config
        stomp_cfg = mv_cfg.get("stomp_noise", {})
        self.stomp_variants = mv_cfg.get("stomp_variants", DEFAULT_STOMP_VARIANTS)
        self.stomp_pos_std = stomp_cfg.get("position_std", STOMP_NOISE_POS_STD)
        self.stomp_heading_std = stomp_cfg.get("heading_std", STOMP_NOISE_HEADING_STD)
        
        teb_cfg = mv_cfg.get("teb_weights", {})
        self.forward_bias = teb_cfg.get("forward_bias", TEB_FORWARD_BIAS)
        self.reverse_bias = teb_cfg.get("reverse_bias", TEB_REVERSE_BIAS)
        self.point_turn_bias = teb_cfg.get("point_turn_bias", TEB_POINT_TURN_BIAS)
        self.wide_sweep_bias = teb_cfg.get("wide_sweep_bias", TEB_WIDE_SWEEP_BIAS)
    
    def refine_segment(self, start_state: Tuple[float, float, float, float, float], end_state: Tuple[float, float, float, float, float], num_samples: int, base_guess: np.ndarray, accuracy_weight: float = 0.0, intermediate_waypoints: Optional[List[Tuple[int, float, float, Optional[float]]]] = None, capture_iterations: bool = False, live_viz: bool = False, override_parallel: Optional[bool] = None) -> Tuple[np.ndarray, str, float]:
        """
        Refine a segment using parallel TEB/STOMP exploration.

        Args:
            start_state: (x, y, theta, vl, vr) tuple
            end_state: (x, y, theta, vl, vr) tuple
            num_samples: Number of samples in the segment
            base_guess: Baseline initial guess array
            accuracy_weight: Weight for smoothness/jerk penalty
            intermediate_waypoints: List of (local_idx, x, y, theta)
            capture_iterations: If True, captures heuristic results for convergence visualization
            live_viz: If True, streams updates to the visualizer
            override_parallel: If provided, overrides self.enable_parallel

        Returns:
            Tuple of (Best trajectory array, name of best heuristic, best cost)
        """
        enable_parallel = self.enable_parallel if override_parallel is None else override_parallel
        # Clear previous refinement history
        self.refinement_history = []
        
        # Generate heuristic guesses
        teb_guesses = self._generate_teb_heuristics(start_state, end_state, num_samples, base_guess)
        stomp_guesses = self._generate_stomp_heuristics(base_guess, num_samples)
        
        all_guesses = teb_guesses + stomp_guesses
        total_guesses = len(all_guesses)
        
        # Filter out infeasible direction constraints
        feasible_guesses = []
        for guess, direction_constraint, name in all_guesses:
            if direction_constraint == 'forward' and (start_state[3] < -0.001 or start_state[4] < -0.001 or end_state[3] < -0.001 or end_state[4] < -0.001):
                continue
            if direction_constraint == 'reverse' and (start_state[3] > 0.001 or start_state[4] > 0.001 or end_state[3] > 0.001 or end_state[4] > 0.001):
                continue
            feasible_guesses.append((guess, direction_constraint, name))
        
        total_feasible = len(feasible_guesses)
        
        best_cost = float('inf')
        best_result = None
        best_name = "None"
        viz = get_visualizer() if live_viz else None
        
        if enable_parallel and total_feasible > 1:
            # Parallel heuristic evaluation using threads
            # CasADi releases the GIL during C-level solves, so threads give real parallelism
            
            def _solve_one(args):
                guess, direction_constraint, name = args
                solver = LocalSegmentOptimizer(self.config)
                success, cost, result = solver.solve_window(
                    start_state, end_state, num_samples, guess, direction_constraint=direction_constraint, accuracy_weight=accuracy_weight, intermediate_waypoints=intermediate_waypoints
                )
                return success, cost, result, name
            
            num_threads = min(self.num_workers, total_feasible)
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(_solve_one, g): i for i, g in enumerate(feasible_guesses)}
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    success, cost, result, name = future.result()
                    
                    if self.verbose:
                        sys.stdout.write(f"\r  Refining: {done_count}/{total_feasible} - Best: {best_cost:.4f}s [{best_name}]")
                        sys.stdout.flush()
                    
                    if success and cost < best_cost:
                        best_cost = cost
                        best_result = result
                        best_name = name
                    
                    if live_viz and success:
                        N = num_samples
                        X_curr = result[1:].reshape((N, 5))
                        traj_data = [{"x": float(X_curr[k, 0]), "y": float(X_curr[k, 1]), "heading": float(X_curr[k, 2])} for k in range(N)]
                        viz.send_state(done_count, traj_data, phase=f"refinement_{name}")
                    
                    if capture_iterations and success:
                        N = num_samples
                        states = result[1:].reshape((N, 5))
                        self.refinement_history.append({
                            'iteration': futures[future],
                            'cost': cost,
                            'trajectory': states.copy(),
                            'dt': float(result[0]),
                            'heuristic_type': name,
                            'heuristic_idx': futures[future]
                        })
        else:
            # Sequential fallback
            for i, (guess, direction_constraint, name) in enumerate(feasible_guesses):
                if self.verbose:
                    progress = (i + 1) / total_feasible * 100
                    sys.stdout.write(f"\r  Refining: {i+1}/{total_feasible} ({progress:.0f}%) - Best cost: {best_cost:.4f} [{name}]")
                    sys.stdout.flush()
                
                success, cost, result = self.local_solver.solve_window(
                    start_state, end_state, num_samples, guess, direction_constraint=direction_constraint, accuracy_weight=accuracy_weight, intermediate_waypoints=intermediate_waypoints
                )
                
                if success and cost < best_cost:
                    best_cost = cost
                    best_result = result
                    best_name = name
        
        if self.verbose:
            sys.stdout.write(f"\r  Refining: {total_feasible}/{total_feasible} (100%) - Best cost: {best_cost:.4f}\n")
            sys.stdout.flush()
        
        if best_result is None:
            # If all refinements failed, calculate cost of base_guess
            dt = base_guess[0]
            states = base_guess[1:].reshape((num_samples, 5))
            time_cost = dt * (num_samples - 1)
            smoothness_cost = 0
            if accuracy_weight > 0 and num_samples > 2:
                for k in range(num_samples - 2):
                    al_k = (states[k+1, 3] - states[k, 3]) / dt
                    al_k1 = (states[k+2, 3] - states[k+1, 3]) / dt
                    ar_k = (states[k+1, 4] - states[k, 4]) / dt
                    ar_k1 = (states[k+2, 4] - states[k+1, 4]) / dt
                    smoothness_cost += (al_k1 - al_k)**2 + (ar_k1 - ar_k)**2
            best_cost = time_cost + accuracy_weight * smoothness_cost

        return best_result if best_result is not None else base_guess, best_name if best_result is not None else "None", best_cost
    
    def _generate_teb_heuristics(self, start_state: Tuple[float, float, float, float, float], end_state: Tuple[float, float, float, float, float], num_samples: int, base_guess: np.ndarray, intermediate_waypoints: Optional[List[Tuple[int, float, float, Optional[float]]]] = None) -> List[Tuple[np.ndarray, str, str]]:
        """Generate TEB topology-based initial guesses with strict bounds."""
        guesses = []
        N = num_samples
        
        # 1. Base Guess
        guesses.append((base_guess.copy(), 'none', 'TEB_Base'))

        # 1b. Direct Path (Straight Heading Candidate)
        # This addresses "loops" by suggesting a straight heading through the segment
        direct_guess = base_guess.copy()
        dx = end_state[0] - start_state[0]
        dy = end_state[1] - start_state[1]
        direct_heading = np.arctan2(dy, dx)
        for i in range(N):
            idx = 1 + i * 5
            frac = i / (N - 1)
            # Interpolate position to break out of loops
            direct_guess[idx] = start_state[0] + frac * dx
            direct_guess[idx+1] = start_state[1] + frac * dy
            # Force straight heading
            direct_guess[idx+2] = direct_heading
            # Reset velocities to small forward nudge to break symmetry
            direct_guess[idx+3] = 0.05
            direct_guess[idx+4] = 0.05
        guesses.append((direct_guess, 'none', 'Direct_Heading'))

        # 1b-rev. Reverse Direct Path
        rev_direct_guess = base_guess.copy()
        rev_direct_heading = (direct_heading + np.pi + np.pi) % (2*np.pi) - np.pi
        for i in range(N):
            idx = 1 + i * 5
            frac = i / (N - 1)
            rev_direct_guess[idx] = start_state[0] + frac * dx
            rev_direct_guess[idx+1] = start_state[1] + frac * dy
            rev_direct_guess[idx+2] = rev_direct_heading
            rev_direct_guess[idx+3] = -0.05
            rev_direct_guess[idx+4] = -0.05
        guesses.append((rev_direct_guess, 'none', 'Reverse_Direct_Heading'))

        # 1c. Intermediate Heading Candidates
        if intermediate_waypoints:
            for local_idx, wx, wy, wtheta in intermediate_waypoints:
                if wtheta is None: # Only if it was unconstrained
                    # Try headings aligned with segments
                    h_in = np.arctan2(wy - start_state[1], wx - start_state[0])
                    h_out = np.arctan2(end_state[1] - wy, end_state[0] - wx)
                    h_avg = np.arctan2(np.sin(h_in)+np.sin(h_out), np.cos(h_in)+np.cos(h_out))

                    for h_cand, name in [(h_in, "In"), (h_out, "Out"), (h_avg, "Avg")]:
                        cand_guess = base_guess.copy()
                        # Force position to be exactly on the waypoint
                        cand_guess[1 + local_idx * 5] = wx
                        cand_guess[1 + local_idx * 5 + 1] = wy
                        # Seed the heading at the intermediate waypoint
                        cand_guess[1 + local_idx * 5 + 2] = h_cand
                        # Also nudge surrounding points to help solver
                        # For h_in, nudge points before. For h_out, nudge points after.
                        if name == "In":
                            for neighbor in range(max(0, local_idx-4), local_idx):
                                cand_guess[1 + neighbor * 5 + 2] = h_cand
                        elif name == "Out":
                            for neighbor in range(local_idx+1, min(N, local_idx+5)):
                                cand_guess[1 + neighbor * 5 + 2] = h_cand
                        else: # Avg
                            for neighbor in range(max(0, local_idx-2), min(N, local_idx+3)):
                                cand_guess[1 + neighbor * 5 + 2] = h_cand

                        guesses.append((cand_guess, 'none', f'Heuristic_Heading_{name}'))

                        # Also try reverse
                        rev_cand = (h_cand + np.pi + np.pi) % (2*np.pi) - np.pi
                        rev_guess = base_guess.copy()
                        rev_guess[1 + local_idx * 5] = wx
                        rev_guess[1 + local_idx * 5 + 1] = wy
                        rev_guess[1 + local_idx * 5 + 2] = rev_cand
                        if name == "In":
                            for neighbor in range(max(0, local_idx-4), local_idx):
                                rev_guess[1 + neighbor * 5 + 2] = rev_cand
                        elif name == "Out":
                            for neighbor in range(local_idx+1, min(N, local_idx+5)):
                                rev_guess[1 + neighbor * 5 + 2] = rev_cand
                        else: # Avg
                            for neighbor in range(max(0, local_idx-2), min(N, local_idx+3)):
                                rev_guess[1 + neighbor * 5 + 2] = rev_cand
                        guesses.append((rev_guess, 'none', f'Heuristic_Heading_{name}_Rev'))
        
        # 2. Strict Forward Bound
        forward_guess = base_guess.copy()
        for i in range(N):
            idx = 1 + i * 5
            forward_guess[idx + 3] = abs(forward_guess[idx + 3]) + 0.1  # vl
            forward_guess[idx + 4] = abs(forward_guess[idx + 4]) + 0.1  # vr
        guesses.append((forward_guess, 'forward', 'Bounded_Forward'))
        
        # 3. Strict Reverse Bound
        reverse_guess = base_guess.copy()
        for i in range(N):
            idx = 1 + i * 5
            reverse_guess[idx + 3] = -abs(reverse_guess[idx + 3]) - 0.1  # vl
            reverse_guess[idx + 4] = -abs(reverse_guess[idx + 4]) - 0.1  # vr
        guesses.append((reverse_guess, 'reverse', 'Bounded_Reverse'))
        
        # Get robot limits for heuristic velocities
        v_limit = self.config.v_max_rad_s * self.config.wheel_radius * 0.7
        spin_v = v_limit * 0.5
        
        # 4. Forward Point-Turn Override (3-phase: Turn-Drive-Turn)
        pt_guess = base_guess.copy()
        dx = end_state[0] - start_state[0]
        dy = end_state[1] - start_state[1]
        target_angle = np.arctan2(dy, dx)
        
        # Divide into 3 phases: turn to target, drive, turn to end heading
        phase_n = N // 3
        for i in range(N):
            idx = 1 + i * 5
            if i < phase_n:
                # Phase 1: Point turn to target_angle
                frac = i / phase_n
                theta_val = start_state[2] + frac * (target_angle - start_state[2])
                pt_guess[idx] = start_state[0]
                pt_guess[idx+1] = start_state[1]
                pt_guess[idx+2] = theta_val
                pt_guess[idx+3], pt_guess[idx+4] = -spin_v, spin_v
            elif i < 2 * phase_n:
                # Phase 2: Drive to end position
                frac = (i - phase_n) / phase_n
                pt_guess[idx] = start_state[0] + frac * dx
                pt_guess[idx+1] = start_state[1] + frac * dy
                pt_guess[idx+2] = target_angle
                pt_guess[idx+3], pt_guess[idx+4] = v_limit, v_limit
            else:
                # Phase 3: Point turn to end heading
                frac = (i - 2 * phase_n) / (N - 1 - 2 * phase_n) if N - 1 > 2 * phase_n else 1.0
                theta_val = target_angle + frac * (end_state[2] - target_angle)
                pt_guess[idx] = end_state[0]
                pt_guess[idx+1] = end_state[1]
                pt_guess[idx+2] = theta_val
                pt_guess[idx+3], pt_guess[idx+4] = -spin_v, spin_v
        guesses.append((pt_guess, 'none', 'Point_Turn_Forward'))

        # 5. Reverse Point-Turn Override (3-phase: Turn-Back-Turn)
        ptr_guess = base_guess.copy()
        target_angle_rev = target_angle + np.pi
        for i in range(N):
            idx = 1 + i * 5
            if i < phase_n:
                frac = i / phase_n
                theta_val = start_state[2] + frac * (target_angle_rev - start_state[2])
                ptr_guess[idx] = start_state[0]
                ptr_guess[idx+1] = start_state[1]
                ptr_guess[idx+2] = theta_val
                ptr_guess[idx+3], ptr_guess[idx+4] = -spin_v, spin_v
            elif i < 2 * phase_n:
                frac = (i - phase_n) / phase_n
                ptr_guess[idx] = start_state[0] + frac * dx
                ptr_guess[idx+1] = start_state[1] + frac * dy
                ptr_guess[idx+2] = target_angle_rev
                ptr_guess[idx+3], ptr_guess[idx+4] = -v_limit, -v_limit
            else:
                frac = (i - 2 * phase_n) / (N - 1 - 2 * phase_n) if N - 1 > 2 * phase_n else 1.0
                theta_val = target_angle_rev + frac * (end_state[2] - target_angle_rev)
                ptr_guess[idx] = end_state[0]
                ptr_guess[idx+1] = end_state[1]
                ptr_guess[idx+2] = theta_val
                ptr_guess[idx+3], ptr_guess[idx+4] = -spin_v, spin_v
        guesses.append((ptr_guess, 'none', 'Point_Turn_Reverse'))
        
        # 6. Wide sweep: add lateral offset
        wide_sweep_guess = base_guess.copy()
        for i in range(N):
            idx = 1 + i * 5
            offset = 0.1 * np.sin(np.pi * i / N)
            wide_sweep_guess[idx + 1] += offset
        guesses.append((wide_sweep_guess, 'none', 'Wide_Sweep'))
        
        # 7. Heading diversity heuristics - vary heading at midpoint
        for heading_offset in [np.pi/4, -np.pi/4, np.pi/2, -np.pi/2]:
            heading_diverse_guess = base_guess.copy()
            mid_idx = N // 2
            for i in range(N):
                idx = 1 + i * 5
                # Apply heading offset with smooth transition
                if i < mid_idx:
                    frac = i / mid_idx
                    heading_diverse_guess[idx + 2] += heading_offset * frac
                else:
                    frac = (i - mid_idx) / (N - mid_idx)
                    heading_diverse_guess[idx + 2] += heading_offset * (1 - frac)
            guesses.append((heading_diverse_guess, 'none', f'Heading_Div_{heading_offset:.2f}'))
        
        # 8. Perpendicular approach heuristic
        perp_guess = base_guess.copy()
        perp_angle = target_angle + np.pi/2
        for i in range(N):
            idx = 1 + i * 5
            frac = i / (N - 1)
            # Blend from start heading to perpendicular, then to end heading
            if i < N // 2:
                perp_guess[idx + 2] = start_state[2] + frac * 2 * (perp_angle - start_state[2])
            else:
                perp_guess[idx + 2] = perp_angle + (frac - 0.5) * 2 * (end_state[2] - perp_angle)
        guesses.append((perp_guess, 'none', 'Perpendicular_Approach'))
        
        return guesses
    
    def _generate_stomp_heuristics(self, base_guess: np.ndarray, num_samples: int) -> List[Tuple[np.ndarray, str, str]]:
        """Generate STOMP stochastic perturbation guesses."""
        guesses = []
        N = num_samples
        
        # Increase heading noise for better diversity at unconstrained waypoints
        increased_heading_std = self.stomp_heading_std * 3.0  # Triple the heading noise
        
        for v in range(self.stomp_variants):
            noisy_guess = base_guess.copy()
            
            # Add Gaussian noise to positions and headings
            for i in range(N):
                idx = 1 + i * 5
                noisy_guess[idx] += np.random.normal(0, self.stomp_pos_std)  # x
                noisy_guess[idx + 1] += np.random.normal(0, self.stomp_pos_std)  # y
                noisy_guess[idx + 2] += np.random.normal(0, increased_heading_std)  # theta (increased noise)
            
            guesses.append((noisy_guess, 'none', f'STOMP_Noise_{v}'))
            
        # Alternate Approach (180 deg flip of unconstrained target)
        # Note: If the target heading was actually constrained, the solver will still
        # enforce it, but the guess will encourage a loop.
        flip_guess = base_guess.copy()
        for i in range(N // 2, N):
            idx = 1 + i * 5
            flip_guess[idx + 2] += np.pi
            flip_guess[idx + 3] *= -1
            flip_guess[idx + 4] *= -1
        guesses.append((flip_guess, 'none', 'STOMP_180_Flip'))
        
        # Add 90-degree flip heuristic
        flip_90_guess = base_guess.copy()
        for i in range(N // 3, 2 * N // 3):
            idx = 1 + i * 5
            flip_90_guess[idx + 2] += np.pi / 2
        guesses.append((flip_90_guess, 'none', 'STOMP_90_Flip'))
        
        # Add -90-degree flip heuristic
        flip_neg_90_guess = base_guess.copy()
        for i in range(N // 3, 2 * N // 3):
            idx = 1 + i * 5
            flip_neg_90_guess[idx + 2] -= np.pi / 2
        guesses.append((flip_neg_90_guess, 'none', 'STOMP_-90_Flip'))
        
        return guesses
    
    def _solve_window_wrapper(self, start_state: Tuple[float, float, float, float, float], end_state: Tuple[float, float, float, float, float], num_samples: int, guess: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """Wrapper for parallel execution."""
        # For now, run sequentially to avoid pickling issues with CasADi
        # True parallel execution would require serializing the config
        return self.local_solver.solve_window(start_state, end_state, num_samples, guess)


class MasterTrajectoryOptimizer:
    """Orchestrates the Multi-Verse refinement pipeline."""

    def __init__(self, config: RobotConfig, enable_parallel: Optional[bool] = None, num_workers: Optional[int] = None, verbose: bool = True) -> None:
        self.config = config
        mv_cfg = config.multiverse_config
        self.enable_parallel = enable_parallel if enable_parallel is not None else mv_cfg.get("enable_parallel", True)
        self.num_workers = num_workers if num_workers is not None else mv_cfg.get("num_workers", 8)
        self.verbose = verbose
        self.bootstrapper = PathBootstrapper(config)
        self.critic = TrajectoryCritic(config)
        self.refiner = MultiVerseRefiner(config, enable_parallel, num_workers, verbose)
        self.iteration_history: List[Dict[str, Any]] = []  # Store convergence data across phases
        self.stats = OptimizationStats()

    
    def solve(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10, accuracy_weight: float = 0.0,
              stop_waypoint_indices: Optional[List[int]] = None, waypoint_events: Optional[Dict[int, str]] = None, apply_headroom: bool = True, verbose: bool = True, capture_iterations: bool = False, live_viz: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Solve trajectory using Multi-Verse refinement pipeline.

        Args:
            waypoints: List of (x, y, heading) tuples
            num_samples_per_segment: Number of samples per segment
            accuracy_weight: Smoothness weight
            stop_waypoint_indices: Indices where robot must stop
            waypoint_events: Event markers
            apply_headroom: Apply safety margins
            verbose: If True, prints progress messages
            capture_iterations: If True, captures intermediate states for convergence visualization

        Returns:
            Tuple of (samples, stats_dict)
        """
        import time
        
        # Reset stats
        self.stats = OptimizationStats()

        total_start_time = time.time()
        num_segments = len(waypoints) - 1
        N = num_segments * num_samples_per_segment + 1
        
        if stop_waypoint_indices is None:
            stop_waypoint_indices = []
        if waypoint_events is None:
            waypoint_events = {}
        
        # Phase 1: Bootstrap with Reeds-Shepp
        phase1_start = time.time()
        if verbose:
            print("Phase 1: Bootstrapping with Reeds-Shepp paths...")
        guess = self.bootstrapper.generate_baseline(waypoints, num_samples_per_segment)
        phase1_time = time.time() - phase1_start
        self.stats.phase_times["bootstrap"] = phase1_time

        
        # Capture bootstrap phase
        if capture_iterations:
            N = num_segments * num_samples_per_segment + 1
            guess_states = guess[1:].reshape((N, 5))
            bootstrap_cost = guess[0] * (N - 1)  # Time cost approximation
            self.iteration_history.append({
                'iteration': len(self.iteration_history),
                'cost': bootstrap_cost,
                'trajectory': guess_states.copy(),
                'dt': float(guess[0]),
                'phase': 'bootstrap'
            })
        
        # Phase 2: Fast global solve
        phase2_start = time.time()
        if verbose:
            print("Phase 2: Fast global optimization...")
        global_traj = self._global_solve(waypoints, num_samples_per_segment, guess, 
                                        accuracy_weight, stop_waypoint_indices, 
                                        waypoint_events, apply_headroom, 
                                        fast_mode=True, capture_iterations=capture_iterations, live_viz=live_viz)
        phase2_time = time.time() - phase2_start
        self.stats.phase_times["global_solve"] = phase2_time
        
        # Calculate initial cost
        if global_traj:
            N_global = len(global_traj)
            dt_global = global_traj[1]['t'] - global_traj[0]['t'] if N_global > 1 else 0.1
            self.stats.initial_cost = dt_global * (N_global - 1)

        
        # Capture global solve phase
        if capture_iterations:
            N = len(global_traj)
            dt = global_traj[1]['t'] - global_traj[0]['t'] if N > 1 else 0.1
            global_states = np.array([[s['x'], s['y'], s['heading'], s['vl'], s['vr']] for s in global_traj])
            global_cost = dt * (N - 1)
            self.iteration_history.append({
                'iteration': len(self.iteration_history),
                'cost': global_cost,
                'trajectory': global_states,
                'dt': dt,
                'phase': 'global_solve'
            })
        
        # Phase 3: Critic evaluation
        phase3_start = time.time()
        if verbose:
            print("Phase 3: Evaluating trajectory quality...")
        bad_windows = self.critic.evaluate(global_traj, num_samples_per_segment)
        phase3_time = time.time() - phase3_start
        self.stats.phase_times["critic"] = phase3_time
        self.stats.bad_segments_found = len(bad_windows)
        
        if verbose:
            print(f"Found {len(bad_windows)} problematic segments")

        
        # Phase 4: Refine bad segments
        phase4_start = time.time()
        refinement_count = 0
        refinement_success_count = 0
        heuristic_improvement_count = 0
        heuristic_win_log = []  # Track which heuristics beat the original
        
        if len(bad_windows) > 0:
            if verbose:
                print(f"Phase 4: Refining {len(bad_windows)} problematic segments...")
            
            # Group windows into disjoint batches to allow parallel refinement across windows
            # Two windows overlap if they share any segments.
            # Window (i, i+2) covers segments i and i+1.
            def windows_overlap(w1, w2):
                s1, e1 = w1[0], w1[1]
                s2, e2 = w2[0], w2[1]
                # Segments for w1: [s1, e1-1]
                # Segments for w2: [s2, e2-1]
                return not (e1 - 1 < s2 or e2 - 1 < s1)

            batches = []
            for w in bad_windows:
                placed = False
                for batch in batches:
                    if not any(windows_overlap(w, bw) for bw in batch):
                        batch.append(w)
                        placed = True
                        break
                if not placed:
                    batches.append([w])
            
            if self.verbose:
                print(f"  Split into {len(batches)} parallel batches")

            for bi, batch in enumerate(batches):
                if verbose:
                    print(f"  Batch {bi+1}/{len(batches)} ({len(batch)} windows)")
                
                if self.enable_parallel and len(batch) > 1:
                    
                    def _refine_window_task(w_info):
                        try:
                            window_start, window_end = w_info
                            # Extract window data
                            start_idx = window_start * num_samples_per_segment
                            end_idx = min(window_end * num_samples_per_segment, len(global_traj) - 1)
                            
                            start_state = (
                                global_traj[start_idx]['x'],
                                global_traj[start_idx]['y'],
                                global_traj[start_idx]['heading'],
                                global_traj[start_idx]['vl'],
                                global_traj[start_idx]['vr']
                            )
                            
                            end_state = (
                                global_traj[end_idx]['x'],
                                global_traj[end_idx]['y'],
                                global_traj[end_idx]['heading'],
                                global_traj[end_idx]['vl'],
                                global_traj[end_idx]['vr']
                            )
                            
                            num_window_samples = end_idx - start_idx + 1
                            original_dt = (global_traj[end_idx]['t'] - global_traj[start_idx]['t']) / max(num_window_samples - 1, 1)

                            # Calculate original cost with smoothness penalty
                            original_time_cost = original_dt * (num_window_samples - 1)
                            original_smoothness_cost = 0
                            if accuracy_weight > 0 and num_window_samples > 2:
                                for k in range(num_window_samples - 2):
                                    al_k = (global_traj[start_idx + k + 1]['vl'] - global_traj[start_idx + k]['vl']) / original_dt
                                    al_k1 = (global_traj[start_idx + k + 2]['vl'] - global_traj[start_idx + k + 1]['vl']) / original_dt
                                    ar_k = (global_traj[start_idx + k + 1]['vr'] - global_traj[start_idx + k]['vr']) / original_dt
                                    ar_k1 = (global_traj[start_idx + k + 2]['vr'] - global_traj[start_idx + k + 1]['vr']) / original_dt
                                    original_smoothness_cost += (al_k1 - al_k)**2 + (ar_k1 - ar_k)**2

                            original_cost = original_time_cost + accuracy_weight * original_smoothness_cost
                            
                            base_guess = np.zeros(1 + num_window_samples * 5)
                            base_guess[0] = original_dt if original_dt > 0.001 else 0.1
                            for j in range(num_window_samples):
                                sample = global_traj[start_idx + j]
                                base_guess[1 + j * 5] = sample['x']
                                base_guess[1 + j * 5 + 1] = sample['y']
                                base_guess[1 + j * 5 + 2] = sample['heading']
                                base_guess[1 + j * 5 + 3] = sample['vl']
                                base_guess[1 + j * 5 + 4] = sample['vr']
                                
                            # Extract intermediate waypoints for this window
                            inter_wps = []
                            for iw in range(window_start + 1, window_end):
                                local_idx = (iw - window_start) * num_samples_per_segment
                                wp = waypoints[iw]
                                inter_wps.append((local_idx, wp[0], wp[1], wp[2]))

                            refined, name, refined_cost = self.refiner.refine_segment(
                                start_state, end_state, num_window_samples, base_guess, accuracy_weight=accuracy_weight,
                                intermediate_waypoints=inter_wps,
                                live_viz=live_viz, override_parallel=False)
                            
                            return (w_info, refined, name, refined_cost, original_cost, start_idx, end_idx)
                        except Exception as e:
                            import traceback
                            print(f"\nError in window task {w_info}: {e}")
                            traceback.print_exc()
                            return (w_info, None, "Error", 0, 0, 0)

                    with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                        futures = [executor.submit(_refine_window_task, w) for w in batch]
                        for future in as_completed(futures):
                            result_data = future.result()
                            if result_data[2] == "Error":
                                continue
                            w_info, refined, best_heuristic_name, refined_cost, original_cost, start_idx, end_idx = result_data
                            window_start, window_end = w_info
                            num_window_samples = end_idx - start_idx + 1
                            
                            refinement_count += 1
                            if refined is not None and best_heuristic_name != "None":
                                refinement_success_count += 1
                                
                                if refined_cost < original_cost:
                                    heuristic_improvement_count += 1
                                    improvement_pct = (1 - refined_cost / original_cost) * 100 if original_cost > 0 else 0
                                    heuristic_win_log.append({
                                        'window': f"W{window_start}-W{window_end}",
                                        'heuristic': best_heuristic_name,
                                        'original_cost': original_cost,
                                        'refined_cost': refined_cost,
                                        'improvement_pct': improvement_pct
                                    })
                                
                                dt_val = refined[0]
                                states = refined[1:].reshape((num_window_samples, 5))
                                for j in range(num_window_samples):
                                    idx = start_idx + j
                                    global_traj[idx]['x'] = float(states[j, 0])
                                    global_traj[idx]['y'] = float(states[j, 1])
                                    global_traj[idx]['heading'] = float(states[j, 2])
                                    global_traj[idx]['vl'] = float(states[j, 3])
                                    global_traj[idx]['vr'] = float(states[j, 4])
                                    global_traj[idx]['omega'] = float((states[j, 4] - states[j, 3]) / self.config.track_width)
                else:
                    # Sequential processing for this batch (or if parallel disabled)
                    for wi, (window_start, window_end) in enumerate(batch):
                        # ... extract window data ...
                        start_idx = window_start * num_samples_per_segment
                        end_idx = min(window_end * num_samples_per_segment, len(global_traj) - 1)
                        
                        start_state = (
                            global_traj[start_idx]['x'], global_traj[start_idx]['y'], global_traj[start_idx]['heading'],
                            global_traj[start_idx]['vl'], global_traj[start_idx]['vr']
                        )
                        end_state = (
                            global_traj[end_idx]['x'], global_traj[end_idx]['y'], global_traj[end_idx]['heading'],
                            global_traj[end_idx]['vl'], global_traj[end_idx]['vr']
                        )
                        
                        num_window_samples = end_idx - start_idx + 1
                        original_dt = (global_traj[end_idx]['t'] - global_traj[start_idx]['t']) / max(num_window_samples - 1, 1)

                        # Calculate original cost with smoothness penalty
                        original_time_cost = original_dt * (num_window_samples - 1)
                        original_smoothness_cost = 0
                        if accuracy_weight > 0 and num_window_samples > 2:
                            for k in range(num_window_samples - 2):
                                al_k = (global_traj[start_idx + k + 1]['vl'] - global_traj[start_idx + k]['vl']) / original_dt
                                al_k1 = (global_traj[start_idx + k + 2]['vl'] - global_traj[start_idx + k + 1]['vl']) / original_dt
                                ar_k = (global_traj[start_idx + k + 1]['vr'] - global_traj[start_idx + k]['vr']) / original_dt
                                ar_k1 = (global_traj[start_idx + k + 2]['vr'] - global_traj[start_idx + k + 1]['vr']) / original_dt
                                original_smoothness_cost += (al_k1 - al_k)**2 + (ar_k1 - ar_k)**2

                        original_cost = original_time_cost + accuracy_weight * original_smoothness_cost
                        
                        base_guess = np.zeros(1 + num_window_samples * 5)
                        base_guess[0] = original_dt if original_dt > 0.001 else 0.1
                        for j in range(num_window_samples):
                            sample = global_traj[start_idx + j]
                            base_guess[1 + j * 5] = sample['x']
                            base_guess[1 + j * 5 + 1] = sample['y']
                            base_guess[1 + j * 5 + 2] = sample['heading']
                            base_guess[1 + j * 5 + 3] = sample['vl']
                            base_guess[1 + j * 5 + 4] = sample['vr']
                        
                        # Extract intermediate waypoints for this window
                        inter_wps = []
                        for iw in range(window_start + 1, window_end):
                            local_idx = (iw - window_start) * num_samples_per_segment
                            wp = waypoints[iw]
                            inter_wps.append((local_idx, wp[0], wp[1], wp[2]))

                        refinement_count += 1
                        refined, best_heuristic_name, refined_cost = self.refiner.refine_segment(
                            start_state, end_state, num_window_samples, base_guess, accuracy_weight=accuracy_weight,
                            intermediate_waypoints=inter_wps, live_viz=live_viz)
                        
                        if refined is not None and best_heuristic_name != "None":
                            refinement_success_count += 1
                            if refined_cost < original_cost:
                                heuristic_improvement_count += 1
                                improvement_pct = (1 - refined_cost / original_cost) * 100 if original_cost > 0 else 0
                                heuristic_win_log.append({
                                    'window': f"W{window_start}-W{window_end}",
                                    'heuristic': best_heuristic_name,
                                    'original_cost': original_cost,
                                    'refined_cost': refined_cost,
                                    'improvement_pct': improvement_pct
                                })
                            
                            dt_val = refined[0]
                            states = refined[1:].reshape((num_window_samples, 5))
                            for j in range(num_window_samples):
                                idx = start_idx + j
                                global_traj[idx]['x'] = float(states[j, 0])
                                global_traj[idx]['y'] = float(states[j, 1])
                                global_traj[idx]['heading'] = float(states[j, 2])
                                global_traj[idx]['vl'] = float(states[j, 3])
                                global_traj[idx]['vr'] = float(states[j, 4])
                                global_traj[idx]['omega'] = float((states[j, 4] - states[j, 3]) / self.config.track_width)

            
            if verbose and len(bad_windows) > 1:
                sys.stdout.write("\n")
                sys.stdout.flush()
        
        phase4_time = time.time() - phase4_start
        
        # Phase 5: Final polish
        phase5_start = time.time()
        if verbose:
            print("Phase 5: Final global polish...")
        final_traj = self._global_solve(waypoints, num_samples_per_segment, guess,
                                       accuracy_weight, stop_waypoint_indices,
                                       waypoint_events, apply_headroom,
                                       fast_mode=False, initial_samples=global_traj,
                                       capture_iterations=capture_iterations, live_viz=live_viz)
        phase5_time = time.time() - phase5_start
        self.stats.phase_times["polish"] = phase5_time
        
        total_time = time.time() - total_start_time
        self.stats.total_time = total_time
        
        # Final cost
        if final_traj:
            N_final = len(final_traj)
            dt_final = final_traj[1]['t'] - final_traj[0]['t'] if N_final > 1 else 0.1
            self.stats.final_cost = dt_final * (N_final - 1)

        # Update stats counts
        self.stats.refinements_attempted = refinement_count
        self.stats.refinements_solved = refinement_success_count
        self.stats.heuristic_wins = heuristic_win_log
        
        # ── Optimization Summary ──────────────────────────────────────────────

        if verbose:
            print(f"\n{'='*60}")
            print(f"  OPTIMIZATION SUMMARY")
            print(f"{'='*60}")
            print(f"  Total time:          {total_time:.2f}s")
            print(f"  Phase 1 (Bootstrap): {phase1_time:.2f}s")
            print(f"  Phase 2 (Global):    {phase2_time:.2f}s")
            print(f"  Phase 3 (Critic):    {phase3_time:.2f}s")
            print(f"  Phase 4 (Refine):    {phase4_time:.2f}s")
            print(f"  Phase 5 (Polish):    {phase5_time:.2f}s")
            print(f"  Bad segments found:  {len(bad_windows)}")
            print(f"  Refinements run:     {refinement_count}")
            print(f"  Refinements solved:  {refinement_success_count}")
            print(f"  Heuristic wins:      {heuristic_improvement_count}")
            
            if heuristic_win_log:
                print(f"\n  {'Window':<12} {'Heuristic':<22} {'Orig (s)':<10} {'New (s)':<10} {'Improvement'}")
                print(f"  {'-'*12} {'-'*22} {'-'*10} {'-'*10} {'-'*12}")
                for win in heuristic_win_log:
                    print(f"  {win['window']:<12} {win['heuristic']:<22} {win['original_cost']:<10.4f} {win['refined_cost']:<10.4f} {win['improvement_pct']:.1f}%")
            
            print(f"{'='*60}\n")
        
        return final_traj, self.stats.to_dict()
    
    def _global_solve(self, waypoints, num_samples_per_segment, initial_guess,
                     accuracy_weight, stop_waypoint_indices, waypoint_events,
                     apply_headroom, fast_mode=False, initial_samples=None, capture_iterations=False, live_viz=False):
        """Global trajectory optimization using CasADi."""
        from optimizer import TrajectoryOptimizer
        
        # Use existing optimizer for global solve
        optimizer = TrajectoryOptimizer(self.config)
        
        # Modify guess if provided
        if initial_samples is not None:
            # Convert samples back to guess format
            N = len(initial_samples)
            dt = initial_samples[1]['t'] - initial_samples[0]['t'] if N > 1 else 0.1
            guess = [dt]
            for sample in initial_samples:
                guess.extend([sample['x'], sample['y'], sample['heading'], 
                            sample['vl'], sample['vr']])
            initial_guess = np.array(guess)
        
        # Adjust solver settings for fast mode
        if fast_mode:
            # Temporarily modify solver settings for fast solve
            # This would require modifying the optimizer class
            pass
        
        samples, _stats = optimizer.solve(waypoints, num_samples_per_segment, accuracy_weight,
                              stop_waypoint_indices, waypoint_events, apply_headroom, 
                              self.verbose, capture_iterations=capture_iterations, live_viz=live_viz)
        return samples
