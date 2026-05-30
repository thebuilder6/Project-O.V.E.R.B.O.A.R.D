"""
Multi-Verse Segment Refinement Pipeline (JAX/Immrax Optimized)

A hybrid trajectory generation architecture combining:
- Gradient-based optimization (CasADi)
- Topology exploration (TEB)
- JAX-accelerated parallel refinement and candidate generation
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

# JAX Imports
import jax
import jax.numpy as jnp
from jax_robot_model import JAXRobotConfig
from jax_optimizer import generate_candidates_jax, get_jax_refiner

def resample_window_cpu(states_array, target_n=10):
    """
    Resamples a trajectory window [M x 5] to exactly target_n samples using linear interpolation.
    states_array: numpy array of shape (M, 5) where columns are [x, y, theta, vl, vr]
    """
    M = states_array.shape[0]
    if M == target_n:
        return states_array

    original_xs = np.linspace(0, 1, M)
    target_xs = np.linspace(0, 1, target_n)

    resampled = np.zeros((target_n, states_array.shape[1]))
    for i in range(states_array.shape[1]):
        if i == 2: # Heading: handle wrapping
            unwrapped = np.unwrap(states_array[:, i])
            resampled[:, i] = (np.interp(target_xs, original_xs, unwrapped) + np.pi) % (2 * np.pi) - np.pi
        else:
            resampled[:, i] = np.interp(target_xs, original_xs, states_array[:, i])

    return resampled

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
TORTUOSITY_THRESHOLD = 1.2
YAW_BUFFER_RAD = 0.3
VELOCITY_CHATTERING_THRESHOLD = 2
JERK_COST_THRESHOLD = 5.0
CURVATURE_COST_THRESHOLD = 2.0
CENTRIPETAL_COST_THRESHOLD = 0.8

STOMP_NOISE_POS_STD = 0.05
STOMP_NOISE_HEADING_STD = 0.1
DEFAULT_STOMP_VARIANTS = 50

# JAX Constants
JAX_TARGET_SAMPLES = 10

class PathBootstrapper:
    """Generates kinematically valid initial guesses using Reeds-Shepp paths."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        mv_cfg = config.multiverse_config
        bootstrap_cfg = mv_cfg.get("bootstrap", {})
        self.turning_radius = bootstrap_cfg.get("turning_radius", 0.0)
        self.resolution = bootstrap_cfg.get("resolution", 0.05)
        self.rs_path = ReedsSheppPath(self.turning_radius)

    def generate_baseline(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10) -> np.ndarray:
        num_segments = len(waypoints) - 1
        N = num_segments * num_samples_per_segment + 1
        initial_dt = 0.1
        guess = [initial_dt]
        for i in range(num_segments):
            p1, p2 = waypoints[i], waypoints[i + 1]
            count = num_samples_per_segment if i < num_segments - 1 else num_samples_per_segment + 1
            segment_waypoints = self._generate_segment_waypoints(p1, p2, count)
            for j in range(count):
                if j < len(segment_waypoints): x, y, theta = segment_waypoints[j]
                else:
                    frac = j / num_samples_per_segment
                    x = p1[0] + (p2[0] - p1[0]) * frac
                    y = p1[1] + (p2[1] - p1[1]) * frac
                    theta = self._interpolate_heading(p1, p2, frac)
                guess.extend([x, y, theta, 0.0, 0.0])
        return np.array(guess)
    
    def _generate_segment_waypoints(self, start, goal, num_points):
        if start[2] is not None and goal[2] is not None:
            rs_path = self.rs_path.plan(start, goal, step_size=self.resolution)
            if rs_path is not None and len(rs_path) > 2: return self._resample_path(rs_path, num_points)
        return linear_interpolation_waypoints(start, goal, num_points)

    def _resample_path(self, path, num_points):
        if len(path) < 2: return path
        dists = [0.0]
        for i in range(len(path) - 1): dists.append(dists[-1] + np.sqrt((path[i+1][0]-path[i][0])**2 + (path[i+1][1]-path[i][1])**2))
        total_dist = dists[-1]
        if total_dist < 1e-6:
            return [path[int(i / (num_points - 1) * (len(path) - 1))] for i in range(num_points)]
        resampled = []
        for i in range(num_points):
            target_d = (i / (num_points - 1)) * total_dist
            for j in range(len(dists) - 1):
                if dists[j] <= target_d <= dists[j+1]:
                    frac = (target_d - dists[j]) / (dists[j+1] - dists[j]) if dists[j+1] > dists[j] else 0.0
                    x = path[j][0] + frac * (path[j+1][0] - path[j][0])
                    y = path[j][1] + frac * (path[j+1][1] - path[j][1])
                    dtheta = (path[j+1][2] - path[j][2] + np.pi) % (2 * np.pi) - np.pi
                    theta = path[j][2] + frac * dtheta
                    resampled.append((x, y, theta))
                    break
        return resampled
    
    def _interpolate_heading(self, p1, p2, frac):
        if p1[2] is not None and p2[2] is not None:
            diff = (p2[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi
            return p1[2] + diff * frac
        elif p1[2] is not None: return p1[2]
        elif p2[2] is not None: return p2[2]
        else: return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])


class TrajectoryCritic:
    """Evaluates trajectory quality and identifies problematic segments."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.tortuosity_threshold = TORTUOSITY_THRESHOLD
        self.yaw_buffer_rad = YAW_BUFFER_RAD
        self.velocity_chattering_threshold = VELOCITY_CHATTERING_THRESHOLD
        self.jerk_cost_threshold = JERK_COST_THRESHOLD
        self.curvature_cost_threshold = CURVATURE_COST_THRESHOLD
        self.centripetal_cost_threshold = CENTRIPETAL_COST_THRESHOLD
    
    def evaluate(self, trajectory: List[Dict[str, Any]], num_samples_per_segment: int) -> List[Tuple[int, int]]:
        bad_windows = []
        num_segments = (len(trajectory) - 1) // num_samples_per_segment
        for i in range(num_segments):
            start_idx = i * num_samples_per_segment
            end_idx = min((i + 2) * num_samples_per_segment, len(trajectory))
            segment = trajectory[start_idx:end_idx]
            if (self._compute_tortuosity(segment) > self.tortuosity_threshold or
                self._compute_yaw_excess(segment) > self.yaw_buffer_rad or
                self._compute_velocity_chattering(segment) > self.velocity_chattering_threshold or
                self._calculate_jerk_cost(segment) > self.jerk_cost_threshold or
                self._calculate_curvature_cost(segment) > self.curvature_cost_threshold or
                self._calculate_centripetal_cost(segment) > self.centripetal_cost_threshold):
                bad_windows.append((i, i + 2))
        return bad_windows
    
    def _compute_tortuosity(self, samples):
        path_length = sum(np.sqrt((samples[i+1]['x']-samples[i]['x'])**2 + (samples[i+1]['y']-samples[i]['y'])**2) for i in range(len(samples)-1))
        straight = np.sqrt((samples[-1]['x']-samples[0]['x'])**2 + (samples[-1]['y']-samples[0]['y'])**2)
        return path_length / straight if straight > 1e-6 else 0.0
    
    def _compute_yaw_excess(self, samples):
        total_yaw = sum(abs(samples[i]['omega']*(samples[i+1]['t']-samples[i]['t'])) for i in range(len(samples)-1))
        expected = abs((samples[-1]['heading']-samples[0]['heading'] + np.pi)%(2*np.pi)-np.pi)
        return max(0, total_yaw - expected)
    
    def _compute_velocity_chattering(self, samples):
        crossings = sum(1 for i in range(1, len(samples)) if samples[i]['vl']*samples[i-1]['vl'] < 0)
        crossings += sum(1 for i in range(1, len(samples)) if samples[i]['vr']*samples[i-1]['vr'] < 0)
        return crossings
    
    def _calculate_jerk_cost(self, samples):
        if len(samples) < 3: return 0.0
        jerk = 0.0
        for k in range(len(samples)-2):
            dt = max(samples[k+1]['t']-samples[k]['t'], 1e-3)
            jerk += ((samples[k+2]['vl']-2*samples[k+1]['vl']+samples[k]['vl'])/dt**2)**2
            jerk += ((samples[k+2]['vr']-2*samples[k+1]['vr']+samples[k]['vr'])/dt**2)**2
        return jerk
    
    def _calculate_curvature_cost(self, samples):
        cost = 0.0
        for k in range(len(samples)-1):
            ds = max(np.sqrt((samples[k+1]['x']-samples[k]['x'])**2 + (samples[k+1]['y']-samples[k]['y'])**2), 1e-6)
            dtheta = (samples[k+1]['heading']-samples[k]['heading']+np.pi)%(2*np.pi)-np.pi
            cost += (abs(dtheta)/ds)**2
        return cost
    
    def _calculate_centripetal_cost(self, samples):
        a_max = self.config.cof * 9.81
        return sum(((abs((s['vl']+s['vr'])/2.0 * s['omega'])/a_max)**2 for s in samples if abs((s['vl']+s['vr'])/2.0 * s['omega']) > 0.8*a_max), 0.0)


class MasterTrajectoryOptimizer:
    """Orchestrates the Multi-Verse refinement pipeline (JAX Enhanced)."""

    def __init__(self, config: RobotConfig, enable_parallel: bool = True, num_workers: int = 8, verbose: bool = True) -> None:
        self.config = config
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers
        self.verbose = verbose
        self.bootstrapper = PathBootstrapper(config)
        self.critic = TrajectoryCritic(config)
        self.iteration_history: List[Dict[str, Any]] = []
        self.stats = OptimizationStats()

        # JAX Initialization
        self.jax_config = JAXRobotConfig(config.__dict__)
        self.jax_refiner = get_jax_refiner(self.jax_config)
        self.prng_key = jax.random.PRNGKey(int(time.time()))

    def solve(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10, accuracy_weight: float = 0.0,
              stop_waypoint_indices: Optional[List[int]] = None, waypoint_events: Optional[Dict[int, str]] = None, apply_headroom: bool = True, verbose: bool = True, capture_iterations: bool = False, live_viz: bool = False) -> List[Dict[str, Any]]:
        self.stats = OptimizationStats()
        total_start_time = time.time()
        
        if stop_waypoint_indices is None: stop_waypoint_indices = []
        if waypoint_events is None: waypoint_events = {}
        
        # Phase 1 & 2: Bootstrap and Global Solve
        phase1_start = time.time()
        if verbose: print("Phase 1: Bootstrapping...")
        guess = self.bootstrapper.generate_baseline(waypoints, num_samples_per_segment)
        self.stats.phase_times["bootstrap"] = time.time() - phase1_start

        phase2_start = time.time()
        if verbose: print("Phase 2: Global optimization (CasADi)...")
        global_traj = self._global_solve(waypoints, num_samples_per_segment, guess, accuracy_weight, stop_waypoint_indices, waypoint_events, apply_headroom, fast_mode=True, capture_iterations=capture_iterations, live_viz=live_viz)
        self.stats.phase_times["global_solve"] = time.time() - phase2_start
        if global_traj: self.stats.initial_cost = (global_traj[1]['t']-global_traj[0]['t']) * (len(global_traj)-1)

        # Phase 3: Critic
        phase3_start = time.time()
        if verbose: print("Phase 3: Evaluating quality...")
        bad_windows = self.critic.evaluate(global_traj, num_samples_per_segment)
        self.stats.phase_times["critic"] = time.time() - phase3_start
        self.stats.bad_segments_found = len(bad_windows)
        
        # Phase 4: JAX Parallel Refinement with Resampling
        phase4_start = time.time()
        if len(bad_windows) > 0:
            if verbose: print(f"Phase 4: Refining {len(bad_windows)} segments using JAX (N={JAX_TARGET_SAMPLES})...")
            
            for window_start, window_end in bad_windows:
                start_idx = window_start * num_samples_per_segment
                end_idx = min(window_end * num_samples_per_segment, len(global_traj) - 1)
                num_window_samples = end_idx - start_idx + 1
                
                # Extract original states for resampling
                orig_states = np.array([[global_traj[idx][k] for k in ['x', 'y', 'heading', 'vl', 'vr']] for idx in range(start_idx, end_idx + 1)])

                # CPU-side Resampling to JAX_TARGET_SAMPLES
                resampled_states = resample_window_cpu(orig_states, JAX_TARGET_SAMPLES)

                s_state = jnp.array(resampled_states[0])
                e_state = jnp.array(resampled_states[-1])

                self.prng_key, subkey = jax.random.split(self.prng_key)
                guesses, initial_costs, biases = generate_candidates_jax(
                    s_state, e_state, JAX_TARGET_SAMPLES, DEFAULT_STOMP_VARIANTS, self.jax_config, subkey
                )

                top_k = 10
                indices = jnp.argsort(initial_costs)[:top_k]

                refined_params, refined_costs, iters = self.jax_refiner(
                    guesses[indices],
                    {'forward_weight': biases['forward_weight'][indices],
                     'reverse_weight': biases['reverse_weight'][indices],
                     'accuracy_weight': jnp.full(top_k, accuracy_weight)},
                    JAX_TARGET_SAMPLES, s_state, e_state, 100
                )

                best_idx = jnp.argmin(refined_costs)
                best_refined = refined_params[best_idx]

                # Resample back to original num_window_samples
                refined_states_fixed = np.array(best_refined[1:].reshape((JAX_TARGET_SAMPLES, 5)))
                refined_states_orig = resample_window_cpu(refined_states_fixed, num_window_samples)

                for j in range(num_window_samples):
                    idx = start_idx + j
                    global_traj[idx].update({
                        'x': float(refined_states_orig[j, 0]), 'y': float(refined_states_orig[j, 1]), 'heading': float(refined_states_orig[j, 2]),
                        'vl': float(refined_states_orig[j, 3]), 'vr': float(refined_states_orig[j, 4]),
                        'omega': float((refined_states_orig[j, 4] - refined_states_orig[j, 3]) / self.config.track_width)
                    })

                self.stats.refinements_attempted += 1
                self.stats.refinements_solved += 1

        self.stats.phase_times["refinement"] = time.time() - phase4_start
        
        # Phase 5: Global Polish
        phase5_start = time.time()
        if verbose: print("Phase 5: Final global polish (CasADi/IPOPT)...")
        final_traj = self._global_solve(waypoints, num_samples_per_segment, guess, accuracy_weight, stop_waypoint_indices, waypoint_events, apply_headroom, fast_mode=False, initial_samples=global_traj, capture_iterations=capture_iterations, live_viz=live_viz)
        self.stats.phase_times["polish"] = time.time() - phase5_start
        
        self.stats.total_time = time.time() - total_start_time
        if final_traj: self.stats.final_cost = (final_traj[1]['t']-final_traj[0]['t']) * (len(final_traj)-1)
        if verbose: print(f"\nOptimization complete in {self.stats.total_time:.2f}s. Final cost: {self.stats.final_cost:.4f}s")
        return final_traj, self.stats.to_dict()
    
    def _global_solve(self, waypoints, num_samples, guess, accuracy, stops, events, headroom, fast_mode, initial_samples=None, capture_iterations=False, live_viz=False):
        from optimizer import TrajectoryOptimizer
        optimizer = TrajectoryOptimizer(self.config)
        if initial_samples:
            dt = initial_samples[1]['t'] - initial_samples[0]['t']
            new_guess = [dt]
            for s in initial_samples: new_guess.extend([s['x'], s['y'], s['heading'], s['vl'], s['vr']])
            guess = np.array(new_guess)
        samples, _ = optimizer.solve(waypoints, num_samples, accuracy, stops, events, headroom, self.verbose, capture_iterations=capture_iterations, live_viz=live_viz)
        return samples
