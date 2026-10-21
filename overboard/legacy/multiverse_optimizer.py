"""
Multi-Verse Segment Refinement Pipeline (JAX/Immrax Optimized)

A hybrid trajectory generation architecture combining:
- Kinematic Profiling (Replaces Initial CasADi Solve)
- Topology exploration (TEB)
- JAX-accelerated parallel refinement and candidate generation
- Gradient-based optimization (CasADi) for final polish
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import casadi as ca
from overboard.models.robot_model import RobotConfig, DifferentialDriveModel, diff_drive_fk, diff_drive_ik, wrap_angle, angle_diff
from overboard.legacy.path_planning import ReedsSheppPath, linear_interpolation_waypoints
import sys
import time

# JAX Imports
try:
    import jax
    import jax.numpy as jnp
    from jax_robot_model import JAXRobotConfig
    from overboard.legacy.jax_optimizer import generate_candidates_jax, get_jax_refiner
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

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
            resampled[:, i] = wrap_angle(np.interp(target_xs, original_xs, unwrapped))
        else:
            resampled[:, i] = np.interp(target_xs, original_xs, states_array[:, i])

    return resampled

class OptimizationStats:
    """Container for detailed optimization telemetry."""
    def __init__(self):
        self.total_time = 0.0
        self.phase_times = {
            "bootstrap": 0.0,
            "kinematic_profile": 0.0,
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
        self.converged = False
        # What the refined-guess probe did (KI-004) and how the polish ran.
        self.refine_probe = None
        self.polish_stats = None

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
            "converged": self.converged,
            "refine_probe": self.refine_probe,
            "polish_stats": self.polish_stats,
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

# Iteration budget for the bounded probe of the refined trajectory (KI-004).
# The probe must stay a fraction of the primary solve, so it gets its own
# small budget instead of the solver default.
REFINE_PROBE_MAX_ITER = 400

# How much faster the refined guess has to be before it replaces the primary
# solve. Below that the difference is solver noise, not a better trajectory.
REFINE_PROBE_MARGIN = 0.001

class PathBootstrapper:
    """Generates kinematically valid initial guesses using Reeds-Shepp paths."""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        mv_cfg = getattr(config, "multiverse_config", {})
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
                    dtheta = angle_diff(path[j+1][2], path[j][2])
                    theta = path[j][2] + frac * dtheta
                    resampled.append((x, y, theta))
                    break
        return resampled
    
    def _interpolate_heading(self, p1, p2, frac):
        if p1[2] is not None and p2[2] is not None:
            diff = angle_diff(p2[2], p1[2])
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
        expected = abs(angle_diff(samples[-1]['heading'], samples[0]['heading']))
        return max(0, total_yaw - expected)
    
    def _compute_velocity_chattering(self, samples):
        crossings = sum(1 for i in range(1, len(samples)) if samples[i]['vl']*samples[i-1]['vl'] < 0)
        crossings += sum(1 for i in range(1, len(samples)) if samples[i]['vr']*samples[i-1]['vr'] < 0)
        return crossings
    
    def _calculate_jerk_cost(self, samples):
        if len(samples) < 3: return 0.0
        jerk = 0.0
        for k in range(len(samples)-2):
            if 'al' in samples[k] and 'al' in samples[k+1]:
                jerk += (samples[k+1]['al'] - samples[k]['al'])**2
                jerk += (samples[k+1]['ar'] - samples[k]['ar'])**2
            else:
                dt = max(samples[k+1]['t']-samples[k]['t'], 1e-3)
                al1 = (samples[k+1]['vl'] - samples[k]['vl']) / dt
                al2 = (samples[k+2]['vl'] - samples[k+1]['vl']) / dt
                ar1 = (samples[k+1]['vr'] - samples[k]['vr']) / dt
                ar2 = (samples[k+2]['vr'] - samples[k+1]['vr']) / dt
                jerk += (al2 - al1)**2 + (ar2 - ar1)**2
        return jerk
    
    def _calculate_curvature_cost(self, samples):
        cost = 0.0
        for k in range(len(samples)-1):
            ds = max(np.sqrt((samples[k+1]['x']-samples[k]['x'])**2 + (samples[k+1]['y']-samples[k]['y'])**2), 1e-6)
            dtheta = angle_diff(samples[k+1]['heading'], samples[k]['heading'])
            cost += (abs(dtheta)/ds)**2
        return cost
    
    def _calculate_centripetal_cost(self, samples):
        a_max = self.config.cof * 9.81
        return sum(((abs(diff_drive_fk(s['vl'], s['vr'], self.config.track_width)[0] * s['omega'])/a_max)**2 for s in samples if abs(diff_drive_fk(s['vl'], s['vr'], self.config.track_width)[0] * s['omega']) > 0.8*a_max), 0.0)


class MasterTrajectoryOptimizer:
    """Orchestrates the Multi-Verse refinement pipeline (JAX Enhanced)."""

    def __init__(self, config: RobotConfig, enable_parallel: bool = True, num_workers: int = 8, verbose: bool = True, seed: int = 0) -> None:
        if not HAS_JAX:
            raise ImportError("JAX is required for Multi-Verse optimization. Please install JAX or run with the --simple flag.")
        self.config = config
        self.enable_parallel = enable_parallel
        self.num_workers = num_workers
        self.verbose = verbose
        self.seed = int(seed)
        self.bootstrapper = PathBootstrapper(config)
        self.critic = TrajectoryCritic(config)
        self.iteration_history: List[Dict[str, Any]] = []
        self.stats = OptimizationStats()

        # JAX Initialization
        self.jax_config = JAXRobotConfig(config.__dict__)
        self.jax_refiner = get_jax_refiner(self.jax_config)
        # The STOMP perturbations are the only random part of the pipeline,
        # and the pipeline has to be reproducible: the same mission must
        # give the same trajectory every time, or a plan cannot be compared
        # with the plan before it. The key used to be seeded from the
        # clock, which made the same mission come out 4.62 s on one run and
        # 5.50 s on the next.
        self.prng_key = jax.random.PRNGKey(self.seed)
        # What the refined-guess probe did in the last solve (KI-004).
        self.refine_probe = None

    def solve(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10, accuracy_weight: float = 0.0,
              stop_waypoint_indices: Optional[List[int]] = None, waypoint_events: Optional[Dict[int, str]] = None, apply_headroom: bool = True, verbose: bool = True, capture_iterations: bool = False, fast_polish: bool = False, **kwargs) -> List[Dict[str, Any]]:
        self.stats = OptimizationStats()
        self.iteration_history = []
        total_start_time = time.time()
        
        if stop_waypoint_indices is None: stop_waypoint_indices = []
        if waypoint_events is None: waypoint_events = {}
        
        # Phase 1: Bootstrap (Geometry Generation)
        phase1_start = time.time()
        if verbose: print("Phase 1: Bootstrapping (Geometry)...")
        guess = self.bootstrapper.generate_baseline(waypoints, num_samples_per_segment)
        self.stats.phase_times["bootstrap"] = time.time() - phase1_start

        # Phase 2: NEW Kinematic Profiler (Replaces Slow CasADi Initial Solve)
        phase2_start = time.time()
        if verbose: print("Phase 2: Kinematic Profiling (Replaces Global CasADi)...")
        global_traj = self._profile_kinematics(guess, num_samples_per_segment, stop_waypoint_indices, waypoint_events, apply_headroom)
        self.stats.phase_times["kinematic_profile"] = time.time() - phase2_start
        if global_traj: self.stats.initial_cost = float(global_traj[-1]['t'])

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
            self.stats.refinements_attempted = self._refine_segments(
                global_traj, num_samples_per_segment, bad_windows,
                accuracy_weight, apply_headroom)
            self.stats.refinements_solved = self.stats.refinements_attempted

        self.stats.phase_times["refinement"] = time.time() - phase4_start
        
        # Phase 5: Global Polish
        phase5_start = time.time()
        if verbose: print("Phase 5: Final global polish (CasADi/IPOPT)...")
        # Warm start CasADi using the fully JAX-refined trajectory!
        final_traj, polish_stats = self._global_solve(waypoints, num_samples_per_segment, None, accuracy_weight, stop_waypoint_indices, waypoint_events, apply_headroom, fast_mode=False, initial_samples=global_traj, capture_iterations=capture_iterations, fast_polish=fast_polish)
        self.stats.phase_times["polish"] = time.time() - phase5_start
        self.stats.converged = bool(polish_stats.get("converged", False))
        self.stats.refine_probe = self.refine_probe
        self.stats.polish_stats = {
            "iterations": polish_stats.get("iter_count"),
            "total_time": polish_stats.get("total_time"),
            "final_cost": polish_stats.get("final_cost"),
            "fast_polish": bool(polish_stats.get("fast_polish", False)),
            "stage_iterations": polish_stats.get("stage_iterations"),
            "stage_times": polish_stats.get("stage_times"),
        }
        
        self.stats.total_time = time.time() - total_start_time
        if final_traj: self.stats.final_cost = float(final_traj[-1]['t'])
        if verbose: print(f"\nOptimization complete in {self.stats.total_time:.2f}s. Final cost: {self.stats.final_cost:.4f}s")
        return final_traj, self.stats.to_dict()

    def _refine_segments(self, global_traj, num_samples_per_segment, bad_windows,
                          accuracy_weight, apply_headroom):
        """Phase 4: refine the flagged segments, one segment at a time.

        The critic reports two-segment windows. A window holds the waypoint
        in its middle, and refining the window as a whole moves that
        waypoint: the refined trajectory no longer passes through it, and
        the waypoint is a hard constraint of the problem. The unit of
        refinement is therefore a **single segment**, whose two endpoints
        are waypoints and are preserved exactly. Nothing between two
        waypoints can be clobbered any more, because there is nothing
        between them.

        Returns the number of segments refined.
        """
        num_segments = (len(global_traj) - 1) // num_samples_per_segment
        flagged = set()
        for window_start, window_end in bad_windows:
            for index in range(window_start, min(window_end, num_segments)):
                flagged.add(index)
        if not flagged:
            return 0

        v_max = float(self.config.max_linear_speed(apply_headroom))
        refined = 0
        for index in sorted(flagged):
            if self._refine_segment(global_traj, index, num_samples_per_segment,
                                     v_max, accuracy_weight):
                refined += 1
        return refined

    def _refine_segment(self, global_traj, index, num_samples_per_segment,
                         v_max, accuracy_weight):
        """Refine one segment and write it back.

        The endpoints are the waypoints at both ends of the segment: they
        are kept exactly, whatever the refinement does in between.
        """
        start_idx = index * num_samples_per_segment
        end_idx = min((index + 1) * num_samples_per_segment, len(global_traj) - 1)
        count = end_idx - start_idx + 1
        if count < 3:
            return False

        # The physics of the segment: how long it is, and therefore how
        # long it should take and how fast it should go. A segment of L
        # metres at 80 % of the robot's top speed takes L / (0.8 * v_max).
        length = 0.0
        for k in range(start_idx, end_idx):
            length += float(np.hypot(global_traj[k + 1]['x'] - global_traj[k]['x'],
                                     global_traj[k + 1]['y'] - global_traj[k]['y']))
        nominal_speed = max(0.8 * v_max, 1e-3)
        dt_window = (length / nominal_speed) / max(JAX_TARGET_SAMPLES - 1, 1)
        dt_window = float(min(max(dt_window, 0.001), 1.0))

        orig_states = np.array([[global_traj[idx][k] for k in ['x', 'y', 'heading', 'vl', 'vr']]
                                for idx in range(start_idx, end_idx + 1)])
        resampled_states = resample_window_cpu(orig_states, JAX_TARGET_SAMPLES)

        s_state = jnp.array(resampled_states[0])
        e_state = jnp.array(resampled_states[-1])

        self.prng_key, subkey = jax.random.split(self.prng_key)
        guesses, initial_costs, biases = generate_candidates_jax(
            s_state, e_state, JAX_TARGET_SAMPLES, DEFAULT_STOMP_VARIANTS,
            self.jax_config, subkey, dt_window, nominal_speed)

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

        # The refined window's own interval, and the segment's interval
        # that it corresponds to: the window holds JAX_TARGET_SAMPLES
        # samples, the segment holds count.
        dt_refined = float(best_refined[0])
        dt_segment = dt_refined * (JAX_TARGET_SAMPLES - 1) / max(count - 1, 1)

        refined_states_fixed = np.array(best_refined[1:].reshape((JAX_TARGET_SAMPLES, 5)))
        refined_states_orig = resample_window_cpu(refined_states_fixed, count)

        # The endpoints are waypoints. Keep them exactly.
        first = {k: global_traj[start_idx][k] for k in ('x', 'y', 'heading', 'vl', 'vr')}
        last = {k: global_traj[end_idx][k] for k in ('x', 'y', 'heading', 'vl', 'vr')}

        for j in range(count):
            idx = start_idx + j
            global_traj[idx].update({
                'x': float(refined_states_orig[j, 0]),
                'y': float(refined_states_orig[j, 1]),
                'heading': float(refined_states_orig[j, 2]),
                'vl': float(refined_states_orig[j, 3]),
                'vr': float(refined_states_orig[j, 4]),
                'omega': float(diff_drive_fk(refined_states_orig[j, 3], refined_states_orig[j, 4], self.config.track_width)[1]),
            })
        global_traj[start_idx].update(first)
        global_traj[end_idx].update(last)

        self._propagate_timing(global_traj, start_idx, end_idx, dt_segment)
        self._refresh_forces(global_traj, start_idx, end_idx)
        return True

    @staticmethod
    def _propagate_timing(global_traj, start_idx, end_idx, dt_segment):
        """Write the refined interval into the timestamps.

        The guess the polish is started from reads its per-segment
        intervals out of the timestamps (``guess_from_samples``), so a
        refinement that changed the timing has to change the timestamps
        too. The segment gets the new interval; everything after it shifts
        by the same amount, so the later segments keep their own.
        """
        count = end_idx - start_idx + 1
        t0 = float(global_traj[start_idx]['t'])
        old_end = float(global_traj[end_idx]['t'])
        new_end = t0 + (count - 1) * dt_segment
        shift = new_end - old_end
        for j in range(count):
            global_traj[start_idx + j]['t'] = t0 + j * dt_segment
        for idx in range(end_idx + 1, len(global_traj)):
            global_traj[idx]['t'] = float(global_traj[idx]['t']) + shift

    def _refresh_forces(self, global_traj, start_idx, end_idx):
        """Recompute the accelerations and forces of a refined segment.

        The refinement changed the wheel speeds, so the accelerations and
        the forces derived from them are stale.
        """
        model = DifferentialDriveModel(self.config)
        for k in range(start_idx, end_idx):
            dt = max(float(global_traj[k + 1]['t']) - float(global_traj[k]['t']), 1e-3)
            al = (global_traj[k + 1]['vl'] - global_traj[k]['vl']) / dt
            ar = (global_traj[k + 1]['vr'] - global_traj[k]['vr']) / dt
            global_traj[k]['al'] = float(al)
            global_traj[k]['ar'] = float(ar)
            fl, fr = model.get_dynamics(global_traj[k]['vl'], global_traj[k]['vr'], al, ar)
            global_traj[k]['fl'] = float(fl)
            global_traj[k]['fr'] = float(fr)

    def _profile_kinematics(self, guess: np.ndarray, num_samples_per_segment: int, stops: List[int], events: Dict[int, str], apply_headroom: bool) -> List[Dict[str, Any]]:
        """Converts geometric Reeds-Shepp arrays into a continuous time-series dictionary format without invoking CasADi."""
        N = (len(guess) - 1) // 5
        states = guess[1:].reshape((N, 5))
        
        # 1. Calculate path distances
        dists = np.zeros(N)
        for i in range(1, N):
            dists[i] = dists[i-1] + np.hypot(states[i,0]-states[i-1,0], states[i,1]-states[i-1,1])
            
        total_dist = dists[-1]
        
        # 2. Determine base dt (using a conservative nominal speed)
        v_max = self.config.max_linear_speed(apply_headroom)
        v_nom = v_max * 0.4  
        total_time = total_dist / v_nom if v_nom > 1e-3 else 1.0
        dt = total_time / max(1, (N - 1))
        dt = max(dt, 0.02)  # Prevent division by zero limits
        
        trajectory = []
        for i in range(N):
            x, y, th = states[i, 0], states[i, 1], states[i, 2]
            
            # Determine velocity using central differences
            if i == 0 or i == N - 1:
                v, w = 0.0, 0.0
            else:
                dx = states[i+1, 0] - states[i-1, 0]
                dy = states[i+1, 1] - states[i-1, 1]
                dth = angle_diff(states[i+1, 2], states[i-1, 2])
                
                v = np.hypot(dx, dy) / (2 * dt)
                w = dth / (2 * dt)
                
                # Reverse motion check
                if dx * np.cos(th) + dy * np.sin(th) < 0:
                    v = -v
            
            # Force stops at requested waypoints
            if i % num_samples_per_segment == 0:
                wp_idx = i // num_samples_per_segment
                if wp_idx in stops:
                    v, w = 0.0, 0.0
                    
            vl, vr = diff_drive_ik(v, w, self.config.track_width)
            
            event = events.get(i // num_samples_per_segment) if i % num_samples_per_segment == 0 else None
            
            trajectory.append({
                "t": float(i * dt),
                "x": float(x),
                "y": float(y),
                "heading": float(th),
                "vl": float(vl),
                "vr": float(vr),
                "omega": float(w),
                "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0
            })
            if event:
                trajectory[-1]["event"] = event
                
        # 3. Calculate Accelerations and Forces
        model = DifferentialDriveModel(self.config)
        for i in range(N - 1):
            al = (trajectory[i+1]['vl'] - trajectory[i]['vl']) / dt
            ar = (trajectory[i+1]['vr'] - trajectory[i]['vr']) / dt
            trajectory[i]['al'] = float(al)
            trajectory[i]['ar'] = float(ar)
            fl, fr = model.get_dynamics(trajectory[i]['vl'], trajectory[i]['vr'], al, ar)
            trajectory[i]['fl'] = float(fl)
            trajectory[i]['fr'] = float(fr)
            
        return trajectory
    
    def _global_solve(self, waypoints, num_samples, guess, accuracy, stops, events, headroom, fast_mode, initial_samples=None, capture_iterations=False, fast_polish=False, probe_refined=True):
        from optimizer import TrajectoryOptimizer
        optimizer = TrajectoryOptimizer(self.config)
        samples, inner_stats = optimizer.solve(waypoints, num_samples, accuracy, stops, events, headroom, self.verbose, capture_iterations=capture_iterations, fast_polish=fast_polish)

        # KI-004: the refined trajectory is a second starting point for the
        # polish, not a dead value. Probe it with a bounded cold pass and keep
        # it when it beats the primary solve. The probe is bounded on purpose:
        # it must never cost more than a fraction of the primary solve.
        if probe_refined and initial_samples and not capture_iterations:
            probe = self._probe_refined_guess(optimizer, waypoints, num_samples, accuracy,
                                              stops, events, headroom, initial_samples)
            if probe is not None:
                probe_samples, probe_stats = probe
                probe_cost = float(probe_stats.get("final_cost", float("inf")))
                primary_cost = float(inner_stats.get("final_cost", float("inf")))
                # Only a real gain replaces the primary solve. A difference in
                # the last digits is solver noise, not a better trajectory.
                used = (probe_stats.get("converged", False)
                        and probe_cost < primary_cost * (1.0 - REFINE_PROBE_MARGIN))
                self.refine_probe = {
                    "converged": bool(probe_stats.get("converged", False)),
                    "cost": probe_cost,
                    "iterations": probe_stats.get("iter_count"),
                    "used": bool(used),
                }
                if used:
                    samples, inner_stats = probe_samples, probe_stats

        # Keep the convergence capture of the solve that was used.
        self.iteration_history = optimizer.iteration_history
        return samples, inner_stats

    def _probe_refined_guess(self, optimizer, waypoints, num_samples, accuracy, stops, events, headroom, initial_samples):
        """Solve once from the refined trajectory with a bounded cold pass.

        The probe reads the **per-segment** timing model, because that is
        what the refined trajectory carries: the refinement gave every
        segment the interval its own length and speed call for, and those
        intervals are in the guess's ``dt`` array. The equal-duration model
        keeps only the first of them and re-reads every sample against one
        interval, so the guess's states - whose wheel speeds and positions
        were computed with the refined intervals - would be read against a
        timing they do not belong to. That is a guess far from the feasible
        manifold, and it is why the probe used to lose (KI-004).

        Returns (samples, stats) or None when the probe cannot run.
        """
        num_segments = len(waypoints) - 1
        try:
            guess = optimizer.guess_from_samples(initial_samples, num_segments, num_samples)
        except (ValueError, KeyError, IndexError):
            return None
        try:
            return optimizer.solve(
                waypoints, num_samples, accuracy, stops, events, headroom, False,
                initial_guess=guess, per_segment_dt=True,
                solver_options={"max_iter": REFINE_PROBE_MAX_ITER, "warm_start_init_point": "no"})
        except Exception:
            return None
