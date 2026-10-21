"""
5-Layer Unified Surrogate-to-Truth Trajectory Pipeline (Parallel Path)
======================================================================

Implements the five-layer optimization & validation architecture:
- Layer 1: CasADi Macro-Solver with Discretization-Aware Metric + Linear Epigraph Force Bounds
- Layer 2: Single-Process ThreadPoolExecutor (Warm JAX + GIL-released C++ IPOPT)
- Layer 3: Compiled jax.lax.scan + vmap Reachability Engine in float64
- Layer 4: Differentiable Immrax Polish (jax.grad through lax.scan)
- Layer 5: Gate-Certified Minimum-Time Candidate Selection
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Layer 3 Hardware Rule: Enforce 64-bit precision in JAX to match IPOPT float64
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad

from immrax import Interval
from immrax_validator import (
    CONTROL_DT,
    PLANT_SUBSTEPS,
    ImmraxValidator,
    unroll_trajectory_headings,
)
from jax_robot_model import JAXRobotConfig
from optimizer import GUESS_STYLES, TrajectoryOptimizer, _GAUSS_LEGENDRE_4
from robot_model import RobotConfig, diff_drive_ik, diff_drive_fk, angle_diff, wrap_angle

from reachability_engine import (
    STATIC_MAX_INTERVALS,
    STATIC_MAX_SEGMENTS,
    MAX_TICKS_PER_INTERVAL,
    GRAD_TICKS_PER_INTERVAL,
    _extract_state_and_dt_arrays,
    _cos_interval,
    _sin_interval,
    _mul_scalar_iv,
    _mul_iv_iv,
    CompiledReachabilityEngine,
    JAX_COMPILE_LOCK,
)


def _gauss_legendre_integrate_jax(
    vl: jnp.ndarray,
    vr: jnp.ndarray,
    theta0: float,
    x0: float,
    y0: float,
    interval_dts: jnp.ndarray,
    track_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiable 4-node Gauss-Legendre integration matching optimizer.py."""
    v1 = 0.5 * (vl[:-1] + vr[:-1])
    v2 = 0.5 * (vl[1:] + vr[1:])
    omega1 = (vr[:-1] - vl[:-1]) / track_width
    omega2 = (vr[1:] - vl[1:]) / track_width

    dtheta = 0.5 * (omega1 + omega2) * interval_dts
    thetas = jnp.concatenate([jnp.array([theta0], dtype=jnp.float64), theta0 + jnp.cumsum(dtheta)])

    theta_k = thetas[:-1]
    d_omega = omega2 - omega1

    dx = jnp.zeros_like(interval_dts)
    dy = jnp.zeros_like(interval_dts)
    for xi, weight in _GAUSS_LEGENDRE_4:
        s = 0.5 * interval_dts * (1.0 + xi)
        w_s = 0.5 * interval_dts * weight
        v_s = v1 + (v2 - v1) * (0.5 * (1.0 + xi))
        theta_s = theta_k + omega1 * s + d_omega * (s * s) / (2.0 * jnp.maximum(interval_dts, 1e-9))
        dx = dx + w_s * v_s * jnp.cos(theta_s)
        dy = dy + w_s * v_s * jnp.sin(theta_s)

    xs = jnp.concatenate([jnp.array([x0], dtype=jnp.float64), x0 + jnp.cumsum(dx)])
    ys = jnp.concatenate([jnp.array([y0], dtype=jnp.float64), y0 + jnp.cumsum(dy)])
    return xs, ys, thetas




# ==============================================================================
# MULTI-FIDELITY PYRAMID: Screen-then-Refine Global Exploration Sieve
# ==============================================================================

def generate_homotopy_bank(
    waypoints: List[Tuple[float, float, Optional[float]]],
    segment_samples: Sequence[int],
    track_width: float,
    v_max: float,
    stop_waypoint_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Stage 1: Generates M = 9-36 constraint-aware geometric and timing hypotheses.
    Prunes the search space using point constraints:
      - Pinned headings (\theta != None) eliminate rotational DOF.
      - Stop waypoints (v = 0) eliminate non-zero boundary speeds.
      - Segment length (L) clamps lateral apex slack (d_perp <= min(0.3*L, 0.25m)).
      - Reverse modes are restricted to stop approaches and short docking segments.
    Returns:
        candidate_states: np.ndarray of shape [M, N, 5]
        candidate_dts: np.ndarray of shape [M, num_segments]
        metadata: List of dicts describing the homotopy parameters of each guess
    """
    num_segments = len(waypoints) - 1
    if num_segments <= 0:
        raise ValueError("At least 2 waypoints required")

    segment_samples = [int(s) for s in segment_samples]
    N = sum(segment_samples) + 1
    wp_indices = [0] + list(np.cumsum(segment_samples))
    stops = set(stop_waypoint_indices or [])

    lengths = [
        float(np.hypot(waypoints[i + 1][0] - waypoints[i][0], waypoints[i + 1][1] - waypoints[i][1]))
        for i in range(num_segments)
    ]
    base_nominal_speed = 0.6 * v_max

    # 1. Lateral apex shift factors: dynamically scaled by segment length
    # If all segments are short (< 15 cm), curvature is impossible -> [0.0]
    has_curvable_segment = any(l >= 0.15 for l in lengths)
    lateral_shift_factors = [-1.0, 0.0, 1.0] if has_curvable_segment else [0.0]

    # 2. Timing regimes (relative to nominal speed)
    speed_factors = [0.80, 1.0, 1.20]

    # 3. Angle perturbations: only active if intermediate waypoints have free headings
    has_unconstrained_heading = any(
        waypoints[i][2] is None for i in range(1, num_segments)
    )
    if has_unconstrained_heading:
        angle_offsets = [-math.radians(20), 0.0, math.radians(20)]
    else:
        angle_offsets = [0.0]

    # 4. Reversal: only plausible on multi-segment missions approaching a stop or short docking
    last_is_stop = (num_segments in stops) or ((len(waypoints) - 1) in stops)
    reverse_allowed = (num_segments > 1) and (last_is_stop or (lengths[-1] < 0.40))
    reverse_modes = [False, True] if reverse_allowed else [False]

    candidates_states_list = []
    candidates_dts_list = []
    metadata = []

    for s_fac in speed_factors:
        v_nom = max(base_nominal_speed * s_fac, 0.1)
        base_dts = np.array([
            float(np.clip((l / v_nom) / max(1, segment_samples[i]), 0.02, 1.0))
            for i, l in enumerate(lengths)
        ], dtype=np.float64)

        for rev_final in reverse_modes:
            for shift_fac in lateral_shift_factors:
                for ang_off in angle_offsets:
                    states = np.zeros((N, 5), dtype=np.float64)

                    for i in range(num_segments):
                        p1 = waypoints[i]
                        p2 = waypoints[i + 1]
                        s_i = segment_samples[i]
                        base_idx = wp_indices[i]
                        count = s_i if i < num_segments - 1 else s_i + 1

                        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                        seg_len = lengths[i]
                        path_angle = math.atan2(dy, dx) if seg_len > 1e-6 else (p1[2] if p1[2] is not None else 0.0)

                        th1 = p1[2] if p1[2] is not None else (path_angle + ang_off)
                        th2 = p2[2] if p2[2] is not None else (path_angle - ang_off)
                        dth = angle_diff(th2, th1)

                        # Clamp lateral apex shift by segment length (max 30% of length or 0.25m)
                        max_lateral = min(0.30 * seg_len, 0.25) if seg_len >= 0.15 else 0.0
                        actual_shift = shift_fac * max_lateral

                        nx, ny = (-dy / seg_len, dx / seg_len) if seg_len > 1e-6 else (0.0, 0.0)
                        is_rev = bool(rev_final and (i == num_segments - 1))

                        for j in range(count):
                            f = j / max(s_i, 1)
                            idx = base_idx + j
                            bump = 4.0 * f * (1.0 - f) * actual_shift

                            states[idx, 0] = p1[0] + dx * f + nx * bump
                            states[idx, 1] = p1[1] + dy * f + ny * bump
                            heading = th1 + dth * f
                            if is_rev:
                                heading = wrap_angle(heading + math.pi)
                            states[idx, 2] = heading

                    # Numerical differencing for wheel speeds
                    for k in range(N - 1):
                        seg_id = 0
                        while seg_id < num_segments - 1 and k >= wp_indices[seg_id + 1]:
                            seg_id += 1
                        dt_k = max(base_dts[seg_id], 1e-4)
                        dx_k = states[k + 1, 0] - states[k, 0]
                        dy_k = states[k + 1, 1] - states[k, 1]
                        dth_k = angle_diff(states[k + 1, 2], states[k, 2])

                        v_lin = math.hypot(dx_k, dy_k) / dt_k
                        fwd_proj = dx_k * math.cos(states[k, 2]) + dy_k * math.sin(states[k, 2])
                        if fwd_proj < 0:
                            v_lin = -v_lin
                        omega = dth_k / dt_k

                        raw_vl, raw_vr = diff_drive_ik(v_lin, omega, track_width)
                        states[k, 3] = float(np.clip(raw_vl, -v_max, v_max))
                        states[k, 4] = float(np.clip(raw_vr, -v_max, v_max))

                    states[-1, 3] = 0.0
                    states[-1, 4] = 0.0
                    states[0, 3:5] = 0.0
                    for stop_idx in stops:
                        if stop_idx < len(wp_indices):
                            states[wp_indices[stop_idx], 3:5] = 0.0

                    candidates_states_list.append(states)
                    candidates_dts_list.append(base_dts.copy())
                    metadata.append({
                        "shift_factor": shift_fac,
                        "speed_factor": s_fac,
                        "angle_offset": ang_off,
                        "reverse_final": rev_final,
                    })

    return (
        np.array(candidates_states_list, dtype=np.float64),
        np.array(candidates_dts_list, dtype=np.float64),
        metadata,
    )


def screen_candidates_jax(
    engine: CompiledReachabilityEngine,
    candidate_states: np.ndarray,  # [M, N, 5]
    candidate_dts: np.ndarray,     # [M, num_segments]
    sample_to_segment: Sequence[int],
    safety_gate_m: float = 0.005,
    tube_gate_m: float = 0.020,
) -> np.ndarray:
    """
    Stage 2: Evaluates all M candidates in parallel in JAX via evaluate_batch_vmap
    and returns their composite proxy score in <25 ms.
    """
    j_states = jnp.asarray(candidate_states, dtype=jnp.float64)
    j_dts = jnp.asarray(candidate_dts, dtype=jnp.float64)

    reports = engine.evaluate_batch_vmap(j_states, j_dts, sample_to_segment)

    times = jnp.sum(j_dts, axis=1)
    track_err = reports["max_tracking_error_m"]
    tube_rad = reports["max_envelope_radius_m"]
    diverged = reports["diverged"]

    proxy_cost = (
        times
        + 120.0 * jnp.maximum(0.0, track_err - float(safety_gate_m))
        + 60.0 * jnp.maximum(0.0, tube_rad - float(tube_gate_m))
        + 1e6 * diverged
    )

    return np.asarray(proxy_cost, dtype=np.float64)


def select_topological_representatives(
    candidate_states: np.ndarray,  # [M, N, 5]
    surrogate_costs: np.ndarray,   # [M]
    k_clusters: int = 3,
    min_separation_m: float = 0.12,
) -> List[int]:
    """
    Stage 3: Selects up to k_clusters structurally distinct candidates with the lowest surrogate cost.
    Deduplicates candidates in the same homotopy basin using spatial signature distance.
    """
    M, N, _ = candidate_states.shape
    if M == 0:
        return []
    if M <= k_clusters:
        return list(range(M))

    key_samples = [max(0, N // 4), max(0, N // 2), max(0, (3 * N) // 4)]
    signatures = candidate_states[:, key_samples, :2].reshape(M, -1)

    valid_mask = np.isfinite(surrogate_costs) & (surrogate_costs < 1e5)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        sorted_indices = list(np.argsort(surrogate_costs))
    else:
        sorted_indices = list(valid_indices[np.argsort(surrogate_costs[valid_indices])])

    selected: List[int] = [int(sorted_indices[0])]

    for idx in sorted_indices[1:]:
        if len(selected) >= k_clusters:
            break

        sig = signatures[idx]
        min_dist = min(float(np.linalg.norm(sig - signatures[s])) for s in selected)
        if min_dist >= min_separation_m:
            selected.append(int(idx))

    if len(selected) < k_clusters:
        for idx in sorted_indices:
            idx_int = int(idx)
            if idx_int not in selected:
                selected.append(idx_int)
                if len(selected) >= k_clusters:
                    break

    return selected


# ==============================================================================
# LAYER 4: Differentiable Immrax Polish (jax.grad through lax.scan)
# ==============================================================================

class SurrogatePolishOptimizer:
    """
    Combines:
      - Layer 1: CasADi Macro-Solver with Discretization-Aware Metric + Linear Epigraph Bounds
      - Layer 2: Single-Process ThreadPoolExecutor for multi-candidate concurrency
      - Layer 3: Compiled jax.lax.scan + vmap Reachability Engine in float64
      - Layer 4: Differentiable Immrax Polish (jax.grad through lax.scan)
      - Layer 5: Gate-Certified Minimum-Time Candidate Selection
    """

    def __init__(
        self,
        config: RobotConfig,
        validator: Optional[ImmraxValidator] = None,
        engine: Optional[CompiledReachabilityEngine] = None,
        safety_gate_m: float = 0.005,
        tube_gate_m: float = 0.020,
    ) -> None:
        self.config = config
        self.casadi_optimizer = TrajectoryOptimizer(config)
        if validator is None:
            jax_cfg = JAXRobotConfig(config)
            validator = ImmraxValidator(jax_cfg)
        self.validator = validator
        self.engine = engine if engine is not None else CompiledReachabilityEngine(validator)
        self.safety_gate_m = float(safety_gate_m)
        self.tube_gate_m = float(tube_gate_m)

    def compute_sample_sensitivities(
        self,
        states: jnp.ndarray,
        dts: jnp.ndarray,
        sample_to_segment: Sequence[int],
    ) -> Tuple[Dict[str, jnp.ndarray], np.ndarray, np.ndarray]:
        """
        Differentiate through the compiled lax.scan reachability rollout to obtain
        exact gradients with respect to every sample state X_k and segment dt_j.
        """
        grad_fn = self.engine._get_grad_fn(self.safety_gate_m, self.tube_gate_m)
        seg_idx_arr = jnp.asarray(sample_to_segment, dtype=jnp.int32)
        active_mask = jnp.ones(states.shape[0] - 1, dtype=bool)
        w_lower, w_upper = self.engine._uncertainty_bounds()
        with JAX_COMPILE_LOCK:
            (_, report), (g_states, g_dts) = grad_fn(states, dts, seg_idx_arr, active_mask, w_lower, w_upper)
        return report, np.asarray(g_states, dtype=np.float64), np.asarray(g_dts, dtype=np.float64)

    def solve(
        self,
        waypoints: List[Tuple[float, float, Optional[float]]],
        num_samples_per_segment: Union[int, Sequence[int]] = 10,
        accuracy_weight: float = 0.0,
        stop_waypoint_indices: Optional[List[int]] = None,
        waypoint_events: Optional[Dict[int, str]] = None,
        apply_headroom: bool = True,
        verbose: bool = True,
        capture_iterations: bool = False,
        guess_style: str = "baseline",
        per_segment_dt: bool = True,
        initial_guess: Optional[np.ndarray] = None,
        fast_polish: bool = False,
        polish_iter_budget: int = 200,
        solver_options: Optional[Dict[str, Any]] = None,
        self_consistent: bool = True,
        proportional_samples: bool = True,
        split_at_boundaries: bool = True,
        force_weight: float = 0.0,
        max_polish_rounds: int = 3,
        build_full_report: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the combined two-scale (Surrogate-to-Truth) trajectory solve:
          Stage 1: CasADi IPOPT with Discretization-Aware Metric + Linear Epigraph Bounds
          Stage 2: Compiled lax.scan Reachability Check + Differentiable Immrax Polish
        """
        t_start = time.time()
        num_segments = len(waypoints) - 1

        # --- STAGE 1: Macro-Solver in CasADi (Discretization Metric + Linear Epigraph) ---
        samples, stats = self.casadi_optimizer.solve(
            waypoints,
            num_samples_per_segment=num_samples_per_segment,
            accuracy_weight=accuracy_weight,
            stop_waypoint_indices=stop_waypoint_indices,
            waypoint_events=waypoint_events,
            apply_headroom=apply_headroom,
            verbose=verbose,
            capture_iterations=capture_iterations,
            guess_style=guess_style,
            per_segment_dt=per_segment_dt,
            initial_guess=initial_guess,
            fast_polish=fast_polish,
            polish_iter_budget=polish_iter_budget,
            solver_options=solver_options,
            self_consistent=self_consistent,
            proportional_samples=proportional_samples,
            split_at_boundaries=split_at_boundaries,
            force_weight=force_weight,
            cost_metric="discretization",
            epigraph_constraints=True,
        )

        if not stats.get("converged", False) or not samples:
            stats["surrogate_polish"] = {"polished": False, "rounds": 0}
            return samples, stats

        segment_samples = stats.get("segment_samples")
        if not segment_samples or len(segment_samples) != num_segments:
            segment_samples = self.casadi_optimizer.get_segment_sample_counts(
                waypoints, num_samples_per_segment, proportional=proportional_samples
            )
        segment_samples = [int(s) for s in segment_samples]
        N = sum(segment_samples) + 1
        sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))

        # --- STAGE 2: Layer 3 (lax.scan Evaluation) + Layer 4 (Differentiable Immrax Polish) ---
        if max_polish_rounds <= 0 and not build_full_report:
            stats["total_time"] = time.time() - t_start
            stats["surrogate_polish"] = {
                "polished": False,
                "rounds": 0,
                "initial_tracking_m": 0.0,
                "final_tracking_m": 0.0,
                "initial_tube_m": 0.0,
                "final_tube_m": 0.0,
            }
            self.casadi_optimizer.iteration_history = list(self.casadi_optimizer.iteration_history)
            return samples, stats

        states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
        with JAX_COMPILE_LOCK:
            report = self.engine.evaluate_states(states, dts, sample_to_segment)

        init_tracking = float(report["max_tracking_error_m"])
        init_tube = float(report["max_envelope_radius_m"])
        best_samples = samples
        best_stats = dict(stats)
        best_tracking = init_tracking
        best_tube = init_tube
        best_report = report

        sample_weights = np.ones(N - 1, dtype=np.float64)
        rounds_run = 0
        cur_samples = samples
        cur_states = states
        cur_dts = dts
        cur_acc_weight = max(accuracy_weight, 0.005)

        for r in range(max_polish_rounds):
            if best_tracking <= self.safety_gate_m - 0.0002 and best_tube <= self.tube_gate_m - 0.0005:
                break

            rounds_run += 1
            _, g_states, g_dts = self.compute_sample_sensitivities(
                cur_states, cur_dts, sample_to_segment
            )
            t_errs = np.asarray(report["tracking_errors"], dtype=np.float64)
            r_errs = np.asarray(report["radii"], dtype=np.float64)

            # Compute per-sample gradient magnitude from jax.grad(immrax_scan)
            grad_norm = np.linalg.norm(g_states[:-1], axis=1) + np.linalg.norm(g_states[1:], axis=1)
            max_grad = float(np.max(grad_norm)) if grad_norm.size > 0 else 0.0
            if max_grad > 1e-12:
                norm_g = grad_norm / max_grad
            else:
                norm_g = np.zeros_like(grad_norm)

            # Boost weights specifically on intervals where tracking error or gradient peaks
            excess_ratio = np.maximum(0.0, t_errs / max(self.safety_gate_m * 0.7, 1e-6) - 0.5)
            sample_weights = sample_weights * (1.0 + 4.0 * norm_g + 6.0 * (excess_ratio ** 2))
            sample_weights = np.clip(sample_weights, 1.0, 200.0)
            cur_acc_weight = min(cur_acc_weight * 3.5, 0.25)

            warm_guess = np.concatenate([
                np.asarray(cur_dts, dtype=np.float64),
                np.asarray(cur_states, dtype=np.float64).flatten(),
            ])

            # Warm-started sensitivity-weighted polish pass in CasADi
            pol_opt = TrajectoryOptimizer(self.config)
            pol_samples, pol_stats = pol_opt.solve(
                waypoints,
                num_samples_per_segment=segment_samples,
                accuracy_weight=cur_acc_weight,
                stop_waypoint_indices=stop_waypoint_indices,
                waypoint_events=waypoint_events,
                apply_headroom=apply_headroom,
                verbose=False,
                capture_iterations=False,
                guess_style=guess_style,
                per_segment_dt=per_segment_dt,
                initial_guess=warm_guess,
                fast_polish=False,
                self_consistent=self_consistent,
                proportional_samples=proportional_samples,
                split_at_boundaries=False,
                force_weight=max(force_weight, 0.002),
                cost_metric="discretization",
                epigraph_constraints=True,
                sample_weights=sample_weights,
            )

            if not pol_stats.get("converged", False) or not pol_samples:
                break

            cur_samples = pol_samples
            cur_states, cur_dts = _extract_state_and_dt_arrays(
                cur_samples, num_segments, segment_samples
            )
            with JAX_COMPILE_LOCK:
                report = self.engine.evaluate_states(cur_states, cur_dts, sample_to_segment)
            cur_tracking = float(report["max_tracking_error_m"])
            cur_tube = float(report["max_envelope_radius_m"])

            if cur_tracking < best_tracking:
                best_tracking = cur_tracking
                best_tube = cur_tube
                best_samples = cur_samples
                best_report = report
                best_stats["final_cost"] = pol_stats.get("final_cost", best_stats.get("final_cost"))
                best_stats["segment_dt"] = pol_stats.get("segment_dt", best_stats.get("segment_dt"))
                best_stats["iter_count"] = int(best_stats.get("iter_count", 0)) + int(
                    pol_stats.get("iter_count", 0)
                )

        if (
            build_full_report
            and not self.engine.active_twin
            and getattr(self.validator, "propagation", "plant") == "plant"
        ):
            with JAX_COMPILE_LOCK:
                best_stats["immrax_report"] = self.engine.build_immrax_report(
                    best_samples, best_report, safety_threshold_m=self.safety_gate_m
                )

        best_stats["total_time"] = time.time() - t_start
        best_stats["surrogate_polish"] = {
            "polished": rounds_run > 0,
            "rounds": rounds_run,
            "initial_tracking_m": init_tracking,
            "final_tracking_m": best_tracking,
            "initial_tube_m": init_tube,
            "final_tube_m": best_tube,
        }
        self.casadi_optimizer.iteration_history = list(self.casadi_optimizer.iteration_history)
        return best_samples, best_stats

    def solve_pyramid(
        self,
        waypoints: List[Tuple[float, float, Optional[float]]],
        num_samples_per_segment: Union[int, Sequence[int]] = 10,
        accuracy_weight: float = 0.0,
        stop_waypoint_indices: Optional[List[int]] = None,
        waypoint_events: Optional[Dict[int, str]] = None,
        apply_headroom: bool = True,
        verbose: bool = True,
        capture_iterations: bool = False,
        fast_polish: bool = False,
        polish_iter_budget: int = 200,
        solver_options: Optional[Dict[str, Any]] = None,
        self_consistent: bool = True,
        proportional_samples: bool = True,
        split_at_boundaries: bool = True,
        force_weight: float = 0.0,
        k_clusters: int = 3,
        min_separation_m: float = 0.12,
        max_polish_rounds: int = 2,
        build_full_report: bool = True,
        max_workers: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the 4-Stage Multi-Fidelity Screen-then-Refine Pyramid:
          Stage 1: Generate M = 100-350 homotopy candidates in NumPy.
          Stage 2: Sieve candidates via compiled float64 evaluate_batch_vmap in <25 ms.
          Stage 3: Topological representative clustering (select Top K distinct seeds).
          Stage 4: Solve survivor candidates in parallel via isolated ProcessPoolExecutor.
          Stage 5: Final differentiable Immrax polish on certified winner.
        """
        t_start = time.time()
        num_segments = len(waypoints) - 1

        segment_samples = self.casadi_optimizer.get_segment_sample_counts(
            waypoints, num_samples_per_segment, proportional=proportional_samples
        )
        segment_samples = [int(s) for s in segment_samples]
        sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))

        # --- STAGE 1: Vectorized Homotopy Bank Generation ---
        cand_states, cand_dts, metadata = generate_homotopy_bank(
            waypoints,
            segment_samples,
            track_width=float(self.config.track_width),
            v_max=float(self.config.max_linear_speed()),
            stop_waypoint_indices=stop_waypoint_indices,
        )

        # --- STAGE 2: Compiled JAX Sieve (evaluate_batch_vmap) ---
        with JAX_COMPILE_LOCK:
            surrogate_costs = screen_candidates_jax(
                self.engine,
                cand_states,
                cand_dts,
                sample_to_segment,
                safety_gate_m=self.safety_gate_m,
                tube_gate_m=self.tube_gate_m,
            )

        # --- STAGE 3: Topological Farthest-Point Clustering ---
        selected_indices = select_topological_representatives(
            cand_states,
            surrogate_costs,
            k_clusters=k_clusters,
            min_separation_m=min_separation_m,
        )

        # --- STAGE 4: Process-Isolated Parallel IPOPT Solves ---
        survivor_guesses = [
            np.concatenate([cand_dts[idx], cand_states[idx].flatten()])
            for idx in selected_indices
        ]
        solve_kw = {
            "num_samples_per_segment": segment_samples,
            "accuracy_weight": accuracy_weight,
            "stop_waypoint_indices": stop_waypoint_indices,
            "waypoint_events": waypoint_events,
            "apply_headroom": apply_headroom,
            "verbose": False,
            "capture_iterations": capture_iterations,
            "fast_polish": fast_polish,
            "polish_iter_budget": polish_iter_budget,
            "solver_options": solver_options,
            "self_consistent": self_consistent,
            "proportional_samples": proportional_samples,
            "split_at_boundaries": split_at_boundaries,
            "force_weight": force_weight,
        }
        solved_pairs = solve_pyramid_candidates(
            self.config,
            waypoints,
            survivor_guesses,
            solve_kw,
            max_workers=max_workers,
        )

        converged_candidates = []
        for i, (cand_samples, cand_stats) in enumerate(solved_pairs):
            if not cand_stats.get("converged", False) or not cand_samples:
                continue
            c_states, c_dts = _extract_state_and_dt_arrays(cand_samples, num_segments, segment_samples)
            with JAX_COMPILE_LOCK:
                c_rep = self.engine.evaluate_states(c_states, c_dts, sample_to_segment)
            track_m = float(c_rep["max_tracking_error_m"])
            tube_m = float(c_rep["max_envelope_radius_m"])
            duration = float(cand_samples[-1]["t"]) if cand_samples else float("inf")
            passed = (track_m <= self.safety_gate_m) and (tube_m <= self.tube_gate_m)
            converged_candidates.append({
                "samples": cand_samples,
                "stats": cand_stats,
                "states": c_states,
                "dts": c_dts,
                "report": c_rep,
                "tracking_m": track_m,
                "tube_m": tube_m,
                "duration": duration,
                "passed": passed,
                "cluster_idx": selected_indices[i],
            })

        if not converged_candidates:
            if solved_pairs:
                first_samples, first_stats = solved_pairs[0]
                first_stats["pyramid"] = {
                    "total_generated": len(cand_states),
                    "screened_clusters": len(selected_indices),
                    "converged_count": 0,
                }
                return first_samples, first_stats
            return [], {"converged": False, "error": "Pyramid solve produced no valid trajectories"}

        passing = [c for c in converged_candidates if c["passed"]]
        if passing:
            winner = min(passing, key=lambda c: c["duration"])
        else:
            winner = min(converged_candidates, key=lambda c: c["tracking_m"])

        best_samples = winner["samples"]
        best_stats = dict(winner["stats"])
        best_tracking = winner["tracking_m"]
        best_tube = winner["tube_m"]
        best_report = winner["report"]
        cur_states = winner["states"]
        cur_dts = winner["dts"]

        # --- STAGE 5: Layer 4 Differentiable Immrax Polish on Winner (if needed) ---
        rounds_run = 0
        if max_polish_rounds > 0 and (
            best_tracking > self.safety_gate_m - 0.0002 or best_tube > self.tube_gate_m - 0.0005
        ):
            cur_acc_weight = max(accuracy_weight, 0.005)
            N = sum(segment_samples) + 1
            sample_weights = np.ones(N - 1, dtype=np.float64)

            for r in range(max_polish_rounds):
                if best_tracking <= self.safety_gate_m - 0.0002 and best_tube <= self.tube_gate_m - 0.0005:
                    break

                rounds_run += 1
                _, g_states, g_dts = self.compute_sample_sensitivities(
                    cur_states, cur_dts, sample_to_segment
                )
                t_errs = np.asarray(best_report["tracking_errors"], dtype=np.float64)
                grad_norm = np.linalg.norm(g_states[:-1], axis=1) + np.linalg.norm(g_states[1:], axis=1)
                max_grad = float(np.max(grad_norm)) if grad_norm.size > 0 else 0.0
                norm_g = grad_norm / max_grad if max_grad > 1e-12 else np.zeros_like(grad_norm)

                excess_ratio = np.maximum(0.0, t_errs / max(self.safety_gate_m * 0.7, 1e-6) - 0.5)
                sample_weights = sample_weights * (1.0 + 4.0 * norm_g + 6.0 * (excess_ratio ** 2))
                sample_weights = np.clip(sample_weights, 1.0, 200.0)
                cur_acc_weight = min(cur_acc_weight * 3.5, 0.25)

                warm_guess = np.concatenate([
                    np.asarray(cur_dts, dtype=np.float64),
                    np.asarray(cur_states, dtype=np.float64).flatten(),
                ])

                pol_opt = TrajectoryOptimizer(self.config)
                pol_samples, pol_stats = pol_opt.solve(
                    waypoints,
                    num_samples_per_segment=segment_samples,
                    accuracy_weight=cur_acc_weight,
                    stop_waypoint_indices=stop_waypoint_indices,
                    waypoint_events=waypoint_events,
                    apply_headroom=apply_headroom,
                    verbose=False,
                    capture_iterations=False,
                    per_segment_dt=True,
                    initial_guess=warm_guess,
                    fast_polish=False,
                    self_consistent=self_consistent,
                    proportional_samples=proportional_samples,
                    split_at_boundaries=False,
                    force_weight=max(force_weight, 0.002),
                    cost_metric="discretization",
                    epigraph_constraints=True,
                    sample_weights=sample_weights,
                )

                if not pol_stats.get("converged", False) or not pol_samples:
                    break

                pol_states, pol_dts = _extract_state_and_dt_arrays(
                    pol_samples, num_segments, segment_samples
                )
                with JAX_COMPILE_LOCK:
                    pol_report = self.engine.evaluate_states(pol_states, pol_dts, sample_to_segment)
                pol_tracking = float(pol_report["max_tracking_error_m"])
                pol_tube = float(pol_report["max_envelope_radius_m"])

                if pol_tracking < best_tracking:
                    best_tracking = pol_tracking
                    best_tube = pol_tube
                    best_samples = pol_samples
                    best_report = pol_report
                    cur_states = pol_states
                    cur_dts = pol_dts
                    best_stats["final_cost"] = pol_stats.get("final_cost", best_stats.get("final_cost"))
                    best_stats["segment_dt"] = pol_stats.get("segment_dt", best_stats.get("segment_dt"))
                    best_stats["iter_count"] = int(best_stats.get("iter_count", 0)) + int(
                        pol_stats.get("iter_count", 0)
                    )

        if (
            build_full_report
            and not self.engine.active_twin
            and getattr(self.validator, "propagation", "plant") == "plant"
        ):
            with JAX_COMPILE_LOCK:
                best_stats["immrax_report"] = self.engine.build_immrax_report(
                    best_samples, best_report, safety_threshold_m=self.safety_gate_m
                )

        best_stats["total_time"] = time.time() - t_start
        best_stats["pyramid"] = {
            "total_generated": len(cand_states),
            "screened_clusters": len(selected_indices),
            "converged_count": len(converged_candidates),
            "selected_cluster": winner["cluster_idx"],
        }
        best_stats["surrogate_polish"] = {
            "polished": rounds_run > 0,
            "rounds": rounds_run,
            "final_tracking_m": best_tracking,
            "final_tube_m": best_tube,
        }
        return best_samples, best_stats


# ==============================================================================
# LAYER 2: Single-Process ThreadPoolExecutor Candidate Solver
# ==============================================================================

def solve_candidates_threaded(
    payloads: Sequence[Tuple[Any, ...]],
    workers: int = 4,
    validator: Optional[ImmraxValidator] = None,
    engine: Optional[CompiledReachabilityEngine] = None,
) -> Tuple[List[Tuple[List[Dict[str, Any]], Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """
    Layer 2 Concurrency:
    Solves candidates concurrently using a single-process ThreadPoolExecutor.
    Shares a single warm JAX/Immrax validator instance across threads so XLA
    kernels are compiled once in-process with zero multiprocessing spawn overhead.
    """
    if not payloads:
        return [], []

    shared_validator = validator
    if shared_validator is None:
        first_cfg = payloads[0][0]
        shared_validator = ImmraxValidator(JAXRobotConfig(first_cfg))
    shared_engine = engine if engine is not None else CompiledReachabilityEngine(shared_validator)

    def _run_one(payload):
        robot_cfg, wps, spec, solve_kwargs, _ = payload
        try:
            opt = SurrogatePolishOptimizer(
                robot_cfg, validator=shared_validator, engine=shared_engine
            )
            kw = dict(solve_kwargs)
            kw.pop("cost_metric", None)
            kw.pop("epigraph_constraints", None)
            samples, stats = opt.solve(wps, **kw)
            return samples, stats, list(getattr(opt.casadi_optimizer, "iteration_history", []))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return [], {"converged": False, "final_cost": float("inf"), "total_time": 0.0, "error": str(exc)}, []

    max_workers = max(1, min(len(payloads), int(workers)))
    if max_workers == 1:
        solved = [_run_one(p) for p in payloads]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            solved = list(pool.map(_run_one, payloads))

    return [pair[:2] for pair in solved], [pair[2] for pair in solved]


# ==============================================================================
# PYRAMID PARALLEL WORKER SOLVERS (ProcessPoolExecutor)
# ==============================================================================

def _worker_pyramid_solve(args: Tuple[Any, ...]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Worker solve function for ProcessPoolExecutor.
    Runs in an isolated process with its own memory space and MUMPS instance,
    completely eliminating Windows CASADI_SOLVE_LOCK contention.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    robot_cfg, waypoints, solve_kwargs = args
    opt = TrajectoryOptimizer(robot_cfg)
    try:
        samples, stats = opt.solve(waypoints, **solve_kwargs)
        return samples, stats
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return [], {"converged": False, "final_cost": float("inf"), "total_time": 0.0, "error": str(exc)}


def solve_pyramid_candidates(
    robot_cfg: RobotConfig,
    waypoints: List[Tuple[float, float, Optional[float]]],
    survivor_guesses: List[np.ndarray],
    solve_kwargs: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    max_workers: Optional[int] = None,
    pool: Optional[ProcessPoolExecutor] = None,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Stage 4: Solves the screened survivor finalists simultaneously across separate processes.
    Zero MUMPS lock contention. Supports persistent worker pools across evolutionary generations
    and per-candidate adaptive solve_kwargs.
    """
    if not survivor_guesses:
        return []

    payloads = []
    for i, guess in enumerate(survivor_guesses):
        if isinstance(solve_kwargs, (list, tuple)):
            kw = dict(solve_kwargs[min(i, len(solve_kwargs) - 1)])
        else:
            kw = dict(solve_kwargs)
        kw["initial_guess"] = guess
        kw["per_segment_dt"] = True
        kw["cost_metric"] = "discretization"
        kw["epigraph_constraints"] = True
        payloads.append((robot_cfg, waypoints, kw))

    if pool is not None:
        try:
            return list(pool.map(_worker_pyramid_solve, payloads))
        except Exception:
            return [_worker_pyramid_solve(p) for p in payloads]

    num_workers = max(1, min(len(payloads), max_workers or len(payloads)))
    if num_workers == 1:
        return [_worker_pyramid_solve(p) for p in payloads]

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as p:
            return list(p.map(_worker_pyramid_solve, payloads))
    except Exception:
        # Fallback to sequential solve if ProcessPoolExecutor cannot spawn in current context
        return [_worker_pyramid_solve(p) for p in payloads]

