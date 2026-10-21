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
import threading
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

# Static upper bound on 20 ms control ticks per sample interval for lax.scan.
# 28 ticks * 0.02 s = 0.56 s per sample interval (covers any trajectory sample).
MAX_TICKS_PER_INTERVAL = 28
GRAD_TICKS_PER_INTERVAL = 6
STATIC_MAX_INTERVALS = 64
STATIC_MAX_SEGMENTS = 16

# MUMPS inside CasADi/IPOPT is not re-entrant across OS threads on Windows;
# serialize the C++ MUMPS call while keeping all candidates in a single warm
# ThreadPoolExecutor process (zero multiprocessing spawn or IPC overhead).
from optimizer import CASADI_SOLVE_LOCK as _CASADI_SOLVE_LOCK
_JAX_COMPILE_LOCK = threading.RLock()


def _extract_state_and_dt_arrays(
    samples: List[Dict[str, Any]],
    num_segments: int,
    segment_samples: Sequence[int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert sample dicts into (states [N, 5], dts [num_segments]) JAX float64 arrays."""
    unrolled = unroll_trajectory_headings(samples)
    states = jnp.array(
        [[s["x"], s["y"], s["heading"], s["vl"], s["vr"]] for s in unrolled],
        dtype=jnp.float64,
    )
    times = np.array([float(s["t"]) for s in unrolled], dtype=np.float64)
    wp_indices = [0] + list(np.cumsum(segment_samples))
    dts = np.zeros(num_segments, dtype=np.float64)
    for j in range(num_segments):
        k = min(wp_indices[j], len(unrolled) - 2)
        dts[j] = max(times[k + 1] - times[k], 1e-4)
    return states, jnp.asarray(dts, dtype=jnp.float64)


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


def _cos_interval(a_min: jnp.ndarray, a_max: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact interval bounds for cos([a_min, a_max]) in pure JAX without Immrax overhead."""
    c1 = jnp.cos(a_min)
    c2 = jnp.cos(a_max)
    c_min = jnp.minimum(c1, c2)
    c_max = jnp.maximum(c1, c2)
    # Check if a multiple of 2*pi (where cos == 1) lies inside [a_min, a_max]
    k_max = jnp.ceil(a_min / (2.0 * math.pi))
    has_peak = (k_max * (2.0 * math.pi)) <= a_max
    c_max = jnp.where(has_peak, 1.0, c_max)
    # Check if pi + 2*k*pi (where cos == -1) lies inside [a_min, a_max]
    k_min = jnp.ceil((a_min - math.pi) / (2.0 * math.pi))
    has_trough = (math.pi + k_min * (2.0 * math.pi)) <= a_max
    c_min = jnp.where(has_trough, -1.0, c_min)
    return c_min, c_max


def _sin_interval(a_min: jnp.ndarray, a_max: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact interval bounds for sin([a_min, a_max]) in pure JAX."""
    return _cos_interval(a_min - 0.5 * math.pi, a_max - 0.5 * math.pi)


def _mul_scalar_iv(s: jnp.ndarray, iv_min: jnp.ndarray, iv_max: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    p1 = s * iv_min
    p2 = s * iv_max
    return jnp.minimum(p1, p2), jnp.maximum(p1, p2)


def _mul_iv_iv(
    a_min: jnp.ndarray, a_max: jnp.ndarray, b_min: jnp.ndarray, b_max: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    p1 = a_min * b_min
    p2 = a_min * b_max
    p3 = a_max * b_min
    p4 = a_max * b_max
    lo = jnp.minimum(jnp.minimum(p1, p2), jnp.minimum(p3, p4))
    hi = jnp.maximum(jnp.maximum(p1, p2), jnp.maximum(p3, p4))
    return lo, hi


# ==============================================================================
# LAYER 3: Compiled jax.lax.scan + vmap Reachability Engine (float64)
# ==============================================================================

class CompiledReachabilityEngine:
    """
    Pure-JAX compiled reachability engine using jax.lax.scan.

    Eliminates the Python for/while loops and .float() host synchronizations
    in ImmraxValidator.compute_reachability so that:
      1. Rollout executes as a single fused XLA kernel (10x-50x faster).
      2. Multiple candidates can be evaluated in parallel via jax.vmap.
      3. Reverse-mode autodiff (jax.grad) flows cleanly through the entire
         20 ms sample-and-hold closed-loop simulation.
    """

    def __init__(self, validator: ImmraxValidator):
        self.validator = validator
        self.cfg = validator.cfg
        self.control_dt = float(validator.control_dt)
        self.ramsete_b = float(getattr(validator, "ramsete_b", 2.0))
        self.track_width = float(validator.cfg.track_width)
        self.active_twin = bool(getattr(validator, "_digital_twin_active", False))
        self.state_len = int(validator.plant.xlen if self.active_twin else 3)
        self._compiled_scan_fn: Any = None
        self._compiled_grad_cache: Dict[Tuple[float, float], Any] = {}
        self._compiled_audit_fn: Any = None

    def _get_scan_fn(self):
        if self._compiled_scan_fn is not None:
            return self._compiled_scan_fn

        with _JAX_COMPILE_LOCK:
            if self._compiled_scan_fn is not None:
                return self._compiled_scan_fn

            state_len = self.state_len
            control_dt = self.control_dt
            ramsete_b = self.ramsete_b
            ramsete_zeta = float(getattr(self.validator, "ramsete_zeta", 0.7))
            nom_track_width = self.track_width
            plant_emb_f = self.validator.plant_emb_sys.f
            controller_fn = self.validator.controller
            odometry_pose_fn = self.validator._odometry_pose
            tick_indices = jnp.arange(MAX_TICKS_PER_INTERVAL, dtype=jnp.float64)
            use_fast_kinematic = (not self.active_twin) and (
                getattr(self.validator, "plant_inclusion", "nat") == "nat"
            )
            from jax_ramsete import _sinc_poly

            def _make_rollout_scan(num_ticks: int, num_substeps: int):
                tick_idx_arr = jnp.arange(num_ticks, dtype=jnp.float64)

                def _rollout_scan(
                    states: jnp.ndarray,
                    dts: jnp.ndarray,
                    seg_idx_arr: jnp.ndarray,
                    active_mask: jnp.ndarray,
                    w_lower: jnp.ndarray,
                    w_upper: jnp.ndarray,
                ) -> Dict[str, jnp.ndarray]:
                    """
                    Differentiable lax.scan rollout over all sample intervals.
                    states: (M+1, 5) [x, y, heading, vl, vr]
                    dts: (S,)
                    seg_idx_arr: (M,) int32 segment index per interval
                    active_mask: (M,) bool mask for non-padded intervals
                    """
                    w_interval = Interval(w_lower, w_upper)
                    backlash_half = 0.5 * w_upper[2]
                    tw_min, tw_max = w_lower[3], w_upper[3]
                    g_min, g_max = w_lower[6], w_upper[6]
                    inv_tw_min = 1.0 / tw_max
                    inv_tw_max = 1.0 / tw_min

                    start_pose = states[0, :3]
                    lower_state = jnp.concatenate([
                        start_pose - 0.001,
                        jnp.zeros(state_len - 3, dtype=jnp.float64),
                    ])
                    upper_state = jnp.concatenate([
                        start_pose + 0.001,
                        jnp.zeros(state_len - 3, dtype=jnp.float64),
                    ])
                    x_flat_init = jnp.concatenate([lower_state, upper_state])

                    raw_interval_dts = dts[seg_idx_arr]
                    interval_dts = jnp.where(active_mask, raw_interval_dts, 0.0)
                    elapsed_times = jnp.cumsum(interval_dts)

                    s0_all = states[:-1]
                    s1_all = states[1:]

                    def _interval_step(carry, step_inputs):
                        x_flat = carry
                        s0, s1, dt_k, elapsed_t, is_active = step_inputs

                        v0 = 0.5 * (s0[3] + s0[4])
                        v1 = 0.5 * (s1[3] + s1[4])
                        dh = (s1[2] - s0[2] + math.pi) % (2.0 * math.pi) - math.pi
                        dt_safe = jnp.maximum(dt_k, 1e-9)
                        ref_v = 0.5 * (v0 + v1)
                        ref_omega = dh / dt_safe
                        ref_kr = jnp.sqrt(ref_omega * ref_omega + ramsete_b * ref_v * ref_v + 1e-16)

                        def _tick_step(x_curr, m_idx):
                            t_rel = m_idx * control_dt
                            dt_ctrl = jnp.clip(dt_k - t_rel, 0.0, control_dt)
                            active = is_active & (dt_ctrl > 1e-12)
                            frac = jnp.where(active, jnp.minimum(t_rel, dt_safe) / dt_safe, 0.0)

                            ref_x = s0[0] + (s1[0] - s0[0]) * frac
                            ref_y = s0[1] + (s1[1] - s0[1]) * frac
                            ref_h = s0[2] + dh * frac

                            sub_dt = dt_ctrl / float(num_substeps)
                            if use_fast_kinematic:
                                cx = 0.5 * (x_curr[0] + x_curr[3])
                                cy = 0.5 * (x_curr[1] + x_curr[4])
                                cth = 0.5 * (x_curr[2] + x_curr[5])
                                ex_g = ref_x - cx
                                ey_g = ref_y - cy
                                cos_cth = jnp.cos(cth)
                                sin_cth = jnp.sin(cth)
                                ex = cos_cth * ex_g + sin_cth * ey_g
                                ey = -sin_cth * ex_g + cos_cth * ey_g
                                etheta = (ref_h - cth + math.pi) % (2.0 * math.pi) - math.pi
                                k1 = 2.0 * ramsete_zeta * ref_kr
                                v_cmd = ref_v * jnp.cos(etheta) + k1 * ex
                                omega_cmd = ref_omega + ramsete_b * ref_v * _sinc_poly(etheta) * ey + k1 * etheta
                                dv_cmd = omega_cmd * nom_track_width

                                v_act_lo, v_act_hi = _mul_scalar_iv(v_cmd, g_min, g_max)
                                dg_lo, dg_hi = _mul_scalar_iv(dv_cmd, g_min, g_max)
                                omega_lo, omega_hi = _mul_iv_iv(dg_lo, dg_hi, inv_tw_min, inv_tw_max)

                                x_sub = x_curr
                                for _ in range(num_substeps):
                                    th_lo = x_sub[2]
                                    th_hi = x_sub[5]
                                    cos_lo, cos_hi = _cos_interval(th_lo, th_hi)
                                    sin_lo, sin_hi = _sin_interval(th_lo, th_hi)
                                    dx_lo, dx_hi = _mul_iv_iv(v_act_lo, v_act_hi, cos_lo, cos_hi)
                                    dy_lo, dy_hi = _mul_iv_iv(v_act_lo, v_act_hi, sin_lo, sin_hi)
                                    step_vec = jnp.array(
                                        [dx_lo, dy_lo, omega_lo, dx_hi, dy_hi, omega_hi],
                                        dtype=jnp.float64,
                                    )
                                    x_sub = x_sub + step_vec * sub_dt
                                return jnp.where(active, x_sub, x_curr), None
                            else:
                                u_ref = jnp.array([ref_x, ref_y, ref_h, ref_v, ref_omega, ref_kr], dtype=jnp.float64)
                                x_center = 0.5 * (x_curr[:3] + x_curr[state_len : state_len + 3])
                                x_center = odometry_pose_fn(x_center, start_pose)
                                u_cmd = controller_fn(
                                    0.0, x_center, u_ref, w_interval, normalize=True, quantize=False
                                )
                                x_sub = x_curr
                                for _ in range(num_substeps):
                                    dx_dt = plant_emb_f(0.0, x_sub, u_cmd, w_interval)
                                    x_sub = x_sub + dx_dt * sub_dt
                                return jnp.where(active, x_sub, x_curr), None

                        x_flat_end, _ = jax.lax.scan(_tick_step, x_flat, tick_idx_arr)
                        x_flat_end = jnp.where(is_active, x_flat_end, x_flat)

                        spread = jnp.maximum(0.0, backlash_half * jnp.maximum(0.0, elapsed_t))
                        x_min = x_flat_end[0] - spread
                        x_max = x_flat_end[state_len] + spread
                        y_min = x_flat_end[1] - spread
                        y_max = x_flat_end[state_len + 1] + spread

                        mid_x = 0.5 * (x_flat_end[0] + x_flat_end[state_len])
                        mid_y = 0.5 * (x_flat_end[1] + x_flat_end[state_len + 1])
                        raw_track_err = jnp.sqrt((mid_x - s1[0]) ** 2 + (mid_y - s1[1]) ** 2 + 1e-16)
                        tracking_err = jnp.where(is_active, raw_track_err, 0.0)

                        mid_h = 0.5 * (x_flat_end[2] + x_flat_end[state_len + 2])
                        raw_head_err = jnp.abs((mid_h - s1[2] + math.pi) % (2.0 * math.pi) - math.pi)
                        heading_err = jnp.where(is_active, raw_head_err, 0.0)

                        raw_radius = jnp.sqrt(((x_max - x_min) * 0.5) ** 2 + ((y_max - y_min) * 0.5) ** 2 + 1e-16)
                        radius = jnp.where(is_active, raw_radius, 0.0)
                        box_vec = jnp.array(
                            [x_min, x_max, y_min, y_max, x_flat_end[2], x_flat_end[state_len + 2]],
                            dtype=jnp.float64,
                        )

                        return x_flat_end, (tracking_err, heading_err, radius, box_vec)

                    _, (tracking_errs, heading_errs, radii, boxes) = jax.lax.scan(
                        _interval_step,
                        x_flat_init,
                        (s0_all, s1_all, interval_dts, elapsed_times, active_mask),
                    )

                    init_box = jnp.array(
                        [
                            x_flat_init[0],
                            x_flat_init[state_len],
                            x_flat_init[1],
                            x_flat_init[state_len + 1],
                            x_flat_init[2],
                            x_flat_init[state_len + 2],
                        ],
                        dtype=jnp.float64,
                    )
                    all_boxes = jnp.vstack([init_box[None, :], boxes])

                    return {
                        "tracking_errors": tracking_errs,
                        "heading_errors": heading_errs,
                        "radii": radii,
                        "boxes": all_boxes,
                        "max_tracking_error_m": jnp.max(tracking_errs),
                        "max_heading_error_rad": jnp.max(heading_errs),
                        "max_envelope_radius_m": jnp.max(radii),
                        "max_error_m": jnp.max(tracking_errs + radii),
                        "diverged": jnp.any(radii > 0.05),
                    }

                return _rollout_scan

            exact_scan = _make_rollout_scan(MAX_TICKS_PER_INTERVAL, PLANT_SUBSTEPS)
            grad_scan = _make_rollout_scan(GRAD_TICKS_PER_INTERVAL, 1)
            self._compiled_scan_fn = jit(exact_scan)
            self._uncompiled_scan_fn = exact_scan
            self._uncompiled_grad_scan_fn = grad_scan
            return self._compiled_scan_fn

    def _pad_inputs(
        self,
        states: jnp.ndarray,
        dts: jnp.ndarray,
        sample_to_segment: Sequence[int],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int, bool]:
        """Pad inputs to static shape (STATIC_MAX_INTERVALS, STATIC_MAX_SEGMENTS) so XLA compiles once."""
        n_states = int(states.shape[0])
        n_intervals = n_states - 1
        n_segs = int(dts.shape[0])
        seg_idx_arr = jnp.asarray(sample_to_segment, dtype=jnp.int32)

        if n_intervals <= STATIC_MAX_INTERVALS and n_segs <= STATIC_MAX_SEGMENTS:
            pad_intervals = STATIC_MAX_INTERVALS - n_intervals
            pad_segs = STATIC_MAX_SEGMENTS - n_segs
            if pad_intervals > 0:
                last_state = jnp.repeat(states[-1:, :], pad_intervals, axis=0)
                p_states = jnp.concatenate([states, last_state], axis=0)
                p_seg_idx = jnp.concatenate(
                    [seg_idx_arr, jnp.zeros(pad_intervals, dtype=jnp.int32)], axis=0
                )
            else:
                p_states = states
                p_seg_idx = seg_idx_arr
            if pad_segs > 0:
                p_dts = jnp.concatenate([dts, jnp.zeros(pad_segs, dtype=jnp.float64)], axis=0)
            else:
                p_dts = dts
            active_mask = jnp.arange(STATIC_MAX_INTERVALS) < n_intervals
            return p_states, p_dts, p_seg_idx, active_mask, n_intervals, n_segs, True

        active_mask = jnp.ones(n_intervals, dtype=bool)
        return states, dts, seg_idx_arr, active_mask, n_intervals, n_segs, False

    def _get_grad_fn(
        self,
        safety_gate_m: float,
        tube_gate_m: float,
    ):
        key = (float(safety_gate_m), float(tube_gate_m))
        if key in self._compiled_grad_cache:
            return self._compiled_grad_cache[key]

        self._get_scan_fn()
        raw_grad_scan_fn = self._uncompiled_grad_scan_fn
        with _JAX_COMPILE_LOCK:
            if key in self._compiled_grad_cache:
                return self._compiled_grad_cache[key]

            target_track = float(safety_gate_m) - 0.0004
            target_tube = float(tube_gate_m) - 0.0010

            def _loss_fn(s_arr, d_arr, seg_idx_arr, active_mask, w_lower, w_upper):
                rep = raw_grad_scan_fn(s_arr, d_arr, seg_idx_arr, active_mask, w_lower, w_upper)
                t_errs = rep["tracking_errors"]
                r_errs = rep["radii"]
                track_hinge = jnp.where(active_mask, jnp.maximum(0.0, t_errs - target_track), 0.0)
                tube_hinge = jnp.where(active_mask, jnp.maximum(0.0, r_errs - target_tube), 0.0)
                loss = (
                    1e6 * jnp.sum(track_hinge ** 2)
                    + 1e5 * jnp.sum(tube_hinge ** 2)
                )
                return loss, rep

            grad_fn = jit(value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True))
            self._compiled_grad_cache[key] = grad_fn
            return grad_fn

    def _uncertainty_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.cfg
        gain_range = getattr(self.validator, "digital_twin_gain_range", (1.0, 1.0))
        w_lower = jnp.array([
            cfg.cof_range[0],
            cfg.torque_headroom_range[0],
            cfg.backlash_range[0],
            cfg.track_width_range[0],
            cfg.wheel_radius_range[0],
            cfg.inertia_range[0],
            gain_range[0],
        ], dtype=jnp.float64)
        w_upper = jnp.array([
            cfg.cof_range[1],
            cfg.torque_headroom_range[1],
            cfg.backlash_range[1],
            cfg.track_width_range[1],
            cfg.wheel_radius_range[1],
            cfg.inertia_range[1],
            gain_range[1],
        ], dtype=jnp.float64)
        return w_lower, w_upper

    def evaluate_states(
        self,
        states: jnp.ndarray,
        dts: jnp.ndarray,
        sample_to_segment: Sequence[int],
    ) -> Dict[str, jnp.ndarray]:
        scan_fn = self._get_scan_fn()
        seg_idx_arr = jnp.asarray(sample_to_segment, dtype=jnp.int32)
        active_mask = jnp.ones(states.shape[0] - 1, dtype=bool)
        w_lower, w_upper = self._uncertainty_bounds()
        return scan_fn(states, dts, seg_idx_arr, active_mask, w_lower, w_upper)

    def build_immrax_report(
        self,
        samples: List[Dict[str, Any]],
        scan_out: Dict[str, jnp.ndarray],
        safety_threshold_m: float = 0.005,
    ) -> Dict[str, Any]:
        """
        Convert compiled lax.scan reachability output + vectorized force/slip audit
        into the exact dict structure returned by ImmraxValidator.validate_trajectory.
        """
        from robot_model import CG_HEIGHT_M

        cfg = self.cfg
        vl = np.array([float(s["vl"]) for s in samples], dtype=np.float64)
        vr = np.array([float(s["vr"]) for s in samples], dtype=np.float64)
        al = np.array([float(s["al"]) for s in samples], dtype=np.float64)
        ar = np.array([float(s["ar"]) for s in samples], dtype=np.float64)

        tw_lo, tw_hi = float(cfg.track_width_range[0]), float(cfg.track_width_range[1])
        wr_lo, wr_hi = float(cfg.wheel_radius_range[0]), float(cfg.wheel_radius_range[1])
        ine_lo, ine_hi = float(cfg.inertia_range[0]), float(cfg.inertia_range[1])
        cof_lo = float(cfg.cof_range[0])
        tm_lo = float(cfg.torque_headroom_range[0])
        mass = float(cfg.mass)
        g = float(cfg.g)
        gearing = float(cfg.gearing)
        v_max_rad_s = float(cfg.v_max_rad_s)
        t_max_nm = float(cfg.t_max_nm)

        a_val = 0.5 * (al + ar)
        f_total = mass * a_val
        da = ar - al

        # Interval alpha = da / [tw_lo, tw_hi]
        alpha_1 = da / tw_lo
        alpha_2 = da / tw_hi
        alpha_lo = np.minimum(alpha_1, alpha_2)
        alpha_hi = np.maximum(alpha_1, alpha_2)

        # Interval m_total = [ine_lo, ine_hi] * [alpha_lo, alpha_hi]
        m_1 = ine_lo * alpha_lo
        m_2 = ine_lo * alpha_hi
        m_3 = ine_hi * alpha_lo
        m_4 = ine_hi * alpha_hi
        m_lo = np.minimum(np.minimum(m_1, m_2), np.minimum(m_3, m_4))
        m_hi = np.maximum(np.maximum(m_1, m_2), np.maximum(m_3, m_4))

        # Interval 2*m_total / [tw_lo, tw_hi]
        mtw_1 = (2.0 * m_lo) / tw_lo
        mtw_2 = (2.0 * m_lo) / tw_hi
        mtw_3 = (2.0 * m_hi) / tw_lo
        mtw_4 = (2.0 * m_hi) / tw_hi
        mtw_lo = np.minimum(np.minimum(mtw_1, mtw_2), np.minimum(mtw_3, mtw_4))
        mtw_hi = np.maximum(np.maximum(mtw_1, mtw_2), np.maximum(mtw_3, mtw_4))

        fr_lo = 0.5 * (f_total + mtw_lo)
        fr_hi = 0.5 * (f_total + mtw_hi)
        fl_lo = f_total - fr_hi
        fl_hi = f_total - fr_lo

        iv_abs_l = np.maximum(np.abs(fl_lo), np.abs(fl_hi))
        iv_abs_r = np.maximum(np.abs(fr_lo), np.abs(fr_hi))

        # Worst-case lower bound on motor force limit occurs at wr_lo for omega_upper and wr_hi for division
        omega_l_hi = (np.abs(vl) / wr_lo) * gearing
        torque_l_lo = np.maximum(0.0, (1.0 - omega_l_hi / v_max_rad_s) * t_max_nm)
        force_limit_l_lo = (torque_l_lo / wr_hi) * gearing * tm_lo

        omega_r_hi = (np.abs(vr) / wr_lo) * gearing
        torque_r_lo = np.maximum(0.0, (1.0 - omega_r_hi / v_max_rad_s) * t_max_nm)
        force_limit_r_lo = (torque_r_lo / wr_hi) * gearing * tm_lo

        motor_vios_l = np.maximum(0.0, iv_abs_l - force_limit_l_lo)
        motor_vios_r = np.maximum(0.0, iv_abs_r - force_limit_r_lo)
        max_motor_vio = float(np.maximum(np.max(motor_vios_l), np.max(motor_vios_r)))

        # Lateral normal force transfer worst-case lower bound
        v_val = 0.5 * (vl + vr)
        dv_val = vr - vl
        ca_1 = v_val * (dv_val / tw_lo)
        ca_2 = v_val * (dv_val / tw_hi)
        ca_lo = np.minimum(ca_1, ca_2)
        ca_hi = np.maximum(ca_1, ca_2)
        lat_1 = (mass * ca_lo * CG_HEIGHT_M) / tw_lo
        lat_2 = (mass * ca_lo * CG_HEIGHT_M) / tw_hi
        lat_3 = (mass * ca_hi * CG_HEIGHT_M) / tw_lo
        lat_4 = (mass * ca_hi * CG_HEIGHT_M) / tw_hi
        lat_lo = np.minimum(np.minimum(lat_1, lat_2), np.minimum(lat_3, lat_4))
        lat_hi = np.maximum(np.maximum(lat_1, lat_2), np.maximum(lat_3, lat_4))

        half_weight = 0.5 * mass * g
        nl_lo = np.maximum(0.0, half_weight - lat_hi)
        nr_lo = np.maximum(0.0, half_weight + lat_lo)

        traction_max = cof_lo * mass * g
        traction_vios = np.maximum(0.0, iv_abs_l + iv_abs_r - traction_max)
        slip_ls = np.maximum(0.0, iv_abs_l - cof_lo * nl_lo)
        slip_rs = np.maximum(0.0, iv_abs_r - cof_lo * nr_lo)

        max_traction_vio = float(np.max(traction_vios))
        max_slip_l = float(np.max(slip_ls))
        max_slip_r = float(np.max(slip_rs))

        max_wheel_speed = max(float(np.max(np.abs(vl))), float(np.max(np.abs(vr))))
        wheel_speed_limit = float(cfg.no_load_wheel_speed_worst_case())
        speed_violation = max(0.0, max_wheel_speed - wheel_speed_limit)

        boxes_np = np.asarray(scan_out["boxes"], dtype=np.float64)
        t_errs_np = np.asarray(scan_out["tracking_errors"], dtype=np.float64)
        h_errs_np = np.asarray(scan_out["heading_errors"], dtype=np.float64)
        radii_np = np.asarray(scan_out["radii"], dtype=np.float64)

        envelope = [{
            "x_min": float(boxes_np[0, 0]),
            "x_max": float(boxes_np[0, 1]),
            "y_min": float(boxes_np[0, 2]),
            "y_max": float(boxes_np[0, 3]),
            "theta_min": float(boxes_np[0, 4]),
            "theta_max": float(boxes_np[0, 5]),
            "tracking_error_m": 0.0,
            "heading_error_rad": 0.0,
        }]
        diverged = False
        for idx in range(len(t_errs_np)):
            if radii_np[idx] > 0.05:
                diverged = True
            env_item = {
                "x_min": float(boxes_np[idx + 1, 0]),
                "x_max": float(boxes_np[idx + 1, 1]),
                "y_min": float(boxes_np[idx + 1, 2]),
                "y_max": float(boxes_np[idx + 1, 3]),
                "theta_min": float(boxes_np[idx + 1, 4]),
                "theta_max": float(boxes_np[idx + 1, 5]),
                "tracking_error_m": float(t_errs_np[idx]),
                "heading_error_rad": float(h_errs_np[idx]),
            }
            if diverged:
                env_item["unknown"] = True
            envelope.append(env_item)

        max_track = float(scan_out["max_tracking_error_m"])
        max_head = float(scan_out["max_heading_error_rad"])
        max_env = float(scan_out["max_envelope_radius_m"])
        max_err = float(scan_out["max_error_m"])
        rms_track = float(np.sqrt(np.mean(t_errs_np ** 2))) if t_errs_np.size else 0.0
        rms_head = float(np.sqrt(np.mean(h_errs_np ** 2))) if h_errs_np.size else 0.0

        reach_report = {
            "max_error_m": max_err,
            "max_tracking_error_m": max_track,
            "rms_tracking_error_m": rms_track,
            "max_heading_error_rad": max_head,
            "rms_heading_error_rad": rms_head,
            "max_envelope_radius_m": max_env,
            "envelope": envelope,
            "diverged": diverged,
            "envelope_truncated": diverged,
            "propagation": "plant",
        }

        is_safe = (
            max_motor_vio < 1e-3
            and max_slip_l < 1e-3
            and max_slip_r < 1e-3
            and max_track <= safety_threshold_m
            and speed_violation <= 1e-6
            and not diverged
        )

        return {
            "max_motor_violation_N": max_motor_vio,
            "max_traction_violation_N": max_traction_vio,
            "max_slip_l_N": max_slip_l,
            "max_slip_r_N": max_slip_r,
            "max_tracking_error_m": max_track,
            "rms_tracking_error_m": rms_track,
            "max_heading_error_rad": max_head,
            "rms_heading_error_rad": rms_head,
            "max_envelope_radius_m": max_env,
            "max_wheel_speed_m_s": max_wheel_speed,
            "wheel_speed_limit_m_s": wheel_speed_limit,
            "speed_violation_m_s": speed_violation,
            "reachability": reach_report,
            "inclusion": {
                "plant": self.validator.plant_inclusion,
                "controller": self.validator.controller_inclusion,
                "tick": self.validator.tick_inclusion,
            },
            "propagation": getattr(self.validator, "propagation", "plant"),
            "digital_twin": (
                self.validator.digital_twin.summary()
                if self.validator.digital_twin is not None
                else None
            ),
            "passed": bool(is_safe),
        }

    def evaluate_batch_vmap(
        self,
        batch_states: jnp.ndarray,
        batch_dts: jnp.ndarray,
        sample_to_segment: Sequence[int],
    ) -> Dict[str, jnp.ndarray]:
        """Layer 3 batched evaluation across multiple candidates simultaneously via jax.vmap."""
        scan_fn = self._get_scan_fn()
        seg_idx_arr = jnp.asarray(sample_to_segment, dtype=jnp.int32)
        active_mask = jnp.ones(batch_states.shape[1] - 1, dtype=bool)
        w_lower, w_upper = self._uncertainty_bounds()
        vmapped_fn = jit(vmap(lambda s, d: scan_fn(s, d, seg_idx_arr, active_mask, w_lower, w_upper)))
        return vmapped_fn(batch_states, batch_dts)


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
                                heading = (heading + math.pi + math.pi) % (2.0 * math.pi) - math.pi
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
        with _JAX_COMPILE_LOCK:
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
        with _JAX_COMPILE_LOCK:
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
            with _JAX_COMPILE_LOCK:
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
            with _JAX_COMPILE_LOCK:
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
        with _JAX_COMPILE_LOCK:
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
            with _JAX_COMPILE_LOCK:
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
                with _JAX_COMPILE_LOCK:
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
            with _JAX_COMPILE_LOCK:
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
    completely eliminating Windows _CASADI_SOLVE_LOCK contention.
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


# ==============================================================================
# SEGMENT-WISE RECOMBINATION CROSSOVER ("CHIMERA SPLICING")
# ==============================================================================

def recombine_candidate_segments(
    candidates_data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    waypoints: List[Tuple[float, float, Optional[float]]],
    robot_cfg: RobotConfig,
    engine: CompiledReachabilityEngine,
    stop_waypoint_indices: Optional[List[int]] = None,
    accuracy_weight: float = 0.0,
    force_weight: float = 0.0,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    max_polish_iters: int = 250,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Advanced Optimization: Segment-Wise Recombination Splicer.
    Evaluates candidate solutions segment-by-segment, extracts the best local
    segment from each candidate, blends transition knots, and rapidly warm-polishes
    the recombined Chimera trajectory.
    """
    valid_cands = [
        (samples, stats) for samples, stats in candidates_data
        if stats.get("converged", False) and samples
    ]
    if len(valid_cands) < 2:
        return None, None, {"recombined": False, "reason": "Fewer than 2 valid converged candidates"}

    num_segments = len(waypoints) - 1
    sample_0, stats_0 = valid_cands[0]
    segment_samples = stats_0.get("segment_samples")
    if not segment_samples or len(segment_samples) != num_segments:
        opt = TrajectoryOptimizer(robot_cfg)
        segment_samples = opt.get_segment_sample_counts(
            waypoints, len(sample_0) // max(1, num_segments), proportional=True
        )
    segment_samples = [int(s) for s in segment_samples]
    N = sum(segment_samples) + 1
    wp_indices = [0] + list(np.cumsum(segment_samples))
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    stops = set(stop_waypoint_indices or [])

    # Evaluate segment-by-segment score for each candidate
    candidate_states_list = []
    candidate_dts_list = []
    candidate_segment_scores = []  # shape: [num_candidates, num_segments]

    for c_idx, (samples, stats) in enumerate(valid_cands):
        states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
        candidate_states_list.append(states)
        candidate_dts_list.append(dts)

        with _JAX_COMPILE_LOCK:
            rep = engine.evaluate_states(states, dts, sample_to_segment)
        track_errs = np.asarray(rep["tracking_errors"])
        radii = np.asarray(rep["radii"])

        seg_scores = []
        for s in range(num_segments):
            k_start = wp_indices[s]
            k_end = wp_indices[s + 1]
            seg_dt = float(dts[s])
            seg_time = seg_dt * segment_samples[s]
            seg_track_err = float(np.max(track_errs[k_start:k_end]))
            seg_tube = float(np.max(radii[k_start:k_end]))

            score = (
                seg_time
                + 120.0 * max(0.0, seg_track_err - 0.005)
                + 60.0 * max(0.0, seg_tube - 0.020)
            )
            seg_scores.append(score)
        candidate_segment_scores.append(seg_scores)

    # Determine segment winners
    winners_per_segment = []
    for s in range(num_segments):
        best_cand = min(range(len(valid_cands)), key=lambda c: candidate_segment_scores[c][s])
        winners_per_segment.append(best_cand)

    unique_winners = set(winners_per_segment)
    if len(unique_winners) <= 1:
        # One candidate dominated all segments, no chimera splicing needed
        dominant_cand = list(unique_winners)[0]
        return (
            valid_cands[dominant_cand][0],
            valid_cands[dominant_cand][1],
            {"recombined": False, "reason": "Single candidate won all segments", "dominant_candidate": dominant_cand},
        )

    # Build Chimera States and DTs
    chimera_states = np.zeros((N, 5), dtype=np.float64)
    chimera_dts = np.zeros(num_segments, dtype=np.float64)

    for s in range(num_segments):
        k_win = winners_per_segment[s]
        win_states = candidate_states_list[k_win]
        win_dts = candidate_dts_list[k_win]
        k_start = wp_indices[s]
        k_end = wp_indices[s + 1]

        chimera_dts[s] = win_dts[s]
        chimera_states[k_start : k_end + 1] = win_states[k_start : k_end + 1]

    # Knot Alignment & Transition Smoothing at Waypoints
    for s in range(1, num_segments):
        idx = wp_indices[s]
        if s in stops:
            chimera_states[idx, 3:5] = 0.0
        else:
            k_prev = winners_per_segment[s - 1]
            k_next = winners_per_segment[s]
            if k_prev != k_next:
                prev_v = candidate_states_list[k_prev][idx, 3:5]
                next_v = candidate_states_list[k_next][idx, 3:5]
                chimera_states[idx, 3:5] = 0.5 * (prev_v + next_v)

    # Rapid Warm-Start IPOPT Polish
    chimera_guess = np.concatenate([chimera_dts, chimera_states.flatten()])
    pol_kw = dict(solve_kwargs or {})
    pol_kw["initial_guess"] = chimera_guess
    pol_kw["num_samples_per_segment"] = segment_samples
    pol_kw["per_segment_dt"] = True
    pol_kw["cost_metric"] = "discretization"
    pol_kw["epigraph_constraints"] = True
    pol_kw["accuracy_weight"] = accuracy_weight
    pol_kw["force_weight"] = force_weight
    pol_kw["verbose"] = False
    pol_kw["self_consistent"] = False
    pol_kw["solver_options"] = {"max_iter": max_polish_iters}

    pol_opt = TrajectoryOptimizer(robot_cfg)
    try:
        pol_samples, pol_stats = pol_opt.solve(waypoints, **pol_kw)
    except Exception as exc:
        return None, None, {"recombined": False, "reason": f"Polish solve failed: {exc}"}

    if not pol_stats.get("converged", False) or not pol_samples:
        return None, None, {"recombined": False, "reason": "Polish solve did not converge"}

    c_states, c_dts = _extract_state_and_dt_arrays(pol_samples, num_segments, segment_samples)
    with _JAX_COMPILE_LOCK:
        rep = engine.evaluate_states(c_states, c_dts, sample_to_segment)

    recomb_info = {
        "recombined": True,
        "segment_winners": winners_per_segment,
        "unique_contributors": sorted(list(unique_winners)),
        "polish_iters": pol_stats.get("iter_count", 0),
        "tracking_m": float(rep["max_tracking_error_m"]),
        "envelope_m": float(rep["max_envelope_radius_m"]),
        "duration_s": float(pol_samples[-1]["t"]) if pol_samples else 0.0,
    }
    pol_stats["chimera_recombination"] = recomb_info
    return pol_samples, pol_stats, recomb_info


# ==============================================================================
# MEMETIC EVOLUTION MODULE (GENETIC SPLICING + CREEP MUTATION + LOCAL SEARCH)
# ==============================================================================

def _compute_hierarchical_weights(num_survivors: int, power: float = 1.6) -> np.ndarray:
    """
    Computes monotonically increasing element weights across survivors:
    - 4th-worst (index num_survivors - 1): small element (~0.05 - 0.08)
    - 5th best (index 4): bigger element
    - 4th best (index 3): even bigger element
    - 3rd, 2nd best: increasingly larger elements
    - Best (index 0): 100% baseline weight (1.0)
    """
    ranks_from_bottom = np.arange(1, num_survivors + 1, dtype=np.float64)
    weights = (ranks_from_bottom / float(num_survivors)) ** power
    return weights[::-1]


def _compute_shared_fitness(
    population: List[Dict[str, Any]],
    d_niche: float = 0.08,
    sharing_power: float = 0.5,
) -> np.ndarray:
    """
    Computes niche-discounted shared fitness to prevent premature diversity collapse.
    Individuals clustered in the same spatial basin receive a density penalty,
    preserving structurally distinct alternative homotopies in the surviving pool.
    """
    n = len(population)
    if n <= 1:
        return np.array([p["score"] for p in population])

    positions = np.array([p["states"][:, 0:2] for p in population])  # [n, N, 2]
    diff = positions[:, None, :, :] - positions[None, :, :, :]        # [n, n, N, 2]
    dist_matrix = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)    # [n, n]

    sharing = np.maximum(0.0, 1.0 - dist_matrix / max(d_niche, 1e-4))
    m = np.sum(sharing, axis=1)  # niche count

    raw_scores = np.array([p["score"] for p in population])
    return raw_scores * (m ** sharing_power)


def run_memetic_evolution(
    candidates_data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    waypoints: List[Tuple[float, float, Optional[float]]],
    robot_cfg: RobotConfig,
    engine: CompiledReachabilityEngine,
    num_generations: int = 3,
    population_size: int = 10,
    discard_worst_count: int = 3,
    stop_waypoint_indices: Optional[List[int]] = None,
    accuracy_weight: float = 0.0,
    force_weight: float = 0.0,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    max_polish_iters: int = 150,
    max_workers: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Memetic Evolution Trajectory Optimizer:
    Iterative population-based genetic algorithm that models robot trajectories as
    discrete chromosomes of segment timings and spatial state sequences.

    Lifecycle per generation:
    1. Evaluate & Rank: Rank population by multi-objective fitness (duration + tracking error + envelope radius).
    2. Truncation Selection: Discard worst 30% candidates (lowest 3 scoring paths).
    3. Elitism: Keep #1 overall best path intact and unaltered (100% of the best).
    4. Hierarchical Recombination: Combine a small element from the 4th-worst, a bigger element
       from the 5th best, an even bigger element from the 4th best, etc., and 100% of the best,
       incorporating a calibrated degree of randomness.
    5. Creep Mutation: Apply Gaussian perturbation to segment dts and lateral path coordinates.
    6. Memetic Local Search: Fast warm-start IPOPT polish on newly bred offspring in parallel.
    """
    valid_cands = [
        (samples, stats) for samples, stats in candidates_data
        if stats.get("converged", False) and samples
    ]
    if not valid_cands:
        return None, None, {"converged": False, "reason": "No valid converged candidates to seed evolution"}

    rng = np.random.default_rng(seed)
    num_segments = len(waypoints) - 1
    sample_0, stats_0 = valid_cands[0]
    segment_samples = stats_0.get("segment_samples")
    if not segment_samples or len(segment_samples) != num_segments:
        opt = TrajectoryOptimizer(robot_cfg)
        segment_samples = opt.get_segment_sample_counts(
            waypoints, len(sample_0) // max(1, num_segments), proportional=True
        )
    segment_samples = [int(s) for s in segment_samples]
    N = sum(segment_samples) + 1
    wp_indices = [0] + list(np.cumsum(segment_samples))
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    stops = set(stop_waypoint_indices or [])

    def _eval_individual(samples, stats, tag="seed", gen=0):
        states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
        with _JAX_COMPILE_LOCK:
            rep = engine.evaluate_states(states, dts, sample_to_segment)
        track_errs = np.asarray(rep["tracking_errors"])
        radii = np.asarray(rep["radii"])

        seg_scores = []
        for s in range(num_segments):
            k_start = wp_indices[s]
            k_end = wp_indices[s + 1]
            seg_dt = float(dts[s])
            seg_time = seg_dt * segment_samples[s]
            seg_track = float(np.max(track_errs[k_start:k_end]))
            seg_tube = float(np.max(radii[k_start:k_end]))
            seg_score = (
                seg_time
                + 120.0 * max(0.0, seg_track - 0.005)
                + 60.0 * max(0.0, seg_tube - 0.020)
            )
            seg_scores.append(seg_score)

        duration = float(samples[-1]["t"]) if samples else float(np.sum(dts * np.array(segment_samples)))
        max_track = float(rep["max_tracking_error_m"])
        max_tube = float(rep["max_envelope_radius_m"])
        total_score = duration + 120.0 * max(0.0, max_track - 0.005) + 60.0 * max(0.0, max_tube - 0.020)

        return {
            "samples": samples,
            "stats": stats,
            "states": np.asarray(states),
            "dts": np.asarray(dts),
            "score": total_score,
            "duration": duration,
            "tracking_m": max_track,
            "envelope_m": max_tube,
            "segment_scores": seg_scores,
            "tag": tag,
            "generation": gen,
        }

    # Initialize population from valid candidates
    population: List[Dict[str, Any]] = []
    for idx, (samples, stats) in enumerate(valid_cands):
        pop_ind = _eval_individual(samples, stats, tag=f"seed_{idx+1}", gen=0)
        population.append(pop_ind)

    # If population is smaller than target population_size, fill with mutated seeds
    while len(population) < population_size:
        parent = population[len(population) % len(valid_cands)]
        mut_dts = parent["dts"] * (1.0 + rng.normal(0.0, 0.05, size=num_segments))
        mut_dts = np.clip(mut_dts, 0.005, 0.25)
        mut_states = np.copy(parent["states"])
        # Subtle Gaussian lateral jitter on intermediate states
        for s in range(num_segments):
            k_start = wp_indices[s]
            k_end = wp_indices[s + 1]
            if k_end - k_start > 2:
                noise = rng.normal(0.0, 0.01, size=(k_end - k_start - 1, 2))
                mut_states[k_start + 1 : k_end, 0:2] += noise
        mut_guess = np.concatenate([mut_dts, mut_states.flatten()])

        pol_kw = dict(solve_kwargs or {})
        pol_kw["initial_guess"] = mut_guess
        pol_kw["num_samples_per_segment"] = segment_samples
        pol_kw["per_segment_dt"] = True
        pol_kw["cost_metric"] = "discretization"
        pol_kw["epigraph_constraints"] = True
        pol_kw["accuracy_weight"] = accuracy_weight
        pol_kw["force_weight"] = force_weight
        pol_kw["verbose"] = False
        pol_kw["self_consistent"] = False
        pol_kw["solver_options"] = {"max_iter": max_polish_iters}

        opt_mut = TrajectoryOptimizer(robot_cfg)
        try:
            m_samples, m_stats = opt_mut.solve(waypoints, **pol_kw)
            if m_stats.get("converged", False) and m_samples:
                population.append(_eval_individual(m_samples, m_stats, tag=f"seed_mut_{len(population)}", gen=0))
            else:
                population.append(dict(parent))
        except Exception:
            population.append(dict(parent))

    # Clamp population to population_size
    population = population[:population_size]

    history: List[Dict[str, Any]] = []
    snapshots: Dict[int, List[Dict[str, Any]]] = {}

    # Sort initial population
    population.sort(key=lambda ind: ind["score"])
    snapshots[0] = population[0]["samples"]
    init_best = population[0]
    init_mean_score = float(np.mean([ind["score"] for ind in population]))
    history.append({
        "generation": 0,
        "best_duration": init_best["duration"],
        "best_score": init_best["score"],
        "mean_score": init_mean_score,
        "best_tracking_m": init_best["tracking_m"],
        "best_envelope_m": init_best["envelope_m"],
        "best_tag": init_best["tag"],
    })

    if verbose:
        print(f"  [Gen 0] Leader: {init_best['duration']:.3f} s (score {init_best['score']:.3f}, "
              f"track {init_best['tracking_m']*100:.2f} cm, tube {init_best['envelope_m']*100:.2f} cm) | "
              f"Pop mean score: {init_mean_score:.3f}")

    # Evolutionary Loop with Persistent ProcessPoolExecutor
    num_w = max(1, min(discard_worst_count, max_workers or discard_worst_count, os.cpu_count() or 8))
    persistent_pool = None
    if num_w > 1:
        try:
            persistent_pool = ProcessPoolExecutor(max_workers=num_w)
        except Exception:
            persistent_pool = None

    track_w = float(robot_cfg.track_width)
    v_max = float(robot_cfg.max_linear_speed())

    try:
        for gen in range(1, num_generations + 1):
            # 1. Truncation Selection with Spatial Niching (Diversity Preservation)
            k_discard = min(discard_worst_count, len(population) - 2)
            shared_costs = _compute_shared_fitness(population, d_niche=0.08, sharing_power=0.5)

            # Elite is strictly the individual with lowest raw objective score (100% preserved)
            elite_idx = int(np.argmin([p["score"] for p in population]))
            elite = population[elite_idx]

            # Rank the remaining candidates by shared fitness to penalize redundant clones
            other_indices = [i for i in range(len(population)) if i != elite_idx]
            other_indices.sort(key=lambda i: shared_costs[i])

            num_surv_needed = len(population) - k_discard - 1
            survivor_indices = [elite_idx] + other_indices[:num_surv_needed]
            survivors = [population[i] for i in survivor_indices]

            num_surv = len(survivors)
            # Compute hierarchical element weights:
            # survivors[0]: 1.0 (100% of best)
            # survivors[3]: even bigger element (4th best)
            # survivors[4]: bigger element (5th best)
            # survivors[num_surv-1]: small element (4th-worst)
            hier_weights = _compute_hierarchical_weights(num_surv, power=1.6)

            offspring_guesses = []
            offspring_tags = []
            offspring_kws = []

            base_kw = dict(solve_kwargs or {})
            base_kw["num_samples_per_segment"] = segment_samples
            base_kw["per_segment_dt"] = True
            base_kw["cost_metric"] = "discretization"
            base_kw["epigraph_constraints"] = True
            base_kw["accuracy_weight"] = accuracy_weight
            base_kw["force_weight"] = force_weight
            base_kw["verbose"] = False
            base_kw["self_consistent"] = False
            base_kw["solver_options"] = {"max_iter": max_polish_iters}

            # 2. Hierarchical Recombination Breeding
            # Generate k_discard offspring combining hierarchical elements + macro-mutation
            for off_idx in range(k_discard):
                # Calibrated degree of randomness for this offspring
                sigma_rand = 0.20 + 0.15 * off_idx  # 0.20 for first, up to 0.50 for exploration
                off_dts = np.zeros(num_segments, dtype=np.float64)
                off_states = np.zeros((N, 5), dtype=np.float64)
                donor_trail = []

                for s in range(num_segments):
                    # Calculate priority for each survivor on segment s:
                    # combines overall hierarchical rank weight with segment performance + randomness
                    seg_perf = np.array([p["segment_scores"][s] for p in survivors], dtype=np.float64)
                    min_p = float(np.min(seg_perf))
                    # Normalized relative segment gap
                    seg_gap = (seg_perf - min_p) / (min_p + 1e-4)

                    # Stochastic perturbation (degree of randomness)
                    rand_jitter = rng.normal(0.0, sigma_rand, size=num_surv)

                    # Priority score: higher is better
                    log_priorities = np.log(np.maximum(hier_weights, 1e-6)) - 1.5 * seg_gap + rand_jitter
                    # Softmax probabilities
                    max_log = np.max(log_priorities)
                    probs = np.exp(log_priorities - max_log)
                    probs = probs / np.sum(probs)

                    # Sample donor parent according to hierarchical distribution
                    chosen_r = int(rng.choice(num_surv, p=probs))
                    donor_trail.append(chosen_r)
                    donor = survivors[chosen_r]

                    k_start = wp_indices[s]
                    k_end = wp_indices[s + 1]

                    # Pure Structural Segment Splicing:
                    # Inherit the donor segment states directly without Euclidean coordinate interpolation.
                    # This prevents cutting inside the turning center or violating non-holonomic no-slip.
                    seg_states_donor = np.copy(donor["states"][k_start : k_end + 1])
                    seg_dts_donor = float(donor["dts"][s])

                    # Kinematic Curvature & Tangent Speed Perturbation on the donor segment
                    v_scale = 1.0 + rng.normal(0.0, 0.02 * sigma_rand)
                    kappa_jitter = rng.normal(0.0, 0.04 * sigma_rand)
                    v_lin = 0.5 * (seg_states_donor[:, 3] + seg_states_donor[:, 4]) * v_scale
                    seg_states_donor[:, 3] = np.clip(v_lin * (1.0 - kappa_jitter * track_w * 0.5), -v_max, v_max)
                    seg_states_donor[:, 4] = np.clip(v_lin * (1.0 + kappa_jitter * track_w * 0.5), -v_max, v_max)

                    timing_creep = 1.0 + rng.normal(0.0, 0.02 * sigma_rand)
                    off_dts[s] = np.clip(seg_dts_donor * timing_creep, 0.005, 0.25)
                    off_states[k_start : k_end + 1] = seg_states_donor

                # Macro-Homotopy Mutation: on exploratory offspring, shift lateral apex and perturb timing
                # to break free from local basins and discover alternative geometric corridors
                if off_idx >= 1:
                    mut_segs = rng.choice(num_segments, size=min(2, num_segments), replace=False)
                    for ms in mut_segs:
                        off_dts[ms] = np.clip(off_dts[ms] * rng.uniform(0.80, 1.25), 0.005, 0.25)

                    disp_seg = int(rng.integers(0, num_segments))
                    ks = wp_indices[disp_seg]
                    ke = wp_indices[disp_seg + 1]
                    if ke - ks > 2:
                        p1 = off_states[ks, 0:2]
                        p2 = off_states[ke, 0:2]
                        chord = p2 - p1
                        chord_len = float(np.linalg.norm(chord))
                        if chord_len > 0.08:
                            norm_vec = np.array([-chord[1], chord[0]]) / chord_len
                            shift_mag = float(rng.uniform(-0.06, 0.06))
                            tau = np.linspace(0.0, 1.0, ke - ks + 1)[:, None]
                            bump = 4.0 * tau * (1.0 - tau)  # parabola: 0 at endpoints, 1 at midpoint
                            off_states[ks : ke + 1, 0:2] += bump * shift_mag * norm_vec

                # SE(2) Continuous Phase Unwrapping across spliced knot boundaries
                for s in range(1, num_segments):
                    idx = wp_indices[s]
                    dth = off_states[idx, 2] - off_states[idx - 1, 2]
                    wrapped_dth = (dth + math.pi) % (2.0 * math.pi) - math.pi
                    off_states[idx:, 2] += (wrapped_dth - dth)
                    if s in stops:
                        off_states[idx, 3:5] = 0.0
                    else:
                        # Smooth knot linear and angular velocities to preserve dynamic continuity
                        off_states[idx, 3:5] = 0.5 * (off_states[idx - 1, 3:5] + off_states[min(idx + 1, N - 1), 3:5])

                # Layer 4 Directed Memetic Mutation (Gradient Guidance on Exploratory Offspring)
                if gen > 1 and off_idx == 0:
                    try:
                        grad_fn = engine._get_grad_fn(safety_gate_m=0.005, tube_gate_m=0.020)
                        seg_idx_arr = jnp.asarray(sample_to_segment, dtype=jnp.int32)
                        active_mask = jnp.ones(off_states.shape[0] - 1, dtype=bool)
                        w_lower, w_upper = engine._uncertainty_bounds()
                        with _JAX_COMPILE_LOCK:
                            (_, _), (g_states, g_dts) = grad_fn(
                                jnp.asarray(off_states), jnp.asarray(off_dts), seg_idx_arr, active_mask, w_lower, w_upper
                            )
                        g_s = np.asarray(g_states)
                        g_d = np.asarray(g_dts)
                        if np.all(np.isfinite(g_s)) and np.all(np.isfinite(g_d)):
                            off_states[:, 0:2] -= np.clip(0.004 * g_s[:, 0:2], -0.010, 0.010)
                            off_dts = np.clip(off_dts - np.clip(0.004 * g_d, -0.010, 0.010), 0.005, 0.25)
                    except Exception:
                        pass

                offspring_guesses.append(np.concatenate([off_dts, off_states.flatten()]))
                offspring_tags.append(f"hier_mix_g{gen}_off{off_idx+1}")

                # Fix 1: Dynamically Align IPOPT with Reachability Errors
                # If donor candidates have tracking errors > 5 mm, dynamically increase accuracy_weight
                # so IPOPT actively penalizes discretization error and avoids flooring throttle in corners
                max_donor_track = max(survivors[r]["tracking_m"] for r in donor_trail)
                excess_err = max(0.0, max_donor_track - 0.005)
                adaptive_acc = max(accuracy_weight, float(np.clip(0.002 + 0.06 * (excess_err / 0.02), 0.0, 0.012)))

                off_kw = dict(base_kw)
                off_kw["accuracy_weight"] = adaptive_acc
                offspring_kws.append(off_kw)

            # 3. Memetic Local Search: Fast Parallel IPOPT Polish on Offspring via Persistent Pool
            max_w = min(len(offspring_guesses), max_workers or len(offspring_guesses))
            polished_results = solve_pyramid_candidates(
                robot_cfg, waypoints, offspring_guesses, offspring_kws, max_workers=max_w, pool=persistent_pool
            )

            new_offspring_inds = []
            for o_idx, (o_samples, o_stats) in enumerate(polished_results):
                tag = offspring_tags[o_idx]
                if o_stats.get("converged", False) and o_samples:
                    ind = _eval_individual(o_samples, o_stats, tag=tag, gen=gen)
                    new_offspring_inds.append(ind)
                else:
                    # If polish failed to converge, evaluate unpolished guess with penalty
                    g = offspring_guesses[o_idx]
                    dts_guess = g[:num_segments]
                    states_guess = g[num_segments:].reshape((N, 5))
                    mock_samples = []
                    cur_t = 0.0
                    for i in range(N):
                        s_idx = sample_to_segment[min(i, len(sample_to_segment) - 1)]
                        mock_samples.append({
                            "t": cur_t,
                            "x": float(states_guess[i, 0]),
                            "y": float(states_guess[i, 1]),
                            "heading": float(states_guess[i, 2]),
                            "vl": float(states_guess[i, 3]),
                            "vr": float(states_guess[i, 4]),
                        })
                        cur_t += float(dts_guess[s_idx])
                    mock_stats = {"converged": False, "segment_dt": list(dts_guess)}
                    ind = _eval_individual(mock_samples, mock_stats, tag=f"{tag}_unpolished", gen=gen)
                    ind["score"] += 1.0  # unconverged penalty
                    new_offspring_inds.append(ind)

            # 4. Integrate survivors + new offspring into population
            population = survivors + new_offspring_inds
            population.sort(key=lambda ind: ind["score"])

            current_best = population[0]
            current_mean = float(np.mean([ind["score"] for ind in population]))
            snapshots[gen] = current_best["samples"]
            history.append({
                "generation": gen,
                "best_duration": current_best["duration"],
                "best_score": current_best["score"],
                "mean_score": current_mean,
                "best_tracking_m": current_best["tracking_m"],
                "best_envelope_m": current_best["envelope_m"],
                "best_tag": current_best["tag"],
            })

            if verbose:
                print(f"  [Gen {gen}] Leader: {current_best['duration']:.3f} s (score {current_best['score']:.3f}, "
                      f"track {current_best['tracking_m']*100:.2f} cm, tube {current_best['envelope_m']*100:.2f} cm, "
                      f"origin '{current_best['tag']}') | Pop mean: {current_mean:.3f}")
    finally:
        if persistent_pool is not None:
            try:
                persistent_pool.shutdown(wait=False)
            except Exception:
                pass

    champion = population[0]
    champ_stats = dict(champion["stats"])
    champ_stats["converged"] = True
    champ_stats["segment_dt"] = list(champion["dts"])

    evolution_report = {
        "generations": history,
        "snapshots": snapshots,
        "population_leaderboard": [
            {
                "rank": i + 1,
                "tag": ind["tag"],
                "duration_s": ind["duration"],
                "score": ind["score"],
                "tracking_cm": ind["tracking_m"] * 100.0,
                "envelope_cm": ind["envelope_m"] * 100.0,
                "generation": ind["generation"],
            }
            for i, ind in enumerate(population)
        ],
        "champion_duration": champion["duration"],
        "champion_score": champion["score"],
        "champion_tag": champion["tag"],
    }
    champ_stats["memetic_evolution"] = evolution_report

    return champion["samples"], champ_stats, evolution_report

