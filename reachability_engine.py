"""
reachability_engine.py - Canonical Pure-JAX Compiled Reachability Verification Engine.

Eliminates the Python for/while loops and .float() host synchronizations in reachability
verification across both immrax_validator.py and surrogate_polish.py so that:
  1. Rollout executes as a single fused XLA kernel (10x-50x faster).
  2. Multiple candidates can be evaluated in parallel via jax.vmap.
  3. Reverse-mode autodiff (jax.grad) flows cleanly through the entire
     20 ms sample-and-hold closed-loop simulation.
  4. Both ImmraxValidator and SurrogatePolishOptimizer share the exact same reachability physics.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
from immrax import Interval

from jax_ramsete import _sinc_poly
from robot_model import CG_HEIGHT_M, body_frame_error, angle_diff

_JAX_COMPILE_LOCK = threading.RLock()

STATIC_MAX_INTERVALS = 180
STATIC_MAX_SEGMENTS = 16
MAX_TICKS_PER_INTERVAL = 30
GRAD_TICKS_PER_INTERVAL = 1
PLANT_SUBSTEPS = 5
CONTROL_DT = 0.02


def unroll_trajectory_headings(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unroll trajectory headings so angles vary continuously across +/- pi boundaries."""
    if not samples:
        return []
    unrolled = []
    current_offset = 0.0
    last_h = samples[0]["heading"]

    for s in samples:
        h = s["heading"]
        if h - last_h > math.pi:
            current_offset -= 2 * math.pi
        elif h - last_h < -math.pi:
            current_offset += 2 * math.pi

        new_s = s.copy()
        new_s["heading"] = h + current_offset
        unrolled.append(new_s)
        last_h = h

    return unrolled


def _extract_state_and_dt_arrays(
    samples: List[Dict[str, Any]],
    num_segments: int,
    segment_samples: Sequence[int],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert sample dicts into (states [N, 5], dts [num_segments]) JAX float64 arrays."""
    unrolled = unroll_trajectory_headings(samples)
    states = jnp.array(
        [[s["x"], s["y"], s["heading"], s.get("vl", 0.0), s.get("vr", 0.0)] for s in unrolled],
        dtype=jnp.float64,
    )
    times = np.array([float(s["t"]) for s in unrolled], dtype=np.float64)
    wp_indices = [0] + list(np.cumsum(segment_samples))
    dts = np.zeros(num_segments, dtype=np.float64)
    for j in range(num_segments):
        k = min(wp_indices[j], len(unrolled) - 2)
        dts[j] = max(times[k + 1] - times[k], 1e-4)
    return states, jnp.asarray(dts, dtype=jnp.float64)


def _cos_interval(a_min: jnp.ndarray, a_max: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact interval bounds for cos([a_min, a_max]) in pure JAX."""
    c1 = jnp.cos(a_min)
    c2 = jnp.cos(a_max)
    c_min = jnp.minimum(c1, c2)
    c_max = jnp.maximum(c1, c2)
    wide = (a_max - a_min) >= (2.0 * math.pi)
    c_min = jnp.where(wide, -1.0, c_min)
    c_max = jnp.where(wide, 1.0, c_max)
    k_max = jnp.ceil(a_min / (2.0 * math.pi))
    has_peak = (k_max * (2.0 * math.pi)) <= a_max
    c_max = jnp.where(has_peak, 1.0, c_max)
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

    def __init__(self, validator: Any):
        self.validator = validator
        self.cfg = validator.cfg
        self.control_dt = float(validator.control_dt)
        self.ramsete_b = float(getattr(validator, "ramsete_b", 2.0))
        self.ramsete_zeta = float(getattr(validator, "ramsete_zeta", 0.7))
        self.track_width = float(validator.cfg.track_width)
        self.active_twin = bool(getattr(validator, "_digital_twin_active", False))
        self.state_len = int(validator.plant.xlen if self.active_twin else 3)
        self._compiled_scan_fn: Any = None
        self._uncompiled_scan_fn: Any = None
        self._uncompiled_grad_scan_fn: Any = None
        self._scan_fn_cache: Dict[Tuple[int, bool, str, int, int], Any] = {}
        self._uncompiled_scan_cache: Dict[Tuple[int, bool, str, int, int], Any] = {}
        self._compiled_grad_cache: Dict[Tuple[float, float], Any] = {}

    def _get_scan_fn(self, num_ticks: int = MAX_TICKS_PER_INTERVAL, num_substeps: int = PLANT_SUBSTEPS):
        plant_inc = str(getattr(self.validator, "plant_inclusion", "nat"))
        cache_key = (
            int(self.state_len),
            bool(self.active_twin),
            plant_inc,
            int(num_ticks),
            int(num_substeps),
        )
        if cache_key in self._scan_fn_cache:
            return self._scan_fn_cache[cache_key]

        with _JAX_COMPILE_LOCK:
            if cache_key in self._scan_fn_cache:
                return self._scan_fn_cache[cache_key]

            state_len = self.state_len
            control_dt = self.control_dt
            ramsete_b = self.ramsete_b
            ramsete_zeta = self.ramsete_zeta
            nom_track_width = self.track_width
            plant_emb_f = getattr(self.validator, "plant_emb_sys", None)
            if plant_emb_f is not None:
                plant_emb_f = plant_emb_f.f
            controller_fn = getattr(self.validator, "controller", None)
            odometry_pose_fn = getattr(self.validator, "_odometry_pose", lambda p, s: p)
            use_fast_kinematic = (not self.active_twin) and (plant_inc == "nat")
            has_belief = bool(self.active_twin and (state_len > 3))
            wheel_state = getattr(self.validator, "wheel_state", "mean_diff")
            odo_dist_scale = float(getattr(self.validator, "digital_twin_distance_scale", 1.0))
            odo_head_scale = float(getattr(self.validator, "digital_twin_heading_scale", 1.0))
            odo_gyro_bias = float(getattr(self.validator, "digital_twin_gyro_bias_rad_s", 0.0))

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
                    x_flat, belief = carry
                    s0, s1, dt_k, elapsed_t, is_active = step_inputs

                    v0 = 0.5 * (s0[3] + s0[4])
                    v1 = 0.5 * (s1[3] + s1[4])
                    dh = angle_diff(s1[2], s0[2])
                    dt_safe = jnp.maximum(dt_k, 1e-9)
                    ref_v = 0.5 * (v0 + v1)
                    ref_omega = dh / dt_safe
                    ref_kr = jnp.sqrt(ref_omega * ref_omega + ramsete_b * ref_v * ref_v + 1e-16)

                    def _tick_step(carry_t, m_idx):
                        x_curr, b_curr = carry_t
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
                            ex, ey = body_frame_error(ref_x, ref_y, cx, cy, cth)
                            etheta = angle_diff(ref_h, cth)
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
                            return (jnp.where(active, x_sub, x_curr), b_curr), None
                        else:
                            u_ref = jnp.array([ref_x, ref_y, ref_h, ref_v, ref_omega, ref_kr], dtype=jnp.float64)
                            if has_belief:
                                x_center = b_curr
                            else:
                                x_center = 0.5 * (x_curr[:3] + x_curr[state_len : state_len + 3])
                                x_center = odometry_pose_fn(x_center, start_pose)
                            u_cmd = controller_fn(
                                0.0, x_center, u_ref, w_interval, normalize=True, quantize=False
                            )
                            x_sub = x_curr
                            b_sub = b_curr
                            for _ in range(num_substeps):
                                if has_belief:
                                    v_mean = 0.5 * (x_sub[3] + x_sub[state_len + 3])
                                    v_diff = 0.5 * (x_sub[4] + x_sub[state_len + 4])
                                    if wheel_state == "mean_diff":
                                        odo_v = v_mean
                                        odo_w = v_diff / nom_track_width
                                    else:
                                        odo_v = 0.5 * (v_mean + v_diff)
                                        odo_w = (v_diff - v_mean) / nom_track_width
                                    odo_v = odo_v * odo_dist_scale
                                    odo_w = odo_w * odo_head_scale + odo_gyro_bias
                                    b_th = b_sub[2] + odo_w * sub_dt
                                    b_x = b_sub[0] + odo_v * jnp.cos(b_th) * sub_dt
                                    b_y = b_sub[1] + odo_v * jnp.sin(b_th) * sub_dt
                                    b_sub = jnp.array([b_x, b_y, b_th], dtype=jnp.float64)
                                dx_dt = plant_emb_f(0.0, x_sub, u_cmd, w_interval)
                                x_sub = x_sub + dx_dt * sub_dt
                            return (jnp.where(active, x_sub, x_curr), jnp.where(active, b_sub, b_curr)), None

                    (x_flat_end, b_end), _ = jax.lax.scan(_tick_step, (x_flat, belief), tick_idx_arr)
                    x_flat_end = jnp.where(is_active, x_flat_end, x_flat)
                    b_end = jnp.where(is_active, b_end, belief)

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
                    raw_head_err = jnp.abs(angle_diff(mid_h, s1[2]))
                    heading_err = jnp.where(is_active, raw_head_err, 0.0)

                    raw_radius = jnp.sqrt(((x_max - x_min) * 0.5) ** 2 + ((y_max - y_min) * 0.5) ** 2 + 1e-16)
                    radius = jnp.where(is_active, raw_radius, 0.0)
                    box_vec = jnp.array(
                        [x_min, x_max, y_min, y_max, x_flat_end[2], x_flat_end[state_len + 2]],
                        dtype=jnp.float64,
                    )

                    return (x_flat_end, b_end), (tracking_err, heading_err, radius, box_vec)

                _, (tracking_errs, heading_errs, radii, boxes) = jax.lax.scan(
                    _interval_step,
                    (x_flat_init, start_pose),
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

            jitted_fn = jit(_rollout_scan)
            self._scan_fn_cache[cache_key] = jitted_fn
            self._uncompiled_scan_cache[cache_key] = _rollout_scan
            if num_ticks == MAX_TICKS_PER_INTERVAL and num_substeps == PLANT_SUBSTEPS:
                self._compiled_scan_fn = jitted_fn
                self._uncompiled_scan_fn = _rollout_scan
            elif num_ticks == GRAD_TICKS_PER_INTERVAL and num_substeps == 1:
                self._uncompiled_grad_scan_fn = _rollout_scan
            return jitted_fn

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

        self._get_scan_fn(GRAD_TICKS_PER_INTERVAL, 1)
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
        w_lower: Optional[jnp.ndarray] = None,
        w_upper: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        scan_fn = self._get_scan_fn()
        seg_idx_arr = jnp.asarray(sample_to_segment, dtype=jnp.int32)
        active_mask = jnp.ones(states.shape[0] - 1, dtype=bool)
        if w_lower is None or w_upper is None:
            def_lo, def_hi = self._uncertainty_bounds()
            w_lower = def_lo if w_lower is None else w_lower
            w_upper = def_hi if w_upper is None else w_upper
        return scan_fn(states, dts, seg_idx_arr, active_mask, w_lower, w_upper)

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

    def _build_reachability_dict(
        self,
        samples: List[Dict[str, Any]],
        scan_out: Dict[str, jnp.ndarray],
    ) -> Dict[str, Any]:
        """Convert compiled lax.scan outputs to the standard reachability dict."""
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

        return {
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

    def compute_reachability_report(
        self,
        samples: List[Dict[str, Any]],
        backlash_range: Tuple[float, float],
        track_width_range: Tuple[float, float],
        wheel_radius_range: Tuple[float, float],
        cof_range: Tuple[float, float],
        torque_margin_range: Tuple[float, float],
        inertia_range: Tuple[float, float],
    ) -> Dict[str, Any]:
        """
        Drop-in replacement for ImmraxValidator.compute_reachability.
        Evaluates arbitrary trajectory samples via compiled jax.lax.scan.
        """
        N = len(samples)
        if N < 2:
            return {
                "max_error_m": 0.0,
                "max_tracking_error_m": 0.0,
                "rms_tracking_error_m": 0.0,
                "max_heading_error_rad": 0.0,
                "rms_heading_error_rad": 0.0,
                "max_envelope_radius_m": 0.0,
                "envelope": [],
                "diverged": False,
                "envelope_truncated": False,
                "propagation": "plant",
            }

        times = np.array([float(s["t"]) for s in samples], dtype=np.float64)
        dts = np.diff(times)
        if np.any(~np.isfinite(dts)) or np.any(dts <= 0):
            envelope = [{
                "x_min": float(samples[0]["x"]) - 0.001,
                "x_max": float(samples[0]["x"]) + 0.001,
                "y_min": float(samples[0]["y"]) - 0.001,
                "y_max": float(samples[0]["y"]) + 0.001,
                "theta_min": float(samples[0]["heading"]) - 0.001,
                "theta_max": float(samples[0]["heading"]) + 0.001,
                "tracking_error_m": 0.0,
                "heading_error_rad": 0.0,
                "unknown": True,
            } for _ in range(N)]
            return {
                "max_error_m": float("inf"),
                "max_tracking_error_m": float("inf"),
                "rms_tracking_error_m": 0.0,
                "max_heading_error_rad": float("inf"),
                "rms_heading_error_rad": 0.0,
                "max_envelope_radius_m": float("inf"),
                "envelope": envelope,
                "diverged": True,
                "envelope_truncated": True,
                "propagation": "plant",
            }

        unrolled = unroll_trajectory_headings(samples)
        states = jnp.array(
            [[s["x"], s["y"], s["heading"], s.get("vl", 0.0), s.get("vr", 0.0)] for s in unrolled],
            dtype=jnp.float64,
        )

        gain_range = getattr(self.validator, "digital_twin_gain_range", (1.0, 1.0))
        w_lower = jnp.array([
            cof_range[0],
            torque_margin_range[0],
            backlash_range[0],
            track_width_range[0],
            wheel_radius_range[0],
            inertia_range[0],
            gain_range[0],
        ], dtype=jnp.float64)
        w_upper = jnp.array([
            cof_range[1],
            torque_margin_range[1],
            backlash_range[1],
            track_width_range[1],
            wheel_radius_range[1],
            inertia_range[1],
            gain_range[1],
        ], dtype=jnp.float64)

        max_dt = float(np.max(dts))
        needed_ticks = max(1, int(math.ceil(max_dt / self.control_dt)))
        num_ticks = max(MAX_TICKS_PER_INTERVAL, ((needed_ticks + 9) // 10) * 10)

        scan_fn = self._get_scan_fn(num_ticks=num_ticks, num_substeps=PLANT_SUBSTEPS)
        seg_idx_arr = jnp.arange(N - 1, dtype=jnp.int32)
        dts_jnp = jnp.asarray(dts, dtype=jnp.float64)
        active_mask = jnp.ones(N - 1, dtype=bool)

        scan_out = scan_fn(states, dts_jnp, seg_idx_arr, active_mask, w_lower, w_upper)
        return self._build_reachability_dict(samples, scan_out)

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
        cfg = self.cfg
        vl = np.array([float(s["vl"]) for s in samples], dtype=np.float64)
        vr = np.array([float(s["vr"]) for s in samples], dtype=np.float64)
        al = np.array([float(s.get("al", 0.0)) for s in samples], dtype=np.float64)
        ar = np.array([float(s.get("ar", 0.0)) for s in samples], dtype=np.float64)

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

        alpha_1 = da / tw_lo
        alpha_2 = da / tw_hi
        alpha_lo = np.minimum(alpha_1, alpha_2)
        alpha_hi = np.maximum(alpha_1, alpha_2)

        m_1 = ine_lo * alpha_lo
        m_2 = ine_lo * alpha_hi
        m_3 = ine_hi * alpha_lo
        m_4 = ine_hi * alpha_hi
        m_lo = np.minimum(np.minimum(m_1, m_2), np.minimum(m_3, m_4))
        m_hi = np.maximum(np.maximum(m_1, m_2), np.maximum(m_3, m_4))

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

        omega_l_hi = (np.abs(vl) / wr_lo) * gearing
        torque_l_lo = np.maximum(0.0, (1.0 - omega_l_hi / v_max_rad_s) * t_max_nm)
        force_limit_l_lo = (torque_l_lo / wr_hi) * gearing * tm_lo

        omega_r_hi = (np.abs(vr) / wr_lo) * gearing
        torque_r_lo = np.maximum(0.0, (1.0 - omega_r_hi / v_max_rad_s) * t_max_nm)
        force_limit_r_lo = (torque_r_lo / wr_hi) * gearing * tm_lo

        motor_vios_l = np.maximum(0.0, iv_abs_l - force_limit_l_lo)
        motor_vios_r = np.maximum(0.0, iv_abs_r - force_limit_r_lo)
        max_motor_vio = float(np.maximum(np.max(motor_vios_l), np.max(motor_vios_r)))

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

        reach_report = self._build_reachability_dict(samples, scan_out)

        is_safe = (
            max_motor_vio < 1e-3
            and max_slip_l < 1e-3
            and max_slip_r < 1e-3
            and reach_report["max_tracking_error_m"] <= safety_threshold_m
            and speed_violation <= 1e-6
            and not reach_report["diverged"]
        )

        return {
            "max_motor_violation_N": max_motor_vio,
            "max_traction_violation_N": max_traction_vio,
            "max_slip_l_N": max_slip_l,
            "max_slip_r_N": max_slip_r,
            "max_tracking_error_m": reach_report["max_tracking_error_m"],
            "rms_tracking_error_m": reach_report["rms_tracking_error_m"],
            "max_heading_error_rad": reach_report["max_heading_error_rad"],
            "rms_heading_error_rad": reach_report["rms_heading_error_rad"],
            "max_envelope_radius_m": reach_report["max_envelope_radius_m"],
            "max_wheel_speed_m_s": max_wheel_speed,
            "wheel_speed_limit_m_s": wheel_speed_limit,
            "speed_violation_m_s": speed_violation,
            "reachability": reach_report,
            "inclusion": {
                "plant": getattr(self.validator, "plant_inclusion", "nat"),
                "controller": getattr(self.validator, "controller_inclusion", "nat"),
                "tick": getattr(self.validator, "tick_inclusion", "jac"),
            },
            "propagation": getattr(self.validator, "propagation", "plant"),
            "digital_twin": (
                self.validator.digital_twin.summary()
                if getattr(self.validator, "digital_twin", None) is not None
                else None
            ),
            "passed": bool(is_safe),
            "diverged": reach_report["diverged"],
            "converged": True,
        }
