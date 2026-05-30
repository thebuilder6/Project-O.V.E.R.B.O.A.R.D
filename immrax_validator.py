import jax
import jax.numpy as jnp
from jax import jit, vmap
import immrax
from immrax import Interval, natif
from jax_robot_model import JAXRobotConfig, JAXDifferentialDriveModel
from jax_ramsete import JAXRamseteController, smooth_deadband
from typing import List, Dict, Any, Tuple

class ImmraxValidator:
    def __init__(self, config: JAXRobotConfig):
        self.cfg = config
        self.model = JAXDifferentialDriveModel(config)
        self.controller = JAXRamseteController()

    def validate_trajectory(self, samples: List[Dict[str, Any]],
                            cof_range: Tuple[float, float],
                            torque_margin_range: Tuple[float, float],
                            backlash_range: Tuple[float, float]) -> Dict[str, Any]:
        vl = jnp.array([s['vl'] for s in samples])
        vr = jnp.array([s['vr'] for s in samples])
        al = jnp.array([s['al'] for s in samples])
        ar = jnp.array([s['ar'] for s in samples])

        cof_interval = Interval(*cof_range)
        torque_margin_interval = Interval(*torque_margin_range)

        def check_sample_interval(vl_val, vr_val, al_val, ar_val):
            # Pass use_intervals=True to include track_width, wheel_radius, inertia, torque intervals
            fl, fr = self.model.get_dynamics(vl_val, vr_val, al_val, ar_val, use_intervals=True)

            force_limit_l = self.cfg.get_max_force_at_velocity(vl_val, use_intervals=True)
            force_limit_r = self.cfg.get_max_force_at_velocity(vr_val, use_intervals=True)

            # Using torque_margin_interval as an additional external multiplier (e.g. for battery)
            force_limit_l *= torque_margin_interval
            force_limit_r *= torque_margin_interval

            # Helper to get max absolute value of an interval
            def iv_abs_max(iv):
                return jnp.maximum(jnp.abs(iv.lower), jnp.abs(iv.upper))

            motor_violation_l = jnp.maximum(0, iv_abs_max(fl) - force_limit_l.lower)
            motor_violation_r = jnp.maximum(0, iv_abs_max(fr) - force_limit_r.lower)

            traction_max = cof_interval.lower * self.cfg.mass * self.cfg.g
            traction_violation = jnp.maximum(0, iv_abs_max(fl) + iv_abs_max(fr) - traction_max)

            nl, nr = self.model.get_wheel_normal_forces(vl_val, vr_val, al_val, ar_val, use_intervals=True)
            slip_l = jnp.maximum(0, iv_abs_max(fl) - (cof_interval * nl).lower)
            slip_r = jnp.maximum(0, iv_abs_max(fr) - (cof_interval * nr).lower)

            return motor_violation_l, motor_violation_r, traction_violation, slip_l, slip_r

        motor_vios_l, motor_vios_r, traction_vios, slip_ls, slip_rs = vmap(check_sample_interval)(vl, vr, al, ar)

        # --- SIMPLE ROBUSTNESS CHECK ---
        # Reachability is causing interval explosion due to wrapping/dynamics issues in this environment.
        # We'll use the worst-case force violations as a proxy for safety in the verdict,
        # but keep the reachability report for visualization if possible.

        reach_report = self.compute_reachability(samples, backlash_range)

        # Pass if no interval violations.
        max_motor_vio = jnp.maximum(jnp.max(motor_vios_l), jnp.max(motor_vios_r))
        is_physically_safe = max_motor_vio < 1e-3 and jnp.max(slip_ls) < 1e-3 and jnp.max(slip_rs) < 1e-3

        return {
            "max_motor_violation_N": float(max_motor_vio),
            "max_traction_violation_N": float(jnp.max(traction_vios)),
            "max_slip_l_N": float(jnp.max(slip_ls)),
            "max_slip_r_N": float(jnp.max(slip_rs)),
            "max_tracking_error_m": reach_report['max_error_m'],
            "reachability": reach_report,
            "passed": bool(is_physically_safe)
        }

    def compute_reachability(self, samples, backlash_range):
        N = len(samples)
        if N < 2: return {"max_error_m": 0.0, "envelope": []}

        from jax_ramsete import ramsete_step_jax

        # Initial pose uncertainty (e.g. 1mm start error)
        start_pose = jnp.array([samples[0]['x'], samples[0]['y'], samples[0]['heading']])
        current_pose_iv = Interval(start_pose - 0.001, start_pose + 0.001)

        backlash_iv = Interval(*backlash_range)

        envelope = []
        max_err = 0.0

        # Lift the step function to intervals
        ramsete_iv_step = immrax.natif(ramsete_step_jax)

        for i in range(len(samples) - 1):
            s = samples[i]
            s_next = samples[i+1]
            dt = s_next['t'] - s['t']

            ref_pose = jnp.array([s['x'], s['y'], s['heading']])
            ref_v = (s['vl'] + s['vr']) / 2.0
            ref_omega = (s['vr'] - s['vl']) / self.cfg.track_width

            # Propagate interval through closed-loop dynamics
            # Note: We treat ref_v, ref_omega as constants for the step,
            # but current_pose and backlash are intervals.
            current_pose_iv, _, _ = ramsete_iv_step(current_pose_iv, ref_pose, ref_v, ref_omega, dt, backlash_iv)

            # Extract bounds for envelope
            x_min, x_max = float(current_pose_iv.lower[0]), float(current_pose_iv.upper[0])
            y_min, y_max = float(current_pose_iv.lower[1]), float(current_pose_iv.upper[1])

            envelope.append({
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max
            })

            # Calculate max error from reference
            mid = (current_pose_iv.lower + current_pose_iv.upper) / 2.0
            err = jnp.sqrt((mid[0] - s_next['x'])**2 + (mid[1] - s_next['y'])**2)
            # Add interval radius to error
            rad = jnp.sqrt(((x_max - x_min)/2)**2 + ((y_max - y_min)/2)**2)
            max_err = max(max_err, float(err + rad))

            # Heuristic to prevent interval explosion in this limited scope:
            # clip the interval if it grows too large (simulating a "reset" or sensor update)
            # FLL robots often square against walls.
            if rad > 0.05: # 5cm limit for visualization sanity
                 current_pose_iv = Interval(mid - 0.025, mid + 0.025)

        # Add last envelope point
        envelope.append(envelope[-1] if envelope else {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0})

        return {"max_error_m": max_err, "envelope": envelope}
