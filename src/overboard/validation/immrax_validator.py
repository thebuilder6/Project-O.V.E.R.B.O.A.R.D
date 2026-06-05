import jax
import jax.numpy as jnp
from jax import jit, vmap
import immrax
from immrax import Interval, interval
from ..core.jax_robot_model import JAXRobotConfig, JAXDifferentialDriveModel
from ..core.jax_ramsete import JAXRamseteController
from typing import List, Dict, Any, Tuple, Optional
import math

def unroll_trajectory_headings(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not samples: return []
    unrolled = []
    current_offset = 0.0
    last_h = samples[0]['heading']
    
    for s in samples:
        h = s['heading']
        if h - last_h > math.pi:
            current_offset -= 2 * math.pi
        elif h - last_h < -math.pi:
            current_offset += 2 * math.pi
            
        new_s = s.copy()
        new_s['heading'] = h + current_offset
        unrolled.append(new_s)
        last_h = h
        
    return unrolled

# ==============================================================================
# 1. THE OPEN LOOP SYSTEM (Physical Plant Dynamics)
# ==============================================================================
class RobotOpenLoop(immrax.system.OpenLoopSystem):
    def __init__(self, config: JAXRobotConfig):
        self.evolution = 'continuous'
        self.xlen = 3
        self.config = config

    def f(self, t, x, u, w):
        """
        u: Control inputs (Commanded wheel velocities) [vl_cmd, vr_cmd]
        """
        vl_cmd, vr_cmd = u[0], u[1]
        track_width, wheel_radius, backlash, headroom = w[3], w[4], w[2], w[1]

        # 1. Smooth Backlash
        epsilon = 1e-3  
        vl_act = vl_cmd - (backlash / 2.0) * jnp.tanh(vl_cmd / epsilon)
        vr_act = vr_cmd - (backlash / 2.0) * jnp.tanh(vr_cmd / epsilon)

        # 2. Enforce physical constraints with smooth saturation
        limit = self.config.v_max_rad_s * wheel_radius
        vl_sat = limit * jnp.tanh(vl_act / limit)
        vr_sat = limit * jnp.tanh(vr_act / limit)

        # 3. Kinematics
        v_act = (vl_sat + vr_sat) / 2.0
        omega_act = (vr_sat - vl_sat) / track_width

        dx = v_act * jnp.cos(x[2])
        dy = v_act * jnp.sin(x[2])
        dtheta = omega_act

        return jnp.array([dx, dy, dtheta])

# ==============================================================================
# 2. THE CONTROLLER (Control Strategy)
# ==============================================================================
class RamseteController:
    def __init__(self, ramsete_b=2.0, ramsete_zeta=0.7):
        self.controller = JAXRamseteController(ramsete_b, ramsete_zeta)

    def __call__(self, t, x, u_ref, w):
        """
        u_ref: Reference trajectory [xr, yr, thetar, vr, omegar]
        """
        xr, yr, thetar, vr, omegar = u_ref[0], u_ref[1], u_ref[2], u_ref[3], u_ref[4]
        track_width = w[3]
        
        ref_pose = jnp.array([xr, yr, thetar])
        v_cmd, omega_cmd = self.controller.calculate(x, ref_pose, vr, omegar)

        vl_cmd = v_cmd - (omega_cmd * track_width) / 2.0
        vr_cmd = v_cmd + (omega_cmd * track_width) / 2.0

        return jnp.array([vl_cmd, vr_cmd])

# ==============================================================================
# 3. THE CLOSED LOOP SYSTEM
# ==============================================================================
class ClosedLoopRobotSystem(immrax.System):
    def __init__(self, config: JAXRobotConfig, ramsete_b: float = 2.0, ramsete_zeta: float = 0.7):
        self.evolution = 'continuous'
        self.xlen = 3                 
        self.plant = RobotOpenLoop(config)
        self.controller = RamseteController(ramsete_b, ramsete_zeta)

    def f(self, t, x, u, w):
        # 1. Controller computes commands
        u_cmd = self.controller(t, x, u, w)
        # 2. Plant applies commands to physical dynamics
        return self.plant.f(t, x, u_cmd, w)

class ImmraxValidator:
    def __init__(self, config: JAXRobotConfig):
        self.cfg = config
        self.model = JAXDifferentialDriveModel(config)
        self.sys = ClosedLoopRobotSystem(config)
        self.emb_sys = immrax.natemb(self.sys)

    def validate_trajectory(self, samples: List[Dict[str, Any]],
                            cof_range: Tuple[float, float],
                            torque_margin_range: Tuple[float, float],
                            backlash_range: Tuple[float, float],
                            track_width_range: Optional[Tuple[float, float]] = None,
                            wheel_radius_range: Optional[Tuple[float, float]] = None,
                            inertia_range: Optional[Tuple[float, float]] = None,
                            safety_threshold_m: float = 0.005) -> Dict[str, Any]:
        vl = jnp.array([s['vl'] for s in samples])
        vr = jnp.array([s['vr'] for s in samples])
        al = jnp.array([s['al'] for s in samples])
        ar = jnp.array([s['ar'] for s in samples])

        cof_interval = Interval(*cof_range)
        torque_margin_interval = Interval(*torque_margin_range)
        
        tw_range = track_width_range if track_width_range else (self.cfg.track_width * 0.99, self.cfg.track_width * 1.01)
        wr_range = wheel_radius_range if wheel_radius_range else (self.cfg.wheel_radius * 0.99, self.cfg.wheel_radius * 1.01)
        ine_range = inertia_range if inertia_range else (self.cfg.inertia * 0.90, self.cfg.inertia * 1.10)

        track_width_iv = Interval(*tw_range)
        wheel_radius_iv = Interval(*wr_range)
        inertia_iv = Interval(*ine_range)

        def check_sample_interval(vl_val, vr_val, al_val, ar_val):
            # 1. Interval Dynamics
            a_val = (al_val + ar_val) / 2.0
            alpha_val = interval(ar_val - al_val) / track_width_iv
            f_total = self.cfg.mass * a_val
            m_total = inertia_iv * alpha_val

            fr = (interval(f_total) + (m_total * interval(2.0) / track_width_iv)) / interval(2.0)
            fl = interval(f_total) - fr

            # 2. Interval-Aware Motor Torque Curves
            omega_l = (interval(jnp.abs(vl_val)) / wheel_radius_iv) * interval(self.cfg.gearing)
            torque_l = (interval(1.0) - omega_l / interval(self.cfg.v_max_rad_s)) * interval(self.cfg.t_max_nm)
            torque_l = Interval(jnp.maximum(0.0, torque_l.lower), jnp.maximum(0.0, torque_l.upper))
            force_limit_l = (torque_l / wheel_radius_iv) * interval(self.cfg.gearing) * torque_margin_interval

            omega_r = (interval(jnp.abs(vr_val)) / wheel_radius_iv) * interval(self.cfg.gearing)
            torque_r = (interval(1.0) - omega_r / interval(self.cfg.v_max_rad_s)) * interval(self.cfg.t_max_nm)
            torque_r = Interval(jnp.maximum(0.0, torque_r.lower), jnp.maximum(0.0, torque_r.upper))
            force_limit_r = (torque_r / wheel_radius_iv) * interval(self.cfg.gearing) * torque_margin_interval

            def iv_abs_max(iv):
                return jnp.maximum(jnp.abs(iv.lower), jnp.abs(iv.upper))

            motor_violation_l = jnp.maximum(0.0, iv_abs_max(fl) - force_limit_l.lower)
            motor_violation_r = jnp.maximum(0.0, iv_abs_max(fr) - force_limit_r.lower)

            # 3. Dynamic Normal Force Transfers
            h_cg = interval(0.05)
            base_normal = interval((self.cfg.mass * self.cfg.g) / 2.0)
            longitudinal_transfer = (interval(self.cfg.mass) * interval(a_val) * h_cg) / track_width_iv
            
            v_val = (vl_val + vr_val) / 2.0
            omega_val = interval(vr_val - vl_val) / track_width_iv
            centripetal_accel = interval(v_val) * omega_val
            lateral_transfer = (interval(self.cfg.mass) * h_cg * centripetal_accel) / track_width_iv
            
            nl = base_normal - longitudinal_transfer - lateral_transfer
            nr = base_normal - longitudinal_transfer + lateral_transfer

            # 4. Global Friction Circle & Local Wheel Slip Violations
            traction_max = cof_interval.lower * self.cfg.mass * self.cfg.g
            traction_violation = jnp.maximum(0.0, iv_abs_max(fl) + iv_abs_max(fr) - traction_max)

            slip_l = jnp.maximum(0.0, iv_abs_max(fl) - (cof_interval * nl).lower)
            slip_r = jnp.maximum(0.0, iv_abs_max(fr) - (cof_interval * nr).lower)

            return motor_violation_l, motor_violation_r, traction_violation, slip_l, slip_r

        motor_vios_l, motor_vios_r, traction_vios, slip_ls, slip_rs = vmap(check_sample_interval)(vl, vr, al, ar)

        unrolled_samples = unroll_trajectory_headings(samples)
        reach_report = self.compute_reachability(
            unrolled_samples, backlash_range, tw_range, wr_range, cof_range, torque_margin_range, ine_range
        )

        max_motor_vio = jnp.maximum(jnp.max(motor_vios_l), jnp.max(motor_vios_r))
        is_physically_safe = (
            max_motor_vio < 1e-3 and 
            jnp.max(slip_ls) < 1e-3 and 
            jnp.max(slip_rs) < 1e-3 and 
            reach_report['max_error_m'] <= safety_threshold_m
        )

        return {
            "max_motor_violation_N": float(max_motor_vio),
            "max_traction_violation_N": float(jnp.max(traction_vios)),
            "max_slip_l_N": float(jnp.max(slip_ls)),
            "max_slip_r_N": float(jnp.max(slip_rs)),
            "max_tracking_error_m": reach_report['max_error_m'],
            "reachability": reach_report,
            "passed": bool(is_physically_safe)
        }

    def compute_reachability(self, samples: List[Dict[str, Any]], 
                             backlash_range: Tuple[float, float],
                             track_width_range: Tuple[float, float],
                             wheel_radius_range: Tuple[float, float],
                             cof_range: Tuple[float, float],
                             torque_margin_range: Tuple[float, float],
                             inertia_range: Tuple[float, float]) -> Dict[str, Any]:
        N = len(samples)
        if N < 2: return {"max_error_m": 0.0, "envelope": []}

        # Create FLAT array for uncertainty 'w' (lower bounds concatenated with upper bounds)
        w_lower = jnp.array([cof_range[0], torque_margin_range[0], backlash_range[0], track_width_range[0], wheel_radius_range[0], inertia_range[0]])
        w_upper = jnp.array([cof_range[1], torque_margin_range[1], backlash_range[1], track_width_range[1], wheel_radius_range[1], inertia_range[1]])
        w_flat = jnp.concatenate([w_lower, w_upper])

        nominal_inputs = []
        dts = []
        for i in range(N - 1):
            s = samples[i]
            s_next = samples[i+1]
            dt = s_next['t'] - s['t']
            dts.append(dt)
            
            ref_v = (s['vl'] + s['vr']) / 2.0
            ref_omega = (s['vr'] - s['vl']) / self.cfg.track_width
            nominal_inputs.append([s['x'], s['y'], s['heading'], ref_v, ref_omega])
            
        nominal_inputs = jnp.array(nominal_inputs)
        dts = jnp.array(dts)

        # Create FLAT array for state 'x' (lower bounds concatenated with upper bounds)
        start_pose = jnp.array([samples[0]['x'], samples[0]['y'], samples[0]['heading']])
        lower_pose = start_pose - 0.001
        upper_pose = start_pose + 0.001
        x_flat = jnp.concatenate([lower_pose, upper_pose])

        envelope = []
        max_err = 0.0

        for i in range(N - 1):
            u_k = nominal_inputs[i]
            dt = dts[i]
            s_next = samples[i+1]

            # Evaluate the compiled embedding system using the FLAT arrays
            dx_dt_flat = self.emb_sys.f(0.0, x_flat, u_k, w_flat)
            x_flat = x_flat + dx_dt_flat * dt

            # Slicing: [0, 1, 2] are lower bounds, [3, 4, 5] are upper bounds
            x_min, x_max = float(x_flat[0]), float(x_flat[3])
            y_min, y_max = float(x_flat[1]), float(x_flat[4])

            envelope.append({
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max
            })

            mid_x = (x_flat[0] + x_flat[3]) / 2.0
            mid_y = (x_flat[1] + x_flat[4]) / 2.0
            err = jnp.sqrt((mid_x - s_next['x'])**2 + (mid_y - s_next['y'])**2)
            
            rad = jnp.sqrt(((x_max - x_min)/2.0)**2 + ((y_max - y_min)/2.0)**2)
            max_err = max(max_err, float(err + rad))

            # Rescale pose interval bounds if they exceed 5cm
            if rad > 0.05:
                mid_theta = (x_flat[2] + x_flat[5]) / 2.0
                # Clamp spatial error to 2.5cm and heading error to 0.03 rad (gyro scale)
                new_lower = jnp.array([mid_x - 0.025, mid_y - 0.025, mid_theta - 0.03])
                new_upper = jnp.array([mid_x + 0.025, mid_y + 0.025, mid_theta + 0.03])
                x_flat = jnp.concatenate([new_lower, new_upper])

        envelope.append(envelope[-1] if envelope else {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0})

        return {"max_error_m": max_err, "envelope": envelope}

    def get_safety_ribbon_data(self, samples: List[Dict[str, Any]],
                               cof_range: Tuple[float, float],
                               torque_margin_range: Tuple[float, float],
                               backlash_range: Tuple[float, float],
                               track_width_range: Optional[Tuple[float, float]] = None,
                               wheel_radius_range: Optional[Tuple[float, float]] = None,
                               inertia_range: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        unrolled = unroll_trajectory_headings(samples)
        
        tw_range = track_width_range if track_width_range else (self.cfg.track_width * 0.99, self.cfg.track_width * 1.01)
        wr_range = wheel_radius_range if wheel_radius_range else (self.cfg.wheel_radius * 0.99, self.cfg.wheel_radius * 1.01)
        ine_range = inertia_range if inertia_range else (self.cfg.inertia * 0.90, self.cfg.inertia * 1.10)

        report = self.compute_reachability(
            unrolled, backlash_range, tw_range, wr_range, cof_range, torque_margin_range, ine_range
        )
        
        ribbon_data = []
        envelope = report["envelope"]
        
        for i, s in enumerate(samples):
            env_point = envelope[i] if i < len(envelope) else envelope[-1]
            
            ribbon_data.append({
                "t": float(s["t"]),
                "x": float(s["x"]),
                "y": float(s["y"]),
                "heading": float(s["heading"]),
                "x_min": float(env_point["x_min"]),
                "x_max": float(env_point["x_max"]),
                "y_min": float(env_point["y_min"]),
                "y_max": float(env_point["y_max"])
            })
            
        return ribbon_data
