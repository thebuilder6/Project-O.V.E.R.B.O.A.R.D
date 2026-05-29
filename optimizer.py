from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import casadi as ca
from robot_model import RobotConfig, DifferentialDriveModel
from live_visualizer import get_visualizer

class TrajectoryOptimizer:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.model = DifferentialDriveModel(config)
        self.iteration_history: List[Dict[str, Any]] = []  # Store convergence data

    def solve(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10, accuracy_weight: float = 0.0, stop_waypoint_indices: Optional[List[int]] = None, waypoint_events: Optional[Dict[int, str]] = None, apply_headroom: bool = True, verbose: bool = True, capture_iterations: bool = False, live_viz: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        waypoints: list of (x, y, heading)
        heading is in radians. None means unconstrained.
        accuracy_weight: weight for smoothness/jerk penalty (0 = pure time-optimal).
        stop_waypoint_indices: list of waypoint indices where robot must come to rest (vl=vr=0).
                             None means only start and end at rest.
        waypoint_events: dict mapping waypoint indices to event names (e.g., {2: "lower_arm", 5: "release"}).
        apply_headroom: If True, applies safety margin for real-world tracking.
        verbose: If True, prints progress messages.
        capture_iterations: If True, captures intermediate solver states for convergence visualization.
        live_viz: If True, streams live updates to the WebSocket visualizer.
        """
        num_segments = len(waypoints) - 1
        N = num_segments * num_samples_per_segment + 1

        if stop_waypoint_indices is None:
            stop_waypoint_indices = []
        if waypoint_events is None:
            waypoint_events = {}
        
        import time
        start_time = time.time()

        # Clear previous iteration history
        self.iteration_history = []

        opti = ca.Opti()

        # Decision variables
        dt = opti.variable()
        X = opti.variable(N, 5)  # x, y, theta, vl, vr

        # Unpack columns for readability
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
        opti.minimize(time_cost + accuracy_weight * smoothness_cost)

        # Bounds on dt
        opti.subject_to(dt >= 0.001)
        opti.subject_to(dt <= 1.0)

        # Bounds on states
        opti.subject_to(x >= -20)
        opti.subject_to(x <= 20)
        opti.subject_to(y >= -20)
        opti.subject_to(y <= 20)
        opti.subject_to(theta >= -100)
        opti.subject_to(theta <= 100)

        # Wheel-speed bounds derived from motor no-load speed
        v_max = self.config.max_linear_speed(apply_headroom)
        # Use 99 % of no-load speed to avoid the exact zero-force singularity
        v_bound = 0.99 * v_max
        opti.subject_to(vl >= -v_bound)
        opti.subject_to(vl <= v_bound)
        opti.subject_to(vr >= -v_bound)
        opti.subject_to(vr <= v_bound)

        # Trapezoidal collocation & physical constraints
        for k in range(N - 1):
            v1 = (vl[k] + vr[k]) / 2.0
            v2 = (vl[k + 1] + vr[k + 1]) / 2.0
            omega1 = (vr[k] - vl[k]) / self.config.track_width
            omega2 = (vr[k + 1] - vl[k + 1]) / self.config.track_width

            # Kinematic integration
            opti.subject_to(
                x[k + 1]
                == x[k]
                + 0.5 * (v1 * ca.cos(theta[k]) + v2 * ca.cos(theta[k + 1])) * dt
            )
            opti.subject_to(
                y[k + 1]
                == y[k]
                + 0.5 * (v1 * ca.sin(theta[k]) + v2 * ca.sin(theta[k + 1])) * dt
            )
            opti.subject_to(
                theta[k + 1] == theta[k] + 0.5 * (omega1 + omega2) * dt
            )

            # Accelerations across interval [k, k + 1]
            al = (vl[k + 1] - vl[k]) / dt
            ar = (vr[k + 1] - vr[k]) / dt

            # Force calculations (evaluated at start of interval to match legacy behaviour)
            fl, fr = self.model.get_dynamics_symbolic(vl[k], vr[k], al, ar)
            max_fl = self.config.get_max_force_symbolic(vl[k], apply_headroom)
            max_fr = self.config.get_max_force_symbolic(vr[k], apply_headroom)

            # Motor limits
            opti.subject_to(ca.fabs(fl) <= max_fl)
            opti.subject_to(ca.fabs(fr) <= max_fr)

            # Traction limit
            f_total = ca.fabs(fl) + ca.fabs(fr)
            f_traction_max = self.config.cof * self.config.mass * self.config.g
            opti.subject_to(f_total <= f_traction_max)

        # Waypoint constraints
        for i, wp in enumerate(waypoints):
            idx = i * num_samples_per_segment
            opti.subject_to(x[idx] == wp[0])
            opti.subject_to(y[idx] == wp[1])
            if wp[2] is not None:
                opti.subject_to(theta[idx] == wp[2])

        # Start and end at rest
        opti.subject_to(vl[0] == 0)
        opti.subject_to(vr[0] == 0)
        opti.subject_to(vl[N - 1] == 0)
        opti.subject_to(vr[N - 1] == 0)

        # Intermediate stop waypoints
        for stop_idx in stop_waypoint_indices:
            if stop_idx < 0 or stop_idx >= len(waypoints):
                continue
            sample_idx = stop_idx * num_samples_per_segment
            opti.subject_to(vl[sample_idx] == 0)
            opti.subject_to(vr[sample_idx] == 0)

        # Initial guess (linear interpolation between waypoints, zero wheelspeed)
        guess = self._build_initial_guess(waypoints, num_samples_per_segment, N)
        opti.set_initial(dt, float(guess[0]))
        guess_states = guess[1:].reshape((N, 5))
        opti.set_initial(X, guess_states)
        
        # Capture initial guess for convergence visualization
        if capture_iterations:
            initial_cost = self._compute_cost(guess, N, num_samples_per_segment, accuracy_weight, waypoints)
            self.iteration_history.append({
                'iteration': 0,
                'cost': initial_cost,
                'trajectory': guess_states.copy(),
                'dt': float(guess[0]),
                'phase': 'initial_guess'
            })

        # Solver setup
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 5000,
            "print_level": 0,
            "tol": 1e-2,
            "constr_viol_tol": 1e-2,
            "acceptable_tol": 1e-1,
            "acceptable_constr_viol_tol": 1e-1,
            "acceptable_iter": 5,
            "nlp_scaling_method": "gradient-based",
            "hessian_approximation": "limited-memory",
            "sb": "yes",  # Suppress Ipopt banner
        }
        opti.solver("ipopt", p_opts, s_opts)

        if live_viz:
            viz = get_visualizer()
            def callback(iteration):
                try:
                    # Get current state from debug
                    X_curr = np.array(opti.debug.value(X))
                    dt_curr = float(opti.debug.value(dt))
                    # Convert to list of dicts for JSON serialization
                    traj_data = []
                    for k in range(N):
                        vl_k = float(X_curr[k, 3])
                        vr_k = float(X_curr[k, 4])

                        # Rough force/torque estimate for live viz
                        fl, fr = 0.0, 0.0
                        if k < N - 1:
                            al = (float(X_curr[k+1, 3]) - vl_k) / dt_curr
                            ar = (float(X_curr[k+1, 4]) - vr_k) / dt_curr
                            fl, fr = self.model.get_dynamics(vl_k, vr_k, al, ar)

                        traj_data.append({
                            "t": k * dt_curr,
                            "x": float(X_curr[k, 0]),
                            "y": float(X_curr[k, 1]),
                            "heading": float(X_curr[k, 2]),
                            "vl": vl_k,
                            "vr": vr_k,
                            "fl": fl,
                            "fr": fr
                        })
                    viz.send_trajectory(traj_data, phase="global_solve", iteration=iteration)
                except Exception:
                    pass
            
            opti.callback(callback)

        try:
            sol = opti.solve()
            dt_val = float(sol.value(dt))
            X_val = np.array(sol.value(X))
            params = np.concatenate([[dt_val], X_val.flatten()])
            
            # Capture final solution for convergence visualization
            if capture_iterations:
                final_cost = self._compute_cost(params, N, num_samples_per_segment, accuracy_weight, waypoints)
                self.iteration_history.append({
                    'iteration': len(self.iteration_history),
                    'cost': final_cost,
                    'trajectory': X_val.copy(),
                    'dt': dt_val,
                    'phase': 'final_solution'
                })
            
            print("Optimization converged. Total time: {:.4f}s".format(dt_val * (N - 1)))
            
            stats = sol.stats()
            stats["total_time"] = time.time() - start_time
            stats["initial_cost"] = self._compute_cost(guess, N, num_samples_per_segment, accuracy_weight, waypoints)
            stats["final_cost"] = dt_val * (N - 1)
            stats["converged"] = True
            
            return self.format_output(params, N, num_samples_per_segment, waypoint_events), stats
        except Exception as e:
            print("Optimization failed or timed out:", e)
            # Return best-effort from debug values
            dt_val = float(opti.debug.value(dt))
            X_val = np.array(opti.debug.value(X))
            params = np.concatenate([[dt_val], X_val.flatten()])
            
            # Capture failed solution for convergence visualization
            if capture_iterations:
                failed_cost = self._compute_cost(params, N, num_samples_per_segment, accuracy_weight, waypoints)
                self.iteration_history.append({
                    'iteration': len(self.iteration_history),
                    'cost': failed_cost,
                    'trajectory': X_val.copy(),
                    'dt': dt_val,
                    'phase': 'failed_solution'
                })
            
            stats = {
                "total_time": time.time() - start_time,
                "initial_cost": self._compute_cost(guess, N, num_samples_per_segment, accuracy_weight, waypoints),
                "final_cost": dt_val * (N - 1),
                "converged": False,
                "error": str(e)
            }
            return self.format_output(params, N, num_samples_per_segment, waypoint_events), stats

    def _compute_cost(self, params: np.ndarray, N: int, num_samples_per_segment: int, accuracy_weight: float, waypoints: List[Tuple[float, float, Optional[float]]]) -> float:
        """Compute cost function value for a given trajectory."""
        dt = params[0]
        states = params[1:].reshape((N, 5))
        
        # Time cost
        time_cost = dt * (N - 1)
        
        # Smoothness cost
        smoothness_cost = 0
        if accuracy_weight > 0 and N > 2:
            for k in range(N - 2):
                al_k = (states[k + 1, 3] - states[k, 3]) / dt
                al_k1 = (states[k + 2, 3] - states[k + 1, 3]) / dt
                ar_k = (states[k + 1, 4] - states[k, 4]) / dt
                ar_k1 = (states[k + 2, 4] - states[k + 1, 4]) / dt
                smoothness_cost += (al_k1 - al_k)**2 + (ar_k1 - ar_k)**2
        
        return time_cost + accuracy_weight * smoothness_cost


    def _build_initial_guess(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int, N: int) -> np.ndarray:
        num_segments = len(waypoints) - 1
        initial_dt = 0.1
        guess = [initial_dt]
        for i in range(num_segments):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            count = (
                num_samples_per_segment
                if i < num_segments - 1
                else num_samples_per_segment + 1
            )
            for j in range(count):
                frac = j / num_samples_per_segment
                x = p1[0] + (p2[0] - p1[0]) * frac
                y = p1[1] + (p2[1] - p1[1]) * frac
                
                # Compute heading based on local path direction
                if p1[2] is not None and p2[2] is not None:
                    # Both headings known: interpolate
                    diff = (p2[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi
                    theta = p1[2] + diff * frac
                elif p1[2] is not None:
                    # Only start heading known: blend towards path direction
                    path_dir = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                    # Smoothly transition from constrained heading to path direction
                    diff = (path_dir - p1[2] + np.pi) % (2 * np.pi) - np.pi
                    theta = p1[2] + diff * frac
                elif p2[2] is not None:
                    # Only end heading known: blend from path direction
                    path_dir = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                    diff = (p2[2] - path_dir + np.pi) % (2 * np.pi) - np.pi
                    theta = path_dir + diff * frac
                else:
                    # No heading known: use local path direction
                    # Look ahead to next waypoint for better direction estimate
                    if i + 2 < len(waypoints):
                        p3 = waypoints[i + 2]
                        # Blend direction to p2 with direction to p3
                        dir_to_p2 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                        dir_to_p3 = np.arctan2(p3[1] - p1[1], p3[0] - p1[0])
                        # Weight by progress through segment
                        theta = (1 - frac) * dir_to_p2 + frac * dir_to_p3
                    else:
                        theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                
                guess.extend([x, y, theta, 0.0, 0.0])
        return np.array(guess)

    def format_output(self, params: np.ndarray, N: int, num_samples_per_segment: int = 10, waypoint_events: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        dt = params[0]
        states = params[1:].reshape((N, 5))
        samples = []
        
        for k in range(N):
            s = states[k]
            vl, vr = s[3], s[4]
            
            # Check if this sample corresponds to a waypoint with an event
            event = None
            if waypoint_events and k % num_samples_per_segment == 0:
                waypoint_idx = k // num_samples_per_segment
                if waypoint_idx in waypoint_events:
                    event = waypoint_events[waypoint_idx]
            if k < N - 1:
                vl_next, vr_next = states[k + 1][3], states[k + 1][4]
                al, ar = (vl_next - vl) / dt, (vr_next - vr) / dt
            else:
                al, ar = 0.0, 0.0

            fl, fr = self.model.get_dynamics(vl, vr, al, ar)
            sample_dict = {
                "t": float(k * dt),
                "x": float(s[0]),
                "y": float(s[1]),
                "heading": float(s[2]),
                "vl": float(vl),
                "vr": float(vr),
                "omega": float((vr - vl) / self.config.track_width),
                "al": float(al),
                "ar": float(ar),
                "fl": float(fl),
                "fr": float(fr),
            }
            if event is not None:
                sample_dict["event"] = event
            samples.append(sample_dict)
        
        return samples
