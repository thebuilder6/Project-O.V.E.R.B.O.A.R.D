from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import casadi as ca
from robot_model import RobotConfig, DifferentialDriveModel
from live_visualizer import get_visualizer

# Distinct deterministic initial guesses for candidate generation. They are
# only guesses: IPOPT still enforces every waypoint, stop, and force limit.
GUESS_STYLES = ("baseline", "rolling", "bow_left", "bow_right", "turn_first")

# Gauss-Legendre nodes and weights on [-1, 1] for the self-consistent
# collocation. Four nodes integrate one interval's velocity profile to
# O(dt^7); the trapezoid rule, which samples only the two endpoints, is
# O(dt^3) and leaves the plan off the path its own profile traces.
_GAUSS_LEGENDRE_4 = (
    (-0.8611363115940526, 0.3478548451374538),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538),
)



class TrajectoryOptimizer:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.model = DifferentialDriveModel(config)
        self.iteration_history: List[Dict[str, Any]] = []  # Store convergence data

    def solve(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10, accuracy_weight: float = 0.0, stop_waypoint_indices: Optional[List[int]] = None, waypoint_events: Optional[Dict[int, str]] = None, apply_headroom: bool = True, verbose: bool = True, capture_iterations: bool = False, live_viz: bool = False, guess_style: str = "baseline", per_segment_dt: bool = False, initial_guess: Optional[np.ndarray] = None, fast_polish: bool = False, polish_iter_budget: int = 200, solver_options: Optional[Dict[str, Any]] = None, self_consistent: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
        guess_style: initial-guess seed, one of GUESS_STYLES. Different seeds
                     explore different local optima; the constraints are unchanged.
        per_segment_dt: if True, each waypoint segment gets its own sample
                        interval. If False (default), all segments share one
                        interval, preserving the validated uniform-time behavior.
        initial_guess: optional [dt..., states...] array (the layout
                        _build_initial_guess returns) to start IPOPT from.
                        None means build the guess from guess_style.
        fast_polish: if True, use the fast solve path. The single pass asks
                   IPOPT to treat the initial point as a previous iterate,
                   which on a large mission makes it crawl through thousands
                   of iterations for an answer a cold pass reaches in a
                   fraction of the time. The fast path gives the warm pass an
                   iteration budget, then finishes from that point with a
                   cold pass. A mission that solves within the budget returns
                   exactly the single-pass result, and a mission where the
                   cold finish fails falls back to the unbudgeted pass, so
                   the fast path never loses a mission that solves today.
        polish_iter_budget: iteration budget for the warm pass in the fast
                   path. Missions that need more iterations than this are
                   finished with a cold pass instead.
        self_consistent: if True, place the samples on the path the
                    trajectory's own (v, omega) profile traces, by
                    integrating that profile with a Gauss-Legendre rule
                    instead of the trapezoid rule. The plan is then
                    self-consistent by construction: the samples lie on the
                    path and the headings are its tangents. Off by default
                    so the measured plans stay reproducible.
        solver_options: extra IPOPT options that override the defaults.
        """
        num_segments = len(waypoints) - 1
        N = num_segments * num_samples_per_segment + 1

        if stop_waypoint_indices is None:
            stop_waypoint_indices = []
        if waypoint_events is None:
            waypoint_events = {}

        common = dict(
            num_samples_per_segment=num_samples_per_segment,
            accuracy_weight=accuracy_weight,
            stop_waypoint_indices=stop_waypoint_indices,
            waypoint_events=waypoint_events,
            apply_headroom=apply_headroom,
            verbose=verbose,
            capture_iterations=capture_iterations,
            live_viz=live_viz,
            self_consistent=self_consistent,
            guess_style=guess_style,
            per_segment_dt=per_segment_dt,
        )

        if not fast_polish:
            return self._solve_once(waypoints, initial_guess=initial_guess,
                                    warm_start=True, stage_label=None,
                                    solver_options=solver_options, **common)

        # Fast path.
        #
        # 1. The warm pass with an iteration budget. A mission that converges
        #    inside the budget returns exactly what the single pass returns.
        # 2. If the budget ran out, finish from that point with a cold pass.
        #    The cold pass starts on the warm pass's own trajectory, so it
        #    refines the same solution instead of searching a new basin.
        # 3. If the cold finish does not converge, run the unbudgeted warm
        #    pass, so a mission that solves today still solves.
        budget = max(1, int(polish_iter_budget))
        warm_options = dict(solver_options or {})
        warm_options["max_iter"] = budget
        stage_a_samples, stage_a_stats = self._solve_once(
            waypoints, initial_guess=initial_guess, warm_start=True,
            stage_label="1/2", solver_options=warm_options, **common)

        if stage_a_stats.get("converged", False):
            stats = dict(stage_a_stats)
            stats["fast_polish"] = True
            stats["stage_count"] = 1
            stats["stage_times"] = [stage_a_stats.get("total_time")]
            stats["stage_iterations"] = [stage_a_stats.get("iter_count")]
            stats["stage_costs"] = [stage_a_stats.get("final_cost")]
            return stage_a_samples, stats

        guess_b = self.guess_from_samples(
            stage_a_samples, num_segments, num_samples_per_segment)
        stage_b_samples, stage_b_stats = self._solve_once(
            waypoints, initial_guess=guess_b, warm_start=False,
            stage_label="2/2", solver_options=solver_options, **common)

        if stage_b_stats.get("converged", False):
            samples, stats = stage_b_samples, stage_b_stats
            stage_stats = [stage_a_stats, stage_b_stats]
        else:
            # The cold finish failed. Fall back to the unbudgeted warm pass.
            samples, stats = self._solve_once(
                waypoints, initial_guess=initial_guess, warm_start=True,
                stage_label=None, solver_options=solver_options, **common)
            stage_stats = [stage_a_stats, stage_b_stats, stats]

        stats = dict(stats)
        stats["fast_polish"] = True
        stats["stage_count"] = len(stage_stats)
        stats["stage_times"] = [s.get("total_time") for s in stage_stats]
        stats["stage_iterations"] = [s.get("iter_count") for s in stage_stats]
        stats["stage_costs"] = [s.get("final_cost") for s in stage_stats]
        stats["total_time"] = sum(float(s.get("total_time", 0.0))
                                  for s in stage_stats)
        return samples, stats

    @staticmethod
    def guess_from_samples(samples: List[Dict[str, Any]], num_segments: int,
                           num_samples_per_segment: int) -> np.ndarray:
        """Rebuild a [dt..., states...] guess array from formatted samples.

        The layout matches _build_initial_guess: one sample interval per
        segment, then the flattened (x, y, heading, vl, vr) states. Used to
        restart IPOPT from a previous pass.
        """
        count = len(samples)
        if count == 0:
            raise ValueError("Cannot build a guess from an empty trajectory")
        times = np.array([float(s["t"]) for s in samples], dtype=float)
        states = np.array([[s["x"], s["y"], s["heading"], s["vl"], s["vr"]]
                           for s in samples], dtype=float)
        dts = np.zeros(num_segments, dtype=float)
        for i in range(num_segments):
            k = min(i * num_samples_per_segment, count - 2)
            dts[i] = max(times[k + 1] - times[k], 1e-3)
        return np.concatenate([dts, states.flatten()])

    def _solve_once(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int = 10, accuracy_weight: float = 0.0, stop_waypoint_indices: Optional[List[int]] = None, waypoint_events: Optional[Dict[int, str]] = None, apply_headroom: bool = True, verbose: bool = True, capture_iterations: bool = False, live_viz: bool = False, guess_style: str = "baseline", per_segment_dt: bool = False, initial_guess: Optional[np.ndarray] = None, warm_start: bool = True, stage_label: Optional[str] = None, solver_options: Optional[Dict[str, Any]] = None, self_consistent: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run one IPOPT pass over the trajectory NLP.

        warm_start=True asks IPOPT to treat the initial point as a previous
        iterate. warm_start=False is a plain cold start, which converges in far
        fewer iterations when no earlier iterate exists.
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

        # Decision variables. With per-segment timing, dt[i] is the constant
        # sample interval inside waypoint segment i; every segment has exactly
        # num_samples_per_segment intervals, so time is s * sum(dt). The default
        # keeps the original scalar dt so the validated solve is unchanged.
        if per_segment_dt:
            dt = opti.variable(num_segments)
            time_cost = num_samples_per_segment * ca.sum1(dt)
        else:
            dt = opti.variable()
            time_cost = dt * (N - 1)
        X = opti.variable(N, 5)  # x, y, theta, vl, vr

        def interval_dt(k):
            """Interval length for the interval starting at sample k."""
            return dt[k // num_samples_per_segment] if per_segment_dt else dt

        # Unpack columns for readability
        x = X[:, 0]
        y = X[:, 1]
        theta = X[:, 2]
        vl = X[:, 3]
        vr = X[:, 4]

        # Objective: minimize total time + smoothness penalty.
        #
        # The penalty is on the wheel jerk J = (a[k+1] - a[k]) / dt, where a
        # is the interval-average acceleration, charged with a path-wide
        # reference interval so every boundary costs the same per unit of
        # physical jerk.
        #
        # Without that normalisation the raw acceleration jump (a[k+1]-a[k])
        # is proportional to J*dt, so a segment with a short dt is charged
        # dt^2 less per unit jerk and can buy violent wheel motion almost for
        # free. That let the per-segment model reach 68.9 m/s^3 of wheel jerk
        # where the uniform model reached 9.5 m/s^3 at the same weight
        # (KI-022).
        #
        # The reference interval is the mean interval over the whole path, so
        # when every dt is equal the scale is exactly 1 and the per-segment
        # objective reduces to the uniform one. The per-segment model is then
        # a strict generalisation of the uniform model, not a different
        # objective. The uniform branch keeps its original expression so the
        # validated uniform results stay bit-identical.
        smoothness_cost = 0
        if accuracy_weight > 0 and N > 2:
            if per_segment_dt:
                # time_cost is samples_per_segment * sum(dt) over N-1
                # intervals, so this is the mean interval over the path.
                dt_ref = time_cost / (N - 1)
            for k in range(N - 2):
                al_k = (vl[k + 1] - vl[k]) / interval_dt(k)
                al_k1 = (vl[k + 2] - vl[k + 1]) / interval_dt(k + 1)
                ar_k = (vr[k + 1] - vr[k]) / interval_dt(k)
                ar_k1 = (vr[k + 2] - vr[k + 1]) / interval_dt(k + 1)
                if per_segment_dt:
                    # Time between the midpoints of the two intervals: the
                    # acceleration jump happens over that span. Scaling by
                    # (dt_ref/dt_span)^2 makes the term dt_ref^2 * J^2, so
                    # every boundary pays the same per unit of physical jerk.
                    dt_span = 0.5 * (interval_dt(k) + interval_dt(k + 1))
                    scale = dt_ref / dt_span
                    smoothness_cost += ((al_k1 - al_k)**2
                                        + (ar_k1 - ar_k)**2) * scale**2
                else:
                    smoothness_cost += (al_k1 - al_k)**2 + (ar_k1 - ar_k)**2
        opti.minimize(time_cost + accuracy_weight * smoothness_cost)

        # Bounds on the interval length(s)
        if per_segment_dt:
            for i in range(num_segments):
                opti.subject_to(dt[i] >= 0.001)
                opti.subject_to(dt[i] <= 1.0)
        else:
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
            dt_k = interval_dt(k)
            v1 = (vl[k] + vr[k]) / 2.0
            v2 = (vl[k + 1] + vr[k + 1]) / 2.0
            omega1 = (vr[k] - vl[k]) / self.config.track_width
            omega2 = (vr[k + 1] - vl[k + 1]) / self.config.track_width

            # Kinematic integration
            if self_consistent:
                # Place the samples on the path the trajectory's own
                # (v, omega) profile traces, by integrating that profile
                # instead of sampling only the two endpoints. The heading
                # along the interval is the one the robot will hold: the
                # turn rate integrated, theta_k + omega1*s + domega*s^2/2dt.
                # With that, the samples lie on the path by construction and
                # the heading samples are its tangents.
                dx = 0.0
                dy = 0.0
                d_omega = omega2 - omega1
                for xi, weight in _GAUSS_LEGENDRE_4:
                    s = 0.5 * dt_k * (1.0 + xi)
                    w_s = 0.5 * dt_k * weight
                    v_s = v1 + (v2 - v1) * s / dt_k
                    theta_s = (theta[k] + omega1 * s
                               + d_omega * s * s / (2.0 * dt_k))
                    dx = dx + w_s * v_s * ca.cos(theta_s)
                    dy = dy + w_s * v_s * ca.sin(theta_s)
                opti.subject_to(x[k + 1] == x[k] + dx)
                opti.subject_to(y[k + 1] == y[k] + dy)
            else:
                opti.subject_to(
                    x[k + 1]
                    == x[k]
                    + 0.5 * (v1 * ca.cos(theta[k]) + v2 * ca.cos(theta[k + 1])) * dt_k
                )
                opti.subject_to(
                    y[k + 1]
                    == y[k]
                    + 0.5 * (v1 * ca.sin(theta[k]) + v2 * ca.sin(theta[k + 1])) * dt_k
                )
            opti.subject_to(
                theta[k + 1] == theta[k] + 0.5 * (omega1 + omega2) * dt_k
            )

            # Accelerations across interval [k, k + 1]
            al = (vl[k + 1] - vl[k]) / dt_k
            ar = (vr[k + 1] - vr[k]) / dt_k

            # Worst-case wheel force over the declared inertia/track-width
            # ranges. The optimizer must bound the same quantity Immrax will
            # check, not the nominal value, or a trajectory can pass here and
            # slip or over-torque there.
            f_bound = self.model.get_max_wheel_force_symbolic(al, ar, apply_headroom)

            # Per-wheel torque-speed limits. With headroom enabled, use the
            # declared worst case (KI-009/KI-010) so the optimizer and the
            # Immrax validator agree on the force available at each wheel.
            if apply_headroom:
                max_fl = self.config.get_max_force_symbolic_robust(vl[k])
                max_fr = self.config.get_max_force_symbolic_robust(vr[k])
            else:
                max_fl = self.config.get_max_force_symbolic(vl[k], apply_headroom=False)
                max_fr = self.config.get_max_force_symbolic(vr[k], apply_headroom=False)

            opti.subject_to(f_bound <= max_fl)
            opti.subject_to(f_bound <= max_fr)

            # Per-wheel traction limit with lateral weight transfer (KI-010),
            # matching the validator's per-wheel slip check.
            f_traction_max = self.model.get_traction_limit_symbolic(vl[k], vr[k], apply_headroom)
            opti.subject_to(f_bound <= f_traction_max)

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

        # Initial guess (seeded by guess_style; baseline is the old linear/zero-speed guess).
        # An explicit guess (e.g. a previous pass's solution) wins over the seed.
        if initial_guess is not None:
            guess = np.asarray(initial_guess, dtype=float)
            if guess.shape[0] != num_segments + N * 5:
                raise ValueError(
                    f"Initial guess has {guess.shape[0]} entries; expected "
                    f"{num_segments + N * 5} ({num_segments} intervals + "
                    f"{N} states x 5)"
                )
        else:
            guess = self._build_initial_guess(
                waypoints, num_samples_per_segment, N,
                style=guess_style, stop_waypoint_indices=stop_waypoint_indices)
        guess_dt = guess[:num_segments]
        if per_segment_dt:
            opti.set_initial(dt, guess_dt)
        else:
            opti.set_initial(dt, float(guess_dt[0]))
        guess_states = guess[num_segments:].reshape((N, 5))
        opti.set_initial(X, guess_states)
        
        # Capture initial guess for convergence visualization
        if capture_iterations:
            initial_cost = self._compute_cost(guess, N, num_samples_per_segment, accuracy_weight, waypoints)
            self.iteration_history.append({
                'iteration': 0,
                'cost': initial_cost,
                'trajectory': guess_states.copy(),
                'dt': np.asarray(guess_dt, dtype=float).tolist(),
                'phase': 'initial_guess'
            })

        # Solver setup
        # print_time=False silences the IPOPT statistics table CasADi prints
        # on every solve; it belongs with the solver internals.
        p_opts = {"expand": True, "print_time": bool(verbose)}
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
            "mu_strategy": "adaptive",             # Massively speeds up Phase 2
        }
        if warm_start:
            # Treat the initial point as a previous iterate. This is what a
            # polish pass wants, but it is also what makes IPOPT crawl through
            # thousands of iterations when there is no earlier iterate.
            s_opts["warm_start_init_point"] = "yes"
            s_opts["warm_start_bound_push"] = 1e-6  # Prevent boundary sticking
        else:
            s_opts["warm_start_init_point"] = "no"
        if solver_options:
            # Caller overrides win over the defaults above.
            s_opts.update(solver_options)
        opti.solver("ipopt", p_opts, s_opts)

        if live_viz:
            viz = get_visualizer()
            def callback(iteration):
                try:
                    # Get current state from debug
                    X_curr = np.array(opti.debug.value(X))
                    if per_segment_dt:
                        dt_curr = np.asarray(opti.debug.value(dt), dtype=float).reshape(-1)
                    else:
                        dt_curr = np.full(num_segments, float(opti.debug.value(dt)))
                    times = self._sample_times(dt_curr, num_samples_per_segment, N)
                    # Convert to list of dicts for JSON serialization
                    traj_data = []
                    for k in range(N):
                        vl_k = float(X_curr[k, 3])
                        vr_k = float(X_curr[k, 4])

                        # Rough force/torque estimate for live viz
                        fl, fr = 0.0, 0.0
                        if k < N - 1:
                            dt_k = dt_curr[k // num_samples_per_segment]
                            al = (float(X_curr[k+1, 3]) - vl_k) / dt_k
                            ar = (float(X_curr[k+1, 4]) - vr_k) / dt_k
                            fl, fr = self.model.get_dynamics(vl_k, vr_k, al, ar)

                        traj_data.append({
                            "t": float(times[k]),
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

        stage_prefix = f"[stage {stage_label}] " if stage_label else ""
        try:
            sol = opti.solve()
            if per_segment_dt:
                dt_val = np.asarray(sol.value(dt), dtype=float).reshape(-1)
            else:
                dt_val = np.full(num_segments, float(sol.value(dt)))
            X_val = np.array(sol.value(X))
            params = np.concatenate([dt_val, X_val.flatten()])
            total_time = float(num_samples_per_segment * np.sum(dt_val))
            
            # Capture final solution for convergence visualization
            if capture_iterations:
                final_cost = self._compute_cost(params, N, num_samples_per_segment, accuracy_weight, waypoints)
                self.iteration_history.append({
                    'iteration': len(self.iteration_history),
                    'cost': final_cost,
                    'trajectory': X_val.copy(),
                    'dt': dt_val.tolist(),
                    'phase': 'final_solution'
                })
            
            if verbose:
                print(f"{stage_prefix}Optimization converged. Total time: {total_time:.4f}s")
            
            stats = sol.stats()
            stats["total_time"] = time.time() - start_time
            stats["initial_cost"] = self._compute_cost(guess, N, num_samples_per_segment, accuracy_weight, waypoints)
            stats["final_cost"] = total_time
            stats["segment_dt"] = dt_val.tolist()
            stats["converged"] = True
            
            return self.format_output(params, N, num_samples_per_segment, waypoint_events), stats
        except Exception as e:
            if verbose:
                if "Maximum_Iterations_Exceeded" in str(e):
                    print(f"{stage_prefix}Optimization stopped at the iteration budget.")
                else:
                    print(f"{stage_prefix}Optimization failed or timed out:", e)
            # Return best-effort from debug values
            if per_segment_dt:
                dt_val = np.asarray(opti.debug.value(dt), dtype=float).reshape(-1)
            else:
                dt_val = np.full(num_segments, float(opti.debug.value(dt)))
            X_val = np.array(opti.debug.value(X))
            params = np.concatenate([dt_val, X_val.flatten()])
            total_time = float(num_samples_per_segment * np.sum(dt_val))
            
            # Capture failed solution for convergence visualization
            if capture_iterations:
                failed_cost = self._compute_cost(params, N, num_samples_per_segment, accuracy_weight, waypoints)
                self.iteration_history.append({
                    'iteration': len(self.iteration_history),
                    'cost': failed_cost,
                    'trajectory': X_val.copy(),
                    'dt': dt_val.tolist(),
                    'phase': 'failed_solution'
                })
            
            stats = {
                "total_time": time.time() - start_time,
                "initial_cost": self._compute_cost(guess, N, num_samples_per_segment, accuracy_weight, waypoints),
                "final_cost": total_time,
                "segment_dt": dt_val.tolist(),
                "converged": False,
                "error": str(e)
            }
            return self.format_output(params, N, num_samples_per_segment, waypoint_events), stats

    def _compute_cost(self, params: np.ndarray, N: int, num_samples_per_segment: int, accuracy_weight: float, waypoints: List[Tuple[float, float, Optional[float]]]) -> float:
        """Compute cost function value for a given trajectory."""
        num_segments = N // num_samples_per_segment
        dt_values = np.asarray(params[:num_segments], dtype=float)
        states = params[num_segments:].reshape((N, 5))
        
        # Time cost: every segment has exactly num_samples_per_segment intervals
        time_cost = float(num_samples_per_segment * np.sum(dt_values))
        
        # Smoothness cost. Mirrors the solver's objective: the wheel jerk
        # charged with a path-wide reference interval, so the per-segment
        # model is held to the same jerk discipline as the uniform one
        # (KI-022). With all dt equal the scale is exactly 1.
        smoothness_cost = 0
        if accuracy_weight > 0 and N > 2:
            dt_ref = time_cost / (N - 1)
            for k in range(N - 2):
                dt_k = dt_values[k // num_samples_per_segment]
                dt_k1 = dt_values[(k + 1) // num_samples_per_segment]
                al_k = (states[k + 1, 3] - states[k, 3]) / dt_k
                al_k1 = (states[k + 2, 3] - states[k + 1, 3]) / dt_k1
                ar_k = (states[k + 1, 4] - states[k, 4]) / dt_k
                ar_k1 = (states[k + 2, 4] - states[k + 1, 4]) / dt_k1
                dt_span = 0.5 * (dt_k + dt_k1)
                scale = dt_ref / dt_span
                smoothness_cost += ((al_k1 - al_k)**2 + (ar_k1 - ar_k)**2) * scale**2
        
        return time_cost + accuracy_weight * smoothness_cost


    def _build_initial_guess(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: int, N: int, style: str = "baseline", stop_waypoint_indices: Optional[List[int]] = None) -> np.ndarray:
        """Build the IPOPT starting point for a given seed style.

        baseline: the original linear/heading interpolation with zero wheel speeds.
        rolling: baseline geometry, with wheel speeds implied by the sampled motion.
        bow_left/bow_right: interior samples pushed to one side of each segment.
        turn_first: rotate at the start of heading-change segments, then drive.

        Every style keeps the waypoint samples on the waypoints, so the guess never
        moves a constrained waypoint. The solver still enforces all constraints.
        """
        if style not in GUESS_STYLES:
            raise ValueError(f"Unknown guess_style {style!r}; expected one of {GUESS_STYLES}")

        num_segments = len(waypoints) - 1
        initial_dt = 0.1
        states = []
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
                x, y, theta = self._guess_point(waypoints, i, p1, p2, frac)
                states.append([x, y, theta, 0.0, 0.0])

        states = np.array(states, dtype=float)
        if states.shape != (N, 5):
            raise ValueError(f"Initial guess shape {states.shape} does not match {(N, 5)}")

        if style == "bow_left":
            self._offset_segment_interiors(states, waypoints, num_samples_per_segment, left=True)
        elif style == "bow_right":
            self._offset_segment_interiors(states, waypoints, num_samples_per_segment, left=False)
        elif style == "turn_first":
            self._rotate_before_driving(states, waypoints, num_samples_per_segment)

        if style != "baseline":
            self._set_rolling_speeds(states, initial_dt, stop_waypoint_indices, num_samples_per_segment)

        # One initial interval length per segment, then the flattened states.
        guess = np.concatenate([
            np.full(num_segments, initial_dt, dtype=float),
            states.flatten(),
        ])
        return guess

    @staticmethod
    def _guess_point(waypoints, i, p1, p2, frac):
        """Position and heading for one sample of the baseline guess."""
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
        return float(x), float(y), float(theta)

    def _offset_segment_interiors(self, states, waypoints, num_samples_per_segment, left=True):
        """Push interior samples sideways so IPOPT starts in an arc-shaped basin."""
        s = num_samples_per_segment
        num_segments = len(waypoints) - 1
        sign = 1.0 if left else -1.0
        for i in range(num_segments):
            p1, p2 = waypoints[i], waypoints[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = float(np.hypot(dx, dy))
            if length < 1e-9:
                continue
            # Unit left normal and a bounded bow (10% of the segment, max 3 cm).
            nx, ny = -dy / length, dx / length
            amplitude = min(0.10 * length, 0.03)
            count = s if i < num_segments - 1 else s + 1
            base = i * s
            for j in range(count):
                frac = j / s
                bump = 4.0 * frac * (1.0 - frac)  # zero at both waypoints
                states[base + j, 0] += sign * nx * amplitude * bump
                states[base + j, 1] += sign * ny * amplitude * bump

    def _rotate_before_driving(self, states, waypoints, num_samples_per_segment):
        """Guess turn-in-place first, then a straight drive, for sharp turns."""
        s = num_samples_per_segment
        if s <= 1:
            # One interval cannot hold both a spin and a drive; keep the
            # baseline geometry rather than move a constrained waypoint.
            return
        num_segments = len(waypoints) - 1
        spin_samples = max(1, int(round(0.30 * s)))
        for i in range(num_segments):
            p1, p2 = waypoints[i], waypoints[i + 1]
            base = i * s
            theta_start = float(states[base, 2])
            theta_end = float(states[base + s, 2])
            delta = (theta_end - theta_start + np.pi) % (2 * np.pi) - np.pi
            if abs(delta) < 0.4:
                continue
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            count = s if i < num_segments - 1 else s + 1
            for j in range(count):
                k = base + j
                if j <= spin_samples:
                    f = j / spin_samples
                    states[k, 0] = p1[0]
                    states[k, 1] = p1[1]
                    states[k, 2] = theta_start + delta * f
                else:
                    f = (j - spin_samples) / (s - spin_samples)
                    f = min(max(f, 0.0), 1.0)
                    states[k, 0] = p1[0] + dx * f
                    states[k, 1] = p1[1] + dy * f
                    states[k, 2] = theta_end

    def _set_rolling_speeds(self, states, dt, stop_waypoint_indices, num_samples_per_segment):
        """Infer wheel speeds from the guessed motion so IPOPT starts moving."""
        n = len(states)
        v = np.zeros(n)
        omega = np.zeros(n)
        for k in range(n - 1):
            dx = states[k + 1, 0] - states[k, 0]
            dy = states[k + 1, 1] - states[k, 1]
            dtheta = (states[k + 1, 2] - states[k, 2] + np.pi) % (2 * np.pi) - np.pi
            v[k] = float(np.hypot(dx, dy) / dt)
            omega[k] = float(dtheta / dt)

        half_track = self.config.track_width / 2.0
        cap = float(self.config.max_linear_speed(apply_headroom=True))
        for k in range(n):
            states[k, 3] = float(np.clip(v[k] - omega[k] * half_track, -cap, cap))
            states[k, 4] = float(np.clip(v[k] + omega[k] * half_track, -cap, cap))

        # The solver requires rest at the endpoints and at declared stops.
        states[0, 3] = states[0, 4] = 0.0
        states[-1, 3] = states[-1, 4] = 0.0
        for index in (stop_waypoint_indices or []):
            k = int(index) * num_samples_per_segment
            if 0 <= k < n:
                states[k, 3] = states[k, 4] = 0.0

    @staticmethod
    def _sample_times(dt_values: np.ndarray, num_samples_per_segment: int, N: int) -> np.ndarray:
        """Timestamps when each segment has a constant sample interval dt[i]."""
        dt_values = np.asarray(dt_values, dtype=float)
        s = int(num_samples_per_segment)
        # k // s is the segment; clamp the final sample to the last segment.
        seg = np.minimum(np.arange(N) // s, len(dt_values) - 1)
        starts = np.concatenate(([0.0], np.cumsum(dt_values * s)))
        return starts[seg] + (np.arange(N) - seg * s) * dt_values[seg]

    def format_output(self, params: np.ndarray, N: int, num_samples_per_segment: int = 10, waypoint_events: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        num_segments = N // num_samples_per_segment
        dt_values = np.asarray(params[:num_segments], dtype=float)
        states = params[num_segments:].reshape((N, 5))
        times = self._sample_times(dt_values, num_samples_per_segment, N)
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
                dt_k = dt_values[k // num_samples_per_segment]
                vl_next, vr_next = states[k + 1][3], states[k + 1][4]
                al, ar = (vl_next - vl) / dt_k, (vr_next - vr) / dt_k
            else:
                al, ar = 0.0, 0.0

            fl, fr = self.model.get_dynamics(vl, vr, al, ar)
            sample_dict = {
                "t": float(times[k]),
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
