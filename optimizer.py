import math
import time
from typing import List, Dict, Tuple, Any, Optional, Union, Sequence
import numpy as np
import casadi as ca
from robot_model import RobotConfig, DifferentialDriveModel

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

    @staticmethod
    def get_segment_sample_counts(
        waypoints: Sequence[Sequence[Any]],
        num_samples_per_segment: Union[int, Sequence[int]] = 10,
        proportional: bool = True,
        min_samples: int = 3,
    ) -> List[int]:
        """Allocate sample counts across segments proportional to arc length (KI-024).

        When proportional=True and num_samples_per_segment is an integer, short
        and long segments do not receive the same fixed count. Each segment receives
        at least min_samples, while keeping the total sample budget (num_segments *
        num_samples_per_segment).
        """
        num_segments = len(waypoints) - 1
        if num_segments <= 0:
            return []
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            if len(num_samples_per_segment) != num_segments:
                raise ValueError(
                    f"Sample counts list has {len(num_samples_per_segment)} entries; "
                    f"expected {num_segments}"
                )
            return [int(c) for c in num_samples_per_segment]

        s_int = int(num_samples_per_segment)
        if not proportional or num_segments == 1:
            return [s_int] * num_segments

        lengths = []
        for i in range(num_segments):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            lengths.append(float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))

        total_length = sum(lengths)
        if total_length < 1e-6:
            return [s_int] * num_segments

        total_budget = num_segments * s_int
        m = min(s_int, max(1, min_samples))
        if total_budget < num_segments * m:
            return [s_int] * num_segments

        rem_budget = total_budget - num_segments * m
        raw_alloc = [rem_budget * (l / total_length) for l in lengths]
        int_alloc = [int(np.floor(a)) for a in raw_alloc]
        leftover = rem_budget - sum(int_alloc)

        remainders = [(raw_alloc[i] - int_alloc[i], i) for i in range(num_segments)]
        remainders.sort(key=lambda x: x[0], reverse=True)
        for _, idx in remainders[:leftover]:
            int_alloc[idx] += 1

        return [m + int_alloc[i] for i in range(num_segments)]

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
        per_segment_dt: bool = False,
        initial_guess: Optional[np.ndarray] = None,
        fast_polish: bool = False,
        polish_iter_budget: int = 200,
        solver_options: Optional[Dict[str, Any]] = None,
        self_consistent: bool = True,
        proportional_samples: bool = True,
        split_at_boundaries: bool = True,
        force_weight: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        waypoints: list of (x, y, heading)
        heading is in radians. None means unconstrained.
        accuracy_weight: weight for smoothness/jerk penalty (0 = pure time-optimal).
        force_weight: weight for soft wheel-force and traction-utilization penalty (0 = none).
        stop_waypoint_indices: list of waypoint indices where robot must come to rest (vl=vr=0).
                             None means only start and end at rest.
        waypoint_events: dict mapping waypoint indices to event names (e.g., {2: "lower_arm", 5: "release"}).
        apply_headroom: If True, applies safety margin for real-world tracking.
        verbose: If True, prints progress messages.
        capture_iterations: If True, captures intermediate solver states for convergence visualization.
        guess_style: initial-guess seed, one of GUESS_STYLES. Different seeds
                     explore different local optima; the constraints are unchanged.
        per_segment_dt: if True, each waypoint segment gets its own sample
                        interval. If False (default), all segments share one
                        interval, preserving the validated uniform-time behavior.
        initial_guess: optional [dt..., states...] array (the layout
                        _build_initial_guess returns) to start IPOPT from.
                        None means build the guess from guess_style.
        fast_polish: if True, use the fast solve path.
        polish_iter_budget: iteration budget for the warm pass in the fast path.
        solver_options: extra IPOPT options that override the defaults.
        self_consistent: if True (default), place the samples on the path the
                    trajectory's own (v, omega) profile traces, by
                    integrating that profile with a Gauss-Legendre rule.
        proportional_samples: if True (default), allocate sample counts
                    proportional to segment arc length (KI-024).
        split_at_boundaries: if True (default), split trajectories at waypoints
                    where stop=True and heading is not None (boundary stops)
                    into independent sub-missions and stitch solutions.
        """
        num_segments = len(waypoints) - 1

        if stop_waypoint_indices is None:
            stop_waypoint_indices = []
        if waypoint_events is None:
            waypoint_events = {}

        # Split trajectories at boundary stops (stop=True and heading is pinned)
        # into independent sub-missions and stitch (KI-024, RQ-A OQ5).
        conds = self.merge_conditions(waypoints, stop_waypoint_indices)
        split_indices = [c["index"] for c in conds if c["boundary"] and 0 < c["index"] < len(waypoints) - 1]

        if split_at_boundaries and split_indices and initial_guess is None:
            splits = [0] + split_indices + [len(waypoints) - 1]
            sub_results = []
            for m in range(len(splits) - 1):
                start_i = splits[m]
                end_i = splits[m + 1]
                sub_wps = waypoints[start_i : end_i + 1]
                sub_stops = [idx - start_i for idx in stop_waypoint_indices if start_i <= idx <= end_i]
                sub_events = {}
                for idx, ev in waypoint_events.items():
                    if m == 0 and start_i <= idx <= end_i:
                        sub_events[idx - start_i] = ev
                    elif m > 0 and start_i < idx <= end_i:
                        sub_events[idx - start_i] = ev

                if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
                    sub_samples_alloc = list(num_samples_per_segment[start_i:end_i])
                else:
                    sub_samples_alloc = num_samples_per_segment

                best_sub_samples = None
                best_sub_stats = None
                best_sub_hist = []
                styles_to_try = [guess_style] + [s for s in GUESS_STYLES if s != guess_style]
                for style in styles_to_try:
                    try:
                        self.iteration_history = []
                        s_samples, s_stats = self.solve(
                            sub_wps,
                            num_samples_per_segment=sub_samples_alloc,
                            accuracy_weight=accuracy_weight,
                            stop_waypoint_indices=sub_stops,
                            waypoint_events=sub_events,
                            apply_headroom=apply_headroom,
                            verbose=verbose and (m == 0),
                            capture_iterations=capture_iterations,
                            guess_style=style,
                            per_segment_dt=per_segment_dt,
                            fast_polish=fast_polish,
                            polish_iter_budget=polish_iter_budget,
                            solver_options=solver_options,
                            self_consistent=self_consistent,
                            proportional_samples=proportional_samples,
                            split_at_boundaries=False,
                            force_weight=force_weight,
                        )
                        if s_stats.get("converged", False):
                            best_sub_samples = s_samples
                            best_sub_stats = s_stats
                            best_sub_hist = list(self.iteration_history)
                            break
                        elif best_sub_samples is None:
                            best_sub_samples = s_samples
                            best_sub_stats = s_stats
                            best_sub_hist = list(self.iteration_history)
                    except Exception as e:
                        if best_sub_samples is None and style == styles_to_try[-1]:
                            raise e

                sub_results.append((best_sub_samples, best_sub_stats, best_sub_hist))

            stitched_samples = []
            stitched_dts = []
            stitched_samples_counts = []
            current_t = 0.0
            for m, (sub_samples, sub_stats, _) in enumerate(sub_results):
                if m == 0:
                    for s in sub_samples:
                        stitched_samples.append(dict(s))
                    current_t = sub_samples[-1]["t"]
                else:
                    for s in sub_samples[1:]:
                        s_copy = dict(s)
                        s_copy["t"] = float(s["t"]) + current_t
                        stitched_samples.append(s_copy)
                    current_t = stitched_samples[-1]["t"]
                stitched_dts.extend(sub_stats.get("segment_dt", []))
                stitched_samples_counts.extend(sub_stats.get("segment_samples", []))

            stats = {
                "total_time": current_t,
                "initial_cost": sum(s.get("initial_cost", 0.0) for _, s, _ in sub_results),
                "final_cost": sum(s.get("final_cost", 0.0) for _, s, _ in sub_results),
                "segment_dt": stitched_dts,
                "segment_samples": stitched_samples_counts,
                "converged": all(s.get("converged", False) for _, s, _ in sub_results),
                "sub_mission_count": len(splits) - 1,
            }
            if any(s.get("fast_polish", False) for _, s, _ in sub_results):
                stats["fast_polish"] = True
                stats["stage_count"] = sum(s.get("stage_count", 1) for _, s, _ in sub_results)
                stats["stage_times"] = [s.get("total_time", 0.0) for _, s, _ in sub_results]
                stage_iters = []
                for _, s, _ in sub_results:
                    if "stage_iterations" in s and s["stage_iterations"]:
                        stage_iters.extend(s["stage_iterations"])
                    elif "iter_count" in s:
                        stage_iters.append(s["iter_count"])
                stats["stage_iterations"] = stage_iters

            if capture_iterations and all(len(hist) >= 2 for _, _, hist in sub_results):
                stitched_guess_traj = sub_results[0][2][0]["trajectory"]
                stitched_final_traj = sub_results[0][2][-1]["trajectory"]
                for _, _, hist in sub_results[1:]:
                    stitched_guess_traj = np.vstack([stitched_guess_traj, hist[0]["trajectory"][1:]])
                    stitched_final_traj = np.vstack([stitched_final_traj, hist[-1]["trajectory"][1:]])
                self.iteration_history = [
                    {
                        "iteration": 0,
                        "cost": stats["initial_cost"],
                        "trajectory": stitched_guess_traj,
                        "dt": stats["segment_dt"],
                        "phase": "initial_guess",
                    },
                    {
                        "iteration": 1,
                        "cost": stats["final_cost"],
                        "trajectory": stitched_final_traj,
                        "dt": stats["segment_dt"],
                        "phase": "final_solution",
                    },
                ]
            else:
                self.iteration_history = []

            return stitched_samples, stats

        common = dict(
            num_samples_per_segment=num_samples_per_segment,
            accuracy_weight=accuracy_weight,
            stop_waypoint_indices=stop_waypoint_indices,
            waypoint_events=waypoint_events,
            apply_headroom=apply_headroom,
            verbose=verbose,
            capture_iterations=capture_iterations,
            guess_style=guess_style,
            per_segment_dt=per_segment_dt,
            self_consistent=self_consistent,
            proportional_samples=proportional_samples,
            force_weight=force_weight,
        )

        if not fast_polish:
            return self._solve_once(waypoints, initial_guess=initial_guess,
                                    warm_start=True, stage_label=None,
                                    solver_options=solver_options, **common)

        # Fast path
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

        stage_b_samples, stage_b_stats = self._solve_once(
            waypoints, initial_guess=initial_guess, warm_start=False,
            stage_label="2/2", solver_options=solver_options, **common)

        if stage_b_stats.get("converged", False):
            samples, stats = stage_b_samples, stage_b_stats
            stage_stats = [stage_a_stats, stage_b_stats]
        else:
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
    def merge_conditions(waypoints, stop_waypoint_indices=None):
        stops = set(stop_waypoint_indices or [])
        out = []
        for i, wp in enumerate(waypoints):
            is_stop = i in stops
            pinned = len(wp) > 2 and wp[2] is not None
            boundary = bool(is_stop and pinned)
            out.append({
                "index": i,
                "stop": bool(is_stop),
                "heading_pinned": bool(pinned),
                "boundary": boundary,
                "agree_velocity": not boundary,
                "agree_heading": not boundary,
                "continuous_force": bool(not is_stop),
            })
        return out

    @staticmethod
    def guess_from_samples(samples: List[Dict[str, Any]], num_segments: int,
                           num_samples_per_segment: Union[int, Sequence[int]]) -> np.ndarray:
        count = len(samples)
        if count == 0:
            raise ValueError("Cannot build a guess from an empty trajectory")
        times = np.array([float(s["t"]) for s in samples], dtype=float)
        states = np.array([[s["x"], s["y"], s["heading"], s["vl"], s["vr"]]
                           for s in samples], dtype=float)
        dts = np.zeros(num_segments, dtype=float)
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            seg_samples = [int(s) for s in num_samples_per_segment]
            waypoint_sample_indices = [0] + list(np.cumsum(seg_samples))
        else:
            waypoint_sample_indices = [i * int(num_samples_per_segment) for i in range(num_segments + 1)]

        for i in range(num_segments):
            k = min(waypoint_sample_indices[i], count - 2)
            dts[i] = max(times[k + 1] - times[k], 1e-3)
        return np.concatenate([dts, states.flatten()])

    def _solve_once(
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
        per_segment_dt: bool = False,
        initial_guess: Optional[np.ndarray] = None,
        warm_start: bool = True,
        stage_label: Optional[str] = None,
        solver_options: Optional[Dict[str, Any]] = None,
        self_consistent: bool = True,
        proportional_samples: bool = True,
        force_weight: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        num_segments = len(waypoints) - 1
        segment_samples = self.get_segment_sample_counts(
            waypoints, num_samples_per_segment, proportional=proportional_samples)
        N = sum(segment_samples) + 1
        waypoint_sample_indices = [0] + list(np.cumsum(segment_samples))
        sample_to_segment = np.repeat(np.arange(num_segments), segment_samples)

        if stop_waypoint_indices is None:
            stop_waypoint_indices = []
        if waypoint_events is None:
            waypoint_events = {}

        start_time = time.time()
        self.iteration_history = []

        opti = ca.Opti()

        if per_segment_dt:
            dt = opti.variable(num_segments)
            time_cost = ca.dot(ca.DM(segment_samples), dt)
        else:
            dt = opti.variable()
            time_cost = dt * (N - 1)

        X = opti.variable(N, 5)

        def interval_dt(k):
            return dt[sample_to_segment[k]] if per_segment_dt else dt

        x = X[:, 0]
        y = X[:, 1]
        theta = X[:, 2]
        vl = X[:, 3]
        vr = X[:, 4]

        merge_constraints = []
        if per_segment_dt:
            _conds = self.merge_conditions(waypoints, stop_waypoint_indices)
            for _j in range(1, num_segments):
                if not _conds[_j]["continuous_force"]:
                    continue
                _k = waypoint_sample_indices[_j]
                merge_constraints.append((
                    (vl[_k] - vl[_k - 1]) * dt[_j]
                    - (vl[_k + 1] - vl[_k]) * dt[_j - 1],
                    (vr[_k] - vr[_k - 1]) * dt[_j]
                    - (vr[_k + 1] - vr[_k]) * dt[_j - 1],
                ))

        smoothness_cost = 0
        if accuracy_weight > 0 and N > 2:
            if per_segment_dt:
                dt_ref = time_cost / (N - 1)
            for k in range(N - 2):
                al_k = (vl[k + 1] - vl[k]) / interval_dt(k)
                al_k1 = (vl[k + 2] - vl[k + 1]) / interval_dt(k + 1)
                ar_k = (vr[k + 1] - vr[k]) / interval_dt(k)
                ar_k1 = (vr[k + 2] - vr[k + 1]) / interval_dt(k + 1)
                if per_segment_dt:
                    dt_span = 0.5 * (interval_dt(k) + interval_dt(k + 1))
                    scale = dt_ref / dt_span
                    smoothness_cost += ((al_k1 - al_k)**2
                                        + (ar_k1 - ar_k)**2) * scale**2
                else:
                    smoothness_cost += (al_k1 - al_k)**2 + (ar_k1 - ar_k)**2

        if per_segment_dt:
            for i in range(num_segments):
                opti.subject_to(dt[i] >= 0.001)
                opti.subject_to(dt[i] <= 1.0)
        else:
            opti.subject_to(dt >= 0.001)
            opti.subject_to(dt <= 1.0)

        opti.subject_to(x >= -20)
        opti.subject_to(x <= 20)
        opti.subject_to(y >= -20)
        opti.subject_to(y <= 20)
        opti.subject_to(theta >= -100)
        opti.subject_to(theta <= 100)

        v_max = self.config.max_linear_speed(apply_headroom)
        v_bound = 0.99 * v_max
        opti.subject_to(vl >= -v_bound)
        opti.subject_to(vl <= v_bound)
        opti.subject_to(vr >= -v_bound)
        opti.subject_to(vr <= v_bound)

        force_cost = 0
        f_stall_ref = max(
            self.config.get_max_force_at_velocity_robust(0.0)
            if apply_headroom
            else self.config.get_max_force_at_velocity(0.0, apply_headroom=False),
            1e-6,
        )

        for k in range(N - 1):
            dt_k = interval_dt(k)
            v1 = (vl[k] + vr[k]) / 2.0
            v2 = (vl[k + 1] + vr[k + 1]) / 2.0
            omega1 = (vr[k] - vl[k]) / self.config.track_width
            omega2 = (vr[k + 1] - vl[k + 1]) / self.config.track_width

            if self_consistent:
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

            al = (vl[k + 1] - vl[k]) / dt_k
            ar = (vr[k + 1] - vr[k]) / dt_k

            f_bound = self.model.get_max_wheel_force_symbolic(al, ar, apply_headroom)

            if apply_headroom:
                max_fl = self.config.get_max_force_symbolic_robust(vl[k])
                max_fr = self.config.get_max_force_symbolic_robust(vr[k])
            else:
                max_fl = self.config.get_max_force_symbolic(vl[k], apply_headroom=False)
                max_fr = self.config.get_max_force_symbolic(vr[k], apply_headroom=False)

            opti.subject_to(f_bound <= max_fl)
            opti.subject_to(f_bound <= max_fr)

            f_traction_max = self.model.get_traction_limit_symbolic(vl[k], vr[k], apply_headroom)
            opti.subject_to(f_bound <= f_traction_max)

            if force_weight > 0:
                a_lin = (al + ar) / 2.0
                alpha_ang = (ar - al) / self.config.track_width
                f_lin = 0.5 * self.config.mass * a_lin
                f_ang = (self.config.inertia * alpha_ang) / self.config.track_width
                fl_k = f_lin - f_ang
                fr_k = f_lin + f_ang
                f_trac_ref = ca.fmax(f_traction_max, 1e-6)
                force_cost += (
                    0.5 * ((fl_k / f_stall_ref) ** 2 + (fr_k / f_stall_ref) ** 2)
                    + 0.5 * ((fl_k / f_trac_ref) ** 2 + (fr_k / f_trac_ref) ** 2)
                ) * dt_k

        opti.minimize(
            time_cost
            + accuracy_weight * smoothness_cost
            + force_weight * force_cost
        )

        for i, wp in enumerate(waypoints):
            idx = waypoint_sample_indices[i]
            opti.subject_to(x[idx] == wp[0])
            opti.subject_to(y[idx] == wp[1])
            if wp[2] is not None:
                opti.subject_to(theta[idx] == wp[2])

        opti.subject_to(vl[0] == 0)
        opti.subject_to(vr[0] == 0)
        opti.subject_to(vl[N - 1] == 0)
        opti.subject_to(vr[N - 1] == 0)

        for stop_idx in stop_waypoint_indices:
            if stop_idx < 0 or stop_idx >= len(waypoints):
                continue
            sample_idx = waypoint_sample_indices[stop_idx]
            opti.subject_to(vl[sample_idx] == 0)
            opti.subject_to(vr[sample_idx] == 0)

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
                waypoints, segment_samples, N,
                style=guess_style, stop_waypoint_indices=stop_waypoint_indices)
        guess_dt = guess[:num_segments]
        if per_segment_dt:
            opti.set_initial(dt, guess_dt)
        else:
            opti.set_initial(dt, float(np.mean(guess_dt)))
        guess_states = guess[num_segments:].reshape((N, 5))
        opti.set_initial(X, guess_states)

        if merge_constraints:
            dt_ref = float(np.mean(guess_dt)) or 0.1
            scale = 1.0 / (dt_ref * dt_ref)
            for _vl_res, _vr_res in merge_constraints:
                opti.subject_to(_vl_res * scale == 0)
                opti.subject_to(_vr_res * scale == 0)
        
        if capture_iterations:
            initial_cost = self._compute_cost(
                guess, N, segment_samples, accuracy_weight, waypoints, force_weight=force_weight)
            self.iteration_history.append({
                'iteration': 0,
                'cost': initial_cost,
                'trajectory': guess_states.copy(),
                'dt': np.asarray(guess_dt, dtype=float).tolist(),
                'phase': 'initial_guess'
            })

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
            "sb": "yes",
            "mu_strategy": "adaptive",
        }
        if warm_start:
            s_opts["warm_start_init_point"] = "yes"
            s_opts["warm_start_bound_push"] = 1e-6
        else:
            s_opts["warm_start_init_point"] = "no"
        if solver_options:
            s_opts.update(solver_options)
        opti.solver("ipopt", p_opts, s_opts)

        stage_prefix = f"[stage {stage_label}] " if stage_label else ""
        try:
            sol = opti.solve()
            if per_segment_dt:
                dt_val = np.asarray(sol.value(dt), dtype=float).reshape(-1)
            else:
                dt_val = np.full(num_segments, float(sol.value(dt)))
            X_val = np.array(sol.value(X))
            params = np.concatenate([dt_val, X_val.flatten()])
            total_time = float(np.sum(np.array(segment_samples) * dt_val))
            
            if capture_iterations:
                final_cost = self._compute_cost(
                    params, N, segment_samples, accuracy_weight, waypoints, force_weight=force_weight)
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
            stats["initial_cost"] = self._compute_cost(
                guess, N, segment_samples, accuracy_weight, waypoints, force_weight=force_weight)
            stats["final_cost"] = total_time
            stats["segment_dt"] = dt_val.tolist()
            stats["segment_samples"] = [int(s) for s in segment_samples]
            stats["converged"] = True
            
            return self.format_output(params, N, segment_samples, waypoint_events), stats
        except Exception as e:
            if verbose:
                if "Maximum_Iterations_Exceeded" in str(e):
                    print(f"{stage_prefix}Optimization stopped at the iteration budget.")
                else:
                    print(f"{stage_prefix}Optimization failed or timed out:", e)
            if per_segment_dt:
                dt_val = np.asarray(opti.debug.value(dt), dtype=float).reshape(-1)
            else:
                dt_val = np.full(num_segments, float(opti.debug.value(dt)))
            X_val = np.array(opti.debug.value(X))
            params = np.concatenate([dt_val, X_val.flatten()])
            total_time = float(np.sum(np.array(segment_samples) * dt_val))
            
            if capture_iterations:
                failed_cost = self._compute_cost(
                    params, N, segment_samples, accuracy_weight, waypoints, force_weight=force_weight)
                self.iteration_history.append({
                    'iteration': len(self.iteration_history),
                    'cost': failed_cost,
                    'trajectory': X_val.copy(),
                    'dt': dt_val.tolist(),
                    'phase': 'failed_solution'
                })
            
            stats = {
                "total_time": time.time() - start_time,
                "initial_cost": self._compute_cost(
                    guess, N, segment_samples, accuracy_weight, waypoints, force_weight=force_weight),
                "final_cost": total_time,
                "segment_dt": dt_val.tolist(),
                "segment_samples": [int(s) for s in segment_samples],
                "converged": False,
                "error": str(e)
            }
            if "Maximum_Iterations_Exceeded" in str(e):
                stats["iter_count"] = int(s_opts.get("max_iter", 0) or 0)
            return self.format_output(params, N, segment_samples, waypoint_events), stats

    def _compute_cost(
        self,
        params: np.ndarray,
        N: int,
        num_samples_per_segment: Union[int, Sequence[int]],
        accuracy_weight: float,
        waypoints: List[Tuple[float, float, Optional[float]]],
        force_weight: float = 0.0,
    ) -> float:
        if len(params) == 1 + N * 5:
            num_dt = 1
            num_segments = len(waypoints) - 1
            dt_values = np.asarray([params[0]] * num_segments, dtype=float)
            states = params[1:].reshape((N, 5))
        else:
            num_dt = len(params) - N * 5
            num_segments = num_dt
            dt_values = np.asarray(params[:num_dt], dtype=float)
            states = params[num_dt:].reshape((N, 5))

        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            segment_samples = [int(s) for s in num_samples_per_segment]
        else:
            segment_samples = [int(num_samples_per_segment)] * num_segments
        sample_to_segment = np.repeat(np.arange(num_segments), segment_samples)
        
        time_cost = float(np.sum(np.array(segment_samples) * dt_values))
        
        smoothness_cost = 0
        if accuracy_weight > 0 and N > 2:
            dt_ref = time_cost / (N - 1)
            for k in range(N - 2):
                dt_k = dt_values[sample_to_segment[k]]
                dt_k1 = dt_values[sample_to_segment[k + 1]]
                al_k = (states[k + 1, 3] - states[k, 3]) / dt_k
                al_k1 = (states[k + 2, 3] - states[k + 1, 3]) / dt_k1
                ar_k = (states[k + 1, 4] - states[k, 4]) / dt_k
                ar_k1 = (states[k + 2, 4] - states[k + 1, 4]) / dt_k1
                dt_span = 0.5 * (dt_k + dt_k1)
                scale = dt_ref / dt_span
                smoothness_cost += ((al_k1 - al_k)**2 + (ar_k1 - ar_k)**2) * scale**2

        force_cost = 0.0
        if force_weight > 0 and N > 1:
            f_stall_ref = max(self.config.get_max_force_at_velocity_robust(0.0), 1e-6)
            cof_min = min(self.config.uncertainties["cof"])
            for k in range(N - 1):
                dt_k = dt_values[sample_to_segment[k]]
                if dt_k <= 0:
                    continue
                vl_k = states[k, 3]
                vr_k = states[k, 4]
                al_k = (states[k + 1, 3] - vl_k) / dt_k
                ar_k = (states[k + 1, 4] - vr_k) / dt_k
                a_lin = 0.5 * (al_k + ar_k)
                alpha_ang = (ar_k - al_k) / self.config.track_width
                f_lin = 0.5 * self.config.mass * a_lin
                f_ang = (self.config.inertia * alpha_ang) / self.config.track_width
                fl_k = f_lin - f_ang
                fr_k = f_lin + f_ang
                nl_k, nr_k = self.model.get_wheel_normal_forces(vl_k, vr_k, al_k, ar_k)
                f_trac_ref = max(cof_min * min(nl_k, nr_k), 1e-6)
                force_cost += (
                    0.5 * ((fl_k / f_stall_ref) ** 2 + (fr_k / f_stall_ref) ** 2)
                    + 0.5 * ((fl_k / f_trac_ref) ** 2 + (fr_k / f_trac_ref) ** 2)
                ) * dt_k
        
        return time_cost + accuracy_weight * smoothness_cost + force_weight * force_cost

    def _build_initial_guess(self, waypoints: List[Tuple[float, float, Optional[float]]], num_samples_per_segment: Union[int, Sequence[int]], N: int, style: str = "baseline", stop_waypoint_indices: Optional[List[int]] = None) -> np.ndarray:
        if style not in GUESS_STYLES:
            raise ValueError(f"Unknown guess_style {style!r}; expected one of {GUESS_STYLES}")

        num_segments = len(waypoints) - 1
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            segment_samples = [int(s) for s in num_samples_per_segment]
        else:
            segment_samples = [int(num_samples_per_segment)] * num_segments

        lengths = []
        for i in range(num_segments):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            lengths.append(float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
        total_length = sum(lengths)

        # Initial dt: length-proportional (Q2 / F9)
        total_intervals = sum(segment_samples)
        total_time_budget = 0.1 * total_intervals
        initial_dts = np.zeros(num_segments, dtype=float)
        if total_length > 1e-6:
            for i in range(num_segments):
                seg_time = total_time_budget * (lengths[i] / total_length)
                initial_dts[i] = float(np.clip(seg_time / max(1, segment_samples[i]), 0.01, 1.0))
        else:
            initial_dts.fill(0.1)

        states = []
        for i in range(num_segments):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            s_i = segment_samples[i]
            count = s_i if i < num_segments - 1 else s_i + 1
            for j in range(count):
                frac = j / s_i
                x, y, theta = self._guess_point(waypoints, i, p1, p2, frac)
                states.append([x, y, theta, 0.0, 0.0])

        states = np.array(states, dtype=float)
        if states.shape != (N, 5):
            raise ValueError(f"Initial guess shape {states.shape} does not match {(N, 5)}")

        if style == "bow_left":
            self._offset_segment_interiors(states, waypoints, segment_samples, left=True)
        elif style == "bow_right":
            self._offset_segment_interiors(states, waypoints, segment_samples, left=False)
        elif style == "turn_first":
            self._rotate_before_driving(states, waypoints, segment_samples)

        if style != "baseline":
            self._set_rolling_speeds(states, initial_dts, stop_waypoint_indices, segment_samples)

        guess = np.concatenate([
            initial_dts,
            states.flatten(),
        ])
        return guess

    @staticmethod
    def _guess_point(waypoints, i, p1, p2, frac):
        x = p1[0] + (p2[0] - p1[0]) * frac
        y = p1[1] + (p2[1] - p1[1]) * frac

        if p1[2] is not None and p2[2] is not None:
            diff = (p2[2] - p1[2] + np.pi) % (2 * np.pi) - np.pi
            theta = p1[2] + diff * frac
        elif p1[2] is not None:
            path_dir = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            diff = (path_dir - p1[2] + np.pi) % (2 * np.pi) - np.pi
            theta = p1[2] + diff * frac
        elif p2[2] is not None:
            path_dir = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            diff = (p2[2] - path_dir + np.pi) % (2 * np.pi) - np.pi
            theta = path_dir + diff * frac
        else:
            if i + 2 < len(waypoints):
                p3 = waypoints[i + 2]
                dir_to_p2 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                dir_to_p3 = np.arctan2(p3[1] - p1[1], p3[0] - p1[0])
                delta = (dir_to_p3 - dir_to_p2 + np.pi) % (2 * np.pi) - np.pi
                theta = dir_to_p2 + delta * frac
            else:
                theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        return float(x), float(y), float(theta)

    def _offset_segment_interiors(self, states, waypoints, num_samples_per_segment: Union[int, Sequence[int]], left=True):
        num_segments = len(waypoints) - 1
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            segment_samples = [int(s) for s in num_samples_per_segment]
        else:
            segment_samples = [int(num_samples_per_segment)] * num_segments

        waypoint_sample_indices = [0] + list(np.cumsum(segment_samples))
        sign = 1.0 if left else -1.0
        for i in range(num_segments):
            p1, p2 = waypoints[i], waypoints[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = float(np.hypot(dx, dy))
            if length < 1e-9:
                continue
            nx, ny = -dy / length, dx / length
            s_i = segment_samples[i]
            base = waypoint_sample_indices[i]

            theta_start = float(states[base, 2])
            theta_end = float(states[base + s_i, 2])
            delta = (theta_end - theta_start + np.pi) % (2 * np.pi) - np.pi
            if abs(delta) > 1e-4:
                sagitta = 0.5 * length * math.tan(abs(delta) / 4.0)
                amplitude = min(sagitta, 0.35 * length)
            else:
                amplitude = min(0.10 * length, 0.03)

            count = s_i if i < num_segments - 1 else s_i + 1
            for j in range(count):
                frac = j / s_i
                bump = 4.0 * frac * (1.0 - frac)
                states[base + j, 0] += sign * nx * amplitude * bump
                states[base + j, 1] += sign * ny * amplitude * bump

    def _rotate_before_driving(self, states, waypoints, num_samples_per_segment: Union[int, Sequence[int]]):
        num_segments = len(waypoints) - 1
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            segment_samples = [int(s) for s in num_samples_per_segment]
        else:
            segment_samples = [int(num_samples_per_segment)] * num_segments

        waypoint_sample_indices = [0] + list(np.cumsum(segment_samples))
        r_min = getattr(self.config, "track_width", 0.14)

        for i in range(num_segments):
            s_i = segment_samples[i]
            if s_i <= 1:
                continue
            base = waypoint_sample_indices[i]
            p1, p2 = waypoints[i], waypoints[i + 1]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = float(np.hypot(dx, dy))

            theta_start = float(states[base, 2])
            theta_end = float(states[base + s_i, 2])
            delta = (theta_end - theta_start + np.pi) % (2 * np.pi) - np.pi

            ratio = length / (2.0 * max(1e-3, r_min))
            if ratio < 1.0:
                threshold = max(0.2, 2.0 * math.asin(ratio))
            else:
                threshold = 0.4

            if abs(delta) < threshold:
                continue

            spin_samples = max(1, int(round(0.30 * s_i)))
            count = s_i if i < num_segments - 1 else s_i + 1
            for j in range(count):
                k = base + j
                if j <= spin_samples:
                    f = j / spin_samples
                    states[k, 0] = p1[0]
                    states[k, 1] = p1[1]
                    states[k, 2] = theta_start + delta * f
                else:
                    f = (j - spin_samples) / (s_i - spin_samples)
                    f = min(max(f, 0.0), 1.0)
                    states[k, 0] = p1[0] + dx * f
                    states[k, 1] = p1[1] + dy * f
                    states[k, 2] = theta_end

    def _set_rolling_speeds(self, states, dt, stop_waypoint_indices, num_samples_per_segment: Union[int, Sequence[int]]):
        n = len(states)
        v = np.zeros(n)
        omega = np.zeros(n)
        
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            segment_samples = [int(s) for s in num_samples_per_segment]
            num_segments = len(segment_samples)
            sample_to_segment = np.repeat(np.arange(num_segments), segment_samples)
            waypoint_sample_indices = [0] + list(np.cumsum(segment_samples))
        else:
            s_int = int(num_samples_per_segment)
            num_segments = (n - 1) // s_int if s_int > 0 else 1
            segment_samples = [s_int] * num_segments
            sample_to_segment = np.repeat(np.arange(num_segments), segment_samples)
            waypoint_sample_indices = [i * s_int for i in range(num_segments + 1)]

        is_dt_array = isinstance(dt, (list, tuple, np.ndarray))

        for k in range(n - 1):
            dx = states[k + 1, 0] - states[k, 0]
            dy = states[k + 1, 1] - states[k, 1]
            dtheta = (states[k + 1, 2] - states[k, 2] + np.pi) % (2 * np.pi) - np.pi
            dt_k = float(dt[sample_to_segment[k]]) if is_dt_array else float(dt)
            dt_k = max(dt_k, 1e-4)
            v[k] = float(np.hypot(dx, dy) / dt_k)
            omega[k] = float(dtheta / dt_k)

        half_track = self.config.track_width / 2.0
        cap = float(self.config.max_linear_speed(apply_headroom=True))
        for k in range(n):
            states[k, 3] = float(np.clip(v[k] - omega[k] * half_track, -cap, cap))
            states[k, 4] = float(np.clip(v[k] + omega[k] * half_track, -cap, cap))

        states[0, 3] = states[0, 4] = 0.0
        states[-1, 3] = states[-1, 4] = 0.0
        for index in (stop_waypoint_indices or []):
            if 0 <= index < len(waypoint_sample_indices):
                k = waypoint_sample_indices[index]
                if 0 <= k < n:
                    states[k, 3] = states[k, 4] = 0.0

    @staticmethod
    def _sample_times(dt_values: np.ndarray, num_samples_per_segment: Union[int, Sequence[int]], N: int) -> np.ndarray:
        dt_values = np.asarray(dt_values, dtype=float)
        num_segments = len(dt_values)
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            seg_samples = [int(s) for s in num_samples_per_segment]
        else:
            seg_samples = [int(num_samples_per_segment)] * num_segments
        
        sample_to_segment = np.repeat(np.arange(num_segments), seg_samples)
        segment_durations = [seg_samples[i] * dt_values[i] for i in range(num_segments)]
        starts = np.concatenate(([0.0], np.cumsum(segment_durations)))
        waypoint_sample_indices = [0] + list(np.cumsum(seg_samples))
        
        times = np.zeros(N)
        for k in range(N - 1):
            seg = sample_to_segment[k]
            offset = k - waypoint_sample_indices[seg]
            times[k] = starts[seg] + offset * dt_values[seg]
        times[N - 1] = starts[-1]
        return times

    def format_output(self, params: np.ndarray, N: int, num_samples_per_segment: Union[int, Sequence[int]] = 10, waypoint_events: Optional[Dict[int, str]] = None) -> List[Dict[str, Any]]:
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            segment_samples = [int(s) for s in num_samples_per_segment]
            num_segments = len(segment_samples)
        else:
            num_segments = (N - 1) // int(num_samples_per_segment) if N > 1 else 1
            segment_samples = [int(num_samples_per_segment)] * num_segments

        dt_values = np.asarray(params[:num_segments], dtype=float)
        states = params[num_segments:].reshape((N, 5))
        times = self._sample_times(dt_values, segment_samples, N)
        samples = []
        waypoint_sample_indices = [0] + list(np.cumsum(segment_samples))
        sample_to_segment = np.repeat(np.arange(num_segments), segment_samples)

        event_by_sample = {}
        if waypoint_events:
            for wp_idx, evt_name in waypoint_events.items():
                if 0 <= wp_idx < len(waypoint_sample_indices):
                    event_by_sample[waypoint_sample_indices[wp_idx]] = evt_name

        for k in range(N):
            s = states[k]
            vl, vr = s[3], s[4]
            event = event_by_sample.get(k, None)

            if k < N - 1:
                dt_k = dt_values[sample_to_segment[k]]
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
