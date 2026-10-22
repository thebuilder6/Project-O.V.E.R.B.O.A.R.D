import threading
import time
from typing import List, Dict, Tuple, Any, Optional, Union, Sequence
import numpy as np
import casadi as ca
from overboard.models.robot_model import RobotConfig, DifferentialDriveModel, CG_HEIGHT_M, wheel_normal_forces, diff_drive_fk, diff_drive_wheel_forces

# The canonical guess styles live in the seed provider; re-export them here so
# the optimizer and its callers keep one list.
from overboard.core.frontier_seed import GUESS_STYLES

# MUMPS inside CasADi/IPOPT is not re-entrant across OS threads on Windows;
# serialize the C++ MUMPS call to guarantee thread-safe solves repository-wide.
CASADI_IPOPT_LOCK = threading.RLock()

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


def _waypoint_heading(waypoint):
    """The declared heading of a waypoint, or None when it is not pinned."""
    if isinstance(waypoint, dict):
        return waypoint.get("heading")
    return waypoint[2] if len(waypoint) > 2 else None


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
        num_samples_per_segment) - except that a segment which turns by more than
        about 20 degrees is raised to the full per-segment count. An arc-length
        split starves a short turn (11 samples on a 90-degree 0.3 m segment, where
        the controller follows chords), and that shows up as tracking error; the
        raised count cost 0.6 cm of envelope and 0.02 s on the stops example.
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

        counts = [m + int_alloc[i] for i in range(num_segments)]
        # A short turn is starved by the arc-length split. The controller
        # follows the chord between samples, so give a turning segment the
        # full count (KI-024, RQ-A); it costs a little envelope, not time.
        for i in range(num_segments):
            h1 = _waypoint_heading(waypoints[i])
            h2 = _waypoint_heading(waypoints[i + 1])
            if h1 is None or h2 is None:
                continue
            turn = abs((h2 - h1 + np.pi) % (2.0 * np.pi) - np.pi)
            if turn > 0.35:
                counts[i] = max(counts[i], s_int)
        return counts

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
        proportional_samples: bool = True,
        split_at_boundaries: bool = True,
        force_weight: float = 0.0,
        cost_metric: str = "jerk",
        epigraph_constraints: bool = False,
        sample_weights: Optional[Sequence[float]] = None,
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
                            proportional_samples=proportional_samples,
                            split_at_boundaries=False,
                            force_weight=force_weight,
                            cost_metric=cost_metric,
                            epigraph_constraints=epigraph_constraints,
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
                # The per-iteration costs of every sub-solve, in order. They
                # carry no trajectory, so they ride between the two stitched
                # ends: the ghost path and the final plan read the ends, and
                # the convergence curve keeps the shape of the whole solve.
                curve = []
                for _, _, hist in sub_results:
                    for entry in hist:
                        if entry.get("phase") == "ipopt":
                            curve.append({"iteration": len(curve) + 1,
                                          "cost": float(entry["cost"]),
                                          "phase": "ipopt"})
                self.iteration_history = [
                    {
                        "iteration": 0,
                        "cost": stats["initial_cost"],
                        "trajectory": stitched_guess_traj,
                        "dt": stats["segment_dt"],
                        "phase": "initial_guess",
                    },
                    *curve,
                    {
                        "iteration": len(curve) + 1,
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
            proportional_samples=proportional_samples,
            force_weight=force_weight,
            cost_metric=cost_metric,
            epigraph_constraints=epigraph_constraints,
            sample_weights=sample_weights,
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

        # The cold finish starts from the warm stage's iterate, not from the
        # original guess. The budgeted warm stage has usually found the good
        # basin already (on the complex mission it is at 6.08 s when the 200
        # iterations run out); restarting cold from the original guess lands
        # in a worse local optimum (8.67 s) even though it converges quickly.
        # Cold-starting from the half-finished iterate keeps KI-047's lesson
        # (do not warm-start the restart, which crawls) and the good optimum.
        stage_b_guess = initial_guess
        if stage_a_samples:
            try:
                stage_b_guess = self.guess_from_samples(
                    stage_a_samples, num_segments,
                    stage_a_stats.get("segment_samples")
                    or ([int(num_samples_per_segment)] * num_segments
                        if not isinstance(num_samples_per_segment, (list, tuple, np.ndarray))
                        else list(num_samples_per_segment)))
            except Exception:
                stage_b_guess = initial_guess

        stage_b_samples, stage_b_stats = self._solve_once(
            waypoints, initial_guess=stage_b_guess, warm_start=False,
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
        proportional_samples: bool = True,
        force_weight: float = 0.0,
        cost_metric: str = "jerk",
        epigraph_constraints: bool = False,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build and solve the NLP for one candidate.

        The method is long: it lays out the variables, writes the collocation
        and the boundary conditions, prices the objective, sets the epigraph
        bounds and hands the problem to IPOPT, in that order, and the order is
        the point - the cost reads the constraints and the constraints read
        the variables.

        Splitting it into a variable builder, a collocation builder, an
        epigraph builder and a boundary builder was considered and deferred.
        The build is linear and the pieces share a dozen locals, so the split
        would move the code rather than clarify it, and it would put the
        solver's numerics at risk for no behaviour change: the plan would have
        to be re-measured against `benchmark.py --mode solver`, which is the
        reference table for a solver change. The iteration log is the one
        part that already lives outside this method (it is on the instance,
        and `solve` reads it after the call).

        When it is done, do it as one extraction at a time, with the solver
        benchmark before and after, and keep the objective last: it is the
        piece that reads the most from the others.
        """
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

        if sample_weights is not None:
            w_arr = np.asarray(sample_weights, dtype=float).reshape(-1)
            if w_arr.shape[0] < N - 1:
                w_arr = np.pad(w_arr, (0, (N - 1) - w_arr.shape[0]), constant_values=1.0)
        else:
            w_arr = np.ones(N - 1, dtype=float)

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
        f_bound_vec = opti.variable(N - 1) if epigraph_constraints else None

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
        if (accuracy_weight > 0 or cost_metric == "discretization") and N > 2:
            if per_segment_dt:
                dt_ref = time_cost / (N - 1)
            else:
                dt_ref = dt
            for k in range(N - 2):
                al_k = (vl[k + 1] - vl[k]) / interval_dt(k)
                al_k1 = (vl[k + 2] - vl[k + 1]) / interval_dt(k + 1)
                ar_k = (vr[k + 1] - vr[k]) / interval_dt(k)
                ar_k1 = (vr[k + 2] - vr[k + 1]) / interval_dt(k + 1)
                w_k = float(0.5 * (w_arr[k] + w_arr[k + 1])) if sample_weights is not None else 1.0
                if per_segment_dt:
                    dt_span = 0.5 * (interval_dt(k) + interval_dt(k + 1))
                    scale = dt_ref / dt_span
                    smoothness_cost += w_k * ((al_k1 - al_k)**2
                                              + (ar_k1 - ar_k)**2) * scale**2
                else:
                    smoothness_cost += w_k * ((al_k1 - al_k)**2 + (ar_k1 - ar_k)**2)

        discretization_cost = 0
        if cost_metric == "discretization":
            # Penalize the three physical mechanisms of the 5 mm tracking gate:
            # 1. Chord-vs-arc mid-interval sagitta: (1/8) * v_mid * omega_mid * dt_k^2
            # 2. Parabolic-vs-linear heading mismatch: (1/8) * v_mid * (omega2 - omega1) * dt_k^2
            # 3. 20 ms sample-and-hold staircase lag: 0.5 * (v2 - v1) * 0.02
            gate_ref_m = 0.005
            ctrl_dt_s = 0.02
            for k in range(N - 1):
                dt_k = interval_dt(k)
                v1_k = (vl[k] + vr[k]) / 2.0
                v2_k = (vl[k + 1] + vr[k + 1]) / 2.0
                w1_k = (vr[k] - vl[k]) / self.config.track_width
                w2_k = (vr[k + 1] - vl[k + 1]) / self.config.track_width
                v_mid = 0.5 * (v1_k + v2_k)
                w_mid = 0.5 * (w1_k + w2_k)
                sagitta_err = (0.125 * v_mid * w_mid * (dt_k ** 2)) / gate_ref_m
                parab_err = (0.125 * v_mid * (w2_k - w1_k) * (dt_k ** 2)) / gate_ref_m
                hold_err = (0.5 * (v2_k - v1_k) * ctrl_dt_s) / gate_ref_m
                w_k = float(w_arr[k])
                discretization_cost += w_k * (sagitta_err ** 2 + parab_err ** 2 + 0.25 * (hold_err ** 2))

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

        def motor_limit(v_wheel):
            """Worst-case per-wheel force limit at a wheel speed."""
            if apply_headroom:
                return self.config.get_max_force_symbolic_robust(v_wheel)
            return self.config.get_max_force_symbolic(v_wheel, apply_headroom=False)

        force_cost = 0
        f_stall_ref = max(
            self.config.get_max_force_at_velocity_robust(0.0)
            if apply_headroom
            else self.config.get_max_force_at_velocity(0.0, apply_headroom=False),
            1e-6,
        )

        # Precompute epigraph physical constants when epigraph_constraints=True
        if epigraph_constraints:
            if apply_headroom:
                epi_inertia = max(self.config.uncertainties["inertia_kg_m2"])
                epi_tw = min(self.config.uncertainties["track_width_range"])
                epi_cof = min(self.config.uncertainties["cof"])
                epi_f_stall, epi_v_noload = self.config.get_motor_linear_limits(apply_headroom=True)
            else:
                epi_inertia = self.config.inertia
                epi_tw = self.config.track_width
                epi_cof = self.config.cof
                epi_f_stall, epi_v_noload = self.config.get_motor_linear_limits(apply_headroom=False)

        for k in range(N - 1):
            dt_k = interval_dt(k)
            v1 = (vl[k] + vr[k]) / 2.0
            v2 = (vl[k + 1] + vr[k + 1]) / 2.0
            omega1 = (vr[k] - vl[k]) / self.config.track_width
            omega2 = (vr[k + 1] - vl[k + 1]) / self.config.track_width

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
            opti.subject_to(
                theta[k + 1] == theta[k] + 0.5 * (omega1 + omega2) * dt_k
            )

            al = (vl[k + 1] - vl[k]) / dt_k
            ar = (vr[k + 1] - vr[k]) / dt_k

            if epigraph_constraints:
                # Layer 1 Linear Epigraph Force Bounds:
                # Replaces sqrt(f_total^2 + 1e-12) and sqrt(split^2 + 1e-12)
                # (which create a 10^6 Hessian spike at al == ar) with 4 exact
                # linear inequalities and smooth motor/normal-force upper bounds.
                f_b = f_bound_vec[k]
                f_half = 0.25 * self.config.mass * (al + ar)
                split = epi_inertia * (ar - al) / (epi_tw ** 2)
                opti.subject_to(f_b >= f_half + split)
                opti.subject_to(f_b >= f_half - split)
                opti.subject_to(f_b >= -f_half + split)
                opti.subject_to(f_b >= -f_half - split)

                # Bound the interval at both ends, as the plain branch does:
                # the linear motor curve is tightest at the fastest wheel of
                # the interval, the load transfer at the tightest v * omega.
                for v_wheel in (vl[k], vl[k + 1], vr[k], vr[k + 1]):
                    opti.subject_to(f_b <= epi_f_stall * (1.0 - v_wheel / epi_v_noload))
                    opti.subject_to(f_b <= epi_f_stall * (1.0 + v_wheel / epi_v_noload))

                omega_epi = (vr[k] - vl[k]) / epi_tw
                nl_epi, nr_epi = wheel_normal_forces(
                    self.config.mass, self.config.g, CG_HEIGHT_M, epi_tw, v1 * omega_epi
                )
                nl_end, nr_end = wheel_normal_forces(
                    self.config.mass, self.config.g, CG_HEIGHT_M, epi_tw, v2 * omega2
                )
                for normal in (nl_epi, nr_epi, nl_end, nr_end):
                    opti.subject_to(f_b <= epi_cof * normal)
                f_traction_max = epi_cof * 0.5 * (ca.fmin(nl_epi, nl_end) + ca.fmin(nr_epi, nr_end))
            else:
                f_bound = self.model.get_max_wheel_force_symbolic(al, ar, apply_headroom)

                # One acceleration holds for the whole interval, but the limit
                # it is measured against moves with the speed, and the motor's
                # force limit falls as the wheel speeds up: the same burst is
                # tightest at the interval's fast end. Bound both ends, or the
                # solve approves a force the motor cannot make at the speed it
                # reaches one sample later.
                # One row per end, not one row through fmin(): the same bound
                # as two half-spaces keeps the kink out of the expression tree,
                # which is what keeps IPOPT from creeping at the speed bound.
                for v_wheel in (vl[k], vl[k + 1], vr[k], vr[k + 1]):
                    opti.subject_to(f_bound <= motor_limit(v_wheel))

                trac_start = self.model.get_traction_limit_symbolic(vl[k], vr[k], apply_headroom)
                trac_end = self.model.get_traction_limit_symbolic(vl[k + 1], vr[k + 1], apply_headroom)
                opti.subject_to(f_bound <= trac_start)
                # shortcut: the interval's traction limit at its end speed is
                # only priced, not enforced - the extra rows pushed the
                # 10-waypoint mission from 112 to 3145 iterations and a MUMPS
                # crash, and at this robot's speeds the motor limit bites first.
                # The audit checks it, so a turning plan that leans on it is
                # refused rather than exported; upgrade when a mission turns
                # hard enough for the transfer to unload a wheel.
                f_traction_max = ca.fmin(trac_start, trac_end)

            if force_weight > 0:
                fl_k, fr_k = diff_drive_wheel_forces(al, ar, self.config.mass, self.config.inertia, self.config.track_width)
                f_trac_ref = ca.fmax(f_traction_max, 1e-6)
                force_cost += (
                    0.5 * ((fl_k / f_stall_ref) ** 2 + (fr_k / f_stall_ref) ** 2)
                    + 0.5 * ((fl_k / f_trac_ref) ** 2 + (fr_k / f_trac_ref) ** 2)
                ) * dt_k

        if cost_metric == "discretization":
            eff_acc_w = accuracy_weight if accuracy_weight > 0 else 0.002
            disc_w = max(accuracy_weight * 10.0, 0.02)
            opti.minimize(
                time_cost
                + eff_acc_w * smoothness_cost
                + disc_w * discretization_cost
                + force_weight * force_cost
            )
        else:
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
        if epigraph_constraints and f_bound_vec is not None:
            guess_dt_per_sample = guess_dt[sample_to_segment] if per_segment_dt else np.full(N - 1, float(np.mean(guess_dt)))
            guess_dt_per_sample = np.maximum(guess_dt_per_sample, 1e-4)
            g_al = (guess_states[1:, 3] - guess_states[:-1, 3]) / guess_dt_per_sample
            g_ar = (guess_states[1:, 4] - guess_states[:-1, 4]) / guess_dt_per_sample
            g_f_half = np.abs(0.25 * self.config.mass * (g_al + g_ar))
            g_split = np.abs(epi_inertia * (g_ar - g_al) / (epi_tw ** 2))
            opti.set_initial(f_bound_vec, g_f_half + g_split + 1e-4)

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

        if capture_iterations:
            # The cost at each IPOPT iterate, so the Convergence plot is a
            # curve and not two dots. Decimated: the log feeds a plot, and
            # every iterate of an iteration-budgeted solve is not needed.
            def record_iteration(iteration):
                if not iteration or iteration % 5:
                    return
                try:
                    if per_segment_dt:
                        dt_val = np.asarray(opti.debug.value(dt), dtype=float).reshape(-1)
                    else:
                        dt_val = np.full(num_segments, float(opti.debug.value(dt)))
                    states = np.asarray(opti.debug.value(X), dtype=float).flatten()
                    cost = self._compute_cost(
                        np.concatenate([dt_val, states]), N, segment_samples,
                        accuracy_weight, waypoints, force_weight=force_weight)
                except Exception:
                    # A plot is not worth losing the solve.
                    return
                self.iteration_history.append({
                    "iteration": len(self.iteration_history),
                    "cost": float(cost),
                    "phase": "ipopt",
                })
            try:
                opti.callback(record_iteration)
            except Exception:
                pass

        stage_prefix = f"[stage {stage_label}] " if stage_label else ""
        try:
            with CASADI_IPOPT_LOCK:
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
                fl_k, fr_k = diff_drive_wheel_forces(al_k, ar_k, self.config.mass, self.config.inertia, self.config.track_width)
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

        from overboard.core.frontier_seed import generate_frontier_seed
        return generate_frontier_seed(
            waypoints=waypoints,
            segment_samples=num_samples_per_segment,
            robot_config=self.config,
            stop_waypoint_indices=stop_waypoint_indices,
            style=style,
        )

    @staticmethod
    def _sample_times(dt_values: np.ndarray, num_samples_per_segment: Union[int, Sequence[int]], N: int) -> np.ndarray:
        dts = np.asarray(dt_values, dtype=float)
        num_segments = len(dts)
        if N <= 1 or num_segments == 0:
            return np.zeros(N, dtype=float)
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            seg_samples = np.asarray(num_samples_per_segment, dtype=int)
        else:
            seg_samples = np.full(num_segments, int(num_samples_per_segment), dtype=int)

        starts = np.concatenate(([0.0], np.cumsum(seg_samples * dts)))
        wp_idx = np.concatenate(([0], np.cumsum(seg_samples)))
        sample_to_seg = np.repeat(np.arange(num_segments), seg_samples)[:N - 1]
        offsets = np.arange(len(sample_to_seg)) - np.repeat(wp_idx[:-1], seg_samples)[:N - 1]

        times = np.empty(N, dtype=float)
        times[:len(sample_to_seg)] = starts[sample_to_seg] + offsets * dts[sample_to_seg]
        times[-1] = starts[-1]
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
                "omega": float(diff_drive_fk(vl, vr, self.config.track_width)[1]),
                "al": float(al),
                "ar": float(ar),
                "fl": float(fl),
                "fr": float(fr),
            }
            if event is not None:
                sample_dict["event"] = event
            samples.append(sample_dict)

        return samples
