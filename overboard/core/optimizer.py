"""
Trajectory Optimizer (CasADi/IPOPT Implementation)
=================================================
Deterministic NLP direct collocation for differential-drive robotics.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import casadi as ca
import numpy as np

from overboard.core.frontier_seed import GUESS_STYLES, LEGACY_GUESS_STYLE_MAP, SUPPORTED_GUESS_STYLES
from overboard.models.drivetrain_physics import (
    CG_HEIGHT_M,
    angle_diff,
    diff_drive_fk,
    diff_drive_wheel_forces,
    wheel_normal_forces,
)
from overboard.models.robot_model import DifferentialDriveModel, RobotConfig

CASADI_IPOPT_LOCK = CASADI_SOLVE_LOCK = threading.RLock()

_GAUSS_LEGENDRE_4 = (
    (-0.8611363115940526, 0.3478548451374538),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538),
)


def _formulate_collocation(
    opti: ca.Opti,
    X: ca.MX,
    dt_k: List[ca.MX],
    N: int,
    config: RobotConfig,
    model: DifferentialDriveModel,
    *,
    self_consistent: bool,
    epigraph_constraints: bool,
    f_bound: Optional[ca.MX],
    apply_headroom: bool,
    force_weight: float,
    cost_metric: str,
    w_arr: np.ndarray,
    time_cost: ca.MX,
    per_segment_dt: bool,
    dt: ca.MX,
    accuracy_weight: float,
) -> ca.MX:
    x, y, theta, vl, vr = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    tw = config.track_width

    if epigraph_constraints:
        unc = config.uncertainties if apply_headroom else {}
        epi_inertia = max(unc.get("inertia_kg_m2", [config.inertia]))
        epi_tw = min(unc.get("track_width_range", [config.track_width]))
        epi_cof = min(unc.get("cof", [config.cof]))
        epi_f_stall, epi_v_noload = config.get_motor_linear_limits(apply_headroom)

    f_stall_ref = max(
        config.get_max_force_at_velocity_robust(0.0) if apply_headroom else config.get_max_force_at_velocity(0.0, apply_headroom=False),
        1e-6,
    )
    force_cost, disc_cost = 0, 0

    for k in range(N - 1):
        h = dt_k[k]
        v1, v2 = 0.5 * (vl[k] + vr[k]), 0.5 * (vl[k + 1] + vr[k + 1])
        w1, w2 = (vr[k] - vl[k]) / tw, (vr[k + 1] - vl[k + 1]) / tw

        if self_consistent:
            dx, dy, dw = 0.0, 0.0, w2 - w1
            for xi, w_gl in _GAUSS_LEGENDRE_4:
                s_gl = 0.5 * h * (1.0 + xi)
                v_s = v1 + (v2 - v1) * (s_gl / h)
                th_s = theta[k] + w1 * s_gl + dw * (s_gl * s_gl) / (2.0 * h)
                dx += (0.5 * h * w_gl) * v_s * ca.cos(th_s)
                dy += (0.5 * h * w_gl) * v_s * ca.sin(th_s)
            opti.subject_to(x[k + 1] == x[k] + dx)
            opti.subject_to(y[k + 1] == y[k] + dy)
        else:
            opti.subject_to(x[k + 1] == x[k] + 0.5 * h * (v1 * ca.cos(theta[k]) + v2 * ca.cos(theta[k + 1])))
            opti.subject_to(y[k + 1] == y[k] + 0.5 * h * (v1 * ca.sin(theta[k]) + v2 * ca.sin(theta[k + 1])))
        opti.subject_to(theta[k + 1] == theta[k] + 0.5 * h * (w1 + w2))

        al_k, ar_k = (vl[k + 1] - vl[k]) / h, (vr[k + 1] - vr[k]) / h
        if epigraph_constraints:
            fb = f_bound[k]
            f_half = 0.25 * config.mass * (al_k + ar_k)
            split = epi_inertia * (ar_k - al_k) / (epi_tw**2)
            opti.subject_to(fb >= f_half + split)
            opti.subject_to(fb >= f_half - split)
            opti.subject_to(fb >= -f_half + split)
            opti.subject_to(fb >= -f_half - split)
            opti.subject_to(fb <= epi_f_stall * (1.0 - vl[k] / epi_v_noload))
            opti.subject_to(fb <= epi_f_stall * (1.0 + vl[k] / epi_v_noload))
            opti.subject_to(fb <= epi_f_stall * (1.0 - vr[k] / epi_v_noload))
            opti.subject_to(fb <= epi_f_stall * (1.0 + vr[k] / epi_v_noload))
            nl_epi, nr_epi = wheel_normal_forces(config.mass, config.g, CG_HEIGHT_M, epi_tw, v1 * (vr[k] - vl[k]) / epi_tw)
            opti.subject_to(fb <= epi_cof * nl_epi)
            opti.subject_to(fb <= epi_cof * nr_epi)
            f_trac_max = epi_cof * 0.5 * (nl_epi + nr_epi)
        else:
            fb = model.get_max_wheel_force_symbolic(al_k, ar_k, apply_headroom)
            opti.subject_to(fb <= (config.get_max_force_symbolic_robust(vl[k]) if apply_headroom else config.get_max_force_symbolic(vl[k], apply_headroom=False)))
            opti.subject_to(fb <= (config.get_max_force_symbolic_robust(vr[k]) if apply_headroom else config.get_max_force_symbolic(vr[k], apply_headroom=False)))
            f_trac_max = model.get_traction_limit_symbolic(vl[k], vr[k], apply_headroom)
            opti.subject_to(fb <= f_trac_max)

        if force_weight > 0:
            fl_k, fr_k = diff_drive_wheel_forces(al_k, ar_k, config.mass, config.inertia, config.track_width)
            f_trac_ref = ca.fmax(f_trac_max, 1e-6)
            force_cost += (0.5 * ((fl_k / f_stall_ref) ** 2 + (fr_k / f_stall_ref) ** 2) + 0.5 * ((fl_k / f_trac_ref) ** 2 + (fr_k / f_trac_ref) ** 2)) * h

        if cost_metric == "discretization":
            sag = (0.125 * 0.5 * (v1 + v2) * (0.5 * (w1 + w2)) * (h**2)) / 0.005
            par = (0.125 * 0.5 * (v1 + v2) * (w2 - w1) * (h**2)) / 0.005
            hld = (0.5 * (v2 - v1) * 0.02) / 0.005
            disc_cost += w_arr[k] * (sag**2 + par**2 + 0.25 * (hld**2))

    smoothness_cost = 0
    if (accuracy_weight > 0 or cost_metric == "discretization") and N > 2:
        dt_ref = time_cost / (N - 1) if per_segment_dt else dt
        for k in range(N - 2):
            al0, al1 = (vl[k + 1] - vl[k]) / dt_k[k], (vl[k + 2] - vl[k + 1]) / dt_k[k + 1]
            ar0, ar1 = (vr[k + 1] - vr[k]) / dt_k[k], (vr[k + 2] - vr[k + 1]) / dt_k[k + 1]
            scale = (dt_ref / (0.5 * (dt_k[k] + dt_k[k + 1]))) if per_segment_dt else 1.0
            smoothness_cost += 0.5 * (w_arr[k] + w_arr[k + 1]) * scale**2 * ((al1 - al0) ** 2 + (ar1 - ar0) ** 2)

    eff_acc_w = accuracy_weight if accuracy_weight > 0 else 0.002
    obj = time_cost + (eff_acc_w if cost_metric == "discretization" else accuracy_weight) * smoothness_cost
    if cost_metric == "discretization":
        obj += max(accuracy_weight * 10.0, 0.02) * disc_cost
    if force_weight > 0:
        obj += force_weight * force_cost
    return obj


class TrajectoryOptimizer:
    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.model = DifferentialDriveModel(config)
        # The iteration log is per instance *and* per thread. A class-level
        # thread-local would isolate threads but not instances, and two
        # optimizers in one thread would share one slot: the CLI's optimizer
        # and the unified planner's would overwrite each other's log, and
        # the convergence plot would show the wrong one.
        self._local = threading.local()
        self._local.history = []

    @property
    def iteration_history(self) -> List[Dict[str, Any]]:
        return getattr(self._local, "history", [])

    @iteration_history.setter
    def iteration_history(self, val: List[Dict[str, Any]]) -> None:
        self._local.history = val

    @staticmethod
    def get_adaptive_mesh_sample_counts(
        waypoints: Sequence[Sequence[Any]], min_samples: int = 3, apex_samples: int = 8, straight_samples: int = 4
    ) -> List[int]:
        S = len(waypoints) - 1
        if S <= 0:
            return []
        pts = np.array([[w[0], w[1]] for w in waypoints], dtype=float)
        L = np.hypot(*(d := np.diff(pts, axis=0)).T)
        th = np.arctan2(d[:, 1], d[:, 0])
        counts = []
        for i in range(S):
            h1 = waypoints[i][2] if len(waypoints[i]) > 2 else None
            h2 = waypoints[i + 1][2] if len(waypoints[i + 1]) > 2 else None
            dth = (
                abs(angle_diff(float(h2), float(h1)))
                if h1 is not None and h2 is not None
                else max(
                    abs(angle_diff(th[i], th[i - 1 if i > 0 else 0])),
                    abs(angle_diff(th[i + 1 if i < S - 1 else i], th[i])),
                )
            )
            c = (
                apex_samples
                if dth > 0.70
                else (apex_samples + straight_samples) // 2
                if dth > 0.30
                else (straight_samples + 1 if L[i] > 0.35 else straight_samples)
            )
            counts.append(int(max(c, min_samples)))
        return counts

    @staticmethod
    def get_segment_sample_counts(
        waypoints: Sequence[Sequence[Any]],
        num_samples_per_segment: Union[int, Sequence[int]] = 10,
        proportional: bool = True,
        min_samples: int = 3,
        h_adaptive: bool = False,
    ) -> List[int]:
        if h_adaptive:
            return TrajectoryOptimizer.get_adaptive_mesh_sample_counts(waypoints, min_samples=min_samples)
        S = len(waypoints) - 1
        if S <= 0:
            return []
        if isinstance(num_samples_per_segment, (list, tuple, np.ndarray)):
            if len(num_samples_per_segment) != S:
                raise ValueError(f"Sample counts list has {len(num_samples_per_segment)} entries; expected {S}")
            return [int(c) for c in num_samples_per_segment]
        s_int = int(num_samples_per_segment)
        if not proportional or S == 1:
            return [s_int] * S
        lengths = np.hypot(*np.diff([[w[0], w[1]] for w in waypoints], axis=0).T)
        tot_len, m = float(lengths.sum()), min(s_int, max(1, min_samples))
        if tot_len < 1e-6 or (rem := S * (s_int - m)) <= 0:
            return [s_int] * S
        raw = rem * (lengths / tot_len)
        alloc = np.floor(raw).astype(int)
        for idx in np.argsort(-(raw - alloc))[: rem - int(alloc.sum())]:
            alloc[idx] += 1
        return (m + alloc).tolist()

    @staticmethod
    def merge_conditions(waypoints: Sequence[Sequence[Any]], stop_waypoint_indices: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
        stops = set(stop_waypoint_indices or [])
        return [
            {
                "index": i,
                "stop": (s := i in stops),
                "heading_pinned": (p := len(wp) > 2 and wp[2] is not None),
                "boundary": (b := bool(s and p)),
                "agree_velocity": not b,
                "agree_heading": not b,
                "continuous_force": not s,
            }
            for i, wp in enumerate(waypoints)
        ]

    @staticmethod
    def guess_from_samples(
        samples: List[Dict[str, Any]], num_segments: int, num_samples_per_segment: Union[int, Sequence[int]]
    ) -> np.ndarray:
        if not samples:
            raise ValueError("Cannot build a guess from an empty trajectory")
        times = np.array([s["t"] for s in samples], dtype=float)
        states = np.array([[s["x"], s["y"], s["heading"], s["vl"], s["vr"]] for s in samples], dtype=float)
        counts = (
            [int(s) for s in num_samples_per_segment]
            if isinstance(num_samples_per_segment, (list, tuple, np.ndarray))
            else [int(num_samples_per_segment)] * num_segments
        )
        k_idx = np.minimum(np.cumsum([0] + counts[:-1]), len(samples) - 2)
        dts = np.maximum(times[k_idx + 1] - times[k_idx], 1e-3)
        return np.concatenate([dts, states.flatten()])

    @staticmethod
    def _sample_times(dt_values: np.ndarray, num_samples_per_segment: Union[int, Sequence[int]], N: int) -> np.ndarray:
        counts = (
            [int(s) for s in num_samples_per_segment]
            if isinstance(num_samples_per_segment, (list, tuple, np.ndarray))
            else [int(num_samples_per_segment)] * len(dt_values)
        )
        return np.concatenate([[0.0], np.cumsum(np.repeat(np.asarray(dt_values, dtype=float), counts))])

    def format_output(
        self,
        params: np.ndarray,
        N: int,
        num_samples_per_segment: Union[int, Sequence[int]] = 10,
        waypoint_events: Optional[Dict[int, str]] = None,
    ) -> List[Dict[str, Any]]:
        counts = (
            [int(s) for s in num_samples_per_segment]
            if isinstance(num_samples_per_segment, (list, tuple, np.ndarray))
            else [int(num_samples_per_segment)] * ((N - 1) // int(num_samples_per_segment) if N > 1 else 1)
        )
        S = len(counts)
        dt_val, states = np.asarray(params[:S], dtype=float), params[S:].reshape((N, 5))
        times = self._sample_times(dt_val, counts, N)
        dt_k = np.append(dt_val[np.repeat(np.arange(S), counts)], 1.0)
        al = np.append((states[1:, 3] - states[:-1, 3]) / dt_k[:-1], 0.0)
        ar = np.append((states[1:, 4] - states[:-1, 4]) / dt_k[:-1], 0.0)
        fl, fr = diff_drive_wheel_forces(al, ar, self.config.mass, self.config.inertia, self.config.track_width)
        wp_idx = [0] + list(np.cumsum(counts))
        ev_map = {wp_idx[i]: ev for i, ev in (waypoint_events or {}).items() if 0 <= i < len(wp_idx)}

        return [
            {
                "t": float(times[k]),
                "x": float(states[k, 0]),
                "y": float(states[k, 1]),
                "heading": float(states[k, 2]),
                "vl": float(states[k, 3]),
                "vr": float(states[k, 4]),
                "omega": float(diff_drive_fk(states[k, 3], states[k, 4], self.config.track_width)[1]),
                "al": float(al[k]),
                "ar": float(ar[k]),
                "fl": float(fl[k]),
                "fr": float(fr[k]),
                **({"event": ev_map[k]} if k in ev_map else {}),
            }
            for k in range(N)
        ]

    def _build_initial_guess(
        self,
        waypoints: List[Tuple[float, float, Optional[float]]],
        num_samples_per_segment: Union[int, Sequence[int]],
        N: int,
        style: str = "baseline",
        stop_waypoint_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        if style not in SUPPORTED_GUESS_STYLES:
            raise ValueError(f"Unknown guess_style {style!r}; expected one of {SUPPORTED_GUESS_STYLES}")
        if style in LEGACY_GUESS_STYLE_MAP:
            warnings.warn(f"guess_style '{style}' deprecated; mapped to {LEGACY_GUESS_STYLE_MAP[style]}", DeprecationWarning, stacklevel=2)
            style = LEGACY_GUESS_STYLE_MAP[style]
        from overboard.core.frontier_seed import generate_frontier_seed
        return generate_frontier_seed(waypoints, num_samples_per_segment, self.config, stop_waypoint_indices, style=style)

    def _compute_cost(
        self,
        params: np.ndarray,
        N: int,
        num_samples_per_segment: Union[int, Sequence[int]],
        accuracy_weight: float,
        waypoints: List[Tuple[float, float, Optional[float]]],
        force_weight: float = 0.0,
    ) -> float:
        """The cost of a plan, for the convergence log.

        The params vector carries the layout: it is the segment durations
        followed by the N states. The number of segments therefore comes
        from the params, not from the waypoint list, so a caller that
        hands in a plan whose layout the waypoints do not describe (the
        KI-022 fixture is a two-waypoint list with an eleven-state plan)
        is still evaluated on the layout it actually carries.
        """
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
                smoothness_cost += ((al_k1 - al_k) ** 2 + (ar_k1 - ar_k) ** 2) * scale ** 2
            cost = time_cost + accuracy_weight * smoothness_cost
        else:
            cost = time_cost

        if force_weight > 0 and N > 1:
            dt_k = dt_values[sample_to_segment]
            al = (states[1:, 3] - states[:-1, 3]) / dt_k
            ar = (states[1:, 4] - states[:-1, 4]) / dt_k
            fl, fr = diff_drive_wheel_forces(al, ar, self.config.mass, self.config.inertia, self.config.track_width)
            f_stall = max(self.config.get_max_force_at_velocity_robust(0.0), 1e-6)
            cof = min(self.config.uncertainties["cof"])
            nl, nr = wheel_normal_forces(
                self.config.mass,
                self.config.g,
                CG_HEIGHT_M,
                self.config.track_width,
                0.5 * (states[:-1, 3] + states[:-1, 4]) * (states[:-1, 4] - states[:-1, 3]) / self.config.track_width,
            )
            f_trac = np.maximum(cof * np.minimum(nl, nr), 1e-6)
            cost += force_weight * float(
                np.sum(
                    (0.5 * ((fl / f_stall) ** 2 + (fr / f_stall) ** 2)
                     + 0.5 * ((fl / f_trac) ** 2 + (fr / f_trac) ** 2)) * dt_k
                )
            )
        return cost

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
        cost_metric: str = "jerk",
        epigraph_constraints: bool = False,
        sample_weights: Optional[Sequence[float]] = None,
        h_adaptive: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Build and solve the NLP for one candidate.

        The collocation, the force bounds and the objective are built by
        `_formulate_collocation`, which returns the objective and leaves the
        constraints it wrote in the Opti. This method lays the variables and
        the bounds out, calls it, and then pins the waypoints and the stops.

        The order matters and must not be rearranged: the cost reads the
        constraints, and the constraints read the variables.

        The encoding is part of the plan. Registering the bounds with
        `opti.bounded` instead of explicit inequality pairs keeps the same
        feasible set and the same objective, and it still moves the plan -
        IPOPT's interior-point path changes, and on the complex mission it
        lands in a different local optimum (4.96 s against 5.78 s, and
        118 mm of trajectory). `tests/test_optimizer.py::TestGoldenPlan`
        pins the plan sample by sample, so that change fails a test instead
        of silently re-basing the ledger.
        """
        S = len(waypoints) - 1
        segment_samples = self.get_segment_sample_counts(
            waypoints, num_samples_per_segment, proportional=proportional_samples, h_adaptive=h_adaptive
        )
        N = sum(segment_samples) + 1
        wp_indices = [0] + list(np.cumsum(segment_samples))
        s2s = np.repeat(np.arange(S), segment_samples)
        stops = set(stop_waypoint_indices or [])
        events = waypoint_events or {}
        w_arr = (
            np.ones(N - 1, dtype=float)
            if sample_weights is None
            else np.pad(np.asarray(sample_weights, dtype=float).ravel(), (0, max(0, N - 1 - len(sample_weights))), constant_values=1.0)[
                : N - 1
            ]
        )

        start_time = time.time()
        local_history: List[Dict[str, Any]] = []
        opti = ca.Opti()

        dt = opti.variable(S) if per_segment_dt else opti.variable()
        time_cost = ca.dot(ca.DM(segment_samples), dt) if per_segment_dt else dt * (N - 1)
        X = opti.variable(N, 5)
        f_bound = opti.variable(N - 1) if epigraph_constraints else None

        x, y, theta, vl, vr = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        dt_k = [dt[s2s[k]] if per_segment_dt else dt for k in range(N - 1)]

        opti.subject_to(dt >= 0.001)
        opti.subject_to(dt <= 1.0)
        opti.subject_to(x >= -20.0)
        opti.subject_to(x <= 20.0)
        opti.subject_to(y >= -20.0)
        opti.subject_to(y <= 20.0)
        opti.subject_to(theta >= -100.0)
        opti.subject_to(theta <= 100.0)
        v_bound = 0.99 * self.config.max_linear_speed(apply_headroom)
        opti.subject_to(vl >= -v_bound)
        opti.subject_to(vl <= v_bound)
        opti.subject_to(vr >= -v_bound)
        opti.subject_to(vr <= v_bound)

        obj = _formulate_collocation(
            opti,
            X,
            dt_k,
            N,
            self.config,
            self.model,
            self_consistent=self_consistent,
            epigraph_constraints=epigraph_constraints,
            f_bound=f_bound,
            apply_headroom=apply_headroom,
            force_weight=force_weight,
            cost_metric=cost_metric,
            w_arr=w_arr,
            time_cost=time_cost,
            per_segment_dt=per_segment_dt,
            dt=dt,
            accuracy_weight=accuracy_weight,
        )
        opti.minimize(obj)

        for i, wp in enumerate(waypoints):
            idx = wp_indices[i]
            opti.subject_to(x[idx] == wp[0])
            opti.subject_to(y[idx] == wp[1])
            if wp[2] is not None:
                opti.subject_to(theta[idx] == wp[2])

        opti.subject_to(vl[0] == 0)
        opti.subject_to(vr[0] == 0)
        opti.subject_to(vl[N - 1] == 0)
        opti.subject_to(vr[N - 1] == 0)

        for s_idx in stops:
            if 0 <= s_idx < len(wp_indices):
                opti.subject_to(vl[wp_indices[s_idx]] == 0)
                opti.subject_to(vr[wp_indices[s_idx]] == 0)

        if initial_guess is not None:
            guess = np.asarray(initial_guess, dtype=float)
            if guess.shape[0] != S + N * 5:
                raise ValueError(f"Initial guess has {guess.shape[0]} entries; expected {S + N * 5}")
        else:
            guess = self._build_initial_guess(waypoints, segment_samples, N, style=guess_style, stop_waypoint_indices=stop_waypoint_indices)
        guess_dt, guess_states = guess[:S], guess[S:].reshape((N, 5))
        opti.set_initial(dt, guess_dt if per_segment_dt else float(np.mean(guess_dt)))
        opti.set_initial(X, guess_states)

        if epigraph_constraints and f_bound is not None:
            unc = self.config.uncertainties if apply_headroom else {}
            epi_inertia = max(unc.get("inertia_kg_m2", [self.config.inertia]))
            epi_tw = min(unc.get("track_width_range", [self.config.track_width]))
            g_dt = np.maximum(guess_dt[s2s] if per_segment_dt else np.mean(guess_dt), 1e-4)
            g_al, g_ar = (guess_states[1:, 3] - guess_states[:-1, 3]) / g_dt, (guess_states[1:, 4] - guess_states[:-1, 4]) / g_dt
            opti.set_initial(f_bound, np.abs(0.25 * self.config.mass * (g_al + g_ar)) + np.abs(epi_inertia * (g_ar - g_al) / (epi_tw**2)) + 1e-4)

        if per_segment_dt:
            conds = self.merge_conditions(waypoints, stop_waypoint_indices)
            dt_ref = float(np.mean(guess_dt)) or 0.1
            scale = 1.0 / (dt_ref * dt_ref)
            for j in range(1, S):
                if conds[j]["continuous_force"]:
                    k = wp_indices[j]
                    opti.subject_to(((vl[k] - vl[k - 1]) * dt[j] - (vl[k + 1] - vl[k]) * dt[j - 1]) * scale == 0)
                    opti.subject_to(((vr[k] - vr[k - 1]) * dt[j] - (vr[k + 1] - vr[k]) * dt[j - 1]) * scale == 0)

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
            "warm_start_init_point": "yes" if warm_start else "no",
            **({"warm_start_bound_push": 1e-6} if warm_start else {}),
            **(solver_options or {}),
        }
        opti.solver("ipopt", p_opts, s_opts)

        if capture_iterations:
            local_history.append({
                "iteration": 0,
                "cost": self._compute_cost(guess, N, segment_samples, accuracy_weight, waypoints, force_weight),
                "trajectory": guess_states.copy(),
                "dt": guess_dt.tolist(),
                "phase": "initial_guess",
            })

        stage_prefix = f"[stage {stage_label}] " if stage_label else ""
        try:
            with CASADI_IPOPT_LOCK:
                sol = opti.solve()
            dt_val = np.asarray(sol.value(dt), dtype=float).reshape(-1) if per_segment_dt else np.full(S, float(sol.value(dt)))
            X_val = np.array(sol.value(X))
            params = np.concatenate([dt_val, X_val.flatten()])
            total_time = float(np.sum(segment_samples * dt_val))
            if verbose:
                print(f"{stage_prefix}Optimization converged. Total time: {total_time:.4f}s")
            stats = {
                **sol.stats(),
                "total_time": time.time() - start_time,
                "initial_cost": self._compute_cost(guess, N, segment_samples, accuracy_weight, waypoints, force_weight),
                "final_cost": total_time,
                "segment_dt": dt_val.tolist(),
                "segment_samples": segment_samples,
                "converged": True,
            }
            if capture_iterations:
                local_history.append({
                    "iteration": len(local_history),
                    "cost": total_time,
                    "trajectory": X_val.copy(),
                    "dt": dt_val.tolist(),
                    "phase": "final_solution",
                })
            self.iteration_history = local_history
            return self.format_output(params, N, segment_samples, events), stats
        except Exception as e:
            if verbose:
                print(f"{stage_prefix}Optimization {'stopped at budget' if 'Maximum_Iterations_Exceeded' in str(e) else 'failed'}: {e}")
            dt_val = np.asarray(opti.debug.value(dt), dtype=float).reshape(-1) if per_segment_dt else np.full(S, float(opti.debug.value(dt)))
            X_val = np.array(opti.debug.value(X))
            params = np.concatenate([dt_val, X_val.flatten()])
            total_time = float(np.sum(segment_samples * dt_val))
            stats = {
                "total_time": time.time() - start_time,
                "initial_cost": self._compute_cost(guess, N, segment_samples, accuracy_weight, waypoints, force_weight),
                "final_cost": total_time,
                "segment_dt": dt_val.tolist(),
                "segment_samples": segment_samples,
                "converged": False,
                "error": str(e),
            }
            if "Maximum_Iterations_Exceeded" in str(e):
                stats["iter_count"] = int(s_opts.get("max_iter", 0) or 0)
            if capture_iterations:
                local_history.append({
                    "iteration": len(local_history),
                    "cost": total_time,
                    "trajectory": X_val.copy(),
                    "dt": dt_val.tolist(),
                    "phase": "failed_solution",
                })
            self.iteration_history = local_history
            return self.format_output(params, N, segment_samples, events), stats

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
        cost_metric: str = "jerk",
        epigraph_constraints: bool = False,
        sample_weights: Optional[Sequence[float]] = None,
        h_adaptive: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        conds = self.merge_conditions(waypoints, stop_waypoint_indices)
        splits = [0] + [c["index"] for c in conds if c["boundary"] and 0 < c["index"] < len(waypoints) - 1] + [len(waypoints) - 1]

        if split_at_boundaries and len(splits) > 2 and initial_guess is None:
            sub_res = []
            for m in range(len(splits) - 1):
                i0, i1 = splits[m], splits[m + 1]
                sub_wps = waypoints[i0 : i1 + 1]
                sub_stops = [idx - i0 for idx in (stop_waypoint_indices or []) if i0 <= idx <= i1]
                sub_events = {idx - i0: ev for idx, ev in (waypoint_events or {}).items() if (i0 <= idx <= i1 if m == 0 else i0 < idx <= i1)}
                sub_samples = (
                    list(num_samples_per_segment[i0:i1])
                    if isinstance(num_samples_per_segment, (list, tuple, np.ndarray))
                    else num_samples_per_segment
                )

                best_pair = None
                for style in [guess_style] + [s for s in GUESS_STYLES if s != guess_style]:
                    try:
                        s_samp, s_stat = self.solve(
                            sub_wps,
                            num_samples_per_segment=sub_samples,
                            accuracy_weight=accuracy_weight,
                            stop_waypoint_indices=sub_stops,
                            waypoint_events=sub_events,
                            apply_headroom=apply_headroom,
                            verbose=verbose and m == 0,
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
                            cost_metric=cost_metric,
                            epigraph_constraints=epigraph_constraints,
                        )
                        if s_stat.get("converged", False) or best_pair is None:
                            best_pair = (s_samp, s_stat, list(self.iteration_history))
                        if s_stat.get("converged", False):
                            break
                    except Exception:
                        if style == GUESS_STYLES[-1] and best_pair is None:
                            raise
                sub_res.append(best_pair)

            stitched, cur_t = [], 0.0
            for m, (s_samples, _, _) in enumerate(sub_res):
                for s in (s_samples if m == 0 else s_samples[1:]):
                    stitched.append({**s, "t": s["t"] + cur_t})
                cur_t = stitched[-1]["t"]

            stats = {
                "total_time": cur_t,
                "initial_cost": sum(s.get("initial_cost", 0.0) for _, s, _ in sub_res),
                "final_cost": sum(s.get("final_cost", 0.0) for _, s, _ in sub_res),
                "segment_dt": [dt for _, s, _ in sub_res for dt in s.get("segment_dt", [])],
                "segment_samples": [cnt for _, s, _ in sub_res for cnt in s.get("segment_samples", [])],
                "converged": all(s.get("converged", False) for _, s, _ in sub_res),
                "sub_mission_count": len(splits) - 1,
            }
            if any(s.get("fast_polish", False) for _, s, _ in sub_res):
                stats["fast_polish"] = True
                stats["stage_count"] = sum(s.get("stage_count", 1) for _, s, _ in sub_res)
                stats["stage_times"] = [s.get("total_time", 0.0) for _, s, _ in sub_res]
                stage_iters = []
                for _, s, _ in sub_res:
                    if "stage_iterations" in s and s["stage_iterations"]:
                        stage_iters.extend(s["stage_iterations"])
                    elif "iter_count" in s:
                        stage_iters.append(s["iter_count"])
                stats["stage_iterations"] = stage_iters
            if capture_iterations and all(len(h) >= 2 for _, _, h in sub_res):
                self.iteration_history = [
                    {
                        "iteration": 0,
                        "cost": stats["initial_cost"],
                        "trajectory": np.vstack([sub_res[0][2][0]["trajectory"]] + [h[0]["trajectory"][1:] for _, _, h in sub_res[1:]]),
                        "dt": stats["segment_dt"],
                        "phase": "initial_guess",
                    },
                    {
                        "iteration": 1,
                        "cost": stats["final_cost"],
                        "trajectory": np.vstack([sub_res[0][2][-1]["trajectory"]] + [h[-1]["trajectory"][1:] for _, _, h in sub_res[1:]]),
                        "dt": stats["segment_dt"],
                        "phase": "final_solution",
                    },
                ]
            else:
                self.iteration_history = []
            return stitched, stats

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
            cost_metric=cost_metric,
            epigraph_constraints=epigraph_constraints,
            sample_weights=sample_weights,
            h_adaptive=h_adaptive,
        )
        if not fast_polish:
            return self._solve_once(waypoints, initial_guess=initial_guess, warm_start=True, stage_label=None, solver_options=solver_options, **common)

        s_a, st_a = self._solve_once(
            waypoints,
            initial_guess=initial_guess,
            warm_start=True,
            stage_label="1/2",
            solver_options={**(solver_options or {}), "max_iter": max(1, int(polish_iter_budget))},
            **common,
        )
        if st_a.get("converged", False):
            return s_a, {
                **st_a,
                "fast_polish": True,
                "stage_count": 1,
                "stage_times": [st_a.get("total_time")],
                "stage_iterations": [st_a.get("iter_count")],
                "stage_costs": [st_a.get("final_cost")],
            }

        s_b, st_b = self._solve_once(waypoints, initial_guess=initial_guess, warm_start=False, stage_label="2/2", solver_options=solver_options, **common)
        stages = (
            [st_a, st_b]
            if st_b.get("converged", False)
            else [
                st_a,
                st_b,
                (s_c := self._solve_once(waypoints, initial_guess=initial_guess, warm_start=True, stage_label=None, solver_options=solver_options, **common))[1],
            ]
        )
        winner_samples = s_b if st_b.get("converged", False) else s_c[0]
        return winner_samples, {
            **stages[-1],
            "fast_polish": True,
            "stage_count": len(stages),
            "stage_times": [s.get("total_time") for s in stages],
            "stage_iterations": [s.get("iter_count") for s in stages],
            "stage_costs": [s.get("final_cost") for s in stages],
            "total_time": sum(float(s.get("total_time", 0.0)) for s in stages),
        }
