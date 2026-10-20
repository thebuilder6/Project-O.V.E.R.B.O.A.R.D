"""Extra initial guesses, built outside the production tree.

Two research questions need a starting point the optimizer does not have:

Q1 asks whether a real geometric path family beats the straight-line seed.
The tree has no path family at all (``docs/HEURISTICS_AND_VERIFICATION.md``
section 2), so one is built here: a per-segment cubic Bezier whose endpoint
tangents follow the waypoint headings, sampled at the segment boundaries so
the waypoint samples land exactly on the waypoints.

Q2 asks whether a length-proportional time allocation converges faster than
the flat ``dt = 0.1 s`` start every seed uses.

Both are handed to the solver through the existing ``initial_guess``
argument of ``TrajectoryOptimizer.solve``, so nothing in ``optimizer.py`` or
``main.py`` changes and the default behaviour is untouched.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from optimizer import TrajectoryOptimizer

# The flat interval every production seed starts from.
FLAT_DT = 0.1


def _unit(angle: float) -> Tuple[float, float]:
    return (math.cos(angle), math.sin(angle))


def _chord(p1: Sequence[float], p2: Sequence[float]) -> Optional[Tuple[float, float]]:
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    return (dx / length, dy / length)


def _miter(p_prev: Optional[Sequence[float]], p_next: Optional[Sequence[float]],
           point: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Direction at a waypoint with no pinned heading.

    The sum of the incoming and outgoing chord directions, so the path does
    not kink at the waypoint.
    """
    vectors = []
    if p_prev is not None:
        vector = _chord(p_prev, point)
        if vector is not None:
            vectors.append(vector)
    if p_next is not None:
        vector = _chord(point, p_next)
        if vector is not None:
            vectors.append(vector)
    if not vectors:
        return None
    sx = sum(v[0] for v in vectors)
    sy = sum(v[1] for v in vectors)
    length = math.hypot(sx, sy)
    if length < 1e-9:
        return vectors[0]
    return (sx / length, sy / length)


def _bezier_point(control: Sequence[Sequence[float]], u: float) -> Tuple[float, float]:
    """Cubic Bezier position at ``u``."""
    p0, p1, p2, p3 = control
    v = 1.0 - u
    x = (v ** 3) * p0[0] + 3 * (v ** 2) * u * p1[0] + 3 * v * (u ** 2) * p2[0] + (u ** 3) * p3[0]
    y = (v ** 3) * p0[1] + 3 * (v ** 2) * u * p1[1] + 3 * v * (u ** 2) * p2[1] + (u ** 3) * p3[1]
    return x, y


def _bezier_tangent(control: Sequence[Sequence[float]], u: float) -> Tuple[float, float]:
    """Cubic Bezier tangent (unnormalised) at ``u``."""
    p0, p1, p2, p3 = control
    v = 1.0 - u
    dx = 3 * (v ** 2) * (p1[0] - p0[0]) + 6 * v * u * (p2[0] - p1[0]) + 3 * (u ** 2) * (p3[0] - p2[0])
    dy = 3 * (v ** 2) * (p1[1] - p0[1]) + 6 * v * u * (p2[1] - p1[1]) + 3 * (u ** 2) * (p3[1] - p2[1])
    return dx, dy


def segment_tangents(waypoints: Sequence[Sequence[float]],
                     index: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Start and end tangent direction of segment ``index``.

    A pinned heading wins. Otherwise the direction is the chord, or the miter
    of the two chords when the waypoint is interior and unpinned.
    """
    num = len(waypoints)
    p1 = waypoints[index]
    p2 = waypoints[index + 1]

    start = None
    if len(p1) > 2 and p1[2] is not None:
        start = _unit(float(p1[2]))
    if start is None:
        if index == 0:
            start = _chord(p1, p2)
        else:
            start = _miter(waypoints[index - 1], p2, p1)
    if start is None:
        start = (1.0, 0.0)

    end = None
    if len(p2) > 2 and p2[2] is not None:
        end = _unit(float(p2[2]))
    if end is None:
        if index + 2 >= num:
            end = _chord(p1, p2)
        else:
            end = _miter(p1, waypoints[index + 2], p2)
    if end is None:
        end = (1.0, 0.0)
    return start, end


def spline_states(waypoints: Sequence[Sequence[float]], num_samples_per_segment: int
                  ) -> np.ndarray:
    """Sample a per-segment cubic Bezier path through the waypoints.

    The tangent at each end of a segment follows the pinned heading, so the
    sample on a waypoint carries that waypoint's heading and the path has no
    heading discontinuity there.
    """
    num_segments = len(waypoints) - 1
    rows: List[List[float]] = []
    for index in range(num_segments):
        p1 = waypoints[index]
        p2 = waypoints[index + 1]
        t1, t2 = segment_tangents(waypoints, index)
        chord = _chord(p1, p2)
        distance = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
        if distance < 1e-9:
            distance = 0.1
        handle = distance / 3.0
        control = [
            (float(p1[0]), float(p1[1])),
            (float(p1[0]) + handle * t1[0], float(p1[1]) + handle * t1[1]),
            (float(p2[0]) - handle * t2[0], float(p2[1]) - handle * t2[1]),
            (float(p2[0]), float(p2[1])),
        ]
        count = (num_samples_per_segment if index < num_segments - 1
                 else num_samples_per_segment + 1)
        for j in range(count):
            u = j / num_samples_per_segment
            x, y = _bezier_point(control, u)
            dx, dy = _bezier_tangent(control, u)
            if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                theta = math.atan2(t1[1], t1[0])
            else:
                theta = math.atan2(dy, dx)
            rows.append([x, y, theta, 0.0, 0.0])
        if chord is None:
            pass
    return np.array(rows, dtype=float)


def _rolling_speeds(states: np.ndarray, dt: float,
                    stop_indices: Sequence[int], num_samples_per_segment: int,
                    robot_cfg) -> None:
    """Fill in the wheel speeds implied by the guessed motion.

    Same rule as ``TrajectoryOptimizer._set_rolling_speeds``: the speeds come
    from the sampled geometry, they are clipped to the speed limit, and they
    are zero at the endpoints and at every declared stop.
    """
    count = len(states)
    half_track = robot_cfg.track_width / 2.0
    cap = float(robot_cfg.max_linear_speed(apply_headroom=True))
    for k in range(count - 1):
        dx = states[k + 1, 0] - states[k, 0]
        dy = states[k + 1, 1] - states[k, 1]
        dtheta = (states[k + 1, 2] - states[k, 2] + math.pi) % (2 * math.pi) - math.pi
        speed = math.hypot(dx, dy) / dt
        omega = dtheta / dt
        states[k, 3] = float(np.clip(speed - omega * half_track, -cap, cap))
        states[k, 4] = float(np.clip(speed + omega * half_track, -cap, cap))
    states[count - 1, 3] = states[count - 1, 4] = 0.0
    states[0, 3] = states[0, 4] = 0.0
    for index in (stop_indices or []):
        k = int(index) * num_samples_per_segment
        if 0 <= k < count:
            states[k, 3] = states[k, 4] = 0.0


def spline_guess(waypoints: Sequence[Sequence[float]], num_samples_per_segment: int,
                 robot_cfg, stop_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """The Q1 guess: a Bezier path family with rolling speeds.

    The layout is the one ``TrajectoryOptimizer._build_initial_guess`` returns:
    one interval length per segment, then the flattened (x, y, heading, vl, vr)
    states.
    """
    num_segments = len(waypoints) - 1
    states = spline_states(waypoints, num_samples_per_segment)
    _rolling_speeds(states, FLAT_DT, stop_indices or [], num_samples_per_segment,
                     robot_cfg)
    guess = np.concatenate([
        np.full(num_segments, FLAT_DT, dtype=float),
        states.flatten(),
    ])
    return guess


def proportional_dt_guess(waypoints: Sequence[Sequence[float]],
                          num_samples_per_segment: int, robot_cfg,
                          stop_indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """The Q2 guess: the baseline geometry with a length-proportional start.

    ``dt_i`` is proportional to the length of segment ``i`` and scaled so the
    total time matches the flat ``dt = 0.1 s`` start. Only the *initial* time
    allocation changes; the geometry and the rolling speeds are the ones the
    ``rolling`` seed already uses.
    """
    num_segments = len(waypoints) - 1
    optimizer = TrajectoryOptimizer(robot_cfg)
    guess = optimizer._build_initial_guess(
        waypoints, num_samples_per_segment, num_segments * num_samples_per_segment + 1,
        style="rolling", stop_waypoint_indices=list(stop_indices or []))

    lengths = []
    for index in range(num_segments):
        p1 = waypoints[index]
        p2 = waypoints[index + 1]
        lengths.append(max(math.hypot(float(p2[0]) - float(p1[0]),
                                      float(p2[1]) - float(p1[1])), 1e-6))
    total = sum(lengths)
    scale = (num_segments * FLAT_DT) / total if total > 0 else FLAT_DT
    dts = np.array([min(max(length * scale, 0.001), 1.0) for length in lengths])
    guess[:num_segments] = dts
    return guess
