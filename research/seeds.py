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


# ---------------------------------------------------------------------------
# The Q12 repair: the Multi-Verse bootstrap, working.
#
# The pipeline's bootstrap was supposed to give the solver a kinematically
# valid path. It never did: ``path_planning.py`` read rsplan's Path object
# as a list, the TypeError was swallowed, and every call returned None - so
# the "Reeds-Shepp bootstrap" fell back to the straight line every time. The
# shipped config compounded it with ``turning_radius: 0.0``, which rsplan
# cannot plan with at all.
#
# A Reeds-Shepp path is worth having as a guess because the straight-line
# guess is kinematically wrong whenever a waypoint's heading does not point
# along the chord: the guess says "the robot is here, facing that way", and
# the solver has to undo the impossibility. The RS path is feasible by
# construction, including the reverse manoeuvres the TEB topologies
# approximate.
# ---------------------------------------------------------------------------

# The robot can turn on the spot, so its minimum turning radius is small.
# The half track is the radius of the tightest arc it can hold while moving.
DEFAULT_TURNING_RADIUS = 0.05


def _arc_lengths(points: Sequence[Tuple[float, float]]) -> List[float]:
    lengths = [0.0]
    for index in range(1, len(points)):
        dx = points[index][0] - points[index - 1][0]
        dy = points[index][1] - points[index - 1][1]
        lengths.append(lengths[-1] + math.hypot(dx, dy))
    return lengths


def _resample_path(states: Sequence[Tuple[float, float, float, int]],
                   fractions: Sequence[float]
                   ) -> List[Tuple[float, float, float, int]]:
    """Sample a path at the given fractions of its arc length.

    The path's states carry a direction of travel, so a resampled point
    knows whether the robot drives forwards or backwards there.
    """
    lengths = _arc_lengths([(s[0], s[1]) for s in states])
    total = lengths[-1]
    if total <= 0.0:
        return [states[0]] * len(fractions)

    out = []
    index = 0
    for fraction in fractions:
        target = fraction * total
        while index < len(lengths) - 2 and lengths[index + 1] < target:
            index += 1
        span = lengths[index + 1] - lengths[index]
        blend = 0.0 if span <= 0 else (target - lengths[index]) / span
        a = states[index]
        b = states[index + 1]
        x = a[0] + blend * (b[0] - a[0])
        y = a[1] + blend * (b[1] - a[1])
        # The heading is the path's own tangent, so it is the tangent of the
        # chord between two samples too - the guess is self-consistent.
        heading = a[2] + blend * ((b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi)
        direction = a[3] if blend < 0.5 else b[3]
        out.append((x, y, heading, direction))
    return out


def _signed_rolling_speeds(states: np.ndarray, dt: float,
                           stop_indices: Sequence[int], num_samples_per_segment: int,
                           robot_cfg) -> None:
    """Wheel speeds implied by the guessed motion, forwards or backwards.

    ``TrajectoryOptimizer._set_rolling_speeds`` assumes the robot moves
    along its heading. A Reeds-Shepp path may drive backwards, so the speed
    is the motion projected onto the heading: negative when the robot
    reverses.
    """
    count = len(states)
    half_track = robot_cfg.track_width / 2.0
    cap = float(robot_cfg.max_linear_speed(apply_headroom=True))
    for k in range(count - 1):
        dx = states[k + 1, 0] - states[k, 0]
        dy = states[k + 1, 1] - states[k, 1]
        dtheta = (states[k + 1, 2] - states[k, 2] + math.pi) % (2 * math.pi) - math.pi
        speed = (dx * math.cos(states[k, 2]) + dy * math.sin(states[k, 2])) / dt
        omega = dtheta / dt
        states[k, 3] = float(np.clip(speed - omega * half_track, -cap, cap))
        states[k, 4] = float(np.clip(speed + omega * half_track, -cap, cap))
    states[count - 1, 3] = states[count - 1, 4] = 0.0
    states[0, 3] = states[0, 4] = 0.0
    for index in (stop_indices or []):
        k = int(index) * num_samples_per_segment
        if 0 <= k < count:
            states[k, 3] = states[k, 4] = 0.0


def reeds_shepp_guess(waypoints: Sequence[Sequence[float]],
                      num_samples_per_segment: int, robot_cfg,
                      stop_indices: Optional[Sequence[int]] = None,
                      turning_radius: float = DEFAULT_TURNING_RADIUS
                      ) -> np.ndarray:
    """The Q12 guess: the Multi-Verse bootstrap, with the API fixed.

    Every segment whose two endpoints have a heading is planned as a
    Reeds-Shepp path and sampled along its arc length; a segment without
    both headings keeps the straight-line geometry, because the planner has
    nothing to plan from.
    """
    from path_planning import ReedsSheppPath

    num_segments = len(waypoints) - 1
    planner = ReedsSheppPath(turning_radius=turning_radius)
    rows: List[List[float]] = []

    for index in range(num_segments):
        p1 = waypoints[index]
        p2 = waypoints[index + 1]
        h1 = p1[2] if len(p1) > 2 else None
        h2 = p2[2] if len(p2) > 2 else None
        count = (num_samples_per_segment if index < num_segments - 1
                 else num_samples_per_segment + 1)
        fractions = [j / num_samples_per_segment for j in range(count)]

        states = None
        if h1 is not None and h2 is not None:
            path = planner.plan_with_direction(
                (float(p1[0]), float(p1[1]), float(h1)),
                (float(p2[0]), float(p2[1]), float(h2)),
                step_size=0.02)
            if path and len(path) >= 2:
                states = _resample_path(path, fractions)

        if states is None:
            # No plan: the straight-line geometry, as the stock seeds use.
            for j in range(count):
                frac = fractions[j]
                if h1 is not None and h2 is not None:
                    diff = (float(h2) - float(h1) + math.pi) % (2 * math.pi) - math.pi
                    heading = float(h1) + diff * frac
                elif h1 is not None:
                    heading = float(h1)
                elif h2 is not None:
                    heading = float(h2)
                else:
                    heading = math.atan2(float(p2[1]) - float(p1[1]),
                                         float(p2[0]) - float(p1[0]))
                rows.append([float(p1[0]) + frac * (float(p2[0]) - float(p1[0])),
                             float(p1[1]) + frac * (float(p2[1]) - float(p1[1])),
                             heading, 0.0, 0.0])
            continue

        for x, y, heading, _direction in states:
            rows.append([x, y, heading, 0.0, 0.0])

    states_array = np.array(rows, dtype=float)
    _signed_rolling_speeds(states_array, FLAT_DT, stop_indices or [],
                            num_samples_per_segment, robot_cfg)
    return np.concatenate([
        np.full(num_segments, FLAT_DT, dtype=float),
        states_array.flatten(),
    ])


def refined_guess(waypoints: Sequence[Sequence[float]],
                  num_samples_per_segment: int, robot_cfg,
                  stop_indices: Optional[Sequence[int]] = None,
                  accuracy_weight: float = 0.0) -> np.ndarray:
    """The Q12 guess: the repaired Multi-Verse pipeline's own output.

    Phases 1 to 4 of the pipeline - the Reeds-Shepp bootstrap, the
    kinematic profile, the critic and the segment-preserving refinement -
    produce a trajectory that passes through every waypoint, carries a
    per-segment interval for every segment, and is kinematically valid.
    That trajectory is the guess; the real solver does the magnitudes.

    This is the re-envisioning of the pipeline: it stops being a solver and
    becomes a *seed provider* for the solver that has the right objective.
    """
    from multiverse_optimizer import MasterTrajectoryOptimizer

    num_segments = len(waypoints) - 1
    pipeline = MasterTrajectoryOptimizer(robot_cfg, enable_parallel=False,
                                         verbose=False)
    geometry = pipeline.bootstrapper.generate_baseline(list(waypoints),
                                                       num_samples_per_segment)
    trajectory = pipeline._profile_kinematics(
        geometry, num_samples_per_segment, list(stop_indices or []), {}, True)
    bad_windows = pipeline.critic.evaluate(trajectory, num_samples_per_segment)
    pipeline._refine_segments(trajectory, num_samples_per_segment, bad_windows,
                              accuracy_weight, True)
    optimizer = TrajectoryOptimizer(robot_cfg)
    return optimizer.guess_from_samples(trajectory, num_segments,
                                        num_samples_per_segment)
