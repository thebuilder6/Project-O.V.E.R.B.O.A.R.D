"""
The field: the mat, the mission models standing on it, and whether a planned
path drives into one.

What this is for
----------------
A trajectory is a path on a real mat, and the mat has things on it. A plan
that runs through a mission model is not a plan, whatever the optimiser
thought. Nothing here *changes* a plan: the module only answers "does this
path touch anything", so the studio can mark the touch and a human can
decide.

Where the geometry comes from
-----------------------------
The footprints are measured off FIRST's own field wireframe, which is public
(``fields/*.json`` says which one, per season). The mat's picture is
FIRST's copyright and is not in this repository: the operator supplies their
own image of their mat, and the geometry is the *coordinates* on it.

The measurement is good, but it is not a survey. Two independent readings of
the same printed wireframe agree on the sizes to a few millimetres and place
the features up to about 100 mm apart, so a footprint carries an uncertainty
and a collision that is shallower than that uncertainty is reported as
*uncertain* rather than as a hit.

The frames
----------
Everything is in the **mat's** frame: the mat's south-west corner is
``(0, 0)``, its north edge is up, and the units are metres. The studio's
field canvas uses the same frame, so the footprints and the mat image line
up. The table is bigger than the mat (2362 x 1143 mm against 2004 x 1143),
with the mat centred in it, so the table's frame is the mat's frame plus
181 mm in x.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import os

from overboard.paths import FIELDS_DIR

# The path between two samples is walked at half the robot's radius, so a
# model that lies between two samples is not missed. The step is never finer
# than MIN_STEP_M, and never more than MAX_STEPS_PER_SEGMENT steps long.
MIN_STEP_M = 0.005
MAX_STEPS_PER_SEGMENT = 512


def _steps_between(x0: float, y0: float, x1: float, y1: float,
                   robot_radius_m: float) -> int:
    """How many equal steps to walk the straight line between two samples.

    The step is half the robot's radius: the disc reaches ``robot_radius_m``
    sideways from the path, so a point the samples miss is at most that far
    from the line, and half a radius of spacing keeps the loss to a graze.
    """
    spacing = max(robot_radius_m / 2.0, MIN_STEP_M)
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0.0:
        return 1
    return max(1, min(MAX_STEPS_PER_SEGMENT, int(math.ceil(length / spacing))))


def _effective_radius(robot_radius_m: Any) -> float:
    """The robot's body radius, from a number or a RobotConfig-like object.

    ``RobotConfig`` may carry a chassis radius, or only an axle track to
    halve; anything else is read as a radius in metres.
    """
    if hasattr(robot_radius_m, "chassis_radius_m"):
        return float(robot_radius_m.chassis_radius_m())
    if hasattr(robot_radius_m, "track_width"):
        return float(robot_radius_m.track_width) / 2.0
    return float(robot_radius_m or 0.0)


def _point_segment_distance(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float,
) -> float:
    """Distance from a point to a segment."""
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def _segment_segment_distance(
    x0: float, y0: float, x1: float, y1: float,
    x2: float, y2: float, x3: float, y3: float,
) -> float:
    """Distance between two segments (Ericson, RTCD 5.1.9)."""
    d1x, d1y = x1 - x0, y1 - y0
    d2x, d2y = x3 - x2, y3 - y2
    rx, ry = x0 - x2, y0 - y2
    a = d1x * d1x + d1y * d1y
    e = d2x * d2x + d2y * d2y
    f = d2x * rx + d2y * ry
    if a <= 0.0 and e <= 0.0:
        return math.hypot(rx, ry)
    if a <= 0.0:
        s = 0.0
        t = min(1.0, max(0.0, f / e))
    else:
        c = d1x * rx + d1y * ry
        if e <= 0.0:
            t = 0.0
            s = min(1.0, max(0.0, -c / a))
        else:
            b = d1x * d2x + d1y * d2y
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = min(1.0, max(0.0, (b - c) / a))
    return math.hypot(x0 + d1x * s - (x2 + d2x * t),
                      y0 + d1y * s - (y2 + d2y * t))


def _point_box_signed(
    px: float, py: float, lo_x: float, hi_x: float, lo_y: float, hi_y: float,
) -> float:
    """Signed distance from a point to an axis-aligned box.

    Zero on the edge, positive outside, and inside the negative depth to
    the nearest edge - the rectangle branch of ``Obstacle.signed_distance``.
    """
    outside_x = max(lo_x - px, px - hi_x)
    outside_y = max(lo_y - py, py - hi_y)
    if outside_x > 0.0 or outside_y > 0.0:
        return math.hypot(max(outside_x, 0.0), max(outside_y, 0.0))
    return min(max(outside_x, outside_y), 0.0)


def _quad_aabb_signed(quad, half_w: float, half_h: float) -> float:
    """Signed distance between a convex quad and a centred axis-aligned box.

    The quad's corners are in cyclic order. Positive when disjoint (the
    distance between the shapes), negative when they overlap (minus the
    minimum translation that separates). The axes are the box's two and
    the quad's edge normals - the Minkowski difference's edge normals, so
    the penetration is exact for two convex polygons. A degenerate quad (a
    segment or a point) keeps its own exit distances on the same axes.
    """
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for (ax, ay), (bx, by) in zip(quad, list(quad[1:]) + [quad[0]]):
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        if length > 0.0:
            axes.append((-dy / length, dx / length))
    exits = []
    for nx, ny in axes:
        projections = [nx * px + ny * py for px, py in quad]
        q_lo, q_hi = min(projections), max(projections)
        extent = half_w * abs(nx) + half_h * abs(ny)
        if min(q_hi, extent) - max(q_lo, -extent) < 0.0:
            break
        exits.append(min(extent - q_lo, q_hi + extent))
    else:
        return -min(exits)
    # Disjoint: the nearest feature is an edge of one of the two shapes.
    best = math.inf
    for (qx0, qy0), (qx1, qy1) in zip(quad, list(quad[1:]) + [quad[0]]):
        for edge in ((-half_w, -half_h, half_w, -half_h),
                     (half_w, -half_h, half_w, half_h),
                     (half_w, half_h, -half_w, half_h),
                     (-half_w, half_h, -half_w, -half_h)):
            best = min(best, _segment_segment_distance(
                qx0, qy0, qx1, qy1, *edge))
    return best


def _segment_aabb_signed(
    x0: float, y0: float, x1: float, y1: float,
    half_w: float, half_h: float,
) -> float:
    """Signed distance between a segment and a centred axis-aligned box.

    Positive when disjoint (the distance between the shapes), negative when
    the segment penetrates (minus the minimum translation that separates).
    A degenerate segment is a point, and takes the point's own branch -
    the same math ``Obstacle.signed_distance`` uses for a rectangle.
    """
    if x0 == x1 and y0 == y1:
        return _point_box_signed(x0, y0, -half_w, half_w, -half_h, half_h)

    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)
    overlap_x = min(hi_x, half_w) - max(lo_x, -half_w)
    overlap_y = min(hi_y, half_h) - max(lo_y, -half_h)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    nx, ny = -dy / length, dx / length
    # The box's extent along the segment's own normal: the third axis of
    # the Minkowski difference (box swept along the segment), so a corner
    # clip that misses the box on x and y is still caught.
    overlap_normal = (half_w * abs(nx) + half_h * abs(ny)
                      - abs(nx * x0 + ny * y0))
    if overlap_x > 0.0 and overlap_y > 0.0 and overlap_normal > 0.0:
        # The minimum translation to separate: the exit distance along
        # each axis (a projection that contains the other needs the exit
        # form, not the overlap length).
        return -min(min(half_w - lo_x, hi_x + half_w),
                    min(half_h - lo_y, hi_y + half_h),
                    overlap_normal)
    # Disjoint: the nearest feature is an edge of the box.
    best = math.inf
    for edge in ((-half_w, -half_h, half_w, -half_h),
                 (half_w, -half_h, half_w, half_h),
                 (half_w, half_h, -half_w, half_h),
                 (-half_w, half_h, -half_w, -half_h)):
        best = min(best, _segment_segment_distance(x0, y0, x1, y1, *edge))
    return best


@dataclass(frozen=True)
class Body:
    """The heading-aware box the robot's body is at one pose.

    The rectangle ``[-back_bumper_m, +front_bumper_m] x [+/- width_m / 2]``
    at the sample's heading (counter-clockwise positive, the mat frame).
    Its circumscribed disc is exactly ``RobotConfig.chassis_radius_m()`` -
    ``hypot(bumper, width / 2)`` at the corners - and the gate measures
    this box, not that disc.
    """

    width_m: float
    front_bumper_m: float
    back_bumper_m: float

    @classmethod
    def from_config(cls, config: Any) -> "Body":
        """The body from a RobotConfig's own width and bumper numbers."""
        return cls(
            width_m=float(config.width_m),
            front_bumper_m=float(config.front_bumper_m),
            back_bumper_m=float(config.back_bumper_m),
        )

    def core(self, x: float, y: float, heading: float):
        """The forward-back segment at one pose (the zero-width body)."""
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        return (x - self.back_bumper_m * cos_h, y - self.back_bumper_m * sin_h,
                x + self.front_bumper_m * cos_h, y + self.front_bumper_m * sin_h)

    def corners(self, x: float, y: float, heading: float):
        """The rectangle's four corners at one pose, in cyclic order."""
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        half = self.width_m / 2.0
        out = []
        for fx, fy in ((-self.back_bumper_m, -half),
                       (self.front_bumper_m, -half),
                       (self.front_bumper_m, half),
                       (-self.back_bumper_m, half)):
            out.append((x + fx * cos_h - fy * sin_h,
                        y + fx * sin_h + fy * cos_h))
        return out


@dataclass(frozen=True)
class Obstacle:
    """One thing standing on the mat.

    Attributes:
        id: a short name, unique in the field.
        name: what it is, for the report.
        kind: ``rect`` or ``circle``.
        x_m: the centre, in the mat's frame.
        y_m: the centre, in the mat's frame.
        w_m: the width, or the diameter for a circle.
        h_m: the height, or the diameter for a circle.
        rot_deg: the rotation, for a rectangle.
    """

    id: str
    name: str
    kind: str
    x_m: float
    y_m: float
    w_m: float
    h_m: float
    rot_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("rect", "circle"):
            raise ValueError(
                f"obstacle {self.id}: kind must be rect or circle, "
                f"got {self.kind!r}")
        if self.w_m <= 0.0 or self.h_m <= 0.0:
            raise ValueError(
                f"obstacle {self.id}: a footprint needs a positive size")

    @property
    def radius_m(self) -> float:
        """The distance from the centre to the edge, for a circle."""
        return self.w_m / 2.0

    def _local(self, px: float, py: float):
        """A point in the rectangle's own frame (no-op frame for others)."""
        angle = math.radians(self.rot_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        dx = px - self.x_m
        dy = py - self.y_m
        return dx * cos_a + dy * sin_a, -dx * sin_a + dy * cos_a

    def signed_distance(self, px: float, py: float) -> float:
        """Distance from a point to the footprint, negative inside it.

        This is the same geometry ``overlap`` is built on: zero on the edge,
        positive outside, and inside the negative depth from the edge.
        """
        if self.kind == "circle":
            return math.hypot(px - self.x_m, py - self.y_m) - self.radius_m

        # A rectangle: turn the point into the rectangle's own frame, then
        # measure the distance to the rectangle.
        local_x, local_y = self._local(px, py)
        half_w = self.w_m / 2.0
        half_h = self.h_m / 2.0
        return _point_box_signed(local_x, local_y,
                                 -half_w, half_w, -half_h, half_h)

    def signed_distance_segment(
        self, x0: float, y0: float, x1: float, y1: float,
    ) -> float:
        """Distance from a segment to the footprint, negative when it enters.

        The zero-width body's own case: the convex signed distance - the
        distance between the shapes when they are apart, minus the minimum
        translation that separates them when they overlap. For a degenerate
        segment this is ``signed_distance`` of that point, so the two
        cannot disagree.
        """
        if self.kind == "circle":
            return (_point_segment_distance(self.x_m, self.y_m, x0, y0, x1, y1)
                    - self.radius_m)
        ax0, ay0 = self._local(x0, y0)
        ax1, ay1 = self._local(x1, y1)
        return _segment_aabb_signed(ax0, ay0, ax1, ay1,
                                    self.w_m / 2.0, self.h_m / 2.0)

    def signed_distance_body(
        self, x: float, y: float, heading: float, body: "Body",
    ) -> float:
        """Distance from the body box at one pose to the footprint.

        The heading-aware rectangle, exactly: negative when the body is
        inside the footprint. Degenerate bodies fall into the segment and
        point branches above, so the three cannot disagree.
        """
        if body.width_m <= 0.0:
            if body.front_bumper_m <= 0.0 and body.back_bumper_m <= 0.0:
                return self.signed_distance(x, y)
            x0, y0, x1, y1 = body.core(x, y, heading)
            return self.signed_distance_segment(x0, y0, x1, y1)
        if self.kind == "circle":
            # The circle's centre into the body frame: the distance to the
            # body's rectangle, less the circle's own radius.
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            dx = self.x_m - x
            dy = self.y_m - y
            half = body.width_m / 2.0
            return (_point_box_signed(dx * cos_h + dy * sin_h,
                                      -dx * sin_h + dy * cos_h,
                                      -body.back_bumper_m, body.front_bumper_m,
                                      -half, half)
                    - self.radius_m)
        # Two rectangles: the body's corners into this footprint's own
        # frame, then box against box.
        quad = [self._local(*corner) for corner in body.corners(x, y, heading)]
        return _quad_aabb_signed(quad, self.w_m / 2.0, self.h_m / 2.0)

    def overlap(self, px: float, py: float, robot_radius_m: float) -> float:
        """How deep a disc at ``(px, py)`` bites into this obstacle.

        Args:
            px: the point's x, in the mat's frame.
            py: the point's y, in the mat's frame.
            robot_radius_m: the radius of the disc.

        Returns:
            The overlap depth in metres. Zero means no overlap, and the
            number is how far inside the footprint the disc reached.
        """
        return max(0.0, robot_radius_m - self.signed_distance(px, py))

    def contains_point(self, px: float, py: float) -> bool:
        """Whether a point is inside the footprint."""
        return self.overlap(px, py, 0.0) > 0.0


@dataclass(frozen=True)
class Collision:
    """A planned sample that runs into something."""

    obstacle: Obstacle
    index: int
    t: float
    x_m: float
    y_m: float
    depth_m: float
    uncertain: bool

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-safe row for the report."""
        return {
            "id": self.obstacle.id,
            "name": self.obstacle.name,
            "index": self.index,
            "t": self.t,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "depth_m": self.depth_m,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True)
class FieldModel:
    """A mat and what stands on it."""

    season: str
    label: str
    mat_width_m: float
    mat_height_m: float
    placement_uncertainty_m: float
    obstacles: Tuple[Obstacle, ...]
    source: Dict[str, Any] = None  # type: ignore[assignment]
    mat_image: Dict[str, Any] = None  # type: ignore[assignment]
    mat_source: Dict[str, Any] = None  # type: ignore[assignment]
    launch_areas: Tuple[Dict[str, Any], ...] = ()

    def obstacle(self, obstacle_id: str) -> Optional[Obstacle]:
        """The obstacle with that id, or None."""
        for item in self.obstacles:
            if item.id == obstacle_id:
                return item
        return None

    def collisions(
        self,
        samples: List[Dict[str, Any]],
        robot_radius_m: Any = 0.0,
    ) -> List[Collision]:
        """Where the planned path runs into something.

        The robot is treated as a disc of ``robot_radius_m`` centred on the
        path, which is the axle-to-axle width of a differential drive, or
        computed from a ``RobotConfig`` footprint.

        The path *between* the samples is checked too. The samples are ten
        per waypoint segment, so their spacing is set by the length of the
        segment, not by the clock: a two-metre straight is sampled every
        200 mm. A model narrower than that can be driven straight through
        without any sample landing on it. The path is therefore walked at
        half the robot's radius, which is close enough that nothing the
        samples cover is missed.

        A collision shallower than the placement uncertainty is still
        reported, but marked ``uncertain``: the footprint's own position is
        not known to better than that, so it is not a hit.

        Args:
            samples: the planned samples, each with ``x``, ``y`` and
                optionally ``t``.
            robot_radius_m: the radius of the robot's body, or a RobotConfig.

        Returns:
            The collisions, in the order the samples appear. One entry per
            obstacle: the deepest overlap wins.
        """
        eff_radius = _effective_radius(robot_radius_m)

        found: Dict[str, Collision] = {}
        for index, sample in enumerate(samples):
            try:
                x = float(sample["x"])
                y = float(sample["y"])
            except (KeyError, TypeError, ValueError):
                continue
            self._check_point(found, index, x, y,
                              float(sample.get("t", 0.0) or 0.0),
                              eff_radius)

        # The path between the samples.
        for index in range(len(samples) - 1):
            first, second = samples[index], samples[index + 1]
            try:
                x0, y0 = float(first["x"]), float(first["y"])
                x1, y1 = float(second["x"]), float(second["y"])
            except (KeyError, TypeError, ValueError):
                continue
            t0 = float(first.get("t", 0.0) or 0.0)
            t1 = float(second.get("t", 0.0) or 0.0)
            steps = _steps_between(x0, y0, x1, y1, eff_radius)
            for step in range(1, steps):
                frac = step / steps
                self._check_point(
                    found, index,
                    x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac,
                    t0 + (t1 - t0) * frac,
                    eff_radius)
        return sorted(found.values(), key=lambda item: item.index)

    def sample_clearances(
        self,
        samples: List[Dict[str, Any]],
        robot_radius_m: Any = 0.0,
    ) -> List[float]:
        """One clearance in metres per trajectory sample.

        The clearance is the distance from the robot's disc - the sample's
        position, less ``robot_radius_m`` - to the nearest obstacle or to the
        nearest relevant field boundary. Negative means the disc penetrates
        that obstacle or boundary by that much.

        What is measured
        ----------------
        Obstacles use real geometry, not overlap-only tests: the signed
        distance to the circle or to the (possibly rotated) rectangle, less
        the robot's radius. The boundary is the **mat's own rectangle**
        (south-west corner ``(0, 0)``, north edge up). The table is bigger
        than the mat (2362 x 1143 against 2004 x 1143, the mat centred in
        it), so the mat edge is the conservative available boundary: the
        real wall is at least that far away. No wall or table coordinates are
        invented; a field whose mat size is not declared contributes no
        boundary at all.

        This is nominal geometry. The footprints are read off FIRST's
        wireframe and agree between readings only to about 100 mm (see the
        module docstring), so a clearance of a few centimetres is not a
        survey. The uncertainty is deliberately **not** subtracted here: the
        caller decides what to do with it.

        Unlike ``collisions``, only the samples themselves are measured; the
        path between two samples is not walked. A sample without a usable
        ``x``/``y`` is the one case ``collisions`` skips, and here it gets
        ``nan`` so the result is always one value per sample - a ``nan``
        clearance must never be read as clearance.

        Args:
            samples: the planned samples, each with ``x`` and ``y``.
            robot_radius_m: the radius of the robot's body, or a RobotConfig.

        Returns:
            One clearance in metres per sample, in the order they appear.
        """
        eff_radius = _effective_radius(robot_radius_m)
        clearances: List[float] = []
        for sample in samples:
            try:
                x = float(sample["x"])
                y = float(sample["y"])
            except (KeyError, TypeError, ValueError):
                clearances.append(float("nan"))
                continue
            best = self._wall_clearance([(x, y)]) - eff_radius
            for obstacle in self.obstacles:
                best = min(best, obstacle.signed_distance(x, y) - eff_radius)
            clearances.append(best)
        return clearances

    def _wall_clearance(self, points) -> float:
        """Distance from the mat's edge to the nearest of ``points``.

        The mat is the rectangle south-west ``(0, 0)``, north up - the
        conservative available boundary. Positive inside and negative once
        a point leaves the mat; ``inf`` when the mat size is not declared.
        """
        if self.mat_width_m <= 0.0 or self.mat_height_m <= 0.0:
            return math.inf
        return min(min(x, y, self.mat_width_m - x, self.mat_height_m - y)
                   for x, y in points)

    @staticmethod
    def _sample_pose(sample):
        """One sample's ``(x, y, heading)``, or None if it is malformed."""
        try:
            x = float(sample["x"])
            y = float(sample["y"])
            heading = float(sample["heading"])
        except (KeyError, TypeError, ValueError):
            return None
        return x, y, heading

    def body_clearances(self, samples, body: Body) -> List[float]:
        """One clearance per sample around the body box (the gate).

        The signed distance from the box's surface to the nearest mission
        model **or mat boundary**, negative inside: the space the body has
        around it. A sample without a usable ``x``/``y``/``heading`` gets
        ``nan`` - a nan clearance must never be read as clearance.
        """
        clearances: List[float] = []
        for sample in samples:
            pose = self._sample_pose(sample)
            if pose is None:
                clearances.append(float("nan"))
                continue
            x, y, heading = pose
            best = self._wall_clearance(body.corners(x, y, heading))
            for obstacle in self.obstacles:
                best = min(best, obstacle.signed_distance_body(
                    x, y, heading, body))
            clearances.append(best)
        return clearances

    def body_obstacles(self, samples, body: Body):
        """One ``(clearance, name)`` per sample, mission models only.

        The physical channel: the mat boundary is a game-rule boundary,
        not a crash - the chassis may legally overhang it - so it is not
        measured here. Negative means the body box is inside that
        obstacle, and the name says which one. A sample without a usable
        pose gets ``(nan, None)``; one with nothing on the field gets
        ``(inf, None)``.
        """
        out: List[Tuple[float, Optional[str]]] = []
        for sample in samples:
            pose = self._sample_pose(sample)
            if pose is None:
                out.append((float("nan"), None))
                continue
            x, y, heading = pose
            best = math.inf
            name = None
            for obstacle in self.obstacles:
                distance = obstacle.signed_distance_body(x, y, heading, body)
                if distance < best:
                    best = distance
                    name = obstacle.name
            out.append((best, name))
        return out

    def body_collisions(self, samples, body: Body) -> List[Collision]:
        """Where the body box runs into a mission model.

        The same body the gate measures, at the same samples, so the
        overlay and the gate cannot disagree about what touches what.
        One entry per obstacle, at its deepest sample. A collision
        shallower than the placement uncertainty is still reported, but
        marked ``uncertain``: the footprint's own position is not known
        to better than that.
        """
        found: Dict[str, Collision] = {}
        for index, sample in enumerate(samples):
            pose = self._sample_pose(sample)
            if pose is None:
                continue
            x, y, heading = pose
            for obstacle in self.obstacles:
                depth = -obstacle.signed_distance_body(x, y, heading, body)
                if depth <= 0.0:
                    continue
                existing = found.get(obstacle.id)
                if existing is not None and existing.depth_m >= depth:
                    continue
                found[obstacle.id] = Collision(
                    obstacle=obstacle,
                    index=index,
                    t=float(sample.get("t", 0.0) or 0.0),
                    x_m=float(sample["x"]),
                    y_m=float(sample["y"]),
                    depth_m=depth,
                    uncertain=depth <= self.placement_uncertainty_m,
                )
        return sorted(found.values(), key=lambda item: item.index)

    def sensor_hits(
        self,
        samples: List[Dict[str, Any]],
        sensor_offsets: Dict[str, Tuple[float, float]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Check where robot sensors pass over obstacles on the field.

        Args:
            samples: trajectory samples with 'x', 'y', and 'heading'.
            sensor_offsets: mapping of sensor name to (x_offset, y_offset) in
                the robot frame (metres).

        Returns:
            Dictionary mapping sensor name to list of hit obstacle dictionaries.
        """
        results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in sensor_offsets}
        for name, (dx, dy) in sensor_offsets.items():
            seen_obstacles = set()
            for sample in samples:
                try:
                    x = float(sample["x"])
                    y = float(sample["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                theta = float(sample.get("heading", 0.0) or 0.0)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                sx = x + dx * cos_t - dy * sin_t
                sy = y + dx * sin_t + dy * cos_t
                for obs in self.obstacles:
                    if obs.contains_point(sx, sy):
                        if obs.id not in seen_obstacles:
                            seen_obstacles.add(obs.id)
                            results[name].append({
                                "id": obs.id,
                                "name": obs.name,
                                "t": float(sample.get("t", 0.0) or 0.0),
                                "sensor_x": sx,
                                "sensor_y": sy,
                            })
        return results

    def _check_point(self, found, index, x, y, t, robot_radius_m):
        """Record the deepest overlap at one point of the path."""
        for obstacle in self.obstacles:
            depth = obstacle.overlap(x, y, robot_radius_m)
            if depth <= 0.0:
                continue
            existing = found.get(obstacle.id)
            if existing is not None and existing.depth_m >= depth:
                continue
            found[obstacle.id] = Collision(
                obstacle=obstacle,
                index=index,
                t=t,
                x_m=x,
                y_m=y,
                depth_m=depth,
                uncertain=depth <= self.placement_uncertainty_m,
            )

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-safe description, for the browser."""
        return {
            "season": self.season,
            "label": self.label,
            "mat": {
                "width_m": self.mat_width_m,
                "height_m": self.mat_height_m,
            },
            "placement_uncertainty_m": self.placement_uncertainty_m,
            "obstacles": [
                {
                    "id": item.id,
                    "name": item.name,
                    "kind": item.kind,
                    "x_m": item.x_m,
                    "y_m": item.y_m,
                    "w_m": item.w_m,
                    "h_m": item.h_m,
                    "rot_deg": item.rot_deg,
                }
                for item in self.obstacles
            ],
            "source": dict(self.source or {}),
            "mat_image": dict(self.mat_image or {}),
            "mat_source": dict(self.mat_source or {}),
            "launch_areas": [dict(area) for area in self.launch_areas],
        }


def _obstacle_from_dict(payload: Dict[str, Any]) -> Obstacle:
    return Obstacle(
        id=str(payload["id"]),
        name=str(payload.get("name", payload["id"])),
        kind=str(payload.get("kind", "rect")),
        x_m=float(payload["x_m"]),
        y_m=float(payload["y_m"]),
        w_m=float(payload["w_m"]),
        h_m=float(payload["h_m"]),
        rot_deg=float(payload.get("rot_deg", 0.0) or 0.0),
    )


def field_from_dict(payload: Dict[str, Any]) -> FieldModel:
    """Build a field model from its JSON description."""
    mat = payload.get("mat", {}) or {}
    obstacles = tuple(
        _obstacle_from_dict(item)
        for item in payload.get("obstacles", []) or [])
    return FieldModel(
        season=str(payload.get("season", "")),
        label=str(payload.get("label", payload.get("season", ""))),
        mat_width_m=float(mat.get("width_m", 0.0) or 0.0),
        mat_height_m=float(mat.get("height_m", 0.0) or 0.0),
        placement_uncertainty_m=float(
            payload.get("placement_uncertainty_m", 0.0) or 0.0),
        obstacles=obstacles,
        source=dict(payload.get("source", {}) or {}),
        mat_image=dict(payload.get("mat_image", {}) or {}),
        mat_source=dict(payload.get("mat_source", {}) or {}),
        launch_areas=tuple(
            dict(area) for area in payload.get("launch_areas", []) or []),
    )


def list_fields() -> List[Dict[str, str]]:
    """The fields this repository knows about.

    The rows come from the fields that load, so the list and the models
    can never disagree about which fields exist.

    Returns:
        One row per season file: its id and its label.
    """
    return [
        {"id": model.season, "label": model.label}
        for model in load_all_fields().values()
    ]


def load_field(season: str) -> Optional[FieldModel]:
    """Load one field by its season id, or None when there is no such file."""
    if not season:
        return None
    safe = "".join(
        ch for ch in str(season) if ch.isalnum() or ch in "-_").strip()
    if not safe:
        return None
    path = os.path.join(FIELDS_DIR, f"{safe}.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return field_from_dict(json.load(handle))
    # The file may be named after the season and its edition, so fall back to
    # looking at what the files say they are.
    if not os.path.isdir(FIELDS_DIR):
        return None
    for name in sorted(os.listdir(FIELDS_DIR)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(FIELDS_DIR, name), "r",
                      encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if str(payload.get("season", "")) == str(season):
            return field_from_dict(payload)
    return None


def load_all_fields() -> Dict[str, FieldModel]:
    """Every field this repository knows about, by season id."""
    result: Dict[str, FieldModel] = {}
    if not os.path.isdir(FIELDS_DIR):
        return result
    for name in sorted(os.listdir(FIELDS_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(FIELDS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            model = field_from_dict(payload)
        except (OSError, ValueError, KeyError):
            continue
        result[model.season or os.path.splitext(name)[0]] = model
    return result
