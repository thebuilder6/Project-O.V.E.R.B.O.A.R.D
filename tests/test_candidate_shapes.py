"""
Q17 candidate shapes (docs/RESEARCH_AGENDA.md §24): the mirrored bow, the
bezier geometry, reverse seeding, and the candidate list they sit in.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from overboard.core.frontier_seed import GUESS_STYLES, generate_frontier_seed
from overboard.core.unified_planner import UnifiedPathEngine
from overboard.models.robot_model import RobotConfig


def _states(seed, num_segments):
    """Split a seed vector into (dts, states) rows of (x, y, heading, vl, vr)."""
    states = np.asarray(seed[num_segments:], dtype=float).reshape(-1, 5)
    return states


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class TestMirrorBow(unittest.TestCase):
    """frontier_sprint_mirror flips the sign of the apex amplitude."""

    def setUp(self):
        self.config = RobotConfig({})
        # A corner at the middle waypoint, so the apex amplitudes are nonzero.
        self.wps = [(0.0, 0.0, 0.0), (0.4, 0.2, None), (0.8, 0.0, 0.0)]
        self.seg_samples = [10, 10]

    def _seed(self, style):
        return generate_frontier_seed(
            self.wps, segment_samples=self.seg_samples,
            robot_config=self.config, style=style)

    def test_apex_amplitude_sign_flips(self):
        plain = _states(self._seed("frontier_sprint"), 2)
        mirror = _states(self._seed("frontier_sprint_mirror"), 2)

        # Signed lateral displacement from the first chord, sample by sample.
        chord = np.array([0.4, 0.2])
        chord_u = chord / np.linalg.norm(chord)
        normal = np.array([-chord_u[1], chord_u[0]])
        start = np.array([0.0, 0.0])
        worst = 0.0
        for k in range(self.seg_samples[0]):
            u = k / self.seg_samples[0]
            on_chord = start + chord * u
            lat_plain = float(np.dot(plain[k, :2] - on_chord, normal))
            lat_mirror = float(np.dot(mirror[k, :2] - on_chord, normal))
            self.assertAlmostEqual(lat_mirror, -lat_plain, places=9,
                                   msg=f"sample {k}: amplitudes did not flip")
            worst = max(worst, abs(lat_plain))
        self.assertGreater(worst, 1e-4,
                           "fixture never left the chord; the flip is unproven")

        # The flip is the only difference: same path length budget and pins.
        self.assertTrue(np.allclose(plain[[0, 10, 20], :2],
                                    mirror[[0, 10, 20], :2]))


class TestSplineGeometry(unittest.TestCase):
    """frontier_spline seeds a bezier path through the waypoints."""

    def setUp(self):
        self.config = RobotConfig({})
        # An L: the bezier bows off the chords at the corner.
        self.wps = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, math.pi / 2.0)]
        self.seg_samples = [10, 10]

    def _seed(self, style):
        return generate_frontier_seed(
            self.wps, segment_samples=self.seg_samples,
            robot_config=self.config, style=style)

    def test_path_passes_through_the_waypoints_and_differs_from_the_straight_seed(self):
        spline = _states(self._seed("frontier_spline"), 2)
        straight = _states(self._seed("frontier_sprint"), 2)

        for idx, wp in zip((0, 10, 20), self.wps):
            self.assertAlmostEqual(float(spline[idx, 0]), wp[0], places=9)
            self.assertAlmostEqual(float(spline[idx, 1]), wp[1], places=9)

        # The path family is a real shape, not a re-timed chord.
        offsets = np.hypot(spline[:, 0] - straight[:, 0],
                           spline[:, 1] - straight[:, 1])
        self.assertGreater(float(np.max(offsets)), 0.005)


class TestReverseSeeding(unittest.TestCase):
    """frontier_reverse backs through segments whose pinned heading faces
    more than 120 degrees off the chord, and leaves the rest alone."""

    def setUp(self):
        self.config = RobotConfig({})
        # Segment 0 is reverse: wp0 is pinned 180 degrees off its chord.
        # Segment 1 is forward: wp2 is pinned along its chord.
        self.wps = [(0.0, 0.0, math.pi), (1.0, 0.0, None), (2.0, 0.0, 0.0)]
        self.seg_samples = [4, 4]

    def _seed(self, style):
        return generate_frontier_seed(
            self.wps, segment_samples=self.seg_samples,
            robot_config=self.config, style=style)

    def test_reverse_segments_get_heading_plus_pi_and_negative_v(self):
        reverse = _states(self._seed("frontier_reverse"), 2)
        plain = _states(self._seed("frontier_sprint"), 2)

        # Exactly + pi and v < 0 on the reverse segment (samples 1-3; sample 0
        # is the path start, at rest by construction).
        for k in range(1, 4):
            self.assertAlmostEqual(
                abs(_wrap(float(reverse[k, 2]) - float(plain[k, 2]))), math.pi,
                places=6, msg=f"sample {k}: heading is not plain + pi")
            self.assertLess(float(reverse[k, 3]), 0.0, f"sample {k}: vl not < 0")
            self.assertLess(float(reverse[k, 4]), 0.0, f"sample {k}: vr not < 0")

        # Untouched elsewhere: no flip on the forward segment (samples 5-7,
        # away from the boundary), and the path geometry never changes.
        for k in range(5, 8):
            self.assertAlmostEqual(
                _wrap(float(reverse[k, 2]) - float(plain[k, 2])), 0.0,
                places=6, msg=f"sample {k}: forward sample was flipped")
            self.assertGreater(float(reverse[k, 3]), 0.0, f"sample {k}: vl not > 0")
            self.assertGreater(float(reverse[k, 4]), 0.0, f"sample {k}: vr not > 0")
        self.assertTrue(np.allclose(reverse[:, :2], plain[:, :2]))


class TestCandidateSpecs(unittest.TestCase):
    """The candidate list is positional: the triad first, then the Q17 tail."""

    TRIAD = ["frontier_sprint", "frontier_sweeper", "frontier_exploratory"]
    TAIL = ["frontier_sprint_mirror", "frontier_spline", "frontier_reverse"]

    def setUp(self):
        self.planner = UnifiedPathEngine(RobotConfig({}))

    def test_ordering_and_count_slicing(self):
        specs = self.planner.candidate_specs()
        self.assertEqual([s["guess_style"] for s in specs],
                         self.TRIAD + self.TAIL)
        self.assertEqual([s["guess_style"] for s in self.planner.candidate_specs(count=3)],
                         self.TRIAD)
        self.assertEqual([s["guess_style"] for s in self.planner.candidate_specs(count=6)],
                         self.TRIAD + self.TAIL)
        self.assertEqual([s["guess_style"] for s in self.planner.candidate_specs(count=2)],
                         self.TRIAD[:2])
        # Every spec is a style the optimizer's guess builder accepts.
        for spec in specs:
            self.assertIn(spec["guess_style"], GUESS_STYLES)


if __name__ == "__main__":
    unittest.main()
