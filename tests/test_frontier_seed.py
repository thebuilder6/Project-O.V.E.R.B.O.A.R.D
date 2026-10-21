"""
Unit tests for the Kinetic Frontier Analytical Seed Generator (frontier_seed.py).
"""

from __future__ import annotations

import math
import unittest
import numpy as np

from overboard.models.robot_model import RobotConfig
from overboard.core.frontier_seed import (
    compute_tangent_headings,
    _asymmetric_late_apex_bump,
    generate_frontier_seed,
)


class TestFrontierSeed(unittest.TestCase):
    """Tests for the analytical kinetic frontier seed generator."""

    def setUp(self):
        self.config = RobotConfig({})

    def test_compute_tangent_headings_straight(self):
        wps = [(0.0, 0.0, None), (0.5, 0.0, None), (1.0, 0.0, None)]
        headings = compute_tangent_headings(wps)
        self.assertEqual(len(headings), 3)
        for h in headings:
            self.assertAlmostEqual(h, 0.0, places=6)

    def test_compute_tangent_headings_corner(self):
        wps = [(0.0, 0.0, None), (1.0, 0.0, None), (1.0, 1.0, None)]
        headings = compute_tangent_headings(wps)
        self.assertAlmostEqual(headings[0], 0.0, places=6)
        self.assertAlmostEqual(headings[1], math.pi / 4.0, places=6)
        self.assertAlmostEqual(headings[2], math.pi / 2.0, places=6)

    def test_compute_tangent_headings_pinned(self):
        wps = [(0.0, 0.0, 0.1), (0.5, 0.5, 0.5), (1.0, 1.0, None)]
        headings = compute_tangent_headings(wps)
        self.assertAlmostEqual(headings[0], 0.1, places=6)
        self.assertAlmostEqual(headings[1], 0.5, places=6)
        self.assertAlmostEqual(headings[2], math.pi / 4.0, places=6)

    def test_asymmetric_late_apex_bump(self):
        u = np.linspace(0.0, 1.0, 101)
        bump = _asymmetric_late_apex_bump(u, u_peak=0.68)
        self.assertEqual(bump[0], 0.0)
        self.assertEqual(bump[-1], 0.0)
        self.assertTrue(np.all(bump >= 0.0))
        # Peak should be at or near u = 0.68
        peak_idx = int(np.argmax(bump))
        peak_u = u[peak_idx]
        self.assertAlmostEqual(peak_u, 0.68, delta=0.02)
        self.assertAlmostEqual(bump[peak_idx], 1.0, delta=0.01)

    def test_generate_frontier_seed_shape(self):
        wps = [(0.0, 0.0, 0.0), (0.4, 0.2, None), (0.8, 0.0, 0.0)]
        seg_samples = [10, 12]
        seed = generate_frontier_seed(
            wps,
            segment_samples=seg_samples,
            robot_config=self.config,
            style="sprint",
        )
        total_samples = sum(seg_samples) + 1
        expected_len = len(seg_samples) + total_samples * 5
        self.assertEqual(len(seed), expected_len)
        self.assertTrue(np.all(np.isfinite(seed)))

    def test_generate_frontier_seed_styles(self):
        wps = [(0.0, 0.0, 0.0), (0.4, 0.2, None), (0.8, 0.0, 0.0)]
        seg_samples = [8, 8]
        styles = ["sprint", "sweeper", "exploratory", "direct_chord"]
        for st in styles:
            seed = generate_frontier_seed(
                wps,
                segment_samples=seg_samples,
                robot_config=self.config,
                style=st,
            )
            self.assertTrue(np.all(np.isfinite(seed)), f"Style {st} produced non-finite values")
            # First 2 values are dts, must be positive
            dts = seed[:2]
            self.assertTrue(np.all(dts > 0.0), f"Style {st} dts not positive")

    def test_stop_waypoint_velocity_zero(self):
        wps = [(0.0, 0.0, 0.0), (0.5, 0.0, None), (1.0, 0.0, 0.0)]
        seg_samples = [10, 10]
        # Intermediate stop at waypoint 1
        seed = generate_frontier_seed(
            wps,
            segment_samples=seg_samples,
            robot_config=self.config,
            stop_waypoint_indices=[0, 1, 2],
            style="sprint",
        )
        # Parse states [x, y, theta, vl, vr]
        num_segs = 2
        dts = seed[:num_segs]
        states = seed[num_segs:].reshape(-1, 5)
        # Waypoint 0 (index 0), Waypoint 1 (index 10), Waypoint 2 (index 20)
        for wp_idx in [0, 10, 20]:
            vl = states[wp_idx, 3]
            vr = states[wp_idx, 4]
            self.assertAlmostEqual(vl, 0.0, delta=1e-3, msg=f"vl at wp {wp_idx} not zero")
            self.assertAlmostEqual(vr, 0.0, delta=1e-3, msg=f"vr at wp {wp_idx} not zero")

    def test_momentum_preservation_unpinned(self):
        wps = [(0.0, 0.0, 0.0), (0.5, 0.0, None), (1.0, 0.0, 0.0)]
        seg_samples = [10, 10]
        seed = generate_frontier_seed(
            wps,
            segment_samples=seg_samples,
            robot_config=self.config,
            stop_waypoint_indices=[0, 2],  # WP 1 is not a stop
            style="sprint",
        )
        states = seed[2:].reshape(-1, 5)
        # Waypoint 1 is at index 10
        vl_wp1 = states[10, 3]
        vr_wp1 = states[10, 4]
        v_lin = 0.5 * (vl_wp1 + vr_wp1)
        v_max = float(self.config.max_linear_speed())
        # Rule 3 says momentum is preserved through non-stop waypoints (should cruise > 0.4 v_max)
        self.assertGreater(v_lin, 0.4 * v_max)

    def test_kinematic_diamond_boundary_saturation(self):
        wps = [(0.0, 0.0, 0.0), (0.4, 0.3, None), (0.8, 0.0, 0.0)]
        seg_samples = [15, 15]
        seed = generate_frontier_seed(
            wps,
            segment_samples=seg_samples,
            robot_config=self.config,
            style="sprint",
        )
        states = seed[2:].reshape(-1, 5)
        B = float(self.config.track_width)
        v_wheel_max = float(self.config.max_linear_speed())

        # Check every point respects the kinematic diamond: |v| + (B/2)|omega| = max(|vl|, |vr|) <= v_wheel_max
        for idx in range(len(states)):
            vl = states[idx, 3]
            vr = states[idx, 4]
            v = 0.5 * (vl + vr)
            omega = (vr - vl) / B
            diamond_norm = abs(v) + 0.5 * B * abs(omega)
            self.assertLessEqual(
                diamond_norm,
                v_wheel_max + 1e-3,
                f"Diamond violation at index {idx}: {diamond_norm} > {v_wheel_max}",
            )


    def test_baseline_seed_is_the_straight_chord_at_rest(self):
        """The baseline style is the chord with the wheels stopped."""
        wps = [(0.0, 0.0, 0.0), (0.4, 0.2, None), (0.8, 0.0, 0.0)]
        seg_samples = [10, 12]
        seed = generate_frontier_seed(
            wps,
            segment_samples=seg_samples,
            robot_config=self.config,
            style="baseline",
        )
        num_segs = len(seg_samples)
        dts = seed[:num_segs]
        states = seed[num_segs:].reshape(-1, 5)
        self.assertTrue(np.all(dts > 0.0))
        # The wheel speeds are zero everywhere: the baseline says where the
        # robot is, not how it gets there.
        self.assertTrue(np.allclose(states[:, 3], 0.0))
        self.assertTrue(np.allclose(states[:, 4], 0.0))
        # Every sample lies on the chord of its segment.
        for i in range(num_segs):
            p1, p2 = np.array(wps[i][:2]), np.array(wps[i + 1][:2])
            start = sum(seg_samples[:i])
            seg = states[start:start + seg_samples[i], :2]
            rel = seg - p1
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length > 1e-9:
                cross = np.abs(rel[:, 0] * direction[1] - rel[:, 1] * direction[0])
                self.assertTrue(np.all(cross / length < 1e-9), f"sample off the chord in segment {i}")


if __name__ == "__main__":
    unittest.main()
