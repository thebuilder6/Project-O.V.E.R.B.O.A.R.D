"""
Unit tests for the Unified Path Planner (unified_planner.py).
"""

from __future__ import annotations

import unittest
import numpy as np

from overboard.models.robot_model import RobotConfig
from overboard.core.unified_planner import UnifiedPathEngine, PlannerCandidateResult, UnifiedPlanResult


class TestUnifiedPlanner(unittest.TestCase):
    """Tests for the 3-Pillar Unified Trajectory Planner."""

    def setUp(self):
        self.config = RobotConfig({})
        self.engine = UnifiedPathEngine(
            robot_config=self.config,
            control_dt=0.02,
            tracking_gate_m=0.005,
            envelope_gate_m=0.020,
        )

    def test_init(self):
        self.assertEqual(self.engine.control_dt, 0.02)
        self.assertEqual(self.engine.tracking_gate_m, 0.005)
        self.assertEqual(self.engine.envelope_gate_m, 0.020)

    def test_compute_invariant_mesh_samples(self):
        wps = [(0.0, 0.0, 0.0), (0.2, 0.1, None), (0.8, 0.0, 0.0)]
        samples = self.engine.compute_invariant_mesh_samples(wps, min_samples=10)
        self.assertEqual(len(samples), 2)
        for s in samples:
            self.assertGreaterEqual(s, 10)
            self.assertLessEqual(s, 18)

    def test_generate_seed(self):
        wps = [(0.0, 0.0, 0.0), (0.4, 0.2, None), (0.8, 0.0, 0.0)]
        seg_samples = [10, 10]
        for style in ["sprint", "sweeper", "exploratory"]:
            seed_arr = self.engine.generate_seed(wps, seg_samples, style=style)
            self.assertTrue(isinstance(seed_arr, np.ndarray))
            self.assertTrue(np.all(np.isfinite(seed_arr)))
            self.assertEqual(len(seed_arr), len(seg_samples) + (sum(seg_samples) + 1) * 5)

    def test_solve_candidate_simple(self):
        wps = [(0.0, 0.0, 0.0), (0.5, 0.0, None), (1.0, 0.0, 0.0)]
        seg_samples = [8, 8]
        seed_arr = self.engine.generate_seed(wps, seg_samples, style="sprint")
        cand = self.engine.solve_candidate(
            waypoints=wps,
            seg_samples=seg_samples,
            name="Test Sprint",
            guess=seed_arr,
            accuracy_weight=0.002,
            stop_indices=[0, 2],
            max_iter=150,
        )
        self.assertIsInstance(cand, PlannerCandidateResult)
        self.assertGreater(cand.duration_s, 0.0)
        self.assertTrue(len(cand.samples) > 0)
        self.assertTrue(cand.iterations > 0)

    def test_plan_end_to_end(self):
        wps = [(0.0, 0.0, 0.0), (0.4, 0.1, None), (0.8, 0.0, 0.0)]
        result = self.engine.plan(
            waypoints=wps,
            mesh_samples=[8, 8],
            verbose=False,
        )
        self.assertIsInstance(result, UnifiedPlanResult)
        self.assertIn(result.status, ["CERTIFIED", "CONVERGED_UNCERTIFIED"])
        self.assertIsNotNone(result.champion)
        self.assertGreater(result.champion.duration_s, 0.0)
        self.assertGreater(len(result.candidates), 0)


if __name__ == "__main__":
    unittest.main()
