"""Tests for the Q25 trajectory-oracle comparison (compare_traj.py)."""

import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "crosscheck"))

import compare_traj  # noqa: E402


class TestOracleMatching(unittest.TestCase):
    def test_interpolates_position_and_short_way_heading(self):
        rows = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 3.0},
            {"t": 1.0, "x": 2.0, "y": 0.0, "heading": -3.0},
        ]
        mid = compare_traj._oracle_at(rows, 0.5)
        self.assertAlmostEqual(mid["x"], 1.0)
        self.assertAlmostEqual(mid["y"], 0.0)
        # Short way: 3.0 -> -3.0 crosses pi; the midpoint is ~pi or ~-pi.
        self.assertAlmostEqual(abs(mid["heading"]), math.pi, places=6)

    def test_clamps_outside_the_trace(self):
        rows = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
            {"t": 1.0, "x": 1.0, "y": 0.0, "heading": 0.0},
        ]
        self.assertEqual(compare_traj._oracle_at(rows, -1.0)["x"], 0.0)
        self.assertEqual(compare_traj._oracle_at(rows, 2.0)["x"], 1.0)


class TestComparePair(unittest.TestCase):
    def test_metrics_and_containment(self):
        oracle = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
            {"t": 1.0, "x": 1.0, "y": 0.0, "heading": 0.0},
        ]
        twin = [
            {"t": 0.0, "x": 0.001, "y": 0.0, "heading": 0.0, "tube": 0.002},
            {"t": 0.5, "x": 0.5, "y": 0.010, "heading": 0.02, "tube": 0.005},
            {"t": 1.0, "x": 1.0, "y": 0.0, "heading": 0.0, "tube": None},
        ]
        result = compare_traj.compare_pair(twin, oracle)
        self.assertAlmostEqual(result["e_pos_m"], 0.010, places=6)
        self.assertAlmostEqual(result["e_head_rad"], 0.02, places=6)
        # Sample 0 is inside its 2 mm tube, sample 1 is outside its 5 mm tube:
        # 1 of 2 tubes contain the oracle sample.
        self.assertAlmostEqual(result["containment"], 0.5)
        self.assertEqual(result["n_tubes"], 2)
        self.assertAlmostEqual(result["worst"]["t"], 0.5)

    def test_no_tubes_means_no_containment(self):
        oracle = [{"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0}]
        twin = [{"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "tube": None}]
        self.assertIsNone(compare_traj.compare_pair(twin, oracle)["containment"])


class TestCliSummary(unittest.TestCase):
    def _write(self, path, rows):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(rows)

    def test_json_summary_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            twin = os.path.join(tmp, "twin.csv")
            oracle = os.path.join(tmp, "oracle.csv")
            self._write(twin, "t,x,y,heading,tube_radius_m\n"
                             "0.0,0.0,0.0,0.0,0.01\n"
                             "0.5,0.5,0.0,0.0,0.01\n")
            self._write(oracle, "t_ms,true_x_m,true_y_m,true_heading_rad\n"
                                "0,0.0,0.0,0.0\n"
                                "250,0.25,0.0,0.0\n"
                                "500,0.5,0.0,0.0\n")
            out = StringIO()
            with redirect_stdout(out):
                code = compare_traj.main([twin, oracle])
            self.assertEqual(code, 0)
            line = out.getvalue().strip().splitlines()[-1]
            summary = json.loads(line)
            self.assertEqual(summary["n_trajectories"], 1)
            self.assertAlmostEqual(summary["mean_e_pos_m"], 0.0, places=6)
            self.assertAlmostEqual(summary["containment"], 1.0)


if __name__ == "__main__":
    unittest.main()
