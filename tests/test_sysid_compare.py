"""Tests for the measured-pose comparison report."""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest

from overboard.sysid.compare import compare, main


def _trajectory():
    return {
        "name": "test",
        "version": 3,
        "trajectory": {
            "config": {},
            "samples": [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0},
                {"t": 1.0, "x": 0.1, "y": 0.0, "heading": 0.0},
                {"t": 2.0, "x": 0.2, "y": 0.0, "heading": 0.0,
                 "reachability_envelope": {
                     "x_min": 0.19, "x_max": 0.21,
                     "y_min": -0.01, "y_max": 0.01,
                 }},
                {"t": 3.0, "x": 0.3, "y": 0.0, "heading": 0.0},
                {"t": 4.0, "x": 0.4, "y": 0.0, "heading": math.pi},
            ],
        },
    }


def _waypoints():
    return [
        {"x": 0.0, "y": 0.0, "heading": 0.0, "stop": True},
        {"x": 0.2, "y": 0.0, "heading": 0.0, "stop": True},
        {"x": 0.4, "y": 0.0, "heading": 3.14},
    ]


class TestSysIdCompare(unittest.TestCase):
    def _write(self, tmp, poses_text, config=None):
        traj_path = os.path.join(tmp, "out.traj")
        poses_path = os.path.join(tmp, "poses.csv")
        waypoints_path = os.path.join(tmp, "waypoints.json")
        trajectory = _trajectory()
        trajectory["trajectory"]["config"] = config or {}
        with open(traj_path, "w", encoding="utf-8") as handle:
            json.dump(trajectory, handle)
        with open(poses_path, "w", encoding="utf-8") as handle:
            handle.write(poses_text)
        with open(waypoints_path, "w", encoding="utf-8") as handle:
            json.dump(_waypoints(), handle)
        return traj_path, poses_path, waypoints_path

    def test_final_pose_fallback_for_single_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "4,0,0.402,0.003,-179.0\n",
            )
            report = compare(traj, poses)
        self.assertEqual(report["run_id"], "4")
        self.assertEqual(len(report["rows"]), 1)
        self.assertEqual(report["rows"][0]["sample_index"], 4)
        self.assertAlmostEqual(
            report["rows"][0]["position_error_m"],
            math.hypot(0.002, 0.003),
            places=9,
        )
        # predicted pi vs measured -179 deg: shortest error is -1 deg.
        self.assertAlmostEqual(
            math.degrees(report["rows"][0]["heading_error_rad"]),
            -1.0,
            delta=0.2,
        )

    def test_waypoint_mapping_includes_final_and_skips_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, waypoints = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "12,0,0.2,0.0,0.0\n"
                "12,1,0.4,0.0,180.0\n",
            )
            report = compare(traj, poses, run_id="12",
                             waypoints_path=waypoints)
        self.assertEqual(
            [row["sample_index"] for row in report["rows"]], [2, 4])
        self.assertTrue(report["rows"][0]["inside_envelope"])
        self.assertIsNone(report["rows"][1]["inside_envelope"])

    def test_explicit_stop_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "7,0,0.105,0.0,0.0\n"
                "7,1,0.3,0.0,0.0\n",
            )
            report = compare(traj, poses, run_id="7", stop_indices="1,3")
        self.assertEqual(
            [row["sample_index"] for row in report["rows"]], [1, 3])

    def test_outside_envelope_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "1,0,0.25,0.0,0.0\n",
            )
            report = compare(traj, poses, run_id="1",
                             stop_indices="2")
        self.assertFalse(report["rows"][0]["inside_envelope"])

    def test_multiple_runs_require_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "1,0,0.4,0.0,0.0\n"
                "2,0,0.4,0.0,0.0\n",
            )
            with self.assertRaises(ValueError):
                compare(traj, poses)

    def test_explicit_index_count_must_match_pose_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "1,0,0.4,0.0,0.0\n"
                "1,1,0.4,0.0,0.0\n",
            )
            with self.assertRaises(ValueError):
                compare(traj, poses, run_id="1", stop_indices="4")

    def test_predicted_pose_csv_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "9,0,0.2,0.0,180.0\n",
            )
            predicted = os.path.join(tmp, "predicted.csv")
            with open(predicted, "w", encoding="utf-8") as handle:
                handle.write(
                    "t_ms,x,y,heading_deg\n"
                    "0,0.0,0.0,0\n"
                    "1000,0.1,0.0,0\n"
                    "2000,0.2,0.0,180\n"
                )
            report = compare(predicted, poses)
        self.assertEqual(report["run_id"], "9")
        self.assertEqual(report["rows"][0]["sample_index"], 2)
        self.assertAlmostEqual(report["rows"][0]["t"], 2.0)
        self.assertAlmostEqual(report["rows"][0]["position_error_m"], 0.0)

    def test_profile_geometry_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "4,0,0.402,0.003,-179.0\n",
                config={"wheel_radius": 0.03, "track_width": 0.0965},
            )
            report = compare(
                traj, poses,
                profile={
                    "name": "test drivebase",
                    "wheel_radius_m": 0.028,
                    "track_width_m": 0.0965,
                },
            )
        self.assertEqual(report["drivebase"]["name"], "test drivebase")
        self.assertEqual(len(report["geometry_warnings"]), 1)
        self.assertIn("wheel radius", report["geometry_warnings"][0])

    def test_cli_prints_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "4,0,0.402,0.003,-179.0\n",
            )
            profile = os.path.join(tmp, "drivebase.json")
            with open(profile, "w", encoding="utf-8") as handle:
                json.dump({
                    "name": "test drivebase",
                    "wheel_radius_m": 0.028,
                    "track_width_m": 0.0965,
                }, handle)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main([traj, poses, "--profile", profile])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Drivebase: test drivebase", text)
        self.assertIn("wheel radius 28.00 mm", text)
        self.assertIn("track width 96.50 mm", text)

    def test_cli_prints_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "4,0,0.402,0.003,-179.0\n",
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main([traj, poses])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("run_id=4", text)
        self.assertIn("Max position error", text)
        self.assertIn("mm", text)

    def test_cli_returns_nonzero_for_bad_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            traj, poses, _ = self._write(
                tmp,
                "run_id,stop,x_m,y_m,heading_deg\n"
                "4,0,0.402,0.003,-179.0\n"
                "4,1,0.402,0.003,-179.0\n",
            )
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main([traj, poses])
        self.assertEqual(code, 1)
        self.assertIn("multiple measured poses", err.getvalue())


if __name__ == "__main__":
    unittest.main()
