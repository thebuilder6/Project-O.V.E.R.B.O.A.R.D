"""Synthetic parameter-recovery tests for the SysId log fitter."""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest

import numpy as np

from sysid_fit import (
    fit_session,
    load_poses,
    main as sysid_main,
    parse_session,
    poses_from_hub_session,
    read_drivebase_profile,
    write_poses_csv,
)


def _first_order(times, command, gain, tau_s, delay_s):
    delayed = np.interp(times - delay_s, times, command, left=command[0])
    output = np.zeros_like(delayed)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        alpha = 1.0 - math.exp(-dt / tau_s)
        output[i] = output[i - 1] + alpha * (gain * delayed[i - 1] - output[i - 1])
    return output


def _session(runs):
    lines = []
    for name, run_id, header, rows in runs:
        lines.append("TEST_START " + name)
        lines.append("META,run_id," + str(run_id))
        lines.append("META,test," + name)
        lines.append("DATA")
        lines.append(header)
        for row in rows:
            lines.append(",".join(str(value) for value in row))
        lines.append("TEST_END," + str(len(rows)))
    return "\n".join(lines)


def _straight_run(name, run_id, gain=0.95, tau_s=0.08, delay_s=0.02):
    times = np.arange(0.0, 4.0, 0.01)
    command = np.zeros_like(times)
    command[(times >= 0.2) & (times < 1.2)] = 0.4
    command[(times >= 1.6) & (times < 2.6)] = -0.3
    speed = _first_order(times, command, gain, tau_s, delay_s)
    angle_deg = np.cumsum(speed) * 0.01 / 0.028 * 180.0 / math.pi
    rows = [
        [t * 1000.0, cmd * 1000.0, 0.0, angle_deg[i], angle_deg[i]]
        for i, (t, cmd) in enumerate(zip(times, command))
    ]
    return name, run_id, "t_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg", rows


def _turn_run(name, run_id, gain=0.95, tau_s=0.08, delay_s=0.02):
    times = np.arange(0.0, 4.0, 0.01)
    command_deg = np.zeros_like(times)
    command_deg[(times >= 0.2) & (times < 1.2)] = 90.0
    command_deg[(times >= 1.6) & (times < 2.6)] = -60.0
    command_rad = command_deg * math.pi / 180.0
    turn_rate = _first_order(times, command_rad, gain, tau_s, delay_s)
    rows = [
        [t * 1000.0, cmd, -turn_rate[i]]
        for i, (t, cmd) in enumerate(zip(times, command_deg))
    ]
    return name, run_id, "t_ms,w_cmd_deg_s,gyro_z_rad_s", rows


class TestSysIdFit(unittest.TestCase):
    def test_parameter_recovery(self):
        runs = [
            _straight_run("straight_steps_fwd", 1),
            _turn_run("turn_steps_cw", 2),
            (
                "odometry_straight",
                3,
                "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,"
                "db_a_deg,db_w_deg_s,batt_mv",
                [[10000, 100, 0, 1000, 0, 0, 0, 8000]],
            ),
            (
                "brake_from_300",
                4,
                "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,"
                "db_a_deg,db_w_deg_s,batt_mv",
                [[3000, 0, 0, 600, 0, 0, 0, 8000]],
            ),
        ]
        parsed = parse_session(_session(runs))
        poses = {
            "3": [{"stop": 0, "x_m": 0.98, "y_m": 0.0, "heading_deg": 0.0}],
            "4": [{"stop": 0, "x_m": 0.57, "y_m": 0.0, "heading_deg": 0.0}],
        }

        result = fit_session(parsed, poses=poses, wheel_radius_m=0.028)

        straight = result["actuator"]["straight"]
        turn = result["actuator"]["turn"]
        self.assertTrue(straight["calibrated"])
        self.assertTrue(turn["calibrated"])
        self.assertAlmostEqual(straight["gain"], 0.95, delta=0.03)
        self.assertAlmostEqual(straight["tau_s"], 0.08, delta=0.02)
        self.assertAlmostEqual(straight["delay_s"], 0.02, delta=0.02)
        self.assertLessEqual(straight["gain_min"], straight["gain"])
        self.assertGreaterEqual(straight["gain_max"], straight["gain"])
        self.assertGreaterEqual(straight["tau_min"], 0.0)
        self.assertLessEqual(straight["delay_min"], straight["delay_max"])
        self.assertAlmostEqual(turn["gain"], 0.95, delta=0.03)
        self.assertAlmostEqual(turn["tau_s"], 0.08, delta=0.02)

        self.assertTrue(result["odometry"]["calibrated"])
        self.assertAlmostEqual(result["odometry"]["distance_scale"], 0.98, delta=0.01)

        self.assertTrue(result["slip"]["calibrated"])
        expected_slip = 1.0 - 0.57 / (0.6 * 0.98)
        self.assertAlmostEqual(
            result["slip"]["brake_ratio_median"], expected_slip, delta=0.01
        )

    def test_parser_ignores_status_lines(self):
        text = _session([
            (
                "multi_stop",
                5,
                "t_ms,target_mm,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv",
                [],
            ),
        ]).replace(
            "TEST_END,0",
            "1000,500,500,0,0,0,8000\n"
            "STOP_AT 500 (500, 0, 0, 0)\n"
            "2000,1000,1000,0,0,0,8000\n"
            "TEST_END,2",
        )
        parsed = parse_session(text)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["data"].shape, (2, 7))
        self.assertEqual(parsed[0]["meta"]["run_id"], "5")

    def test_chirp_validation_is_reported(self):
        times = np.arange(0.0, 5.0, 0.01)
        command = 0.25 * np.sin(2.0 * math.pi * 0.5 * times)
        speed = _first_order(times, command, 0.95, 0.08, 0.02)
        angle_deg = np.cumsum(speed) * 0.01 / 0.028 * 180.0 / math.pi
        rows = [
            [t * 1000.0, cmd * 1000.0, 0.0, angle_deg[i], angle_deg[i]]
            for i, (t, cmd) in enumerate(zip(times, command))
        ]
        parsed = parse_session(_session([
            _straight_run("straight_steps_fwd", 1),
            (
                "chirp_straight",
                2,
                "t_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg",
                rows,
            ),
        ]))
        result = fit_session(parsed, wheel_radius_m=0.028)
        self.assertIsNotNone(result["validation"]["chirp_straight_nrmse"])
        self.assertLess(result["validation"]["chirp_straight_nrmse"], 0.10)


class TestDrivebaseProfile(unittest.TestCase):
    def _write_profile(self, tmp, payload):
        path = os.path.join(tmp, "drivebase.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_reads_minimal_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(tmp, {
                "name": "test drivebase",
                "wheel_radius_m": 0.028,
                "track_width_m": 0.0965,
            })
            profile = read_drivebase_profile(path)
        self.assertEqual(profile["name"], "test drivebase")
        self.assertAlmostEqual(profile["wheel_radius_m"], 0.028)
        self.assertAlmostEqual(profile["track_width_m"], 0.0965)

    def test_reads_pybricks_style_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(tmp, {
                "wheel_diameter_mm": 56.0,
                "axle_track_mm": 96.5,
            })
            profile = read_drivebase_profile(path)
        self.assertAlmostEqual(profile["wheel_radius_m"], 0.028)
        self.assertAlmostEqual(profile["track_width_m"], 0.0965)

    def test_reads_full_robot_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(tmp, {
                "robot": {
                    "wheel_radius": 0.03,
                    "track_width": 0.11,
                },
            })
            profile = read_drivebase_profile(path)
        self.assertAlmostEqual(profile["wheel_radius_m"], 0.03)
        self.assertAlmostEqual(profile["track_width_m"], 0.11)

    def test_reads_legacy_choreo_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(tmp, {
                "config": {
                    "wheel_radius": {"val": 0.028},
                    "differentialTrackWidth": {"val": 0.0965},
                },
            })
            profile = read_drivebase_profile(path)
        self.assertAlmostEqual(profile["wheel_radius_m"], 0.028)
        self.assertAlmostEqual(profile["track_width_m"], 0.0965)

    def test_rejects_profile_without_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_profile(tmp, {"name": "empty"})
            with self.assertRaises(ValueError):
                read_drivebase_profile(path)

    def test_cli_writes_drivebase_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            profile_path = self._write_profile(tmp, {
                "name": "test drivebase",
                "wheel_radius_m": 0.028,
                "track_width_m": 0.0965,
            })
            out = os.path.join(tmp, "sysid_fit.json")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write(_session([_straight_run("straight_steps_fwd", 1)]))
            with contextlib.redirect_stdout(io.StringIO()):
                code = sysid_main([session, "--profile", profile_path,
                                   "--out", out])
            with open(out, "r", encoding="utf-8") as handle:
                result = json.load(handle)
        self.assertEqual(code, 0)
        self.assertEqual(result["drivebase"]["name"], "test drivebase")
        self.assertAlmostEqual(result["drivebase"]["wheel_radius_m"], 0.028)
        self.assertAlmostEqual(result["drivebase"]["track_width_m"], 0.0965)
        self.assertEqual(result["drivebase"]["source"], profile_path)


class TestHubSessionPoses(unittest.TestCase):
    """KI-020: the hub session has to become a measured-pose CSV.

    hub_runner.py prints the pose of a run as "POSE,x_m,y_m,heading_deg" and
    every exported event as "EVENT,name". sysid_fit.py and sysid_compare.py
    read measured poses as CSV rows with run_id, stop, x_m, y_m, heading_deg,
    so the conversion has to exist somewhere or the robot run ends in a
    hand-edited file.
    """

    SESSION = [
        "Hub runner: following the pushed trajectory...",
        "EVENT,lower_arm",
        "POSE,0.4123,0.1000,-12.50",
        "EVENT,release",
        "POSE,1.2000,0.0500,3.25",
        "Hub runner: done.",
    ]

    def test_pose_lines_become_rows(self):
        rows, events = poses_from_hub_session(self.SESSION, run_id="run1")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_id"], "run1")
        self.assertEqual(rows[0]["stop"], 0)
        self.assertAlmostEqual(rows[0]["x_m"], 0.4123)
        self.assertAlmostEqual(rows[0]["y_m"], 0.1000)
        self.assertAlmostEqual(rows[0]["heading_deg"], -12.50)
        self.assertEqual(rows[1]["stop"], 1)
        self.assertEqual(events, ["lower_arm", "release"])

    def test_other_lines_are_ignored(self):
        rows, events = poses_from_hub_session([
            "Battery Voltage: 8100 mV",
            "POSE,not,a,number",
            "POSE,0.1,0.2",
            "Trajectory complete.",
        ])
        self.assertEqual(rows, [])
        self.assertEqual(events, [])

    def test_written_csv_reads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "poses.csv")
            rows, _ = poses_from_hub_session(self.SESSION, run_id="run1")
            write_poses_csv(rows, path)
            poses = load_poses(path)

        self.assertIn("run1", poses)
        entries = poses["run1"]
        self.assertEqual(len(entries), 2)
        self.assertAlmostEqual(entries[0]["x_m"], 0.4123)
        self.assertAlmostEqual(entries[1]["heading_deg"], 3.25)

    def test_cli_converts_a_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.log")
            poses_path = os.path.join(tmp, "poses.csv")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self.SESSION) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                code = sysid_main([
                    "--from-hub-session", session,
                    "--out-poses", poses_path,
                    "--run-id", "run1",
                ])
            poses = load_poses(poses_path)

        self.assertEqual(code, 0)
        self.assertEqual(len(poses["run1"]), 2)

    def test_cli_reports_a_session_without_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.log")
            poses_path = os.path.join(tmp, "poses.csv")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write("Hub runner: done.\n")
            with contextlib.redirect_stdout(io.StringIO()):
                code = sysid_main([
                    "--from-hub-session", session, "--out-poses", poses_path,
                ])
            self.assertFalse(os.path.exists(poses_path))
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
