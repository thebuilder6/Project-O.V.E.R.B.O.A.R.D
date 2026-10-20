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


_RATE_HEADER = (
    "t_ms,loop_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg,"
    "gyro_z_rad_s,ax_m_s2,ay_m_s2,batt_mv,batt_ma,l_load_mNm,r_load_mNm"
)


def _top_speed_run(name, run_id, speed_mm_s=600.0, command_mm_s=1000.0,
                   wheel_radius_m=0.028):
    """A free-running run: the wheels spin at a constant speed."""
    times = np.arange(0.0, 3.0, 0.01)
    omega_deg_s = speed_mm_s / 1000.0 / wheel_radius_m * 180.0 / math.pi
    angle_deg = np.cumsum(np.full_like(times, omega_deg_s) * 0.01)
    rows = [
        [t * 1000.0, 10.0, command_mm_s, 0.0, angle_deg[i], angle_deg[i],
         0.0, 0.0, 0.0, 8000.0, 0.0, 0.0, 0.0]
        for i, t in enumerate(times)
    ]
    return name, run_id, _RATE_HEADER, rows


class TestNoLoadSpeed(unittest.TestCase):
    """The motor's free-running speed, measured with the wheels lifted."""

    def test_the_free_running_speed_is_fitted(self):
        parsed = parse_session(_session([_top_speed_run("top_speed", 9)]))
        result = fit_session(parsed, wheel_radius_m=0.028)

        drivetrain = result["drivetrain"]
        self.assertTrue(drivetrain["calibrated"])
        # 600 mm/s on a 28 mm wheel is 21.43 rad/s.
        self.assertAlmostEqual(drivetrain["no_load_speed_rad_s"], 21.43,
                               delta=0.2)

    def test_a_run_that_reached_the_command_is_not_used(self):
        """The command was the limit then, so the number is not the motor's."""
        parsed = parse_session(_session([
            _top_speed_run("top_speed", 9, speed_mm_s=1000.0,
                           command_mm_s=1000.0)]))
        result = fit_session(parsed, wheel_radius_m=0.028)

        drivetrain = result["drivetrain"]
        self.assertFalse(drivetrain["calibrated"])
        self.assertTrue(drivetrain.get("rejected"))

    def test_a_session_without_the_test_has_no_number(self):
        parsed = parse_session(_session([_straight_run("straight_steps_fwd", 1)]))
        result = fit_session(parsed, wheel_radius_m=0.028)
        self.assertFalse(result["drivetrain"]["calibrated"])


def _reject_constant(name):
    """A bare NaN or Infinity is not JSON, so the file must not carry one."""
    raise ValueError(f"{name} is not valid JSON")


class TestBadRows(unittest.TestCase):
    """A non-finite cell is a bad cell, not data.

    One NaN in the command column used to turn the whole fit into NaN, and
    the fit file was written with a bare NaN in it, which is not JSON and
    fails later, far from the cause.
    """

    @staticmethod
    def _session_with_nan():
        name, run_id, header, rows = _straight_run("straight_steps_fwd", 1)
        rows[50][1] = float("nan")
        return _session([(name, run_id, header, rows)])

    def test_the_nan_row_is_dropped_and_counted(self):
        parsed = parse_session(self._session_with_nan())
        result = fit_session(parsed, wheel_radius_m=0.028)

        straight = result["actuator"]["straight"]
        self.assertTrue(straight["calibrated"])
        self.assertTrue(np.isfinite(straight["gain"]))
        self.assertAlmostEqual(straight["gain"], 0.95, delta=0.05)
        self.assertEqual(result["bad_rows"], 1)

    def test_the_written_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            out = os.path.join(tmp, "fit.json")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write(self._session_with_nan())
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = sysid_main([session, "--wheel-radius", "0.028",
                                   "--out", out])
            with open(out, "r", encoding="utf-8") as handle:
                text = handle.read()

        self.assertEqual(code, 0)
        # A strict reader refuses a bare NaN, so this is the real test.
        result = json.loads(text, parse_constant=_reject_constant)
        self.assertTrue(
            np.isfinite(result["actuator"]["straight"]["gain"]))

    def test_the_summary_says_how_many_rows_were_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            out = os.path.join(tmp, "fit.json")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write(self._session_with_nan())
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = sysid_main([session, "--wheel-radius", "0.028",
                                   "--out", out])

        self.assertEqual(code, 0)
        self.assertIn("1 row", buffer.getvalue())


class TestDistanceScale(unittest.TestCase):
    """The scale the validator plans against has to be one of the estimates.

    The tape and the wall are two estimates of the same number. Pooling them
    into one median gave a number that was neither, so the fit could not be
    reproduced from the numbers it reported.
    """

    @staticmethod
    def _homing_run(run_id, odometry_mm=1000.0):
        """A run that drove into a wall: the wall is the reference."""
        rows = [[0, 60.0, 0, 0, 60.0, 0, 0, 8000, 0, 0],
                [1000, 60.0, 0, odometry_mm, 0.0, 0, 0, 8000, 0, 0],
                [16000, 0, 0, odometry_mm, 0.0, 0.3, 0, 8000, 130, 130]]
        return "\n".join([
            "TEST_START wall_homing",
            f"META,run_id,{run_id}",
            "META,test,wall_homing",
            "META,repeat,0",
            "META,poses,0",
            "DATA",
            "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,db_a_deg,"
            "db_w_deg_s,batt_mv,l_load_mNm,r_load_mNm",
        ] + [",".join(str(v) for v in row) for row in rows] + ["TEST_END,3"])

    @staticmethod
    def _odometry_run(run_id):
        """A run with a tape-measured pose: the tape says 990 mm."""
        return "\n".join([
            "TEST_START odometry_straight",
            f"META,run_id,{run_id}",
            "META,test,odometry_straight",
            "META,repeat,0",
            "META,poses,1",
            "DATA",
            "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,db_a_deg,"
            "db_w_deg_s,batt_mv",
            "0,100,0,0,0,0,0,8000",
            "10000,0,0,1000,0,0,0,8000",
            "TEST_END,2",
        ])

    def _odometry(self):
        session = self._homing_run(1) + "\n" + self._odometry_run(2) + "\n"
        poses = {"2": [{"stop": 0, "x_m": 0.990, "y_m": 0.0,
                        "heading_deg": 0.0}]}
        parsed = parse_session(session)
        return fit_session(parsed, poses=poses, wheel_radius_m=0.028,
                           wall_distance_m=1.05)["odometry"]

    def test_the_scale_is_the_tape_estimate(self):
        odometry = self._odometry()

        self.assertAlmostEqual(odometry["distance_scale_tape"], 0.990,
                               places=6)
        self.assertAlmostEqual(odometry["distance_scale_homing"], 1.05,
                               places=6)
        # The number the validator plans against is the tape's number, not
        # the average of the two.
        self.assertAlmostEqual(odometry["distance_scale"],
                               odometry["distance_scale_tape"], places=9)
        # The cross-check is still reported.
        self.assertAlmostEqual(odometry["distance_scale_disagreement"], 0.06,
                               places=6)

    def test_the_wall_is_used_when_there_is_no_tape(self):
        session = self._homing_run(1) + "\n"
        odometry = fit_session(parse_session(session),
                               wheel_radius_m=0.028,
                               wall_distance_m=1.05)["odometry"]

        self.assertAlmostEqual(odometry["distance_scale"], 1.05, places=6)
        self.assertNotIn("distance_scale_tape", odometry)


class TestFitWithoutPoses(unittest.TestCase):
    """The actuator needs no measurement, so it is fitted without poses."""

    def test_the_actuator_is_fitted_without_a_measurement(self):
        parsed = parse_session(_session([_straight_run("straight_steps_fwd", 1)]))
        result = fit_session(parsed, poses={}, wheel_radius_m=0.028)

        straight = result["actuator"]["straight"]
        self.assertTrue(straight["calibrated"])
        self.assertAlmostEqual(straight["gain"], 0.95, delta=0.03)
        # The odometry needs a measurement, so it stays at its default.
        self.assertFalse(result["odometry"]["calibrated"])
        self.assertFalse(result["slip"]["calibrated"])
        self.assertAlmostEqual(result["odometry"]["distance_scale"], 1.0,
                               places=9)


class TestIntervalLowerBound(unittest.TestCase):
    """The interval over two repeats can reach below zero.

    A gain band that is not positive is refused by the loader, so the fit
    file has to be usable: the raw interval is kept, and the reported one
    is clamped.
    """

    def _two_repeat_fit(self):
        lines = []
        for repeat in range(2):
            _name, _run_id, header, rows = _straight_run(
                "straight_steps_fwd", repeat + 1, gain=0.90 + 0.20 * repeat)
            lines += [
                "TEST_START straight_steps_fwd",
                f"META,run_id,{repeat + 1}",
                "META,test,straight_steps_fwd",
                f"META,repeat,{repeat}",
                "META,poses,0",
                "DATA",
                header,
            ]
            lines += [",".join(str(v) for v in row) for row in rows]
            lines.append(f"TEST_END,{len(rows)}")
        return parse_session("\n".join(lines) + "\n")

    def test_the_reported_interval_has_a_positive_lower_bound(self):
        result = fit_session(self._two_repeat_fit(), wheel_radius_m=0.028)
        straight = result["actuator"]["straight"]

        self.assertIsNotNone(straight.get("gain_ci95"))
        self.assertGreater(straight["gain_ci95"][0], 0.0)
        # The raw t-interval is kept, so the operator can see what it was.
        self.assertIsNotNone(straight.get("gain_ci95_raw"))
        self.assertLess(straight["gain_ci95_raw"][0], 0.0)

    def test_the_fit_file_loads(self):
        from digital_twin import load_sysid_fit

        result = fit_session(self._two_repeat_fit(), wheel_radius_m=0.028)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fit.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
            twin = load_sysid_fit(path)

        self.assertTrue(twin.enabled)
        self.assertGreater(twin.actuator_gain_range()[0], 0.0)
