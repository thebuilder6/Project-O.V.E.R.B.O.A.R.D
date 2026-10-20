"""
Tests for the automated SysId collection over Bluetooth.

The collector is firmware code, so it is stubbed here the way
``tests/test_firmware_events.py`` does it: the metadata block it prints is
the real code, only the hardware is fake.

What matters is the contract between the hub and the host: the hub prints
``META,key,value`` and ``POSE,x_m,y_m,heading_deg`` lines, and the host
turns them into the poses.csv the fitter reads.
"""

import io
import json
import math
import os
import struct
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from tests.firmware_stubs import install_firmware_stubs, GcStub

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYBRICKS_CODE = os.path.join(REPO_ROOT, "pybricks_code")

if PYBRICKS_CODE not in sys.path:
    sys.path.insert(0, PYBRICKS_CODE)


def _fake_robot_config():
    """A robot_config with just enough hardware for the collector."""
    module = types.ModuleType("robot_config")

    class _Battery:
        def voltage(self):
            return 8000

    class _Hub:
        battery = _Battery()

    class _Motor:
        pass

    class _DriveBase:
        def settings(self):
            return (0, 0, 0, 0)

        def reset(self, distance=0, angle=0):
            pass

        def drive(self, speed, turn_rate):
            pass

        def stop(self):
            pass

        def distance(self):
            return 0

        def angle(self):
            return 0.0

        def state(self):
            return (0, 0, 0, 0)

    module.hub = _Hub()
    module.motorLeft = _Motor()
    module.motorRight = _Motor()
    module.drivebase = _DriveBase()
    module.WHEEL_DIAMETER_MM = 56
    module.AXLE_TRACK_MM = 96.5
    return module


def _load_collector():
    """Import the collector with stubbed hardware."""
    install_firmware_stubs()
    sys.modules.setdefault("robot_config", _fake_robot_config())
    import sysid_digital_twin

    # MicroPython's gc has mem_free(); CPython's does not.
    sysid_digital_twin.gc = GcStub
    # The run id counter is module state; reset it so each test starts at 1.
    sysid_digital_twin._RUN_ID = 0
    return sysid_digital_twin


# The session format the collector prints.
_RATE_HEADER = (
    "t_ms,loop_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg,"
    "gyro_z_rad_s,ax_m_s2,ay_m_s2,batt_mv,batt_ma,l_load_mNm,r_load_mNm"
)
_ODOM_HEADER = "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv"


def _session():
    """A session in the collector's format, with measured poses.

    The poses are 0.998 of the odometry, so both the straight run and the
    multi-stop run give the same distance scale.
    """
    return "\n".join([
        "TEST_START straight_steps_fwd",
        "META,run_id,1",
        "META,test,straight_steps_fwd",
        "DATA",
        _RATE_HEADER,
        "0.00,10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8000,0,0.0,0.0",
        "100.00,10.0,100.0,0.0,10.0,9.0,0.0,0.0,0.0,8000,0,0.0,0.0",
        "200.00,10.0,100.0,0.0,20.0,18.0,0.0,0.0,0.0,8000,0,0.0,0.0",
        "TEST_END,3",
        "TEST_START odometry_straight",
        "META,run_id,2",
        "META,test,odometry_straight",
        "DATA",
        "POSE,0.9980,0.0020,0.00",
        _ODOM_HEADER,
        "0.00,100.0,0.0,0.0,100.0,0.0,0.0,8000",
        "10000.00,0.0,0.0,1000.0,0.0,0.0,0.0,8000",
        "TEST_END,2",
        "TEST_START multi_stop",
        "META,run_id,3",
        "META,test,multi_stop",
        "DATA",
        "t_ms,target_mm,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv",
        "POSE,0.4990,0.0000,0.00",
        "POSE,0.9980,0.0000,0.00",
        "POSE,1.4970,0.0000,0.00",
        "5000.0,500,500,0.0,0.0,0.0,8000",
        "10000.0,1000,1000,0.0,0.0,0.0,8000",
        "15000.0,1500,1500,0.0,0.0,0.0,8000",
        "TEST_END,3",
    ]) + "\n"


def _turn_session():
    """A session with a turn test, so the heading scale has a source.

    The robot turns clockwise at 50 deg/s for 1.8 s, so the odometry angle
    reads +90 (clockwise positive) and the pose reads -90 (counter-clockwise
    positive, the trajectory convention). The scale between them is +1.
    """
    return "\n".join([
        "TEST_START odometry_turn_cw",
        "META,run_id,7",
        "META,test,odometry_turn_cw",
        "DATA",
        "POSE,0.0000,-0.1257,-90.00",
        _ODOM_HEADER,
        "0.00,0.0,50.0,0.0,0.0,0.0,50.0,8000",
        "1800.00,0.0,0.0,0.0,0.0,90.0,0.0,8000",
        "TEST_END,2",
    ]) + "\n"


class TestCollectorMetadata(unittest.TestCase):
    """The collector's META lines have to match what the host parses."""

    def test_meta_lines_use_the_key_value_form(self):
        collector = _load_collector()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            collector._print_metadata("odometry_straight", "notes,with,commas")
        text = buffer.getvalue()

        self.assertIn("TEST_START odometry_straight", text)
        self.assertIn("META,run_id,", text)
        self.assertIn("META,test,odometry_straight", text)
        self.assertIn("META,notes,notes;with;commas", text)

    def test_the_session_parses_with_a_run_id(self):
        from sysid_fit import parse_session

        collector = _load_collector()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            collector._print_metadata("odometry_straight", "")
            print("DATA")
            print("t_ms,speed")
            print("0,0")
            print("TEST_END,1")

        runs = parse_session(buffer.getvalue())
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["meta"]["run_id"], "1")
        self.assertEqual(runs[0]["name"], "odometry_straight")


class TestSessionPoses(unittest.TestCase):
    """A pose belongs to the test that was open when it was printed."""

    def test_poses_take_the_run_id_of_their_test(self):
        from sysid_fit import poses_from_hub_session

        rows, events = poses_from_hub_session(_session().splitlines())
        by_run = {}
        for row in rows:
            by_run.setdefault(row["run_id"], []).append(row)

        self.assertEqual(sorted(by_run), ["2", "3"])
        self.assertEqual([row["stop"] for row in by_run["3"]], [0, 1, 2])
        self.assertAlmostEqual(by_run["2"][0]["x_m"], 0.998)
        self.assertAlmostEqual(by_run["3"][2]["y_m"], 0.0)

    def test_a_session_without_meta_keeps_one_run(self):
        from sysid_fit import poses_from_hub_session

        rows, _ = poses_from_hub_session(["POSE,0.1000,0.0000,0.00"])
        self.assertEqual([row["run_id"] for row in rows], ["hub"])
        self.assertEqual([row["stop"] for row in rows], [0])

    def test_the_written_csv_is_read_back(self):
        from sysid_fit import load_poses, poses_from_hub_session, write_poses_csv

        rows, _ = poses_from_hub_session(_session().splitlines())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "poses.csv")
            write_poses_csv(rows, path)
            poses = load_poses(path)

        self.assertEqual(sorted(poses), ["2", "3"])
        self.assertEqual(len(poses["3"]), 3)
        self.assertAlmostEqual(poses["3"][1]["x_m"], 0.998)


class TestFittedScales(unittest.TestCase):
    """The scale between the measured pose and the odometry must be +1.

    A negative heading scale is rejected by the loader, so this is the
    difference between a fit that can be used and one that cannot.
    """

    def _fit(self, session):
        from sysid_fit import (fit_session, load_poses, load_runs,
                               poses_from_hub_session, write_poses_csv)
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "session.csv")
            poses = os.path.join(tmp, "poses.csv")
            with open(log, "w", encoding="utf-8") as handle:
                handle.write(session)
            rows, _ = poses_from_hub_session(session.splitlines())
            write_poses_csv(rows, poses)
            _, runs = load_runs([log])
            return fit_session(runs, poses=load_poses(poses),
                               wheel_radius_m=0.028)

    def test_a_clockwise_turn_gives_a_positive_heading_scale(self):
        result = self._fit(_turn_session())
        self.assertAlmostEqual(result["odometry"]["heading_scale"], 1.0,
                               places=6)
        self.assertGreater(result["odometry"]["heading_scale"], 0.0)

    def test_the_fit_with_a_turn_loads(self):
        from digital_twin import load_sysid_fit

        result = self._fit(_turn_session())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fit.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
            fit = load_sysid_fit(path)
        self.assertAlmostEqual(fit.odometry_heading_scale(), 1.0, places=6)


class TestMeasurementSheet(unittest.TestCase):
    """The operator's sheet: what they fill in, and what it turns into."""

    @staticmethod
    def _session(repeats=1):
        lines = [
            "TEST_START static",
            "META,run_id,1",
            "META,test,static",
            "META,repeat,0",
            "META,poses,0",
            "DATA",
            "t_ms,gyro_z_deg_s",
            "0,0.1",
            "TEST_END,1",
        ]
        run_id = 2
        for repeat in range(repeats):
            for name, poses, rows in (
                ("odometry_straight", 1, 2),
                ("odometry_turn_cw", 1, 2),
                ("multi_stop", 3, 3),
            ):
                lines += [
                    f"TEST_START {name}",
                    f"META,run_id,{run_id}",
                    f"META,test,{name}",
                    f"META,repeat,{repeat}",
                    f"META,poses,{poses}",
                    "DATA",
                    "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,"
                    "db_a_deg,db_w_deg_s,batt_mv",
                ]
                for i in range(rows):
                    lines.append(f"{i * 1000},100,0,{i * 1000},0,0,0,8000")
                lines.append(f"TEST_END,{rows}")
                run_id += 1
        return "\n".join(lines) + "\n"

    def _runs(self, repeats=1):
        from sysid_fit import load_runs
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._session(repeats))
            return load_runs([path])[1]

    def test_the_sheet_lists_the_tests_that_need_a_measurement(self):
        from sysid_fit import read_sheet, write_sheet

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.csv")
            write_sheet(self._runs(repeats=3), path)
            rows = read_sheet(path)
            # The blank rows are not measurements yet, so read_sheet skips them.
            self.assertEqual(rows, [])
            with open(path, encoding="utf-8") as handle:
                text = handle.read()

        # One row per measurement, per repeat.
        self.assertIn("odometry_straight,0,0", text)
        self.assertIn("odometry_straight,2,0", text)
        self.assertIn("multi_stop,2,2", text)
        # The test that needs no measurement is not in the sheet.
        self.assertNotIn("static", text)

    def test_the_sheet_becomes_measured_poses(self):
        from sysid_fit import sheet_to_poses

        rows = [
            {"test": "odometry_straight", "repeat": 0, "stop": 0,
             "x_mm": 998.0, "y_mm": 2.0, "turn_right_deg": 0.0},
            {"test": "odometry_turn_cw", "repeat": 0, "stop": 0,
             "x_mm": 0.0, "y_mm": 0.0, "turn_right_deg": 90.0},
            {"test": "multi_stop", "repeat": 0, "stop": 0,
             "x_mm": 499.0, "y_mm": 0.0, "turn_right_deg": 0.0},
        ]
        poses, problems = sheet_to_poses(rows, self._runs())

        self.assertEqual([p["run_id"] for p in poses], ["2", "3", "4"])
        self.assertAlmostEqual(poses[0]["x_m"], 0.998)
        self.assertAlmostEqual(poses[0]["y_m"], 0.002)
        # A turn to the right is a negative heading in the trajectory frame.
        self.assertAlmostEqual(poses[1]["heading_deg"], -90.0)
        # The two multi_stop stops that were not measured are reported.
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("multi_stop" in p for p in problems))

    def test_a_repeat_is_matched_to_its_own_run(self):
        """The same test in two repeats is two different runs."""
        from sysid_fit import sheet_to_poses

        rows = [
            {"test": "odometry_straight", "repeat": 0, "stop": 0,
             "x_mm": 998.0, "y_mm": 0.0, "turn_right_deg": 0.0},
            {"test": "odometry_straight", "repeat": 1, "stop": 0,
             "x_mm": 997.0, "y_mm": 0.0, "turn_right_deg": 0.0},
        ]
        poses, _problems = sheet_to_poses(rows, self._runs(repeats=2))
        # Repeat 0 of odometry_straight is run 2, repeat 1 is run 5.
        self.assertEqual([p["run_id"] for p in poses], ["2", "5"])

    def test_a_row_for_a_test_that_did_not_run_is_reported(self):
        from sysid_fit import sheet_to_poses

        rows = [{"test": "no_such_test", "repeat": 0, "stop": 0,
                 "x_mm": 1.0, "y_mm": 0.0, "turn_right_deg": 0.0}]
        poses, problems = sheet_to_poses(rows, self._runs())
        self.assertEqual(poses, [])
        self.assertTrue(any("no_such_test" in p for p in problems))

    def test_the_fit_uses_the_measured_poses(self):
        """A measured pose has to beat the hub's own odometry."""
        from sysid_fit import (fit_session, load_poses, parse_session,
                               sheet_to_poses, write_poses_csv)

        rows = [{"test": "odometry_straight", "repeat": 0, "stop": 0,
                 "x_mm": 998.0, "y_mm": 0.0, "turn_right_deg": 0.0}]
        poses, _problems = sheet_to_poses(rows, self._runs())

        with tempfile.TemporaryDirectory() as tmp:
            poses_path = os.path.join(tmp, "poses.csv")
            write_poses_csv(poses, poses_path)
            loaded = load_poses(poses_path)

        runs = parse_session(self._session())
        result = fit_session(runs, poses=loaded, wheel_radius_m=0.028)
        # The hub's odometry says 1000 mm, the operator measured 998 mm.
        self.assertAlmostEqual(result["odometry"]["distance_scale"], 0.998,
                               places=3)


class TestRepeatStatistics(unittest.TestCase):
    """The numbers have to carry an interval, and say what it covers."""

    def test_one_run_has_no_interval(self):
        from sysid_fit import repeat_statistics

        stats = repeat_statistics([0.98])
        self.assertEqual(stats["n"], 1)
        self.assertIsNone(stats["ci95"])

    def test_three_runs_give_a_t_interval(self):
        from sysid_fit import repeat_statistics

        stats = repeat_statistics([0.98, 1.00, 1.02])
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 1.0)
        self.assertAlmostEqual(stats["std"], 0.02, places=9)
        # t(0.975, 2) = 4.303, so the half width is 4.303 * 0.02 / sqrt(3).
        half = 4.303 * 0.02 / (3 ** 0.5)
        self.assertAlmostEqual(stats["ci95"][0], 1.0 - half, places=9)
        self.assertAlmostEqual(stats["ci95"][1], 1.0 + half, places=9)

    def test_the_fit_reports_the_interval_over_the_repeats(self):
        from sysid_fit import fit_session, parse_session

        lines = []
        for repeat in range(3):
            rows = []
            for i in range(120):
                t = i * 0.01
                gain = 0.98 + 0.01 * repeat
                speed = 0.4 * gain if 0.2 <= t < 1.2 else 0.0
                angle = speed / 0.028 * 180.0 / 3.141592653589793 * t
                rows.append([t * 1000.0, 400.0 * gain, 0.0, angle, angle])
            lines += [
                f"TEST_START straight_steps_fwd_{repeat}",
                f"META,run_id,{repeat + 1}",
                "META,test,straight_steps_fwd",
                f"META,repeat,{repeat}",
                "META,poses,0",
                "DATA",
                "t_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg",
            ]
            lines += [",".join(str(v) for v in row) for row in rows]
            lines.append(f"TEST_END,{len(rows)}")

        result = fit_session(parse_session("\n".join(lines) + "\n"),
                             wheel_radius_m=0.028)
        straight = result["actuator"]["straight"]
        self.assertEqual(straight["n_runs"], 3)
        self.assertEqual(straight["gain_between_runs"]["n"], 3)
        self.assertIsNotNone(straight.get("gain_ci95"))

    def test_the_validator_band_is_the_interval(self):
        from digital_twin import load_sysid_fit

        fit = {
            "actuator": {
                "straight": {"calibrated": True, "gain": 0.98,
                             "gain_min": 0.97, "gain_max": 0.99,
                             "gain_ci95": [0.95, 1.01], "tau_s": 0.08,
                             "delay_s": 0.02, "n_steps": 42, "n_runs": 3},
                "turn": {"calibrated": False},
            },
            "sensors": {"calibrated": False},
            "odometry": {"calibrated": False},
            "slip": {"calibrated": False},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fit.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            loaded = load_sysid_fit(path)
            self.assertEqual(loaded.actuator_gain_range(), (0.95, 1.01))

            # A session with one run has no interval, so the observed range
            # is the band.
            del fit["actuator"]["straight"]["gain_ci95"]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            loaded = load_sysid_fit(path)
            self.assertEqual(loaded.actuator_gain_range(), (0.97, 0.99))


class TestDriftAndStallTorque(unittest.TestCase):
    """Two things the suite already measured but the fit ignored."""

    @staticmethod
    def _session():
        lines = []
        lines += [
            "TEST_START static",
            "META,run_id,1",
            "META,test,static",
            "META,repeat,0",
            "META,poses,0",
            "DATA",
            "t_ms,loop_ms,l_deg,r_deg,gyro_x_deg_s,gyro_y_deg_s,"
            "gyro_z_deg_s,ax_mm_s2,ay_mm_s2,az_mm_s2,batt_mv,batt_ma",
        ]
        for i in range(50):
            lines.append(f"{i * 10},10,0,0,0,0,0.05,0,0,9800,8000,0")
        lines.append("TEST_END,50")

        for repeat in range(3):
            lines += [
                "TEST_START odometry_straight",
                f"META,run_id,{2 + repeat}",
                "META,test,odometry_straight",
                f"META,repeat,{repeat}",
                "META,poses,1",
                "DATA",
                "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,db_a_deg,"
                "db_w_deg_s,batt_mv",
            ]
            for i in range(11):
                # 1 m in 10 s. The heading changed by -2.5 deg: 2 deg of real
                # left curve plus 0.5 deg of gyro bias over the run.
                last = i == 10
                lines.append(f"{i * 1000},100,0,{1000 if last else 0},0,"
                             f"{-2.5 if last else 0},0,8000")
            lines.append("TEST_END,11")

        for repeat in range(3):
            lines += [
                "TEST_START wall_stall",
                f"META,run_id,{5 + repeat}",
                "META,test,wall_stall",
                f"META,repeat,{repeat}",
                "META,poses,0",
                "DATA",
                "t_ms,loop_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg,"
                "gyro_z_rad_s,ax_m_s2,ay_m_s2,batt_mv,batt_ma,l_load_mNm,"
                "r_load_mNm",
            ]
            for i in range(50):
                # The impact overshoots, then the load settles at 130 mNm.
                load = 300.0 if i <= 2 else 130.0
                lines.append(f"{i * 10},10,30,0,0,0,0,0,0,8000,50,{load},{load}")
            lines.append("TEST_END,50")
        return "\n".join(lines) + "\n"

    def _fit(self, poses=None):
        from sysid_fit import fit_session, load_poses, parse_session
        if poses is None:
            poses = {}
        return fit_session(parse_session(self._session()), poses=poses,
                           wheel_radius_m=0.028)

    def test_the_drift_comes_from_the_gyro_with_the_bias_removed(self):
        result = self._fit()
        drift = result["drift"]
        self.assertTrue(drift["calibrated"])
        # 2 deg/m of real curve. Without the bias correction the gyro would
        # report 2.5 deg/m.
        self.assertAlmostEqual(drift["drift_deg_per_m"], 2.0, places=6)
        self.assertEqual(drift["n_sources"], 3)

    def test_the_tape_gives_an_independent_estimate_of_the_drift(self):
        poses = {"2": [{"stop": 0, "x_m": 0.998, "y_m": 0.01745,
                        "heading_deg": 2.0}]}
        result = self._fit(poses)
        drift = result["drift"]
        # A drift of k over a distance L leaves k * L^2 / 2 behind.
        self.assertAlmostEqual(drift["drift_deg_per_m_tape"], 2.007, places=2)
        self.assertLess(drift["drift_disagreement_deg_per_m"], 0.1)

    def test_the_stall_torque_ignores_the_impact(self):
        result = self._fit()
        stall = result["stall_torque"]
        self.assertTrue(stall["calibrated"])
        # The steady load is 130 mNm, which is 0.13 Nm. The impact spike
        # must not raise it.
        self.assertAlmostEqual(stall["t_max_nm"], 0.13, places=6)
        self.assertEqual(stall["n_sources"], 3)

    def test_without_the_tests_neither_is_calibrated(self):
        from sysid_fit import fit_session, parse_session
        session = ("TEST_START static\nMETA,run_id,1\nMETA,test,static\n"
                   "META,poses,0\nDATA\nt_ms,speed\n0,0\nTEST_END,1\n")
        result = fit_session(parse_session(session), wheel_radius_m=0.028)
        self.assertFalse(result["drift"]["calibrated"])
        self.assertFalse(result["stall_torque"]["calibrated"])


class TestCollectCommand(unittest.TestCase):

    def _run(self, tmp, extra=None):
        import sysid_collect

        session = os.path.join(tmp, "session.csv")
        poses = os.path.join(tmp, "poses.csv")
        fit = os.path.join(tmp, "sysid_fit.json")
        profile = os.path.join(tmp, "profile.json")
        with open(profile, "w", encoding="utf-8") as handle:
            json.dump({"name": "test", "wheel_radius_m": 0.028,
                       "track_width_m": 0.0965}, handle)

        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            with open(stream_path, "w", encoding="utf-8") as handle:
                handle.write(_session())
            return _session().splitlines()

        args = ["--conn", "ble", "--session", session, "--poses", poses,
                "--profile", profile, "--out", fit] + (extra or [])
        with patch.object(sysid_collect, "deploy", fake_deploy):
            status = sysid_collect.main(args)
        return status, session, poses, fit

    def test_it_writes_the_session_the_poses_and_the_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, session, poses, fit = self._run(tmp)
            self.assertEqual(status, 0)
            self.assertTrue(os.path.exists(session))
            self.assertTrue(os.path.exists(poses))
            self.assertTrue(os.path.exists(fit))
            with open(fit, "r", encoding="utf-8") as handle:
                result = json.load(handle)

        self.assertTrue(result["odometry"]["calibrated"])
        # The pose says the robot went 998 mm and the odometry says 1000 mm.
        self.assertAlmostEqual(result["odometry"]["distance_scale"], 0.998,
                               places=3)

    def test_a_missing_profile_is_reported(self):
        import sysid_collect

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                sysid_collect.main(["--session", os.path.join(tmp, "s.csv")])

    def test_a_dropped_link_still_fits_what_arrived(self):
        """One flaky link must not throw away a ten minute session."""
        import sysid_collect
        from hub_deploy import HubLinkError

        def dropping_deploy(script, connection="ble", name=None, wait=True,
                            print_output=True, stream_path=None):
            with open(stream_path, "w", encoding="utf-8") as handle:
                handle.write(_session())
            raise HubLinkError("hub disconnected")

        with tempfile.TemporaryDirectory() as tmp:
            profile = os.path.join(tmp, "profile.json")
            with open(profile, "w", encoding="utf-8") as handle:
                json.dump({"wheel_radius_m": 0.028,
                           "track_width_m": 0.0965}, handle)
            fit = os.path.join(tmp, "sysid_fit.json")
            args = ["--session", os.path.join(tmp, "session.csv"),
                    "--poses", os.path.join(tmp, "poses.csv"),
                    "--profile", profile, "--out", fit]
            with patch.object(sysid_collect, "deploy", dropping_deploy):
                status = sysid_collect.main(args)

            self.assertEqual(status, 0)
            self.assertTrue(os.path.exists(fit))
            with open(fit, "r", encoding="utf-8") as handle:
                result = json.load(handle)
        self.assertTrue(result["odometry"]["calibrated"])


if __name__ == "__main__":
    unittest.main()
