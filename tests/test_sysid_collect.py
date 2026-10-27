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
from overboard.sysid.fit import sheet_rows, write_sheet

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
        from overboard.sysid.fit import parse_session

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
        from overboard.sysid.fit import poses_from_hub_session

        rows, events = poses_from_hub_session(_session().splitlines())
        by_run = {}
        for row in rows:
            by_run.setdefault(row["run_id"], []).append(row)

        self.assertEqual(sorted(by_run), ["2", "3"])
        self.assertEqual([row["stop"] for row in by_run["3"]], [0, 1, 2])
        self.assertAlmostEqual(by_run["2"][0]["x_m"], 0.998)
        self.assertAlmostEqual(by_run["3"][2]["y_m"], 0.0)

    def test_a_session_without_meta_keeps_one_run(self):
        from overboard.sysid.fit import poses_from_hub_session

        rows, _ = poses_from_hub_session(["POSE,0.1000,0.0000,0.00"])
        self.assertEqual([row["run_id"] for row in rows], ["hub"])
        self.assertEqual([row["stop"] for row in rows], [0])

    def test_the_written_csv_is_read_back(self):
        from overboard.sysid.fit import load_poses, poses_from_hub_session, write_poses_csv

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
        from overboard.sysid.fit import fit_session, load_poses, load_runs, poses_from_hub_session, write_poses_csv
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
        from overboard.sysid.digital_twin import load_sysid_fit

        result = self._fit(_turn_session())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fit.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
            fit = load_sysid_fit(path)
        self.assertAlmostEqual(fit.odometry_heading_scale(), 1.0, places=6)


def _measurement_session(repeats=1):
    """A session with the tests that need a tape measurement.

    Each of the three measured tests runs once per repeat, and the multi-stop
    test needs a pose at each of its three stops.
    """
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


class TestMeasurementSheet(unittest.TestCase):
    """The operator's sheet: what they fill in, and what it turns into."""

    @staticmethod
    def _session(repeats=1):
        return _measurement_session(repeats)

    def _runs(self, repeats=1):
        from overboard.sysid.fit import load_runs
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._session(repeats))
            return load_runs([path])[1]

    def test_the_sheet_lists_the_tests_that_need_a_measurement(self):
        from overboard.sysid.fit import read_sheet, write_sheet

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
        from overboard.sysid.fit import sheet_to_poses

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
        from overboard.sysid.fit import sheet_to_poses

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
        from overboard.sysid.fit import sheet_to_poses

        rows = [{"test": "no_such_test", "repeat": 0, "stop": 0,
                 "x_mm": 1.0, "y_mm": 0.0, "turn_right_deg": 0.0}]
        poses, problems = sheet_to_poses(rows, self._runs())
        self.assertEqual(poses, [])
        self.assertTrue(any("no_such_test" in p for p in problems))

    def test_the_fit_uses_the_measured_poses(self):
        """A measured pose has to beat the hub's own odometry."""
        from overboard.sysid.fit import fit_session, load_poses, parse_session, sheet_to_poses, write_poses_csv

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
        from overboard.sysid.fit import repeat_statistics

        stats = repeat_statistics([0.98])
        self.assertEqual(stats["n"], 1)
        self.assertIsNone(stats["ci95"])

    def test_three_runs_give_a_t_interval(self):
        from overboard.sysid.fit import repeat_statistics

        stats = repeat_statistics([0.98, 1.00, 1.02])
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 1.0)
        self.assertAlmostEqual(stats["std"], 0.02, places=9)
        # t(0.975, 2) = 4.303, so the half width is 4.303 * 0.02 / sqrt(3).
        half = 4.303 * 0.02 / (3 ** 0.5)
        self.assertAlmostEqual(stats["ci95"][0], 1.0 - half, places=9)
        self.assertAlmostEqual(stats["ci95"][1], 1.0 + half, places=9)

    def test_the_fit_reports_the_interval_over_the_repeats(self):
        from overboard.sysid.fit import fit_session, parse_session

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
        from overboard.sysid.digital_twin import load_sysid_fit

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
        from overboard.sysid.fit import fit_session, load_poses, parse_session
        if poses is None:
            poses = {}
        return fit_session(parse_session(self._session()), poses=poses,
                           wheel_radius_m=0.028)

    def test_the_stall_torque_ignores_the_impact(self):
        result = self._fit()
        stall = result["stall_torque"]
        self.assertTrue(stall["calibrated"])
        # The steady load is 130 mNm, which is 0.13 Nm. The impact spike
        # must not raise it.
        self.assertAlmostEqual(stall["t_max_nm"], 0.13, places=6)
        self.assertEqual(stall["n_sources"], 3)

    def test_without_the_tests_neither_is_calibrated(self):
        from overboard.sysid.fit import fit_session, parse_session
        session = ("TEST_START static\nMETA,run_id,1\nMETA,test,static\n"
                   "META,poses,0\nDATA\nt_ms,speed\n0,0\nTEST_END,1\n")
        result = fit_session(parse_session(session), wheel_radius_m=0.028)
        self.assertFalse(result["drift"]["calibrated"])
        self.assertFalse(result["stall_torque"]["calibrated"])


class TestRefinedStepFit(unittest.TestCase):
    """The step fit is a two-stage search, so it is not quantised."""

    @staticmethod
    def _step(gain=0.98, tau=0.08, delay=0.02, dt=0.01, n=90, command=0.4):
        import math
        times = [i * dt for i in range(n)]
        measured = [
            gain * command * (1.0 - math.exp(-max(t - delay, 0.0) / tau))
            for t in times
        ]
        return times, measured, command

    def test_it_recovers_a_known_step(self):
        from overboard.sysid.fit import _fit_first_order

        times, measured, command = self._step()
        fit = _fit_first_order(times, measured, command)

        self.assertAlmostEqual(fit["gain"], 0.98, places=6)
        self.assertAlmostEqual(fit["tau_s"], 0.08, places=6)
        self.assertAlmostEqual(fit["delay_s"], 0.02, places=6)
        self.assertAlmostEqual(fit["rmse"], 0.0, places=9)

    def test_the_answer_is_not_on_the_ten_millisecond_grid(self):
        """The first pass can only land on a 10 ms boundary."""
        from overboard.sysid.fit import _fit_first_order

        times, measured, command = self._step(tau=0.0837, delay=0.0163)
        fit = _fit_first_order(times, measured, command)

        self.assertAlmostEqual(fit["tau_s"], 0.0837, places=3)
        self.assertAlmostEqual(fit["delay_s"], 0.0163, places=3)


class TestDriftRegression(unittest.TestCase):
    """The drift is a slope against the speed, not a subtraction."""

    @staticmethod
    def _session(drift=2.0, bias=0.05, gain=0.98, dt=0.01):
        import math
        lines = []

        def block(name, repeat, run_id, header, rows):
            lines.append(f"TEST_START {name}")
            lines.append(f"META,run_id,{run_id}")
            lines.append(f"META,test,{name}")
            lines.append(f"META,repeat,{repeat}")
            lines.append("META,poses,0")
            lines.append("DATA")
            lines.append(header)
            for row in rows:
                lines.append(",".join(str(v) for v in row))
            lines.append(f"TEST_END,{len(rows)}")

        rate = ("t_ms,loop_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg,gyro_z_rad_s,"
                "ax_m_s2,ay_m_s2,batt_mv,batt_ma,l_load_mNm,r_load_mNm")
        block("static", 0, 1,
              "t_ms,loop_ms,l_deg,r_deg,gyro_x_deg_s,gyro_y_deg_s,"
              "gyro_z_deg_s,ax_mm_s2,ay_mm_s2,az_mm_s2,batt_mv,batt_ma",
              [[i * 10, 10, 0, 0, 0, 0, bias, 0, 0, 9800, 8000, 0]
               for i in range(50)])

        run_id = 2
        for repeat in range(3):
            for direction, tag in ((1, "fwd"), (-1, "rev")):
                rows = []
                t = 0.0
                angle = 0.0
                for speed_mm_s in (50, 100, 200, 400, 600):
                    command = direction * speed_mm_s
                    for i in range(90):
                        t += dt
                        value = gain * (command / 1000.0)
                        angle += value * dt / 0.028 * 180.0 / math.pi
                        gyro = math.radians(drift * value + bias)
                        rows.append([t * 1000, 10, command, 0, angle, angle,
                                     gyro, 0, 0, 8000, 50, 0, 0])
                block(f"straight_steps_{tag}", repeat, run_id, rate, rows)
                run_id += 1
            # The odometry run, for the tape's independent estimate.
            block("odometry_straight", repeat, run_id,
                  "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,db_a_deg,"
                  "db_w_deg_s,batt_mv",
                  [[i * 1000, 100, 0, 0, 0, 0, 0, 8000] for i in range(11)])
            run_id += 1
        return "\n".join(lines) + "\n"

    @staticmethod
    def _step_run(name, run_id, repeat, command, steps=1, drift=2.0,
                  bias=0.05, gain=0.98, dt=0.01):
        """A step run: the wheel angles give the speed, the gyro the rate."""
        import math
        rows = []
        t = 0.0
        angle = 0.0
        for _ in range(steps):
            for _ in range(90):
                t += dt
                value = gain * command
                angle += value * dt / 0.028 * 180.0 / math.pi
                gyro = math.radians(drift * value + bias)
                rows.append([t * 1000, 10, command * 1000, 0, angle, angle,
                             gyro, 0, 0, 8000, 50, 0, 0])
        return "\n".join([
            f"TEST_START {name}",
            f"META,run_id,{run_id}",
            f"META,test,{name}",
            f"META,repeat,{repeat}",
            "META,poses,0",
            "DATA",
            "t_ms,loop_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg,gyro_z_rad_s,"
            "ax_m_s2,ay_m_s2,batt_mv,batt_ma,l_load_mNm,r_load_mNm",
        ] + [",".join(str(v) for v in row) for row in rows] + [f"TEST_END,{len(rows)}"])

    def _fit(self, poses=None, **kwargs):
        from overboard.sysid.fit import fit_session, load_poses, parse_session
        if poses is None:
            poses = {}
        return fit_session(parse_session(self._session(**kwargs)), poses=poses,
                           wheel_radius_m=0.028)

    def test_the_slope_is_the_drift(self):
        drift = self._fit()["drift"]
        self.assertTrue(drift["calibrated"])
        self.assertAlmostEqual(drift["drift_deg_per_m"], 2.0, places=4)

    def test_the_intercept_is_the_gyro_bias(self):
        """The bias stops contaminating the drift: it is the intercept."""
        drift = self._fit()["drift"]
        self.assertAlmostEqual(drift["gyro_bias_deg_s_moving"], 0.05, places=6)
        self.assertLess(drift["gyro_bias_disagreement_deg_s"], 1e-6)

    def test_the_forward_and_reverse_runs_are_in_one_fit(self):
        drift = self._fit()["drift"]
        for source in drift["drift_sources"]:
            low, high = source["speed_range_m_s"]
            self.assertLess(low, 0.0)
            self.assertGreater(high, 0.0)

    def test_the_tape_gives_an_independent_estimate(self):
        """A drift of k over a distance L leaves k * L^2 / 2 behind."""
        poses = {"4": [{"stop": 0, "x_m": 0.998, "y_m": 0.01745,
                        "heading_deg": 2.0}]}
        drift = self._fit(poses)["drift"]
        self.assertAlmostEqual(drift["drift_deg_per_m_tape"], 2.007, places=2)
        self.assertLess(drift["drift_disagreement_deg_per_m"], 0.1)

    def test_too_few_points_does_not_crash_the_fit(self):
        """A line through two points has no standard error to report."""
        from overboard.sysid.fit import fit_session, parse_session

        # One step in the forward run and one in the reverse run: two points.
        session = "\n".join([
            self._step_run("straight_steps_fwd", 1, 0, 0.4),
            self._step_run("straight_steps_rev", 2, 0, -0.4),
        ]) + "\n"
        result = fit_session(parse_session(session), wheel_radius_m=0.028)
        self.assertFalse(result["drift"]["calibrated"])


class TestWallHoming(unittest.TestCase):
    """The wall is a reference that does not move and is not read by hand."""

    @staticmethod
    def _homing_run(name, run_id, repeat, odometry_mm, final_speed,
                    command=60.0, reached=True):
        """A homing run written the way the collector writes it.

        The collector logs the drive, and then a settle row after the robot
        has stopped, with the command at zero. The settle row is the last
        one, so the check has to look at the last row that was still
        commanded.
        """
        rows = [[0, command, 0, 0, command, 0, 0, 8000, 0, 0],
                [1000, command, 0, odometry_mm,
                 0.0 if reached else command, 0, 0, 8000, 0, 0],
                [16000, 0, 0, odometry_mm, final_speed, 0.3, 0, 8000,
                 130, 130]]
        return "\n".join([
            f"TEST_START {name}",
            f"META,run_id,{run_id}",
            f"META,test,{name}",
            f"META,repeat,{repeat}",
            "META,poses,0",
            "DATA",
            "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,db_a_deg,"
            "db_w_deg_s,batt_mv,l_load_mNm,r_load_mNm",
        ] + [",".join(str(v) for v in row) for row in rows] + ["TEST_END,3"])

    def _fit(self, session, wall_distance_m):
        from overboard.sysid.fit import fit_session, parse_session
        return fit_session(parse_session(session), wheel_radius_m=0.028,
                           wall_distance_m=wall_distance_m)

    def test_the_wall_gives_the_distance_scale(self):
        session = "\n".join([
            self._homing_run("wall_homing", 1, 0, 1000.0, 0.0),
            self._homing_run("wall_homing", 2, 1, 1000.0, 0.0),
        ]) + "\n"
        odometry = self._fit(session, 0.998)["odometry"]

        self.assertTrue(odometry["calibrated"])
        self.assertAlmostEqual(odometry["distance_scale_homing"], 0.998,
                               places=6)
        self.assertEqual(odometry["n_homing_sources"], 2)

    def test_a_run_that_never_reached_the_wall_is_rejected(self):
        """The robot was still moving when the run ended: no wall."""
        session = self._homing_run("wall_homing", 1, 0, 1000.0, 0.0,
                                   reached=False) + "\n"
        odometry = self._fit(session, 0.998)["odometry"]

        self.assertNotIn("distance_scale_homing", odometry)
        self.assertFalse(odometry["calibrated"])

    def test_without_a_wall_distance_the_run_is_ignored(self):
        session = self._homing_run("wall_homing", 1, 0, 1000.0, 0.0) + "\n"
        odometry = self._fit(session, None)["odometry"]

        self.assertNotIn("distance_scale_homing", odometry)

    def test_the_wall_and_the_tape_are_both_reported(self):
        """Two estimates of the same scale, so a bad mark shows up."""
        session = "\n".join([
            self._homing_run("wall_homing", 1, 0, 1000.0, 0.0),
            "TEST_START odometry_straight",
            "META,run_id,2",
            "META,test,odometry_straight",
            "META,repeat,0",
            "META,poses,1",
            "DATA",
            "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,db_a_deg,"
            "db_w_deg_s,batt_mv",
            "0,100,0,0,0,0,0,8000",
            "10000,0,0,1000,0,0,0,8000",
            "TEST_END,2",
        ]) + "\n"
        poses = {"2": [{"stop": 0, "x_m": 0.990, "y_m": 0.0,
                        "heading_deg": 0.0}]}
        from overboard.sysid.fit import fit_session, load_poses, parse_session
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "poses.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("run_id,stop,x_m,y_m,heading_deg\n"
                             "2,0,0.990,0.0,0.0\n")
            loaded = load_poses(path)
        odometry = fit_session(parse_session(session), poses=loaded,
                              wheel_radius_m=0.028,
                              wall_distance_m=0.998)["odometry"]

        self.assertAlmostEqual(odometry["distance_scale_homing"], 0.998,
                               places=6)
        self.assertAlmostEqual(odometry["distance_scale_tape"], 0.990,
                               places=6)
        self.assertAlmostEqual(odometry["distance_scale_disagreement"], 0.008,
                               places=6)

    def test_the_profile_carries_the_wall_distance(self):
        from overboard.sysid.fit import read_drivebase_profile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "profile.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"name": "test", "wheel_radius_m": 0.028,
                           "wall": {"distance_m": 1.05}}, handle)
            profile = read_drivebase_profile(path)
        self.assertAlmostEqual(profile["wall_distance_m"], 1.05)


class TestSheetFormAndOneStepFit(unittest.TestCase):
    """The sheet as a form, and the fit straight from the sheet.

    The form is a browser page that downloads the CSV the fitter reads, so
    the operator types numbers instead of editing a spreadsheet. The tests
    here pin the contract between the two: the form holds every measurement
    the sheet asks for, and it writes the file the fitter takes.
    """

    def _runs(self, repeats=1):
        from overboard.sysid.fit import load_runs
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(_measurement_session(repeats))
            return load_runs([path])[1]

    def test_the_form_holds_every_measurement(self):
        from overboard.sysid.sheet_form import weigh_in_rows, write_form

        rows = sheet_rows(self._runs(repeats=2))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.html")
            write_form(rows, path, csv_name="calibration_sheet.csv",
                       fit_command="python sysid_fit.py session.csv")
            with open(path, encoding="utf-8") as handle:
                text = handle.read()

        # One row per measurement, keyed the way the sheet keys it, plus
        # the weigh-in rows the sheet always carries.
        self.assertEqual(
            text.count("<tr data-test="), len(rows) + len(weigh_in_rows()))
        self.assertIn(
            'data-test="multi_stop" data-repeat="1" data-stop="2"', text)
        self.assertIn('data-test="weigh_in" data-repeat="2"', text)
        # The header the fitter reads, and the file the form downloads.
        self.assertIn("test,repeat,stop,x_mm,y_mm,turn_right_deg,mass_kg",
                      text)
        self.assertIn('var HEADER = "test,repeat,stop,x_mm,y_mm,'
                      'turn_right_deg,mass_kg"', text)
        self.assertIn("calibration_sheet.csv", text)

    def test_the_form_and_the_sheet_agree_on_the_rows(self):
        from overboard.sysid.sheet_form import write_form

        runs = self._runs(repeats=2)
        rows = sheet_rows(runs)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sheet.csv")
            html_path = os.path.join(tmp, "sheet.html")
            write_sheet(runs, csv_path)
            write_form(rows, html_path)
            with open(csv_path, encoding="utf-8") as handle:
                csv_text = handle.read()
            with open(html_path, encoding="utf-8") as handle:
                html_text = handle.read()

        for row in rows:
            key = (row["test"], str(row["repeat"]), str(row["stop"]))
            self.assertIn(",".join(key) + ",", csv_text)
            self.assertIn(
                'data-test="{0}" data-repeat="{1}" data-stop="{2}"'.format(*key),
                html_text)

    def test_a_sheet_fills_in_the_browser_is_read_by_the_fitter(self):
        """The CSV the form downloads is the CSV the fitter takes.

        The form writes the columns in the order the fitter reads them, so a
        file that came out of the browser is fitted without being touched.
        """
        from overboard.sysid.fit import read_sheet

        csv_text = ("test,repeat,stop,x_mm,y_mm,turn_right_deg\n"
                    "odometry_straight,0,0,998,2,0\n"
                    "multi_stop,0,1,499,0,0\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(csv_text)
            rows = read_sheet(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["test"], "odometry_straight")
        self.assertAlmostEqual(rows[0]["x_mm"], 998.0)
        self.assertAlmostEqual(rows[1]["x_mm"], 499.0)

    def test_the_fit_takes_the_sheet_directly(self):
        """--sheet fits in one step: there is no intermediate pose CSV."""
        from overboard.sysid.fit import main as fit_main

        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            sheet = os.path.join(tmp, "sheet.csv")
            profile = os.path.join(tmp, "profile.json")
            fit = os.path.join(tmp, "fit.json")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write(_measurement_session())
            with open(profile, "w", encoding="utf-8") as handle:
                json.dump({"wheel_radius_m": 0.028}, handle)
            with open(sheet, "w", encoding="utf-8") as handle:
                handle.write("test,repeat,stop,x_mm,y_mm,turn_right_deg\n"
                             "odometry_straight,0,0,998,2,0\n")

            with redirect_stdout(io.StringIO()):
                code = fit_main([session, "--sheet", sheet,
                                 "--profile", profile, "--out", fit])
            with open(fit, encoding="utf-8") as handle:
                result = json.load(handle)

        self.assertEqual(code, 0)
        # The hub's odometry says 1000 mm, the operator measured 998 mm.
        self.assertAlmostEqual(result["odometry"]["distance_scale"], 0.998,
                               places=3)

    def test_a_sheet_with_no_measurement_is_reported(self):
        from overboard.sysid.fit import main as fit_main

        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            sheet = os.path.join(tmp, "sheet.csv")
            profile = os.path.join(tmp, "profile.json")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write(_measurement_session())
            with open(profile, "w", encoding="utf-8") as handle:
                json.dump({"wheel_radius_m": 0.028}, handle)
            with open(sheet, "w", encoding="utf-8") as handle:
                handle.write("test,repeat,stop,x_mm,y_mm,turn_right_deg\n")

            with redirect_stdout(io.StringIO()):
                code = fit_main([session, "--sheet", sheet,
                                 "--profile", profile,
                                 "--out", os.path.join(tmp, "fit.json")])

        self.assertEqual(code, 1)

    def test_the_cli_writes_the_form(self):
        from overboard.sysid.fit import main as fit_main

        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            form = os.path.join(tmp, "sheet.html")
            with open(session, "w", encoding="utf-8") as handle:
                handle.write(_measurement_session(repeats=2))
            with redirect_stdout(io.StringIO()):
                code = fit_main([session, "--write-form", form])
            self.assertTrue(os.path.exists(form))
            with open(form, encoding="utf-8") as handle:
                text = handle.read()

        self.assertEqual(code, 0)
        # The form names the CSV it downloads, and the command that fits it.
        self.assertIn("sheet.csv", text)
        self.assertIn("--sheet", text)


class TestGuidedSheetForm(unittest.TestCase):
    """The guided form: cards, diagram, units, warnings and progress.

    The form is the operator's side of the measurement convention, so it has
    to teach it: what each test does, what to measure, and the frame the
    numbers are written in. These pins hold the guidance in place; the CSV
    contract is pinned by the tests above.
    """

    def _form(self):
        from overboard.sysid.sheet_form import write_form

        rows = [
            {"test": "odometry_straight", "repeat": 0, "stop": 0},
            {"test": "odometry_turn_cw", "repeat": 0, "stop": 0},
            {"test": "multi_stop", "repeat": 0, "stop": 0},
            {"test": "multi_stop", "repeat": 0, "stop": 1},
            {"test": "multi_stop", "repeat": 0, "stop": 2},
            {"test": "brake_from_300", "repeat": 0, "stop": 0},
            {"test": "brake_from_600", "repeat": 0, "stop": 0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.html")
            write_form(rows, path, csv_name="calibration_sheet.csv",
                       fit_command="python sysid_fit.py s.csv --sheet c.csv")
            with open(path, encoding="utf-8") as handle:
                return handle.read()

    def test_every_test_gets_a_card_with_instructions(self):
        text = self._form()
        for test in ("odometry_straight", "odometry_turn_cw", "multi_stop",
                     "brake_from_300", "brake_from_600"):
            self.assertIn(f"<h2>{test} ", text)
        # What the robot does, what to measure, and the numbers wanted.
        self.assertEqual(text.count("What the robot does:"), 6)
        self.assertEqual(text.count("What to measure:"), 6)
        self.assertEqual(text.count("The numbers wanted:"), 6)
        # Plain-language content per test.
        self.assertIn("100 mm/s for 10 seconds", text)     # odometry_straight
        self.assertIn("pauses at three stops", text)       # multi_stop
        self.assertIn("brake slip", text)                  # brake_from_*
        self.assertIn("kitchen scale", text)               # weigh_in

    def test_an_unknown_test_still_gets_the_generic_instruction(self):
        from overboard.sysid.sheet_form import write_form

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sheet.html")
            write_form([{"test": "custom_test", "repeat": 0, "stop": 0}], path)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        self.assertIn("<h2>custom_test ", text)
        self.assertIn("Where the robot's centre ended at each stop", text)
        self.assertIn("x_mm: how far forward", text)

    def test_the_field_frame_diagram_is_inline(self):
        text = self._form()
        self.assertIn("<svg", text)
        self.assertIn("x_mm: forward", text)
        self.assertIn("y_mm: to the left", text)
        self.assertIn("turn_right_deg:", text)
        self.assertIn("to the right", text)
        # The convention is spelled out at the top, in its exact words.
        self.assertIn("forward from the start pose", text)
        self.assertIn("Measure from the robot's centre, not its bumper", text)

    def test_every_input_carries_a_unit_label(self):
        text = self._form()
        inputs = text.count("<input type=\"number\"")
        self.assertEqual(text.count('class="unit"'), inputs)
        self.assertIn(">mm</span>", text)
        self.assertIn(">deg</span>", text)
        self.assertIn(">kg</span>", text)

    def test_the_validation_warns_but_never_blocks(self):
        text = self._form()
        self.assertIn('id="warnings"', text)
        self.assertIn("turn_right_deg is usually 0 to 360 to the right", text)
        self.assertIn("mass_kg is outside 0.05 to 5 kg", text)
        self.assertIn("check the units (millimetres)", text)
        # Wires: the notes are filled in on input, and the download is not
        # gated on them.
        self.assertIn("function rowWarning", text)
        self.assertIn('q("#download").addEventListener', text)

    def test_the_progress_counts_what_is_missing(self):
        text = self._form()
        self.assertIn('id="progress-fill"', text)
        self.assertIn('id="missing"', text)
        self.assertIn("measurements still missing", text)
        self.assertIn("function done", text)
        self.assertIn(".card-count", text)


class TestCollectCommand(unittest.TestCase):

    def _run(self, tmp, extra=None):
        import overboard.sysid.collect as sysid_collect

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

        # The poses the hub reported are written out for the operator, but
        # they are the robot's own odometry, so the fit does not use them.
        self.assertFalse(result["odometry"]["calibrated"])
        self.assertFalse(result["slip"]["calibrated"])

    def test_the_hub_poses_are_not_handed_to_the_fit(self):
        """The robot must not be calibrated against its own odometry."""
        import overboard.sysid.collect as sysid_collect

        recorded = {}

        def fake_fit(argv):
            recorded["argv"] = list(argv)
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            poses = os.path.join(tmp, "poses.csv")
            profile = os.path.join(tmp, "profile.json")
            fit = os.path.join(tmp, "sysid_fit.json")
            with open(profile, "w", encoding="utf-8") as handle:
                json.dump({"wheel_radius_m": 0.028}, handle)

            def fake_deploy(script, connection="ble", name=None, wait=True,
                           print_output=True, stream_path=None):
                with open(stream_path, "w", encoding="utf-8") as handle:
                    handle.write(_session())
                return _session().splitlines()

            args = ["--session", session, "--poses", poses,
                    "--profile", profile, "--out", fit]
            with patch.object(sysid_collect, "deploy", fake_deploy):
                with patch.object(sysid_collect, "fit_main", fake_fit):
                    status = sysid_collect.main(args)

            self.assertEqual(status, 0)
            # The poses are still written out for the operator.
            self.assertTrue(os.path.exists(poses))

        argv = recorded["argv"]
        self.assertNotIn("--poses", argv)
        self.assertNotIn(poses, argv)

    def test_a_missing_profile_does_not_stop_the_collect(self):
        """Only the fit needs the wheel radius; the collect does not.

        The procedure documents the collect as its own step, so the command
        it names has to work. The fit comes after the measurements, and it
        gets the profile then.
        """
        import overboard.sysid.collect as sysid_collect

        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            poses = os.path.join(tmp, "poses.csv")
            fit = os.path.join(tmp, "sysid_fit.json")

            def fake_deploy(script, connection="ble", name=None, wait=True,
                            print_output=True, stream_path=None):
                with open(stream_path, "w", encoding="utf-8") as handle:
                    handle.write(_session())
                return _session().splitlines()

            args = ["--session", session, "--poses", poses]
            with patch.object(sysid_collect, "deploy", fake_deploy):
                with redirect_stdout(io.StringIO()):
                    status = sysid_collect.main(args)

            self.assertEqual(status, 0)
            self.assertTrue(os.path.exists(session))
            # There is no wheel radius, so there is no fit yet.
            self.assertFalse(os.path.exists(fit))

    def test_the_sheet_and_the_form_are_written(self):
        import overboard.sysid.collect as sysid_collect

        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "session.csv")
            poses = os.path.join(tmp, "poses.csv")
            profile = os.path.join(tmp, "profile.json")
            sheet = os.path.join(tmp, "calibration_sheet.csv")
            with open(profile, "w", encoding="utf-8") as handle:
                json.dump({"wheel_radius_m": 0.028}, handle)

            def fake_deploy(script, connection="ble", name=None, wait=True,
                            print_output=True, stream_path=None):
                with open(stream_path, "w", encoding="utf-8") as handle:
                    handle.write(_measurement_session())
                return _measurement_session().splitlines()

            args = ["--session", session, "--poses", poses,
                    "--profile", profile, "--sheet", sheet]
            with patch.object(sysid_collect, "deploy", fake_deploy):
                with redirect_stdout(io.StringIO()):
                    status = sysid_collect.main(args)

            self.assertEqual(status, 0)
            self.assertTrue(os.path.exists(sheet))
            form = os.path.splitext(sheet)[0] + ".html"
            self.assertTrue(os.path.exists(form))
            with open(sheet, encoding="utf-8") as handle:
                sheet_text = handle.read()
            with open(form, encoding="utf-8") as handle:
                form_text = handle.read()

        self.assertIn("test,repeat,stop,x_mm,y_mm,turn_right_deg", sheet_text)
        # The form and the sheet hold the same rows.
        self.assertEqual(form_text.count("<tr data-test="),
                         sheet_text.count("\n") - 1)

    def test_a_dropped_link_still_fits_what_arrived(self):
        """One flaky link must not throw away a ten minute session."""
        import overboard.sysid.collect as sysid_collect
        from overboard.deploy.hub_deploy import HubLinkError

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
        # What arrived is fitted, but the robot's own odometry is not a
        # measurement, so the odometry block stays uncalibrated.
        self.assertFalse(result["odometry"]["calibrated"])


if __name__ == "__main__":
    unittest.main()
