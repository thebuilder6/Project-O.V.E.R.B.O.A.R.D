"""
Tests for the user-facing CLI behaviour: defaults, error messages, presets,
the environment report and the live visualizer's port handling.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import immrax_validator  # noqa: F401
    HAS_IMMRAX = True
except ImportError:
    HAS_IMMRAX = False


def _samples(n=11, dt=0.1, speed=0.1):
    """A straight line that ends at x = 0.1 m, like the other test helpers."""
    return [{
        "t": i * dt, "x": speed * i * dt, "y": 0.0, "heading": 0.0,
        "vl": speed, "vr": speed, "omega": 0.0,
        "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
    } for i in range(n)]


class _RecordingOptimizer:
    """Stands in for the solver and records how it was called."""

    calls = []

    def __init__(self, config):
        self.config = config

    def solve(self, waypoints, **kwargs):
        _RecordingOptimizer.calls.append(kwargs)
        return _samples(), {"converged": True, "final_cost": 1.0}


class TestCliBasics(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def _invoke(self, args):
        import main
        return self.runner.invoke(main.main, args)

    def test_help_lists_the_user_facing_flags(self):
        result = self._invoke(["--help"])
        self.assertEqual(result.exit_code, 0)
        for flag in ("--gui", "--doctor", "--multiverse", "--preset"):
            self.assertIn(flag, result.output)

    def test_missing_waypoints_explains_what_to_pass(self):
        result = self._invoke([])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("--waypoints", result.output)
        self.assertIn("example_straight", result.output)

    def test_waypoints_file_must_be_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            result = self._invoke(["-w", bad])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("not valid JSON", result.output)

    def test_waypoints_file_needs_two_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wp.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}], handle)
            result = self._invoke(["-w", path])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("at least two waypoints", result.output)

    def test_waypoint_missing_coordinate_names_the_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wp.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"y": 1.0}], handle)
            result = self._invoke(["-w", path])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Waypoint 1", result.output)
        self.assertIn("'x'", result.output)

    def test_non_numeric_coordinate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wp.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"x": "left", "y": 0.0}, {"x": 1.0, "y": 0.0}], handle)
            result = self._invoke(["-w", path])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("must be a number", result.output)

    def test_stop_waypoint_out_of_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wp.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}], handle)
            result = self._invoke(["-w", path, "--stop-waypoints", "7"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("out of range", result.output)

    def test_malformed_event_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wp.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}], handle)
            result = self._invoke(["-w", path, "--events", "lower_arm"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("index:event", result.output)

    def test_simple_and_multiverse_cannot_be_combined(self):
        result = self._invoke(["-w", "examples/example_straight.json",
                                "--simple", "--multiverse"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("not both", result.output)

    def test_quiet_and_verbose_cannot_be_combined(self):
        result = self._invoke(["-w", "examples/example_straight.json",
                                "--quiet", "--verbose"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("cannot be combined", result.output)

    def test_doctor_reports_the_environment(self):
        result = self._invoke(["--doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Python", result.output)
        self.assertIn("casadi", result.output)

    def test_simple_solve_uses_the_default_config(self):
        _RecordingOptimizer.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            wp = os.path.join(tmp, "wp.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}], handle)
            import main
            with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer):
                result = self.runner.invoke(main.main, ["-w", wp, "-o", out, "-q"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(out))
        self.assertEqual(len(_RecordingOptimizer.calls), 1)

    def test_preset_fills_in_options_the_user_did_not_set(self):
        _RecordingOptimizer.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            wp = os.path.join(tmp, "wp.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}], handle)
            import main
            with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer):
                result = self.runner.invoke(
                    main.main, ["-w", wp, "-o", out, "-q", "--preset", "fast"])
            self.assertEqual(result.exit_code, 0, result.output)
        call = _RecordingOptimizer.calls[0]
        self.assertEqual(call["num_samples_per_segment"], 8)

    def test_explicit_option_beats_the_preset(self):
        _RecordingOptimizer.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            wp = os.path.join(tmp, "wp.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}], handle)
            import main
            with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer):
                result = self.runner.invoke(
                    main.main, ["-w", wp, "-o", out, "-q", "-n", "12",
                                "--preset", "fast"])
            self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_RecordingOptimizer.calls[0]["num_samples_per_segment"], 12)

    def test_solve_does_not_import_jax(self):
        """A simple solve must not pay for the JAX import."""
        code = "import sys, main; print('jax' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


class TestParallelCandidates(unittest.TestCase):
    """The candidates may be solved in worker processes; that must not
    change the trajectory the CLI exports."""

    def setUp(self):
        self.runner = CliRunner()
        self.waypoints = os.path.join(REPO_ROOT, "examples", "example_straight.json")

    def _run(self, extra_args):
        import main
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            result = self.runner.invoke(
                main.main, ["-w", self.waypoints, "-o", out, "--validate",
                            "--candidates", "2"] + extra_args)
            data = None
            if os.path.exists(out):
                with open(out, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
        return result, data

    def test_parallel_and_sequential_candidates_agree(self):
        try:
            import immrax  # noqa: F401
        except ImportError:
            self.skipTest("immrax not installed")

        parallel_result, parallel_data = self._run([])
        sequential_result, sequential_data = self._run(["--no-parallel"])

        self.assertEqual(parallel_result.exit_code, 0, parallel_result.output)
        self.assertEqual(sequential_result.exit_code, 0, sequential_result.output)
        self.assertIsNotNone(parallel_data)
        self.assertIsNotNone(sequential_data)

        parallel_samples = parallel_data["trajectory"]["samples"]
        sequential_samples = sequential_data["trajectory"]["samples"]
        self.assertEqual(len(parallel_samples), len(sequential_samples))
        for a, b in zip(parallel_samples, sequential_samples):
            for key in ("t", "x", "y", "heading", "vl", "vr"):
                self.assertAlmostEqual(a[key], b[key], places=9)

    def test_a_failing_candidate_keeps_the_export_closed(self):
        """A candidate that fails the gates must not be exported."""
        try:
            import immrax  # noqa: F401
        except ImportError:
            self.skipTest("immrax not installed")

        result, data = self._run([])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNotNone(data)


@unittest.skipUnless(HAS_IMMRAX, "jax/immrax not installed")
class TestValidationReport(unittest.TestCase):
    """The CLI prints a per-segment report the studio can read."""

    def setUp(self):
        import main
        self.main = main
        self.runner = CliRunner()
        self.repo = REPO_ROOT

    class _Optimizer:
        def __init__(self, config):
            pass

        def solve(self, *args, **kwargs):
            return [{
                "t": i * 0.1, "x": 0.01 * i, "y": 0.0, "heading": 0.0,
                "vl": 0.1, "vr": 0.1, "omega": 0.0,
                "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
            } for i in range(11)], {"converged": True, "final_cost": 1.0}

    class _WideImmrax:
        """An envelope that is over the local threshold."""

        def __init__(self, config, digital_twin=None, **kwargs):
            pass

        def validate_trajectory(self, samples, *ranges, **kwargs):
            envelope = [{
                "x_min": s["x"] - 0.05, "x_max": s["x"] + 0.05,
                "y_min": s["y"] - 0.05, "y_max": s["y"] + 0.05,
            } for s in samples]
            return {
                "passed": True,
                "max_motor_violation_N": 0.0,
                "max_traction_violation_N": 0.0,
                "max_slip_l_N": 0.0,
                "max_slip_r_N": 0.0,
                "max_tracking_error_m": 0.001,
                "reachability": {"max_error_m": 0.07, "envelope": envelope,
                                 "diverged": False},
            }

    def _run(self):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0, "heading": 0.0},
                           {"x": 0.1, "y": 0.0, "heading": 0.0}], handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._WideImmrax):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--simple", "--validate"])
        return result

    def test_report_names_the_failing_segment(self):
        result = self._run()
        marker = [line for line in result.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        self.assertEqual(len(marker), 1, result.output)
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        self.assertEqual(payload["selected"], 1)
        candidate = payload["candidates"][0]
        self.assertFalse(candidate["ok"])
        self.assertEqual(len(candidate["segments"]), 1)
        segment = candidate["segments"][0]
        self.assertFalse(segment["ok"])
        self.assertGreater(segment["envelope_m"], segment["limit_m"])
        self.assertEqual(segment["from"], 0)
        self.assertEqual(segment["to"], 1)

    def test_report_is_valid_json_without_infinities(self):
        result = self._run()
        marker = [line for line in result.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        # json.dumps would write Infinity, which is not valid JSON.
        self.assertNotIn("Infinity", marker[0])
        self.assertIsNotNone(payload["candidates"][0]["duration_s"])


class TestIterationHistory(unittest.TestCase):
    """--iteration-history writes how the trajectory converged."""

    def test_the_history_has_the_first_guess_and_the_solution(self):
        import main
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            history = os.path.join(tmp, "iterations.json")
            result = runner.invoke(main.main, [
                "-w", "examples/example_straight.json", "-o", out,
                "--iteration-history", history, "-q"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(history))
            with open(history, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["phase"], "initial_guess")
        self.assertEqual(entries[1]["phase"], "final_solution")
        self.assertIn("cost", entries[0])
        self.assertTrue(entries[0]["trajectory"])


class TestStudioEntryPoints(unittest.TestCase):
    """--gui and --live both open the studio; --live solves the mission."""

    def setUp(self):
        import main
        self.main = main
        self.runner = CliRunner()

    def test_live_and_gui_cannot_be_combined(self):
        result = self.runner.invoke(self.main.main, [
            "--gui", "--live", "-w", "examples/example_straight.json"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("both open the studio", result.output)

    def test_live_needs_a_mission(self):
        result = self.runner.invoke(self.main.main, ["--live"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("--waypoints", result.output)

    def test_studio_options_carry_the_cli_settings(self):
        options = self.main._studio_options(
            samples=14, accuracy_weight=0.005, candidates=3, dt_mode="per-segment",
            validate=True, export_format="controller", controller_dt=0.02,
            fast_polish=True, sysid_fit="fit.json")
        self.assertEqual(options["samples"], 14)
        self.assertEqual(options["timing"], "per-segment")
        self.assertTrue(options["validate"])
        self.assertEqual(options["export_format"], "controller")
        self.assertTrue(options["fast_polish"])
        self.assertEqual(options["sysid_fit"], "fit.json")


if __name__ == "__main__":
    unittest.main()
