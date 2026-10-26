"""
Tests for the user-facing CLI behaviour: defaults, error messages, presets,
the environment report and the live visualizer's port handling.
"""

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from click.testing import CliRunner

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import overboard.verification.immrax_validator as immrax_validator  # noqa: F401
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


def _mission(points, target=1.0, max_time=2.0):
    """A mission file body with the scoring timing a validated run needs."""
    return {"waypoints": points,
            "scoring": {"target_time_s": target, "max_time_s": max_time}}


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
        import overboard.main as main
        return self.runner.invoke(main.main, args)

    def test_help_lists_the_user_facing_flags(self):
        result = self._invoke(["--help"])
        self.assertEqual(result.exit_code, 0)
        for flag in ("--gui", "--doctor", "--preset", "--candidates", "--polish"):
            self.assertIn(flag, result.output)

    def test_the_propagation_flag_is_offered(self):
        """The reachability model is a user-facing choice, so it is a flag."""
        result = self._invoke(["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--propagation", result.output)
        self.assertIn("tick", result.output)

    def test_the_default_propagation_is_the_validator_default(self):
        """The CLI and the validator must not drift apart on the default.

        The CLI names the propagation explicitly, so the validator's own
        default is only reachable in tests and in the research scripts. If
        the two disagree, a run through the CLI measures a different model
        from the one the research scripts measured.
        """
        if not HAS_IMMRAX:
            self.skipTest("immrax not installed")
        import overboard.verification.immrax_validator as immrax_validator
        import overboard.main as main
        for param in main.main.params:
            if param.name == "propagation":
                self.assertEqual(param.default, immrax_validator.PROPAGATION)
                return
        self.fail("main.main has no --propagation option")

    def test_the_report_names_the_model_that_produced_the_numbers(self):
        """An envelope is the width of a tube, and the tube has a model.

        The report is the machine-readable line the studio parses and the
        offline analysis reads, so the propagation and the inclusion
        functions have to be in it: a number whose model is not stated
        cannot be compared with a number from another run (KI-049).
        """
        import overboard.main as main
        attempts = [{
            "guess_style": "baseline", "per_segment_dt": False,
            "accuracy_weight": 0.0, "duration": 2.0, "tube": 0.003,
            "envelope_truncated": False, "propagation": "tick",
            "inclusion": {"plant": "nat", "controller": "nat", "tick": "mjac"},
            "tracking_m": 0.001, "ok": True, "converged": True, "reasons": [],
            "segments": [],
        }]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main._emit_validation_report(attempts, 1, propagation="tick")
        payload = json.loads(
            buffer.getvalue().split(main.VALIDATION_MARKER)[1])
        self.assertEqual(payload["propagation"], "tick")
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["propagation"], "tick")
        self.assertEqual(candidate["inclusion"]["tick"], "mjac")
        # The envelope is still there, with its truncation flag.
        self.assertAlmostEqual(candidate["envelope_cm"], 0.3)
        self.assertFalse(candidate["envelope_truncated"])

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
            import overboard.main as main
            with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer):
                result = self.runner.invoke(main.main, ["-w", wp, "-o", out, "-q"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(out))
            # No --validate: the file says so, in the same place a passed one
            # records its pass, so it cannot be read as an approved plan.
            with open(out, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(data["validation"]["status"], "unvalidated")
            self.assertFalse(data["validation"]["passed"])
        self.assertEqual(len(_RecordingOptimizer.calls), 1)

    def test_preset_fills_in_options_the_user_did_not_set(self):
        _RecordingOptimizer.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            wp = os.path.join(tmp, "wp.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}], handle)
            import overboard.main as main
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
            import overboard.main as main
            with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer):
                result = self.runner.invoke(
                    main.main, ["-w", wp, "-o", out, "-q", "-n", "12",
                                "--preset", "fast"])
            self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(_RecordingOptimizer.calls[0]["num_samples_per_segment"], 12)

    def test_solve_does_not_import_jax(self):
        """A simple solve must not pay for the JAX import."""
        code = "import sys, overboard.main; print('jax' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


class TestUnifiedDefaults(unittest.TestCase):
    """The unified engine is the default; explicit flags still win."""

    def setUp(self):
        self.runner = CliRunner()

    def _invoke(self, extra):
        _RecordingOptimizer.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            wp = os.path.join(tmp, "wp.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}], handle)
            import overboard.main as main
            with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer):
                result = self.runner.invoke(
                    main.main, ["-w", wp, "-o", out, "-q"] + extra)
        self.assertEqual(result.exit_code, 0, result.output)
        return list(_RecordingOptimizer.calls)

    def test_unified_is_the_default(self):
        call = self._invoke([])[0]
        self.assertEqual(call["guess_style"], "frontier_sprint")
        self.assertTrue(call["per_segment_dt"])
        self.assertEqual(call["cost_metric"], "discretization")
        self.assertTrue(call["epigraph_constraints"])


    def test_explicit_samples_and_weight_reach_the_unified_specs(self):
        call = self._invoke(["-n", "12", "-a", "0.005"])[0]
        self.assertEqual(call["num_samples_per_segment"], 12)
        self.assertAlmostEqual(call["accuracy_weight"], 0.005)


class TestFirstRun(unittest.TestCase):
    """A first-time user must get the install command, not a traceback."""

    def test_missing_core_packages_names_what_is_missing(self):
        import overboard.diagnostics as diagnostics
        with patch.object(diagnostics, "module_available",
                          side_effect=lambda name: name != "casadi"):
            self.assertEqual(diagnostics.missing_core_packages(), ["casadi"])

    def test_the_doctor_names_the_install_command(self):
        import overboard.diagnostics as diagnostics
        with patch.object(diagnostics, "module_available",
                          side_effect=lambda name: name != "casadi"):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = diagnostics.run_doctor()
        output = buffer.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("pip install -r requirements.txt", output)
        self.assertIn(diagnostics.install_script(), output)

    def test_main_without_the_packages_reports_the_install(self):
        """The first run on a clean machine must not die with a traceback."""
        code = (
            "import sys, runpy\n"
            "for name in ('click', 'numpy', 'casadi', 'plotly', 'matplotlib'):\n"
            "    sys.modules[name] = None\n"
            "sys.argv = ['main.py', '--doctor']\n"
            "runpy.run_path('overboard/main.py', run_name='__main__')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertNotIn("Traceback", result.stdout)
        self.assertIn("pip install -r requirements.txt", result.stdout)

    def test_the_batch_files_keep_crlf(self):
        """cmd.exe cannot find a batch label in a file with LF endings.

        install.bat calls a subroutine, so an LF file makes it fall over with
        "The system cannot find the batch label specified".
        """
        for name in ("install.bat", "start_studio.bat"):
            with open(os.path.join(REPO_ROOT, name), "rb") as handle:
                data = handle.read()
            self.assertIn(b"\r\n", data, name)
            self.assertNotIn(b"\n", data.replace(b"\r\n", b""), name)


class TestParallelCandidates(unittest.TestCase):
    """The candidates may be solved in worker processes; that must not
    change the trajectory the CLI exports."""

    def setUp(self):
        self.runner = CliRunner()
        self.waypoints = os.path.join(REPO_ROOT, "examples", "example_straight.json")

    def _run(self, extra_args):
        import overboard.main as main
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
        import overboard.main as main
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
                json.dump(_mission([{"x": 0.0, "y": 0.0, "heading": 0.0},
                                    {"x": 0.1, "y": 0.0, "heading": 0.0}]),
                          handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._WideImmrax):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--validate"])
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

    def _invoke(self, extra_args, validator=None):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission([{"x": 0.0, "y": 0.0, "heading": 0.0},
                                    {"x": 0.1, "y": 0.0, "heading": 0.0}]),
                          handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator",
                                 validator or self._WideImmrax):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--validate"] + list(extra_args))
            data = None
            if os.path.exists(out):
                with open(out, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
        return result, data

    def test_best_effort_writes_the_failed_path_with_its_verdict(self):
        """The path is shown, and the file says it is not an approved one."""
        result, data = self._invoke(["--best-effort"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("BEST EFFORT", result.output)
        self.assertIsNotNone(data)
        self.assertTrue(data["validation"]["best_effort"])
        self.assertFalse(data["validation"]["passed"])
        self.assertGreater(data["validation"]["max_excess_m"], 0.0)
        self.assertIn("exceeds the local threshold",
                      " ".join(data["validation"]["reasons"]))
        self.assertIn("not approved", result.output)
        # The old second report called the same file safe to run.
        self.assertNotIn("safe to run", result.output)

    def test_a_missing_output_directory_is_a_plain_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission([{"x": 0.0, "y": 0.0, "heading": 0.0},
                                    {"x": 0.1, "y": 0.0, "heading": 0.0}]),
                          handle)
            out = os.path.join(tmp, "missing", "out.traj")
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "-n", "5", "--candidates", "1"])
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("Output directory", result.output)
            self.assertNotIn("Traceback", result.output)
            self.assertFalse(os.path.exists(out))

    def test_a_failed_write_keeps_the_previous_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission([{"x": 0.0, "y": 0.0, "heading": 0.0},
                                    {"x": 0.1, "y": 0.0, "heading": 0.0}]),
                          handle)
            with open(out, "w", encoding="utf-8") as handle:
                handle.write("previous plan")
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch("overboard.main.json.dump", side_effect=OSError("disk full")):
                self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "-n", "5", "--candidates", "1"])
            with open(out, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "previous plan")

    def test_best_effort_still_refuses_a_physically_unsound_plan(self):
        """A slip is not an accuracy drop, so it is not exported this way."""
        class _SlippingImmrax(self._WideImmrax):
            def validate_trajectory(self, samples, *ranges, **kwargs):
                report = super().validate_trajectory(samples, *ranges, **kwargs)
                report["max_slip_l_N"] = 0.5
                return report

        result, data = self._invoke(["--best-effort"], validator=_SlippingImmrax)
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIsNone(data)
        self.assertIn("No candidate failed on accuracy alone", result.output)
        # The message names the categories; the report names the number that
        # decided, so a refusal still leaves it behind.
        marker = [line for line in result.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        self.assertEqual(len(marker), 1, result.output)
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        self.assertIn("wheel slip", " ".join(payload["candidates"][0]["reasons"]))

    def test_best_effort_needs_validation(self):
        result = self.runner.invoke(self.main.main, [
            "-w", os.path.join(self.repo, "examples", "example_straight.json"),
            "--best-effort"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--best-effort needs --validate", result.output)

    def test_the_accuracy_limit_is_configurable_and_recorded(self):
        """The budget is a declaration: 4.2 cm fails at 2 cm, passes at 5 cm."""

        class _MediumImmrax(self._WideImmrax):
            def validate_trajectory(self, samples, *ranges, **kwargs):
                report = super().validate_trajectory(samples, *ranges, **kwargs)
                report["reachability"]["envelope"] = [{
                    "x_min": s["x"] - 0.03, "x_max": s["x"] + 0.03,
                    "y_min": s["y"] - 0.03, "y_max": s["y"] + 0.03,
                } for s in samples]
                return report

        failed, data = self._invoke([], validator=_MediumImmrax)
        self.assertNotEqual(failed.exit_code, 0, failed.output)
        self.assertIsNone(data)

        passed, data = self._invoke(["--envelope-limit", "0.05"],
                                    validator=_MediumImmrax)
        self.assertEqual(passed.exit_code, 0, passed.output)
        self.assertTrue(data["validation"]["passed"])
        self.assertAlmostEqual(data["validation"]["envelope_limit_m"], 0.05)

        marker = [line for line in passed.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        self.assertAlmostEqual(payload["envelope_limit_m"], 0.05)
        self.assertAlmostEqual(payload["candidates"][0]["segments"][0]["limit_m"],
                               0.05)

    def test_the_envelope_limit_can_grow_per_metre(self):
        """A tube that widens with distance can be priced per metre (KI-054)."""

        class _GrowingImmrax(self._WideImmrax):
            def validate_trajectory(self, samples, *ranges, **kwargs):
                report = super().validate_trajectory(samples, *ranges, **kwargs)
                report["reachability"]["envelope"] = []
                for s in samples:
                    half = 0.001 + 0.04 * s["x"]
                    report["reachability"]["envelope"].append({
                        "x_min": s["x"] - half, "x_max": s["x"] + half,
                        "y_min": s["y"], "y_max": s["y"],
                    })
                return report

        base = ["--envelope-limit", "0.001"]
        failed, data = self._invoke(base, validator=_GrowingImmrax)
        self.assertNotEqual(failed.exit_code, 0, failed.output)
        self.assertIsNone(data)

        passed, data = self._invoke(
            base + ["--envelope-limit-per-metre", "0.05"],
            validator=_GrowingImmrax)
        self.assertEqual(passed.exit_code, 0, passed.output)
        self.assertTrue(data["validation"]["passed"])
        self.assertAlmostEqual(
            data["validation"]["envelope_limit_per_metre"], 0.05)

        marker = [line for line in passed.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        self.assertAlmostEqual(payload["envelope_limit_per_metre"], 0.05)

    @staticmethod
    def _field(payload):
        from overboard.viz.fields import field_from_dict
        return field_from_dict(payload)

    @staticmethod
    def _envelope_validator(half):
        class _Envelope:
            def __init__(self, config, digital_twin=None, **kwargs):
                pass

            def validate_trajectory(self, samples, *ranges, **kwargs):
                return {
                    "passed": True,
                    "max_motor_violation_N": 0.0,
                    "max_traction_violation_N": 0.0,
                    "max_slip_l_N": 0.0,
                    "max_slip_r_N": 0.0,
                    "max_tracking_error_m": 0.001,
                    "reachability": {
                        "max_error_m": half,
                        "envelope": [{
                            "x_min": s["x"] - half, "x_max": s["x"] + half,
                            "y_min": s["y"] - half, "y_max": s["y"] + half,
                        } for s in samples],
                        "diverged": False},
                }
        return _Envelope

    def test_the_field_limit_clamps_per_sample(self):
        """The clearance is the limit, clamped to 0.5-5 cm (-KI-054)."""
        validator = self._envelope_validator(0.02)
        # The path runs along y = 0: with the mat's south edge declared there,
        # the disc penetrates the boundary and the limit clamps down.
        tight = self._field({
            "season": "tight-field", "placement_uncertainty_m": 0.1,
            "mat": {"width_m": 2.0, "height_m": 1.0},
            "obstacles": []})
        with patch("overboard.viz.fields.load_field", return_value=tight):
            failed, data = self._invoke(
                ["--envelope-from-field", "tight-field",
                 "--envelope-limit", "0.05"], validator=validator)
        self.assertNotEqual(failed.exit_code, 0, failed.output)
        self.assertIsNone(data)
        marker = [line for line in failed.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        self.assertAlmostEqual(
            payload["candidates"][0]["segments"][0]["limit_m"], 0.005)

        # No mat and nothing on the field: the clearance is unbounded and
        # clamps up to the tube's own 50 mm guard.
        loose = self._field({
            "season": "loose-field", "placement_uncertainty_m": 0.1,
            "obstacles": []})
        with patch("overboard.viz.fields.load_field", return_value=loose):
            passed, data = self._invoke(
                ["--envelope-from-field", "loose-field",
                 "--envelope-limit", "0.001"], validator=validator)
        self.assertEqual(passed.exit_code, 0, passed.output)
        self.assertTrue(data["validation"]["passed"])

    def test_a_waypoint_threshold_wins_over_the_field(self):
        """error_threshold / precision_mode keep their local authority."""
        validator = self._envelope_validator(0.02)
        tight = self._field({
            "season": "tight-field", "placement_uncertainty_m": 0.1,
            "mat": {"width_m": 2.0, "height_m": 1.0},
            "obstacles": []})
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission(
                    [{"x": 0.0, "y": 0.0, "heading": 0.0},
                     {"x": 0.1, "y": 0.0, "heading": 0.0,
                      "error_threshold": 0.05}]), handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", validator), \
                    patch("overboard.viz.fields.load_field", return_value=tight):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--validate",
                    "--envelope-from-field", "tight-field",
                    "--envelope-limit", "0.001"])
            with open(out, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(data["validation"]["passed"])

    def test_a_sample_the_field_cannot_measure_falls_back(self):
        """NaN is no field limit - it is never clamped (KI-054)."""
        class _Unmeasured:
            season = "unmeasured"
            placement_uncertainty_m = 0.1

            def sample_clearances(self, samples, robot_radius_m):
                return [float("nan")] * len(samples)

        validator = self._envelope_validator(0.02)
        with patch("overboard.viz.fields.load_field", return_value=_Unmeasured()):
            passed, data = self._invoke(
                ["--envelope-from-field", "unmeasured",
                 "--envelope-limit", "0.05"], validator=validator)
        # Clamped, a NaN would become 0.005 and fail the 2.8 cm envelope.
        self.assertEqual(passed.exit_code, 0, passed.output)
        self.assertTrue(data["validation"]["passed"])
        self.assertAlmostEqual(data["validation"]["envelope_limit_m"], 0.05)
        summary = data["validation"]["envelope_limit_field_m"]
        self.assertIsNone(summary["min_m"])
        self.assertIsNone(summary["max_m"])

    def test_the_field_limit_and_the_uncertainty_are_recorded(self):
        validator = self._envelope_validator(0.02)
        loose = self._field({
            "season": "loose-field", "placement_uncertainty_m": 0.1,
            "obstacles": []})
        with patch("overboard.viz.fields.load_field", return_value=loose):
            passed, data = self._invoke(
                ["--envelope-from-field", "loose-field"], validator=validator)
        self.assertEqual(passed.exit_code, 0, passed.output)
        summary = data["validation"]["envelope_limit_field_m"]
        self.assertEqual(summary["season"], "loose-field")
        self.assertAlmostEqual(summary["min_m"], 0.05)
        self.assertAlmostEqual(summary["max_m"], 0.05)
        self.assertAlmostEqual(data["validation"]["placement_uncertainty_m"], 0.1)

        marker = [line for line in passed.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        self.assertEqual(len(marker), 1)
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        self.assertEqual(payload["envelope_limit_field_m"]["season"], "loose-field")
        self.assertAlmostEqual(payload["placement_uncertainty_m"], 0.1)
        self.assertNotIn("NaN", marker[0])
        self.assertNotIn("Infinity", marker[0])

    def test_an_unknown_field_season_is_refused(self):
        result, data = self._invoke(["--envelope-from-field", "no-such-season"])
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIsNone(data)
        self.assertIn("There is no field for the season", result.output)
        self.assertNotIn("Traceback", result.output)

    def test_every_candidate_carries_its_score_vector(self):
        """The selection rule can be re-run offline on these numbers.

        Q10's step 1 and Q3's step 1 are the same instrumentation: the
        duration, the envelope, the tracking error, the biggest excess over
        the local threshold, and the peak wheel jerk.
        """
        result = self._run()
        marker = [line for line in result.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        for candidate in payload["candidates"]:
            for field in ("duration_s", "envelope_cm", "tracking_cm",
                          "max_excess_cm", "peak_jerk_m_s3",
                          "peak_force_util", "peak_traction_util"):
                self.assertIn(field, candidate)
                self.assertIsNotNone(candidate[field], field)
            # The excess agrees with the segments it came from.
            segments = candidate["segments"]
            excess = max(segment["envelope_m"] - segment["limit_m"]
                         for segment in segments)
            self.assertAlmostEqual(candidate["max_excess_cm"],
                                   excess * 100.0, places=6)

    class _RecordingImmrax(_WideImmrax):
        """A passing envelope, plus the propagation it was built with."""

        seen = []

        def __init__(self, config, digital_twin=None, propagation="plant", **kwargs):
            type(self).seen.append(propagation)

        def validate_trajectory(self, samples, *ranges, **kwargs):
            return self._report(samples)

        def _report(self, samples):
            return {
                "passed": True,
                "max_motor_violation_N": 0.0,
                "max_traction_violation_N": 0.0,
                "max_slip_l_N": 0.0,
                "max_slip_r_N": 0.0,
                "max_tracking_error_m": 0.001,
                "reachability": {"max_error_m": 0.001, "diverged": False,
                                 "envelope": [{"x_min": s["x"] - 0.001,
                                               "x_max": s["x"] + 0.001,
                                               "y_min": s["y"] - 0.001,
                                               "y_max": s["y"] + 0.001}
                                              for s in samples]},
            }

    def test_safe_preset_validates_with_the_closed_loop_model(self):
        """The widened example ranges do not fit an open-loop tube.

        `safe` is the preset that pays for the closed-loop model, so it has
        to reach the validator as `tick` - and an explicit flag still wins.
        """
        self._RecordingImmrax.seen = []
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission([{"x": 0.0, "y": 0.0, "heading": 0.0},
                                    {"x": 0.1, "y": 0.0, "heading": 0.0}]),
                          handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator",
                                 self._RecordingImmrax):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "-q", "--preset", "safe",
                    "--no-parallel"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(self._RecordingImmrax.seen)
            self.assertEqual(set(self._RecordingImmrax.seen), {"tick"})

            self._RecordingImmrax.seen = []
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator",
                                 self._RecordingImmrax):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "-q", "--preset", "safe",
                    "--propagation", "plant", "--no-parallel"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(set(self._RecordingImmrax.seen), {"plant"})

    # --- The decision score contract --------------------------------------

    def test_a_validated_run_needs_the_scoring_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            # A plain list is still a valid mission - but a validated run
            # then needs the timing on the command line.
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0, "heading": 0.0},
                           {"x": 0.1, "y": 0.0, "heading": 0.0}], handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._WideImmrax):
                missing = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--validate"])
            self.assertNotEqual(missing.exit_code, 0, missing.output)
            self.assertIn("target_time_s", missing.output)
            self.assertIn("--target-time", missing.output)
            self.assertIn("max_time_s", missing.output)
            self.assertIn("--max-time", missing.output)
            self.assertFalse(os.path.exists(out))
            # The flags supply it, and the run goes through.
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._RecordingImmrax):
                flagged = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--validate",
                    "--target-time", "1", "--max-time", "2"])
            self.assertEqual(flagged.exit_code, 0, flagged.output)

    def test_the_mission_scoring_budget_is_read_and_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission([{"x": 0.0, "y": 0.0, "heading": 0.0},
                                    {"x": 0.1, "y": 0.0, "heading": 0.0}],
                                   target=1.0, max_time=2.0), handle)
            args = ["-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--validate", "-q"]
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._RecordingImmrax):
                result = self.runner.invoke(self.main.main, args)
            self.assertEqual(result.exit_code, 0, result.output)
            with open(out, "r", encoding="utf-8") as handle:
                record = json.load(handle)["validation"]
            self.assertEqual(record["target_time_s"], 1.0)
            self.assertEqual(record["max_time_s"], 2.0)
            # The flags win over the mission metadata.
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._RecordingImmrax):
                result = self.runner.invoke(self.main.main, args + [
                    "--target-time", "0.5", "--max-time", "6"])
            self.assertEqual(result.exit_code, 0, result.output)
            with open(out, "r", encoding="utf-8") as handle:
                record = json.load(handle)["validation"]
            self.assertEqual(record["target_time_s"], 0.5)
            self.assertEqual(record["max_time_s"], 6.0)

    def test_a_bad_scoring_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission([{"x": 0.0, "y": 0.0},
                                    {"x": 0.1, "y": 0.0}],
                                   target=2.0, max_time=1.0), handle)
            result = self.runner.invoke(self.main.main, [
                "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                "-w", wp, "-o", out, "--validate"])
            self.assertNotEqual(result.exit_code, 0, result.output)
            self.assertIn("target_time_s", result.output)
            self.assertFalse(os.path.exists(out))

    def test_the_report_carries_the_score_contract_fields(self):
        result, data = self._invoke([], validator=self._RecordingImmrax)
        self.assertEqual(result.exit_code, 0, result.output)
        marker = [line for line in result.output.splitlines()
                  if line.startswith(self.main.VALIDATION_MARKER)]
        payload = json.loads(marker[0][len(self.main.VALIDATION_MARKER):])
        self.assertEqual(payload["score_version"], 1)
        self.assertEqual(payload["profile"], "balanced")
        self.assertIsNone(payload["accept_score"])
        self.assertFalse(payload["accepted"])
        self.assertFalse(payload["best_effort"])
        self.assertTrue(payload["hub_approved"])
        for candidate in payload["candidates"]:
            self.assertIn("score", candidate)
            self.assertEqual(candidate["score_version"], 1)
            self.assertEqual(sorted(candidate["categories"]),
                             ["accuracy", "duration", "physical", "quality"])
            self.assertEqual(sorted(candidate["ratios"]), ["r_acc", "r_phys"])
            self.assertTrue(candidate["eligibility"])
            self.assertFalse(candidate["structural_failure"])
        # The artifact carries the same verdict, with its hash.
        self.assertTrue(data["validation"]["passed"])
        self.assertFalse(data["validation"]["accepted"])
        self.assertTrue(data["validation"]["hub_approved"])
        self.assertEqual(data["validation"]["score_version"], 1)
        self.assertEqual(sorted(data["validation"]["categories"]),
                         ["accuracy", "duration", "physical", "quality"])
        self.assertEqual(data["validation"]["profile"], "balanced")
        self.assertEqual(data["validation"]["target_time_s"], 1.0)
        self.assertEqual(data["validation"]["max_time_s"], 2.0)
        self.assertRegex(data["validation"]["trajectory_sha256"],
                         r"^[0-9a-f]{64}$")

    def test_the_named_profiles_change_the_weights(self):
        for profile in ("precision", "balanced", "speed"):
            result, data = self._invoke(["--scoring-profile", profile],
                                        validator=self._RecordingImmrax)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(data["validation"]["profile"], profile)
            self.assertEqual(sum(data["validation"]["weights"].values()), 100.0)

    def test_an_accepted_score_exports_and_approves_a_failed_plan(self):
        """A quantified accuracy excess can be accepted and pushed."""

        class _SlightlyWideImmrax(self._WideImmrax):
            def validate_trajectory(self, samples, *ranges, **kwargs):
                report = super().validate_trajectory(samples, *ranges, **kwargs)
                # 2.5 cm of envelope against a 2 cm limit: ratio 1.25.
                report["reachability"]["envelope"] = [{
                    "x_min": s["x"] - 0.025, "x_max": s["x"] + 0.025,
                    "y_min": s["y"], "y_max": s["y"],
                } for s in samples]
                return report

        refused, data = self._invoke([], validator=_SlightlyWideImmrax)
        self.assertNotEqual(refused.exit_code, 0, refused.output)
        self.assertIsNone(data)

        accepted, data = self._invoke(
            ["--accept-score", "50"], validator=_SlightlyWideImmrax)
        self.assertEqual(accepted.exit_code, 0, accepted.output)
        self.assertIsNotNone(data)
        record = data["validation"]
        self.assertFalse(record["passed"])
        self.assertTrue(record["accepted"])
        self.assertTrue(record["hub_approved"])
        self.assertFalse(record["best_effort"])
        self.assertIsNotNone(record["score"])
        self.assertGreaterEqual(record["score"], 50.0)
        self.assertEqual(record["accept_score"], 50)
        self.assertIn("ACCEPTED", accepted.output)

        # Below the score nothing is written.
        high, data = self._invoke(["--accept-score", "100"],
                                  validator=_SlightlyWideImmrax)
        self.assertNotEqual(high.exit_code, 0, high.output)
        self.assertIsNone(data)

    def test_a_structural_failure_is_never_accepted(self):
        """A diverged tube cannot be bought with any --accept-score."""

        class _DivergedImmrax(self._WideImmrax):
            def validate_trajectory(self, samples, *ranges, **kwargs):
                report = super().validate_trajectory(samples, *ranges, **kwargs)
                report["reachability"]["diverged"] = True
                return report

        result, data = self._invoke(["--accept-score", "1"],
                                    validator=_DivergedImmrax)
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIsNone(data)

    def test_accept_score_needs_validation(self):
        result = self.runner.invoke(self.main.main, [
            "-w", os.path.join(self.repo, "examples", "example_straight.json"),
            "--accept-score", "50"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--accept-score needs --validate", result.output)

    def test_the_unvalidated_file_is_never_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump(_mission([{"x": 0.0, "y": 0.0, "heading": 0.0},
                                    {"x": 0.1, "y": 0.0, "heading": 0.0}]),
                          handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "-q", "-n", "5", "--candidates", "1"])
            self.assertEqual(result.exit_code, 0, result.output)
            with open(out, "r", encoding="utf-8") as handle:
                record = json.load(handle)["validation"]
        self.assertFalse(record["hub_approved"])
        self.assertFalse(record["passed"])
        self.assertFalse(record["best_effort"])
        self.assertRegex(record["trajectory_sha256"], r"^[0-9a-f]{64}$")

    def test_best_effort_is_written_unapproved_with_its_score(self):
        result, data = self._invoke(["--best-effort"])
        self.assertEqual(result.exit_code, 0, result.output)
        record = data["validation"]
        self.assertTrue(record["best_effort"])
        self.assertFalse(record["passed"])
        self.assertFalse(record["accepted"])
        self.assertFalse(record["hub_approved"])
        self.assertEqual(record["score_version"], 1)
        self.assertIsNotNone(record["score"])
        self.assertRegex(record["trajectory_sha256"], r"^[0-9a-f]{64}$")


class TestVariableSampleLayout(unittest.TestCase):
    """The gate follows the optimizer's own sample layout.

    The optimizer splits the sample budget across the waypoint segments in
    proportion to their length, so the waypoints are not k *
    samples_per_segment apart. A consumer that assumes they are applies a
    waypoint's envelope threshold to the wrong samples, and groups the
    per-segment report wrongly.
    """

    def setUp(self):
        import overboard.main as main
        self.main = main
        self.runner = CliRunner()
        self.repo = REPO_ROOT

    # Three segments with four, two and three samples: ten samples in all.
    COUNTS = [4, 2, 3]

    class _Optimizer:
        def __init__(self, config):
            pass

        def solve(self, *args, **kwargs):
            n = sum(TestVariableSampleLayout.COUNTS) + 1
            samples = [{
                "t": i * 0.1, "x": 0.01 * i, "y": 0.0, "heading": 0.0,
                "vl": 0.1, "vr": 0.1, "omega": 0.0,
                "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
            } for i in range(n)]
            return samples, {"converged": True, "final_cost": 1.0,
                             "segment_samples":
                                 list(TestVariableSampleLayout.COUNTS)}

    class _Immrax:
        """A narrow envelope, except on the second segment's samples."""

        def __init__(self, config, digital_twin=None, **kwargs):
            pass

        def validate_trajectory(self, samples, *ranges, **kwargs):
            envelope = []
            for i, s in enumerate(samples):
                # Samples 4 and 5 are the second segment's.
                half = 0.05 if i in (4, 5) else 0.001
                envelope.append({
                    "x_min": s["x"] - half, "x_max": s["x"] + half,
                    "y_min": s["y"] - half, "y_max": s["y"] + half,
                })
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
                json.dump(_mission([
                    {"x": 0.0, "y": 0.0, "heading": 0.0},
                    {"x": 0.1, "y": 0.0, "heading": 0.0},
                    {"x": 0.2, "y": 0.0, "heading": 0.0},
                    {"x": 0.3, "y": 0.0, "heading": 0.0}]), handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._Immrax):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--validate"])
        marker = [line for line in result.output.splitlines()
                   if line.startswith(self.main.VALIDATION_MARKER)]
        self.assertEqual(len(marker), 1, result.output)
        return json.loads(marker[0][len(self.main.VALIDATION_MARKER):])

    def test_the_segments_follow_the_optimizers_layout(self):
        """Each segment holds its own samples, not a tenth of the run."""
        payload = self._run()
        segments = payload["candidates"][0]["segments"]
        self.assertEqual(len(segments), 3)
        # The envelope of a segment is the half-diagonal of the widest box
        # among its own samples. The first and the last segment hold narrow
        # samples, and the second holds the two wide ones - which is what
        # says the grouping follows the optimizer's layout: with the uniform
        # layout every sample would be in the first segment, and the first
        # segment would carry the wide box.
        diagonal = 0.001 * (2 ** 0.5)
        wide = 0.05 * (2 ** 0.5)
        widths = [round(segment["envelope_m"], 6) for segment in segments]
        self.assertEqual(widths, [round(diagonal, 6), round(wide, 6),
                                  round(diagonal, 6)])

    def test_a_tight_threshold_applies_to_its_own_waypoint(self):
        """The failing segment is the one the tight threshold belongs to."""
        payload = self._run()
        candidate = payload["candidates"][0]
        failing = [segment["from"] for segment in candidate["segments"]
                   if not segment["ok"]]
        # The wide envelope is on the second segment's samples, and that
        # segment is the one a tight threshold on its waypoint applies to.
        # With the uniform layout those samples would be read as the first
        # segment's, and the first segment would be the one that failed.
        self.assertEqual(failing, [1])


class TestIterationHistory(unittest.TestCase):
    """--iteration-history writes how the trajectory converged."""

    def test_the_history_has_the_first_guess_and_the_solution(self):
        import overboard.main as main
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
        self.assertGreaterEqual(len(entries), 2)
        self.assertEqual(entries[0]["phase"], "initial_guess")
        self.assertEqual(entries[-1]["phase"], "final_solution")
        self.assertIn("cost", entries[0])
        self.assertTrue(entries[0]["trajectory"])
        # Between the ends are the decimated IPOPT iterates the Convergence
        # plot draws: a cost, and no trajectory (the ghost path reads the ends).
        for entry in entries[1:-1]:
            self.assertEqual(entry["phase"], "ipopt")
            self.assertIn("cost", entry)
            self.assertNotIn("trajectory", entry)

    def test_the_history_survives_parallel_candidates(self):
        """The candidates may be solved in worker processes.

        The solver's log belongs to the process that ran the solver, so it
        has to travel with the result. Without that the studio's
        convergence plot and its first-guess path are empty whenever the
        candidates run in parallel, which is the studio's default.
        """
        import overboard.main as main
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            history = os.path.join(tmp, "iterations.json")
            result = runner.invoke(main.main, [
                "-w", "examples/example_mission_first4.json", "-o", out,
                "--validate", "--candidates", "2",
                "--iteration-history", history, "-q"])
            self.assertEqual(result.exit_code, 0, result.output)
            with open(history, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
        self.assertGreater(len(entries), 2)
        self.assertEqual(entries[0]["phase"], "initial_guess")
        self.assertEqual(entries[-1]["phase"], "final_solution")
        self.assertTrue(entries[0]["trajectory"])
        self.assertIn("ipopt", [entry["phase"] for entry in entries])

class TestCandidateComparison(unittest.TestCase):
    """--candidates-out writes every candidate, and --pick-candidate chooses.

    The studio draws the three frontier paths together; the operator's pick
    is exported instead of the fastest passing candidate, and only when that
    candidate passed.
    """

    def test_the_candidate_file_counts_the_rescue(self):
        """A rescue that never pays is a cost nobody can see until it is counted."""
        import overboard.main as main
        attempts = [
            {"guess_style": "frontier_sprint", "ok": True},
            {"guess_style": "rolling", "ok": False, "rescue": True,
             "reasons": ["envelope radius 0.55 cm exceeds the local threshold"]},
        ]
        payload = main._candidate_summary(attempts, 1)
        self.assertEqual(payload["rescue"], {"tried": 1, "passed": 0})

    def test_every_candidate_is_written_with_its_samples(self):
        import overboard.main as main
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            comparison = os.path.join(tmp, "candidates.json")
            result = runner.invoke(main.main, [
                "-w", "examples/example_straight.json", "-o", out,
                "--validate", "--candidates", "2",
                "--candidates-out", comparison, "-q"])
            self.assertEqual(result.exit_code, 0, result.output)
            with open(comparison, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertEqual(len(payload["candidates"]), 2)
        # The decision score selects: the highest-scoring candidate, the
        # lower index breaking a tie.
        scores = {c["index"]: (c["score"] or 0.0) for c in payload["candidates"]}
        self.assertEqual(payload["selected"],
                         max(scores, key=lambda i: (scores[i], -i)))
        for candidate in payload["candidates"]:
            self.assertTrue(candidate["samples"], candidate["style"])
            self.assertIn(candidate["style"], ("frontier_sprint", "frontier_sweeper",
                                               "frontier_exploratory"))

    def test_a_progress_write_carries_the_path_before_the_verdict(self):
        """The reader polls this file mid-run, so it is never half written."""
        import overboard.main as main
        with tempfile.TemporaryDirectory() as tmp:
            comparison = os.path.join(tmp, "candidates.json")
            attempts = [{"guess_style": "frontier_sprint", "converged": True,
                         "samples": [{"t": 0.0, "x": 0.1, "y": 0.2}]}]
            main._write_candidates(comparison, attempts, None, quiet=True)
            with open(comparison, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertIsNone(payload["selected"])
            self.assertEqual(payload["candidates"][0]["samples"][0]["x"], 0.1)
            self.assertEqual(sorted(os.listdir(tmp)), ["candidates.json"])

    def test_a_refused_rename_is_retried(self):
        """Windows refuses the swap while the studio holds the file open."""
        import overboard.main as main
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "candidates.json")
            real_replace = os.replace
            calls = []

            def flaky(source, destination):
                calls.append(destination)
                if len(calls) < 3:
                    raise PermissionError(13, "Access is denied", destination)
                return real_replace(source, destination)

            with patch("overboard.main.os.replace", flaky):
                main._write_candidates(target, [], None, quiet=True)
            self.assertEqual(len(calls), 3)
            self.assertTrue(os.path.isfile(target))
            self.assertEqual(sorted(os.listdir(tmp)), ["candidates.json"])

    def test_a_candidate_that_failed_cannot_be_picked(self):
        """The rule is overridden by a human, never suspended."""
        import click
        import overboard.main as main
        attempts = [{"samples": [{"t": 0.0}], "stats": {}, "converged": True,
                     "ok": False, "reasons": ["envelope 4.00 cm over 2.00 cm"],
                     "tube": 0.04}]
        with self.assertRaises(click.ClickException):
            main._choose_attempt(attempts, validate=True, quiet=True, samples=10, pick=1)

    def test_a_passing_pick_is_the_one_exported(self):
        import overboard.main as main
        attempt = {"samples": [{"t": 0.0}], "stats": {}, "converged": True,
                   "ok": True, "reasons": [], "tube": 0.01}
        chosen = main._choose_attempt([attempt], validate=True, quiet=True,
                                      samples=10, pick=1)
        self.assertEqual(chosen[5], 1)
        self.assertIs(chosen[0], attempt["samples"])

    # --- The decision score selects (score contract) ----------------------

    @staticmethod
    def scored(name, score, ok=False, categories=None, duration=1.0,
               structural=False):
        return {"samples": name, "stats": {}, "converged": True, "ok": ok,
                "structural_failure": structural, "score": score,
                "categories": categories or {"accuracy": 50.0, "physical": 50.0},
                "duration": duration, "cost": 1.0, "violation": 0.0,
                "reasons": [], "tube": 0.01}

    def test_the_highest_score_selects(self):
        import overboard.main as main
        low = self.scored("low", 60.0, ok=True, duration=1.0)
        high = self.scored("high", 90.0, ok=True, duration=5.0)
        best, passed_any = main._select_attempt_scored([low, high])
        self.assertTrue(passed_any)
        self.assertEqual(best["samples"], "high")

    def test_a_score_tie_is_broken_by_the_documented_chain(self):
        import overboard.main as main
        # Same score: the ordinary pass wins before an accepted one.
        accepted = self.scored("accepted", 80.0, ok=False, duration=1.0)
        passed = self.scored("passed", 80.0, ok=True, duration=2.0)
        best, _ = main._select_attempt_scored([accepted, passed],
                                              accept_score=50)
        self.assertEqual(best["samples"], "passed")
        # Same score and verdict: the higher accuracy score wins.
        low_a = self.scored("low_a", 80.0, ok=True,
                            categories={"accuracy": 40.0, "physical": 90.0})
        high_a = self.scored("high_a", 80.0, ok=True,
                             categories={"accuracy": 90.0, "physical": 40.0})
        best, _ = main._select_attempt_scored([low_a, high_a])
        self.assertEqual(best["samples"], "high_a")
        # Same accuracy: the higher physical score wins.
        low_p = self.scored("low_p", 80.0, ok=True,
                            categories={"accuracy": 50.0, "physical": 30.0})
        high_p = self.scored("high_p", 80.0, ok=True,
                             categories={"accuracy": 50.0, "physical": 60.0})
        best, _ = main._select_attempt_scored([low_p, high_p])
        self.assertEqual(best["samples"], "high_p")
        # Then the shorter duration, then the lower index.
        slower = self.scored("slower", 80.0, ok=True, duration=4.0)
        quicker = self.scored("quicker", 80.0, ok=True, duration=2.0)
        best, _ = main._select_attempt_scored([slower, quicker])
        self.assertEqual(best["samples"], "quicker")
        first = self.scored("first", 80.0, ok=True, duration=2.0)
        second = self.scored("second", 80.0, ok=True, duration=2.0)
        best, _ = main._select_attempt_scored([first, second])
        self.assertEqual(best["samples"], "first")

    def test_an_accepted_candidate_only_counts_over_its_score(self):
        import click
        import overboard.main as main
        attempt = {"samples": [{"t": 0.0}], "stats": {}, "converged": True,
                   "ok": False, "reasons": ["envelope"], "tube": 0.04,
                   "score": 70.0, "categories": {"accuracy": 70.0,
                                                 "physical": 100.0},
                   "duration": 1.0, "structural_failure": False}
        # Under the score it may be picked, and it exports as accepted.
        chosen = main._choose_attempt([attempt], validate=True, quiet=True,
                                      samples=10, pick=1, accept_score=50)
        self.assertEqual(chosen[5], 1)
        self.assertFalse(chosen[2])
        # Over the score the pick is refused, and without --accept-score too.
        with self.assertRaises(click.ClickException):
            main._choose_attempt([attempt], validate=True, quiet=True,
                                 samples=10, pick=1, accept_score=80)
        with self.assertRaises(click.ClickException):
            main._choose_attempt([attempt], validate=True, quiet=True,
                                 samples=10, pick=1)

    def test_a_structural_failure_is_never_eligible(self):
        import overboard.main as main
        structural = self.scored("structural", 99.0, ok=True, structural=True)
        self.assertFalse(main._attempt_eligible(structural, 1))
        self.assertFalse(main._attempt_accepted(structural, 1))
        best, passed_any = main._select_attempt_scored([structural], 1)
        self.assertFalse(passed_any)

    def test_a_failed_score_beats_a_passing_one_only_on_the_score(self):
        """A failed plan accepted by its score can outscore a pass."""
        import overboard.main as main
        passing = self.scored("passing", 55.0, ok=True, duration=1.0)
        accepted = self.scored("accepted", 75.0, ok=False, duration=3.0)
        best, passed_any = main._select_attempt_scored([passing, accepted],
                                                       accept_score=50)
        self.assertEqual(best["samples"], "accepted")
        # The verdict of the chosen candidate travels back: it failed, and
        # only the accept-score makes it exportable.
        self.assertFalse(passed_any)
        # Without the accept-score the failed one is not eligible at all.
        best, _ = main._select_attempt_scored([passing, accepted])
        self.assertEqual(best["samples"], "passing")

    def test_best_effort_opens_the_gate_only_for_a_converged_plan(self):
        import overboard.main as main
        self.assertIsNone(main._export_block_reason(
            {"converged": True}, True, False, best_effort=True))
        self.assertEqual(main._export_block_reason(
            {"converged": False}, True, False, best_effort=True),
            "solver did not converge")
        # Without the flag the hard gate is unchanged.
        self.assertEqual(main._export_block_reason(
            {"converged": True}, True, False), "validation failed")

    def test_the_pick_needs_validation(self):
        import overboard.main as main
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(main.main, [
                "-w", "examples/example_straight.json", "-o", os.path.join(tmp, "out.traj"),
                "--pick-candidate", "2", "-q"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--pick-candidate needs --validate", result.output)

    def test_one_validator_serves_every_candidate(self):
        """A fresh validator per candidate pays the XLA compile again."""
        import overboard.main as main

        class FakeValidator:
            built = []

            def __init__(self, cfg, **kwargs):
                self.cfg = cfg
                self.kwargs = kwargs
                FakeValidator.built.append(self)

        class Ctx:
            validator = None
            robot_cfg = object()
            control_dt = 0.02
            propagation = "plant"
            digital_twin = None

        with patch.object(main, "_ensure_jax_classes",
                          return_value=(FakeValidator, lambda cfg: cfg)):
            first = main._immrax_validator(Ctx)
            second = main._immrax_validator(Ctx)

        self.assertIs(first, second)
        self.assertEqual(len(FakeValidator.built), 1)

    def test_a_slower_candidate_is_not_checked_after_a_faster_one_passes(self):
        """Validation is skipped when the solved duration can no longer win."""
        import overboard.main as main

        class Ctx:
            polish = False

        specs = [
            {"accuracy_weight": 0.001, "guess_style": "frontier_sprint"},
            {"accuracy_weight": 0.002, "guess_style": "frontier_sweeper"},
        ]
        solved = [
            ([{"t": duration}], {"converged": True, "final_cost": duration})
            for duration in (3.0, 1.0)
        ]
        checked = []

        def fake_dispatch(batch, ctx):
            return solved, [[], []]

        def fake_score(record, ctx):
            checked.append(record["duration"])
            record["ok"] = record["duration"] == 1.0
            return False

        with patch.object(main, "_dispatch_solve", fake_dispatch), \
                patch.object(main, "_score_attempt", fake_score):
            attempts = main._run_candidates(
                specs, Ctx(), object(), accuracy_weight=0.001,
                use_validation=True, quiet=True)

        self.assertEqual(checked, [1.0])
        self.assertTrue(attempts[1]["ok"])
        self.assertTrue(attempts[0]["validation_skipped"])

        # A compare run or an explicit pick needs every verdict.
        checked.clear()
        with patch.object(main, "_dispatch_solve", fake_dispatch), \
                patch.object(main, "_score_attempt", fake_score):
            main._run_candidates(
                specs, Ctx(), object(), accuracy_weight=0.001,
                use_validation=True, quiet=True, early_stop=False)
        self.assertEqual(checked, [3.0, 1.0])


class TestStudioEntryPoints(unittest.TestCase):
    """--gui opens the studio, and the CLI keeps no wrapper for it."""

    def setUp(self):
        import overboard.main as main
        self.main = main
        self.runner = CliRunner()

    def test_the_wrappers_that_duplicated_the_studio_are_gone(self):
        """--live, --show-convergence and --convergence-mode duplicated what
        the studio does natively, so they are not options any more."""
        for flag in ("--live", "--show-convergence", "--convergence-mode"):
            result = self.runner.invoke(
                self.main.main, ["-w", "examples/example_straight.json", flag])
            self.assertNotEqual(result.exit_code, 0, flag)
            self.assertIn("no such option", result.output.lower(), flag)

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
        self.assertTrue(options["auto_mesh"])
        self.assertFalse(options["polish"])

    def _studio_options_for(self, argv):
        seen = {}
        with patch("overboard.gui_server.run_studio",
                   side_effect=lambda **kwargs: seen.update(kwargs)):
            result = self.runner.invoke(self.main.main, argv)
        self.assertEqual(result.exit_code, 0, result.output)
        return seen["options"]

    def test_an_explicit_sample_count_reaches_the_studio(self):
        """-n has to beat the studio's automatic mesh."""
        options = self._studio_options_for(["--gui", "-n", "20"])
        self.assertEqual(options["samples"], 20)
        self.assertFalse(options["auto_mesh"])

    def test_the_studio_keeps_the_automatic_mesh_by_default(self):
        options = self._studio_options_for(["--gui"])
        self.assertTrue(options["auto_mesh"])
        self.assertFalse(options["polish"])

    def test_a_preset_is_the_same_run_in_the_studio(self):
        """A preset's own sample count reaches the solve, not the mesh."""
        options = self._studio_options_for(["--gui", "--preset", "fast"])
        self.assertEqual(options["samples"], self.main.PRESETS["fast"]["samples"])
        self.assertFalse(options["auto_mesh"])



class TestDeprecatedCliFlags(unittest.TestCase):
    """Verification of thin deprecation notices for superseded flags."""

    def setUp(self):
        import overboard.main as main
        self.main = main
        self.runner = CliRunner()

    def test_the_dead_legacy_flags_are_gone(self):
        """Purged flags are rejected as unknown options."""
        for flag in ("--simple", "--sweep", "--uniform-dt", "--per-segment-dt",
                     "--surrogate-polish", "--pyramid", "--pyramid-candidates",
                     "--h-adaptive", "--unified", "--no-unified",
                     "--self-consistent", "--no-self-consistent",
                     "--recombine-segments", "--memetic-evolution",
                     "--generations", "--population-size"):
            result = self.runner.invoke(
                self.main.main, ["-w", "examples/example_straight.json", flag])
            self.assertNotEqual(result.exit_code, 0, flag)
            self.assertIn("no such option", result.output.lower(), flag)

    def test_help_shows_active_flags(self):
        result = self.runner.invoke(self.main.main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--candidates", result.output)
        self.assertIn("--polish", result.output)
        self.assertIn("--gui", result.output)

if __name__ == "__main__":
    unittest.main()
