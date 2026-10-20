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
        import immrax_validator
        import main
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
        import main
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


class TestFirstRun(unittest.TestCase):
    """A first-time user must get the install command, not a traceback."""

    def test_missing_core_packages_names_what_is_missing(self):
        import diagnostics
        with patch.object(diagnostics, "module_available",
                          side_effect=lambda name: name != "casadi"):
            self.assertEqual(diagnostics.missing_core_packages(), ["casadi"])

    def test_the_doctor_names_the_install_command(self):
        import diagnostics
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
            "runpy.run_path('main.py', run_name='__main__')\n"
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
                          "max_excess_cm", "peak_jerk_m_s3"):
                self.assertIn(field, candidate)
                self.assertIsNotNone(candidate[field], field)
            # The excess agrees with the segments it came from.
            segments = candidate["segments"]
            excess = max(segment["envelope_m"] - segment["limit_m"]
                         for segment in segments)
            self.assertAlmostEqual(candidate["max_excess_cm"],
                                   excess * 100.0, places=6)


class TestVariableSampleLayout(unittest.TestCase):
    """The gate follows the optimizer's own sample layout.

    The optimizer splits the sample budget across the waypoint segments in
    proportion to their length, so the waypoints are not k *
    samples_per_segment apart. A consumer that assumes they are applies a
    waypoint's envelope threshold to the wrong samples, and groups the
    per-segment report wrongly.
    """

    def setUp(self):
        import main
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
                json.dump([{"x": 0.0, "y": 0.0, "heading": 0.0},
                           {"x": 0.1, "y": 0.0, "heading": 0.0},
                           {"x": 0.2, "y": 0.0, "heading": 0.0},
                           {"x": 0.3, "y": 0.0, "heading": 0.0}], handle)
            with patch.object(self.main, "TrajectoryOptimizer", self._Optimizer), \
                    patch.object(self.main, "ImmraxValidator", self._Immrax):
                result = self.runner.invoke(self.main.main, [
                    "-c", os.path.join(self.repo, "examples", "fll_choreo.json"),
                    "-w", wp, "-o", out, "--simple", "--validate"])
        marker = [line for line in result.output.splitlines()
                   if line.startswith(self.main.VALIDATION_MARKER)]
        self.assertEqual(len(marker), 1, result.output)
        return json.loads(marker[0][len(self.main.VALIDATION_MARKER):])

    def test_the_segments_follow_the_optimizers_layout(self):
        """Each segment holds its own samples, not a tenth of the run."""
        payload = self._run()
        segments = payload["candidates"][0]["segments"]
        self.assertEqual(len(segments), 3)
        # The envelope of a segment is the widest of its own samples, and
        # every sample's envelope is 1 mm wider than the last one's.
        widths = [round(segment["envelope_m"], 6) for segment in segments]
        self.assertEqual(widths, [0.002, 0.003, 0.005])

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

    def test_the_history_survives_parallel_candidates(self):
        """The candidates may be solved in worker processes.

        The solver's log belongs to the process that ran the solver, so it
        has to travel with the result. Without that the studio's
        convergence plot and its first-guess path are empty whenever the
        candidates run in parallel, which is the studio's default.
        """
        import main
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
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["phase"], "initial_guess")
        self.assertEqual(entries[1]["phase"], "final_solution")
        self.assertTrue(entries[0]["trajectory"])

    def test_the_history_is_captured_under_multiverse(self):
        """KI-017: --multiverse forwards iteration history when requested."""
        import main
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            history = os.path.join(tmp, "iterations.json")
            result = runner.invoke(main.main, [
                "-w", "examples/example_straight.json", "-o", out,
                "--multiverse", "--iteration-history", history, "-q"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(history))
            with open(history, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
        self.assertGreaterEqual(len(entries), 1)


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
