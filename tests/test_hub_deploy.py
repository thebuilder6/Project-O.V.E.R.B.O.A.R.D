"""
Tests for pushing a trajectory to a Pybricks hub.

The hub cannot be reached from a test, so the transport is mocked. What is
checked here is the part that runs on the host: the session file the hub
console output is written to, the push wiring in the CLI, and the contract
between the generated mission module and the runner that imports it.
"""

import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import overboard.deploy.hub_deploy as hub_deploy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYBRICKS_CODE = os.path.join(REPO_ROOT, "pybricks_code")

try:
    import mpy_cross_v6
    HAS_MPY_CROSS = True
except ImportError:
    HAS_MPY_CROSS = False


def _samples(n=11, dt=0.1, speed=0.1):
    """A straight line, like the other test helpers."""
    return [{
        "t": i * dt, "x": speed * i * dt, "y": 0.0, "heading": 0.0,
        "vl": speed, "vr": speed, "omega": 0.0,
        "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
    } for i in range(n)]


class _RecordingOptimizer:
    """Stands in for the solver."""

    def __init__(self, config):
        self.config = config

    def solve(self, waypoints, **kwargs):
        return _samples(), {"converged": True, "final_cost": 1.0}


class TestSessionFile(unittest.TestCase):
    """The captured console output is a session file, not a log dump."""

    def test_lines_are_written_verbatim(self):
        lines = ["Battery Voltage: 8100 mV", "POSE,0.1000,0.0000,0.00"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.csv")
            hub_deploy.write_session(lines, path)
            with open(path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "\n".join(lines) + "\n")

    def test_the_sysid_fitter_can_read_the_session(self):
        from overboard.sysid.fit import parse_session

        lines = [
            "TEST_START,drive",
            "META,run_id,7",
            "DATA",
            "t_ms,speed",
            "0,0",
            "10,1",
            "TEST_END",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.csv")
            hub_deploy.write_session(lines, path)
            with open(path, "r", encoding="utf-8") as handle:
                runs = parse_session(handle.read())

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["meta"]["run_id"], "7")
        self.assertEqual(runs[0]["columns"]["speed"], 1)


class _PassingImmrax:
    """A narrow envelope that passes every gate."""

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
            "reachability": {"max_error_m": 0.001, "diverged": False,
                             "envelope": [{"x_min": s["x"] - 0.001,
                                           "x_max": s["x"] + 0.001,
                                           "y_min": s["y"] - 0.001,
                                           "y_max": s["y"] + 0.001}
                                          for s in samples]},
        }


class TestPushHubCli(unittest.TestCase):
    """--push-hub exports the mission and hands it to the deployer.

    Only an ordinary validator pass or an explicitly accepted score may
    drive the hub, so these runs validate first (score contract).
    """

    def setUp(self):
        self.runner = CliRunner()

    def _run(self, tmp, deploy, extra=None, immrax=None):
        import overboard.main as main

        out = os.path.join(tmp, "out.traj")
        wp = os.path.join(tmp, "wp.json")
        with open(wp, "w", encoding="utf-8") as handle:
            json.dump({"waypoints": [{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}],
                       "scoring": {"target_time_s": 1.0, "max_time_s": 2.0}},
                      handle)

        # Keep the generated hub files out of the repository.
        mission = os.path.join(tmp, "hub_mission.py")
        runner = os.path.join(tmp, "hub_runner.py")
        with open(runner, "w", encoding="utf-8") as handle:
            handle.write("# stand-in runner\n")

        args = ["-w", wp, "-o", out, "-q", "--validate", "--push-hub"] + (extra or [])
        with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer), \
                patch.object(main, "HUB_MISSION_MODULE", Path(mission)), \
                patch.object(main, "HUB_RUNNER", Path(runner)), \
                patch.object(main, "HAS_JAX", True), \
                patch.object(main, "ImmraxValidator", immrax or _PassingImmrax), \
                patch.object(main, "JAXRobotConfig", lambda cfg: cfg), \
                patch.object(hub_deploy, "deploy", deploy):
            result = self.runner.invoke(main.main, args)
        return result, out, mission

    def test_push_writes_the_mission_and_runs_the_runner(self):
        calls = {}

        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            calls["script"] = script
            calls["connection"] = connection
            return ["POSE,0.1000,0.0000,0.00"]

        with tempfile.TemporaryDirectory() as tmp:
            result, out, mission = self._run(tmp, fake_deploy)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(mission))
            log = os.path.join(tmp, "hub.log")
            with open(log, "w", encoding="utf-8") as handle:
                handle.write("")

        self.assertTrue(calls["script"].endswith("hub_runner.py"))
        self.assertEqual(calls["connection"], "ble")

    def test_the_console_output_is_written_to_the_log(self):
        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            return ["POSE,0.1000,0.0000,0.00"]

        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "hub.log")
            result, _, _ = self._run(tmp, fake_deploy, ["--hub-log", log])
            self.assertEqual(result.exit_code, 0, result.output)
            with open(log, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "POSE,0.1000,0.0000,0.00\n")

    def test_missing_tooling_is_reported_clearly(self):
        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            raise RuntimeError(
                "Install the hub tooling with: pip install pybricksdev")

        with tempfile.TemporaryDirectory() as tmp:
            result, _, _ = self._run(tmp, fake_deploy)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("pip install pybricksdev", result.output)

    def test_push_without_validation_is_refused(self):
        """An unvalidated trajectory may not drive the hub."""
        calls = []

        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            calls.append(script)
            return []

        import overboard.main as main
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            wp = os.path.join(tmp, "wp.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}], handle)
            with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer), \
                    patch.object(hub_deploy, "deploy", fake_deploy):
                result = self.runner.invoke(
                    main.main, ["-w", wp, "-o", out, "-q", "--push-hub"])
            self.assertNotEqual(result.exit_code, 0, result.output)
            self.assertIn("--push-hub needs --validate", result.output)
        self.assertEqual(calls, [])

    def test_a_best_effort_export_never_pushes(self):
        """A best-effort path is for inspection, never for the hub."""

        class _WideImmrax(_PassingImmrax):
            def validate_trajectory(self, samples, *ranges, **kwargs):
                report = super().validate_trajectory(samples, *ranges, **kwargs)
                report["reachability"]["envelope"] = [{
                    "x_min": s["x"] - 0.05, "x_max": s["x"] + 0.05,
                    "y_min": s["y"] - 0.05, "y_max": s["y"] + 0.05,
                } for s in samples]
                return report

        calls = []

        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            calls.append(script)
            return []

        with tempfile.TemporaryDirectory() as tmp:
            result, out, _ = self._run(
                tmp, fake_deploy, ["--best-effort"], immrax=_WideImmrax)
            self.assertNotEqual(result.exit_code, 0, result.output)
            self.assertIn("not approved", result.output)
            self.assertTrue(os.path.exists(out))
        self.assertEqual(calls, [])

    def test_a_plain_solve_does_not_import_the_hub_tooling(self):
        code = "import sys, overboard.main; print('pybricksdev' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    # --- push_trajectory's own approval check (score contract) ------------

    def _doc(self, validation):
        samples = _samples()
        return {
            "name": "out",
            "version": 3,
            "trajectory": {"config": {"mass": 1.0}, "samples": samples},
            "validation": validation,
        }

    def _approved(self, **overrides):
        doc = self._doc(None)
        record = {
            "passed": True, "accepted": False, "best_effort": False,
            "hub_approved": True, "score": 91.0, "score_version": 1,
            "trajectory_sha256": hub_deploy.trajectory_sha256(
                doc["trajectory"]["config"], doc["trajectory"]["samples"]),
        }
        record.update(overrides)
        doc["validation"] = record
        return doc

    def _write(self, tmp, doc):
        path = os.path.join(tmp, "out.traj")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
        return path

    def _refused(self, doc, needle):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, doc)
            mission = os.path.join(tmp, "hub_mission.py")
            with patch.object(hub_deploy, "deploy",
                              lambda *a, **k: self.fail("deployed an unapproved trajectory")):
                with self.assertRaises(ValueError) as caught:
                    hub_deploy.push_trajectory(path, mission_module=Path(mission))
            self.assertIn(needle, str(caught.exception))
            # The refusal happens before the mission is written.
            self.assertFalse(os.path.exists(mission))

    def test_an_unvalidated_file_is_refused(self):
        self._refused(self._doc(None), "no validation record")

    def test_a_best_effort_file_is_refused(self):
        self._refused(
            self._approved(best_effort=True, passed=False, hub_approved=False),
            "not approved")

    def test_an_unapproved_verdict_is_refused(self):
        self._refused(self._approved(hub_approved=False), "not approved")

    def test_an_unsupported_score_version_is_refused(self):
        self._refused(self._approved(score_version=2), "score_version")

    def test_a_stale_hash_is_refused(self):
        doc = self._approved()
        doc["trajectory"]["samples"][0]["x"] += 0.01
        self._refused(doc, "hash")

    def test_an_edited_record_is_refused(self):
        doc = self._approved()
        doc["validation"]["trajectory_sha256"] = "0" * 64
        self._refused(doc, "hash")

    def test_an_approved_trajectory_is_pushed(self):
        calls = []

        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            calls.append(script)
            return ["POSE,0.1000,0.0000,0.00"]

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, self._approved())
            mission = os.path.join(tmp, "hub_mission.py")
            with patch.object(hub_deploy, "deploy", fake_deploy):
                lines = hub_deploy.push_trajectory(path, mission_module=Path(mission))
            self.assertTrue(os.path.exists(mission))
        self.assertEqual(len(calls), 1)
        self.assertEqual(lines, ["POSE,0.1000,0.0000,0.00"])

    def test_an_accepted_failure_is_approved_too(self):
        """A failed plan accepted by --accept-score may drive the hub."""
        calls = []

        def fake_deploy(script, connection="ble", name=None, wait=True,
                       print_output=True, stream_path=None):
            calls.append(script)
            return []

        doc = self._approved(passed=False, accepted=True, score=62.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, doc)
            mission = os.path.join(tmp, "hub_mission.py")
            with patch.object(hub_deploy, "deploy", fake_deploy):
                hub_deploy.push_trajectory(path, mission_module=Path(mission))
        self.assertEqual(len(calls), 1)

    def test_the_hash_is_canonical_and_shared(self):
        """The writer and the pusher compute the same hash."""
        doc = self._doc({"status": "unvalidated", "passed": False})
        config, samples = doc["trajectory"]["config"], doc["trajectory"]["samples"]
        # Key order and whitespace must not matter.
        self.assertEqual(
            hub_deploy.trajectory_sha256(config, samples),
            hub_deploy.trajectory_sha256(dict(reversed(list(config.items()))),
                                         list(samples)))
        self.assertEqual(
            hub_deploy.trajectory_sha256(config, samples),
            hub_deploy.trajectory_sha256(json.loads(json.dumps(config)),
                                         json.loads(json.dumps(samples))))
        self.assertNotEqual(
            hub_deploy.trajectory_sha256(config, samples),
            hub_deploy.trajectory_sha256(config, samples[:-1]))


@unittest.skipUnless(HAS_MPY_CROSS, "mpy-cross is not installed")
class TestFirmwareCompiles(unittest.TestCase):
    """The firmware has to compile for MPY, or it cannot be pushed to a hub.

    MicroPython cannot concatenate two f-strings, and that only shows up
    here: the file is valid Python, so the host tools and the tests never
    notice.
    """

    def test_every_firmware_module_compiles(self):
        for path in sorted(glob.glob(os.path.join(PYBRICKS_CODE, "*.py"))):
            name = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            proc, _ = mpy_cross_v6.mpy_cross_compile(name, source)
            self.assertEqual(
                proc.returncode, 0,
                f"{name} does not compile for MPY: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}",
            )


class TestHubRunnerContract(unittest.TestCase):
    """The runner and the generated mission module must fit together."""

    def test_runner_parses_and_its_local_imports_exist(self):
        path = os.path.join(PYBRICKS_CODE, "hub_runner.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        compile(source, path, "exec")
        for name in ("robot_config", "ramsete_compact_events"):
            self.assertTrue(
                os.path.exists(os.path.join(PYBRICKS_CODE, name + ".py")), name)

    def test_generated_mission_module_has_what_the_runner_imports(self):
        from overboard.deploy.export import write_compact_python_file

        config = {
            "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
            "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
            "gearing": 1.0, "cof": 1.37,
        }
        with tempfile.TemporaryDirectory() as tmp:
            traj = os.path.join(tmp, "out.traj")
            with open(traj, "w", encoding="utf-8") as handle:
                json.dump({"trajectory": {"config": config,
                                          "samples": _samples()}}, handle)
            module = os.path.join(tmp, "hub_mission.py")
            write_compact_python_file(traj, module)

            with open(module, "r", encoding="utf-8") as handle:
                namespace = {}
                exec(compile(handle.read(), module, "exec"), namespace)

        self.assertIn("samples", namespace)
        self.assertIn("config", namespace)
        self.assertIn("events", namespace)


if __name__ == "__main__":
    unittest.main()
