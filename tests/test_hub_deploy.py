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

import hub_deploy

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
        from sysid_fit import parse_session

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


class TestPushHubCli(unittest.TestCase):
    """--push-hub exports the mission and hands it to the deployer."""

    def setUp(self):
        self.runner = CliRunner()

    def _run(self, tmp, deploy, extra=None):
        import main

        out = os.path.join(tmp, "out.traj")
        wp = os.path.join(tmp, "wp.json")
        with open(wp, "w", encoding="utf-8") as handle:
            json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}], handle)

        # Keep the generated hub files out of the repository.
        mission = os.path.join(tmp, "hub_mission.py")
        runner = os.path.join(tmp, "hub_runner.py")
        with open(runner, "w", encoding="utf-8") as handle:
            handle.write("# stand-in runner\n")

        args = ["-w", wp, "-o", out, "-q", "--push-hub"] + (extra or [])
        with patch.object(main, "TrajectoryOptimizer", _RecordingOptimizer), \
                patch.object(main, "HUB_MISSION_MODULE", Path(mission)), \
                patch.object(main, "HUB_RUNNER", Path(runner)), \
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

    def test_a_plain_solve_does_not_import_the_hub_tooling(self):
        code = "import sys, main; print('pybricksdev' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


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
        from export import write_compact_python_file

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
