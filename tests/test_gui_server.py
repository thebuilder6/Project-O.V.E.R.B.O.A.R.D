"""
Tests for the browser studio server.

The studio must stay a thin front end: it serves the UI, runs the command line
in a subprocess and never writes an artifact the CLI did not write.
"""

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import gui_server

REPO_ROOT = Path(__file__).resolve().parent.parent


def _request(url, method="GET", payload=None, headers=None):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        if response.headers.get("Content-Type", "").startswith("application/json"):
            return response.status, json.loads(body.decode("utf-8"))
        return response.status, body


class StudioTestCase(unittest.TestCase):
    """Starts one studio server for the whole test case."""

    def setUp(self):
        self.server = gui_server.create_server(
            0, {"fast": {"samples": 4}}, {"config": None, "waypoints": None})
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()


class TestStudioRoutes(StudioTestCase):
    def test_index_page_is_served(self):
        status, body = _request(f"{self.base}/")
        self.assertEqual(status, 200)
        self.assertIn(b"OVERBOARD", body)

    def test_static_assets_are_served(self):
        for path in ("/studio.css", "/studio.js"):
            status, body = _request(f"{self.base}{path}")
            self.assertEqual(status, 200, path)
            self.assertTrue(body)

    def test_unknown_path_is_not_found(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_traversal_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/jobs/../../main.py")
        self.assertIn(ctx.exception.code, (400, 404))

    def test_foreign_host_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/health", headers={"Host": "evil.example.com"})
        self.assertEqual(ctx.exception.code, 403)

    def test_health_reports_the_optional_packages(self):
        status, payload = _request(f"{self.base}/api/health")
        self.assertEqual(status, 200)
        self.assertIn("packages", payload)
        self.assertIn("presets", payload)

    def test_examples_list_configs_and_missions(self):
        status, payload = _request(f"{self.base}/api/examples")
        self.assertEqual(status, 200)
        names = [config["name"] for config in payload["configs"]]
        self.assertIn("fll_choreo", names)
        missions = [mission["name"] for mission in payload["missions"]]
        self.assertIn("example_straight", missions)


class TestStudioSolve(StudioTestCase):
    def _config(self):
        with open(REPO_ROOT / "examples" / "fll_choreo.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _wait(self, job_id, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, payload = _request(f"{self.base}/api/jobs/{job_id}")
            if payload["status"] != "running":
                return payload
            time.sleep(0.3)
        self.fail("the solve did not finish in time")

    def test_solve_writes_a_trajectory_and_reports_it(self):
        payload = {
            "name": "unit test",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0, "heading": 0.0},
                          {"x": 0.3, "y": 0.0, "heading": 0.0}],
            "options": {"samples": 4, "candidates": 1, "accuracy_weight": 0.0,
                        "timing": "auto", "validate": False, "export_format": "none"},
        }
        status, job = _request(f"{self.base}/api/solve", method="POST", payload=payload)
        self.assertEqual(status, 202)
        self.assertEqual(job["status"], "running")

        done = self._wait(job["id"])
        self.assertEqual(done["status"], "ok", done["log"][-5:])
        self.assertEqual(done["exit_code"], 0)
        self.assertIsNotNone(done["summary"])
        self.assertGreater(done["summary"]["duration_s"], 0.0)
        kinds = [artifact["kind"] for artifact in done["artifacts"]]
        self.assertIn("traj", kinds)

        status, body = _request(f"{self.base}/api/jobs/{job['id']}/trajectory")
        self.assertEqual(status, 200)
        self.assertIn("samples", body["trajectory"])

    def test_second_solve_is_refused_while_one_runs(self):
        payload = {
            "name": "busy",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 0.3, "y": 0.0}],
            "options": {"samples": 4, "candidates": 1},
        }
        _request(f"{self.base}/api/solve", method="POST", payload=payload)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/solve", method="POST", payload=payload)
        self.assertEqual(ctx.exception.code, 409)

    def test_a_mission_with_one_waypoint_is_rejected(self):
        payload = {
            "name": "too short",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0}],
            "options": {},
        }
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/solve", method="POST", payload=payload)
        self.assertEqual(ctx.exception.code, 400)

    def test_a_failed_solve_writes_no_artifact(self):
        payload = {
            "name": "bad",
            "config": self._config(),
            # A waypoint the solver cannot reach makes the CLI exit non-zero.
            "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 40.0, "y": 40.0}],
            "options": {"samples": 4, "candidates": 1},
        }
        status, job = _request(f"{self.base}/api/solve", method="POST", payload=payload)
        done = self._wait(job["id"])
        if done["status"] == "ok":
            self.skipTest("the solver accepted the far waypoint")
        self.assertNotEqual(done["exit_code"], 0)
        self.assertEqual(done["artifacts"], [])


class TestStudioHelpers(unittest.TestCase):
    def test_slugify_keeps_file_names_safe(self):
        self.assertEqual(gui_server.slugify("My Mission!"), "my-mission")
        self.assertEqual(gui_server.slugify(""), "mission")
        self.assertEqual(gui_server.slugify("../../etc/passwd"), "etc-passwd")

    def test_command_contains_the_expected_flags(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 8, "accuracy_weight": 0.5, "candidates": 3,
             "timing": "per-segment", "validate": True,
             "export_format": "controller", "controller_dt": 0.02})
        joined = " ".join(str(part) for part in command)
        self.assertIn("-n 8", joined)
        self.assertIn("-a 0.5", joined)
        self.assertIn("--candidates 3", joined)
        self.assertIn("--per-segment-dt", joined)
        self.assertIn("--validate", joined)
        self.assertIn("--export-format controller", joined)
        self.assertIn("--controller-dt 0.02", joined)

    def test_command_does_not_force_a_timing_mode_by_default(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 10, "timing": "auto"})
        joined = " ".join(str(part) for part in command)
        self.assertNotIn("--uniform-dt", joined)
        self.assertNotIn("--per-segment-dt", joined)

    def test_summary_reads_the_written_trajectory(self):
        samples = [{
            "t": 0.1 * i, "x": 0.1 * i, "y": 0.0, "heading": 0.0,
            "vl": 0.4, "vr": 0.4, "omega": 0.0,
            "reachability_envelope": {"x_min": -0.01, "x_max": 0.01,
                                      "y_min": -0.01, "y_max": 0.01},
        } for i in range(11)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.traj"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"trajectory": {"samples": samples}}, handle)
            summary = gui_server.summarize_trajectory(path)
        self.assertAlmostEqual(summary["duration_s"], 1.0)
        self.assertAlmostEqual(summary["path_length_m"], 1.0)
        self.assertAlmostEqual(summary["peak_wheel_speed_m_s"], 0.4)
        self.assertTrue(summary["has_envelope"])

    def test_saving_a_mission_writes_to_the_missions_folder(self):
        missions = gui_server.MISSIONS_DIR
        existed = missions.is_dir()
        try:
            server = gui_server.create_server(0, {}, {})
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            status, payload = _request(f"{base}/api/missions", method="POST", payload={
                "name": "unit test mission",
                "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
            })
            self.assertEqual(status, 200)
            self.assertEqual(payload["name"], "unit-test-mission")
            self.assertTrue((missions / "unit-test-mission.json").is_file())
            server.shutdown()
            server.server_close()
        finally:
            saved = missions / "unit-test-mission.json"
            if saved.is_file():
                saved.unlink()
            if not existed and missions.is_dir() and not any(missions.iterdir()):
                missions.rmdir()


if __name__ == "__main__":
    unittest.main()
