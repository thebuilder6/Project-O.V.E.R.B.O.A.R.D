"""
Tests for the browser studio server.

The studio must stay a thin front end: it serves the UI, runs the command line
in a subprocess and never writes an artifact the CLI did not write.
"""

import json
import math
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import overboard.gui_server as gui_server

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
        for path in ("/studio.css", "/studio.js", "/report.js"):
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

    def test_initial_payload_carries_the_auto_solve_options(self):
        initial = gui_server.build_initial(
            config_path=None,
            waypoints_path=str(REPO_ROOT / "examples" / "example_straight.json"),
            auto_solve=True,
            options={"samples": 14, "validate": True, "timing": "per-segment"},
        )
        self.assertTrue(initial["auto_solve"])
        self.assertEqual(initial["options"]["samples"], 14)
        self.assertTrue(initial["options"]["validate"])
        self.assertEqual(len(initial["waypoints"]), 2)

    def test_a_fitted_plant_turns_validation_on(self):
        initial = gui_server.build_initial(
            options={"validate": False, "sysid_fit": "fit.json"})
        self.assertTrue(initial["options"]["validate"])

    def test_current_job_is_null_when_idle(self):
        status, payload = _request(f"{self.base}/api/jobs/current")
        self.assertEqual(status, 200)
        self.assertIsNone(payload["job"])


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
        self.assertIsNotNone(done["wall_s"])
        self.assertGreater(done["wall_s"], 0.0)
        kinds = [artifact["kind"] for artifact in done["artifacts"]]
        self.assertIn("traj", kinds)

        # The plots need the limits and the way the solve converged.
        limits = done["result"]["limits"]
        self.assertIn("speed_limit_m_s", limits)
        self.assertGreater(len(limits["force_limit_n"]), 0)
        iterations = done["result"]["iterations"]
        self.assertEqual(len(iterations), 2)
        self.assertEqual(iterations[0]["phase"], "initial_guess")
        # Every candidate was written with its samples, so the field can draw
        # the comparison (one candidate for this run).
        candidates = done["result"]["candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["samples"])

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

    def test_two_solves_cannot_start_at_the_same_time(self):
        """The check and the registration are one step.

        Two requests that arrive together used to both pass the "is
        something running" check, and two solves ran at once.
        """
        payload = {
            "name": "race",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 0.3, "y": 0.0}],
            "options": {"samples": 4, "candidates": 1},
        }
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def ask():
            barrier.wait()
            try:
                status, _ = _request(f"{self.base}/api/solve", method="POST",
                                     payload=payload)
                code = status
            except urllib.error.HTTPError as exc:
                code = exc.code
            with lock:
                results.append(code)

        threads = [threading.Thread(target=ask) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        self.assertEqual(len(results), 4)
        self.assertEqual(results.count(202), 1,
                         f"more than one solve started: {results}")
        self.assertEqual(results.count(409), 3, f"the rest were not refused: {results}")

    def test_a_failed_post_processing_does_not_block_the_studio(self):
        """The job must leave "running" whatever the post-processing does.

        A job that stays "running" is the studio's active job, so every
        later solve, push and collection is refused with 409 and the studio
        cannot be used until the server is restarted.
        """
        class Boom(BaseException):
            """Not an Exception, so the guards do not catch it."""

        payload = {
            "name": "broken result",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 0.3, "y": 0.0}],
            "options": {"samples": 4, "candidates": 1},
        }
        # The worker thread's traceback is expected here; keep it out of the
        # test output.
        quiet = threading.excepthook
        threading.excepthook = lambda args: None
        try:
            with patch.object(gui_server, "trajectory_limits", side_effect=Boom()):
                status, job = _request(f"{self.base}/api/solve", method="POST",
                                       payload=payload)
                self.assertEqual(status, 202)
                done = self._wait(job["id"])
        finally:
            threading.excepthook = quiet
        self.assertNotEqual(done["status"], "running")
        # The studio still takes work.
        status, _ = _request(f"{self.base}/api/solve", method="POST",
                             payload=payload)
        self.assertEqual(status, 202)

    def test_a_running_job_is_reported_as_current(self):
        payload = {
            "name": "current",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 0.3, "y": 0.0}],
            "options": {"samples": 4, "candidates": 1},
        }
        _request(f"{self.base}/api/solve", method="POST", payload=payload)
        status, body = _request(f"{self.base}/api/jobs/current")
        self.assertEqual(status, 200)
        self.assertIsNotNone(body["job"])
        self.assertEqual(body["job"]["status"], "running")


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


class TestStudioPush(StudioTestCase):
    """The push runs an approved trajectory on the hub."""

    def _config(self):
        with open(REPO_ROOT / "examples" / "fll_choreo.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.fake_deploy = os.path.join(self.tmp, "fake_hub_deploy.py")
        with open(self.fake_deploy, "w", encoding="utf-8") as handle:
            handle.write(
                "import sys\n"
                "out = sys.argv[sys.argv.index('--out') + 1]\n"
                "with open(out, 'w', encoding='utf-8') as handle:\n"
                "    handle.write('Hub runner: following the pushed trajectory...\\n')\n"
                "    handle.write('EVENT,lower_arm\\n')\n"
                "    handle.write('POSE,0.0900,0.0000,1.50\\n')\n"
                "print('Hub runner: following the pushed trajectory...')\n"
                "print('POSE,0.0900,0.0000,1.50')\n"
            )
        self._patch = patch.object(gui_server, "HUB_DEPLOY", Path(self.fake_deploy))
        self._patch.start()
        self.addCleanup(self._patch.stop)
        # The push writes the hub session next to the mission. That is the
        # operator's own missions folder in a real run, so the test must not
        # leave a session in it (and a test that writes in the repository is
        # never skipped by the cache).
        self.missions = Path(self.tmp) / "missions"
        self.missions.mkdir()
        self._missions_patch = patch.object(gui_server, "MISSIONS_DIR", self.missions)
        self._missions_patch.start()
        self.addCleanup(self._missions_patch.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _solve(self, validated=True):
        payload = {
            "name": "pushable",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": 0.0}],
            "options": {"samples": 4, "candidates": 1, "iteration_history": True},
        }
        status, job = _request(f"{self.base}/api/solve", method="POST", payload=payload)
        self.assertEqual(status, 202)
        deadline = time.time() + 120
        while time.time() < deadline:
            status, body = _request(f"{self.base}/api/jobs/{job['id']}")
            if body["status"] != "running":
                if validated:
                    # The studio only pushes a validated trajectory, and the
                    # verdict is on the solve's own output.
                    self.server.get_job(job["id"]).log(
                        gui_server.VALIDATION_MARKER
                        + json.dumps({"selected": 1,
                                      "candidates": [{"ok": True}]}))
                return job, body
            time.sleep(0.3)
        self.fail("the solve did not finish in time")

    def test_an_unvalidated_trajectory_is_not_pushable(self):
        """The rule is the server's, not only the browser's.

        A page on this machine can post to the studio, so the endpoint
        that drives the robot has to check the verdict itself (KI-036).
        """
        solve_job, solved = self._solve(validated=False)
        self.assertEqual(solved["status"], "ok")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/push", method="POST", payload={
                "job_id": solve_job["id"], "connection": "ble", "hub_name": "HUB"})
        self.assertEqual(ctx.exception.code, 400)

    def test_pushing_an_unknown_job_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/push", method="POST",
                      payload={"job_id": "nope", "connection": "ble"})
        self.assertEqual(ctx.exception.code, 400)

    def test_pushing_without_a_trajectory_is_refused(self):
        # A solve that writes nothing must not be pushable.
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/push", method="POST",
                      payload={"job_id": "nope", "connection": "ble"})
        self.assertEqual(ctx.exception.code, 400)

    def test_a_push_reports_the_measured_pose_and_the_events(self):
        solve_job, solved = self._solve()
        self.assertEqual(solved["status"], "ok")
        kinds = [artifact["kind"] for artifact in solved["artifacts"]]
        self.assertIn("traj", kinds)

        status, job = _request(f"{self.base}/api/push", method="POST", payload={
            "job_id": solve_job["id"], "connection": "ble", "hub_name": "HUB"})
        self.assertEqual(status, 202)
        self.assertEqual(job["kind"], "push")

        deadline = time.time() + 120
        while time.time() < deadline:
            status, body = _request(f"{self.base}/api/jobs/{job['id']}")
            if body["status"] != "running":
                break
            time.sleep(0.3)
        self.assertEqual(body["status"], "ok", body["log"][-5:])
        self.assertEqual(body["result"]["connection"], "ble")
        self.assertEqual(body["result"]["poses"],
                         [{"x_m": 0.09, "y_m": 0.0, "heading_deg": 1.5}])
        self.assertEqual(body["result"]["events"], ["lower_arm"])
        # The comparison: the robot stopped 10 mm short of the 0.1 m plan.
        report = body["result"]["report"]
        self.assertIsNotNone(report)
        self.assertEqual(len(report["rows"]), 1)
        row = report["rows"][0]
        self.assertAlmostEqual(row["position_error_m"], 0.01, places=6)
        self.assertAlmostEqual(row["measured"]["x"], 0.09, places=6)
        self.assertAlmostEqual(row["predicted"]["x"], 0.1, places=6)
        kinds = [artifact["kind"] for artifact in body["artifacts"]]
        self.assertIn("session", kinds)

    def test_the_comparison_puts_both_poses_in_the_same_frame(self):
        """The hub's pose is in the robot's frame; the plan is in the field's."""
        with tempfile.TemporaryDirectory() as tmp:
            traj = os.path.join(tmp, "plan.traj")
            session = os.path.join(tmp, "session.csv")
            # The plan starts at (1, 2) pointing along +y.
            samples = [{"t": 0.0, "x": 1.0, "y": 2.0, "heading": math.pi / 2,
                        "vl": 0.0, "vr": 0.0, "omega": 0.0}]
            with open(traj, "w", encoding="utf-8") as handle:
                json.dump({"trajectory": {"samples": samples}}, handle)
            with open(session, "w", encoding="utf-8") as handle:
                handle.write("POSE,1.0000,0.0000,0.00\n")

            report = gui_server.compare_with_plan(traj, session)
        self.assertIsNotNone(report)
        row = report["rows"][0]
        # One metre along the robot's own axis is one metre along +y here.
        self.assertAlmostEqual(row["measured"]["x"], 1.0, places=6)
        self.assertAlmostEqual(row["measured"]["y"], 3.0, places=6)

    def test_the_session_lands_next_to_the_mission(self):
        solve_job, _ = self._solve()
        target = gui_server.MISSIONS_DIR / "pushable.session.csv"
        target.write_text("POSE,0.0000,0.0000,0.00\n", encoding="utf-8")
        status, job = _request(f"{self.base}/api/push", method="POST", payload={
            "job_id": solve_job["id"], "connection": "ble"})
        deadline = time.time() + 120
        while time.time() < deadline:
            status, body = _request(f"{self.base}/api/jobs/{job['id']}")
            if body["status"] != "running":
                break
            time.sleep(0.3)
        try:
            self.assertTrue(target.is_file())
            self.assertIn("POSE,0.0900", target.read_text(encoding="utf-8"))
            archived = [path for path in
                        gui_server.MISSIONS_DIR.glob("pushable*session.csv")
                        if path != target]
            self.assertEqual(len(archived), 1)
            self.assertIn("POSE,0.0000", archived[0].read_text(encoding="utf-8"))
        finally:
            for path in gui_server.MISSIONS_DIR.glob("pushable*session.csv"):
                path.unlink()

class TestStudioHubScan(StudioTestCase):
    """The Run tab's device picker: what a Bluetooth scan reports."""

    class _Device:
        def __init__(self, name, address):
            self.name = name
            self.address = address

    class _Advertisement:
        def __init__(self, local_name="", uuids=()):
            self.local_name = local_name
            self.service_uuids = list(uuids)

    def _patch_scan(self, rows):
        from overboard.deploy import hub_ble

        async def fake(seconds=0, timeout=0, limit=4):
            for row in rows:
                yield row

        patcher = patch.object(hub_ble, "list_hubs", fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_scan_names_the_hubs_and_counts_what_it_saw(self):
        named = self._Device("Technic Hub", "11:22:33:44:55:66")
        unnamed = self._Device("", "AA:BB:CC:DD:EE:FF")
        phone = self._Device("Phone", "99:88:77:66:55:44")
        self._patch_scan([
            (named, self._Advertisement("Technic Hub"), True),
            (unnamed, self._Advertisement(), True),
            (phone, self._Advertisement("Phone"), False),
        ])
        status, body = _request(f"{self.base}/api/hubs", method="POST")
        self.assertEqual(status, 200)
        self.assertEqual(body["seen"], 3)
        # Only the hubs are offered, and an unnamed one is named by its
        # address, which is what --name matches on.
        self.assertEqual([hub["name"] for hub in body["hubs"]],
                         ["Technic Hub", "AA:BB:CC:DD:EE:FF"])
        self.assertIn("Technic Hub (11:22:33:44:55:66)", body["hubs"][0]["label"])

    def test_a_scan_is_refused_while_a_job_runs(self):
        job = gui_server.Job("jobscan", "solve", Path(tempfile.mkdtemp()))
        self.addCleanup(shutil.rmtree, str(job.output_dir), True)
        self.server.jobs[job.id] = job
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/hubs", method="POST")
        self.assertEqual(ctx.exception.code, 409)


class TestStudioReport(StudioTestCase):
    """The report the browser built is written and served back."""

    def _config(self):
        with open(REPO_ROOT / "examples" / "fll_choreo.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _solve(self):
        payload = {
            "name": "unit test report",
            "config": self._config(),
            "waypoints": [{"x": 0.0, "y": 0.0, "heading": 0.0},
                          {"x": 0.3, "y": 0.0, "heading": 0.0}],
            "options": {"samples": 4, "candidates": 1, "accuracy_weight": 0.0,
                        "timing": "auto", "validate": False, "export_format": "none"},
        }
        status, job = _request(f"{self.base}/api/solve", method="POST", payload=payload)
        self.assertEqual(status, 202)
        deadline = time.time() + 180
        while time.time() < deadline:
            status, done = _request(f"{self.base}/api/jobs/{job['id']}")
            if done["status"] != "running":
                self.assertEqual(done["status"], "ok", done["log"][-5:])
                return job["id"]
            time.sleep(0.3)
        self.fail("the solve did not finish in time")

    def test_the_report_is_written_and_served(self):
        job_id = self._solve()
        html = "<!DOCTYPE html><html><body><h1>report</h1></body></html>"
        status, result = _request(f"{self.base}/api/jobs/{job_id}/report",
                                   method="POST", payload={"html": html})
        self.assertEqual(status, 200)
        self.assertTrue(result["name"].endswith("_report.html"))
        self.assertEqual(result["url"], f"/api/jobs/{job_id}/report")

        status, body = _request(f"{self.base}/api/jobs/{job_id}/report")
        self.assertEqual(status, 200)
        self.assertIn(b"report", body)

        # It is an artifact of the solve, so it is listed with the others.
        status, done = _request(f"{self.base}/api/jobs/{job_id}")
        kinds = [artifact["kind"] for artifact in done["artifacts"]]
        self.assertIn("report", kinds)

    def test_an_empty_report_is_refused(self):
        job_id = self._solve()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/jobs/{job_id}/report", method="POST",
                     payload={"html": "  "})
        self.assertEqual(ctx.exception.code, 400)

    def test_a_report_for_an_unknown_job_is_not_found(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/jobs/nope/report", method="POST",
                     payload={"html": "<html></html>"})
        self.assertEqual(ctx.exception.code, 404)


class TestStudioModel(StudioTestCase):
    """The measured model belongs to the robot and is used when planning."""

    def setUp(self):
        super().setUp()
        # The measured models are written next to the robot; the tests keep
        # them out of the repository.
        self.calibration = tempfile.mkdtemp()
        self._patch = patch.object(gui_server, "CALIBRATION_DIR", Path(self.calibration))
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(shutil.rmtree, self.calibration, True)

    def _fit(self):
        return {
            "wheel_radius_m": 0.028,
            "actuator": {
                "straight": {"calibrated": True, "n_runs": 3, "gain": 0.981,
                             "gain_between_runs": {"n": 3, "ci95": [0.971, 0.991]},
                             "gain_ci95": [0.971, 0.991], "tau_s": 0.081, "delay_s": 0.021},
                "turn": {"calibrated": True, "n_runs": 3, "gain": 0.976,
                         "gain_ci95": [0.966, 0.986], "tau_s": 0.079, "delay_s": 0.018},
            },
            "sensors": {"calibrated": True, "gyro_z_bias_deg_s": 0.0041},
            "odometry": {"calibrated": True, "distance_scale": 0.998,
                         "heading_scale": 1.0},
            "slip": {"calibrated": True, "brake_ratio_median": 0.011,
                     "brake_ratio_min": 0.008, "brake_ratio_max": 0.013},
        }

    def test_a_fit_is_stored_with_the_robot_and_summarised(self):
        status, model = _request(f"{self.base}/api/robot/model/store", method="POST",
                                  payload={"name": "my robot", "fit": self._fit()})
        self.assertEqual(status, 200)
        self.assertTrue(model["path"].endswith("my-robot.sysid_fit.json"))
        self.assertAlmostEqual(model["straight"]["gain"], 0.981)
        self.assertEqual(model["straight"]["gain_ci95"], [0.971, 0.991])
        self.assertAlmostEqual(model["odometry"]["distance_scale"], 0.998)
        self.assertAlmostEqual(model["slip"]["brake_ratio_median"], 0.011)
        self.assertTrue(gui_server.CALIBRATION_DIR.is_dir())

    def test_the_stored_models_are_listed(self):
        _request(f"{self.base}/api/robot/model/store", method="POST",
                 payload={"name": "listed robot", "fit": self._fit()})
        status, payload = _request(f"{self.base}/api/robot/models")
        self.assertEqual(status, 200)
        names = [model["name"] for model in payload["models"]]
        self.assertIn("listed-robot", names)

    def test_a_fit_is_loaded_from_a_path(self):
        # The route only reads a file under the project folder or the
        # calibration folder (KI-036), so the fit is written into the
        # calibration folder the test already redirects.
        path = str(gui_server.calibration_paths("loaded")["fit"])
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._fit(), handle)
        status, model = _request(f"{self.base}/api/robot/model/load", method="POST",
                                  payload={"path": path})
        self.assertEqual(status, 200)
        self.assertAlmostEqual(model["straight"]["gain"], 0.981)

    def test_a_fit_outside_the_project_folder_is_refused(self):
        """Any page on this machine must not be able to read a file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sysid_fit.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._fit(), handle)
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                _request(f"{self.base}/api/robot/model/load", method="POST",
                         payload={"path": path})
            self.assertEqual(ctx.exception.code, 400)

    def test_a_file_that_is_not_a_fit_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/robot/model/store", method="POST",
                     payload={"name": "nonsense", "fit": {"hello": "world"}})
        self.assertEqual(ctx.exception.code, 400)

    def test_a_solve_with_a_model_turns_validation_on(self):
        path = str(gui_server.calibration_paths("fitted robot")["fit"])
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._fit(), handle)
        status, model = _request(f"{self.base}/api/robot/model/load", method="POST",
                                 payload={"path": path, "name": "fitted robot"})
        self.assertEqual(status, 200)

        payload = {
            "name": "fitted",
            "config": {
                "robot": {"mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
                          "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
                          "gearing": 1.0, "cof": 1.37}},
            "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 0.3, "y": 0.0}],
            "options": {"samples": 4, "candidates": 1, "sysid_fit": model["path"]},
        }
        status, job = _request(f"{self.base}/api/solve", method="POST", payload=payload)
        self.assertEqual(status, 202)
        deadline = time.time() + 120
        while time.time() < deadline:
            status, body = _request(f"{self.base}/api/jobs/{job['id']}")
            if body["status"] != "running":
                break
            time.sleep(0.3)
        # The fitted plant is only used by the validator, so it turns on.
        log = " ".join(body["log"])
        self.assertIn("Digital twin", log)
        self.assertIn("Running the Immrax robustness check", log)


class TestStudioCalibration(StudioTestCase):
    """The calibration wizard: collect, measure, fit."""

    SESSION = """TEST_START odometry_straight
META,run_id,11
META,test,odometry_straight
META,repeat,0
META,poses,1
META,wheel_diameter_mm,56
META,axle_track_mm,96.5
DATA
t_ms,v_cmd_mm_s,gyro_z_deg_s
0,0,0
10,100,0
20,100,0
TEST_END,3
TEST_START multi_stop
META,run_id,12
META,test,multi_stop
META,repeat,0
META,poses,3
META,wheel_diameter_mm,56
META,axle_track_mm,96.5
DATA
t_ms,v_cmd_mm_s,gyro_z_deg_s
0,0,0
10,100,0
20,100,0
TEST_END,3
"""

    def setUp(self):
        super().setUp()
        self.calibration = tempfile.mkdtemp()
        self._patch = patch.object(gui_server, "CALIBRATION_DIR", Path(self.calibration))
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(shutil.rmtree, self.calibration, True)
        # A stand-in for the hub: it writes a session and prints what the hub
        # would print.
        self.fake = os.path.join(self.calibration, "fake_collect.py")
        with open(self.fake, "w", encoding="utf-8") as handle:
            handle.write(
                "import sys\n"
                f"session = {self.SESSION!r}\n"
                "out = sys.argv[sys.argv.index('--out') + 1]\n"
                "with open(out, 'w', encoding='utf-8') as handle:\n"
                "    handle.write(session)\n"
                "print('Hub runner: collecting')\n"
            )
        self._deploy_patch = patch.object(gui_server, "HUB_DEPLOY", Path(self.fake))
        self._deploy_patch.start()
        self.addCleanup(self._deploy_patch.stop)

    def _write_session(self):
        paths = gui_server.calibration_paths("my robot")
        paths["session"].parent.mkdir(parents=True, exist_ok=True)
        with open(paths["session"], "w", encoding="utf-8") as handle:
            handle.write(self.SESSION)

    def test_the_sheet_lists_the_measurements_the_session_needs(self):
        self._write_session()
        status, payload = _request(
            f"{self.base}/api/calibration/sheet?name=my%20robot")
        self.assertEqual(status, 200)
        self.assertEqual(payload["total"], 4)
        self.assertEqual(payload["missing"], 4)
        tests = {row["test"] for row in payload["rows"]}
        self.assertEqual(tests, {"odometry_straight", "multi_stop"})

    def test_saving_the_sheet_keeps_what_was_filled_in(self):
        self._write_session()
        rows = [
            {"test": "odometry_straight", "repeat": "0", "stop": "0",
             "x_mm": "998", "y_mm": "2", "turn_right_deg": "0"},
            {"test": "multi_stop", "repeat": "0", "stop": "0",
             "x_mm": "500", "y_mm": "0", "turn_right_deg": "0"},
            {"test": "multi_stop", "repeat": "0", "stop": "1",
             "x_mm": "", "y_mm": "", "turn_right_deg": ""},
            {"test": "multi_stop", "repeat": "0", "stop": "2",
             "x_mm": "", "y_mm": "", "turn_right_deg": ""},
        ]
        status, payload = _request(f"{self.base}/api/calibration/sheet", method="POST",
                                   payload={"name": "my robot", "rows": rows})
        self.assertEqual(status, 200)
        # Four measurements are needed; two were filled in.
        self.assertEqual(payload["missing"], 2)

        status, payload = _request(
            f"{self.base}/api/calibration/sheet?name=my%20robot")
        filled = {(row["test"], row["stop"]): row for row in payload["rows"]}
        self.assertEqual(filled[("odometry_straight", 0)]["x_mm"], "998")
        self.assertEqual(filled[("multi_stop", 0)]["x_mm"], "500")
        self.assertEqual(filled[("multi_stop", 1)]["x_mm"], "")

    def test_a_partly_filled_row_is_still_missing(self):
        """The fitter needs all three numbers of a row.

        A row with a blank turn is skipped whole, so the studio must not
        count it as measured.
        """
        self._write_session()
        rows = [
            {"test": "odometry_straight", "repeat": "0", "stop": "0",
             "x_mm": "998", "y_mm": "2", "turn_right_deg": ""},
            {"test": "multi_stop", "repeat": "0", "stop": "0",
             "x_mm": "500", "y_mm": "0", "turn_right_deg": "0"},
            {"test": "multi_stop", "repeat": "0", "stop": "1",
             "x_mm": "500", "y_mm": "0", "turn_right_deg": "0"},
            {"test": "multi_stop", "repeat": "0", "stop": "2",
             "x_mm": "500", "y_mm": "0", "turn_right_deg": "0"},
        ]
        status, payload = _request(f"{self.base}/api/calibration/sheet", method="POST",
                                   payload={"name": "my robot", "rows": rows})
        self.assertEqual(status, 200)
        # Three rows are complete; the straight one has no turn, so the
        # fitter will skip it whole.
        self.assertEqual(payload["missing"], 1)

    def test_a_stale_tab_does_not_wipe_what_was_filled_in(self):
        """The rows are merged with the ones on disk.

        A tab that was open before the collection posts rows it knows
        nothing about, and must not erase what another tab put there.
        """
        self._write_session()
        filled = [
            {"test": "odometry_straight", "repeat": "0", "stop": "0",
             "x_mm": "998", "y_mm": "2", "turn_right_deg": "0"},
        ]
        _request(f"{self.base}/api/calibration/sheet", method="POST",
                 payload={"name": "my robot", "rows": filled})
        # The stale tab posts the same rows, empty.
        stale = [
            {"test": "odometry_straight", "repeat": "0", "stop": "0",
             "x_mm": "", "y_mm": "", "turn_right_deg": ""},
            {"test": "multi_stop", "repeat": "0", "stop": "0",
             "x_mm": "", "y_mm": "", "turn_right_deg": ""},
        ]
        status, payload = _request(f"{self.base}/api/calibration/sheet", method="POST",
                                   payload={"name": "my robot", "rows": stale})
        self.assertEqual(status, 200)
        kept = {(row["test"], row["stop"]): row for row in payload["rows"]}
        self.assertEqual(kept[("odometry_straight", 0)]["x_mm"], "998",
                         "the stale tab wiped the measurement")

    def test_a_collect_pushes_the_collector_and_saves_the_session(self):
        status, job = _request(f"{self.base}/api/calibration/collect", method="POST",
                               payload={"name": "my robot", "connection": "ble"})
        self.assertEqual(status, 202)
        self.assertEqual(job["kind"], "collect")
        deadline = time.time() + 120
        while time.time() < deadline:
            status, body = _request(f"{self.base}/api/jobs/{job['id']}")
            if body["status"] != "running":
                break
            time.sleep(0.2)
        self.assertEqual(body["status"], "ok", body["log"][-3:])
        # The session is kept with the robot.
        session = gui_server.calibration_paths("my robot")["session"]
        self.assertTrue(session.is_file())
        self.assertIn("odometry_straight", session.read_text(encoding="utf-8"))

    def test_a_dropped_link_keeps_the_session_that_arrived(self):
        """A link that drops costs the last test, not the run.

        The collector writes the session as the output arrives, so a run
        that ends with a non-zero exit still holds a measurement.
        """
        broken = os.path.join(self.calibration, "fake_collect_broken.py")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write(
                "import sys\n"
                f"session = {self.SESSION!r}\n"
                "out = sys.argv[sys.argv.index('--out') + 1]\n"
                "with open(out, 'w', encoding='utf-8') as handle:\n"
                "    handle.write(session)\n"
                "print('Hub runner: collecting')\n"
                "print('Hub link lost')\n"
                "sys.exit(1)\n"
            )
        with patch.object(gui_server, "HUB_DEPLOY", Path(broken)):
            status, job = _request(f"{self.base}/api/calibration/collect",
                                   method="POST",
                                   payload={"name": "dropped", "connection": "ble"})
            self.assertEqual(status, 202)
            deadline = time.time() + 120
            while time.time() < deadline:
                status, body = _request(f"{self.base}/api/jobs/{job['id']}")
                if body["status"] != "running":
                    break
                time.sleep(0.2)
        self.assertEqual(body["status"], "failed")
        session = gui_server.calibration_paths("dropped")["session"]
        self.assertTrue(session.is_file(),
                        "the session that arrived was thrown away")
        self.assertIn("odometry_straight", session.read_text(encoding="utf-8"))

    def test_fitting_without_a_session_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            _request(f"{self.base}/api/calibration/fit", method="POST",
                     payload={"name": "nobody"})
        self.assertEqual(ctx.exception.code, 400)

    def test_the_fit_uses_the_wheel_diameter_from_the_session(self):
        self._write_session()
        paths = gui_server.calibration_paths("my robot")
        gui_server.save_sheet(paths["sheet"], [
            {"test": "odometry_straight", "repeat": "0", "stop": "0",
             "x_mm": "998", "y_mm": "0", "turn_right_deg": ""},
        ])
        status, summary = _request(f"{self.base}/api/calibration/fit", method="POST",
                                   payload={"name": "my robot"})
        self.assertEqual(status, 200)
        # 56 mm from the session's own metadata, not from a profile file.
        self.assertAlmostEqual(summary["wheel_radius_m"], 0.028)
        self.assertTrue(paths["fit"].is_file())


class TestStudioHelpers(unittest.TestCase):
    def test_slugify_keeps_file_names_safe(self):
        self.assertEqual(gui_server.slugify("My Mission!"), "my-mission")
        self.assertEqual(gui_server.slugify(""), "mission")
        self.assertEqual(gui_server.slugify("../../etc/passwd"), "etc-passwd")

    def test_command_contains_the_expected_flags(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 8, "auto_mesh": False, "accuracy_weight": 0.5,
             "candidates": 3, "timing": "per-segment", "validate": True,
             "export_format": "controller", "controller_dt": 0.02})
        joined = " ".join(str(part) for part in command)
        self.assertIn("-n 8", joined)
        self.assertIn("-a 0.5", joined)
        self.assertIn("--candidates 3", joined)
        self.assertIn("--validate", joined)
        self.assertIn("--export-format controller", joined)
        self.assertIn("--controller-dt 0.02", joined)

    def test_the_automatic_mesh_leaves_the_sample_count_out(self):
        """Without -n the planner sizes the invariant mesh from the path."""
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 10})
        self.assertNotIn("-n", command)

    def test_the_sample_count_is_sent_when_the_automatic_mesh_is_off(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 12, "auto_mesh": False})
        joined = " ".join(str(part) for part in command)
        self.assertIn("-n 12", joined)

    def test_the_propagation_model_is_passed_through(self):
        base = (Path("."), Path("c.json"), Path("w.json"), Path("o.traj"))
        command = gui_server.build_command(*base, {"propagation": "tick"})
        joined = " ".join(str(part) for part in command)
        self.assertIn("--propagation tick", joined)
        command = gui_server.build_command(*base, {"propagation": "plant"})
        joined = " ".join(str(part) for part in command)
        self.assertIn("--propagation plant", joined)
        # Nothing said is the command line's own default.
        command = gui_server.build_command(*base, {})
        self.assertNotIn("--propagation", command)
        command = gui_server.build_command(*base, {"propagation": "sideways"})
        self.assertNotIn("--propagation", command)

    def test_the_force_weight_and_the_polish_are_passed_through(self):
        base = (Path("."), Path("c.json"), Path("w.json"), Path("o.traj"))
        command = gui_server.build_command(*base, {"force_weight": 0.005, "polish": True})
        joined = " ".join(str(part) for part in command)
        self.assertIn("--force-weight 0.005", joined)
        self.assertIn("--polish", joined)

        command = gui_server.build_command(*base, {"force_weight": 0.0, "polish": False})
        joined = " ".join(str(part) for part in command)
        self.assertNotIn("--force-weight", joined)
        self.assertNotIn("--polish", joined)

    def test_best_effort_is_passed_through_only_with_validation(self):
        base = (Path("."), Path("c.json"), Path("w.json"), Path("o.traj"))
        command = gui_server.build_command(
            *base, {"validate": True, "best_effort": True})
        self.assertIn("--best-effort", command)

        # Without a verdict there is nothing to override.
        command = gui_server.build_command(*base, {"best_effort": True})
        self.assertNotIn("--best-effort", command)

    def test_the_accuracy_limit_is_passed_through_only_with_validation(self):
        base = (Path("."), Path("c.json"), Path("w.json"), Path("o.traj"))
        command = gui_server.build_command(
            *base, {"validate": True, "envelope_limit": 0.03})
        self.assertIn("--envelope-limit 0.03",
                      " ".join(str(part) for part in command))

        # The limit only means something to the validator.
        command = gui_server.build_command(*base, {"envelope_limit": 0.03})
        self.assertNotIn("--envelope-limit", command)

    def test_the_accuracy_allowance_per_metre_is_passed_through(self):
        base = (Path("."), Path("c.json"), Path("w.json"), Path("o.traj"))
        command = gui_server.build_command(
            *base, {"validate": True, "envelope_limit_per_metre": 0.006})
        self.assertIn("--envelope-limit-per-metre 0.006",
                      " ".join(str(part) for part in command))

        command = gui_server.build_command(
            *base, {"envelope_limit_per_metre": 0.006})
        self.assertNotIn("--envelope-limit-per-metre", command)

    def test_the_candidate_comparison_is_always_requested(self):
        command = gui_server.build_command(
            Path("jobs"), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 10})
        joined = " ".join(str(part) for part in command)
        self.assertIn("--candidates-out", joined)
        self.assertIn(gui_server.CANDIDATES_FILE, joined)

    def test_a_running_job_shows_the_paths_it_already_solved(self):
        """The field is drawn from the file, not only from the finished job."""
        with tempfile.TemporaryDirectory() as folder:
            job = gui_server.Job("job1", "test", Path(folder))
            self.assertEqual(job.snapshot()["result"], {})
            (Path(folder) / gui_server.CANDIDATES_FILE).write_text(
                json.dumps({"selected": None, "candidates": [
                    {"index": 1, "style": "frontier_sprint",
                     "samples": [{"t": 0.0, "x": 0.25, "y": 0.5}]}]}),
                encoding="utf-8")
            candidates = job.snapshot()["result"]["candidates"]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["samples"][0]["x"], 0.25)
            self.assertEqual(job.status, "running")

    def test_the_operator_pick_is_passed_through(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"pick_candidate": 2})
        joined = " ".join(str(part) for part in command)
        self.assertIn("--pick-candidate 2", joined)

        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"), {})
        self.assertNotIn("--pick-candidate", command)

    def test_the_candidate_count_is_clamped_to_the_frontier_triad(self):
        """The planner has three frontier candidates, so 9 is not a choice."""
        base = (Path("."), Path("c.json"), Path("w.json"), Path("o.traj"))
        joined = " ".join(str(part) for part in gui_server.build_command(*base, {"candidates": 9}))
        self.assertIn("--candidates 3", joined)
        joined = " ".join(str(part) for part in gui_server.build_command(*base, {"candidates": 0}))
        self.assertIn("--candidates 1", joined)
        command = gui_server.build_command(*base, {"candidates": 2})
        self.assertNotIn("--candidates", command)

    def test_command_does_not_force_a_timing_mode_by_default(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 10, "timing": "auto"})
        joined = " ".join(str(part) for part in command)
        self.assertNotIn("--uniform-dt", joined)
        self.assertNotIn("--per-segment-dt", joined)

    def test_command_leaves_unified_to_the_cli_default(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 10})
        self.assertNotIn("--unified", command)
        self.assertNotIn("--no-unified", command)

    def test_there_is_no_planner_choice_to_forward(self):
        """The 3-Pillar engine is the only planner, so the studio has no toggle."""
        base = (Path("."), Path("c.json"), Path("w.json"), Path("o.traj"))
        for options in ({"samples": 10},
                        {"samples": 10, "unified": True},
                        {"samples": 10, "unified": False}):
            command = gui_server.build_command(*base, options)
            self.assertNotIn("--unified", command)
            self.assertNotIn("--no-unified", command)

    def test_fast_polish_is_passed_through(self):
        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 10, "fast_polish": True})
        self.assertIn("--fast-polish", " ".join(str(part) for part in command))

        command = gui_server.build_command(
            Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
            {"samples": 10, "fast_polish": False})
        self.assertNotIn("--fast-polish", " ".join(str(part) for part in command))

    def test_a_stale_self_consistent_option_never_reaches_the_command(self):
        for value in (True, False):
            command = gui_server.build_command(
                Path("."), Path("c.json"), Path("w.json"), Path("o.traj"),
                {"samples": 10, "self_consistent": value})
            joined = " ".join(str(part) for part in command)
            self.assertNotIn("--self-consistent", joined)
            self.assertNotIn("--no-self-consistent", joined)

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
        # The missions folder is the operator's own git repository. The
        # test must not commit into it: it did, and the operator's history
        # carried "Mission unit-test-mission" commits (KI-037).
        with tempfile.TemporaryDirectory() as tmp:
            missions = Path(tmp)
            with patch.object(gui_server, "MISSIONS_DIR", missions):
                server = gui_server.create_server(0, {}, {})
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    status, payload = _request(f"{base}/api/missions", method="POST", payload={
                        "name": "unit test mission",
                        "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
                    })
                    self.assertEqual(status, 200)
                    self.assertEqual(payload["name"], "unit-test-mission")
                    self.assertTrue((missions / "unit-test-mission.json").is_file())
                finally:
                    server.shutdown()
                    server.server_close()

    def test_the_mission_history_is_a_git_history(self):
        """Every save is a commit, and an earlier version can be restored."""
        with tempfile.TemporaryDirectory() as tmp:
            missions = Path(tmp)
            with patch.object(gui_server, "MISSIONS_DIR", missions):
                server = gui_server.create_server(0, {}, {})
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    status, first = _request(f"{base}/api/missions", method="POST", payload={
                        "name": "history test",
                        "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
                    })
                    self.assertEqual(status, 200)
                    self.assertTrue(first["commit"])
                    self.assertEqual(len(first["history"]), 1)

                    status, second = _request(f"{base}/api/missions", method="POST", payload={
                        "name": "history test",
                        "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.0}],
                    })
                    self.assertEqual(status, 200)
                    self.assertEqual(len(second["history"]), 2)

                    status, history = _request(
                        f"{base}/api/missions/history?name=history%20test")
                    self.assertEqual(status, 200)
                    self.assertEqual(len(history["history"]), 2)

                    # Put the first version back.
                    oldest = history["history"][-1]["sha"]
                    status, restored = _request(f"{base}/api/missions/restore", method="POST",
                                                payload={"name": "history test", "sha": oldest})
                    self.assertEqual(status, 200)
                    self.assertEqual(restored["waypoints"][1]["x"], 1.0)
                finally:
                    server.shutdown()
                    server.server_close()

    def test_a_failed_commit_is_reported_not_hidden(self):
        """The mission is saved, but the studio says the history lost it."""
        real_git = gui_server._git

        def flaky_git(args, cwd):
            if "commit" in args:
                return False, "fatal: index.lock exists"
            return real_git(args, cwd)

        with tempfile.TemporaryDirectory() as tmp:
            missions = Path(tmp)
            with patch.object(gui_server, "MISSIONS_DIR", missions), \
                    patch.object(gui_server, "_git", flaky_git):
                server = gui_server.create_server(0, {}, {})
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                base = f"http://127.0.0.1:{server.server_address[1]}"
                try:
                    status, saved = _request(f"{base}/api/missions", method="POST", payload={
                        "name": "commit failure",
                        "waypoints": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
                    })
                finally:
                    server.shutdown()
                    server.server_close()

            # The mission itself is saved either way.
            self.assertTrue((missions / "commit-failure.json").is_file())

        self.assertEqual(status, 200)
        # No commit, but the reason is there: a silent None would leave the
        # operator believing the mission is in the history.
        self.assertIsNone(saved["commit"])
        self.assertIn("index.lock", saved["commit_error"])


class TestValidationVerdict(unittest.TestCase):
    """The server reads the validator's verdict back from the solve's output.

    The rule "only a validated trajectory goes to the robot" must not live
    only in the browser: the endpoint that drives the robot has to enforce
    it (KI-036).
    """

    def _job(self, lines):
        job = gui_server.Job("job1", "t", Path(tempfile.gettempdir()))
        for line in lines:
            job.log(line)
        return job

    def _report(self, selected, first_ok):
        """A report with two candidates; the second is the opposite of the first."""
        return gui_server.VALIDATION_MARKER + json.dumps({
            "selected": selected,
            "candidates": [{"ok": first_ok}, {"ok": not first_ok}],
        })

    def test_a_passing_run_reads_as_a_pass(self):
        job = self._job([self._report(1, True)])
        self.assertIs(gui_server._validation_verdict(job), True)

    def test_a_failing_run_reads_as_a_fail(self):
        job = self._job([self._report(1, False)])
        self.assertIs(gui_server._validation_verdict(job), False)

    def test_the_selected_candidate_is_the_one_that_counts(self):
        # Candidate 1 failed and candidate 2 passed.
        job = self._job([self._report(2, False)])
        self.assertIs(gui_server._validation_verdict(job), True)
        job = self._job([self._report(1, False)])
        self.assertIs(gui_server._validation_verdict(job), False)

    def test_an_unvalidated_run_has_no_verdict(self):
        job = self._job(["Trajectory written to output.traj"])
        self.assertIsNone(gui_server._validation_verdict(job))

    def test_a_broken_report_has_no_verdict(self):
        job = self._job([gui_server.VALIDATION_MARKER + "{not json"])
        self.assertIsNone(gui_server._validation_verdict(job))

    def test_the_last_report_wins(self):
        job = self._job([self._report(1, False), self._report(1, True)])
        self.assertIs(gui_server._validation_verdict(job), True)


if __name__ == "__main__":
    unittest.main()
