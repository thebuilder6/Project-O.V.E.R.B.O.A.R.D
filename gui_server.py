"""
Local browser studio for OVERBOARD.

The studio serves a small web app that lets someone draw a mission, solve it
and download the result. It has no extra dependencies: the HTTP server is
stdlib, and every solve runs the command line in a subprocess so the export
gate, the validation rules and the file formats stay in one place.

    python main.py --gui
"""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent
VIZ_DIR = REPO_ROOT / "viz"
EXAMPLES_DIR = REPO_ROOT / "examples"
MISSIONS_DIR = REPO_ROOT / "missions"

MAX_BODY_BYTES = 2_000_000
LOG_LINES = 800
SLUG_RE = re.compile(r"[^a-z0-9_-]+")

# Files the studio is allowed to serve from viz/.
STATIC_FILES = {
    "/studio.css": ("studio.css", "text/css; charset=utf-8"),
    "/studio.js": ("studio.js", "application/javascript; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
}


def slugify(name, fallback="mission"):
    """A safe file name from a user-supplied mission name."""
    slug = SLUG_RE.sub("-", str(name or "").strip().lower()).strip("-_")
    slug = slug[:60]
    return slug or fallback


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _is_robot_config(data):
    if not isinstance(data, dict):
        return False
    if "robot" in data or "config" in data:
        return True
    return "mass" in data and "wheel_radius" in data and "track_width" in data


def collect_examples():
    """Return the example robot configs and the saved missions."""
    configs = []
    missions = []

    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        data = _read_json(path)
        if data is None:
            continue
        if isinstance(data, list):
            missions.append({"name": path.stem, "source": "example", "data": data})
        elif _is_robot_config(data):
            configs.append({"name": path.stem, "source": "example", "data": data})

    if MISSIONS_DIR.is_dir():
        for path in sorted(MISSIONS_DIR.glob("*.json")):
            data = _read_json(path)
            if isinstance(data, list):
                missions.append({"name": path.stem, "source": "saved", "data": data})

    return {"configs": configs, "missions": missions}


def summarize_trajectory(path):
    """Pull the numbers a user cares about out of a written trajectory."""
    data = _read_json(path)
    if not data:
        return None
    samples = data.get("trajectory", {}).get("samples", [])
    if not samples:
        return None

    # The samples carry wheel velocities in m/s (robot_model.get_dynamics).
    peak_wheel = max(max(abs(s.get("vl", 0.0)), abs(s.get("vr", 0.0))) for s in samples)
    path_length = sum(math.hypot(samples[i + 1]["x"] - samples[i]["x"],
                                 samples[i + 1]["y"] - samples[i]["y"])
                      for i in range(len(samples) - 1))
    envelope = 0.0
    for sample in samples:
        env = sample.get("reachability_envelope")
        if not env:
            continue
        radius = math.hypot((env["x_max"] - env["x_min"]) / 2.0,
                             (env["y_max"] - env["y_min"]) / 2.0)
        envelope = max(envelope, radius)

    return {
        "duration_s": float(samples[-1]["t"]),
        "path_length_m": path_length,
        "peak_wheel_speed_m_s": peak_wheel,
        "max_envelope_m": envelope,
        "samples": len(samples),
        "has_envelope": envelope > 0.0,
    }


class Job:
    """One solve running in the background."""

    def __init__(self, job_id, name, output_dir):
        self.id = job_id
        self.name = name
        self.output_dir = output_dir
        self.status = "running"
        self.exit_code = None
        self.lines = deque(maxlen=LOG_LINES)
        self.proc = None
        self.started = time.time()
        self.finished = None
        self.error = ""

    def log(self, text):
        self.lines.append(text)

    def snapshot(self):
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "exit_code": self.exit_code,
            "started": self.started,
            "finished": self.finished,
            "error": self.error,
            "log": list(self.lines),
        }


class StudioServer(ThreadingHTTPServer):
    """HTTP server that owns the solve jobs."""

    daemon_threads = True
    # A busy port must fail loudly, never be taken from another program.
    allow_reuse_address = False

    def __init__(self, address, handler_class, presets, initial):
        super().__init__(address, handler_class)
        self.presets = presets or {}
        self.initial = initial or {}
        self.jobs = {}
        self.lock = threading.Lock()
        self.counter = 0

    def next_job_id(self):
        with self.lock:
            self.counter += 1
            return f"job{self.counter}"

    def add_job(self, job):
        with self.lock:
            self.jobs[job.id] = job
            return job

    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def active_job(self):
        with self.lock:
            for job in self.jobs.values():
                if job.status == "running":
                    return job
            return None


def build_command(job_dir, config_path, waypoints_path, output_path, options):
    """The command line for one solve."""
    command = [sys.executable, "-u", str(REPO_ROOT / "main.py"),
               "-c", str(config_path),
               "-w", str(waypoints_path),
               "-o", str(output_path)]

    samples = int(options.get("samples", 10))
    command += ["-n", str(samples)]

    accuracy = float(options.get("accuracy_weight", 0.0))
    if accuracy:
        command += ["-a", str(accuracy)]

    candidates = int(options.get("candidates", 2))
    if candidates != 2:
        command += ["--candidates", str(candidates)]

    timing = options.get("timing", "auto")
    if timing == "uniform":
        command.append("--uniform-dt")
    elif timing == "per-segment":
        command.append("--per-segment-dt")

    if options.get("validate"):
        command.append("--validate")

    export_format = options.get("export_format", "none")
    if export_format and export_format != "none":
        command += ["--export-format", str(export_format)]
        controller_dt = float(options.get("controller_dt", 0.02))
        command += ["--controller-dt", str(controller_dt)]

    return command


def start_job(server, payload):
    """Validate a solve request, then run it in the background."""
    name = slugify(payload.get("name"))
    config = payload.get("config")
    waypoints = payload.get("waypoints")
    options = payload.get("options") or {}

    if not isinstance(config, dict):
        raise ValueError("A robot configuration is required.")
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError("A mission needs at least two waypoints.")
    for index, point in enumerate(waypoints):
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            raise ValueError(f"Waypoint {index} needs an x and a y value.")
        try:
            float(point["x"])
            float(point["y"])
        except (TypeError, ValueError):
            raise ValueError(f"Waypoint {index} has a non-numeric position.")

    samples = int(options.get("samples", 10))
    if not 2 <= samples <= 200:
        raise ValueError("Samples per segment must be between 2 and 200.")
    candidates = int(options.get("candidates", 2))
    if not 1 <= candidates <= 5:
        raise ValueError("Candidates must be between 1 and 5.")
    if float(options.get("accuracy_weight", 0.0)) < 0:
        raise ValueError("The accuracy weight cannot be negative.")
    if options.get("validate") and options.get("export_format") == "none":
        pass

    job_dir = Path(tempfile.mkdtemp(prefix="overboard_studio_"))
    config_path = job_dir / "config.json"
    waypoints_path = job_dir / "waypoints.json"
    output_path = job_dir / f"{name}.traj"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=1)
    with open(waypoints_path, "w", encoding="utf-8") as handle:
        json.dump(waypoints, handle, indent=1)

    command = build_command(job_dir, config_path, waypoints_path, output_path, options)
    job = Job(server.next_job_id(), name, job_dir)
    server.add_job(job)

    def run():
        try:
            proc = subprocess.Popen(
                command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace")
            job.proc = proc
            for line in proc.stdout:
                job.log(line.rstrip("\n"))
            proc.stdout.close()
            job.exit_code = proc.wait()
        except Exception as exc:  # noqa: BLE001 - report it in the UI
            job.error = str(exc)
            job.log(f"Studio error: {exc}")
            job.exit_code = -1
        finally:
            job.finished = time.time()
            if job.status == "running":
                job.status = "ok" if job.exit_code == 0 else "failed"
            if job.status == "failed" and not job.error:
                job.error = job.lines[-1] if job.lines else "The solve failed."

    thread = threading.Thread(target=run, daemon=True, name=f"Solve{job.id}")
    thread.start()
    job.thread = thread
    return job


def artifact_path(job, kind):
    """Map an artifact kind to a file inside the job directory."""
    names = {
        "traj": f"{job.name}.traj",
        "controller": f"{job.name}_controller.json",
        "python": f"{job.name}.py",
        "stats": f"{job.name}_stats.json",
    }
    if kind not in names:
        return None
    return job.output_dir / names[kind]


class StudioHandler(BaseHTTPRequestHandler):
    """Routes for the studio UI."""

    server_version = "OVERBOARD"
    protocol_version = "HTTP/1.1"

    # --- plumbing -------------------------------------------------------
    def log_message(self, fmt, *args):
        pass  # keep the console readable

    def _host_allowed(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost", "::1", "")

    def _origin_allowed(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = urlparse(origin).hostname
        return host in ("127.0.0.1", "localhost", "::1")

    def _send(self, code, body, content_type="application/json; charset=utf-8",
              extra_headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, payload):
        self._send(code, json.dumps(payload))

    def _error(self, code, message):
        self._json(code, {"error": message})

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("The request is too large.")
        if not length:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # --- routing --------------------------------------------------------
    def do_GET(self):
        if not self._host_allowed():
            return self._error(403, "Only local requests are allowed.")
        path = urlparse(self.path).path
        try:
            if path in ("/", "/studio.html"):
                return self._serve_static("studio.html", "text/html; charset=utf-8")
            if path in STATIC_FILES:
                filename, content_type = STATIC_FILES[path]
                return self._serve_static(filename, content_type)
            if path == "/api/health":
                return self._json(200, self._health())
            if path == "/api/initial":
                return self._json(200, self.server.initial)
            if path == "/api/examples":
                return self._json(200, collect_examples())
            if path.startswith("/api/jobs/"):
                return self._job_route(path)
            return self._error(404, "Not found.")
        except Exception as exc:  # noqa: BLE001 - never take the server down
            traceback.print_exc()
            return self._error(500, str(exc))

    def do_POST(self):
        if not self._host_allowed() or not self._origin_allowed():
            return self._error(403, "Only local requests are allowed.")
        path = urlparse(self.path).path
        try:
            if path == "/api/solve":
                return self._solve()
            if path == "/api/missions":
                return self._save_mission()
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[3]
                return self._cancel_job(job_id)
            return self._error(404, "Not found.")
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._error(500, str(exc))

    # --- handlers -------------------------------------------------------
    def _serve_static(self, filename, content_type):
        path = VIZ_DIR / filename
        if not path.is_file():
            return self._error(404, "Not found.")
        with open(path, "rb") as handle:
            body = handle.read()
        self._send(200, body, content_type=content_type)

    def _health(self):
        from diagnostics import check_environment, summarize
        checks = check_environment(ports=())
        ok, warn, error = summarize(checks)
        packages = {c["name"]: c["status"] for c in checks
                    if c["name"] in ("jax", "jaxopt", "immrax", "casadi", "numpy")}
        return {
            "packages": packages,
            "counts": {"ok": ok, "warn": warn, "error": error},
            "has_validation": packages.get("jax") == "ok" and packages.get("immrax") == "ok",
            "presets": self.server.presets,
        }

    def _job_route(self, path):
        parts = path.strip("/").split("/")
        job = self.server.get_job(parts[2]) if len(parts) >= 3 else None
        if job is None:
            return self._error(404, "Unknown job.")
        if len(parts) == 3:
            payload = job.snapshot()
            summary = None
            if job.status == "ok":
                summary = summarize_trajectory(artifact_path(job, "traj"))
            payload["summary"] = summary
            payload["artifacts"] = self._artifacts(job)
            return self._json(200, payload)
        if len(parts) == 4 and parts[3] == "trajectory":
            path_to_traj = artifact_path(job, "traj")
            if not path_to_traj or not path_to_traj.is_file():
                return self._error(404, "No trajectory was written.")
            with open(path_to_traj, "rb") as handle:
                return self._send(200, handle.read(),
                                  content_type="application/json; charset=utf-8")
        if len(parts) == 5 and parts[3] == "artifact":
            return self._download(job, parts[4])
        return self._error(404, "Not found.")

    def _artifacts(self, job):
        found = []
        for kind in ("traj", "controller", "python", "stats"):
            path = artifact_path(job, kind)
            if path and path.is_file():
                found.append({"kind": kind, "name": path.name,
                              "size": path.stat().st_size})
        return found

    def _download(self, job, kind):
        path = artifact_path(job, kind)
        if not path or not path.is_file():
            return self._error(404, "That file was not written.")
        with open(path, "rb") as handle:
            body = handle.read()
        content_type = "application/json; charset=utf-8"
        if kind == "python":
            content_type = "text/x-python; charset=utf-8"
        return self._send(200, body, content_type=content_type, extra_headers={
            "Content-Disposition": f'attachment; filename="{path.name}"',
        })

    def _solve(self):
        if self.server.active_job() is not None:
            return self._error(409, "A solve is already running. Cancel it first.")
        payload = self._read_body()
        job = start_job(self.server, payload)
        return self._json(202, job.snapshot())

    def _cancel_job(self, job_id):
        job = self.server.get_job(job_id)
        if job is None:
            return self._error(404, "Unknown job.")
        if job.status == "running" and job.proc is not None:
            try:
                job.proc.terminate()
            except OSError:
                pass
            job.status = "cancelled"
        return self._json(200, job.snapshot())

    def _save_mission(self):
        payload = self._read_body()
        waypoints = payload.get("waypoints")
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ValueError("A mission needs at least two waypoints.")
        name = slugify(payload.get("name"))
        MISSIONS_DIR.mkdir(exist_ok=True)
        path = MISSIONS_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(waypoints, handle, indent=1)
        return self._json(200, {"name": name, "path": str(path)})


def create_server(port, presets, initial):
    """Start the studio on the first free port at or after `port`."""
    attempts = [port] if port == 0 else list(range(port, port + 10))
    last_error = None
    for candidate in attempts:
        try:
            return StudioServer(("127.0.0.1", candidate), StudioHandler,
                                presets, initial)
        except OSError as exc:
            last_error = exc
    raise RuntimeError(
        f"No free port between {attempts[0]} and {attempts[-1]}: {last_error}"
    )


def build_initial(config_path=None, waypoints_path=None):
    """The mission and robot the studio opens with."""
    initial = {"config": None, "waypoints": None, "config_name": None}

    config_file = Path(config_path) if config_path else EXAMPLES_DIR / "fll_choreo.json"
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    data = _read_json(config_file)
    if data is not None:
        initial["config"] = data
        initial["config_name"] = config_file.stem

    if waypoints_path:
        waypoint_file = Path(waypoints_path)
        if not waypoint_file.is_absolute():
            waypoint_file = REPO_ROOT / waypoint_file
        data = _read_json(waypoint_file)
        if isinstance(data, list):
            initial["waypoints"] = data
    return initial


def run_studio(config_path=None, waypoints_path=None, port=8000, presets=None):
    """Run the studio until the user stops it."""
    initial = build_initial(config_path=config_path, waypoints_path=waypoints_path)
    try:
        server = create_server(port, presets, initial)
    except RuntimeError as exc:
        print(f"Studio error: {exc}")
        raise SystemExit(1)

    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}"
    print("OVERBOARD studio")
    print(f"  Open {url} in your browser. Press Ctrl+C here to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStudio stopped.")
    finally:
        job = server.active_job()
        if job and job.proc is not None:
            try:
                job.proc.terminate()
            except OSError:
                pass
        server.shutdown()
        server.server_close()
