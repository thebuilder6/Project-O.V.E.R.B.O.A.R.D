"""
Local browser studio for OVERBOARD.

The studio serves a small web app that lets someone draw a mission, solve it
and download the result. It has no extra dependencies: the HTTP server is
stdlib, and every solve runs the command line in a subprocess so the export
gate, the validation rules and the file formats stay in one place.

    python -m overboard.main --gui
"""

import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Ensure repo root is on sys.path if invoked directly as a script
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from overboard.paths import (
    REPO_ROOT,
    VIZ_DIR,
    EXAMPLES_DIR,
    MISSIONS_DIR,
    CALIBRATION_DIR,
    MATS_DIR,
    COLLECTOR_SCRIPT,
    PYBRICKS_DIR,
)

MAX_BODY_BYTES = 2_000_000
# The field guide is under a megabyte; anything larger is not it.
MAX_MAT_DOCUMENT_BYTES = 32_000_000
# How far the picture's shape may differ from the mat's before the studio
# says so: one percent is about 20 mm of a 2 m mat.
MAT_ASPECT_TOLERANCE = 0.01
LOG_LINES = 800
# How many finished jobs the studio keeps. Every job owns a temporary
# folder with its trajectory, its controller and its stats, and nothing
# removed either, so a season of solves filled the temp folder and left
# every old trajectory downloadable by guessing an id (KI-037).
MAX_JOBS = 20
# The machine-readable validation summary the CLI prints for a validated
# run. The server reads it back, so the rule "only a validated trajectory
# goes to the robot" is enforced here and not only in the browser.
VALIDATION_MARKER = "##OVERBOARD-VALIDATION## "
# Unicode letters and digits are kept in a file name. Stripping them
# turned every non-ASCII mission name into the same file name, so one
# mission overwrote another without a word (KI-039).
SLUG_RE = re.compile(r"[^\w-]+", re.UNICODE)

# The host-side push client. Can be a command list or patched with a Path/str in tests.
HUB_DEPLOY = [sys.executable, "-u", "-m", "overboard.deploy.hub_deploy"]


def _deploy_command(*args):
    """Build the command to execute hub_deploy, supporting both patched HUB_DEPLOY and default -m overboard.deploy.hub_deploy."""
    if isinstance(HUB_DEPLOY, (list, tuple)):
        return list(HUB_DEPLOY) + [str(a) for a in args]
    if HUB_DEPLOY is None:
        return [sys.executable, "-u", "-m", "overboard.deploy.hub_deploy"] + [str(a) for a in args]
    return [sys.executable, "-u", str(HUB_DEPLOY)] + [str(a) for a in args]

# Files the studio is allowed to serve from viz/.
STATIC_FILES = {
    "/studio.css": ("studio.css", "text/css; charset=utf-8"),
    "/studio.js": ("studio.js", "application/javascript; charset=utf-8"),
    "/report.js": ("report.js", "application/javascript; charset=utf-8"),
}


def slugify(name, fallback="mission"):
    """A safe file name from a user-supplied mission name.

    Unicode letters and digits are kept. The old rule stripped everything
    outside ``[a-z0-9_-]``, so "任务" and "テスト" both became "mission"
    and the second save overwrote the first one (KI-039).
    """
    slug = SLUG_RE.sub("-", str(name or "").strip().lower()).strip("-_")
    slug = slug[:60]
    return slug or fallback


def _validation_verdict(job):
    """The validator's verdict for a solve, read back from its own output.

    ``main.py`` prints one machine-readable line per validated run. The
    line carries every candidate and says which one was selected, so the
    server can check the selected candidate's own verdict instead of
    trusting the browser.

    Returns True when the run was validated and the selected candidate
    passed, False when it was validated and did not pass, and None when
    the run carried no verdict at all - which is what an unvalidated run
    looks like.
    """
    for line in reversed(list(job.lines)):
        text = line.strip()
        if not text.startswith(VALIDATION_MARKER):
            continue
        try:
            report = json.loads(text[len(VALIDATION_MARKER):])
        except ValueError:
            return None
        candidates = report.get("candidates") or []
        if not candidates:
            return None
        try:
            index = int(report.get("selected") or 1)
        except (TypeError, ValueError):
            return None
        if not 1 <= index <= len(candidates):
            return None
        return bool(candidates[index - 1].get("ok"))
    return None


def _process_group_kwargs():
    """Put the child in its own process group, so it can be killed whole.

    The candidates are solved in worker processes, and killing only the
    parent left those workers running: they kept solving at full speed
    after the studio said "Cancelled" (KI-037).
    """
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _terminate_tree(proc):
    """Stop a solve and everything it started."""
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, AttributeError, ValueError):
        pass
    try:
        proc.terminate()
    except OSError:
        pass


class BusyError(RuntimeError):
    """The studio is already running a job, so this one cannot start."""


def _register_job(server, job):
    """Register a job, or refuse it when one is already running.

    The job's temporary folder is removed when the studio is busy: the job
    never runs, so nothing will ever read it.
    """
    try:
        server.add_job(job)
    except BusyError:
        shutil.rmtree(str(job.output_dir), ignore_errors=True)
        raise


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
    """One solve or push running in the background."""

    def __init__(self, job_id, name, output_dir, kind="solve"):
        self.id = job_id
        self.name = name
        self.output_dir = output_dir
        self.kind = kind
        self.status = "running"
        self.exit_code = None
        self.lines = deque(maxlen=LOG_LINES)
        self.proc = None
        self.started = time.time()
        self.finished = None
        self.error = ""
        self.result = {}
        self.cancel_requested = False

    def log(self, text):
        self.lines.append(text)

    def snapshot(self):
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "exit_code": self.exit_code,
            "started": self.started,
            "finished": self.finished,
            "wall_s": (self.finished - self.started) if self.finished else None,
            "error": self.error,
            "log": list(self.lines),
            "result": self.result,
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
        """Register a job, or refuse it when one is already running.

        The check and the registration are one step under the lock. The
        studio runs one job at a time, so the check the handlers do first
        is only a fast path: two requests that arrive together would both
        pass it, and two solves would run at once.
        """
        with self.lock:
            for other in self.jobs.values():
                if other.status == "running":
                    raise BusyError("A solve is already running. Cancel it first.")
            self.jobs[job.id] = job
            self._evict_old_jobs()
            return job

    def _evict_old_jobs(self):
        """Keep the newest finished jobs and throw the rest away.

        Every job owns a temporary folder with its trajectory, its
        controller and its stats. Nothing removed either, so a season of
        solves filled the temp folder and left every old trajectory
        downloadable by guessing an id (KI-037). The running job is never
        evicted: it is the one the studio is working on.
        """
        finished = [job for job in self.jobs.values() if job.status != "running"]
        finished.sort(key=lambda job: job.started)
        for old in finished[:max(0, len(finished) - MAX_JOBS)]:
            self.jobs.pop(old.id, None)
            shutil.rmtree(str(old.output_dir), ignore_errors=True)

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
    command = [sys.executable, "-u", "-m", "overboard.main",
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

    if options.get("fast_polish"):
        command.append("--fast-polish")

    if options.get("sweep"):
        command.append("--sweep")

    if options.get("self_consistent") is False:
        command.append("--no-self-consistent")
    elif options.get("self_consistent"):
        command.append("--self-consistent")

    # Leave the flag out when the caller has no preference, so the CLI default
    # applies. An explicit false must be forwarded: --unified is now the default.
    if "unified" in options:
        command.append("--unified" if options["unified"] else "--no-unified")

    # The iteration history is always requested: the studio's convergence
    # plot and its candidate explainer both read it, and without it the
    # browser shows nothing about how the solve converged.
    command += ["--iteration-history", str(job_dir / "iterations.json")]

    sysid_fit = options.get("sysid_fit")
    if sysid_fit:
        command += ["--sysid-fit", str(sysid_fit)]

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
    options = dict(payload.get("options") or {})
    # A fitted plant is only used by the validator, so it forces validation on.
    if options.get("sysid_fit") and not options.get("validate"):
        options["validate"] = True

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
    job.result = {"trajectory": {}, "limits": {}, "iterations": [], "collisions": []}
    _register_job(server, job)

    def run():
        try:
            proc = subprocess.Popen(
                command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                **_process_group_kwargs())
            job.proc = proc
            if job.cancel_requested:
                # The cancel arrived before the process was up.
                _terminate_tree(proc)
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
            # The plots need the limits and, when asked for, the way the
            # trajectory converged. Fill them in before the status flips, so a
            # poll that sees the job finished also sees the result.
            #
            # The status is set first, and the post-processing is guarded:
            # a failure here must not leave the job "running", or the studio
            # would refuse every later job and the operator could not even
            # cancel it.
            # Populate trajectory, limits, collisions, and iterations before
            # flipping the status, so a poll that sees the job finished also
            # sees the full result (KI-046.1). The post-processing is in a
            # try...finally block: a failure here must not leave the job
            # "running", or the studio would refuse every later job.
            try:
                samples = []
                try:
                    traj = artifact_path(job, "traj")
                    if traj and traj.is_file():
                        document = _read_json(traj) or {}
                        job.result["trajectory"] = document.get("trajectory") or {}
                        samples = (document.get("trajectory") or {}).get("samples", [])
                except Exception as exc:  # noqa: BLE001 - a plot is not worth a failure
                    job.log(f"Studio could not read the trajectory: {exc}")
                    samples = []
                try:
                    job.result["limits"] = trajectory_limits(config_path, samples)
                except Exception:  # noqa: BLE001 - a plot is not worth a failure
                    job.result["limits"] = {}
                try:
                    job.result["collisions"] = field_collisions(
                        samples, config, options.get("field"))
                except Exception:  # noqa: BLE001 - a mark is not worth a failure
                    job.result["collisions"] = []
                history = job_dir / "iterations.json"
                if history.is_file():
                    job.result["iterations"] = _read_json(history) or []
            finally:
                if job.status == "running":
                    if job.cancel_requested:
                        job.status = "cancelled"
                    else:
                        job.status = "ok" if job.exit_code == 0 else "failed"
                if job.status == "failed" and not job.error:
                    job.error = job.lines[-1] if job.lines else "The solve failed."

    thread = threading.Thread(target=run, daemon=True, name=f"Solve{job.id}")
    thread.start()
    return job


def _interval(summary, name):
    """The 95% interval of one parameter, or None."""
    stats = summary.get(f"{name}_between_runs") or {}
    ci95 = stats.get("ci95")
    if isinstance(ci95, list) and len(ci95) == 2:
        return [float(ci95[0]), float(ci95[1])]
    return None


def summarize_fit(fit):
    """The numbers of a fitted model, the way the fitter prints them."""
    actuator = fit.get("actuator", {}) or {}
    straight = actuator.get("straight", {}) or {}
    turn = actuator.get("turn", {}) or {}
    sensors = fit.get("sensors", {}) or {}
    odometry = fit.get("odometry", {}) or {}
    slip = fit.get("slip", {}) or {}

    def value(source, key):
        item = source.get(key)
        return float(item) if isinstance(item, (int, float)) else None

    return {
        "calibrated": bool(straight.get("calibrated") or turn.get("calibrated")),
        "wheel_radius_m": value(fit, "wheel_radius_m"),
        "straight": {
            "calibrated": bool(straight.get("calibrated")),
            "gain": value(straight, "gain"),
            "gain_ci95": _interval(straight, "gain"),
            "tau_s": value(straight, "tau_s"),
            "delay_s": value(straight, "delay_s"),
            "n_runs": straight.get("n_runs"),
        },
        "turn": {
            "calibrated": bool(turn.get("calibrated")),
            "gain": value(turn, "gain"),
            "gain_ci95": _interval(turn, "gain"),
            "tau_s": value(turn, "tau_s"),
            "delay_s": value(turn, "delay_s"),
            "n_runs": turn.get("n_runs"),
        },
        "gyro_z_bias_deg_s": value(sensors, "gyro_z_bias_deg_s"),
        "odometry": {
            "calibrated": bool(odometry.get("calibrated")),
            "distance_scale": value(odometry, "distance_scale"),
            "distance_scale_ci95": _interval(odometry, "distance_scale"),
            "heading_scale": value(odometry, "heading_scale"),
            "heading_scale_ci95": _interval(odometry, "heading_scale"),
        },
        "slip": {
            "calibrated": bool(slip.get("calibrated")),
            "brake_ratio_median": value(slip, "brake_ratio_median"),
            "brake_ratio_min": value(slip, "brake_ratio_min"),
            "brake_ratio_max": value(slip, "brake_ratio_max"),
        },
    }


def load_fit(path):
    """Read a fitted model and summarise it."""
    fit = _read_json(path)
    if not isinstance(fit, dict):
        raise ValueError(f"'{path}' is not a fitted model.")
    if "actuator" not in fit:
        raise ValueError(f"'{path}' has no actuator model. "
                         "It does not look like a sysid_fit.json.")
    summary = summarize_fit(fit)
    summary["source"] = str(path)
    return summary


def field_payload():
    """The fields the studio can draw on, by season.

    The footprints are measured off FIRST's public field wireframe and live
    in ``fields/``. The mat's picture is not ours to ship, so the operator
    brings their own image; the geometry is only the coordinates on it.
    """
    from overboard.viz.fields import load_all_fields

    models = load_all_fields()
    return {
        "seasons": [
            {"id": model.season, "label": model.label}
            for model in models.values()
        ],
        "fields": {
            model.season: model.as_dict() for model in models.values()
        },
    }


def _read_bounded(response, limit):
    """Read a response, up to ``limit`` bytes. None if it is longer."""
    chunks = []
    total = 0
    while True:
        chunk = response.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def download_field_mat(season, timeout=120):
    """Compose a season's mat picture from FIRST's own documents.

    FIRST's mat artwork is copyrighted, so it is not in this tree and this
    never puts it in one. The picture is *composed* on this machine from the
    document FIRST publishes: the field guide's wireframe page is rendered,
    cropped to the mat and scaled to the mat's physical size, so the picture
    maps exactly onto the mat. This is the same idea as the reference
    simulator's ``tools/mat-import``.

    The composed picture and the document it came from are kept in
    ``mats/``, which is not tracked.

    Returns:
        A dict with the picture's URL in this server, its size and the
        credit line.
    """
    from overboard.viz.fields import load_field

    field = load_field(season)
    if field is None:
        raise ValueError(f"There is no field for the season {season!r}.")
    source = field.mat_source or {}
    url = source.get("url")
    if not url:
        raise ValueError(f"The field {season!r} has no mat document to read.")

    MATS_DIR.mkdir(parents=True, exist_ok=True)
    # The picture is cached under the document's name, so two fields that
    # share a mat share the picture too.
    mat_mm = str(source.get("mat_mm", "2004x1143"))
    stem = hashlib.sha1(f"{url}|{mat_mm}".encode("utf-8")).hexdigest()[:12]
    target = MATS_DIR / f"mat-{stem}.png"
    document = MATS_DIR / f"mat-{stem}-source.pdf"
    if not target.is_file():
        if not document.is_file():
            request = urllib.request.Request(
                url, headers={"User-Agent": "overboard-studio"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    # Read it in chunks and stop at the limit: the document
                    # is under a megabyte, and a server that sends more
                    # than that is not the one the field file names.
                    data = _read_bounded(response, MAX_MAT_DOCUMENT_BYTES)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported to the browser
                raise ValueError(
                    f"Could not fetch FIRST's field guide: {exc}") from exc
            if data is None:
                raise ValueError("The field guide is larger than expected.")
            if not data.startswith(b"%PDF"):
                raise ValueError("What came back is not a PDF.")
            with open(document, "wb") as handle:
                handle.write(data)
        try:
            from overboard.viz.mat_import import compose_mat
            composed = compose_mat(
                str(document), str(target),
                mat_mm=mat_mm,
                page=int(source.get("page", 2) or 2))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        # The picture is stretched to the mat's size, so a crop that is not
        # the mat's shape puts its features up to a tenth of the mat away
        # from where they belong. The composition is still kept - the
        # operator can see it - but the studio is told.
        if composed.get("aspect_error", 0.0) > MAT_ASPECT_TOLERANCE:
            raise ValueError(
                "The mat's picture does not have the mat's shape "
                f"({composed['source_aspect']:.4f} against "
                f"{composed['mat_aspect']:.4f}); the picture would be "
                "stretched. Check the field's mat size or the page.")

    return {
        "url": f"/mats/{target.name}",
        "width_m": field.mat_width_m,
        "height_m": field.mat_height_m,
        "credit": source.get("credit", ""),
        "source_url": url,
    }


def field_collisions(samples, config, season):
    """Where the planned path runs into a mission model.

    The robot is treated as a disc as wide as its axle track, centred on the
    path. Nothing is changed and nothing is blocked: the collisions are only
    marked, so a human can decide what to do about them.
    """
    if not season or not samples:
        return []
    from overboard.viz.fields import load_field
    from overboard.models.robot_model import RobotConfig

    field = load_field(season)
    if field is None:
        return []
    try:
        track = float(RobotConfig(config or {}).track_width)
    except (TypeError, ValueError, KeyError):
        track = 0.0
    radius = max(track, 0.0) / 2.0
    return [item.as_dict() for item in field.collisions(samples, radius)]


def list_fits():
    """The measured models the studio knows about."""
    models = []
    if not CALIBRATION_DIR.is_dir():
        return models
    for path in sorted(CALIBRATION_DIR.glob("*.sysid_fit.json")):
        try:
            summary = load_fit(path)
        except ValueError:
            continue
        summary["name"] = path.stem.replace(".sysid_fit", "")
        summary["path"] = str(path)
        models.append(summary)
    return models


def store_fit(name, fit):
    """Keep a fitted model with the robot, and summarise it."""
    if not isinstance(fit, dict) or "actuator" not in fit:
        raise ValueError("That file has no actuator model. "
                         "It does not look like a sysid_fit.json.")
    CALIBRATION_DIR.mkdir(exist_ok=True)
    path = CALIBRATION_DIR / f"{slugify(name, 'robot')}.sysid_fit.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(fit, handle, indent=2)
        handle.write("\n")
    summary = summarize_fit(fit)
    summary["source"] = str(path)
    summary["name"] = path.stem.replace(".sysid_fit", "")
    summary["path"] = str(path)
    return summary


def calibration_paths(name):
    """Where the files of one calibration live."""
    stem = str(CALIBRATION_DIR / slugify(name, "robot"))
    return {
        "session": Path(stem + ".session.csv"),
        "sheet": Path(stem + ".sheet.csv"),
        "poses": Path(stem + ".poses.csv"),
        "fit": Path(stem + ".sysid_fit.json"),
    }


def _session_runs(path):
    """The runs in a captured session."""
    from overboard.sysid.fit import parse_session
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return []
    return parse_session(text)


def _sheet_rows(runs):
    """One row per measurement the session needs."""
    from overboard.sysid.fit import session_tests
    rows = []
    for (name, repeat), (run_id, poses) in sorted(session_tests(runs).items()):
        for stop in range(poses):
            rows.append({"test": name, "repeat": repeat, "stop": stop})
    return rows


def collect_progress(session_path):
    """Where the collection is: the test running, and what still needs measuring."""
    runs = _session_runs(session_path)
    rows = _sheet_rows(runs)
    current = None
    if runs:
        last = runs[-1]
        current = {
            "test": last["name"],
            "repeat": last["meta"].get("repeat"),
            "run_id": last["meta"].get("run_id"),
        }
    return {
        "current": current,
        "tests_done": len(runs),
        "rows": rows,
        "needed": len(rows),
    }


def read_sheet_rows(sheet_path, rows):
    """The measurement sheet with whatever the operator has filled in.

    The rows are read as text, so a measurement the operator typed is shown
    back even when one of its numbers is blank. The fitter reads the sheet
    with its own reader, which skips a row that is not complete.
    """
    filled = {}
    if sheet_path and sheet_path.is_file():
        try:
            with open(sheet_path, "r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    name = (raw.get("test") or "").strip()
                    if not name:
                        continue
                    key = (name, (raw.get("repeat") or "0").strip(),
                           (raw.get("stop") or "0").strip())
                    filled[key] = raw
        except OSError:
            filled = {}
    out = []
    for row in rows:
        key = (row["test"], str(row["repeat"]), str(row["stop"]))
        value = filled.get(key, {})
        out.append({
            "test": row["test"],
            "repeat": row["repeat"],
            "stop": row["stop"],
            "x_mm": (value.get("x_mm") or "").strip(),
            "y_mm": (value.get("y_mm") or "").strip(),
            "turn_right_deg": (value.get("turn_right_deg") or "").strip(),
        })
    return out


def _sheet_key(row):
    """The identity of one measurement row: the test, the repeat, the stop."""
    return (str(row.get("test", "") or "").strip(),
            str(row.get("repeat", "0") or "0").strip(),
            str(row.get("stop", "0") or "0").strip())


# The three numbers the fitter needs for a row. A row without all three is
# skipped whole, so a partly filled row is still missing.
MEASUREMENT_FIELDS = ("x_mm", "y_mm", "turn_right_deg")


def save_sheet(sheet_path, rows):
    """Write the operator's measurements, in the fitter's sheet format.

    The browser's rows are merged with the ones already on disk. A row the
    browser sends empty keeps what the file holds, and a row the browser
    does not send at all is kept. A tab that was open before the last
    collection must not wipe what another tab - or the operator's own last
    save - put there.
    """
    from overboard.sysid.fit import SHEET_FIELDS
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if sheet_path.is_file():
        try:
            with open(sheet_path, "r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    if not (raw.get("test") or "").strip():
                        continue
                    merged[_sheet_key(raw)] = {
                        field: str(raw.get(field, "") or "").strip()
                        for field in SHEET_FIELDS
                    }
        except OSError:
            merged = {}
    for row in rows:
        key = _sheet_key(row)
        values = {field: str(row.get(field, "") or "").strip()
                  for field in SHEET_FIELDS}
        # Only the measurement cells count: the row's name is never blank.
        measured = any(values.get(field, "") for field in MEASUREMENT_FIELDS)
        if not measured and key in merged:
            # The browser knows nothing about this row: keep what is on disk.
            continue
        merged[key] = values
    with open(sheet_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for key in merged:
            writer.writerow({field: merged[key].get(field, "")
                             for field in SHEET_FIELDS})
    return sheet_path


def count_missing(rows):
    """How many of the measurements the fitter still needs are missing.

    The fitter needs all three numbers of a row: a row with one blank cell
    is skipped whole, so a partly filled row is still missing.
    """
    missing = 0
    for row in rows:
        complete = True
        for field in MEASUREMENT_FIELDS:
            value = str(row.get(field, "") or "").strip()
            if value == "":
                complete = False
                break
            try:
                float(value)
            except (TypeError, ValueError):
                complete = False
                break
        if not complete:
            missing += 1
    return missing


def fit_from_sheet(name):
    """Fit the model from the collected session and the filled sheet."""
    from overboard.sysid.fit import fit_session, sheet_to_poses, write_poses_csv

    paths = calibration_paths(name)
    if not paths["session"].is_file():
        raise ValueError("There is no collected session for this robot yet.")
    if not paths["sheet"].is_file():
        raise ValueError("There is no measurement sheet yet.")

    runs = _session_runs(paths["session"])
    if not runs:
        raise ValueError("The session has no test blocks in it.")

    from overboard.sysid.fit import read_sheet
    rows = read_sheet(str(paths["sheet"]))
    pose_rows, problems = sheet_to_poses(rows, runs)

    # The robot's own wheel diameter, measured by the robot: the hub prints it
    # in the session's metadata, so no drivebase profile file is needed.
    wheel_radius = None
    for run in runs:
        diameter = run["meta"].get("wheel_diameter_mm")
        if diameter:
            wheel_radius = float(diameter) / 2000.0
            break

    poses = {}
    for row in pose_rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        poses.setdefault(run_id, []).append({
            "stop": int(float(row.get("stop") or 0)),
            "x_m": float(row.get("x_m") or 0.0),
            "y_m": float(row.get("y_m") or 0.0),
            "heading_deg": float(row.get("heading_deg") or 0.0),
        })
    for entries in poses.values():
        entries.sort(key=lambda entry: entry["stop"])

    if wheel_radius is None:
        raise ValueError("The session does not say the wheel diameter, so the "
                         "fit has no wheel radius to work with.")

    result = fit_session(runs, poses=poses, wheel_radius_m=wheel_radius)
    result["drivebase"] = {
        "name": name,
        "wheel_radius_m": wheel_radius,
        "source": str(paths["session"]),
    }
    write_poses_csv(pose_rows, str(paths["poses"]))

    paths["fit"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["fit"], "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    summary = summarize_fit(result)
    summary["name"] = slugify(name, "robot")
    summary["path"] = str(paths["fit"])
    summary["problems"] = problems
    return summary


def start_collect(server, payload):
    """Push the calibration suite to the hub and capture the session."""
    name = slugify(payload.get("name") or "robot", "robot")
    connection = str(payload.get("connection") or "ble")
    if connection not in ("ble", "usb"):
        raise ValueError("The connection must be ble or usb.")
    hub_name = payload.get("hub_name") or None

    job_dir = Path(tempfile.mkdtemp(prefix="overboard_collect_"))
    try:
        from overboard.models.robot_model import RobotConfig
        cfg_dict = payload.get("config") or payload.get("robot")
        if cfg_dict:
            RobotConfig(cfg_dict).sync_pybricks_robot_config()
    except Exception:
        pass
    session_path = job_dir / "session.csv"
    command = _deploy_command(str(COLLECTOR_SCRIPT),
                              "--conn", connection, "--out", str(session_path))
    if hub_name:
        command += ["--name", str(hub_name)]

    job = Job(server.next_job_id(), f"collect {name}", job_dir, kind="collect")
    job.result = {"name": name, "connection": connection,
                  "session_path": str(session_path), "progress": None}
    _register_job(server, job)

    def run():
        try:
            proc = subprocess.Popen(
                command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                **_process_group_kwargs())
            job.proc = proc
            if job.cancel_requested:
                _terminate_tree(proc)
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
            # Populate session_path before flipping status, so a poll that sees
            # the job finished also sees the result (KI-046.1).
            try:
                if session_path.is_file():
                    try:
                        paths = calibration_paths(name)
                        paths["session"].parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(str(session_path), str(paths["session"]))
                        job.result["session_path"] = str(paths["session"])
                        if job.exit_code != 0:
                            job.log("The link dropped; the session that arrived "
                                    "was kept.")
                    except OSError as exc:
                        job.log(f"Could not save the session: {exc}")
            finally:
                if job.status == "running":
                    if job.cancel_requested:
                        job.status = "cancelled"
                    else:
                        job.status = "ok" if job.exit_code == 0 else "failed"
                if job.status == "failed" and not job.error:
                    job.error = job.lines[-1] if job.lines else "The collection failed."

    thread = threading.Thread(target=run, daemon=True, name=f"Collect{job.id}")
    thread.start()
    return job


def trajectory_limits(config_path, samples):
    """The limits the studio draws the trajectory against.

    The force limit is the same worst-case curve the validator uses, so a
    plot that touches the limit means the validator would too.
    """
    from overboard.models.robot_model import RobotConfig
    config = _read_json(config_path) or {}
    try:
        robot = RobotConfig(config)
    except Exception:
        return {}
    speed_limit = robot.v_max_rad_s * robot.wheel_radius * robot.speed_headroom
    force_limit = [robot.get_max_force_at_velocity_robust(s.get("vl", 0.0))
                   for s in samples]
    return {
        "speed_limit_m_s": speed_limit,
        "force_limit_n": force_limit,
    }


def _git(args, cwd):
    """Run git in a directory. Returns (ok, output)."""
    try:
        proc = subprocess.run(["git", "-C", str(cwd)] + list(args),
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


# The commits are made with a fixed identity, so the studio does not need a
# git configuration on the machine it runs on.
_GIT_IDENTITY = ["-c", "user.name=OVERBOARD Studio",
                 "-c", "user.email=studio@localhost",
                 "-c", "commit.gpgsign=false"]


def ensure_mission_repo():
    """Make the missions folder its own git repository, so every save is a commit.

    The folder usually sits inside the project's own repository, so the check
    is for a repository *of this folder* and not for one that merely contains
    it: the missions have their own history, and the project does not get
    commits the operator did not ask for.
    """
    MISSIONS_DIR.mkdir(exist_ok=True)
    ok, top = _git(["rev-parse", "--show-toplevel"], MISSIONS_DIR)
    if ok:
        try:
            if Path(top.strip()).resolve() == MISSIONS_DIR.resolve():
                return True
        except OSError:
            pass
    ok, _ = _git(["init", "-q"], MISSIONS_DIR)
    if not ok:
        return False
    _git(["add", "-A"], MISSIONS_DIR)
    _git(_GIT_IDENTITY + ["commit", "-q", "-m", "Missions folder"], MISSIONS_DIR)
    return True


def commit_mission(name, message):
    """Commit a saved mission. Returns (commit id, error text).

    Git is tried twice: a locked index or a busy machine is transient, and a
    lost commit is a lost history. The mission file is saved either way, so
    a failure is reported on the console and in the response, not raised.
    """
    commit, error = "", ""
    for attempt in range(2):
        commit, error = _commit_mission_once(name, message)
        if commit:
            return commit, ""
        if attempt == 0:
            time.sleep(0.3)
    print(f"Studio could not commit the mission '{name}': {error.strip()}")
    return None, error.strip()


def _commit_mission_once(name, message):
    """One attempt at the commit. Returns (commit id, error text)."""
    if not ensure_mission_repo():
        return None, "the missions folder is not a git repository"
    ok, output = _git(["add", "--", f"{name}.json"], MISSIONS_DIR)
    if not ok:
        return None, output
    ok, output = _git(_GIT_IDENTITY + ["commit", "-q", "-m", message], MISSIONS_DIR)
    if not ok:
        # Nothing changed, or git is unhappy: the file is still saved.
        return None, output
    ok, head = _git(["rev-parse", "HEAD"], MISSIONS_DIR)
    if not ok:
        return None, head
    return head.strip(), ""


def mission_history(name):
    """The commits that touched a mission, newest first."""
    if not (MISSIONS_DIR / f"{name}.json").is_file():
        return []
    ok, output = _git(
        ["log", "--format=%H%x1f%s%x1f%cI", "--", f"{name}.json"], MISSIONS_DIR)
    if not ok:
        return []
    entries = []
    for line in output.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3:
            entries.append({"sha": parts[0], "message": parts[1], "when": parts[2]})
    return entries


def restore_mission(name, sha):
    """Put a mission back to an earlier version. Returns the waypoints."""
    # The path is relative to the repository's root, which is the missions
    # folder itself.
    ok, content = _git(["show", f"{sha}:./{name}.json"], MISSIONS_DIR)
    if not ok:
        raise ValueError("That version cannot be read.")
    path = MISSIONS_DIR / f"{name}.json"
    # Parse before writing: a blob that is not a mission must not reach the
    # disk, or the mission silently disappears from the list (KI-039).
    data = json.loads(content)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return data


def read_robot_files():
    """What the robot's own files say: its event actions and its geometry.

    pybricks_code/ is firmware: it imports pybricks.* and cannot be imported
    on the host, so the few values the studio needs are read as text.
    """
    events = []
    geometry = {}

    try:
        runner = (PYBRICKS_DIR / "hub_runner.py").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        runner = ""
    block = re.search(r"MISSION_EVENTS\s*=\s*\{(.*?)\n\s*\}", runner, re.S)
    if block:
        events = re.findall(r"[\"']([^\"']+)[\"']\s*:", block.group(1))

    try:
        config = (PYBRICKS_DIR / "robot_config.py").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        config = ""
    for name, value in re.findall(
            r"^\s*(WHEEL_DIAMETER_MM|AXLE_TRACK_MM)\s*=\s*([0-9.]+)",
            config, re.M):
        geometry[name] = float(value)

    return {"events": events, "geometry": geometry}


def artifact_path(job, kind):
    """Map an artifact kind to a file inside the job directory."""
    names = {
        "traj": f"{job.name}.traj",
        "controller": f"{job.name}_controller.json",
        "python": f"{job.name}.py",
        "stats": f"{job.name}_stats.json",
        "session": "session.csv",
        "report": f"{job.name}_report.html",
    }
    if kind not in names:
        return None
    return job.output_dir / names[kind]


def read_session_poses(path):
    """The measured poses and event names in a captured hub session."""
    from overboard.sysid.fit import poses_from_hub_session

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return [], []
    rows, events = poses_from_hub_session(lines)
    poses = [{"x_m": row["x_m"], "y_m": row["y_m"],
              "heading_deg": row["heading_deg"]} for row in rows]
    return poses, events


def compare_with_plan(traj_path, session_path):
    """The plan against what the robot did, as sysid_compare reports it.

    The hub's pose is in the robot's own frame - the origin is where the robot
    physically sat when it started, which is the plan's first waypoint. The
    plan is in the field frame, so the measured pose is rotated by the plan's
    first heading and moved by its first position before the two are compared.
    """
    from overboard.sysid.compare import compare
    from overboard.sysid.fit import write_poses_csv

    poses, _ = read_session_poses(session_path)
    if not poses:
        return None

    document = _read_json(traj_path) or {}
    samples = document.get("trajectory", {}).get("samples", [])
    if not samples:
        return None
    first = samples[0]
    x0 = float(first.get("x", 0.0))
    y0 = float(first.get("y", 0.0))
    heading0 = float(first.get("heading", 0.0) or 0.0)
    cos = math.cos(heading0)
    sin = math.sin(heading0)

    rows = []
    for index, pose in enumerate(poses):
        xm = float(pose["x_m"])
        ym = float(pose["y_m"])
        rows.append({
            "run_id": "hub",
            "stop": index,
            "x_m": x0 + xm * cos - ym * sin,
            "y_m": y0 + xm * sin + ym * cos,
            "heading_deg": float(pose["heading_deg"]) + math.degrees(heading0),
        })

    with tempfile.TemporaryDirectory() as tmp:
        poses_csv = os.path.join(tmp, "poses.csv")
        write_poses_csv(rows, poses_csv)
        try:
            return compare(str(traj_path), poses_csv)
        except (OSError, ValueError) as exc:
            return {"error": str(exc)}


def _error_headline(lines):
    """The one line that says what went wrong.

    A tool that fails prints "error: ..." first and the detail after it, so
    the first such line is the headline and the last line is the detail.
    The operator reads the headline.
    """
    for line in lines:
        if line.startswith("error:"):
            return line[len("error:"):].strip()
    for line in reversed(lines):
        if line.strip():
            return line
    return "The push failed."


def start_push(server, payload):
    """Push an already-solved trajectory to the hub."""
    source = server.get_job(str(payload.get("job_id") or ""))
    if source is None:
        raise ValueError("Unknown job.")
    traj = artifact_path(source, "traj")
    if traj is None or not traj.is_file():
        # The export gate did not pass, so there is nothing safe to push.
        raise ValueError("That solve produced no trajectory to push.")

    # Only a validated trajectory goes to the robot. The browser checked
    # this too, but the browser is not the gate: a direct POST to this
    # endpoint would otherwise run an unchecked plan on the mat (KI-036).
    verdict = _validation_verdict(source)
    if verdict is not True:
        raise ValueError(
            "That trajectory was never validated against your robot. "
            "Validate it in the Solve tab and solve again before pushing it."
        )

    connection = str(payload.get("connection") or "ble")
    if connection not in ("ble", "usb"):
        raise ValueError("The connection must be ble or usb.")
    hub_name = payload.get("hub_name") or None

    job_dir = Path(tempfile.mkdtemp(prefix="overboard_push_"))
    try:
        from overboard.models.robot_model import RobotConfig
        data = _read_json(traj) or {}
        cfg_dict = data.get("config") or data.get("robot")
        if cfg_dict:
            RobotConfig(cfg_dict).sync_pybricks_robot_config()
    except Exception:
        pass
    session_path = job_dir / "session.csv"
    command = _deploy_command("--trajectory", str(traj), "--conn", connection,
                              "--out", str(session_path))
    if hub_name:
        command += ["--name", str(hub_name)]

    job = Job(server.next_job_id(), source.name, job_dir, kind="push")
    job.result = {"connection": connection, "hub_name": hub_name or "",
                  "poses": [], "events": [], "report": None}
    _register_job(server, job)

    def run():
        try:
            proc = subprocess.Popen(
                command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                **_process_group_kwargs())
            job.proc = proc
            if job.cancel_requested:
                _terminate_tree(proc)
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
            # The measured pose and comparison are populated before status flips,
            # so a poll that sees the job finished also sees the result (KI-046.1).
            # Wrapped in try...finally so a post-processing failure cannot leave
            # the job "running" and block the studio.
            try:
                if session_path.is_file():
                    try:
                        poses, events = read_session_poses(session_path)
                        job.result["poses"] = poses
                        job.result["events"] = events
                        job.result["report"] = compare_with_plan(traj, session_path)
                    except Exception as exc:  # noqa: BLE001 - a comparison is not worth a failure
                        job.log(f"Could not compare the run with the plan: {exc}")
                    # The session belongs next to the mission it came from.
                    try:
                        MISSIONS_DIR.mkdir(exist_ok=True)
                        target = MISSIONS_DIR / f"{source.name}.session.csv"
                        shutil.copyfile(str(session_path), str(target))
                        job.result["session_path"] = str(target)
                    except OSError as exc:
                        job.log(f"Could not save the session next to the mission: {exc}")
            finally:
                if job.status == "running":
                    if job.cancel_requested:
                        job.status = "cancelled"
                    else:
                        job.status = "ok" if job.exit_code == 0 else "failed"
                if job.status == "failed" and not job.error:
                    job.error = _error_headline(job.lines)

    thread = threading.Thread(target=run, daemon=True, name=f"Push{job.id}")
    thread.start()
    return job


class StudioHandler(BaseHTTPRequestHandler):
    """Routes for the studio UI."""

    server_version = "OVERBOARD"
    protocol_version = "HTTP/1.1"

    # --- plumbing -------------------------------------------------------
    def log_message(self, fmt, *args):
        pass  # keep the console readable

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            # The browser closed the connection (a reload, a cancelled fetch).
            # That is normal, not a server error.
            self.close_connection = True

    def _host_allowed(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost", "::1", "")

    def _origin_allowed(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlparse(origin)
        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            return False
        # The port has to be the studio's own. Any other web app on this
        # machine sends a localhost Origin, so a hostname check alone let a
        # page the operator happened to have open drive the studio - and
        # through it the robot (KI-036).
        try:
            port = parsed.port
        except ValueError:
            return False
        if port is None:
            return False
        return port == self.server.server_address[1]

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
            if path == "/api/robot":
                return self._json(200, read_robot_files())
            if path == "/api/robot/models":
                return self._json(200, {"models": list_fits()})
            if path == "/api/missions/history":
                return self._mission_history()
            if path == "/api/fields":
                return self._json(200, field_payload())
            if path.startswith("/mats/"):
                return self._serve_mat(path)
            if path == "/api/calibration/sheet":
                return self._sheet()
            if path == "/api/jobs/current":
                job = self.server.active_job()
                if job is None:
                    return self._json(200, {"job": None})
                return self._json(200, {"job": job.snapshot()})
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
            if path == "/api/push":
                return self._push()
            if path == "/api/robot/model/load":
                return self._load_model()
            if path == "/api/robot/model/store":
                return self._store_model()
            if path == "/api/calibration/collect":
                return self._collect()
            if path == "/api/calibration/sheet":
                return self._save_sheet()
            if path == "/api/calibration/fit":
                return self._fit()
            if path == "/api/field/mat":
                return self._field_mat()
            if path == "/api/missions":
                return self._save_mission()
            if path == "/api/missions/restore":
                return self._restore_mission()
            if path.startswith("/api/jobs/") and path.endswith("/report"):
                job = self.server.get_job(path.split("/")[3])
                if job is None:
                    return self._error(404, "Unknown job.")
                return self._write_report(job)
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

    def _serve_mat(self, path):
        """A mat picture the operator fetched."""
        name = os.path.basename(path)
        if not name.endswith(".png"):
            return self._error(404, "Not found.")
        target = MATS_DIR / name
        if not target.is_file():
            return self._error(404, "That mat has not been fetched yet.")
        with open(target, "rb") as handle:
            return self._send(200, handle.read(), content_type="image/png")

    def _field_mat(self):
        """Fetch a season's mat picture and hand the browser its URL."""
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            payload = {}
        season = str(payload.get("season") or "").strip()
        if not season:
            return self._error(400, "A season is needed.")
        try:
            return self._json(200, download_field_mat(season))
        except ValueError as exc:
            return self._error(400, str(exc))

    def _health(self):
        from overboard.diagnostics import check_environment, summarize
        checks = check_environment(ports=())
        ok, warn, error = summarize(checks)
        packages = {c["name"]: c["status"] for c in checks
                    if c["name"] in ("jax", "immrax", "casadi", "numpy")}
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
            if job.kind == "collect":
                # The session grows as the hub prints, so the checklist is
                # recomputed on every poll.
                payload["progress"] = collect_progress(
                    job.result.get("session_path", ""))
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
        if len(parts) == 4 and parts[3] == "report":
            return self._serve_report(job)
        if len(parts) == 5 and parts[3] == "artifact":
            return self._download(job, parts[4])
        return self._error(404, "Not found.")

    def _artifacts(self, job):
        kinds = ("session",) if job.kind == "push" else (
            "traj", "controller", "python", "stats", "report")
        found = []
        for kind in kinds:
            path = artifact_path(job, kind)
            if path and path.is_file():
                found.append({"kind": kind, "name": path.name,
                              "size": path.stat().st_size})
        return found

    def _serve_report(self, job):
        """The mission report, as a page the operator can print."""
        path = artifact_path(job, "report")
        if not path or not path.is_file():
            return self._error(404, "No report was written.")
        with open(path, "rb") as handle:
            body = handle.read()
        return self._send(200, body, content_type="text/html; charset=utf-8")

    def _write_report(self, job):
        """Write the report the browser built, next to the trajectory."""
        payload = self._read_body()
        html = payload.get("html")
        if not isinstance(html, str) or not html.strip():
            raise ValueError("The report is empty.")
        path = artifact_path(job, "report")
        if not path:
            return self._error(404, "No report can be written for this job.")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return self._json(200, {"name": path.name, "path": str(path),
                                "url": f"/api/jobs/{job.id}/report"})

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
        try:
            job = start_job(self.server, payload)
        except BusyError as exc:
            return self._error(409, str(exc))
        return self._json(202, job.snapshot())

    def _push(self):
        if self.server.active_job() is not None:
            return self._error(409, "Something is already running. Cancel it first.")
        payload = self._read_body()
        try:
            job = start_push(self.server, payload)
        except BusyError as exc:
            return self._error(409, str(exc))
        return self._json(202, job.snapshot())

    def _load_model(self):
        """Take a fitted model from somewhere and keep it with the robot."""
        payload = self._read_body()
        path = payload.get("path")
        if not path:
            return self._error(400, "Give the path of a sysid_fit.json.")
        source = Path(path)
        if not source.is_absolute():
            source = REPO_ROOT / source
        try:
            source = source.resolve()
        except OSError:
            return self._error(400, "That path cannot be read.")
        # Only the project folder and the calibration folder hold models.
        # Without this, any page on this machine could ask the studio to
        # read an arbitrary file and copy it into the calibration folder
        # (KI-036).
        roots = (REPO_ROOT.resolve(), CALIBRATION_DIR.resolve())
        if not any(source == root or root in source.parents for root in roots):
            return self._error(400, "That path is outside the project folder.")
        try:
            fit = _read_json(source)
        except OSError as exc:
            return self._error(400, f"Could not read '{source}': {exc}")
        if not isinstance(fit, dict) or "actuator" not in fit:
            return self._error(400, f"'{source}' has no actuator model. "
                                  "It does not look like a sysid_fit.json.")
        name = payload.get("name") or source.stem.replace(".sysid_fit", "")
        summary = store_fit(name, fit)
        return self._json(200, summary)

    def _store_model(self):
        """Store a fitted model the browser uploaded."""
        payload = self._read_body()
        fit = payload.get("fit")
        name = payload.get("name") or "robot"
        if not isinstance(fit, dict):
            return self._error(400, "That file has no actuator model. "
                                  "It does not look like a sysid_fit.json.")
        summary = store_fit(name, fit)
        return self._json(200, summary)

    def _collect(self):
        """Push the calibration suite to the hub and capture the session."""
        if self.server.active_job() is not None:
            return self._error(409, "Something is already running. Cancel it first.")
        payload = self._read_body()
        try:
            job = start_collect(self.server, payload)
        except BusyError as exc:
            return self._error(409, str(exc))
        return self._json(202, job.snapshot())

    def _sheet(self):
        """The measurement rows the session needs, and what is filled in."""
        query = urlparse(self.path).query
        params = parse_qs(query)
        name = (params.get("name") or ["robot"])[0]
        paths = calibration_paths(name)
        session = paths["session"]
        if not session.is_file():
            # No session yet is not an error. The studio asks for the sheet
            # every time it loads, and a 404 there is noise in the console
            # and in the operator's face. The empty sheet says the same
            # thing: there is nothing to fill in yet.
            return self._json(200, {"rows": [], "missing": 0, "total": 0,
                                    "name": name, "session": False})
        rows = _sheet_rows(_session_runs(session))
        filled = read_sheet_rows(paths["sheet"], rows)
        missing = count_missing(filled)
        return self._json(200, {"rows": filled, "missing": missing,
                                "total": len(filled), "name": name,
                                "session": True})

    def _fit(self):
        """Fit the model from the session and the filled sheet."""
        payload = self._read_body()
        name = payload.get("name") or "robot"
        try:
            summary = fit_from_sheet(name)
        except ValueError as exc:
            return self._error(400, str(exc))
        return self._json(200, summary)

    def _save_sheet(self):
        """Save the operator's measurements."""
        payload = self._read_body()
        name = payload.get("name") or "robot"
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return self._error(400, "The sheet has no rows.")
        paths = calibration_paths(name)
        try:
            save_sheet(paths["sheet"], rows)
        except OSError as exc:
            return self._error(500, f"Could not save the sheet: {exc}")
        # The same rows the GET returns: every measurement the session
        # needs, with whatever is filled in. The rows the browser sent are
        # not the whole sheet.
        if paths["session"].is_file():
            shown = read_sheet_rows(
                paths["sheet"], _sheet_rows(_session_runs(paths["session"])))
        else:
            shown = read_sheet_rows(paths["sheet"], rows)
        missing = count_missing(shown)
        return self._json(200, {"rows": shown, "missing": missing,
                                "total": len(shown), "name": name})

    def _cancel_job(self, job_id):
        job = self.server.get_job(job_id)
        if job is None:
            return self._error(404, "Unknown job.")
        if job.status == "running":
            # Ask for the cancel and kill the whole tree: the candidates
            # are solved in worker processes, and killing only the parent
            # left them running at full speed (KI-037).
            #
            # The job stays "running" until the process is really gone, so
            # the studio does not accept a second solve while the first
            # one is still finishing.
            job.cancel_requested = True
            _terminate_tree(job.proc)
        return self._json(200, job.snapshot())

    def _save_mission(self):
        payload = self._read_body()
        waypoints = payload.get("waypoints")
        if not isinstance(waypoints, list) or len(waypoints) < 2:
            raise ValueError("A mission needs at least two waypoints.")
        name = slugify(payload.get("name"))
        MISSIONS_DIR.mkdir(exist_ok=True)
        # The repository is made before the file is written, so its first
        # commit is the folder itself and the mission gets a commit of its own.
        ensure_mission_repo()
        path = MISSIONS_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(waypoints, handle, indent=1)
        # Every save is a commit, so the mission has a history.
        commit, commit_error = commit_mission(
            name, payload.get("message") or f"Mission {name}")
        response = {"name": name, "path": str(path), "commit": commit,
                    "history": mission_history(name)}
        if commit_error:
            # The mission is saved, but it is not in the history. The studio
            # says so instead of pretending the save is a commit.
            response["commit_error"] = commit_error
        return self._json(200, response)

    def _mission_history(self):
        query = urlparse(self.path).query
        params = parse_qs(query)
        name = slugify((params.get("name") or ["robot"])[0])
        return self._json(200, {"name": name, "history": mission_history(name)})

    def _restore_mission(self):
        payload = self._read_body()
        name = slugify(payload.get("name"))
        sha = payload.get("sha")
        if not sha:
            raise ValueError("Give the commit to restore.")
        waypoints = restore_mission(name, sha)
        return self._json(200, {"name": name, "waypoints": waypoints,
                                "history": mission_history(name)})


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


def build_initial(config_path=None, waypoints_path=None, auto_solve=False, options=None):
    """The mission and robot the studio opens with."""
    initial = {"config": None, "waypoints": None, "config_name": None,
               "auto_solve": bool(auto_solve), "options": dict(options or {})}
    # A fitted plant is only used by the validator, so it turns validation on.
    if initial["options"].get("sysid_fit") and not initial["options"].get("validate"):
        initial["options"]["validate"] = True

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


def run_studio(config_path=None, waypoints_path=None, port=8000, presets=None,
               auto_solve=False, options=None):
    """Run the studio until the user stops it."""
    initial = build_initial(config_path=config_path, waypoints_path=waypoints_path,
                            auto_solve=auto_solve, options=options)
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
