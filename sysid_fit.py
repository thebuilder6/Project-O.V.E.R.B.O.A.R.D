"""
Fit a first-order digital-twin model from ``pybricks_code/sysid_digital_twin.py``
console logs.

This is host-side code. It reads the CSV blocks emitted by the robot logger,
fits straight/turn actuator response, odometry scale, and a bounded braking
slip estimate, then writes a JSON file consumed by ``main.py --sysid-fit``.

Usage:
    python sysid_fit.py path/to/session.csv --poses path/to/poses.csv \
        --profile examples/drivebase_profile.example.json --out sysid_fit.json

A minimal drivebase profile only needs ``wheel_radius_m``; ``track_width_m`` is
optional metadata used by reports. ``--robot-config`` remains accepted for an
existing full robot config, and ``--wheel-radius`` overrides either source.

``poses.csv`` is optional. Its columns are:
    run_id,stop,x_m,y_m,heading_deg
``stop`` defaults to 0. For a single-pose run the pose is measured from the
start origin. For ``multi_stop`` runs, include one pose per logged stop in order.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _finish_run(run, header, rows):
    if run is None or header is None:
        return
    run["columns"] = {name.strip(): i for i, name in enumerate(header)}
    if rows:
        run["data"] = np.asarray(rows, dtype=float)
    else:
        run["data"] = np.empty((0, len(header)), dtype=float)


def parse_session(text):
    """Parse one or more TEST_START ... TEST_END blocks into run dictionaries."""
    runs = []
    current = None
    header = None
    rows = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("TEST_START"):
            _finish_run(current, header, rows)
            name = line[len("TEST_START"):].strip()
            current = {"name": name, "meta": {}, "columns": {}, "data": np.empty((0, 0))}
            runs.append(current)
            header = None
            rows = []
            continue

        if line.startswith("META,"):
            if current is None:
                continue
            parts = line.split(",", 2)
            if len(parts) >= 3:
                current["meta"][parts[1].strip()] = parts[2].strip()
            elif len(parts) == 2:
                current["meta"][parts[1].strip()] = ""
            continue

        if line == "DATA":
            header = None
            rows = []
            continue

        if line.startswith("TEST_END"):
            _finish_run(current, header, rows)
            header = None
            rows = []
            continue

        if current is None:
            continue

        if header is None:
            # The logger prints DATA, then the real header. Ignore free-form
            # status lines such as STOP_AT or ramsete_fast's console output.
            if line.split(",", 1)[0].strip() == "t_ms":
                header = line.split(",")
            continue

        parts = line.split(",")
        try:
            values = [float(part) for part in parts]
        except ValueError:
            # STOP_AT / FINAL_STATE status lines are deliberately ignored.
            continue
        if len(values) == len(header):
            rows.append(values)

    _finish_run(current, header, rows)
    return [run for run in runs if run["columns"]]


def _collect_files(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.csv"))))
        else:
            files.append(path)
    return files


def load_runs(paths):
    """Load all session files, prefixing run IDs when more than one is used."""
    files = _collect_files(paths)
    loaded = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            loaded.append((path, parse_session(handle.read())))

    session_count = sum(1 for _, file_runs in loaded if file_runs)
    runs = []
    for path, file_runs in loaded:
        prefix = os.path.splitext(os.path.basename(path))[0]
        for run in file_runs:
            if session_count > 1 and run["meta"].get("run_id") is not None:
                run["meta"]["run_id"] = prefix + ":" + run["meta"]["run_id"]
            run["meta"]["source_file"] = path
            runs.append(run)
    return files, runs


def load_poses(path):
    """Load run_id -> [{stop, x_m, y_m, heading_deg}, ...]."""
    poses = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_id = (row.get("run_id") or "").strip()
            if not run_id:
                continue
            stop = int(float(row.get("stop") or 0))
            distance = (row.get("distance_m") or "").strip()
            if distance:
                x_m = float(distance)
                y_m = 0.0
                heading_deg = 0.0
            else:
                x_m = float(row.get("x_m") or 0.0)
                y_m = float(row.get("y_m") or 0.0)
                heading_deg = float(row.get("heading_deg") or 0.0)
            poses.setdefault(run_id, []).append({
                "stop": stop,
                "x_m": x_m,
                "y_m": y_m,
                "heading_deg": heading_deg,
            })
    for entries in poses.values():
        entries.sort(key=lambda entry: entry["stop"])
    return poses


# The hub prints the pose of a run as "POSE,x_m,y_m,heading_deg" and every
# exported event as "EVENT,name". These prefixes are a contract: the runner in
# pybricks_code/hub_runner.py and this parser have to agree.
HUB_POSE_PREFIX = "POSE,"
HUB_EVENT_PREFIX = "EVENT,"

POSES_CSV_FIELDS = ["run_id", "stop", "x_m", "y_m", "heading_deg"]


def poses_from_hub_session(lines, run_id="hub"):
    """Turn a captured hub session into measured-pose rows and event names.

    ``lines`` is the hub console output, as written by ``hub_deploy.py`` or
    ``main.py --push-hub --hub-log``. The fitter and the comparator read
    measured poses as CSV rows with run_id, stop, x_m, y_m, heading_deg, so
    the session is converted here instead of by hand.

    A pose belongs to the test that is open when it is printed, so it takes
    the run id of that test's ``META,run_id`` line. The ``run_id`` argument
    is only the fallback for a session without META lines, such as the one a
    pushed trajectory prints.

    Returns (rows, events). The rows are in the order the hub printed them,
    with ``stop`` counting the poses of each run.
    """
    rows = []
    events = []
    current_run = None
    stops = {}
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        text = line.strip()
        if text.startswith("META,"):
            parts = text.split(",", 2)
            if len(parts) >= 3 and parts[1].strip() == "run_id":
                current_run = parts[2].strip()
            continue
        if text.startswith(HUB_POSE_PREFIX):
            parts = text.split(",")
            if len(parts) < 4:
                continue
            try:
                x_m = float(parts[1])
                y_m = float(parts[2])
                heading_deg = float(parts[3])
            except ValueError:
                continue
            owner = current_run if current_run else run_id
            stop = stops.get(owner, 0)
            stops[owner] = stop + 1
            rows.append({
                "run_id": owner,
                "stop": stop,
                "x_m": x_m,
                "y_m": y_m,
                "heading_deg": heading_deg,
            })
        elif text.startswith(HUB_EVENT_PREFIX):
            events.append(text[len(HUB_EVENT_PREFIX):].strip())
    return rows, events


# The measurement sheet is what the operator fills in by hand. It is keyed by
# the name of the test and the repeat of it, so it does not depend on the
# order the tests ran in:
#
#   test,repeat,stop,x_mm,y_mm,turn_right_deg
#   odometry_straight,0,0,998,2,0
#
# x is forward from the start pose, y is to the left, and the turn is how far
# the robot turned to the right. The heading the fitter wants is
# counter-clockwise positive, so the turn is negated on the way in.
SHEET_FIELDS = ["test", "repeat", "stop", "x_mm", "y_mm", "turn_right_deg"]


def session_tests(runs):
    """Map every run to its run id, keyed by its test name and repeat."""
    index = {}
    for run in runs:
        name = (run["meta"].get("test") or run["name"] or "").strip()
        if not name:
            continue
        key = (name, _run_repeat(run))
        try:
            poses = int(run["meta"].get("poses", 0))
        except (TypeError, ValueError):
            poses = 0
        index[key] = (run["meta"].get("run_id"), poses)
    return index


def write_sheet(runs, path):
    """Write a blank measurement sheet for the tests that need one."""
    rows = []
    for (name, repeat), (run_id, poses) in sorted(session_tests(runs).items()):
        for stop in range(poses):
            rows.append({"test": name, "repeat": repeat, "stop": stop})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SHEET_FIELDS})
    return path


def read_sheet(path):
    """Read the operator's measurement sheet.

    Rows the operator left empty are skipped, so a sheet can be filled in
    part by part.
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            name = (raw.get("test") or "").strip()
            if not name:
                continue
            try:
                repeat = int((raw.get("repeat") or "0").strip())
                stop = int((raw.get("stop") or "0").strip())
                x_mm = float((raw.get("x_mm") or "").strip())
                y_mm = float((raw.get("y_mm") or "").strip())
                turn_deg = float((raw.get("turn_right_deg") or "").strip())
            except (TypeError, ValueError):
                continue
            rows.append({
                "test": name,
                "repeat": repeat,
                "stop": stop,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "turn_right_deg": turn_deg,
            })
    return rows


def sheet_to_poses(rows, runs):
    """Turn the operator's sheet into measured-pose rows for the fitter.

    Returns (pose_rows, problems). A problem is a line of plain text for the
    operator: a row for a test that did not run, or a test that needs a
    measurement and did not get one.
    """
    index = session_tests(runs)
    problems = []
    poses = []
    seen = {}
    for row in rows:
        name = row["test"]
        key = (name, row["repeat"])
        if key not in index:
            problems.append(
                f"the sheet has '{name}' repeat {row['repeat']} but the "
                f"session does not have that test and repeat")
            continue
        run_id, _poses = index[key]
        seen_key = (name, row["repeat"], row["stop"])
        if seen_key in seen:
            problems.append(
                f"the sheet has '{name}' repeat {row['repeat']} stop "
                f"{row['stop']} twice")
            continue
        seen[seen_key] = True
        poses.append({
            "run_id": run_id,
            "stop": row["stop"],
            "x_m": row["x_mm"] / 1000.0,
            "y_m": row["y_mm"] / 1000.0,
            # The sheet counts a turn to the right; the trajectory counts a
            # turn to the left.
            "heading_deg": -row["turn_right_deg"],
        })

    measured = {(row["test"], row["repeat"], row["stop"]) for row in rows}
    for (name, repeat), (_run_id, count) in index.items():
        if count <= 0:
            continue
        for stop in range(count):
            if (name, repeat, stop) not in measured:
                problems.append(
                    f"'{name}' repeat {repeat} stop {stop} was not measured, "
                    f"so its scale will not be calibrated")
    return poses, problems


def write_poses_csv(rows, path):
    """Write measured-pose rows in the format load_poses() reads."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSES_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in POSES_CSV_FIELDS})
    return path


def _write_sheet(args, parser):
    """CLI mode: write a blank measurement sheet for the session."""
    files, runs = load_runs(args.logs)
    if not runs:
        print("No TEST_START blocks found.")
        return 1
    write_sheet(runs, args.write_sheet)
    tests = session_tests(runs)
    wanted = sum(1 for _run_id, count in tests.values() if count > 0)
    print(f"Wrote a measurement sheet for {wanted} test run(s) to "
          f"{args.write_sheet}.")
    print("Fill in x_mm, y_mm and turn_right_deg, then run:")
    print(f"  python sysid_fit.py {' '.join(args.logs)} --from-sheet "
          f"{args.write_sheet} --out-poses poses.csv")
    return 0


def _convert_sheet(args, parser):
    """CLI mode: turn the operator's sheet into a measured-pose CSV."""
    if not args.out_poses:
        parser.error("--from-sheet needs --out-poses")
    _files, runs = load_runs(args.logs)
    if not runs:
        print("No TEST_START blocks found.")
        return 1
    try:
        rows = read_sheet(args.from_sheet)
    except OSError as exc:
        print(f"error: cannot read {args.from_sheet}: {exc}")
        return 1

    poses, problems = sheet_to_poses(rows, runs)
    for problem in problems:
        print(f"warning: {problem}.")
    if not poses:
        print(f"No measurements in {args.from_sheet}.")
        return 1

    write_poses_csv(poses, args.out_poses)
    print(f"Wrote {len(poses)} measured pose(s) to {args.out_poses}.")
    return 0


def _convert_hub_session(args, parser):
    """CLI mode: turn a captured hub session into a measured-pose CSV."""
    if not args.out_poses:
        parser.error("--from-hub-session needs --out-poses")
    try:
        with open(args.from_hub_session, "r", encoding="utf-8",
                  errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        print(f"error: cannot read {args.from_hub_session}: {exc}")
        return 1

    rows, events = poses_from_hub_session(lines, run_id=args.run_id)
    if not rows:
        print(f"No POSE lines in {args.from_hub_session}.")
        return 1

    write_poses_csv(rows, args.out_poses)
    print(
        f"Wrote {len(rows)} measured pose(s) to {args.out_poses} "
        f"(run_id={args.run_id})."
    )
    if events:
        print("Events: " + ", ".join(events))
    return 0


# ---------------------------------------------------------------------------
# Column helpers and first-order fitting
# ---------------------------------------------------------------------------

def _column(run, name):
    index = run["columns"].get(name)
    if index is None or run["data"].size == 0:
        return None
    return run["data"][:, index]


def _last(run, name):
    values = _column(run, name)
    if values is None or len(values) == 0:
        return None
    return float(values[-1])


def _find_steps(times, command, threshold):
    changes = np.where(np.abs(np.diff(command)) > threshold)[0] + 1
    edges = np.concatenate(([0], changes, [len(command)]))
    return [(int(edges[i]), int(edges[i + 1]))
            for i in range(len(edges) - 1)
            if edges[i + 1] - edges[i] >= 10]


def _fit_first_order(times, measured, command):
    """Fit y(t) = yss + (y0 - yss) * exp(-(t - t0 - delay) / tau)."""
    if len(times) < 10:
        return None

    y0 = float(np.median(measured[:3]))
    tail_start = max(3, int(0.65 * len(measured)))
    yss = float(np.median(measured[tail_start:]))
    if abs(yss) < 1e-5:
        return None

    t0 = float(times[0])
    best_error = float("inf")
    best_delay = 0.0
    best_tau = 0.1

    for delay in np.arange(0.0, 0.121, 0.01):
        shifted = times - (t0 + delay)
        for tau in np.arange(0.01, 0.501, 0.01):
            model = yss + (y0 - yss) * np.exp(-np.maximum(shifted, 0.0) / tau)
            error = float(np.mean((measured - model) ** 2))
            if error < best_error:
                best_error = error
                best_delay = float(delay)
                best_tau = float(tau)

    gradient = np.abs(np.gradient(measured, times))
    return {
        "gain": yss / command,
        "tau_s": best_tau,
        "delay_s": best_delay,
        "accel_limit": float(np.percentile(gradient, 95)),
        "rmse": float(np.sqrt(best_error)),
    }


# The two-sided 95% t multipliers, for the degrees of freedom of a mean over
# n independent repeats. Beyond the table the normal value is close enough.
_T95 = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
}


def repeat_statistics(values):
    """The spread of one parameter over the repeats of a test.

    ``values`` is one value per repeat. With fewer than two repeats there is
    no interval, because a spread needs two points. The interval is a t
    interval on the mean: it covers the run-to-run variation, and it is
    deliberately wide, because three repeats is a small sample.
    """
    values = [float(v) for v in values]
    count = len(values)
    if count == 0:
        return {"n": 0}
    mean = float(np.mean(values))
    if count < 2:
        return {"n": count, "mean": mean, "std": None, "ci95": None}
    std = float(np.std(values, ddof=1))
    half = _T95.get(count, 1.96) * std / np.sqrt(count)
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95": [float(mean - half), float(mean + half)],
    }


def _between_runs(pairs):
    """The spread of one parameter over the repeats of a test.

    ``pairs`` is (repeat, value). The values of one repeat are pooled first,
    so a repeat that measured the parameter twice counts once.
    """
    groups = {}
    for repeat, value in pairs:
        groups.setdefault(repeat, []).append(float(value))
    return repeat_statistics([float(np.median(group))
                              for group in groups.values()])


def _summary(fits):
    if not fits:
        return {"calibrated": False, "n_steps": 0}

    def median(key):
        return float(np.median([fit[key] for fit in fits]))

    def minimum(key):
        return float(np.min([fit[key] for fit in fits]))

    def maximum(key):
        return float(np.max([fit[key] for fit in fits]))

    gains = [fit["gain"] for fit in fits]
    # One value per repeat, so the between-repeat spread is the honest
    # uncertainty. The steps inside a repeat are sequential and correlated,
    # so their spread is only the within-run precision.
    def per_repeat(key):
        return [(fit.get("repeat", 0), fit[key]) for fit in fits]

    summary = {
        "calibrated": True,
        "n_steps": len(fits),
        "n_runs": len({fit.get("repeat", 0) for fit in fits}),
        "gain": median("gain"),
        "gain_min": minimum("gain"),
        "gain_max": maximum("gain"),
        "gain_std": float(np.std(gains)),
        "tau_s": median("tau_s"),
        "tau_min": minimum("tau_s"),
        "tau_max": maximum("tau_s"),
        "delay_s": median("delay_s"),
        "delay_min": minimum("delay_s"),
        "delay_max": maximum("delay_s"),
        "accel_limit": median("accel_limit"),
        "rmse": median("rmse"),
    }
    for key, name in (("gain", "gain"), ("tau_s", "tau_s"),
                      ("delay_s", "delay_s")):
        stats = _between_runs(per_repeat(key))
        summary[f"{name}_between_runs"] = stats
        if stats.get("ci95"):
            summary[f"{name}_ci95"] = stats["ci95"]
    return summary



# ---------------------------------------------------------------------------
# Measured outputs from raw encoder / gyro rows
# ---------------------------------------------------------------------------

def _straight_measured(run, wheel_radius_m):
    left = _column(run, "l_deg")
    right = _column(run, "r_deg")
    times = _column(run, "t_ms")
    if left is None or right is None or times is None:
        return None, None
    times = times / 1000.0
    average_angle = 0.5 * (left + right) * np.pi / 180.0
    speed = np.gradient(average_angle, times) * wheel_radius_m
    return times, speed


def _turn_measured(run):
    gyro_z = _column(run, "gyro_z_rad_s")
    times = _column(run, "t_ms")
    if gyro_z is None or times is None:
        return None, None
    # Pybricks turn rate is positive clockwise; IMU z is positive CCW.
    return times / 1000.0, -gyro_z


def _add_measured_outputs(runs, wheel_radius_m):
    for run in runs:
        if "t_ms" not in run["columns"]:
            continue
        if "v_meas_m_s" not in run["columns"]:
            times, speed = _straight_measured(run, wheel_radius_m)
            if times is not None:
                run["v_meas_m_s"] = speed
        if "w_meas_rad_s" not in run["columns"]:
            times, turn_rate = _turn_measured(run)
            if times is not None:
                run["w_meas_rad_s"] = turn_rate


def _measured_column(run, name):
    if name in run and run[name] is not None:
        return run[name]
    return _column(run, name)


def _run_repeat(run):
    """The repeat number of a run. A session without the line is repeat 0."""
    try:
        return int(run["meta"].get("repeat", 0))
    except (TypeError, ValueError):
        return 0


def _step_fits_measured(runs, prefixes, command_column, measured_key,
                        command_scale=1.0, command_threshold=0.005):
    fits = []
    for run in runs:
        if not any(run["name"].startswith(prefix) for prefix in prefixes):
            continue
        times = _column(run, "t_ms")
        command = _column(run, command_column)
        measured = _measured_column(run, measured_key)
        if times is None or command is None or measured is None:
            continue

        times = times / 1000.0
        command = command * command_scale
        for start, end in _find_steps(times, command, command_threshold):
            segment_command = float(np.median(command[start:end]))
            if abs(segment_command) < command_threshold:
                continue
            if times[end - 1] - times[start] < 0.30:
                continue
            fit = _fit_first_order(
                times[start:end],
                measured[start:end],
                segment_command,
            )
            if fit is None:
                continue
            fit["run"] = run["name"]
            fit["command"] = segment_command
            # The repeat the run belongs to, so the fit can tell the
            # between-run spread from the within-run one.
            fit["repeat"] = _run_repeat(run)
            fits.append(fit)
    return fits


# ---------------------------------------------------------------------------
# Chirp validation
# ---------------------------------------------------------------------------

def _simulate_first_order(times, command, gain, tau_s, delay_s):
    delayed = np.interp(times - delay_s, times, command, left=command[0])
    model = np.empty_like(delayed)
    model[0] = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        alpha = 1.0 - np.exp(-dt / tau_s)
        model[i] = model[i - 1] + alpha * (gain * delayed[i - 1] - model[i - 1])
    return model


def _chirp_nrmse(runs, names, command_column, measured_key,
                 command_scale, summary):
    if not summary.get("calibrated"):
        return None

    errors = []
    for run in runs:
        if run["name"] not in names:
            continue
        times = _column(run, "t_ms")
        command = _column(run, command_column)
        measured = _measured_column(run, measured_key)
        if times is None or command is None or measured is None:
            continue
        times = times / 1000.0
        command = command * command_scale
        model = _simulate_first_order(
            times, command, summary["gain"], summary["tau_s"], summary["delay_s"]
        )
        span = float(np.max(measured) - np.min(measured))
        if span < 1e-6:
            continue
        errors.append(float(np.sqrt(np.mean((measured - model) ** 2)) / span))
    return float(np.median(errors)) if errors else None


# ---------------------------------------------------------------------------
# Backlash / low-speed gain
# ---------------------------------------------------------------------------

def _static_summary(runs):
    for run in runs:
        if run["name"] != "static":
            continue
        gyro_z = _column(run, "gyro_z_deg_s")
        ax = _column(run, "ax_mm_s2")
        ay = _column(run, "ay_mm_s2")
        if gyro_z is None:
            return {"calibrated": False}
        return {
            "calibrated": True,
            "gyro_z_bias_deg_s": float(np.mean(gyro_z)),
            "gyro_z_std_deg_s": float(np.std(gyro_z)),
            "ax_bias_m_s2": float(np.mean(ax) / 1000.0) if ax is not None else None,
            "ay_bias_m_s2": float(np.mean(ay) / 1000.0) if ay is not None else None,
        }
    return {"calibrated": False}


def _reverse_delay(run, measured_key, threshold=0.01):
    times = _column(run, "t_ms")
    command = _column(run, "v_cmd_mm_s")
    measured = _measured_column(run, measured_key)
    if times is None or command is None or measured is None:
        return None

    times = times / 1000.0
    command = command / 1000.0
    signs = np.sign(command)
    delays = []

    for index in np.where(np.diff(signs) != 0)[0]:
        target = signs[index + 1]
        if target == 0.0:
            continue
        for probe in range(index + 1, len(measured)):
            if np.sign(measured[probe]) == target and abs(measured[probe]) > threshold:
                delays.append(float(times[probe] - times[index]))
                break

    return float(np.median(delays)) if delays else None


# ---------------------------------------------------------------------------
# Odometry and slip from measured poses
# ---------------------------------------------------------------------------

def _pose_distance(pose):
    return float(np.hypot(pose["x_m"], pose["y_m"]))


def _fit_odometry(runs, poses, distance_scale_default=1.0):
    distance_candidates = []
    heading_candidates = []
    brake_pairs = []

    for run in runs:
        run_id = run["meta"].get("run_id")
        if run_id is None:
            continue
        entries = poses.get(str(run_id))
        if not entries:
            continue

        db_distance = _last(run, "db_d_mm")
        db_heading = _last(run, "db_a_deg")

        if run["name"] == "multi_stop":
            db_values = _column(run, "db_d_mm")
            if db_values is None or len(entries) != len(db_values):
                continue
            measured = [_pose_distance(entries[0])]
            odometry = [float(db_values[0]) / 1000.0]
            for i in range(1, len(entries)):
                measured.append(float(np.hypot(
                    entries[i]["x_m"] - entries[i - 1]["x_m"],
                    entries[i]["y_m"] - entries[i - 1]["y_m"],
                )))
                odometry.append((float(db_values[i]) - float(db_values[i - 1])) / 1000.0)
            total_measured = float(np.sum(measured))
            total_odometry = float(np.sum(odometry))
            if total_odometry > 1e-6:
                distance_candidates.append((
                    _run_repeat(run),
                    total_measured / total_odometry,
                ))
            continue

        if db_distance is None:
            continue

        measured_distance = _pose_distance(entries[-1])
        repeat = _run_repeat(run)
        if "odometry_straight" in run["name"] and db_distance > 1e-6:
            distance_candidates.append((
                repeat,
                measured_distance / (db_distance / 1000.0),
            ))
        elif "odometry_turn" in run["name"] and db_heading is not None:
            if abs(db_heading) > 1e-6:
                # The measured heading is counter-clockwise positive, like
                # the planned trajectory, but the odometry angle is
                # clockwise positive. Flip one of them so the ratio is the
                # scale between them and not its negative.
                heading_candidates.append((
                    repeat,
                    entries[-1]["heading_deg"] / -db_heading,
                ))
        elif run["name"].startswith("brake_from_") and db_distance > 1e-6:
            brake_pairs.append((
                run["name"],
                repeat,
                db_distance / 1000.0,
                measured_distance,
            ))

    distance_scale = (
        float(np.median([value for _repeat, value in distance_candidates]))
        if distance_candidates
        else float(distance_scale_default)
    )
    heading_scale = (
        float(np.median([value for _repeat, value in heading_candidates]))
        if heading_candidates
        else 1.0
    )

    slip = []
    for name, repeat, odometry_distance, measured_distance in brake_pairs:
        expected = odometry_distance * distance_scale
        if expected > 1e-6:
            slip.append({
                "run": name,
                "repeat": repeat,
                "ratio": float(1.0 - measured_distance / expected),
                "odometry_m": odometry_distance,
                "measured_m": measured_distance,
            })

    if slip:
        ratios = [entry["ratio"] for entry in slip]
        slip_summary = {
            "calibrated": True,
            "brake_ratio_median": float(np.median(ratios)),
            "brake_ratio_min": float(np.min(ratios)),
            "brake_ratio_max": float(np.max(ratios)),
            "brake_ratio_between_runs": _between_runs(
                [(entry["repeat"], entry["ratio"]) for entry in slip]),
            "runs": slip,
        }
    else:
        slip_summary = {"calibrated": False}

    return {
        "distance_scale": distance_scale,
        "heading_scale": heading_scale,
        "calibrated": bool(distance_candidates or heading_candidates),
        "n_distance_sources": len(distance_candidates),
        "n_heading_sources": len(heading_candidates),
        "distance_scale_between_runs": _between_runs(distance_candidates),
        "heading_scale_between_runs": _between_runs(heading_candidates),
    }, slip_summary


# ---------------------------------------------------------------------------
# Public fitting entry point
# ---------------------------------------------------------------------------

def _fit_drift(runs, poses, gyro_bias_deg_s):
    """The straight-line heading drift of the robot, in degrees per metre.

    A robot whose wheels are not parallel curves even when it is told to
    drive straight. The plan's straight segments are straight, so the
    curvature is a tracking error that grows with the distance travelled.

    Two independent estimates are made. The gyro sees the heading change
    directly, but a gyro bias looks exactly like a drift over a long run, so
    the measured bias is subtracted first. The tape sees the lateral offset
    the drift leaves behind, which is a pure length measurement with no
    sensor bias in it, but it also carries the operator's placement error.
    The two are reported together: if they disagree, one of the two
    systematic errors is wrong and the drift term should not be trusted.
    """
    estimates = []
    tape_estimates = []
    for run in runs:
        if "odometry_straight" not in run["name"]:
            continue
        distance_m = _last(run, "db_d_mm")
        heading_deg = _last(run, "db_a_deg")
        if distance_m is None or heading_deg is None or distance_m <= 0:
            continue
        distance_m = distance_m / 1000.0
        duration_s = _last(run, "t_ms")
        duration_s = (duration_s / 1000.0) if duration_s is not None else 0.0

        # The DriveBase angle is clockwise positive and the IMU is
        # counter-clockwise positive, so the sign flips once. A bias of b
        # deg/s over t seconds is b * t of fake heading change.
        bias = (gyro_bias_deg_s or 0.0) * duration_s
        drift = (-heading_deg - bias) / distance_m
        estimates.append((_run_repeat(run), float(drift)))

        # The tape: a constant drift of k over a distance L leaves a lateral
        # offset of about k * L^2 / 2 behind.
        entries = poses.get(str(run["meta"].get("run_id")))
        if entries:
            lateral_m = float(entries[-1]["y_m"])
            measured_m = _pose_distance(entries[-1])
            if measured_m > 1e-6:
                rad = 2.0 * lateral_m / (measured_m ** 2)
                tape_estimates.append(
                    (_run_repeat(run), float(np.degrees(rad))))

    if not estimates:
        return {"calibrated": False, "n_sources": 0}

    stats = _between_runs(estimates)
    result = {
        "calibrated": True,
        "n_sources": len(estimates),
        "drift_deg_per_m": stats["mean"],
        "drift_deg_per_m_between_runs": stats,
    }
    if stats.get("ci95"):
        result["drift_deg_per_m_ci95"] = stats["ci95"]
    if tape_estimates:
        tape = _between_runs(tape_estimates)
        result["drift_deg_per_m_tape"] = tape["mean"]
        result["drift_disagreement_deg_per_m"] = abs(
            tape["mean"] - stats["mean"])
    return result


def _fit_stall_torque(runs):
    """The motor's stall torque, in newton-metres.

    The wall test drives the robot into a wall and logs the load on each
    motor. The load is the torque the motor is producing, so the steady
    value is the motor's stall torque: the same number the force model
    calls t_max_nm, which the robot config otherwise has to declare.

    The median is used, not the peak: the first sample or two of the run
    are the impact and they overshoot the steady stall value.
    """
    estimates = []
    for run in runs:
        if run["name"] != "wall_stall":
            continue
        loads = [_column(run, "l_load_mNm"), _column(run, "r_load_mNm")]
        peaks = []
        for load in loads:
            if load is None or load.size == 0:
                continue
            # The load is in milli-newton-metres and the torque is in
            # newton-metres, so it is a factor of a thousand.
            peaks.append(float(np.median(np.abs(load))) / 1000.0)
        if not peaks:
            continue
        estimates.append((_run_repeat(run), max(peaks)))

    if not estimates:
        return {"calibrated": False, "n_sources": 0}

    stats = _between_runs(estimates)
    result = {
        "calibrated": True,
        "n_sources": len(estimates),
        "t_max_nm": stats["mean"],
        "t_max_nm_between_runs": stats,
    }
    if stats.get("ci95"):
        result["t_max_nm_ci95"] = stats["ci95"]
    return result


def fit_session(runs, poses=None, wheel_radius_m=0.028):
    poses = poses or {}
    _add_measured_outputs(runs, wheel_radius_m)

    straight_fits = _step_fits_measured(
        runs,
        ("straight_steps",),
        "v_cmd_mm_s",
        "v_meas_m_s",
        command_scale=1.0 / 1000.0,
        command_threshold=0.005,
    )
    turn_fits = _step_fits_measured(
        runs,
        ("turn_steps",),
        "w_cmd_deg_s",
        "w_meas_rad_s",
        command_scale=np.pi / 180.0,
        command_threshold=0.02,
    )

    straight_summary = _summary(straight_fits)
    turn_summary = _summary(turn_fits)

    chirp_straight = _chirp_nrmse(
        runs,
        ("chirp_straight",),
        "v_cmd_mm_s",
        "v_meas_m_s",
        1.0 / 1000.0,
        straight_summary,
    )
    chirp_turn = _chirp_nrmse(
        runs,
        ("chirp_turn",),
        "w_cmd_deg_s",
        "w_meas_rad_s",
        np.pi / 180.0,
        turn_summary,
    )

    reversal = next(
        (run for run in runs if run["name"] == "reversal_backlash"),
        None,
    )
    backlash = {"calibrated": False}
    if reversal is not None:
        reverse_delay = _reverse_delay(reversal, "v_meas_m_s")
        reversal_fits = _step_fits_measured(
            runs,
            ("reversal_backlash",),
            "v_cmd_mm_s",
            "v_meas_m_s",
            command_scale=1.0 / 1000.0,
            command_threshold=0.005,
        )
        low_speed_fits = [
            fit for fit in reversal_fits if abs(fit["command"]) <= 0.10
        ]
        if reverse_delay is not None or low_speed_fits:
            backlash = {
                "calibrated": True,
                "reverse_delay_s": reverse_delay,
                "low_speed_gain": (
                    float(np.median([fit["gain"] for fit in low_speed_fits]))
                    if low_speed_fits else None
                ),
            }

    odometry, slip = _fit_odometry(runs, poses)
    sensors = _static_summary(runs)
    drift = _fit_drift(runs, poses, sensors.get("gyro_z_bias_deg_s"))
    stall = _fit_stall_torque(runs)

    return {
        "wheel_radius_m": float(wheel_radius_m),
        "actuator": {
            "straight": {
                **straight_summary,
                "units": "m/s per m/s",
                "accel_limit_units": "m/s^2",
            },
            "turn": {
                **turn_summary,
                "units": "rad/s per rad/s",
                "accel_limit_units": "rad/s^2",
            },
        },
        "sensors": sensors,
        "odometry": odometry,
        "slip": slip,
        "backlash": backlash,
        "drift": drift,
        "stall_torque": stall,
        "validation": {
            "chirp_straight_nrmse": chirp_straight,
            "chirp_turn_nrmse": chirp_turn,
        },
        "runs": {
            run["name"]: {
                "run_id": run["meta"].get("run_id"),
                "rows": int(run["data"].shape[0]),
            }
            for run in runs
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_wheel_radius(path):
    """Backward-compatible wrapper around read_drivebase_profile."""
    profile = read_drivebase_profile(path)
    if profile.get("wheel_radius_m") is None:
        raise ValueError(f"no wheel_radius in {path}")
    return profile["wheel_radius_m"]


def _profile_sections(data):
    sections = []
    for key in ("drivebase", "robot", "config"):
        section = data.get(key)
        if isinstance(section, dict):
            sections.append(section)
    sections.append(data)
    return sections


def _profile_value(sections, *keys):
    for section in sections:
        for key in keys:
            if key in section and section[key] is not None:
                value = section[key]
                if isinstance(value, dict) and "val" in value:
                    value = value["val"]
                return value
    return None


def _positive_profile_value(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def read_drivebase_profile(path):
    """Read the minimal drivebase geometry needed by the host tools.

    The file may be a small profile, a full robot config, or an existing
    ``sysid_fit.json``. Recognized aliases:

    * wheel radius: ``wheel_radius_m``, ``wheel_radius``, ``radius``,
      ``wheel_diameter_mm``, ``wheel_diameter_m``, ``wheel_diameter``
    * track width: ``track_width_m``, ``track_width``,
      ``differentialTrackWidth``, ``axle_track_mm``, ``axle_track_m``,
      ``axle_track``

    ``wheel_diameter`` and ``axle_track`` are treated as millimetres because
    that is the Pybricks/LEGO convention. Parameters are optional so a profile
    can supply only wheel radius when track width is not known yet.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    sections = _profile_sections(data)

    wheel_radius = _profile_value(
        sections, "wheel_radius_m", "wheel_radius", "radius")
    if wheel_radius is None:
        diameter_mm = _profile_value(
            sections, "wheel_diameter_mm", "wheel_diameter")
        if diameter_mm is not None:
            wheel_radius = float(diameter_mm) / 2000.0
        else:
            diameter_m = _profile_value(sections, "wheel_diameter_m")
            if diameter_m is not None:
                wheel_radius = float(diameter_m) / 2.0
    if wheel_radius is not None:
        wheel_radius = _positive_profile_value(wheel_radius, "wheel radius")

    track_width = _profile_value(
        sections, "track_width_m", "track_width", "differentialTrackWidth")
    if track_width is None:
        axle_mm = _profile_value(sections, "axle_track_mm", "axle_track")
        if axle_mm is not None:
            track_width = float(axle_mm) / 1000.0
        else:
            axle_m = _profile_value(sections, "axle_track_m")
            if axle_m is not None:
                track_width = float(axle_m)
    if track_width is not None:
        track_width = _positive_profile_value(track_width, "track width")

    name = _profile_value(sections, "name", "drivebase_name")
    if wheel_radius is None and track_width is None:
        raise ValueError(
            f"{path} has no drivebase geometry; expected wheel_radius_m or "
            "track_width_m")

    return {
        "name": str(name) if name is not None else None,
        "wheel_radius_m": wheel_radius,
        "track_width_m": track_width,
        "source": path,
    }


def _print_interval(summary, name):
    """Print the 95% interval of a parameter over the repeats."""
    stats = summary.get(f"{name}_between_runs")
    if not stats or not stats.get("ci95"):
        print(f"  {name}: one run only, no interval.")
        return
    low, high = stats["ci95"]
    print(f"  {name}: 95% interval [{low:.4f}, {high:.4f}] over "
          f"{stats['n']} repeats (std {stats['std']:.4f}).")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit a digital-twin model from Pybricks SysId CSV logs."
    )
    parser.add_argument("logs", nargs="*", help="CSV file(s) or a directory of CSV files")
    parser.add_argument("--poses", help="measured pose CSV")
    parser.add_argument(
        "--from-hub-session",
        help=(
            "captured hub console output (hub_deploy.py --out, or "
            "main.py --push-hub --hub-log); its POSE lines are converted to "
            "--out-poses and nothing is fitted"
        ),
    )
    parser.add_argument(
        "--out-poses", help="measured pose CSV to write for --from-hub-session"
    )
    parser.add_argument(
        "--run-id", default="hub",
        help="fallback run id for poses in a session without META lines",
    )
    parser.add_argument(
        "--from-sheet",
        help=(
            "the operator's measurement sheet; it is converted to "
            "--out-poses using the test names in the session and nothing is "
            "fitted"
        ),
    )
    parser.add_argument(
        "--write-sheet",
        help="write a blank measurement sheet for the tests in the session",
    )
    parser.add_argument(
        "--profile",
        help="minimal drivebase profile JSON (wheel_radius_m, track_width_m)",
    )
    parser.add_argument(
        "--robot-config",
        default=None,
        help="full robot config JSON; --profile is preferred",
    )
    parser.add_argument(
        "--wheel-radius",
        type=float,
        help="wheel radius in metres; overrides --profile/--robot-config",
    )
    parser.add_argument("--out", default="sysid_fit.json", help="output JSON path")
    args = parser.parse_args(argv)

    if args.from_hub_session:
        return _convert_hub_session(args, parser)

    if args.write_sheet:
        return _write_sheet(args, parser)

    if args.from_sheet:
        return _convert_sheet(args, parser)

    if args.profile and args.robot_config:
        parser.error("use either --profile or --robot-config, not both")

    profile_path = args.profile or args.robot_config
    profile = read_drivebase_profile(profile_path) if profile_path else None
    wheel_radius = (
        args.wheel_radius
        if args.wheel_radius is not None
        else (profile.get("wheel_radius_m") if profile else None)
    )
    if wheel_radius is None:
        parser.error(
            "provide --profile with wheel_radius_m (or --wheel-radius; "
            "--robot-config is also accepted)"
        )
    wheel_radius = float(wheel_radius)
    if not np.isfinite(wheel_radius) or wheel_radius <= 0.0:
        parser.error("wheel radius must be a positive finite number")
    track_width = profile.get("track_width_m") if profile else None

    files, runs = load_runs(args.logs)
    if not runs:
        print("No TEST_START blocks found.")
        return 1

    poses = load_poses(args.poses) if args.poses else {}
    result = fit_session(runs, poses=poses, wheel_radius_m=wheel_radius)
    result["source"] = files
    result["drivebase"] = {
        "name": profile.get("name") if profile else None,
        "wheel_radius_m": float(wheel_radius),
        "track_width_m": float(track_width) if track_width is not None else None,
        "source": profile_path,
    }

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    if profile is not None:
        label = profile.get("name") or "unnamed"
        print(
            "Drivebase: {} (wheel radius {:.4f} m, track width {})".format(
                label,
                wheel_radius,
                "unknown" if track_width is None else f"{track_width:.4f} m",
            )
        )
    straight = result["actuator"]["straight"]
    turn = result["actuator"]["turn"]
    print(f"Parsed {len(runs)} runs from {len(files)} file(s).")
    if straight.get("calibrated"):
        print(
            "Straight: gain={gain:.3f}, tau={tau_s:.3f}s, delay={delay_s:.3f}s, "
            "n={n_steps} steps in {n_runs} repeats".format(**straight)
        )
        _print_interval(straight, "gain")
    else:
        print("Straight: no usable step data.")
    if turn.get("calibrated"):
        print(
            "Turn: gain={gain:.3f}, tau={tau_s:.3f}s, delay={delay_s:.3f}s, "
            "n={n_steps} steps in {n_runs} repeats".format(**turn)
        )
        _print_interval(turn, "gain")
    else:
        print("Turn: no usable step data.")
    if result["sensors"]["calibrated"]:
        print(
            "Gyro z bias: {gyro_z_bias_deg_s:.4f} deg/s "
            "(std {gyro_z_std_deg_s:.4f})".format(**result["sensors"])
        )
    if result["odometry"]["calibrated"]:
        print(
            "Odometry: distance_scale={distance_scale:.4f}, "
            "heading_scale={heading_scale:.4f}".format(**result["odometry"])
        )
        _print_interval(result["odometry"], "distance_scale")
    else:
        print("Odometry: no measured poses supplied.")
    if result["slip"]["calibrated"]:
        print(
            "Brake slip: median={brake_ratio_median:.3f}, "
            "range=[{brake_ratio_min:.3f}, {brake_ratio_max:.3f}]".format(
                **result["slip"]
            )
        )
    drift = result.get("drift", {})
    if drift.get("calibrated"):
        print(
            "Straight drift: {drift_deg_per_m:.3f} deg/m "
            "(the robot curves when told to drive straight)".format(**drift)
        )
        if "drift_deg_per_m_tape" in drift:
            print(
                "  tape says {drift_deg_per_m_tape:.3f} deg/m, "
                "disagreement {drift_disagreement_deg_per_m:.3f} deg/m. A big "
                "disagreement means the gyro bias or the start mark is off, "
                "so do not trust the drift term.".format(**drift)
            )
    stall = result.get("stall_torque", {})
    if stall.get("calibrated"):
        print("Stall torque: {t_max_nm:.4f} Nm".format(**stall))
    print("The intervals cover the run-to-run variation of the robot. They do "
          "not cover the error of the tape measure, which is the same every "
          "time and never averages out.")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
