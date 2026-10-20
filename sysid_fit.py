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


def _linear_step_fit(times, measured, delay, tau):
    """Least-squares fit of a step response for a given delay and tau.

    The model is ``y = y0 * e + yss * (1 - e)`` with
    ``e = exp(-(t - t0 - delay) / tau)``, which is linear in the two
    amplitudes, so they are solved in closed form instead of being read off
    the data. That uses every sample of the step rather than the median of
    the tail, and it gives the covariance of the amplitudes for free.

    Returns (y0, yss, sse, se_yss).
    """
    t0 = float(times[0])
    times = np.asarray(times)
    measured = np.asarray(measured)
    e = np.exp(-np.maximum(times - (t0 + delay), 0.0) / tau)
    c1 = e
    c2 = 1.0 - e
    a11 = float(np.sum(c1 * c1))
    a12 = float(np.sum(c1 * c2))
    a22 = float(np.sum(c2 * c2))
    b1 = float(np.sum(c1 * measured))
    b2 = float(np.sum(c2 * measured))
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-18:
        return None
    y0 = (b1 * a22 - b2 * a12) / det
    yss = (a11 * b2 - a12 * b1) / det
    sse = float(np.sum(measured * measured) - (y0 * b1 + yss * b2))
    sse = max(sse, 0.0)
    dof = max(len(measured) - 2, 1)
    sigma2 = sse / dof
    se_yss = float(np.sqrt(max(sigma2 * a11 / det, 0.0)))
    return y0, yss, sse, se_yss


def _fit_first_order(times, measured, command):
    """Fit y(t) = yss + (y0 - yss) * exp(-(t - t0 - delay) / tau).

    Two passes. The first is a blind grid over the whole plausible range, so
    it cannot be fooled by a local minimum. The second refines around that
    answer at a thousandth of the resolution, because the first pass can only
    land on a 10 ms boundary and that quantisation is a floor on the
    reported time constant and delay.
    """
    if len(times) < 10:
        return None

    t0 = float(times[0])
    best_error = float("inf")
    best_delay = 0.0
    best_tau = 0.1
    best_yss = None
    best_se = None

    for delay in np.arange(0.0, 0.121, 0.01):
        for tau in np.arange(0.01, 0.501, 0.01):
            fit = _linear_step_fit(times, measured, float(delay), float(tau))
            if fit is None:
                continue
            _y0, yss, sse, se_yss = fit
            if sse < best_error:
                best_error = sse
                best_delay = float(delay)
                best_tau = float(tau)
                best_yss = yss
                best_se = se_yss

    if best_yss is None or abs(best_yss) < 1e-5:
        return None

    # Second pass: a fine grid around the first answer.
    fine_error = best_error
    for delay in np.arange(max(0.0, best_delay - 0.012),
                            best_delay + 0.0121, 0.001):
        for tau in np.arange(max(0.005, best_tau - 0.020),
                             best_tau + 0.0201, 0.001):
            fit = _linear_step_fit(times, measured, float(delay), float(tau))
            if fit is None:
                continue
            _y0, yss, sse, se_yss = fit
            if sse < fine_error:
                fine_error = sse
                best_delay = float(delay)
                best_tau = float(tau)
                best_yss = yss
                best_se = se_yss

    if abs(best_yss) < 1e-5:
        return None

    gradient = np.abs(np.gradient(measured, times))
    return {
        "gain": best_yss / command,
        "gain_se": (best_se / abs(command)) if best_se is not None else None,
        "tau_s": best_tau,
        "delay_s": best_delay,
        "accel_limit": float(np.percentile(gradient, 95)),
        "rmse": float(np.sqrt(fine_error / len(measured))),
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
    step_se = [fit["gain_se"] for fit in fits if fit.get("gain_se") is not None]
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
        # How well one step determines the gain. The steps inside a run are
        # sequential, so this is not a standard error over them.
        "gain_se_step": (float(np.median(step_se)) if step_se else None),
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
    if times.size < 2:
        # A gradient needs two points, and a run that short says nothing.
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
    if times.size < 2:
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


def _wall_homing_scale(run, wall_distance_m):
    """The distance scale from a run that drove into a wall.

    The wall is the reference: the robot was a known distance from it and
    the odometry says how far it thinks it went. The ratio is the same
    distance scale the tape gives, but the reference does not move and
    nobody has to read it, so the run can be repeated unattended.

    Two things are checked. The robot has to have actually reached the
    wall: the last row that was still commanded has to show the wheels
    barely turning, and a run that times out against nothing is rejected.
    And the approach has to be roughly square: a robot that meets the wall
    at an angle travels a little further, so the estimate is wrong by the
    cosine of that angle, which is second order and therefore small.
    """
    if not wall_distance_m or wall_distance_m <= 0:
        return None
    commands = _column(run, "v_cmd_mm_s")
    speeds = _column(run, "db_v_mm_s")
    distances = _column(run, "db_d_mm")
    if commands is None or speeds is None or distances is None:
        return None
    if commands.size == 0 or speeds.size == 0 or distances.size == 0:
        return None

    command = float(np.max(np.abs(commands)))
    if command < 1e-6:
        return None

    # The last row that was still commanded. The settle row the collector
    # writes after the robot stops has the command at zero, so the final
    # row says nothing about whether the robot was blocked.
    commanded = np.nonzero(np.abs(commands) > 1e-6)[0]
    if commanded.size == 0:
        return None
    last = int(commanded[-1])
    if abs(float(speeds[last])) > 0.2 * command:
        # Still moving when the run ended: the robot never reached the wall.
        return None

    odometry_m = float(distances[-1]) / 1000.0
    if odometry_m <= 1e-6:
        return None
    return float(wall_distance_m) / odometry_m


def _fit_odometry(runs, poses, distance_scale_default=1.0, wall_distance_m=None):
    distance_candidates = []
    heading_candidates = []
    brake_pairs = []
    homing_candidates = []

    for run in runs:
        run_id = run["meta"].get("run_id")
        if run_id is None:
            continue

        # The wall homing run needs no measured pose: the wall is the
        # reference, so it is handled before the pose lookup.
        if run["name"] == "wall_homing":
            ratio = _wall_homing_scale(run, wall_distance_m)
            if ratio is not None:
                homing_candidates.append((_run_repeat(run), ratio))
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

    # The wall and the tape are two estimates of the same distance scale, so
    # they are pooled for the value and reported separately, the same way the
    # drift is. A disagreement means the mark or the tape is off.
    all_distance = ([value for _repeat, value in distance_candidates]
                    + [value for _repeat, value in homing_candidates])
    distance_scale = (
        float(np.median(all_distance))
        if all_distance
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

    summary = {
        "distance_scale": distance_scale,
        "heading_scale": heading_scale,
        "calibrated": bool(all_distance or heading_candidates),
        "n_distance_sources": len(distance_candidates),
        "n_heading_sources": len(heading_candidates),
        "distance_scale_between_runs": _between_runs(distance_candidates),
        "heading_scale_between_runs": _between_runs(heading_candidates),
    }
    if distance_candidates:
        summary["distance_scale_tape"] = float(
            np.median([value for _repeat, value in distance_candidates]))
    if homing_candidates:
        summary["distance_scale_homing"] = float(
            np.median([value for _repeat, value in homing_candidates]))
        summary["n_homing_sources"] = len(homing_candidates)
        summary["distance_scale_homing_between_runs"] = _between_runs(
            homing_candidates)
    if distance_candidates and homing_candidates:
        summary["distance_scale_disagreement"] = abs(
            summary["distance_scale_homing"] - summary["distance_scale_tape"])
    return summary, slip_summary


# ---------------------------------------------------------------------------
# Public fitting entry point
# ---------------------------------------------------------------------------

def _drift_points(runs):
    """(speed, heading rate) points from the runs that drive straight.

    The heading rate is the IMU's z rate, which is counter-clockwise
    positive, and the speed is the measured one, so the drift is against the
    motion the robot actually made. A robot that curves while going
    backwards curves the other way, so the signed speed puts the forward and
    the reverse run in the same fit.
    """
    points = []
    for run in runs:
        if not run["name"].startswith("straight_steps"):
            continue
        times = _column(run, "t_ms")
        command = _column(run, "v_cmd_mm_s")
        speed = _measured_column(run, "v_meas_m_s")
        gyro = _column(run, "gyro_z_rad_s")
        if times is None or command is None or speed is None or gyro is None:
            continue
        times = times / 1000.0
        gyro_deg_s = np.degrees(gyro)
        for start, end in _find_steps(times, command, 0.005):
            # The steady part of the step: the transient is not the drift.
            tail = max(start + 3, int(start + 0.65 * (end - start)))
            if end - tail < 3:
                continue
            points.append((
                _run_repeat(run),
                float(np.median(speed[tail:end])),
                float(np.median(gyro_deg_s[tail:end])),
            ))
    return points


def _fit_drift(runs, poses, gyro_bias_deg_s):
    """The straight-line heading drift of the robot, in degrees per metre.

    A robot whose wheels are not parallel curves even when it is told to
    drive straight. The plan's straight segments are straight, so the
    curvature is a tracking error that grows with the distance travelled.

    The drift is a slope, not a subtraction. A run that drives straight at a
    constant speed turns at ``drift * speed``, and the gyro also reads a
    constant bias, so the heading rate against the speed is a straight line
    whose slope is the drift and whose intercept is the bias. Regressing it
    means the bias cannot leak into the drift: subtracting a separately
    measured bias instead would multiply the bias error by the run's length
    over its distance, which is ten for the one-metre odometry run.

    The tape gives an independent estimate from the lateral offset the drift
    leaves behind. That is a pure length measurement with no sensor bias in
    it, but it carries the operator's placement error, so the two are
    reported together with the disagreement between them.
    """
    points = _drift_points(runs)
    by_repeat = {}
    for repeat, speed, rate in points:
        by_repeat.setdefault(repeat, []).append((speed, rate))

    estimates = []
    sources = []
    for repeat in sorted(by_repeat):
        group = by_repeat[repeat]
        if len(group) < 3:
            # A line through two points has no degrees of freedom, so the
            # standard error of its slope is not defined. Three points is the
            # least that has one.
            continue
        x = np.array([point[0] for point in group])
        y = np.array([point[1] for point in group])
        if float(np.ptp(x)) < 1e-6:
            continue
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        slope_se = float(np.sqrt(max(cov[0][0], 0.0)))
        if not np.isfinite(slope_se):
            slope_se = None
        estimates.append((repeat, slope))
        sources.append({
            "repeat": repeat,
            "points": len(group),
            "speed_range_m_s": [float(np.min(x)), float(np.max(x))],
            "slope_se": slope_se,
            # The intercept is the gyro's bias, read off while the robot was
            # moving. It is a second opinion on the static measurement.
            "gyro_bias_deg_s": intercept,
        })

    if not estimates:
        return {"calibrated": False, "n_sources": 0}

    stats = _between_runs(estimates)
    result = {
        "calibrated": True,
        "n_sources": len(estimates),
        "drift_deg_per_m": stats["mean"],
        "drift_deg_per_m_between_runs": stats,
        "drift_sources": sources,
    }
    if stats.get("ci95"):
        result["drift_deg_per_m_ci95"] = stats["ci95"]

    # The tape: a constant drift of k over a distance L leaves a lateral
    # offset of about k * L^2 / 2 behind.
    tape_estimates = []
    for run in runs:
        if "odometry_straight" not in run["name"]:
            continue
        entries = poses.get(str(run["meta"].get("run_id")))
        if not entries:
            continue
        lateral_m = float(entries[-1]["y_m"])
        measured_m = _pose_distance(entries[-1])
        if measured_m > 1e-6:
            tape_estimates.append((
                _run_repeat(run),
                float(np.degrees(2.0 * lateral_m / (measured_m ** 2))),
            ))
    if tape_estimates:
        tape = _between_runs(tape_estimates)
        result["drift_deg_per_m_tape"] = tape["mean"]
        result["drift_disagreement_deg_per_m"] = abs(
            tape["mean"] - stats["mean"])

    # The regression's intercept is the gyro's bias, read while the robot was
    # moving. The static test measured it while the robot stood still, so the
    # two are a check on each other.
    if sources and gyro_bias_deg_s is not None:
        moving = float(np.mean([s["gyro_bias_deg_s"] for s in sources]))
        result["gyro_bias_deg_s_moving"] = moving
        result["gyro_bias_disagreement_deg_s"] = abs(
            moving - float(gyro_bias_deg_s))
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


def _fit_no_load_speed(runs, wheel_radius_m):
    """The motor's free-running speed, in rad/s at the wheel.

    The top-speed test runs the robot with the wheels off the ground, so the
    motor pushes nothing and runs up to the speed at which it makes no
    torque. That is the other end of the torque-speed line the force model
    uses: at this speed the motor's torque is zero.

    The robot config has to declare that number, and a declared number is a
    guess. This measures it.

    The measurement is only valid when the motor was torque-limited: if the
    robot reached the commanded speed, the firmware's speed limit was the
    binding constraint and the number is not the motor's no-load speed. That
    is why the test commands a speed no LEGO motor can reach.
    """
    estimates = []
    rejected = []
    for run in runs:
        if run["name"] != "top_speed":
            continue
        times, speed = _straight_measured(run, wheel_radius_m)
        if times is None or speed is None or len(speed) < 20:
            continue
        # The motor has settled over the last second of the run.
        window = times >= (times[-1] - 1.0)
        settled = speed[window]
        command = _column(run, "v_cmd_mm_s")
        if command is None:
            continue
        commanded = command[window]
        settled = settled[np.isfinite(settled)]
        commanded = commanded[np.isfinite(commanded)]
        if settled.size < 5 or commanded.size < 5:
            continue
        measured_mm_s = float(np.median(settled)) * 1000.0
        commanded_mm_s = float(np.median(commanded))
        if commanded_mm_s <= 0.0:
            continue
        if measured_mm_s >= 0.98 * commanded_mm_s:
            # The robot reached the command, so the command was the limit.
            rejected.append(
                "the robot reached the commanded {:.0f} mm/s, so the command "
                "was the limit and not the motor".format(commanded_mm_s))
            continue
        value = float(np.median(settled)) / wheel_radius_m
        if value <= 0.0:
            continue
        estimates.append((_run_repeat(run), value))

    if not estimates:
        result = {"calibrated": False, "n_sources": 0}
        if rejected:
            result["rejected"] = rejected
        return result

    stats = _between_runs(estimates)
    result = {
        "calibrated": True,
        "n_sources": len(estimates),
        "no_load_speed_rad_s": stats["mean"],
        "no_load_speed_rad_s_between_runs": stats,
    }
    if stats.get("ci95"):
        result["no_load_speed_rad_s_ci95"] = stats["ci95"]
    if rejected:
        result["rejected"] = rejected
    return result


def fit_session(runs, poses=None, wheel_radius_m=0.028, wall_distance_m=None):
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

    odometry, slip = _fit_odometry(runs, poses, wall_distance_m=wall_distance_m)
    sensors = _static_summary(runs)
    drift = _fit_drift(runs, poses, sensors.get("gyro_z_bias_deg_s"))
    stall = _fit_stall_torque(runs)
    drivetrain = _fit_no_load_speed(runs, wheel_radius_m)

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
        "drivetrain": drivetrain,
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

    # The wall the homing run drives into. It is a setup number, not a robot
    # number: how far the start mark is from the wall, measured to where the
    # robot's centre rests against it.
    wall = data.get("wall") if isinstance(data.get("wall"), dict) else {}
    wall_distance = _profile_value([wall, data], "distance_m", "wall_distance_m")
    if wall_distance is not None:
        wall_distance = _positive_profile_value(wall_distance, "wall distance")

    return {
        "name": str(name) if name is not None else None,
        "wheel_radius_m": wheel_radius,
        "track_width_m": track_width,
        "wall_distance_m": wall_distance,
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
    wall_distance = profile.get("wall_distance_m") if profile else None
    result = fit_session(runs, poses=poses, wheel_radius_m=wheel_radius,
                         wall_distance_m=wall_distance)
    result["source"] = files
    result["drivebase"] = {
        "name": profile.get("name") if profile else None,
        "wheel_radius_m": float(wheel_radius),
        "track_width_m": float(track_width) if track_width is not None else None,
        "wall_distance_m": (float(wall_distance)
                            if wall_distance is not None else None),
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
        if "distance_scale_homing" in result["odometry"]:
            print(
                "  the wall says {distance_scale_homing:.4f}, the tape says "
                "{distance_scale_tape:.4f}".format(**result["odometry"])
            )
            if "distance_scale_disagreement" in result["odometry"]:
                print(
                    "  they disagree by "
                    "{distance_scale_disagreement:.4f}. A big disagreement "
                    "means the start mark or the tape is off.".format(
                        **result["odometry"])
                )
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
    drivetrain = result.get("drivetrain", {})
    if drivetrain.get("calibrated"):
        print(
            "No-load speed: {no_load_speed_rad_s:.3f} rad/s at the wheel "
            "(the speed at which the motor makes no torque)".format(
                **drivetrain)
        )
        _print_interval(drivetrain, "no_load_speed_rad_s")
    for reason in drivetrain.get("rejected", []) or []:
        print(f"No-load speed: not used, {reason}.")
    print("The intervals cover the run-to-run variation of the robot. They do "
          "not cover the error of the tape measure, which is the same every "
          "time and never averages out.")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
