"""
Fit a first-order digital-twin model from ``pybricks_code/sysid_digital_twin.py``
console logs.

This is host-side code. It reads the CSV blocks emitted by the robot logger,
fits straight/turn actuator response, odometry scale, and a bounded braking
slip estimate, lands the operator's weigh-in mass and the inertia derived
from the turn steps, then writes a JSON file consumed by ``main.py
--sysid-fit``.

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

The operator's measurement sheet can be fitted directly, with no intermediate
``poses.csv``:

    python sysid_fit.py session.csv --sheet calibration_sheet.csv \
        --profile my_drivebase.json --out sysid_fit.json

``--write-form`` writes the same sheet as a form in a browser, so the numbers
go in as numbers instead of as text in a spreadsheet.
"""

import argparse
import csv
import glob
import io
import json
import os
import sys

import numpy as np

from overboard.sysid import sheet_form


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _finish_run(run, header, rows, bad_rows=0, expected_rows=None, ended=False):
    if run is None or header is None:
        return
    run["columns"] = {name.strip(): i for i, name in enumerate(header)}
    run["bad_rows"] = int(bad_rows)
    run["ended"] = bool(ended)
    if expected_rows is not None:
        run["expected_rows"] = int(expected_rows)
        parsed_count = len(rows)
        missing = expected_rows - (parsed_count + bad_rows)
        run["missing_rows"] = max(0, missing)
        if run["missing_rows"] > 0:
            import warnings
            warnings.warn(
                f"Run '{run.get("name", "unnamed")}' is truncated: received {parsed_count} row(s) "
                f"({bad_rows} bad, {run['missing_rows']} missing of {expected_rows} expected).",
                UserWarning,
                stacklevel=2,
            )
    else:
        run["missing_rows"] = 0
    if rows:
        run["data"] = np.asarray(rows, dtype=float)
    else:
        run["data"] = np.empty((0, len(header)), dtype=float)


def parse_session(text):
    """Parse one or more TEST_START ... TEST_END blocks into run dictionaries.

    A row that carries a non-finite number (a NaN or an infinity) is a bad
    row, not data: it would poison every fit that reads the column. Such a
    row is dropped and counted, so the fit can tell the operator that the
    session file is corrupt. The count is on the run as ``bad_rows``.
    """
    runs = []
    current = None
    header = None
    rows = []
    bad_rows = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("TEST_START"):
            _finish_run(current, header, rows, bad_rows, ended=False)
            name = line[len("TEST_START"):].strip()
            current = {"name": name, "meta": {}, "columns": {},
                       "data": np.empty((0, 0)), "bad_rows": 0, "ended": False}
            runs.append(current)
            header = None
            rows = []
            bad_rows = 0
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
            bad_rows = 0
            continue

        if line.startswith("TEST_END"):
            expected = None
            rest = line[len("TEST_END"):].strip()
            if rest.startswith(","):
                rest = rest[1:].strip()
            if rest:
                try:
                    expected = int(rest.split()[0].split(",")[0])
                except ValueError:
                    expected = None
            _finish_run(current, header, rows, bad_rows, expected_rows=expected, ended=True)
            header = None
            rows = []
            bad_rows = 0
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
            if not np.all(np.isfinite(values)):
                # A non-finite cell is a bad cell, not data. Keeping it
                # would put a NaN into the fit and out into the model.
                bad_rows += 1
                continue
            rows.append(values)

    _finish_run(current, header, rows, bad_rows)
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
#   test,repeat,stop,x_mm,y_mm,turn_right_deg,mass_kg
#   odometry_straight,0,0,998,2,0,
#   weigh_in,0,0,,,,0.851
#
# x is forward from the start pose, y is to the left, and the turn is how far
# the robot turned to the right. The heading the fitter wants is
# counter-clockwise positive, so the turn is negated on the way in. The
# mass_kg column carries the weigh-in rows (test name weigh_in, three repeats
# per the KI-010 discipline): the robot's mass with the battery, weighed on a
# bench scale. They are bench measurements, not hub tests, so their rows are
# appended to every sheet and skipped by read_sheet.
SHEET_FIELDS = ["test", "repeat", "stop", "x_mm", "y_mm", "turn_right_deg",
                "mass_kg"]


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


def sheet_rows(runs):
    """The rows of the measurement sheet: one per measurement the session needs.

    Each row says which test, which repeat and which stop it is, so the
    operator's numbers are keyed by what was measured and not by a run id
    that changes when the session is re-collected.
    """
    rows = []
    for (name, repeat), (_run_id, count) in sorted(session_tests(runs).items()):
        for stop in range(count):
            rows.append({"test": name, "repeat": repeat, "stop": stop})
    return rows


def write_sheet(runs, path):
    """Write a blank measurement sheet for the tests that need one.

    The weigh-in rows are always part of the sheet: the mass is a bench
    measurement every calibration needs, hub tests or not.
    """
    rows = sheet_rows(runs) + sheet_form.weigh_in_rows()
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SHEET_FIELDS})
    return path


def read_sheet(path):
    """Read the operator's measurement sheet.

    Rows the operator left empty are skipped, so a sheet can be filled in
    part by part. The weigh-in rows are not poses and are skipped here; read
    them with read_weigh_in.
    """
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            name = (raw.get("test") or "").strip()
            if not name or name == sheet_form.WEIGH_IN_TEST:
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


def read_weigh_in(path):
    """The operator's weigh-in rows as (repeat, mass_kg) pairs.

    A row the operator left blank, or filled with a non-positive number, is
    not a measurement and is skipped.
    """
    pairs = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if (raw.get("test") or "").strip() != sheet_form.WEIGH_IN_TEST:
                continue
            try:
                repeat = int((raw.get("repeat") or "0").strip())
                mass = float((raw.get("mass_kg") or "").strip())
            except (TypeError, ValueError):
                continue
            if mass > 0.0:
                pairs.append((repeat, mass))
    return pairs


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


def poses_from_sheet_rows(rows, runs):
    """The operator's sheet as the pose map the fitter takes.

    ``sheet_to_poses`` gives the rows; this gives the dict ``fit_session``
    reads, keyed by the run id and sorted by stop. The key is the run id as a
    string, because that is how the fitter looks a run's poses up.

    Returns (poses, problems). A problem is a line of plain text for the
    operator: a row for a test that did not run, or a test that needs a
    measurement and did not get one.
    """
    pose_rows, problems = sheet_to_poses(rows, runs)
    poses = {}
    for row in pose_rows:
        poses.setdefault(str(row["run_id"]), []).append({
            "stop": row["stop"],
            "x_m": row["x_m"],
            "y_m": row["y_m"],
            "heading_deg": row["heading_deg"],
        })
    for entries in poses.values():
        entries.sort(key=lambda entry: entry["stop"])
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
    print("Fill in x_mm, y_mm, turn_right_deg and the weigh-in mass_kg "
          "rows, then run:")
    print(f"  python sysid_fit.py {' '.join(args.logs)} --from-sheet "
          f"{args.write_sheet} --out-poses poses.csv")
    print("Or fit it in one step, with no intermediate file:")
    print(f"  python sysid_fit.py {' '.join(args.logs)} --sheet "
          f"{args.write_sheet} --profile my_drivebase.json "
          f"--out sysid_fit.json")
    return 0


def _write_form(args, parser):
    """CLI mode: write the measurement sheet as a form in a browser."""
    _files, runs = load_runs(args.logs)
    if not runs:
        print("No TEST_START blocks found.")
        return 1
    rows = sheet_rows(runs)
    if not rows:
        print("The session has no test that needs a measurement.")
        return 1

    csv_name = os.path.basename(args.write_form)
    if not csv_name.lower().endswith(".csv"):
        csv_name = sheet_form.DEFAULT_CSV_NAME
    fit_command = (
        f"python sysid_fit.py {' '.join(args.logs)} --sheet {csv_name} "
        f"--profile my_drivebase.json --out sysid_fit.json")
    sheet_form.write_form(rows, args.write_form, csv_name=csv_name,
                          fit_command=fit_command)
    print(f"Wrote a measurement form for {len(rows)} measurement(s) to "
          f"{args.write_form}.")
    print("Open it in a browser, type the numbers the tape says, and click "
          "Download CSV.")
    print(f"Then run: {fit_command}")
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
    gain = best_yss / command
    if not np.isfinite(gain):
        # A step that cannot be fitted to a finite gain is not a step.
        return None
    return {
        "gain": gain,
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


# The smallest lower bound a reported interval may have. A t-interval over
# two repeats is wider than the parameter itself, so its lower bound can be
# negative, and a negative bound makes the fit file unloadable: the loader
# refuses a gain band that is not positive.
_MIN_GAIN_BOUND = 1e-6


def _summary(fits):
    if not fits:
        return {"calibrated": False, "n_steps": 0}

    # A fit that carries a non-finite number is not a fit: it would put a
    # NaN into the summary and out into the model. Such a fit is skipped
    # and is not counted as a fitted step.
    fits = [fit for fit in fits if np.isfinite(fit.get("gain", np.nan))]
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
        ci = stats.get("ci95")
        if not ci:
            continue
        low, high = float(ci[0]), float(ci[1])
        # The raw interval is what the t-multiplier really said. The
        # reported one is clamped so it can be loaded: a gain band that is
        # not positive is refused by the loader, and a negative time
        # constant is not a time constant.
        floor = _MIN_GAIN_BOUND if name == "gain" else 0.0
        summary[f"{name}_ci95"] = [max(low, floor), high]
        summary[f"{name}_ci95_raw"] = [low, high]
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
            if not np.isfinite(segment_command):
                # A step whose command is not finite is a bad step, not a
                # step: fitting it would put a NaN into the whole fit.
                continue
            if abs(segment_command) < command_threshold:
                continue
            if times[end - 1] - times[start] < 0.30:
                continue
            if not np.all(np.isfinite(measured[start:end])):
                # Same for the measurement: a bad sample is not data.
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
            # Where the step lives in the run's data, for the body-parameter
            # derivation that reuses the same windows. The fit dicts are
            # never serialized -- only _summary products are -- so these
            # private keys stay in memory.
            fit["_run"] = run
            fit["_span"] = (start, end)
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

    # The wall and the tape are two estimates of the same distance scale.
    # They are NOT pooled: a median over two different estimates is
    # neither of them, and the number the validator plans against has to
    # be one of the estimates. The tape is the measurement, so it is the
    # one that is used; the wall is the cross-check and is reported next
    # to it, so a disagreement between the two shows up.
    tape_values = [value for _repeat, value in distance_candidates]
    homing_values = [value for _repeat, value in homing_candidates]
    all_distance = tape_values + homing_values
    if tape_values:
        distance_scale = float(np.median(tape_values))
        distance_scale_source = "tape"
    elif homing_values:
        distance_scale = float(np.median(homing_values))
        distance_scale_source = "wall"
    else:
        distance_scale = float(distance_scale_default)
        distance_scale_source = "default"
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
        "distance_scale_source": distance_scale_source,
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


# ---------------------------------------------------------------------------
# Body parameters: the weighed mass and the derived inertia
# ---------------------------------------------------------------------------
#
# Mass and yaw inertia are what the CPU force/slip audit plans against
# (diff_drive_wheel_forces), and no procedure step used to land them. The
# command semantics decide the derivation. The step tests drive
# ``drivebase.drive(v, w)``, a closed-loop speed command: the fitted tau of
# the command-to-speed response is the CLOSED-LOOP time constant, and the
# session carries no duty fraction, so the pinned open-loop line
# "param = tau * (T_stall * c) / y_ss" cannot use the command column. What
# the session does carry is the actuation: every rate-test row logs the
# motor torque (l_load_mNm / r_load_mNm). Both factors of the pinned line
# are therefore measured from the step's actual actuation:
#
#   J = tau_mech * (B + K_v)
#
# * (B + K_v) = K0 / y_ss is the steady state: K0 is the step's zero-speed
#   drive level, G_ss + K_v * y_ss, where G_ss is the logged generalized
#   force at the steady tail and K_v the back-EMF slope of the measured
#   torque-speed line (t_max_nm x no-load speed). B = G_ss / y_ss is the
#   drag the logged torque fights; K_v is what the back-EMF already ate
#   before the torque was logged. Both come from the step's measured
#   steady tail, not from the first-order fit's amplitude: the drivebase's
#   speed loop overshoots slightly, and the fit's asymptote carries that
#   as a 2-3% bias on every step.
# * tau_mech is the MECHANICAL time constant, from the step's transient
#   against the logged drive: J is identified from the exact discrete step
#   response of J dq = (G - B q) dt (so the measured speed is never
#   differentiated and no quadrature error creeps in), then
#   tau_mech = J / (B + K_v).
#
# Conventions, verified against the code that consumes the result:
# * The logged torque is motor-shaft torque in mNm (the convention of
#   _fit_stall_torque), positive along the wheel's forward direction. The
#   generalized force carries the same gearing factor as
#   motor_force_at_velocity: G = (Tl + Tr) * gearing / r_wheel straight;
#   for a turn q is clockwise-positive (the drivebase's turn command and
#   _turn_measured's convention), so G = (Tl - Tr) * gearing * track_width
#   / (2 * r_wheel). gearing is motor revs per wheel rev
#   (RobotConfig.gearing).
# * The identified J is the effective body-side parameter
#   diff_drive_wheel_forces consumes, and it ALREADY includes the reflected
#   motor/gear rotor inertia (the shaft torque that is logged spins the
#   rotors through the gearing). reflected_inertia_kg_m2 is never added to
#   inertia in the force path, so the value is written to "inertia" alone --
#   adding the two would double-count.
# * The turn step exists (turn_steps_cw/ccw), so the yaw inertia has a real
#   measurement; no new hardware test is invented.
#
# The mass that lands in the config is the operator's weigh-in (the sheet),
# 3x per the KI-010 discipline. The straight steps give an independent
# derived mass on the same machinery, reported as a cross-check only.

def _back_emf_slope(axis, t_stall_nm, no_load_wheel_rad_s,
                    wheel_radius_m, track_width_m, gearing):
    """K_v, the back-EMF damping of the measured torque-speed line.

    The motor law is ``T = T_stall * (c - n / n_nl)`` at motor speed n
    (drivetrain.motor_torque), so the damping the command sees is that
    slope mapped onto the axis (v in m/s straight, omega in rad/s for a
    turn). The gearing cancels out of the speed term and appears once in
    the force term.
    """
    if axis == "straight":
        return (2.0 * t_stall_nm * gearing
                / (wheel_radius_m * wheel_radius_m * no_load_wheel_rad_s))
    return (t_stall_nm * gearing * track_width_m * track_width_m
            / (2.0 * wheel_radius_m * wheel_radius_m * no_load_wheel_rad_s))


def _step_j_estimate(t, q, drive, back_emf_slope):
    """One inertia estimate from one step: (J, B, tau_mech) or None.

    t is seconds, q the measured axis speed and drive the logged
    generalized force over the step. The model is J dq = (G - B q) dt.

    B comes from the step's steady tail (B = G_ss / y_ss, the pinned
    "B + K_v = K0 / y_ss" without the back-EMF), where y_ss is the
    measured steady tail and NOT the first-order fit's asymptote: the
    drivebase's speed loop overshoots slightly, which the fit's amplitude
    carries as a 2-3% high bias on every step.

    J comes from the exact discrete step response of the plant against the
    zero-speed drive (G0 held at its sample): q[i+1] = a q[i] +
    (1 - a) G0[i] / (B + K_v) with a = exp(-(B + K_v) dt / J). Regressing
    the settled offset out of both sides gives a with no quadrature error
    -- integrating the residual numerically biases J at the 10 ms sampling
    of a hub log -- and no derivative of the measured speed is taken
    anywhere.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    drive = np.asarray(drive, dtype=float)
    if len(q) < 10 or not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))
                           and np.all(np.isfinite(drive))):
        return None
    tail = max(3, int(0.3 * len(q)))
    y_ss = float(np.median(q[-tail:]))
    g_ss = float(np.median(drive[-tail:]))
    if abs(y_ss) < 1e-6:
        return None
    damping = g_ss / y_ss
    if not np.isfinite(damping) or damping <= 0.0:
        return None
    steps = np.diff(t)
    if float(np.ptp(steps)) > 1e-6:
        # The exact recursion needs a uniform step; a hub log is one.
        return None
    # Fold the back-EMF into the damping: against the zero-speed drive
    # G0 = G + K_v q the plant is J dq = (G0 - (B + K_v) q) dt, and G0 is
    # what the closed loop held -- piecewise constant between the drive
    # base's control ticks -- so the recursion below is exact on hub data.
    # (B + K_v) = K0 / y_ss is the pinned steady state, and
    # tau_mech = J / (B + K_v) the time constant of the open-loop step.
    total_damping = damping + back_emf_slope
    g0_over_b = (drive + back_emf_slope * q) / total_damping
    x = q[:-1] - g0_over_b[:-1]
    y = q[1:] - g0_over_b[:-1]
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return None
    a = float(np.dot(x, y) / denom)
    if not 0.0 < a < 1.0:
        return None
    j = -total_damping * float(steps[0]) / float(np.log(a))
    if not np.isfinite(j) or j <= 0.0:
        return None
    return float(j), float(damping), float(j / total_damping)


def _axis_j_estimates(fits, axis, measured_key, wheel_radius_m,
                      track_width_m, gearing, t_stall_nm,
                      no_load_wheel_rad_s):
    """One inertia estimate per fitted step, from the logged motor torque.

    Steps without the torque columns are skipped: without the actuation the
    closed-loop step cannot identify anything.
    """
    estimates = []
    back_emf = _back_emf_slope(axis, t_stall_nm, no_load_wheel_rad_s,
                               wheel_radius_m, track_width_m, gearing)
    for fit in fits:
        run = fit.get("_run")
        span = fit.get("_span")
        if run is None or span is None:
            continue
        start, end = span
        left = _column(run, "l_load_mNm")
        right = _column(run, "r_load_mNm")
        measured = _measured_column(run, measured_key)
        times = _column(run, "t_ms")
        if (left is None or right is None or measured is None
                or times is None):
            continue
        left = np.asarray(left[start:end], dtype=float) / 1000.0
        right = np.asarray(right[start:end], dtype=float) / 1000.0
        if axis == "straight":
            drive = gearing * (left + right) / wheel_radius_m
        else:
            # q is clockwise-positive; a clockwise spin drives the left
            # wheel forward and the right wheel backward.
            drive = (gearing * track_width_m * (left - right)
                     / (2.0 * wheel_radius_m))
        estimate = _step_j_estimate(
            np.asarray(times[start:end], dtype=float) / 1000.0,
            np.asarray(measured[start:end], dtype=float),
            drive,
            back_emf,
        )
        if estimate is None:
            continue
        j, damping, tau_mech = estimate
        estimates.append({
            "repeat": fit.get("repeat", 0),
            "j": j,
            "damping": damping,
            "tau_mech_s": tau_mech,
        })
    return estimates


def _fit_body_parameters(straight_fits, turn_fits, stall, drivetrain,
                         wheel_radius_m, track_width_m, gearing, weigh_in):
    """The weigh-in mass and the inertia derived from the turn steps.

    Returns the (mass, inertia) result sections. Without the torque-speed
    line (stall torque and no-load speed) the derivation has no K_v, so
    nothing is derived; without the weigh-in rows the mass stays
    uncalibrated -- an old sheet still fits.
    """
    t_stall = stall.get("t_max_nm") if stall.get("calibrated") else None
    n_nl = (drivetrain.get("no_load_speed_rad_s")
            if drivetrain.get("calibrated") else None)

    derived = []
    turn_estimates = []
    if t_stall is not None and n_nl is not None:
        derived = _axis_j_estimates(
            straight_fits, "straight", "v_meas_m_s", wheel_radius_m,
            track_width_m, gearing, float(t_stall), float(n_nl))
        if track_width_m is not None:
            turn_estimates = _axis_j_estimates(
                turn_fits, "turn", "w_meas_rad_s", wheel_radius_m,
                track_width_m, gearing, float(t_stall), float(n_nl))

    # _between_runs reports mean/std/ci95 over the repeats (KI-010).
    mass = {"calibrated": False, "n_sources": 0}
    if weigh_in:
        stats = _between_runs(list(weigh_in))
        mass = {
            "calibrated": True,
            "n_sources": int(stats.get("n", 0) or 0),
            "mass_kg": stats["mean"],
            "mass_kg_between_runs": stats,
            "source": "weigh_in",
        }
        if stats.get("ci95"):
            low, high = float(stats["ci95"][0]), float(stats["ci95"][1])
            # The floor keeps the interval positive; the raw one is kept
            # alongside it so the clamp can never hide the spread.
            mass["mass_kg_ci95"] = [max(low, 1e-9), high]
            mass["mass_kg_ci95_raw"] = [low, high]
    if derived:
        stats = _between_runs([(e["repeat"], e["j"]) for e in derived])
        mass["derived_mass_kg"] = stats["mean"]
        mass["derived_mass_kg_between_runs"] = stats
        mass["derived_source"] = "straight_steps"
        if mass.get("mass_kg") is not None:
            mass["derived_vs_weighed"] = abs(stats["mean"] - mass["mass_kg"])

    inertia = {"calibrated": False, "n_sources": 0}
    if turn_estimates:
        stats = _between_runs([(e["repeat"], e["j"]) for e in turn_estimates])
        inertia = {
            "calibrated": True,
            "n_sources": len(turn_estimates),
            "inertia_kg_m2": stats["mean"],
            "inertia_kg_m2_between_runs": stats,
            "tau_mech_s": float(np.median(
                [e["tau_mech_s"] for e in turn_estimates])),
            "source": "turn_steps",
        }
        if stats.get("ci95"):
            low, high = float(stats["ci95"][0]), float(stats["ci95"][1])
            inertia["inertia_kg_m2_ci95"] = [max(low, 1e-9), high]
            inertia["inertia_kg_m2_ci95_raw"] = [low, high]
    return mass, inertia


def fit_session(runs, poses=None, wheel_radius_m=0.028, wall_distance_m=None,
                track_width_m=None, gearing=1.0, weigh_in=None):
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
    mass, inertia = _fit_body_parameters(
        straight_fits, turn_fits, stall, drivetrain,
        wheel_radius_m, track_width_m,
        float(gearing) if gearing else 1.0,
        weigh_in or [],
    )

    bad_rows = sum(int(run.get("bad_rows", 0) or 0) for run in runs)
    missing_rows = sum(int(run.get("missing_rows", 0) or 0) for run in runs)

    return {
        "wheel_radius_m": float(wheel_radius_m),
        # The rows the session carried with a non-finite number. They are
        # not data, so they are not in the fit; the count is here so the
        # operator learns that the session file is corrupt.
        "bad_rows": bad_rows,
        "missing_rows": missing_rows,
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
        "mass": mass,
        "inertia": inertia,
        "validation": {
            "chirp_straight_nrmse": chirp_straight,
            "chirp_turn_nrmse": chirp_turn,
        },
        "runs": {
            run["name"]: {
                "run_id": run["meta"].get("run_id"),
                "rows": int(run["data"].shape[0]),
                "expected_rows": run.get("expected_rows"),
                "bad_rows": int(run.get("bad_rows", 0) or 0),
                "missing_rows": int(run.get("missing_rows", 0) or 0),
            }
            for run in runs
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    * gearing: ``gearing`` (motor revolutions per wheel revolution; the
      default of 1 is direct drive)

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

    gearing = _profile_value(sections, "gearing")
    if gearing is not None:
        gearing = _positive_profile_value(gearing, "gearing")
    else:
        gearing = 1.0

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
        "gearing": gearing,
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
        "--write-form",
        help=(
            "write the measurement sheet as a form in a browser; the form "
            "downloads the same CSV the fitter reads"
        ),
    )
    parser.add_argument(
        "--sheet",
        help=(
            "the operator's measurement sheet; it is fitted directly, so "
            "there is no intermediate pose CSV"
        ),
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

    if args.write_form:
        return _write_form(args, parser)

    if args.from_sheet:
        return _convert_sheet(args, parser)

    if args.poses and args.sheet:
        parser.error("use either --poses or --sheet, not both")

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
    gearing = profile.get("gearing") if profile else 1.0

    files, runs = load_runs(args.logs)
    if not runs:
        print("No TEST_START blocks found.")
        return 1

    poses = load_poses(args.poses) if args.poses else {}
    weigh_in = []
    if args.sheet:
        try:
            measured = read_sheet(args.sheet)
            weigh_in = read_weigh_in(args.sheet)
        except OSError as exc:
            print(f"error: cannot read {args.sheet}: {exc}")
            return 1
        poses, problems = poses_from_sheet_rows(measured, runs)
        for problem in problems:
            print(f"warning: {problem}.")
        if not poses and not weigh_in:
            print(f"No measurements in {args.sheet}.")
            return 1
        if weigh_in:
            print(f"Read {len(weigh_in)} weigh-in row(s) from {args.sheet}.")
        if poses:
            print(f"Read {len(poses)} measured run(s) from {args.sheet}.")
    wall_distance = profile.get("wall_distance_m") if profile else None
    result = fit_session(runs, poses=poses, wheel_radius_m=wheel_radius,
                         wall_distance_m=wall_distance,
                         track_width_m=track_width, gearing=gearing,
                         weigh_in=weigh_in)
    result["source"] = files
    result["drivebase"] = {
        "name": profile.get("name") if profile else None,
        "wheel_radius_m": float(wheel_radius),
        "track_width_m": float(track_width) if track_width is not None else None,
        "wall_distance_m": (float(wall_distance)
                            if wall_distance is not None else None),
        "source": profile_path,
    }

    # The fit file is written with allow_nan=False, so a fit that still
    # carries a non-finite number fails loudly here, at write time, with a
    # message that says what is wrong. A file with a bare NaN in it is not
    # JSON, and it fails later, far from the cause.
    buffer = io.StringIO()
    try:
        json.dump(result, buffer, indent=2, allow_nan=False)
    except ValueError as exc:
        print(
            "error: the fit carries a non-finite number ({}), so it was not "
            "written. The session file is corrupt.".format(exc),
            file=sys.stderr,
        )
        return 1
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(buffer.getvalue())
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
    bad_rows = int(result.get("bad_rows", 0) or 0)
    if bad_rows:
        print(
            "Dropped {} row(s) with a non-finite value. They are not data, "
            "so they are not in the fit: the session file is corrupt.".format(
                bad_rows)
        )
    missing_rows = int(result.get("missing_rows", 0) or 0)
    if missing_rows:
        print(
            f"warning: session is missing {missing_rows} row(s) against TEST_END; "
            f"capture may be truncated or dropped BLE packets."
        )
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
            "Odometry: distance_scale={distance_scale:.4f} (the "
            "{distance_scale_source}), heading_scale={heading_scale:.4f}".format(
                **result["odometry"])
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
    mass = result.get("mass", {})
    if mass.get("calibrated"):
        print("Mass: {mass_kg:.4f} kg (weighed {n_sources}x, with battery, "
              "source {source})".format(**mass))
        _print_interval(mass, "mass_kg")
        if "derived_mass_kg" in mass:
            print("  the straight steps say {derived_mass_kg:.4f} kg "
                  "(cross-check)".format(**mass))
    elif "derived_mass_kg" in mass:
        print("Mass: not weighed; the straight steps say "
              "{derived_mass_kg:.4f} kg (cross-check only, nothing landed)"
              .format(**mass))
    inertia = result.get("inertia", {})
    if inertia.get("calibrated"):
        print("Inertia: {inertia_kg_m2:.6f} kg m^2 (from the turn steps, "
              "tau_mech={tau_mech_s:.3f}s)".format(**inertia))
        _print_interval(inertia, "inertia_kg_m2")
    else:
        print("Inertia: not derived (needs the turn steps and the "
              "torque-speed line).")
    print("The intervals cover the run-to-run variation of the robot. They do "
          "not cover the error of the tape measure, which is the same every "
          "time and never averages out.")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
