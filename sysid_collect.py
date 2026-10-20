"""
Collect a SysId session from a Pybricks hub over Bluetooth and fit it.

The collector is pushed to the hub, the operator presses the hub button
between tests, and the session is captured as it arrives. The poses the hub
reports are written to a CSV for the operator, and the session is fitted into
sysid_fit.json:

    python sysid_collect.py --conn ble \
        --profile examples/drivebase_profile.example.json

What this automates is the plumbing: the log no longer has to be copied off
the hub by hand, the poses no longer have to be typed into a CSV, and the fit
runs without a second command.

With ``--sheet`` the measurement sheet is written too, as a CSV and as a form
in a browser, so the operator's numbers go in as numbers:

    python sysid_collect.py --conn ble --sheet calibration_sheet.csv

What it does not do is measure the robot. The poses the hub reports are its
own odometry, so they are NOT handed to the fit as a measurement: the
heading scale and the brake slip of the fit carry no information and stay
at their defaults. The distance scale can still come from the wall homing
run, because the wall does not move, but that needs a wall distance in the
profile. Fill the measurement sheet with tape measurements and fit from it:

    python sysid_fit.py session.csv --sheet calibration_sheet.csv \
        --profile examples/drivebase_profile.example.json --out sysid_fit.json

The actuator part of the fit (gain, time constant, command delay, backlash)
comes from the motor logs, so it is calibrated either way.

A dropped Bluetooth link does not lose the run: the session is written as it
arrives, so whatever arrived is fitted.
"""

import argparse
import os
import sys

import sheet_form
from hub_deploy import deploy, HubLinkError
from sysid_fit import (load_runs, poses_from_hub_session, session_tests,
                        sheet_rows, write_poses_csv, write_sheet,
                        main as fit_main)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRIPT = os.path.join(REPO_ROOT, "pybricks_code", "sysid_digital_twin.py")
SHEET_NAME = "calibration_sheet.csv"


def _print_measurement_checklist(session_path):
    """Tell the operator which end poses the session still needs measured."""
    try:
        _files, runs = load_runs([session_path])
    except OSError:
        return
    needed = []
    for (name, repeat), (_run_id, count) in session_tests(runs).items():
        if count > 0:
            needed.append((name, repeat, count))
    if not needed:
        return
    print()
    print("The hub reports its own pose, which is not a measurement. Measure "
          "these with a tape and fill them in:")
    for name, repeat, count in needed:
        print(f"  {name} repeat {repeat}" +
              (f" ({count} stops)" if count > 1 else ""))
    print("  python sysid_fit.py " + session_path +
          f" --write-sheet {SHEET_NAME}")
    print("See docs/CALIBRATION_PROCEDURE.md for the whole procedure.")


def _write_measurement_sheet(session_path, path):
    """Write the blank sheet and the browser form for a collected session.

    ``path`` names the sheet. The CSV is the file the fitter reads and the
    HTML is the form the operator fills in; the form downloads the CSV, so
    the two are the same file by the end.
    """
    _files, runs = load_runs([session_path])
    rows = sheet_rows(runs)
    if not rows:
        return None, None
    stem, ext = os.path.splitext(path)
    if ext.lower() == ".html":
        form_path, sheet_path = path, stem + ".csv"
    else:
        sheet_path, form_path = path, stem + ".html"
    write_sheet(runs, sheet_path)
    fit_command = (
        f"python sysid_fit.py {session_path} --sheet "
        f"{os.path.basename(sheet_path)} --profile my_drivebase.json "
        f"--out sysid_fit.json")
    sheet_form.write_form(rows, form_path,
                          csv_name=os.path.basename(sheet_path),
                          fit_command=fit_command)
    return sheet_path, form_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect a SysId session from a Pybricks hub and fit it."
    )
    parser.add_argument("--conn", choices=("ble", "usb"), default="ble",
                        help="How to reach the hub (default: ble).")
    parser.add_argument("--name", default=None,
                        help="Bluetooth name or address of the hub.")
    parser.add_argument("--script", default=DEFAULT_SCRIPT,
                        help="Collector script to push to the hub.")
    parser.add_argument("--session", default="session.csv",
                        help="Where to write the captured session.")
    parser.add_argument("--poses", default="hub_poses.csv",
                        help="Where to write the poses the hub reported. This "
                             "is not the operator's measured poses.csv, so it "
                             "never overwrites them.")
    parser.add_argument("--profile",
                        help="drivebase profile JSON (wheel_radius_m, "
                             "track_width_m). Without it the session is "
                             "collected but not fitted, because the fit needs "
                             "the wheel radius.")
    parser.add_argument("--wheel-radius", type=float,
                        help="wheel radius in metres; overrides --profile")
    parser.add_argument("--sheet",
                        help="write the measurement sheet here (the CSV the "
                             "fitter reads and the browser form next to it)")
    parser.add_argument("--out", default="sysid_fit.json",
                        help="Where to write the fitted model.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.script):
        parser.error(f"collector script '{args.script}' not found")

    print(f"Pushing {os.path.basename(args.script)} to the hub over {args.conn}.")
    print("Press the hub button when each test asks for it. Do not close this "
          "window; the session is written as it arrives.")
    try:
        lines = deploy(args.script, connection=args.conn, name=args.name,
                       stream_path=args.session)
    except HubLinkError as exc:
        # The session so far is on disk, so fit that instead of losing the
        # whole run to a dropped link.
        print(f"warning: {exc}. Fitting what arrived.", file=sys.stderr)
        lines = []
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Captured {len(lines)} lines of hub output to {args.session}.")

    rows, events = [], []
    if os.path.exists(args.session):
        with open(args.session, "r", encoding="utf-8", errors="replace") as handle:
            rows, events = poses_from_hub_session(handle.read().splitlines())
    if rows:
        write_poses_csv(rows, args.poses)
        print(f"Wrote {len(rows)} pose(s) the hub reported to {args.poses}.")
        if events:
            print("Events: " + ", ".join(events))
    else:
        print("No POSE lines in the session.")

    sheet_path = form_path = None
    if args.sheet:
        try:
            sheet_path, form_path = _write_measurement_sheet(
                args.session, args.sheet)
        except OSError as exc:
            print(f"error: cannot write the measurement sheet: {exc}",
                  file=sys.stderr)
            return 1
        if sheet_path:
            print()
            print(f"Wrote the measurement sheet to {sheet_path} and the form "
                  f"to {form_path}.")
            print("Open the form in a browser, type the numbers the tape "
                  "says, and click Download CSV.")
        else:
            print("The session has no test that needs a measurement, so "
                  "there was no sheet to write.")

    _print_measurement_checklist(args.session)

    if not args.profile and args.wheel_radius is None:
        # The fit needs the wheel radius, so without a profile there is
        # nothing to fit yet. The session is the valuable part and it is on
        # disk, so say what to do next instead of failing.
        print()
        print("No drivebase profile, so the model was not fitted. The session "
              "is on disk; the fit needs the wheel radius:")
        print("  python sysid_fit.py " + args.session +
              " --profile my_drivebase.json --out " + args.out)
        if sheet_path:
            print("  (or, once the sheet is filled in:")
            print(f"  python sysid_fit.py {args.session} --sheet {sheet_path} "
                  f"--profile my_drivebase.json --out {args.out})")
        return 0

    fit_args = [args.session, "--out", args.out]
    if args.profile:
        fit_args += ["--profile", args.profile]
    if args.wheel_radius is not None:
        fit_args += ["--wheel-radius", str(args.wheel_radius)]
    # The hub's own poses are NOT handed to the fit. They are the robot's
    # odometry, so fitting them would calibrate the odometry against itself:
    # the scale would come out at 1.0 because the robot agrees with itself,
    # and the odometry block would claim to be calibrated when it carries no
    # information. The poses are written out for the operator above; the
    # measurement is a tape measurement, which the operator supplies.

    print()
    status = fit_main(fit_args)
    if status == 0:
        print()
        print("The poses the hub reported are the robot's own odometry, so "
              "they were not used as a measurement: the heading scale and "
              "the brake slip carry no information and stay at their "
              "defaults. The distance scale can still come from the wall "
              "homing run, if the profile carries a wall distance. To "
              "calibrate the rest, fill in the sheet and fit from it:")
        if sheet_path:
            print(f"  python sysid_fit.py {args.session} --sheet {sheet_path} "
                  f"--profile {args.profile or 'my_drivebase.json'} "
                  f"--out {args.out}")
        else:
            print("  python sysid_fit.py " + " ".join(fit_args)
                  + " --poses poses.csv")
    return status


if __name__ == "__main__":
    sys.exit(main())
