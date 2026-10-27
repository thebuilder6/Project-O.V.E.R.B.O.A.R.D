"""
Collect a SysId session from a Pybricks hub over Bluetooth and fit it.

The collector is pushed to the hub, the operator presses the hub button
between tests, and the session is captured as it arrives. The poses the hub
reports are written to a CSV for the operator, and the session is fitted into
sysid_fit.json:

    python -m overboard.sysid.collect --conn ble \
        --profile examples/drivebase_profile.example.json

What this automates is the plumbing: the log no longer has to be copied off
the hub by hand, the poses no longer have to be typed into a CSV, and the fit
runs without a second command.

With ``--sheet`` the measurement sheet is written too, as a CSV and as a form
in a browser, so the operator's numbers go in as numbers:

    python -m overboard.sysid.collect --conn ble --sheet calibration_sheet.csv

What it does not do is measure the robot. The poses the hub reports are its
own odometry, so they are NOT handed to the fit as a measurement: the
heading scale and the brake slip of the fit carry no information and stay
at their defaults. The distance scale can still come from the wall homing
run, because the wall does not move, but that needs a wall distance in the
profile. Fill the measurement sheet with tape measurements and fit from it:

    python -m overboard.sysid.fit session.csv --sheet calibration_sheet.csv \
        --profile examples/drivebase_profile.example.json --out sysid_fit.json

The actuator part of the fit (gain, time constant, command delay, backlash)
comes from the motor logs, so it is calibrated either way.

While the suite runs, the terminal says for every test what the robot will
do and what to measure in which columns (the same convention words the
measurement form uses); at the end it prints what the session captured,
what it did not and why, and the exact commands to run next. A dropped
Bluetooth link does not lose the run: the session is written as it
arrives, so whatever arrived is fitted.
"""

import argparse
import os
import sys

from overboard.paths import COLLECTOR_SCRIPT
from overboard.sysid import sheet_form
from overboard.deploy.hub_deploy import deploy, HubLinkError
from overboard.sysid.fit import (load_runs, poses_from_hub_session, session_tests,
                        sheet_rows, write_poses_csv, write_sheet,
                        main as fit_main)

DEFAULT_SCRIPT = str(COLLECTOR_SCRIPT)
SHEET_NAME = "calibration_sheet.csv"


class TestGuide:
    """A step banner for every test the hub starts.

    The banner is the operator's live script: which test is about to run,
    what the robot will do, and - for the measured tests - what to write
    down, in which columns and in the exact words of the measurement
    convention. It is driven by the hub's own ``TEST_START`` lines, so the
    session wire format is untouched.
    """

    MEASURE_MARK = "MEASURE THIS ONE"

    def __init__(self, print_fn=print):
        self._print = print_fn
        self._runs = {}

    def __call__(self, line):
        if not line.startswith("TEST_START"):
            return
        name = line[len("TEST_START"):].strip() or "?"
        run = self._runs.get(name, 0) + 1
        self._runs[name] = run
        self.banner(name, run)

    def banner(self, name, run):
        guide = sheet_form.guide_for(name)
        measured = name in sheet_form.MEASURED_HINTS
        head = f"{name} - run {run}"
        if measured:
            head += f" - {self.MEASURE_MARK}"
        self._print("")
        self._print("=" * 64)
        self._print(head)
        self._print(f"What the robot does: {guide['does']}")
        self._print(f"What to measure: {guide['measure']}")
        if measured:
            self._print(f"The numbers wanted: {guide['numbers']}")
            self._print("Write them down in the sheet's columns:")
            for column in sheet_form.CONVENTION_COLUMNS:
                self._print(f"  {column}")


def _print_closing_summary(session_path, sheet_path, form_path, poses_path,
                           rows, events, profile, out_path):
    """What the run captured, what it did not, and the exact next commands."""
    runs = []
    try:
        _files, runs = load_runs([session_path])
    except OSError:
        runs = []
    tests = session_tests(runs)
    measured = sorted((name, repeat, count)
                      for (name, repeat), (_run_id, count) in tests.items()
                      if count > 0)
    needed = sum(count for _name, _repeat, count in measured)

    print()
    print("=" * 64)
    print("What the session captured:")
    print(f"  {len(runs)} test run(s) in {session_path}.")
    if rows:
        print(f"  {len(rows)} pose line(s) the hub reported, written to "
              f"{poses_path}. They are the robot's own odometry, not a "
              "measurement, so the fit does not take them.")
    else:
        print("  No POSE lines: the hub did not report a pose. That is fine; "
              "the measurement is the tape's job.")
    if events:
        print("  Events: " + ", ".join(events))
    print("What was NOT captured, and why:")
    print("  The weigh-in - it is a bench measurement, not a hub test. Weigh "
          "the robot 3 times with the battery in (pick it up, put it down, "
          "read the scale each time).")
    if measured:
        print(f"  The tape measurements: {needed} number(s) still to enter "
              f"across {len(measured)} test run(s):")
        for name, repeat, count in measured:
            print(f"    {name} repeat {repeat}" +
                  (f" ({count} stops)" if count > 1 else ""))
    print("What to do next:")
    if form_path:
        print(f"  1. Open {form_path} in a browser, type the tape numbers, "
              f"click Download CSV (saves {sheet_path}).")
    elif sheet_path:
        print(f"  1. Fill in {sheet_path} with the tape numbers.")
    else:
        print("  1. Write the measurement sheet first:")
        print(f"     python -m overboard.sysid.fit {session_path} "
              f"--write-sheet {SHEET_NAME}")
        sheet_path = SHEET_NAME
    profile_arg = profile or "my_drivebase.json"
    print("  2. Fit the model:")
    print(f"     python -m overboard.sysid.fit {session_path} --sheet "
          f"{sheet_path} --profile {profile_arg} --out {out_path}")
    print("The whole procedure: docs/CALIBRATION_PROCEDURE.md")


def _hub_error_hints(exc):
    """What to do about the common hub failures, in plain words."""
    text = str(exc)
    hints = []
    if "pybricksdev" in text:
        hints.append("Install the hub tooling with: pip install pybricksdev")
    if "No Pybricks hub" in text or "would not take the connection" in text:
        hints.append("List the hubs Bluetooth can see: "
                     "python -m overboard.deploy.hub_deploy --list-hubs")
        hints.append("Turn the hub on, and close Pybricks Code "
                     "(code.pybricks.com) if it is open.")
    hints.append("The hub must be running the Pybricks firmware to run the "
                 "suite.")
    return hints


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
        f"python -m overboard.sysid.fit {session_path} --sheet "
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

    if args.profile and os.path.exists(args.profile):
        try:
            import json
            from overboard.models.robot_model import RobotConfig
            with open(args.profile, "r", encoding="utf-8") as pf:
                cfg_dict = json.load(pf)
            RobotConfig(cfg_dict).sync_pybricks_robot_config()
        except Exception:
            pass

    print(f"Pushing {os.path.basename(args.script)} to the hub over {args.conn}.")
    print("Press the hub button when each test asks for it. Do not close this "
          "window; the session is written as it arrives.")
    guide = TestGuide()
    try:
        lines = deploy(args.script, connection=args.conn, name=args.name,
                       stream_path=args.session, on_line=guide)
    except HubLinkError as exc:
        # The session so far is on disk, so fit that instead of losing the
        # whole run to a dropped link.
        print(f"warning: {exc}. Fitting what arrived.", file=sys.stderr)
        lines = []
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        for hint in _hub_error_hints(exc):
            print(f"  {hint}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot write the session to {args.session}: {exc}",
              file=sys.stderr)
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

    _print_closing_summary(args.session, sheet_path, form_path, args.poses,
                           rows, events, args.profile, args.out)

    if not args.profile and args.wheel_radius is None:
        # The fit needs the wheel radius, so without a profile there is
        # nothing to fit yet. The session is the valuable part and it is on
        # disk, so say what to do next instead of failing.
        print()
        print("No drivebase profile, so the model was not fitted. The session "
              "is on disk; the fit needs the wheel radius:")
        print("  python -m overboard.sysid.fit " + args.session +
              " --profile my_drivebase.json --out " + args.out)
        if sheet_path:
            print("  (or, once the sheet is filled in:")
            print(f"  python -m overboard.sysid.fit {args.session} --sheet {sheet_path} "
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
            print(f"  python -m overboard.sysid.fit {args.session} --sheet {sheet_path} "
                  f"--profile {args.profile or 'my_drivebase.json'} "
                  f"--out {args.out}")
        else:
            print("  python -m overboard.sysid.fit " + " ".join(fit_args)
                  + " --poses poses.csv")
    return status


if __name__ == "__main__":
    sys.exit(main())
