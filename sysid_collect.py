"""
Collect a SysId session from a Pybricks hub over Bluetooth and fit it.

The collector is pushed to the hub, the operator presses the hub button
between tests, and the session is captured as it arrives. The poses the hub
reports are converted to a poses.csv and fitted into sysid_fit.json:

    python sysid_collect.py --conn ble \
        --profile examples/drivebase_profile.example.json

What this automates is the plumbing: the log no longer has to be copied off
the hub by hand, the poses no longer have to be typed into a CSV, and the fit
runs without a second command.

What it does not do is measure the robot. The poses the hub reports are its
own odometry, so the distance scale, the heading scale and the brake slip of
the fit carry no information and stay at their defaults. Replace the
generated poses.csv with tape measurements and re-run the fit to calibrate
them:

    python sysid_fit.py session.csv --poses poses.csv \
        --profile examples/drivebase_profile.example.json --out sysid_fit.json

The actuator part of the fit (gain, time constant, command delay, backlash)
comes from the motor logs, so it is calibrated either way.

A dropped Bluetooth link does not lose the run: the session is written as it
arrives, so whatever arrived is fitted.
"""

import argparse
import os
import sys

from hub_deploy import deploy, HubLinkError
from sysid_fit import (load_runs, poses_from_hub_session, session_tests,
                        write_poses_csv, main as fit_main)

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
                             "track_width_m)")
    parser.add_argument("--wheel-radius", type=float,
                        help="wheel radius in metres; overrides --profile")
    parser.add_argument("--out", default="sysid_fit.json",
                        help="Where to write the fitted model.")
    args = parser.parse_args(argv)

    if not args.profile and args.wheel_radius is None:
        parser.error(
            "provide --profile with wheel_radius_m (or --wheel-radius); "
            "copy examples/drivebase_profile.example.json and set it to your "
            "robot"
        )
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

    _print_measurement_checklist(args.session)

    fit_args = [args.session, "--out", args.out]
    if args.profile:
        fit_args += ["--profile", args.profile]
    if args.wheel_radius is not None:
        fit_args += ["--wheel-radius", str(args.wheel_radius)]
    if rows:
        fit_args += ["--poses", args.poses]

    print()
    status = fit_main(fit_args)
    if status == 0:
        print()
        print("The poses are the hub's own odometry, so the distance scale, "
              "the heading scale and the brake slip carry no information: the "
              "robot is only agreeing with itself. To calibrate them, replace "
              "the poses with tape measurements and re-run:")
        print("  python sysid_fit.py " + " ".join(fit_args))
    return status


if __name__ == "__main__":
    sys.exit(main())
