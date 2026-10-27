"""
Make a drivebase profile by answering a few questions.

The calibration tools need to know the robot's wheels before anything can be
fitted: the wheel radius, the track width and the gearing, and optionally how
far the start mark is from a wall. This asks for each one with a tip on how
to measure it, checks the answer is plausible for a LEGO robot, and writes
the JSON that every tool takes as ``--profile``:

    python -m overboard.sysid.profile --out my_drivebase.json

The file can also be written by hand (see examples/drivebase_profile.example.json);
the reader accepts both.
"""

import argparse
import json
import os
import sys


# The plausible ranges of a LEGO/FLL robot. A number outside one of these is
# a unit mistake far more often than a real robot (millimetres typed as
# metres, a diameter typed as a radius), so the wizard refuses it and asks
# again instead of writing a profile the fit will silently misread.
WHEEL_RADIUS_MM = (5.0, 100.0)
TRACK_WIDTH_MM = (40.0, 500.0)
GEARING_RANGE = (0.1, 50.0)
WALL_DISTANCE_M = (0.2, 5.0)

DEFAULT_NAME = "my robot"


class AnswerError(ValueError):
    """One answer cannot be a number for a robot; say why and ask again."""


def _number(text, name, low, high, unit):
    try:
        number = float(str(text).strip())
    except (TypeError, ValueError):
        raise AnswerError(f"{name} must be a number in {unit}, got {text!r}")
    if not (low <= number <= high):
        raise AnswerError(
            f"{name} must be between {low:g} and {high:g} {unit}, "
            f"got {number:g}")
    return number


def parse_wheel_radius_mm(text):
    """The wheel radius in millimetres, refused outside the LEGO range."""
    return _number(text, "the wheel radius", WHEEL_RADIUS_MM[0],
                   WHEEL_RADIUS_MM[1], "mm")


def parse_track_width_mm(text):
    """The track width in millimetres, refused outside the LEGO range."""
    return _number(text, "the track width", TRACK_WIDTH_MM[0],
                   TRACK_WIDTH_MM[1], "mm")


def parse_gearing(text):
    """Motor revolutions per wheel revolution, refused outside a sane range."""
    return _number(text, "the gearing", GEARING_RANGE[0], GEARING_RANGE[1],
                   "motor turns per wheel turn")


def parse_wall_distance_m(text):
    """The wall distance in metres; a blank answer skips the wall."""
    if str(text).strip() == "":
        return None
    return _number(text, "the wall distance", WALL_DISTANCE_M[0],
                   WALL_DISTANCE_M[1], "m")


def parse_name(text):
    name = str(text).strip() or DEFAULT_NAME
    return name


def build_profile(name, wheel_radius_m, track_width_m, gearing,
                  wall_distance_m=None):
    """The profile dict, in the shape read_drivebase_profile reads."""
    profile = {
        "name": name,
        "wheel_radius_m": float(wheel_radius_m),
        "track_width_m": float(track_width_m),
        "gearing": float(gearing),
    }
    if wall_distance_m is not None:
        profile["wall"] = {"distance_m": float(wall_distance_m)}
    return profile


def write_profile(profile, path, force=False):
    """Write the profile as JSON. Refuses to overwrite unless asked."""
    with open(path, "w" if force else "x", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2)
        handle.write("\n")
    return path


_TIPS = {
    "wheel_radius": (
        "How to measure it: roll the robot along the floor in a straight line "
        "for one full turn of the wheel, and measure how far it went. Divide "
        "that distance by 6.28 to get the radius. Or hold a ruler across the "
        "wheel at the tyre (the diameter) and halve it."
    ),
    "track_width": (
        "How to measure it: from the centre of one wheel's contact patch to "
        "the centre of the other, along the ground. The fit refines this "
        "number from the turn tests, so a tape measurement is all it needs."
    ),
    "gearing": (
        "What it is: how many times the motor turns for one turn of the "
        "wheel. A motor straight on the wheel is 1. With gears it is the "
        "wheel-side gear teeth divided by the motor-side gear teeth (a "
        "24-tooth motor gear driving a 40-tooth wheel gear is 40/24 = 1.67)."
    ),
    "wall": (
        "What it is: how far the start mark is from a wall the robot can "
        "drive into. Measure it to where the robot's CENTRE rests against "
        "the wall, not to the wall's surface: put the robot against the "
        "wall, mark where its centre is, and measure from that mark. "
        "Press Enter to skip the wall (the homing test is left out then)."
    ),
}


def _ask(input_fn, prompt, parse):
    while True:
        text = input_fn(prompt)
        try:
            return parse(text)
        except AnswerError as exc:
            print(f"  {exc}. Try again.")


def main(argv=None, input_fn=None):
    input_fn = input_fn or input
    parser = argparse.ArgumentParser(
        description="Make a drivebase profile by answering a few questions."
    )
    parser.add_argument("--out", default="my_drivebase.json",
                        help="where to write the profile JSON")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the file if it already exists")
    args = parser.parse_args(argv)

    if not args.force and os.path.exists(args.out):
        print(f"error: {args.out} already exists. Move it away, or pass "
              "--force to overwrite it.", file=sys.stderr)
        return 1

    print("Answer each question; a measurement tip is above it.")
    try:
        name = _ask(input_fn, f"Robot name [{DEFAULT_NAME}]: ", parse_name)
        print(_TIPS["wheel_radius"])
        radius_mm = _ask(input_fn, "Wheel radius (mm): ", parse_wheel_radius_mm)
        print(_TIPS["track_width"])
        track_mm = _ask(input_fn, "Track width (mm): ", parse_track_width_mm)
        print(_TIPS["gearing"])
        gearing = _ask(input_fn, "Gearing (motor turns per wheel turn): ",
                       parse_gearing)
        print(_TIPS["wall"])
        wall = _ask(input_fn, "Wall distance (m, Enter to skip): ",
                    parse_wall_distance_m)
    except EOFError:
        print("\nerror: no answers, nothing written.", file=sys.stderr)
        return 1

    profile = build_profile(name, radius_mm / 1000.0, track_mm / 1000.0,
                            gearing, wall)
    try:
        write_profile(profile, args.out, force=args.force)
    except OSError as exc:
        print(f"error: cannot write {args.out}: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Wrote {args.out}:")
    print("  " + json.dumps(profile))
    print("Check the numbers against the robot before you fit. Next:")
    print(f"  python -m overboard.sysid.collect --conn ble "
          f"--profile {args.out} --session session.csv "
          f"--sheet calibration_sheet.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
