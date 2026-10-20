"""Random robot configurations.

The mission suite varies the path; this varies the machine the path is
planned for. Every constant the planner reads is here - the mass, the
inertia, the wheel radius, the track width, the motor's stall torque and its
top speed, the friction, the gear backlash - because a suite that only
varies the path tests exactly one robot.

The declared uncertainty ranges are the *width* around a constant, so they
move with their constant: a robot with 30 mm wheels has a wheel-radius range
around 30 mm. A range that belongs to nothing (the gear backlash) moves on
its own.

Nothing here is imported by the production code. It is measurement code.

Usage::

    python research/robots.py --config examples/fll_choreo.json --count 8
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from typing import Any, Dict, List, Optional, Tuple

# The constants that move, and how far they can move: (low factor, high
# factor). The factor multiplies the declared value.
PARAMETERS: Dict[str, Tuple[float, float]] = {
    "mass": (0.60, 1.50),
    "inertia": (0.60, 1.60),
    "track_width": (0.85, 1.15),
    "wheel_radius": (0.85, 1.15),
    "v_max_rad_s": (0.85, 1.15),
    "t_max_nm": (0.70, 1.30),
    "cof": (0.75, 1.25),
}

# The declared range that belongs to a constant. A range that is not listed
# stays where it is: it is not a function of the constant.
RANGE_OF: Dict[str, str] = {
    "track_width": "track_width_range",
    "wheel_radius": "wheel_radius_range",
    "inertia": "inertia_kg_m2",
    "cof": "cof",
    "t_max_nm": "t_max_range",
}

# The ranges that belong to nothing: they move on their own.
FREE_RANGES: Dict[str, Tuple[float, float]] = {
    "gear_backlash_m": (0.50, 2.00),
}


def _robot_block(data: Dict[str, Any]) -> Dict[str, Any]:
    """The block that holds the robot's constants.

    A config is canonical (nested under ``robot``), legacy Choreo (nested
    under ``config``) or flat. ``robot_model.RobotConfig`` reads all three,
    so this does too.
    """
    for key in ("robot", "config"):
        block = data.get(key)
        if isinstance(block, dict):
            return block
    return data


def _read(block: Dict[str, Any], key: str) -> Optional[Any]:
    """A constant's value, whether it is plain or wrapped in ``{"val": ...}``."""
    value = block.get(key)
    if isinstance(value, dict) and "val" in value:
        return value["val"]
    return value


def _write(block: Dict[str, Any], key: str, value: Any) -> None:
    """Write a constant back in the shape it was declared in."""
    current = block.get(key)
    if isinstance(current, dict) and "val" in current:
        current["val"] = value
    else:
        block[key] = value


def _uncertainty_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """The block that holds the declared ranges, created if it is missing."""
    ranges = block.get("uncertainties")
    if not isinstance(ranges, dict):
        ranges = {}
        block["uncertainties"] = ranges
    return ranges


def _read_range(block: Dict[str, Any], ranges: Dict[str, Any],
                key: str) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
    """Where a declared range lives, and what it holds."""
    for holder in (ranges, block):
        item = holder.get(key)
        if item is None:
            continue
        if isinstance(item, dict) and "val" in item:
            item = item["val"]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return holder, [float(item[0]), float(item[1])]
    return None, None


def _factor(rng: random.Random, low: float, high: float, spread: float) -> float:
    """The multiplier for one constant.

    ``spread = 0`` is the declared robot, ``spread = 1`` is the full range
    this module lists.
    """
    return 1.0 + float(spread) * (rng.uniform(low, high) - 1.0)


def jitter(config_data: Dict[str, Any], rng: random.Random,
            spread: float = 1.0) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Return the robot's constants, moved away from the declared values.

    The returned config is a copy: the caller's config is untouched. The
    second value is the factor that was applied to each constant, so a
    measurement can say which robot it was run against.
    """
    data = copy.deepcopy(config_data)
    block = _robot_block(data)
    ranges = _uncertainty_block(block)
    factors: Dict[str, float] = {}

    for name, (low, high) in PARAMETERS.items():
        value = _read(block, name)
        range_name = RANGE_OF.get(name)
        holder, declared = _read_range(block, ranges, range_name) if range_name else (None, None)
        if value is None and declared is None:
            continue
        factor = _factor(rng, low, high, spread)
        factors[name] = round(factor, 4)
        if value is not None:
            _write(block, name, float(value) * factor)
        if declared is not None and holder is not None:
            _write(holder, range_name, [declared[0] * factor, declared[1] * factor])

    for name, (low, high) in FREE_RANGES.items():
        holder, declared = _read_range(block, ranges, name)
        if declared is None or holder is None:
            continue
        factor = _factor(rng, low, high, spread)
        factors[name] = round(factor, 4)
        _write(holder, name, [declared[0] * factor, declared[1] * factor])

    return data, factors


def robot_for(config_data: Dict[str, Any], seed: Any, key: str,
              spread: float = 1.0) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """The robot for one (mission, draw) pair. Deterministic.

    The same seed and the same key always give the same robot, so a run is
    reproducible and one mission's robot does not change when another
    mission is added to the run.
    """
    return jitter(config_data, random.Random(f"{seed}|{key}"), spread)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/fll_choreo.json",
                        help="The declared robot config.")
    parser.add_argument("--count", type=int, default=8,
                        help="How many robots to draw.")
    parser.add_argument("--seed", type=int, default=7,
                        help="RNG seed.")
    parser.add_argument("--spread", type=float, default=1.0,
                        help="0 = the declared robot, 1 = the full range.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    names = sorted(PARAMETERS)
    print(f"{'robot':<8}" + "".join(f"{name:>14}" for name in names))
    for index in range(args.count):
        data, factors = robot_for(config_data, args.seed, f"#{index}", args.spread)
        block = _robot_block(data)
        values = []
        for name in names:
            value = _read(block, name)
            values.append(f"{float(value):>14.4f}" if value is not None
                          else f"{'-':>14}")
        print(f"r{index:<7}" + "".join(values))


if __name__ == "__main__":
    main()
