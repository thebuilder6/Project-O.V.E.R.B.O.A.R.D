"""Random mission suite (Q8).

Every heuristic claim in ``docs/HEURISTICS_AND_VERIFICATION.md`` rests on six
hand-written example paths, and the random benchmark that was supposed to
supply a mission suite is broken (KI-015).  This generator replaces it.

It writes mission files in the same format as ``examples/*.json`` from a fixed
seed, so a measurement run is reproducible.

Missions are stratified over four profiles instead of being drawn uniformly.
The comparisons in the research agenda are about *kinds* of path, and a
uniform random draw would mostly produce near-straight paths:

``even``    segments of similar length, small heading changes
``uneven``  segment lengths that differ by up to 10x (the equal-duration
            timing model's worst case, KI-022)
``turny``   large heading changes (the corner-cutting case)
``mixed``   everything random

Usage::

    python research/suite.py --count 24 --seed 7 --out research/missions
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILES = ("even", "uneven", "turny", "mixed")

# The field the missions live on. The example missions use roughly a metre,
# so this is a little larger: enough room for a multi-segment mission.
FIELD_X = (0.05, 2.00)
FIELD_Y = (0.05, 1.00)

# Minimum distance between two waypoints, so no mission is degenerate.
MIN_SEPARATION = 0.12


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _profile_parameters(profile: str, rng: random.Random) -> Dict[str, Any]:
    """Length and heading-change distribution for one profile."""
    if profile == "even":
        return {
            "length": (0.35, 0.60),
            "turn_deg": (0.0, 25.0),
            "pin": 0.20,
        }
    if profile == "uneven":
        # Half the segments are short, half are long: a 5x-10x spread.
        return {
            "length": None,
            "length_short": (0.12, 0.25),
            "length_long": (0.90, 1.60),
            "turn_deg": (0.0, 35.0),
            "pin": 0.30,
        }
    if profile == "turny":
        return {
            "length": (0.30, 0.70),
            "turn_deg": (55.0, 140.0),
            "pin": 0.60,
        }
    return {
        "length": (0.15, 1.20),
        "turn_deg": (0.0, 60.0),
        "pin": 0.40,
    }


def generate_mission(profile: str, rng: random.Random) -> List[Dict[str, Any]]:
    """Build one mission: a list of waypoint dicts in the examples' format."""
    params = _profile_parameters(profile, rng)
    num_waypoints = rng.randint(3, 8)

    # Walk the path: at every step pick a length and a heading change.
    x = rng.uniform(*FIELD_X)
    y = rng.uniform(*FIELD_Y)
    direction = rng.uniform(-math.pi, math.pi)

    points: List[Dict[str, Any]] = []
    headings: List[Optional[float]] = []

    for index in range(num_waypoints):
        if index > 0:
            if params.get("length") is not None:
                length = rng.uniform(*params["length"])
            else:
                length = (rng.uniform(*params["length_short"])
                          if rng.random() < 0.5
                          else rng.uniform(*params["length_long"]))
            low, high = params["turn_deg"]
            turn = math.radians(rng.uniform(low, high) * rng.choice((-1.0, 1.0)))
            direction += turn
            nx = x + length * math.cos(direction)
            ny = y + length * math.sin(direction)
            # Keep the mission on the field: reflect instead of clamping, so a
            # segment never collapses to zero length.
            if not (FIELD_X[0] <= nx <= FIELD_X[1]):
                direction = math.pi - direction
                nx = x + length * math.cos(direction)
            if not (FIELD_Y[0] <= ny <= FIELD_Y[1]):
                direction = -direction
                ny = y + length * math.sin(direction)
            x = _clamp(nx, *FIELD_X)
            y = _clamp(ny, *FIELD_Y)

        points.append({"x": round(x, 4), "y": round(y, 4)})
        headings.append(None)

    # Headings: pin the first waypoint to the travel direction, pin some
    # interior waypoints, and pin the last one half the time.
    for index in range(num_waypoints):
        incoming = None
        outgoing = None
        if index > 0:
            dx = points[index]["x"] - points[index - 1]["x"]
            dy = points[index]["y"] - points[index - 1]["y"]
            if math.hypot(dx, dy) > 1e-6:
                incoming = math.atan2(dy, dx)
        if index < num_waypoints - 1:
            dx = points[index + 1]["x"] - points[index]["x"]
            dy = points[index + 1]["y"] - points[index]["y"]
            if math.hypot(dx, dy) > 1e-6:
                outgoing = math.atan2(dy, dx)

        if index == 0:
            if incoming is not None or outgoing is not None:
                headings[index] = outgoing if outgoing is not None else incoming
        elif index == num_waypoints - 1:
            if rng.random() < 0.5 and incoming is not None:
                headings[index] = incoming
        elif rng.random() < params["pin"]:
            base = incoming if incoming is not None else outgoing
            if base is not None:
                headings[index] = base + math.radians(rng.uniform(-20.0, 20.0))

    # Stops and precision waypoints.
    for index in range(num_waypoints):
        if 0 < index < num_waypoints - 1 and rng.random() < 0.25:
            points[index]["stop"] = True
        if rng.random() < 0.12:
            points[index]["precision_mode"] = True

    for index, heading in enumerate(headings):
        if heading is not None:
            points[index]["heading"] = round(heading, 4)

    return points


def _separated(points: List[Dict[str, Any]]) -> bool:
    """True when no two waypoints sit on top of each other."""
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx = points[i]["x"] - points[j]["x"]
            dy = points[i]["y"] - points[j]["y"]
            if math.hypot(dx, dy) < MIN_SEPARATION:
                return False
    return True


def build_suite(count: int, seed: int) -> List[Dict[str, Any]]:
    """Build ``count`` missions, round-robin over the four profiles."""
    rng = random.Random(seed)
    missions: List[Dict[str, Any]] = []
    attempts = 0
    while len(missions) < count and attempts < count * 50:
        attempts += 1
        profile = PROFILES[len(missions) % len(PROFILES)]
        points = generate_mission(profile, rng)
        if not _separated(points):
            continue
        missions.append({
            "id": f"m{len(missions):03d}",
            "profile": profile,
            "waypoints": points,
        })
    return missions


def characterize(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Describe a mission: how long it is, how hard it turns, how uneven."""
    lengths = []
    turns = []
    for i in range(len(points) - 1):
        dx = points[i + 1]["x"] - points[i]["x"]
        dy = points[i + 1]["y"] - points[i]["y"]
        lengths.append(math.hypot(dx, dy))
    for i in range(len(points) - 1):
        a = points[i].get("heading")
        b = points[i + 1].get("heading")
        if a is not None and b is not None:
            turns.append(abs((b - a + math.pi) % (2 * math.pi) - math.pi))
    return {
        "path_length_m": round(sum(lengths), 4),
        "segment_length_ratio": round(max(lengths) / max(min(lengths), 1e-6), 2)
                                if lengths else 1.0,
        "max_turn_rad": round(max(turns), 4) if turns else 0.0,
        "pinned_headings": sum(1 for p in points if p.get("heading") is not None),
    }


def write_suite(missions: List[Dict[str, Any]], out_dir: Path) -> None:
    """Write every mission as its own file plus one index file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for mission in missions:
        path = out_dir / f"{mission['id']}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(mission["waypoints"], handle, indent=2)
        entry = {
            "id": mission["id"],
            "profile": mission["profile"],
            "file": path.name,
            "num_waypoints": len(mission["waypoints"]),
            "num_segments": len(mission["waypoints"]) - 1,
        }
        entry.update(characterize(mission["waypoints"]))
        index.append(entry)
    with (out_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump({"missions": index}, handle, indent=2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24,
                        help="How many missions to generate.")
    parser.add_argument("--seed", type=int, default=7,
                        help="RNG seed. The same seed gives the same suite.")
    parser.add_argument("--out", default="research/missions",
                        help="Output directory for the mission files.")
    args = parser.parse_args()

    missions = build_suite(args.count, args.seed)
    out_dir = Path(args.out)
    write_suite(missions, out_dir)
    index_path = out_dir / "index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, "missions": index["missions"]}, handle,
                  indent=2)
    print(f"Wrote {len(missions)} missions to {out_dir}")
    print(f"  {'id':<7}{'profile':<9}{'seg':>4}{'length m':>10}{'ratio':>8}"
          f"{'max turn':>10}{'pinned':>8}")
    for mission in index["missions"]:
        print(f"  {mission['id']:<7}{mission['profile']:<9}"
              f"{mission['num_segments']:>4}{mission['path_length_m']:>10.3f}"
              f"{mission['segment_length_ratio']:>8.1f}"
              f"{mission['max_turn_rad']:>10.3f}{mission['pinned_headings']:>8}")


if __name__ == "__main__":
    main()
