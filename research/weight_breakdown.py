"""Per-timing breakdown of the accuracy-weight study (Q6)."""

import json
import sys

rows = []
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        if record.get("record") != "meta":
            rows.append(record)

weights = sorted({r["weight"] for r in rows})
for timing in ("equal-duration", "per-segment"):
    parts = []
    for weight in weights:
        group = [r for r in rows if r["weight"] == weight and r["timing"] == timing]
        if not group:
            continue
        passing = sum(1 for r in group if r["ok"])
        duration = sorted(r["duration_s"] for r in group)
        tracking = sorted(r["tracking_m"] for r in group)
        mid = duration[len(duration) // 2]
        tmid = tracking[len(tracking) // 2]
        parts.append(f"{weight:g}: pass {100 * passing / len(group):3.0f}% "
                     f"dur {mid:5.2f}s trk {tmid * 100:4.2f}cm")
    print(f"{timing}:")
    for part in parts:
        print(f"   {part}")

# The best weight per mission, and what the weight bought.
print("\nper mission: the weight that passed, and the weight that was fastest")
by_mission = {}
for record in rows:
    by_mission.setdefault(record["mission"], []).append(record)
for mission, group in sorted(by_mission.items()):
    passing = [r for r in group if r["ok"]]
    if not passing:
        print(f"  {mission}: nothing passed")
        continue
    fastest_passing = min(passing, key=lambda r: r["duration_s"])
    best_tracking = min(group, key=lambda r: r["tracking_m"])
    print(f"  {mission}: fastest passing a={fastest_passing['weight']:g} "
          f"{fastest_passing['duration_s']:.3f}s trk "
          f"{fastest_passing['tracking_m'] * 100:.2f}cm | "
          f"best tracking a={best_tracking['weight']:g} "
          f"{best_tracking['tracking_m'] * 100:.2f}cm")
