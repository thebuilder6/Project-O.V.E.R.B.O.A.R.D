"""Q13: read the CLI matrix back as a table.

The records are `research/cli_matrix.py`'s: one per mission, carrying the
CLI's own `##OVERBOARD-VALIDATION##` payload. Prints what the shipping
planner did on every mission, then the totals Q13's success criterion needs,
and where the two propagation models disagree.

Usage::

    python research/q13_report.py
    python research/q13_report.py --plant research/results/q13_plant.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["mission"]] = record
    return records


def _number(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def report(propagation: str, records: dict) -> dict:
    print(f"\n== {propagation}: {len(records)} missions")
    print(f"{'mission':10} {'exit':>4} {'sel':>3} {'pass':>4} {'dur_s':>7} "
          f"{'env_cm':>7} {'trk_cm':>7} {'skip':>4} {'wall':>6}")
    passed_missions = 0
    solved = checked = skipped = truncated = 0
    durations = []
    fails = []
    for mission in sorted(records):
        record = records[mission]
        payload = record.get("report") or {}
        candidates = payload.get("candidates") or []
        passing = [c for c in candidates if c.get("ok")]
        selected = payload.get("selected")
        chosen = candidates[selected - 1] if selected and 0 < selected <= len(candidates) else None
        solved += len(candidates)
        skipped += sum(1 for c in candidates if c.get("validation_skipped"))
        checked += sum(1 for c in candidates
                       if not c.get("validation_skipped") and c.get("converged"))
        if chosen is not None:
            truncated += 1 if chosen.get("envelope_truncated") else 0
        if passing:
            passed_missions += 1
            durations.append(chosen["duration_s"])
        else:
            best = min(candidates, key=lambda c: c.get("max_excess_cm") or 0.0) if candidates else None
            fails.append((mission, best.get("max_excess_cm") if best else None,
                          (best.get("reasons") or ["no candidates"])[0] if best else "no candidates"))
        mission_skipped = sum(1 for c in candidates if c.get("validation_skipped"))
        print(f"{mission:10} {record['exit']:>4} {str(selected):>3} {len(passing):>4} "
              f"{_number(chosen['duration_s'] if chosen else None):>7} "
              f"{_number(chosen['envelope_cm'] if chosen else None):>7} "
              f"{_number(chosen['tracking_cm'] if chosen else None):>7} "
              f"{mission_skipped:>4} {record['wall_s']:>5}s")
    print(f"\nmissions passing: {passed_missions}/{len(records)}   "
          f"candidates: {solved} solved, {checked} checked, {skipped} skipped")
    if durations:
        print(f"selected duration: mean {sum(durations) / len(durations):.2f} s, "
              f"min {min(durations):.2f} s, max {max(durations):.2f} s")
    print(f"selected plan's envelope was cut off at the guard: {truncated}")
    if fails:
        print("missions with no passing candidate (mission, smallest excess cm, first reason):")
        for mission, excess, reason in fails:
            print(f"  {mission}  {_number(excess)}  {reason}")
    return {"passed": set(m for m in records
                          if any(c.get("ok") for c in (records[m].get("report") or {}).get("candidates", [])))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", default=str(ROOT / "research/results/q13_plant.jsonl"))
    parser.add_argument("--tick", default=str(ROOT / "research/results/q13_tick.jsonl"))
    args = parser.parse_args()
    seen = {}
    for propagation, path in (("plant", args.plant), ("tick", args.tick)):
        if Path(path).exists():
            seen[propagation] = report(propagation, load(Path(path)))
    if len(seen) == 2:
        plant, tick = seen["plant"]["passed"], seen["tick"]["passed"]
        # KI-054 lives here: the model changes the envelope, so the two
        # verdicts can differ on the same plan.
        if plant != tick:
            print(f"\nmodel disagrees on {len(plant ^ tick)} mission(s): "
                  f"only plant {sorted(plant - tick)}, only tick {sorted(tick - plant)}")
        else:
            print("\nthe two models agree on every mission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
