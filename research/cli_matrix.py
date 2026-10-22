"""Measure the shipping planner, mission by mission, through its own CLI.

`research/harness.py` drives the retired stock spec sets (a guess style per
candidate), so it cannot answer a question about the 3-Pillar frontier
candidates: those are the planner's choice, not a spec's. This drives the CLI
instead - exactly the path that ships - and keeps the one machine-readable
line it prints (`##OVERBOARD-VALIDATION##`). One JSON record per run, written
as the run finishes, so a long matrix can be resumed after an interruption.

The bookkeeping is the whole contribution: no planning or gate logic lives
here, and a record says what the CLI would have said.

Usage::

    python research/cli_matrix.py --missions-from research/results/q14_plant.jsonl \
        --out research/results/q13_plant.jsonl --propagation plant
    python research/cli_matrix.py examples/example_s_curve.json \
        --out research/results/q16_polish.jsonl --polish
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MARKER = "##OVERBOARD-VALIDATION##"
ROOT = Path(__file__).resolve().parents[1]
MISSION_DIR = ROOT / "research" / "missions"


def _names_from(path: Path):
    """The missions a result file already measured, in its own order."""
    names = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                names.add(json.loads(line)["mission"])
    return sorted(names)


def _missions(args) -> list[Path]:
    found = [Path(p) for p in args.missions]
    if args.missions_from:
        names = _names_from(Path(args.missions_from))
        # Suite missions are m000-style; the examples are file names.
        for name in names:
            in_dir = MISSION_DIR / f"{name}.json"
            found.append(in_dir if in_dir.exists() else ROOT / name)
    if not found:
        raise SystemExit("Nothing to run: pass mission files or --missions-from.")
    return found


def run_one(mission: Path, args, work: Path) -> dict:
    out = work / f"{mission.stem}.traj"
    command = [sys.executable, "-m", "overboard.main", "-w", str(mission), "-o", str(out),
               "--validate", "--candidates", str(args.candidates),
               "--propagation", args.propagation]
    if args.polish:
        command.append("--polish")
    started = time.time()
    done = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    wall = round(time.time() - started, 1)
    report = None
    for line in done.stdout.splitlines():
        if line.startswith(MARKER):
            report = json.loads(line[len(MARKER):].strip())
    record = {
        "mission": mission.stem,
        "mission_path": str(mission.relative_to(ROOT)) if mission.is_relative_to(ROOT) else str(mission),
        "propagation": args.propagation,
        "polish": bool(args.polish),
        "candidates": args.candidates,
        "exit": done.returncode,
        "wall_s": wall,
        "polished": done.stdout.count("Polishing the plan"),
        "report": report,
    }
    if report is None:
        # No verdict line: the run stopped before validating. Keep why, so a
        # hole in the matrix is never read as a missing mission.
        record["stdout_tail"] = done.stdout[-600:]
        record["stderr_tail"] = done.stderr[-600:]
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("missions", nargs="*", help="Mission files to run.")
    parser.add_argument("--missions-from",
                        help="A result JSONL whose 'mission' values name the files to run.")
    parser.add_argument("--out", required=True, help="Result JSONL (resumed, not replaced).")
    parser.add_argument("--propagation", choices=("plant", "tick"), default="plant")
    parser.add_argument("--polish", action="store_true", help="Pass --polish to every run.")
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N missions.")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["mission"] for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()}

    missions = _missions(args)
    if args.limit:
        missions = missions[:args.limit]
    todo = [m for m in missions if m.stem not in done]
    print(f"{len(missions)} missions, {len(todo)} to run, propagation={args.propagation} "
          f"polish={args.polish}", flush=True)

    work = Path(tempfile.mkdtemp(prefix="overboard-matrix-"))
    with out_path.open("a", encoding="utf-8") as handle:
        for number, mission in enumerate(todo, 1):
            record = run_one(mission, args, work)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            report = record["report"] or {}
            passing = sum(1 for c in report.get("candidates", []) if c.get("ok"))
            print(f"[{number}/{len(todo)}] {mission.stem} exit={record['exit']} "
                  f"selected={report.get('selected')} passing={passing} "
                  f"polished={record['polished']} wall={record['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
