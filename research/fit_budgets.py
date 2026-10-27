"""Fit measured scoring budgets into generated mission files (the M1 rule).

`research/suite.py` writes plain waypoint lists. A validated run needs the
mission's scoring timing (`target_time_s`, `max_time_s`), and those budgets are
measured, never guessed (docs/EXAMPLES.md, the M1 rule). The measurement:
run the candidate triad once with placeholder budgets - the durations are
solve outputs and do not depend on the budget values - then take

    T_ref = fastest converged duration of the triad   (research/m3_matrix.py)
    max   = 2 * T_ref

and rewrite the mission as {"waypoints": [...], "scoring": {...}}.

    python research/fit_budgets.py research/missions/m012.json ...

Measurement only; the mission files it rewrites are generated artifacts.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "##OVERBOARD-VALIDATION##"


def fit(mission: Path):
    raw = json.loads(mission.read_text(encoding="utf-8"))
    waypoints = raw["waypoints"] if isinstance(raw, dict) else raw
    out = Path(r"C:\Users\jumpi\AppData\Local\Temp\opencode") / (mission.stem + ".budget.traj")
    cmd = [sys.executable, "-m", "overboard.main", "-w", str(mission),
           "-o", str(out), "--validate", "--candidates", "3",
           "--target-time", "100", "--max-time", "200"]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if MARKER not in done.stdout:
        print(f"{mission.stem}: no validation line (exit {done.returncode}); "
              f"stderr tail: {done.stderr.strip()[-200:]}")
        return None
    payload = json.loads(done.stdout.split(MARKER)[1].strip().splitlines()[0])
    durs = [c.get("duration_s") for c in payload.get("candidates") or []
            if c.get("converged") and c.get("duration_s")]
    if not durs:
        print(f"{mission.stem}: nothing converged; mission left unwritten")
        return None
    t_ref = min(durs)
    mission.write_text(json.dumps(
        {"waypoints": waypoints,
         "scoring": {"target_time_s": t_ref, "max_time_s": 2.0 * t_ref}},
        indent=2), encoding="utf-8")
    print(f"{mission.stem}: T_ref={t_ref:.4f}s (from {len(durs)} converged: "
          f"{', '.join(f'{d:.3f}' for d in sorted(durs))}) -> target={t_ref:.4f} "
          f"max={2.0 * t_ref:.4f}")
    return t_ref


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("missions", nargs="+")
    args = ap.parse_args(argv)
    for name in args.missions:
        fit(Path(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
