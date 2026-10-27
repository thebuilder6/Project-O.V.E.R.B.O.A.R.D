"""Compare a twin prediction with a trajectory-oracle trace (Q25, B1).

The twin prediction is one pose per plan sample instant (the closed-loop
rollout centre, e.g. the midpoint of ``report["reachability"]["envelope"]``)
plus the reachability tube radius around it. The oracle trace is what
``research/crosscheck/sim_traj.ts`` wrote: 10 ms rows of the simulator's true
pose. Poses are matched at the twin's sample instants by interpolating the
oracle trace (position linearly, heading the short way), which is the
registered pose-vs-time matching rule (sanity gate 4).

Per trajectory at matched sample instants:
  e_pos   = max over samples of |p_twin - p_oracle|          (metres)
  e_head  = max over samples of the short-way heading error  (radians)
  contain = fraction of oracle samples inside the twin's tube
            (needs the twin's tube_radius_m column)

Set aggregates over all pairs: mean and worst e_pos, mean e_head, and overall
containment. Usage:

    python research/crosscheck/compare_traj.py twin.csv oracle.csv [twin oracle ...]

Twin CSV: t,x,y,heading[,tube_radius_m] at sample instants (SI units).
Oracle trace CSV: the sim_traj.ts trace (true_x_m,true_y_m,true_heading_rad).

The last stdout line is machine-readable JSON.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def _load_twin(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "t": float(row["t"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "heading": float(row["heading"]),
                "tube": float(row["tube_radius_m"]) if row.get("tube_radius_m") else None,
            })
    return rows


def _load_oracle(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "t": float(row["t_ms"]) / 1000.0,
                "x": float(row["true_x_m"]),
                "y": float(row["true_y_m"]),
                "heading": float(row["true_heading_rad"]),
            })
    return rows


def _short_way(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _oracle_at(rows, t: float):
    """Interpolate the oracle trace at time t (the registered matching rule)."""
    if t <= rows[0]["t"]:
        lo = hi = rows[0]
    elif t >= rows[-1]["t"]:
        lo = hi = rows[-1]
    else:
        lo, hi = rows[0], rows[-1]
        for i in range(1, len(rows)):
            if rows[i]["t"] >= t:
                lo, hi = rows[i - 1], rows[i]
                break
    span = hi["t"] - lo["t"]
    a = 0.0 if span <= 0 else (t - lo["t"]) / span
    heading = lo["heading"] + a * _short_way(hi["heading"] - lo["heading"])
    return {
        "x": lo["x"] + a * (hi["x"] - lo["x"]),
        "y": lo["y"] + a * (hi["y"] - lo["y"]),
        "heading": heading,
    }


def compare_pair(twin, oracle):
    e_pos = 0.0
    e_head = 0.0
    contained = 0
    n_tubes = 0
    worst = None
    for s in twin:
        o = _oracle_at(oracle, s["t"])
        pos = math.hypot(s["x"] - o["x"], s["y"] - o["y"])
        head = abs(_short_way(s["heading"] - o["heading"]))
        if pos > e_pos:
            e_pos = pos
            worst = {"t": s["t"], "e_pos_m": pos, "e_head_rad": head}
        e_head = max(e_head, head)
        if s["tube"] is not None:
            n_tubes += 1
            if pos <= s["tube"]:
                contained += 1
    return {
        "e_pos_m": e_pos,
        "e_head_rad": e_head,
        "containment": (contained / n_tubes) if n_tubes else None,
        "n_tubes": n_tubes,
        "worst": worst,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or len(argv) % 2 != 0:
        print(__doc__)
        return 2

    pairs = []
    for i in range(0, len(argv), 2):
        twin_path, oracle_path = Path(argv[i]), Path(argv[i + 1])
        twin = _load_twin(twin_path)
        oracle = _load_oracle(oracle_path)
        if not twin or not oracle:
            print(f"empty input: {twin_path} or {oracle_path}")
            return 2
        result = compare_pair(twin, oracle)
        result["twin"] = twin_path.name
        result["oracle"] = oracle_path.name
        pairs.append(result)

    n_pos = len(pairs)
    mean_e_pos = sum(p["e_pos_m"] for p in pairs) / n_pos
    worst_e_pos = max(p["e_pos_m"] for p in pairs)
    mean_e_head = sum(p["e_head_rad"] for p in pairs) / n_pos
    tubes = [p for p in pairs if p["containment"] is not None]
    containment = (sum(p["containment"] for p in tubes) / len(tubes)) if tubes else None

    for p in pairs:
        cont = "n/a" if p["containment"] is None else f"{p['containment']:.3f}"
        print(f"{p['twin']} vs {p['oracle']}: "
              f"e_pos {p['e_pos_m'] * 1000:.2f} mm, e_head {p['e_head_rad']:.4f} rad, "
              f"containment {cont} ({p['n_tubes']} tubes)")
        if p["worst"]:
            w = p["worst"]
            print(f"    worst sample t={w['t']:.3f}s: "
                  f"{w['e_pos_m'] * 1000:.2f} mm, {w['e_head_rad']:.4f} rad")

    print(f"set ({n_pos} trajectories): mean e_pos {mean_e_pos * 1000:.2f} mm, "
          f"worst e_pos {worst_e_pos * 1000:.2f} mm, "
          f"mean e_head {mean_e_head:.4f} rad, "
          f"containment {'n/a' if containment is None else f'{containment:.4f}'}")

    summary = {
        "n_trajectories": n_pos,
        "mean_e_pos_m": mean_e_pos,
        "worst_e_pos_m": worst_e_pos,
        "mean_e_head_rad": mean_e_head,
        "containment": containment,
        "pairs": [{k: v for k, v in p.items()} for p in pairs],
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
