"""
Re-measure the heuristic matrix on the current tree and record the evidence.

Four configurations per mission, the same four the 2026-10-19 and
2026-10-20-post-ki028 evidence used, so the tables in
``docs/HEURISTICS_AND_VERIFICATION.md`` can be regenerated against the tree
the planner is on now:

  equal-duration  the five seeds under one shared dt
  per-segment     the five seeds under one dt per segment
  auto__sweep     the default set plus the timing x weight grid, per-segment
  full-sweep      the default set plus the grid, both timing modes

One log per run and the parsed validation reports, in the format the earlier
evidence uses. Run it from the repository root:

    python docs/measurements/2026-10-20-variable-samples/run_sweep.py

The tree's fingerprint is taken **before and after** the run. A tree that
changed while the measurement was running is detectable rather than silently
wrong, which is the same rule ``research/harness.py`` follows.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.dirname(os.path.abspath(__file__))

MISSIONS = [
    "example_straight",
    "example_s_curve",
    "example_mission_first4",
    "example_complex_mission",
    "example_with_stops",
    "example_with_events",
]

MODES = {
    "equal-duration": ["--uniform-dt"],
    "per-segment": ["--per-segment-dt"],
    "auto__sweep": ["--per-segment-dt", "--sweep"],
    "full-sweep": ["--uniform-dt", "--sweep"],
}

# The matrix (five seeds, one timing model) and the grids (the default set
# plus the accuracy-weight ladder) are two files, because the tables are
# built from two files.
MATRIX_MODES = ("equal-duration", "per-segment")
GRID_MODES = ("auto__sweep", "full-sweep")

TREE_FILES = [
    "optimizer.py", "immrax_validator.py", "validator.py",
    "robot_model.py", "main.py",
    "pybricks_code/ramsete_fast.py", "pybricks_code/ramsete.py",
    "pybricks_code/ramsete_compact.py",
    "pybricks_code/ramsete_compact_events.py",
]


def run(mission, mode, extra):
    tag = f"{mission}__{mode}"
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, f"{mission}.traj")
        cmd = [sys.executable, "main.py",
               "-c", "examples/fll_choreo.json",
               "-w", f"examples/{mission}.json",
               "-o", out, "--validate", "--candidates", "5"] + extra
        start = time.time()
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        wall = time.time() - start
        stdout = result.stdout
        report = None
        for line in stdout.splitlines():
            if "##OVERBOARD-VALIDATION##" in line:
                report = json.loads(line.split("##OVERBOARD-VALIDATION##", 1)[1])
        header = (
            f"# command: {' '.join(cmd)}\n"
            f"# wall_seconds: {round(wall, 1)}\n"
            f"# exit_code: {result.returncode}\n\n"
        )
        with open(os.path.join(OUT, f"{tag}.log"), "w", encoding="utf-8",
                  newline="\n") as handle:
            handle.write(header + stdout)
            if result.stderr:
                handle.write("\n--- stderr ---\n" + result.stderr)
        return {
            "tag": tag,
            "mission": mission,
            "mode": mode,
            "wall_s": round(wall, 1),
            "exit_code": result.returncode,
            "report": report,
        }


def tree_digests():
    """The fingerprint of the files the numbers depend on.

    It is recorded with the results, so the evidence says which tree it
    describes without anybody having to copy a list by hand.
    """
    digests = {}
    for name in TREE_FILES:
        path = os.path.join(REPO, name)
        try:
            with open(path, "rb") as handle:
                digests[name] = hashlib.md5(handle.read()).hexdigest().upper()
        except OSError:
            digests[name] = "missing"
    return digests


def main():
    matrix, grids = [], []
    before = tree_digests()
    for mission in MISSIONS:
        for mode, extra in MODES.items():
            print(f"running {mission} / {mode} ...", flush=True)
            record = run(mission, mode, extra)
            (matrix if mode in MATRIX_MODES else grids).append(record)
    after = tree_digests()
    for name, records in (("sweep_results.json", matrix),
                          ("full_sweep_results.json", grids)):
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(records, handle, indent=1)
            handle.write("\n")
        print("wrote", path)
    # The tree's fingerprint, so the evidence says which tree it describes.
    path = os.path.join(OUT, "TREE.md5")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for name, digest in after.items():
            handle.write(f"{digest}  {name}\n")
    print("wrote", path)
    if before != after:
        changed = [name for name in before if before[name] != after.get(name)]
        print("\n!! THE TREE CHANGED DURING THE RUN: " + ", ".join(changed))
        print("   The numbers above straddle two trees and cannot be quoted "
              "as one measurement. Re-run the measurement on a settled tree.")
        return 2
    print("\nthe tree did not change during the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
