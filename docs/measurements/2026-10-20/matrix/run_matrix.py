"""
Run the candidate matrix on the current tree and record the evidence.

Four configurations per mission:

  default               the default candidate set (two timing models of the
                        baseline seed)
  seeds                 the default set plus the other initial-guess seeds
                        (--candidates 5), so the multi-start is measured at
                        all: the default set never leaves the baseline seed
  refine                the default set plus the Multi-Verse pipeline's
                        refined trajectory as a candidate (--refine)
  sweep                  the default set plus the timing x accuracy-weight grid
  sweep-self-consistent  the same, with the self-consistent collocation (KI-026)

One log per run and one parsed results file, in the format the 2026-10-19
evidence uses. Run it from the repository root:

    python docs/measurements/2026-10-20/matrix/run_matrix.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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
    "default": [],
    "seeds": ["--candidates", "5"],
    "refine": ["--refine"],
    "sweep": ["--sweep"],
    "sweep-self-consistent": ["--sweep", "--self-consistent"],
}


def run(mission, mode, extra):
    tag = f"{mission}__{mode}"
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, f"{mission}.traj")
        cmd = [sys.executable, "main.py",
               "-c", "examples/fll_choreo.json",
               "-w", f"examples/{mission}.json",
               "-o", out, "--validate"] + extra
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
    import hashlib
    files = [
        "optimizer.py", "immrax_validator.py", "validator.py",
        "robot_model.py", "main.py",
        "pybricks_code/ramsete_fast.py", "pybricks_code/ramsete.py",
        "pybricks_code/ramsete_compact.py",
        "pybricks_code/ramsete_compact_events.py",
    ]
    digests = {}
    for name in files:
        path = os.path.join(REPO, name)
        try:
            with open(path, "rb") as handle:
                digests[name] = hashlib.md5(handle.read()).hexdigest().upper()
        except OSError:
            digests[name] = "missing"
    return digests


def main():
    results = []
    digests = tree_digests()
    for mission in MISSIONS:
        for mode, extra in MODES.items():
            print(f"running {mission} / {mode} ...", flush=True)
            results.append(run(mission, mode, extra))
    path = os.path.join(OUT, "matrix_results.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"tree": digests, "results": results}, handle, indent=1)
        handle.write("\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
