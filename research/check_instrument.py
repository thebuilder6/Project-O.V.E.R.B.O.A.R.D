"""Check the harness against the CLI.

RQ-A's lesson: verify the instrument before believing a large effect. This
compares what the harness records for a candidate with what the CLI's own
machine-readable validation report says about the same candidate, on the same
mission, against the same tree.

    python research/check_instrument.py examples/example_straight.json

With OVERBOARD_TREE set, both sides run against the pinned tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MARKER = "##OVERBOARD-VALIDATION## "


def cli_report(mission: Path, tree: Path) -> dict:
    """Run the CLI and parse its validation report."""
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "main.py", "-w", str(mission), "--validate",
             "--candidates", "5", "-o", str(Path(tmp) / "out.traj")],
            cwd=str(tree), capture_output=True, text=True, timeout=1800)
    for line in result.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
    raise SystemExit(f"no validation report in the CLI output:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def main() -> None:
    mission = Path(sys.argv[1] if len(sys.argv) > 1
                   else "examples/example_straight.json")
    if not mission.is_absolute():
        mission = REPO_ROOT / mission
    tree = Path(os.environ.get("OVERBOARD_TREE") or REPO_ROOT)

    report = cli_report(mission, tree)
    print(f"CLI report on {mission.name} (tree {tree}):")
    for candidate in report["candidates"]:
        print(f"  {candidate['index']} {candidate['style']}/{candidate['timing']}: "
              f"{candidate['duration_s']:.4f}s env {candidate['envelope_cm']:.4f}cm "
              f"trk {candidate['tracking_cm']:.4f}cm ok={candidate['ok']} "
              f"{candidate['reasons']}")

    # The same candidates through the harness.
    from research import harness
    config_data, robot_cfg = harness.cli._load_config(
        str(harness.MEASURE_ROOT / "examples/fll_choreo.json"))
    validator_cls, robot_config_cls = harness.cli._ensure_jax_classes()
    loaded = harness.load_mission(mission)
    specs = harness.build_specs("seeds", 0.0, 10)
    records = harness.run_mission(loaded, specs, robot_cfg, validator_cls,
                                  robot_config_cls, critic=None, workers=4)
    print(f"\nHarness records on {mission.name}:")
    print(f"  {'label':<34}{'dur':>9}{'env cm':>9}{'trk cm':>9}{'ok':>7}")
    for record in records:
        print(f"  {record['label']:<34}{record['duration_s']:9.4f}"
              f"{record['tube_m'] * 100:9.4f}{record['tracking_m'] * 100:9.4f}"
              f"{str(record['ok']):>7}")

    # Compare the two candidates the CLI reported (baseline, both timings).
    mismatches = []
    for candidate in report["candidates"]:
        label = f"{candidate['style']}/{candidate['timing']}"
        match = next((r for r in records if r["label"].startswith(label)), None)
        if match is None:
            mismatches.append(f"{label}: no harness record")
            continue
        for cli_key, harness_key, scale in (
                ("duration_s", "duration_s", 1.0),
                ("envelope_cm", "tube_m", 100.0),
                ("tracking_cm", "tracking_m", 100.0)):
            left = candidate[cli_key]
            right = match[harness_key] * scale
            if abs(left - right) > 1e-3 * max(1.0, abs(left)):
                mismatches.append(
                    f"{label}: {cli_key} cli {left:.6f} vs harness {right:.6f}")
        if bool(candidate["ok"]) != bool(match.get("ok")):
            mismatches.append(f"{label}: ok cli {candidate['ok']} vs harness {match.get('ok')}")
    if mismatches:
        print("\nMISMATCH:")
        for line in mismatches:
            print(f"  {line}")
        raise SystemExit(1)
    print("\nThe harness and the CLI agree on every reported number.")


if __name__ == "__main__":
    main()
