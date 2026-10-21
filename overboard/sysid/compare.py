"""
Compare a planned trajectory with measured stop/final poses.

This is a host-side report, not a validation gate. It answers one calibration
question: after a hardware run, how far is the robot's measured pose from the
pose the trajectory predicted?

Typical use for a single final pose:

    python sysid_compare.py output.traj poses.csv --run-id 4

For a multi-stop run, provide the same waypoint file used to plan it:

    python sysid_compare.py output.traj poses.csv --run-id 12 \
        --waypoints examples/example_with_stops.json

The script maps stop waypoints to trajectory samples using the optimizer's
uniform sampling convention. If that mapping does not fit, pass explicit
trajectory sample indices instead:

    python sysid_compare.py output.traj poses.csv --stop-indices 20,40

Measured poses use the same CSV schema as ``sysid_fit.py``:

    run_id,stop,x_m,y_m,heading_deg

The predicted side can be a Choreo v3 ``.traj`` or a plain CSV with
``t,x,y,heading`` (seconds and radians). ``t_ms`` and ``heading_deg`` are also
accepted. This keeps the report independent of the optimizer that produced the
prediction. A drivebase profile can be supplied for traceability and geometry
checks:

    python sysid_compare.py output.traj poses.csv \
        --profile examples/drivebase_profile.example.json

If the trajectory contains a ``reachability_envelope`` at a stop sample, the
report also says whether the measured pose is inside it. No thresholds are
applied and no files are written.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from overboard.sysid.fit import load_poses, read_drivebase_profile
from overboard.models.robot_model import wrap_angle as _wrap_angle


def _load_predicted_csv(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        fields = {name.strip() for name in reader.fieldnames if name}
        missing = [key for key in ("x", "y") if key not in fields]
        if "t" not in fields and "t_ms" not in fields:
            missing.append("t or t_ms")
        if "heading" not in fields and "heading_deg" not in fields:
            missing.append("heading or heading_deg")
        if missing:
            raise ValueError(f"{path} is missing {', '.join(missing)}")
        samples = []
        for index, row in enumerate(reader):
            row = {(key.strip() if key else key): value
                   for key, value in row.items()}
            try:
                x = float(row["x"])
                y = float(row["y"])
                t = (float(row["t"]) if "t" in fields
                     else float(row["t_ms"]) / 1000.0)
                heading = (float(row["heading"]) if "heading" in fields
                           else math.radians(float(row["heading_deg"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} row {index + 2} has an invalid number") from exc
            samples.append({
                "t": t,
                "x": x,
                "y": y,
                "heading": heading,
            })
    if not samples:
        raise ValueError(f"{path} contains no predicted poses")
    return {"samples": samples, "config": {}}


def load_trajectory_document(path: str) -> Dict[str, Any]:
    """Load samples and optional config from a .traj or predicted-pose CSV."""
    if os.path.splitext(path)[1].lower() == ".csv":
        return _load_predicted_csv(path)

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    trajectory = data.get("trajectory") if isinstance(data, dict) else None
    if not isinstance(trajectory, dict):
        trajectory = data if isinstance(data, dict) else {}
    samples = trajectory.get("samples")

    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} does not contain trajectory.samples")

    required = ("t", "x", "y", "heading")
    for index, sample in enumerate(samples):
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(
                f"{path} sample {index} is missing {', '.join(missing)}")
    config = trajectory.get("config")
    return {
        "samples": samples,
        "config": config if isinstance(config, dict) else {},
    }


def _parse_indices(value: Optional[str], name: str) -> Optional[List[int]]:
    if value is None:
        return None
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            indices.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} must be comma-separated integers") from exc
    if not indices:
        raise ValueError(f"{name} did not contain any indices")
    return indices


def _load_waypoint_stops(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        waypoints = json.load(handle)
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise ValueError(f"{path} must contain at least two waypoints")
    return waypoints


def _waypoint_stop_indices(waypoints: Sequence[Any],
                           explicit: Optional[Sequence[int]]) -> List[int]:
    if explicit is not None:
        indices = list(explicit)
    else:
        indices = [
            index for index, waypoint in enumerate(waypoints)
            if isinstance(waypoint, dict) and waypoint.get("stop")
        ]
        # The trajectory always ends at rest. Include the final waypoint even
        # when the plan does not mark it with "stop": true.
        indices.append(len(waypoints) - 1)

    # A stop at waypoint 0 is the start, not a measured stop after driving.
    indices = [index for index in indices if index > 0]
    indices = sorted(set(indices))
    if not indices:
        raise ValueError("no stop waypoints after the start were found")
    for index in indices:
        if index >= len(waypoints):
            raise ValueError(
                f"stop waypoint index {index} is outside 0..{len(waypoints) - 1}")
    return indices


def _samples_per_segment(samples: Sequence[Dict[str, Any]],
                         waypoints: Sequence[Any],
                         explicit: Optional[int]) -> int:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("--samples-per-segment must be positive")
        return explicit

    intervals = len(samples) - 1
    segments = len(waypoints) - 1
    if segments <= 0 or intervals % segments != 0:
        raise ValueError(
            f"cannot infer samples per segment from {len(samples)} samples and "
            f"{len(waypoints)} waypoints; pass --samples-per-segment")
    return intervals // segments


def resolve_stop_indices(samples: Sequence[Dict[str, Any]],
                         pose_count: int,
                         stop_indices: Optional[str] = None,
                         waypoints_path: Optional[str] = None,
                         stop_waypoints: Optional[str] = None,
                         samples_per_segment: Optional[int] = None) -> List[int]:
    """Return trajectory sample indices for the measured pose rows."""
    explicit_samples = _parse_indices(stop_indices, "--stop-indices")
    if explicit_samples is not None:
        if len(explicit_samples) != pose_count:
            raise ValueError(
                f"--stop-indices has {len(explicit_samples)} entries but the "
                f"pose file has {pose_count} rows")
        for index in explicit_samples:
            if index < 0 or index >= len(samples):
                raise ValueError(
                    f"stop sample index {index} is outside 0..{len(samples) - 1}")
        return explicit_samples

    if waypoints_path is not None:
        waypoints = _load_waypoint_stops(waypoints_path)
        wp_indices = _waypoint_stop_indices(
            waypoints, _parse_indices(stop_waypoints, "--stop-waypoints"))
        per_segment = _samples_per_segment(
            samples, waypoints, samples_per_segment)
        indices = [index * per_segment for index in wp_indices]
        for index in indices:
            if index >= len(samples):
                raise ValueError(
                    f"waypoint-derived sample index {index} is outside "
                    f"0..{len(samples) - 1}")
        if len(indices) != pose_count:
            raise ValueError(
                f"waypoint stops map to {len(indices)} samples but the pose "
                f"file has {pose_count} rows; use --stop-indices to align them")
        return indices

    if pose_count != 1:
        raise ValueError(
            "multiple measured poses need --waypoints or --stop-indices to "
            "map them to trajectory samples")
    return [len(samples) - 1]


# _wrap_angle imported from robot_model.wrap_angle


def _envelope_state(sample: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    envelope = sample.get("reachability_envelope")
    if not isinstance(envelope, dict) or envelope.get("unknown"):
        return None
    keys = ("x_min", "x_max", "y_min", "y_max")
    if not all(key in envelope for key in keys):
        return None
    return tuple(float(envelope[key]) for key in keys)


def _geometry_warnings(profile: Optional[Dict[str, Any]],
                       config: Dict[str, Any]) -> List[str]:
    if not profile or not config:
        return []
    warnings = []
    checks = (
        ("wheel radius", "wheel_radius_m",
         ("wheel_radius", "wheel_radius_m", "radius")),
        ("track width", "track_width_m",
         ("track_width", "track_width_m", "differentialTrackWidth")),
    )
    for label, profile_key, config_keys in checks:
        profile_value = profile.get(profile_key)
        config_value = next(
            (config[key] for key in config_keys
             if config.get(key) is not None), None)
        if profile_value is None or config_value is None:
            continue
        try:
            profile_value = float(profile_value)
            config_value = float(config_value)
        except (TypeError, ValueError):
            continue
        if abs(profile_value - config_value) > max(
                1e-9, 1e-3 * abs(profile_value)):
            warnings.append(
                f"{label} in the profile is {profile_value:g} but the "
                f"trajectory was planned with {config_value:g}")
    return warnings


def compare(trajectory_path: str,
            poses_path: str,
            run_id: Optional[str] = None,
            stop_indices: Optional[str] = None,
            waypoints_path: Optional[str] = None,
            stop_waypoints: Optional[str] = None,
            samples_per_segment: Optional[int] = None,
            profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build comparison rows without printing."""
    if stop_indices is not None and waypoints_path is not None:
        raise ValueError("use either --stop-indices or --waypoints, not both")
    if stop_waypoints is not None and waypoints_path is None:
        raise ValueError("--stop-waypoints requires --waypoints")

    document = load_trajectory_document(trajectory_path)
    samples = document["samples"]
    poses = load_poses(poses_path)
    if not poses:
        raise ValueError(f"{poses_path} contains no poses")

    if run_id is None:
        if len(poses) > 1:
            available = ", ".join(sorted(poses))
            raise ValueError(
                f"{poses_path} contains multiple runs ({available}); "
                "pass --run-id")
        run_id = next(iter(poses))
    if run_id not in poses:
        available = ", ".join(sorted(poses))
        raise ValueError(f"run_id {run_id!r} not found; available: {available}")

    entries = poses[run_id]
    indices = resolve_stop_indices(
        samples,
        len(entries),
        stop_indices=stop_indices,
        waypoints_path=waypoints_path,
        stop_waypoints=stop_waypoints,
        samples_per_segment=samples_per_segment,
    )

    rows = []
    for pose, sample_index in zip(entries, indices):
        sample = samples[sample_index]
        measured_x = float(pose["x_m"])
        measured_y = float(pose["y_m"])
        measured_heading = math.radians(float(pose["heading_deg"]))
        predicted_x = float(sample["x"])
        predicted_y = float(sample["y"])
        predicted_heading = float(sample["heading"])

        position_error = math.hypot(
            predicted_x - measured_x, predicted_y - measured_y)
        heading_error = _wrap_angle(predicted_heading - measured_heading)
        envelope = _envelope_state(sample)
        inside_envelope = None
        if envelope is not None:
            x_min, x_max, y_min, y_max = envelope
            inside_envelope = (
                x_min <= measured_x <= x_max and
                y_min <= measured_y <= y_max
            )

        rows.append({
            "stop": pose["stop"],
            "sample_index": sample_index,
            "t": float(sample["t"]),
            "measured": {
                "x": measured_x,
                "y": measured_y,
                "heading": measured_heading,
            },
            "predicted": {
                "x": predicted_x,
                "y": predicted_y,
                "heading": predicted_heading,
            },
            "position_error_m": position_error,
            "heading_error_rad": heading_error,
            "inside_envelope": inside_envelope,
        })

    position_errors = [row["position_error_m"] for row in rows]
    heading_errors = [abs(row["heading_error_rad"]) for row in rows]
    return {
        "trajectory": trajectory_path,
        "poses": poses_path,
        "run_id": run_id,
        "drivebase": profile,
        "geometry_warnings": _geometry_warnings(
            profile, document.get("config", {})),
        "rows": rows,
        "max_position_error_m": max(position_errors),
        "rms_position_error_m": math.sqrt(
            sum(error * error for error in position_errors) / len(position_errors)),
        "max_heading_error_rad": max(heading_errors),
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(f"Trajectory: {report['trajectory']}")
    print(f"Poses: {report['poses']} (run_id={report['run_id']})")
    drivebase = report.get("drivebase")
    if drivebase:
        label = drivebase.get("name") or "unnamed"
        parts = []
        radius = drivebase.get("wheel_radius_m")
        track = drivebase.get("track_width_m")
        if radius is not None:
            parts.append(f"wheel radius {float(radius) * 1000.0:.2f} mm")
        if track is not None:
            parts.append(f"track width {float(track) * 1000.0:.2f} mm")
        suffix = f" ({', '.join(parts)})" if parts else ""
        print(f"Drivebase: {label}{suffix}")
    for warning in report.get("geometry_warnings", []):
        print(f"WARNING: {warning}")
    print("")
    print(
        "stop  sample      t(s)   measured x/y (mm)     predicted x/y (mm)    "
        "pos err(mm)  heading err(deg)  envelope"
    )
    for row in report["rows"]:
        measured = row["measured"]
        predicted = row["predicted"]
        if row["inside_envelope"] is None:
            envelope = "-"
        else:
            envelope = "inside" if row["inside_envelope"] else "OUTSIDE"
        print(
            f"{row['stop']:>4}  {row['sample_index']:>6}  {row['t']:>7.3f}  "
            f"{measured['x'] * 1000.0:>9.2f}/{measured['y'] * 1000.0:<9.2f}  "
            f"{predicted['x'] * 1000.0:>9.2f}/{predicted['y'] * 1000.0:<9.2f}  "
            f"{row['position_error_m'] * 1000.0:>10.2f}  "
            f"{math.degrees(row['heading_error_rad']):>15.2f}  {envelope}"
        )
    print("")
    print(
        f"Max position error: {report['max_position_error_m'] * 1000.0:.2f} mm | "
        f"RMS: {report['rms_position_error_m'] * 1000.0:.2f} mm | "
        f"Max heading error: "
        f"{math.degrees(report['max_heading_error_rad']):.2f} deg"
    )
    if any(row["inside_envelope"] is False for row in report["rows"]):
        print(
            "WARNING: at least one measured pose is outside the trajectory's "
            "reachability envelope."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a planned trajectory with measured stop/final poses.")
    parser.add_argument(
        "trajectory",
        help="Choreo v3 .traj file or predicted-pose CSV (t,x,y,heading)")
    parser.add_argument("poses", help="CSV with run_id,stop,x_m,y_m,heading_deg")
    parser.add_argument("--run-id", default=None,
                        help="Measured run id; required if the CSV has several")
    parser.add_argument("--stop-indices", default=None,
                        help="Comma-separated trajectory sample indices, in pose order")
    parser.add_argument("--waypoints", default=None,
                        help="Waypoint JSON used to plan the trajectory")
    parser.add_argument("--stop-waypoints", default=None,
                        help="Comma-separated waypoint indices when --waypoints is used")
    parser.add_argument("--samples-per-segment", type=int, default=None,
                        help="Samples per waypoint segment; inferred when possible")
    parser.add_argument(
        "--profile", default=None,
        help="drivebase profile JSON for reporting and geometry checks")
    args = parser.parse_args(argv)

    try:
        profile = (read_drivebase_profile(args.profile)
                   if args.profile else None)
        report = compare(
            args.trajectory,
            args.poses,
            run_id=args.run_id,
            stop_indices=args.stop_indices,
            waypoints_path=args.waypoints,
            stop_waypoints=args.stop_waypoints,
            samples_per_segment=args.samples_per_segment,
            profile=profile,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
