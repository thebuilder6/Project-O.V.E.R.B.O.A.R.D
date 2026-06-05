from typing import List, Dict, Any
import json
import numpy as np


def resample_to_fixed_dt(samples: List[Dict[str, Any]], target_dt: float = 0.02, track_width: float = 0.0965) -> List[Dict[str, Any]]:
    """
    Linearly resample a variable-timestep trajectory to a fixed controller dt.

    Returns a list of dicts with keys:
        t, x, y, heading, vl, vr, v, omega, event (optional)
    """
    if not samples:
        return []

    t_src = np.array([s["t"] for s in samples])
    x_src = np.array([s["x"] for s in samples])
    y_src = np.array([s["y"] for s in samples])
    h_src = np.array([s["heading"] for s in samples])
    vl_src = np.array([s["vl"] for s in samples])
    vr_src = np.array([s["vr"] for s in samples])

    # Collect events from source samples
    event_times = {}
    for i, s in enumerate(samples):
        if "event" in s:
            event_times[s["t"]] = s["event"]

    total_t = t_src[-1]
    num_steps = int(np.floor(total_t / target_dt)) + 1
    target_times = np.arange(num_steps) * target_dt

    def lerp(t_query, t_arr, val_arr):
        if t_query <= t_arr[0]:
            return float(val_arr[0])
        if t_query >= t_arr[-1]:
            return float(val_arr[-1])
        idx = int(np.searchsorted(t_arr, t_query)) - 1
        idx = max(0, min(idx, len(t_arr) - 2))
        t0, t1 = t_arr[idx], t_arr[idx + 1]
        frac = (t_query - t0) / (t1 - t0)
        return float(val_arr[idx] + (val_arr[idx + 1] - val_arr[idx]) * frac)

    out = []
    assigned_events = set()  # Track which events have been assigned
    for t in target_times:
        if t > total_t:
            break
        vl = lerp(t, t_src, vl_src)
        vr = lerp(t, t_src, vr_src)
        v = (vl + vr) / 2.0
        omega = (vr - vl) / track_width
        
        sample_dict = {
            "t": round(float(t), 6),
            "x": round(lerp(t, t_src, x_src), 6),
            "y": round(lerp(t, t_src, y_src), 6),
            "heading": round(lerp(t, t_src, h_src), 6),
            "vl": round(vl, 6),
            "vr": round(vr, 6),
            "v": round(v, 6),
            "omega": round(omega, 6),
        }
        
        # Find the closest event to this time
        closest_event_t = None
        closest_event_name = None
        closest_dist = float('inf')
        for event_t, event_name in event_times.items():
            # Skip if already assigned
            if event_t in assigned_events:
                continue
            dist = abs(event_t - t)
            if dist < closest_dist:
                closest_dist = dist
                closest_event_t = event_t
                closest_event_name = event_name
        
        # Include event if it's within one timestep of this sample
        if closest_event_name is not None and closest_dist <= target_dt:
            sample_dict["event"] = closest_event_name
            assigned_events.add(closest_event_t)
        
        out.append(sample_dict)
    return out


def export_controller_json(samples: List[Dict[str, Any]], target_dt: float = 0.02, track_width: float = 0.0965) -> Dict[str, Any]:
    """Return a JSON-serializable dict with controller-ready samples."""
    resampled = resample_to_fixed_dt(samples, target_dt, track_width)
    return {
        "format": "controller_profile",
        "version": 1,
        "dt": target_dt,
        "num_samples": len(resampled),
        "samples": resampled,
    }


def write_controller_file(
    input_traj_file: str, output_file: str, target_dt: float = 0.02, track_width: float = 0.0965
) -> None:
    with open(input_traj_file, "r") as f:
        traj_data = json.load(f)

    samples = traj_data["trajectory"]["samples"]
    ctrl = export_controller_json(samples, target_dt, track_width)

    with open(output_file, "w") as f:
        json.dump(ctrl, f, indent=1)

    print(f"Exported {ctrl['num_samples']} controller samples at dt={target_dt}s to {output_file}")


def write_python_file(input_traj_file: str, output_file: str) -> None:
    """
    Export trajectory samples and robot config as a Python file.

    Output format:
        # Robot configuration
        config = {
            "mass": 0.723,
            "inertia": 0.0024,
            "track_width": 0.0965,
            "wheel_radius": 0.028,
            "v_max_rad_s": 15.7,
            "t_max_nm": 0.04,
            "gearing": 1.0,
            "cof": 0.65
        }

        # Trajectory samples
        samples = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "vl": 0.0, "vr": 0.0, "omega": 0.0},
            {"t": 0.02, "x": 0.01, "y": 0.0, "heading": 0.0, "vl": 0.5, "vr": 0.5, "omega": 0.0},
            # ... more samples
        ]
    """
    with open(input_traj_file, "r") as f:
        traj_data = json.load(f)

    samples = traj_data["trajectory"]["samples"]
    cfg = traj_data["trajectory"].get("robot", traj_data["trajectory"].get("config", {}))

    # Extract robot config parameters robustly
    def get_val(key, default):
        val = cfg.get(key, default)
        if isinstance(val, dict):
            return val.get("val", default)
        return val

    config_dict = {
        "mass": get_val("mass", 0.8),
        "inertia": get_val("inertia", 0.001),
        "track_width": get_val("differentialTrackWidth", 0.0965),
        "wheel_radius": get_val("radius", 0.028),
        "v_max_rad_s": get_val("vmax", 15.7),
        "t_max_nm": get_val("tmax", 0.04),
        "gearing": get_val("gearing", 1.0),
        "cof": get_val("cof", 1.0),
    }

    # Build Python code string
    lines = [
        "# Trajectory exported from FLL Trajectory Optimizer",
        f"# Source: {input_traj_file}",
        f"# Number of samples: {len(samples)}",
        "",
        "# Robot configuration parameters",
        "config = {",
    ]

    for key, value in config_dict.items():
        if isinstance(value, float):
            lines.append(f'    "{key}": {value:.6f},')
        else:
            lines.append(f'    "{key}": {value},')

    lines.append("}")
    lines.append("")
    lines.append("# Trajectory samples")
    lines.append("samples = [")

    for i, s in enumerate(samples):
        # Build sample dict string
        sample_items = []
        for key in ["t", "x", "y", "heading", "vl", "vr", "omega"]:
            if key in s:
                value = s[key]
                if isinstance(value, float):
                    sample_items.append(f'    "{key}": {value:.6f}')
                else:
                    sample_items.append(f'    "{key}": {value}')

        # Add event if present
        if "event" in s:
            sample_items.append(f'    "event": "{s["event"]}"')

        sample_str = "    {" + ",\n".join(sample_items) + "\n  }"

        if i == len(samples) - 1:
            sample_str = sample_str.rstrip()  # Remove trailing comma for last item
        else:
            sample_str += ","

        lines.append(sample_str)

    lines.append("]")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Exported {len(samples)} samples and config to Python file: {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python export.py <input.traj> <output.json> [target_dt]")
        sys.exit(1)
    dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    write_controller_file(sys.argv[1], sys.argv[2], dt)
