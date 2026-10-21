from typing import List, Dict, Any
import json
import math
import struct
import numpy as np

from robot_model import RobotConfig, diff_drive_fk, wrap_angle


# Compact Python export for Pybricks MicroPython.
#
# Pybricks disables the array module (MICROPY_PY_ARRAY=0), so the trajectory is
# packed into a single bytes object and decoded on the hub with ustruct. A
# bytes blob is also far smaller than a list of dicts: 24 bytes per sample.
#
# Layout, little-endian:
#   header: <HH  version, count
#   sample: <6f  t, x, y, heading, v, omega
COMPACT_VERSION = 1
COMPACT_HEADER = struct.Struct("<HH")
COMPACT_RECORD = struct.Struct("<6f")
COMPACT_HEADER_SIZE = COMPACT_HEADER.size
COMPACT_RECORD_SIZE = COMPACT_RECORD.size


def encode_compact_trajectory(samples: List[Dict[str, Any]]):
    """Pack trajectory samples into the Pybricks controller format.

    Returns (blob, events) where events is ((sample_index, name), ...).
    """
    count = len(samples)
    if count > 0xFFFF:
        raise ValueError("compact export supports at most 65535 samples")

    blob = bytearray(COMPACT_HEADER_SIZE + count * COMPACT_RECORD_SIZE)
    COMPACT_HEADER.pack_into(blob, 0, COMPACT_VERSION, count)
    events = []
    offset = COMPACT_HEADER_SIZE

    for index, sample in enumerate(samples):
        try:
            t = float(sample["t"])
            x = float(sample["x"])
            y = float(sample["y"])
            heading = float(sample["heading"])
            omega = float(sample["omega"])
            if "vl" in sample and "vr" in sample:
                v = (float(sample["vl"]) + float(sample["vr"])) * 0.5
            else:
                v = float(sample["v"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"compact export: sample {index} has a missing or invalid numeric field"
            ) from exc

        for name, value in (
            ("t", t), ("x", x), ("y", y),
            ("heading", heading), ("v", v), ("omega", omega),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"compact export: sample {index} field {name} is not finite"
                )

        try:
            COMPACT_RECORD.pack_into(blob, offset, t, x, y, heading, v, omega)
        except (OverflowError, struct.error) as exc:
            raise ValueError(
                f"compact export: sample {index} does not fit in float32"
            ) from exc

        if "event" in sample:
            events.append((index, str(sample["event"])))
        offset += COMPACT_RECORD_SIZE

    return bytes(blob), tuple(events)


def _format_bytes_literal(blob: bytes, chunk_size: int = 32) -> str:
    """Format a blob as adjacent bytes literals, which MicroPython folds."""
    if not blob:
        return "samples = b''\n"

    lines = ["samples = ("]
    for start in range(0, len(blob), chunk_size):
        parts = []
        for byte in blob[start:start + chunk_size]:
            if byte == 39:          # single quote
                parts.append("\\'")
            elif byte == 92:        # backslash
                parts.append("\\\\")
            elif 32 <= byte <= 126:
                parts.append(chr(byte))
            else:
                parts.append("\\x%02x" % byte)
        lines.append("    b'" + "".join(parts) + "'")
    lines.append(")")
    return "\n".join(lines) + "\n"


def _format_events_literal(events) -> str:
    """Format ((sample_index, name), ...) as Python source."""
    if not events:
        return "events = ()\n"
    lines = ["events = ("]
    for index, name in events:
        lines.append("    (%d, %r)," % (index, name))
    lines.append(")")
    return "\n".join(lines) + "\n"


def write_compact_python_file(input_traj_file: str, output_file: str) -> None:
    """Write a Pybricks ramsete_compact trajectory module."""
    with open(input_traj_file, "r", encoding="utf-8") as f:
        traj_data = json.load(f)

    samples = traj_data["trajectory"]["samples"]
    rc = RobotConfig(traj_data["trajectory"])
    config_dict = {
        "mass": rc.mass,
        "inertia": rc.inertia,
        "track_width": rc.track_width,
        "wheel_radius": rc.wheel_radius,
        "v_max_rad_s": rc.v_max_rad_s,
        "t_max_nm": rc.t_max_nm,
        "gearing": rc.gearing,
        "cof": rc.cof,
    }

    blob, events = encode_compact_trajectory(samples)

    lines = [
        "# Trajectory exported from FLL Trajectory Optimizer (compact v1)",
        f"# Source: {input_traj_file}",
        f"# Number of samples: {len(samples)}",
        f"# Packed size: {len(blob)} bytes ({COMPACT_RECORD_SIZE} bytes/sample)",
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
    lines.append("# Packed trajectory: <HH version,count> + count * <6f")
    lines.append("# (t, x, y, heading, v, omega), little-endian")
    lines.append(_format_bytes_literal(blob).rstrip("\n"))
    lines.append("")
    lines.append("# Events: (sample_index, name)")
    lines.append(_format_events_literal(events).rstrip("\n"))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(
        f"Exported {len(samples)} samples as a packed blob "
        f"({len(blob)} bytes) to {output_file}"
    )


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]. Delegates to canonical wrap_angle."""
    return float(wrap_angle(angle))


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
    # Unwrap before interpolating: a heading that crosses +/-pi must not be
    # interpolated the long way round (KI-011). The optimizer emits a
    # continuous heading, but a hand-edited or external trajectory may jump
    # from +pi to -pi.
    h_src = np.unwrap(np.array([s["heading"] for s in samples]))
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
        v, omega = diff_drive_fk(vl, vr, track_width)
        
        sample_dict = {
            "t": round(float(t), 6),
            "x": round(lerp(t, t_src, x_src), 6),
            "y": round(lerp(t, t_src, y_src), 6),
            "heading": round(_wrap_pi(lerp(t, t_src, h_src)), 6),
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


def write_python_file(input_traj_file: str, output_file: str, compact: bool = False) -> None:
    """
    Export a trajectory as a Python module.

    The default (compact=False) is the legacy list-of-dicts format used by
    ramsete_fast.py and ramsete.py. compact=True writes the packed blob used
    by ramsete_compact.py and ramsete_compact_events.py.
    """
    if compact:
        write_compact_python_file(input_traj_file, output_file)
        return

    with open(input_traj_file, "r") as f:
        traj_data = json.load(f)

    samples = traj_data["trajectory"]["samples"]

    # Parse the robot parameters the same way the optimizer did (handles old and new key names)
    rc = RobotConfig(traj_data["trajectory"])
    config_dict = {
        "mass": rc.mass,
        "inertia": rc.inertia,
        "track_width": rc.track_width,
        "wheel_radius": rc.wheel_radius,
        "v_max_rad_s": rc.v_max_rad_s,
        "t_max_nm": rc.t_max_nm,
        "gearing": rc.gearing,
        "cof": rc.cof,
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

        # Add event if present.
        # repr(), not an f-string: an event name is a free-text name the
        # operator typed, and pasting it into the source unescaped broke
        # the generated file (a name with a quote in it was a SyntaxError
        # on the hub) and could inject keys into the sample dict (KI-038).
        if "event" in s:
            sample_items.append('    "event": %s' % repr(str(s["event"])))

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
