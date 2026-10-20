"""Verify the backlash fix: the backlash must no longer bias the tracking
error, and must now appear in the envelope instead.

Also answers the question the fix raised: on a straight path, what DOES
produce the tracking error now that the backlash lag is gone?
"""
import json
import math
import os
import sys

import numpy as np

PINNED = os.environ.get(
    "OVERBOARD_TREE",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, PINNED)

import immrax_validator as IV  # noqa: E402
from jax_robot_model import JAXRobotConfig  # noqa: E402
from robot_model import RobotConfig  # noqa: E402

CFG = json.load(open(os.path.join(PINNED, "examples", "fll_choreo.json"),
                     encoding="utf-8"))
JAX_CFG = JAXRobotConfig(RobotConfig(CFG))


def straight(length=1.0, v=0.5, n=10):
    """A perfectly straight, constant-speed plan."""
    out = []
    dt = (length / v) / n
    for k in range(n + 1):
        out.append({
            "t": k * dt, "x": v * k * dt, "y": 0.0, "heading": 0.0,
            "vl": v, "vr": v, "omega": 0.0,
            "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
        })
    return out


def validate(samples, backlash=(0.0004, 0.0005)):
    val = IV.ImmraxValidator(JAX_CFG)
    rep = val.validate_trajectory(
        samples, (1.24, 1.51), (0.75, 0.9), backlash,
        (0.0964, 0.0966), (0.0275, 0.0285), (0.0008, 0.0011))
    return rep


print("=== the backlash must not touch the tracking error any more ===")
for length in (0.5, 1.0, 2.0):
    s = straight(length=length)
    a = validate(s, (0.0004, 0.0005))
    b = validate(s, (0.0, 0.0))
    print(f"  {length:.1f} m straight:  backlash on -> track "
          f"{a['max_tracking_error_m']*1000:.4f} mm, env "
          f"{a['max_envelope_radius_m']*1000:.4f} mm")
    print(f"  {'':13}  backlash off -> track "
          f"{b['max_tracking_error_m']*1000:.4f} mm, env "
          f"{b['max_envelope_radius_m']*1000:.4f} mm")
    d_trk = abs(a["max_tracking_error_m"] - b["max_tracking_error_m"])
    d_env = abs(a["max_envelope_radius_m"] - b["max_envelope_radius_m"])
    print(f"  {'':13}  difference: tracking {d_trk*1000:.4f} mm, "
          f"envelope {d_env*1000:.4f} mm")

print("\n=== what drives the straight's tracking error now? ===")
s = straight(length=1.0)
base = validate(s)
print(f"  baseline                          track "
      f"{base['max_tracking_error_m']*1000:8.4f} mm  env "
      f"{base['max_envelope_radius_m']*1000:8.4f} mm")
orig_quant = IV.quantize_drive_command
IV.quantize_drive_command = lambda v, w: (v, w)
nq = validate(s)
IV.quantize_drive_command = orig_quant
print(f"  no command quantisation           track "
      f"{nq['max_tracking_error_m']*1000:8.4f} mm  env "
      f"{nq['max_envelope_radius_m']*1000:8.4f} mm")
print(f"  -> quantisation contributes "
      f"{(base['max_tracking_error_m']-nq['max_tracking_error_m'])*1000:.4f} mm")

print(f"\n  heading error now reported: "
      f"{base.get('max_heading_error_rad', float('nan'))*180/math.pi:.5f} deg")
