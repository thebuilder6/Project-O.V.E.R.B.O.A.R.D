"""Verify the Hermite instrument against the exact circle before trusting it.

If the Hermite is a poor interpolant of the arc, variants 3/4 of exp8 mean
nothing. Compare the Hermite's curve against the true circle and print the
endpoint curvature it implies.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp8_proposal import hermite_reference  # noqa: E402

R = 0.30
ARCLEN = 0.60
V = 0.4
N = 10
DT = (ARCLEN / V) / N
DTHETA = ARCLEN / R


def exact(s):
    th = s / R
    return R * math.sin(th), R * (1.0 - math.cos(th)), th


def sample(k):
    s = ARCLEN * k / N
    x, y, th = exact(s)
    om = V / R
    return {"x": x, "y": y, "heading": th, "vl": V - om * 0.0965 / 2,
            "vr": V + om * 0.0965 / 2, "omega": om}


print(f"arc R={R} length={ARCLEN} n={N} dt={DT:.4f}s dtheta/step={DTHETA/N:.4f} rad")

s0 = sample(0)
s1 = sample(1)

print("\ns   hermite_x  exact_x    dx       hermite_y  exact_y    dy       "
      "herm_head  true_head   d_head")
maxdev = 0.0
for i in range(11):
    s = i / 10.0
    hx, hy, hh, hv, hom, kr = hermite_reference(s0, s1, DT, 0.0965, s)
    arc = s * (ARCLEN / N)
    ex, ey, eth = exact(arc)
    dev = math.hypot(hx - ex, hy - ey)
    maxdev = max(maxdev, dev)
    print(f"{s:.1f} {hx:10.5f} {ex:9.5f} {hx-ex:+8.5f}  {hy:10.5f} {ey:9.5f} "
          f"{hy-ey:+8.5f}  {hh:8.4f} {eth:9.4f} {hh-eth:+8.4f}")

print(f"\nmax deviation from the true circle: {maxdev*1000:.4f} mm "
      f"(chord sagitta = {R*(1-math.cos(DTHETA/N/2))*1000:.4f} mm)")

# endpoint curvature implied by the Hermite vs the true curvature
h0 = hermite_reference(s0, s1, DT, 0.0965, 0.0)
print(f"\nHermite omega at s=0   : {h0[4]:+.4f} rad/s")
print(f"true omega              : {V/R:+.4f} rad/s")
print(f"omega at s=0.5          : "
      f"{hermite_reference(s0, s1, DT, 0.0965, 0.5)[4]:+.4f} rad/s")
print(f"omega at s=1            : "
      f"{hermite_reference(s0, s1, DT, 0.0965, 1.0)[4]:+.4f} rad/s")
