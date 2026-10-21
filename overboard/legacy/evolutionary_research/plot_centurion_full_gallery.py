"""
Generate Complete Centurion Visual Gallery:
1. plot_centurion_velocity_profiles.png: Linear & Differential Wheel Velocities
2. plot_centurion_lateral_accel.png: Curvature & Lateral Acceleration vs. Traction Limit
3. plot_centurion_phase_portrait.png: (v, omega) State-Space Feasibility Diamond
"""

import json
import math
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
from scratch.run_racing_seed_evolution import _solve_task

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 1. Load mission & model
with open("examples/example_complex_mission.json") as f:
    mission = json.load(f)
with open("examples/fll_choreo.json") as f:
    cfg_dict = json.load(f)

config = RobotConfig(cfg_dict)
optimizer = TrajectoryOptimizer(config)
waypoints = [(w["x"], w["y"], w.get("heading")) for w in mission]
stops = [i for i, w in enumerate(mission) if w.get("stop", False)]
num_segments = len(waypoints) - 1
segment_samples = optimizer.get_segment_sample_counts(waypoints, 10)

# 2. Load Stock Baseline
stock_guess = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="rolling", stop_waypoint_indices=stops)
_, s_stock, _, _ = _solve_task((0, stock_guess, waypoints, stops, cfg_dict, 150, 0.002))

# 3. Load Centurion Grand Champion
s_champ = []
with open("output_homotopy/centurion_grand_champion.traj") as f:
    lines = f.readlines()
for line in lines[1:]:
    parts = line.strip().split(",")
    if len(parts) >= 6:
        s_champ.append({
            "t": float(parts[0]),
            "x": float(parts[1]),
            "y": float(parts[2]),
            "heading": float(parts[3]),
            "vl": float(parts[4]),
            "vr": float(parts[5]),
        })

# Compute kinematics
def get_kinematics(samples, tw):
    t = np.array([s["t"] for s in samples])
    vl = np.array([s["vl"] for s in samples])
    vr = np.array([s["vr"] for s in samples])
    v = 0.5 * (vl + vr)
    w = (vr - vl) / tw
    kappa = np.where(np.abs(v) > 0.04, w / np.maximum(np.abs(v), 0.04), 0.0)
    a_lat = v * w
    return t, vl, vr, v, w, kappa, a_lat

t_st, vl_st, vr_st, v_st, w_st, kap_st, alat_st = get_kinematics(s_stock, config.track_width)
t_ch, vl_ch, vr_ch, v_ch, w_ch, kap_ch, alat_ch = get_kinematics(s_champ, config.track_width)

v_max = config.v_max_rad_s * config.wheel_radius / max(config.gearing, 1e-9)
mu_g = config.cof * 9.81

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# =========================================================================
# Plot 1: Linear & Differential Wheel Velocities
# =========================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)

# Linear Speed
ax1.plot(t_st, v_st, "--", color="#d62728", lw=1.8, alpha=0.75, label=f"Stock Baseline ({t_st[-1]:.2f} s)")
ax1.plot(t_ch, v_ch, "-", color="#1f77b4", lw=2.4, label=f"Centurion Champion ({t_ch[-1]:.2f} s)")
ax1.set_title("Centurion Champion vs. Stock: Linear Speed Profile $v(t)$", fontsize=12, weight="bold")
ax1.set_ylabel("Linear Velocity (m/s)", fontsize=11)
ax1.set_xlabel("Time (s)", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper right", frameon=True, fontsize=9)

# Differential Wheel Velocities
ax2.plot(t_ch, vl_ch, color="#2ca02c", lw=1.8, label="Left Wheel $v_l$")
ax2.plot(t_ch, vr_ch, color="#ff7f0e", lw=1.8, label="Right Wheel $v_r$")
ax2.axhline(v_max, color="black", linestyle=":", lw=1.2, label=f"Motor Saturation Limit ($\\pm {v_max:.2f}\\text{{ m/s}}$)")
ax2.axhline(-v_max, color="black", linestyle=":", lw=1.2)
ax2.set_title("Centurion Champion Differential Wheel Speeds", fontsize=12, weight="bold")
ax2.set_ylabel("Wheel Speed (m/s)", fontsize=11)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper right", frameon=True, fontsize=9)

plt.tight_layout()
p1 = os.path.join(ARTIFACT_DIR, "plot_centurion_velocity_profiles.png")
plt.savefig(p1, dpi=200)
plt.close()

# =========================================================================
# Plot 2: Curvature & Lateral Acceleration vs. Traction
# =========================================================================
fig, (ax_lat, ax_curv) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)

# Lateral Accel
ax_lat.plot(t_st, np.abs(alat_st), "--", color="#d62728", lw=1.6, alpha=0.7, label=f"Stock Baseline (Peak: {np.max(np.abs(alat_st)):.2f} m/s²)")
ax_lat.plot(t_ch, np.abs(alat_ch), "-", color="#1f77b4", lw=2.2, label=f"Centurion Champion (Peak: {np.max(np.abs(alat_ch)):.2f} m/s²)")
ax_lat.axhline(mu_g, color="black", linestyle=":", lw=1.5, label=f"Traction Friction Limit ($\\mu g = {mu_g:.2f}\\text{{ m/s}}^2$)")
ax_lat.set_title("Cornering Severity: Lateral Acceleration $|a_{\\text{lat}}| = v \\cdot |\\omega|$", fontsize=12, weight="bold")
ax_lat.set_ylabel("Lateral Accel (m/s²)", fontsize=11)
ax_lat.set_xlabel("Time (s)", fontsize=11)
ax_lat.grid(True, linestyle="--", alpha=0.6)
ax_lat.legend(loc="upper right", frameon=True, fontsize=9)

# Curvature
ax_curv.plot(t_st, np.abs(kap_st), "--", color="#d62728", lw=1.6, alpha=0.7, label="Stock Baseline")
ax_curv.plot(t_ch, np.abs(kap_ch), "-", color="#1f77b4", lw=2.2, label="Centurion Champion (Continuous Clothoid Transitions)")
ax_curv.set_title("Path Curvature Profile $|\\kappa(t)|$", fontsize=12, weight="bold")
ax_curv.set_ylabel("Curvature (1/m)", fontsize=11)
ax_curv.set_xlabel("Time (s)", fontsize=11)
ax_curv.grid(True, linestyle="--", alpha=0.6)
ax_curv.legend(loc="upper right", frameon=True, fontsize=9)

plt.tight_layout()
p2 = os.path.join(ARTIFACT_DIR, "plot_centurion_lateral_accel.png")
plt.savefig(p2, dpi=200)
plt.close()

# =========================================================================
# Plot 3: (v, omega) State-Space Feasibility Diamond Phase Portrait
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 7), dpi=200)

v_boundary = np.linspace(-v_max, v_max, 200)
w_upper = (2.0 / config.track_width) * (v_max - np.abs(v_boundary))
w_lower = -w_upper

ax.fill_between(v_boundary, w_lower, w_upper, color="#e8f4f8", alpha=0.6, label="Feasible Differential Kinematic Envelope")
ax.plot(v_boundary, w_upper, "k-", lw=1.5)
ax.plot(v_boundary, w_lower, "k-", lw=1.5)

ax.plot(v_st, w_st, "--", color="#d62728", lw=1.6, alpha=0.65, label="Stock Baseline (Interior Cluster)")
sc = ax.scatter(v_ch, w_ch, c=t_ch, cmap="plasma", s=45, zorder=5, label="Centurion Champion (Kinematic Frontier)")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Mission Time (s)", fontsize=10)

ax.set_title("State-Space Phase Portrait: Linear Velocity $v$ vs. Yaw Rate $\\omega$", fontsize=13, weight="bold", pad=12)
ax.set_xlabel("Linear Velocity $v$ (m/s)", fontsize=11)
ax.set_ylabel("Yaw Rate $\\omega$ (rad/s)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left", frameon=True, fontsize=9)
plt.tight_layout()
p3 = os.path.join(ARTIFACT_DIR, "plot_centurion_phase_portrait.png")
plt.savefig(p3, dpi=200)
plt.close()

print("All 3 extra centurion plots rendered successfully in:", ARTIFACT_DIR)
