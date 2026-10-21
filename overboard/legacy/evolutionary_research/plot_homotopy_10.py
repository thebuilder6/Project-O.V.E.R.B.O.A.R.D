"""
Plotting script for the 10-Candidate Homotopy Pyramid run.
Renders:
1. plot_homotopy_10_candidates.png: 2D field comparison of all 10 homotopy candidates.
2. plot_homotopy_10_leaderboard.png: Performance leaderboard across all 10 candidates.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Candidate data from ##OVERBOARD-VALIDATION##
candidates = [
    {"index": 1, "duration": 5.711, "envelope_cm": 1.22, "tracking_cm": 0.42, "fitness": 10.604, "ok": True, "style": "pyramid_1"},
    {"index": 2, "duration": 6.499, "envelope_cm": 1.52, "tracking_cm": 1.14, "fitness": 12.380, "ok": False, "style": "pyramid_2"},
    {"index": 3, "duration": 6.499, "envelope_cm": 1.51, "tracking_cm": 0.50, "fitness": 12.278, "ok": True, "style": "pyramid_3"},
    {"index": 4, "duration": 6.369, "envelope_cm": 1.50, "tracking_cm": 0.49, "fitness": 11.843, "ok": True, "style": "pyramid_4"},
    {"index": 5, "duration": 6.355, "envelope_cm": 1.50, "tracking_cm": 0.48, "fitness": 11.822, "ok": True, "style": "pyramid_5"},
    {"index": 6, "duration": 4.961, "envelope_cm": 1.06, "tracking_cm": 0.32, "fitness": 9.549, "ok": True, "style": "pyramid_6 (WINNER)"},
    {"index": 7, "duration": 6.392, "envelope_cm": 1.48, "tracking_cm": 1.13, "fitness": 12.180, "ok": False, "style": "pyramid_7"},
    {"index": 8, "duration": 6.534, "envelope_cm": 1.57, "tracking_cm": 1.85, "fitness": 13.110, "ok": False, "style": "pyramid_8"},
    {"index": 9, "duration": 6.682, "envelope_cm": 1.28, "tracking_cm": 0.45, "fitness": 11.537, "ok": True, "style": "pyramid_9"},
    {"index": 10, "duration": 6.560, "envelope_cm": 1.53, "tracking_cm": 1.36, "fitness": 12.522, "ok": False, "style": "pyramid_10"},
]

# Load waypoints
with open("examples/example_complex_mission.json") as f:
    mission = json.load(f)
waypoints = [(w["x"], w["y"], w.get("heading")) for w in mission]
stops = [i for i, w in enumerate(mission) if w.get("stop", False)]

# Load winner trajectory
traj_path = "output_homotopy/champion.traj"
winner_samples = []
with open(traj_path) as f:
    lines = f.readlines()
# Parse .traj (csv with header)
for line in lines[1:]:
    parts = line.strip().split(",")
    if len(parts) >= 6:
        try:
            winner_samples.append({
                "t": float(parts[0]),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "heading": float(parts[3]),
                "vl": float(parts[4]),
                "vr": float(parts[5]),
            })
        except ValueError:
            pass

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# =========================================================================
# Plot 1: 2D Spatial Mat Path of the 4.961s Winner
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

wx = [w[0] for w in waypoints]
wy = [w[1] for w in waypoints]
ax.scatter(wx, wy, c="#111111", s=95, zorder=5, label="Waypoints")
for i, (x, y, _) in enumerate(waypoints):
    is_stop = i in stops
    lbl = f"W{i}" + (" (STOP)" if is_stop else "")
    ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6),
                fontsize=9, weight="bold", color="#8B0000" if is_stop else "#111111")

if winner_samples:
    cx = [s["x"] for s in winner_samples]
    cy = [s["y"] for s in winner_samples]
    ax.plot(cx, cy, "-", color="#2ca02c", lw=3.2, zorder=4,
            label="Candidate 6 (WINNER: 4.961 s | Envelope 1.06 cm | Tracking 0.32 cm)")

ax.set_title("Homotopy Bank Top 10 Winner: Candidate 6 (Sub-5s Physical Execution)", fontsize=14, weight="bold", pad=12)
ax.set_xlabel("X Field Coordinate (m)", fontsize=11)
ax.set_ylabel("Y Field Coordinate (m)", fontsize=11)
ax.axis("equal")
ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
plt.tight_layout()
p1 = os.path.join(ARTIFACT_DIR, "plot_homotopy_10_candidates.png")
plt.savefig(p1, dpi=200)
plt.close()

# =========================================================================
# Plot 2: 10-Candidate Leaderboard & Gate Status
# =========================================================================
fig, (ax_dur, ax_track) = plt.subplots(2, 1, figsize=(10, 8), dpi=200, sharex=True)

indices = [c["index"] for c in candidates]
durations = [c["duration"] for c in candidates]
trackings = [c["tracking_cm"] for c in candidates]
colors = ["#2ca02c" if c["index"] == 6 else ("#1f77b4" if c["ok"] else "#d62728") for c in candidates]

# Duration Bar
bars_dur = ax_dur.bar(indices, durations, color=colors, edgecolor="black", width=0.55)
ax_dur.set_ylabel("Duration (s)", fontsize=11, weight="bold")
ax_dur.set_title("Homotopy Sieve Top 10 Parallel Candidate Performance", fontsize=13, weight="bold")
ax_dur.grid(True, linestyle="--", alpha=0.5)

# Annotate bars
for bar, c in zip(bars_dur, candidates):
    yval = bar.get_height()
    status_str = "WINNER" if c["index"] == 6 else ("OK" if c["ok"] else "FAIL")
    ax_dur.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}s\n({status_str})",
                ha="center", va="bottom", fontsize=8, weight="bold",
                color="#2ca02c" if c["index"] == 6 else ("#1f77b4" if c["ok"] else "#d62728"))

ax_dur.set_ylim(4.0, 7.5)

# Tracking Error Bar
bars_track = ax_track.bar(indices, trackings, color=colors, edgecolor="black", width=0.55)
ax_track.axhline(0.50, color="crimson", linestyle="--", lw=1.8, label="Tracking Error Gate (0.50 cm / 5.0 mm)")
ax_track.set_ylabel("Tracking Error (cm)", fontsize=11, weight="bold")
ax_track.set_xlabel("Homotopy Candidate Index", fontsize=11, weight="bold")
ax_track.set_xticks(indices)
ax_track.set_xticklabels([f"Cand {i}" for i in indices], rotation=0)
ax_track.grid(True, linestyle="--", alpha=0.5)
ax_track.legend(loc="upper left", frameon=True, fontsize=9)

for bar, c in zip(bars_track, candidates):
    yval = bar.get_height()
    ax_track.text(bar.get_x() + bar.get_width()/2.0, yval + 0.04, f"{yval:.2f}cm",
                  ha="center", va="bottom", fontsize=8, weight="bold")

ax_track.set_ylim(0.0, 2.2)

plt.tight_layout()
p2 = os.path.join(ARTIFACT_DIR, "plot_homotopy_10_leaderboard.png")
plt.savefig(p2, dpi=200)
plt.close()

print("Plots rendered successfully in:", ARTIFACT_DIR)
