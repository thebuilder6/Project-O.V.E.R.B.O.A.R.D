"""
Generate 4 New Advanced Diagnostic Plots:
1. plot_curvature_centripetal.png: Curvature & Lateral Acceleration (Cornering G-Forces)
2. plot_phase_portrait_traction.png: (v, omega) Dynamic Kinematic Envelope
3. plot_tracking_reachability_timeseries.png: Time-series tracking error vs. 5mm gate
4. plot_population_pareto_cloud.png: Multi-generation Pareto frontier cloud (Time vs. Error)
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from optimizer import TrajectoryOptimizer
from immrax_validator import ImmraxValidator
from surrogate_polish import CompiledReachabilityEngine, _extract_state_and_dt_arrays
from scratch.run_racing_seed_evolution import (
    RacingGenome,
    build_racing_guess,
    _solve_task,
)

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def compute_kinematics(samples: List[Dict[str, Any]], track_width: float):
    t = np.array([float(s["t"]) for s in samples])
    x = np.array([float(s["x"]) for s in samples])
    y = np.array([float(s["y"]) for s in samples])
    th = np.array([float(s["heading"]) for s in samples])
    vl = np.array([float(s["vl"]) for s in samples])
    vr = np.array([float(s["vr"]) for s in samples])

    v = 0.5 * (vl + vr)
    omega = (vr - vl) / track_width

    # Curvature kappa = omega / max(|v|, 1e-3)
    kappa = np.where(np.abs(v) > 0.05, omega / np.maximum(np.abs(v), 0.05), 0.0)
    # Lateral acceleration a_lat = v * omega
    a_lat = v * omega

    return {
        "t": t, "x": x, "y": y, "th": th,
        "vl": vl, "vr": vr, "v": v, "omega": omega,
        "kappa": kappa, "a_lat": a_lat,
    }


def main():
    print("Loading mission & model...")
    with open("examples/example_complex_mission.json") as f:
        mission = json.load(f)
    with open("examples/fll_choreo.json") as f:
        cfg_dict = json.load(f)

    config = RobotConfig(cfg_dict)
    jax_cfg = JAXRobotConfig(config)
    val = ImmraxValidator(jax_cfg)
    engine = CompiledReachabilityEngine(val)
    optimizer = TrajectoryOptimizer(config)

    waypoints = [(w["x"], w["y"], w.get("heading")) for w in mission]
    stops = [i for i, w in enumerate(mission) if w.get("stop", False)]
    num_segments = len(waypoints) - 1
    segment_samples = optimizer.get_segment_sample_counts(waypoints, 10)
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    unpinned_count = sum(1 for i, w in enumerate(waypoints) if w[2] is None and 0 < i < len(waypoints) - 1)

    print("1. Solving Stock Baseline & Rolling...")
    g_base = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="baseline", stop_waypoint_indices=stops)
    _, s_base, st_base, _ = _solve_task((0, g_base, waypoints, stops, cfg_dict, 150, 0.002))

    g_roll = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="rolling", stop_waypoint_indices=stops)
    _, s_roll, st_roll, _ = _solve_task((0, g_roll, waypoints, stops, cfg_dict, 150, 0.002))

    # Evaluate reachability time series
    states_b, dts_b = _extract_state_and_dt_arrays(s_base, num_segments, segment_samples)
    rep_b = engine.evaluate_states(states_b, dts_b, sample_to_segment)

    states_r, dts_r = _extract_state_and_dt_arrays(s_roll, num_segments, segment_samples)
    rep_r = engine.evaluate_states(states_r, dts_r, sample_to_segment)

    print("2. Evolving Racing Line Candidates for Pareto Population...")
    # Evolve 5 generations of 10 candidates and archive all for the Pareto cloud
    pop_archive = []
    pop = [RacingGenome.create_default(num_segments, unpinned_count)]
    for _ in range(9):
        pop.append(RacingGenome.create_random(num_segments, unpinned_count))

    best_champ_samples = None
    best_champ_rep = None
    best_champ_cost = float("inf")

    for gen in range(6):
        print(f"  Simulating Gen {gen+1}/6...")
        for ind in pop:
            raw_g = build_racing_guess(ind, waypoints, segment_samples, config, stops)
            _, samples, stats, _ = _solve_task((0, raw_g, waypoints, stops, cfg_dict, 120, 0.002))
            if samples:
                states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
                rep = engine.evaluate_states(states, dts, sample_to_segment)
                dur = float(samples[-1]["t"])
                err_cm = float(rep["max_tracking_error_m"]) * 100.0
                tube_cm = float(rep["max_envelope_radius_m"]) * 100.0
                cost = dur + 120.0 * max(0.0, rep["max_tracking_error_m"] - 0.005) + 60.0 * max(0.0, rep["max_envelope_radius_m"] - 0.020)
                pop_archive.append({
                    "gen": gen + 1, "dur": dur, "err_cm": err_cm, "tube_cm": tube_cm, "cost": cost, "samples": samples, "rep": rep
                })
                if cost < best_champ_cost:
                    best_champ_cost = cost
                    best_champ_samples = samples
                    best_champ_rep = rep

        # Sort and breed next
        pop_archive_recent = pop_archive[-10:]
        pop_archive_recent.sort(key=lambda x: x["cost"])
        new_pop = [pop[0]]
        while len(new_pop) < 10:
            p1, p2 = np.random.choice(len(pop), 2, replace=False)
            child = RacingGenome.crossover(pop[p1], pop[p2])
            child = child.mutate(p_mut=0.35, sigma=0.14 * (0.88**gen))
            new_pop.append(child)
        pop = new_pop

    print(f"Archive collected: {len(pop_archive)} evaluated candidates.")
    print(f"Champion Cost: {best_champ_cost:.3f}")

    kin_base = compute_kinematics(s_base, config.track_width)
    kin_roll = compute_kinematics(s_roll, config.track_width)
    kin_champ = compute_kinematics(best_champ_samples, config.track_width)

    v_max = config.v_max_rad_s * config.wheel_radius / max(config.gearing, 1e-9)
    w_max = (2.0 * v_max) / config.track_width
    mu_g = config.cof * 9.81

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # NEW PLOT 1: Curvature & Lateral Acceleration Profile (Cornering G-Forces)
    # =========================================================================
    print("Rendering Plot 1: Curvature & Lateral Acceleration...")
    fig, (ax_lat, ax_curv) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)

    # Lateral Acceleration a_lat = v * omega
    ax_lat.plot(kin_base["t"], np.abs(kin_base["a_lat"]), "--", color="#d62728", lw=1.6, alpha=0.7, label=f"Stock Baseline (Peak: {np.max(np.abs(kin_base['a_lat'])):.2f} m/s²)")
    ax_lat.plot(kin_roll["t"], np.abs(kin_roll["a_lat"]), "-.", color="#9467bd", lw=1.8, alpha=0.8, label=f"Stock Rolling (Peak: {np.max(np.abs(kin_roll['a_lat'])):.2f} m/s²)")
    ax_lat.plot(kin_champ["t"], np.abs(kin_champ["a_lat"]), "-", color="#2ca02c", lw=2.4, label=f"Evolved Champion (Peak: {np.max(np.abs(kin_champ['a_lat'])):.2f} m/s²)")
    ax_lat.axhline(mu_g, color="black", linestyle=":", lw=1.5, label=f"Traction Friction Limit ($\\mu g = {mu_g:.2f}\\text{{ m/s}}^2$)")

    ax_lat.set_title("Cornering Severity: Lateral Acceleration $|a_{\\text{lat}}| = v \\cdot |\\omega|$", fontsize=12, weight="bold")
    ax_lat.set_ylabel("Lateral Accel (m/s²)", fontsize=11)
    ax_lat.set_xlabel("Mission Time (s)", fontsize=11)
    ax_lat.grid(True, linestyle="--", alpha=0.6)
    ax_lat.legend(loc="upper right", frameon=True, fontsize=9)

    # Path Curvature kappa
    ax_curv.plot(kin_base["t"], np.abs(kin_base["kappa"]), "--", color="#d62728", lw=1.6, alpha=0.7, label="Stock Baseline")
    ax_curv.plot(kin_roll["t"], np.abs(kin_roll["kappa"]), "-.", color="#9467bd", lw=1.8, alpha=0.8, label="Stock Rolling")
    ax_curv.plot(kin_champ["t"], np.abs(kin_champ["kappa"]), "-", color="#2ca02c", lw=2.4, label="Evolved Champion (Smoothed Apex Arcs)")

    ax_curv.set_title("Path Curvature Profile $|\\kappa(t)| = |\\omega / v|$", fontsize=12, weight="bold")
    ax_curv.set_ylabel("Curvature (1/m)", fontsize=11)
    ax_curv.set_xlabel("Mission Time (s)", fontsize=11)
    ax_curv.grid(True, linestyle="--", alpha=0.6)
    ax_curv.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    p1 = os.path.join(ARTIFACT_DIR, "plot_curvature_centripetal.png")
    plt.savefig(p1, dpi=200)
    plt.close()

    # =========================================================================
    # NEW PLOT 2: Dynamic Operating Envelope Phase Portrait (v vs. omega)
    # =========================================================================
    print("Rendering Plot 2: Dynamic Traction Phase Portrait...")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)

    # Differential drive kinematic boundary diamond
    v_boundary = np.linspace(-v_max, v_max, 200)
    w_upper = (2.0 / config.track_width) * (v_max - np.abs(v_boundary))
    w_lower = -w_upper

    ax.fill_between(v_boundary, w_lower, w_upper, color="#e8f4f8", alpha=0.6, label="Feasible Differential Kinematic Envelope")
    ax.plot(v_boundary, w_upper, "k-", lw=1.5)
    ax.plot(v_boundary, w_lower, "k-", lw=1.5)

    # Plot trajectories in phase space
    ax.plot(kin_base["v"], kin_base["omega"], "--", color="#d62728", lw=1.6, alpha=0.65, label="Stock Baseline (Interior Cluster)")
    ax.plot(kin_roll["v"], kin_roll["omega"], "-.", color="#9467bd", lw=1.8, alpha=0.75, label="Stock Rolling")
    ax.scatter(kin_champ["v"], kin_champ["omega"], c=kin_champ["t"], cmap="viridis", s=40, zorder=5, label="Evolved Champion (Rides Kinetic Frontier)")
    cb = plt.colorbar(ax.collections[-1], ax=ax)
    cb.set_label("Mission Time (s)", fontsize=10)

    ax.set_title("State-Space Phase Portrait: Linear Velocity $v$ vs. Yaw Rate $\\omega$", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Linear Velocity $v$ (m/s)", fontsize=11)
    ax.set_ylabel("Yaw Rate $\\omega$ (rad/s)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(ARTIFACT_DIR, "plot_phase_portrait_traction.png")
    plt.savefig(p2, dpi=200)
    plt.close()

    # =========================================================================
    # NEW PLOT 3: Tracking Error & Envelope Reachability Time Series
    # =========================================================================
    print("Rendering Plot 3: Reachability Time Series...")
    fig, (ax_err, ax_tube) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)

    tb = np.array([s["t"] for s in s_base])
    tr = np.array([s["t"] for s in s_roll])
    tc = np.array([s["t"] for s in best_champ_samples])

    err_b = np.array(rep_b["tracking_errors"]) * 100.0
    err_r = np.array(rep_r["tracking_errors"]) * 100.0
    err_c = np.array(best_champ_rep["tracking_errors"]) * 100.0

    t_mid_b = 0.5 * (tb[:-1] + tb[1:]) if len(tb) == len(err_b) + 1 else tb[:len(err_b)]
    t_mid_r = 0.5 * (tr[:-1] + tr[1:]) if len(tr) == len(err_r) + 1 else tr[:len(err_r)]
    t_mid_c = 0.5 * (tc[:-1] + tc[1:]) if len(tc) == len(err_c) + 1 else tc[:len(err_c)]

    ax_err.plot(t_mid_b, err_b, "--", color="#d62728", lw=1.6, alpha=0.75, label=f"Stock Baseline (Max: {np.max(err_b):.2f} cm)")
    ax_err.plot(t_mid_r, err_r, "-.", color="#9467bd", lw=1.8, alpha=0.8, label=f"Stock Rolling (Max: {np.max(err_r):.2f} cm)")
    ax_err.plot(t_mid_c, err_c, "-", color="#2ca02c", lw=2.4, label=f"Evolved Champion (Max: {np.max(err_c):.2f} cm)")
    ax_err.axhline(0.50, color="crimson", linestyle=":", lw=1.8, label="Safety Gate Threshold (0.50 cm / 5.0 mm)")

    ax_err.set_title("Closed-Loop Ramsete Tracking Error Along Path ($e_{\\text{track}}$)", fontsize=12, weight="bold")
    ax_err.set_ylabel("Tracking Error (cm)", fontsize=11)
    ax_err.set_xlabel("Time (s)", fontsize=11)
    ax_err.grid(True, linestyle="--", alpha=0.6)
    ax_err.legend(loc="upper right", frameon=True, fontsize=9)

    # Tube Envelope Radius
    tube_b = np.array(rep_b["radii"]) * 100.0
    tube_r = np.array(rep_r["radii"]) * 100.0
    tube_c = np.array(best_champ_rep["radii"]) * 100.0

    ax_tube.plot(t_mid_b, tube_b, "--", color="#d62728", lw=1.6, alpha=0.75, label="Stock Baseline")
    ax_tube.plot(t_mid_r, tube_r, "-.", color="#9467bd", lw=1.8, alpha=0.8, label="Stock Rolling")
    ax_tube.plot(t_mid_c, tube_c, "-", color="#2ca02c", lw=2.4, label=f"Evolved Champion (Max Tube: {np.max(tube_c):.2f} cm)")
    ax_tube.axhline(2.00, color="black", linestyle=":", lw=1.8, label="Safety Guard Limit (2.00 cm)")

    ax_tube.set_title("Immrax Robust Reachability Envelope Radius ($r_{\\text{tube}}$)", fontsize=12, weight="bold")
    ax_tube.set_ylabel("Tube Radius (cm)", fontsize=11)
    ax_tube.set_xlabel("Time (s)", fontsize=11)
    ax_tube.grid(True, linestyle="--", alpha=0.6)
    ax_tube.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    p3 = os.path.join(ARTIFACT_DIR, "plot_tracking_reachability_timeseries.png")
    plt.savefig(p3, dpi=200)
    plt.close()

    # =========================================================================
    # NEW PLOT 4: Multi-Generation Pareto Frontier Cloud
    # =========================================================================
    print("Rendering Plot 4: Pareto Population Cloud...")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

    durs = [p["dur"] for p in pop_archive]
    errs = [p["err_cm"] for p in pop_archive]
    gens = [p["gen"] for p in pop_archive]

    sc = ax.scatter(durs, errs, c=gens, cmap="plasma", s=55, alpha=0.85, edgecolors="none", label="Evolved Population Candidates")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Evolutionary Generation (1 to 6)", fontsize=10)

    # Reference points
    ax.scatter([s_base[-1]["t"]], [np.max(err_b)], color="#d62728", s=130, marker="X", zorder=10, label=f"Stock Baseline ({s_base[-1]['t']:.2f}s, {np.max(err_b):.2f}cm)")
    ax.scatter([s_roll[-1]["t"]], [np.max(err_r)], color="#9467bd", s=130, marker="^", zorder=10, label=f"Stock Rolling ({s_roll[-1]['t']:.2f}s, {np.max(err_r):.2f}cm)")
    ax.scatter([best_champ_samples[-1]["t"]], [np.max(err_c)], color="#2ca02c", s=160, marker="*", zorder=10, label=f"Pareto Champion ({best_champ_samples[-1]['t']:.2f}s, {np.max(err_c):.2f}cm)")

    ax.axhline(0.50, color="crimson", linestyle="--", lw=1.5, label="5 mm Gate Frontier")

    ax.set_title("Pareto Exploration Cloud: Mission Duration vs. Tracking Error", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Trajectory Duration (s) [Lower is Better]", fontsize=11)
    ax.set_ylabel("Max Tracking Error (cm) [Lower is Better]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    p4 = os.path.join(ARTIFACT_DIR, "plot_population_pareto_cloud.png")
    plt.savefig(p4, dpi=200)
    plt.close()

    print("All 4 new advanced plots saved successfully to:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
