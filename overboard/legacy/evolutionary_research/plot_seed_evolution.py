"""
Plotting script for Seed-Level / Starting-Value Evolutionary Algorithm.
Generates comprehensive figures and saves them to the conversation artifact directory.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from typing import Any, List, Dict, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from optimizer import TrajectoryOptimizer
from immrax_validator import ImmraxValidator
from surrogate_polish import CompiledReachabilityEngine, _extract_state_and_dt_arrays
from scratch.test_seed_evolution import (
    InitialGuessGenome,
    build_guess_from_genome,
    evaluate_guess,
)

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def run_and_plot():
    # 1. Setup
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
    unpinned_count = sum(1 for i, w in enumerate(waypoints) if w[2] is None and 0 < i < len(waypoints) - 1)

    print("Running baseline seed...")
    guess_base = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="baseline", stop_waypoint_indices=stops)
    samples_base, cost_base, meta_base = evaluate_guess(guess_base, waypoints, config, optimizer, engine, stops, max_iter=150)

    print("Running rolling seed...")
    guess_rolling = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="rolling", stop_waypoint_indices=stops)
    samples_rolling, cost_rolling, meta_rolling = evaluate_guess(guess_rolling, waypoints, config, optimizer, engine, stops, max_iter=150)

    print("Running starting-value evolutionary loop...")
    pop_size = 6
    num_gens = 3
    pop = [InitialGuessGenome.create_default(num_segments, unpinned_count)]
    for _ in range(pop_size - 1):
        pop.append(InitialGuessGenome.create_random(num_segments, unpinned_count))

    gen_bests = []
    gen_means = []
    best_overall_cost = float("inf")
    best_overall_samples = None
    best_overall_meta = None
    best_genome = None
    best_guess_raw = None

    for gen in range(num_gens):
        gen_evals = []
        for idx, ind in enumerate(pop):
            raw_g = build_guess_from_genome(ind, waypoints, segment_samples, config, stops)
            samples, cost, meta = evaluate_guess(raw_g, waypoints, config, optimizer, engine, stops, max_iter=120)
            gen_evals.append((ind, cost, samples, meta, raw_g))

        gen_evals.sort(key=lambda x: x[1])
        g_best = gen_evals[0]
        gen_bests.append(g_best[1])
        gen_means.append(np.mean([e[1] for e in gen_evals if e[1] < 900]))

        if g_best[1] < best_overall_cost:
            best_overall_cost = g_best[1]
            best_overall_samples = g_best[2]
            best_overall_meta = g_best[3]
            best_genome = g_best[0]
            best_guess_raw = g_best[4]

        # Breed
        new_pop = [gen_evals[0][0], gen_evals[1][0]]
        while len(new_pop) < pop_size:
            c1, c2 = np.random.choice(len(gen_evals[:4]), size=2, replace=False)
            child = InitialGuessGenome.crossover(gen_evals[c1][0], gen_evals[c2][0])
            child = child.mutate(p_mut=0.35, sigma=0.15)
            new_pop.append(child)
        pop = new_pop

    print(f"Done! Best cost: {best_overall_cost:.3f}")

    # -------------------------------------------------------------
    # Plot 1: 2D Spatial Mat Paths (Field Trajectory Comparison)
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Waypoints
    wx = [w[0] for w in waypoints]
    wy = [w[1] for w in waypoints]
    ax.scatter(wx, wy, c="#333333", s=90, zorder=5, label="Waypoints")
    for i, (x, y, _) in enumerate(waypoints):
        is_stop = i in stops
        lbl = f"W{i}" + (" (STOP)" if is_stop else "")
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, weight="bold", color="#8B0000" if is_stop else "#111111")

    # Baseline
    if samples_base:
        bx = [s["x"] for s in samples_base]
        by = [s["y"] for s in samples_base]
        ax.plot(bx, by, "--", color="#d62728", lw=2.0, alpha=0.8,
                label=f"Stock Baseline ({meta_base['duration']:.2f}s, cost {meta_base['cost']:.2f})")

    # Rolling
    if samples_rolling:
        rx = [s["x"] for s in samples_rolling]
        ry = [s["y"] for s in samples_rolling]
        ax.plot(rx, ry, "-.", color="#9467bd", lw=2.2, alpha=0.85,
                label=f"Stock Rolling ({meta_rolling['duration']:.2f}s, cost {meta_rolling['cost']:.2f})")

    # Evolved Initial Guess (Before IPOPT)
    if best_guess_raw is not None:
        N = sum(segment_samples) + 1
        guess_states = best_guess_raw[num_segments:].reshape((N, 5))
        ax.plot(guess_states[:, 0], guess_states[:, 1], ":", color="#ff7f0e", lw=2.0,
                label="Evolved Guess Genome $z_0$ (Pre-IPOPT)")

    # Evolved Champion Solved Trajectory
    if best_overall_samples:
        cx = [s["x"] for s in best_overall_samples]
        cy = [s["y"] for s in best_overall_samples]
        ax.plot(cx, cy, "-", color="#2ca02c", lw=2.8,
                label=f"Evolved Champion $z^*$ ({best_overall_meta['duration']:.2f}s, err {best_overall_meta['tracking_error_cm']:.2f}cm, cost {best_overall_cost:.2f})")

    ax.set_title("2D Mat Trajectory Comparison: Starting-Value Genome vs. Stock Seeds", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("X Field Coordinate (m)", fontsize=11)
    ax.set_ylabel("Y Field Coordinate (m)", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    p1_path = os.path.join(ARTIFACT_DIR, "plot_seed_spatial_paths.png")
    plt.savefig(p1_path, dpi=200)
    plt.close()
    print("Saved:", p1_path)

    # -------------------------------------------------------------
    # Plot 2: Convergence Dynamics across Generations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    gens = np.arange(1, num_gens + 1)
    ax.plot(gens, gen_bests, "o-", color="#2ca02c", lw=2.5, markersize=8, label="Generation Champion Cost")
    ax.plot(gens, gen_means, "s--", color="#1f77b4", lw=2.0, markersize=6, alpha=0.7, label="Population Mean Cost")

    # Add reference lines for stock baselines
    ax.axhline(meta_base["cost"], color="#d62728", linestyle=":", lw=1.8, label=f"Stock Baseline ({meta_base['cost']:.3f})")
    ax.axhline(meta_rolling["cost"], color="#9467bd", linestyle="-.", lw=1.8, label=f"Stock Rolling ({meta_rolling['cost']:.3f})")

    ax.set_title("Evolutionary Progress of Starting-Value Genomes", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Evolutionary Generation", fontsize=11)
    ax.set_ylabel("Composite Cost Metric ($J$)", fontsize=11)
    ax.set_xticks(gens)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    p2_path = os.path.join(ARTIFACT_DIR, "plot_seed_convergence.png")
    plt.savefig(p2_path, dpi=200)
    plt.close()
    print("Saved:", p2_path)

    # -------------------------------------------------------------
    # Plot 3: Kinematic Velocity Profiles
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)

    if samples_base:
        tb = [s["t"] for s in samples_base]
        vb = [0.5 * (s["vl"] + s["vr"]) for s in samples_base]
        ax1.plot(tb, vb, "--", color="#d62728", lw=1.8, alpha=0.8, label="Stock Baseline")

    if samples_rolling:
        tr = [s["t"] for s in samples_rolling]
        vr = [0.5 * (s["vl"] + s["vr"]) for s in samples_rolling]
        ax1.plot(tr, vr, "-.", color="#9467bd", lw=1.8, alpha=0.85, label="Stock Rolling")

    if best_overall_samples:
        tc = [s["t"] for s in best_overall_samples]
        vc = [0.5 * (s["vl"] + s["vr"]) for s in best_overall_samples]
        ax1.plot(tc, vc, "-", color="#2ca02c", lw=2.4, label="Evolved Seed Champion")

    ax1.set_title("Linear Velocity Profiles: Agile Cornering and Acceleration", fontsize=12, weight="bold")
    ax1.set_ylabel("Linear Speed $v$ (m/s)", fontsize=11)
    ax1.set_xlabel("Trajectory Time (s)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Differential wheel velocity spread for champion
    if best_overall_samples:
        v_l = [s["vl"] for s in best_overall_samples]
        v_r = [s["vr"] for s in best_overall_samples]
        ax2.plot(tc, v_l, color="#1f77b4", lw=1.8, label="Left Wheel $v_l$")
        ax2.plot(tc, v_r, color="#ff7f0e", lw=1.8, label="Right Wheel $v_r$")
        ax2.axhline(0.88, color="black", linestyle=":", lw=1.2, label="Motor Limit ($0.88\\text{ m/s}$)")
        ax2.axhline(-0.88, color="black", linestyle=":", lw=1.2)
        ax2.set_title("Evolved Champion Differential Wheel Speeds", fontsize=12, weight="bold")
        ax2.set_ylabel("Wheel Velocity (m/s)", fontsize=11)
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    p3_path = os.path.join(ARTIFACT_DIR, "plot_seed_velocity_profiles.png")
    plt.savefig(p3_path, dpi=200)
    plt.close()
    print("Saved:", p3_path)

    # -------------------------------------------------------------
    # Plot 4: Evolved Chromosome Map (The Anatomy of the Genes)
    # -------------------------------------------------------------
    fig, (ax_dt, ax_apex, ax_spd) = plt.subplots(3, 1, figsize=(10, 8), dpi=200, sharex=True)
    segs = np.arange(1, num_segments + 1)

    if best_genome is not None:
        # 1. Segment Duration Multiplier
        ax_dt.bar(segs - 0.2, best_genome.dt_mults, width=0.4, color="#1f77b4", edgecolor="black", label="Evolved $dt$ Multiplier")
        ax_dt.axhline(1.0, color="gray", linestyle="--", lw=1.2, label="Nominal (1.0x)")
        ax_dt.set_ylabel("Duration Mult", fontsize=10, weight="bold")
        ax_dt.set_title("Anatomy of the 31-Gene Evolved Starting Solution", fontsize=13, weight="bold")
        ax_dt.legend(loc="upper right", frameon=True, fontsize=9)
        ax_dt.grid(True, linestyle="--", alpha=0.5)

        # 2. Apex Offset
        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in best_genome.apex_offsets * 100.0]
        ax_apex.bar(segs, best_genome.apex_offsets * 100.0, width=0.4, color=colors, edgecolor="black", label="Apex Offset (cm)")
        ax_apex.axhline(0.0, color="black", lw=1.0)
        ax_apex.set_ylabel("Apex Shift (cm)", fontsize=10, weight="bold")
        ax_apex.legend(loc="upper right", frameon=True, fontsize=9)
        ax_apex.grid(True, linestyle="--", alpha=0.5)

        # 3. Speed Scale
        ax_spd.bar(segs + 0.2, best_genome.speed_scales, width=0.4, color="#ff7f0e", edgecolor="black", label="Speed Scale ($v / v_{max}$)")
        ax_spd.axhline(0.6, color="gray", linestyle="--", lw=1.2, label="Default Guess (0.6)")
        ax_spd.set_ylabel("Speed Fraction", fontsize=10, weight="bold")
        ax_spd.set_xlabel("Mission Segment Index", fontsize=11, weight="bold")
        ax_spd.set_xticks(segs)
        ax_spd.legend(loc="upper right", frameon=True, fontsize=9)
        ax_spd.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p4_path = os.path.join(ARTIFACT_DIR, "plot_seed_chromosome_map.png")
    plt.savefig(p4_path, dpi=200)
    plt.close()
    print("Saved:", p4_path)


if __name__ == "__main__":
    run_and_plot()
