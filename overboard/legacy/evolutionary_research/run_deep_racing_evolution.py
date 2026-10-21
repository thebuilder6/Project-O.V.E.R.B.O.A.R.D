"""
Deep 20-Generation Racing-Line Evolution with Expanded Iteration Budget
========================================================================
- 20 Generations with 12 Individuals
- Max IPOPT Iterations: 200 (allows full convergence beyond the 120 cap)
- 44-Gene Racing Line Architecture (Asymmetric Apex + Passing Speeds)
- ProcessPoolExecutor on 8 CPU Workers
"""

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

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


def run_deep_racing_evolution():
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

    print("================ DEEP 20-GENERATION RACING-LINE EVOLUTION ================")
    print(f"Mission: {len(waypoints)} waypoints, {num_segments} segments, {unpinned_count} unpinned headings")
    print(f"Population: 12 Genomes across 20 Generations | Solver Budget: 200 Iterations")
    print(f"Parallel Execution on 8 CPU Workers")

    # Stock baseline for benchmark reference
    stock_guess = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="rolling", stop_waypoint_indices=stops)
    _, s_stock, st_stock, _ = _solve_task((0, stock_guess, waypoints, stops, cfg_dict, 200, 0.002))
    states_st, dts_st = _extract_state_and_dt_arrays(s_stock, num_segments, segment_samples)
    rep_st = engine.evaluate_states(states_st, dts_st, sample_to_segment)
    cost_stock = float(s_stock[-1]["t"]) + 120.0 * max(0.0, float(rep_st["max_tracking_error_m"]) - 0.005) + 60.0 * max(0.0, float(rep_st["max_envelope_radius_m"]) - 0.020)
    print(f"Best Stock Baseline (rolling): Cost = {cost_stock:.3f} | Dur = {s_stock[-1]['t']:.3f}s | Err = {rep_st['max_tracking_error_m']*100:.2f}cm")

    num_gens = 20
    pop_size = 12
    num_workers = 8

    # Seed population with default + diverse random exploration
    pop: List[RacingGenome] = [RacingGenome.create_default(num_segments, unpinned_count)]
    for _ in range(pop_size - 1):
        pop.append(RacingGenome.create_random(num_segments, unpinned_count))

    best_overall_cost = float("inf")
    best_overall_samples = None
    best_overall_meta = None
    best_genome = None
    best_guess_raw = None

    history = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for gen in range(num_gens):
            gen_start = time.time()
            # Dynamic simulated annealing schedule with periodic micro-diversity injections
            sigma_decay = 0.18 * (0.89 ** (gen % 10))

            tasks = []
            guesses_raw = []
            for idx, ind in enumerate(pop):
                raw_g = build_racing_guess(ind, waypoints, segment_samples, config, stops)
                guesses_raw.append(raw_g)
                tasks.append((idx, raw_g, waypoints, stops, cfg_dict, 200, 0.002))

            results = list(executor.map(_solve_task, tasks))
            results.sort(key=lambda x: x[0])

            gen_evals = []
            for idx, samples, stats, solve_t in results:
                ind = pop[idx]
                raw_g = guesses_raw[idx]
                if not samples:
                    gen_evals.append((ind, 999.0, None, {"error": "failed"}, raw_g))
                    continue

                states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
                rep = engine.evaluate_states(states, dts, sample_to_segment)
                T = float(samples[-1]["t"])
                e_track = float(rep["max_tracking_error_m"])
                tube = float(rep["max_envelope_radius_m"])
                cost = T + 120.0 * max(0.0, e_track - 0.005) + 60.0 * max(0.0, tube - 0.020)
                meta = {
                    "duration": T,
                    "tracking_error_cm": e_track * 100.0,
                    "tube_cm": tube * 100.0,
                    "cost": cost,
                    "iter": stats.get("iter_count", 200),
                    "solve_t": solve_t,
                }
                gen_evals.append((ind, cost, samples, meta, raw_g))

            gen_evals.sort(key=lambda x: x[1])
            g_best = gen_evals[0]
            valid_costs = [e[1] for e in gen_evals if e[1] < 900]
            g_mean = float(np.mean(valid_costs)) if valid_costs else 999.0
            wall_t = time.time() - gen_start

            history.append({
                "gen": gen + 1,
                "best_cost": g_best[1],
                "mean_cost": g_mean,
                "best_meta": g_best[3],
                "wall_time": wall_t,
            })

            if g_best[1] < best_overall_cost:
                best_overall_cost = g_best[1]
                best_overall_samples = g_best[2]
                best_overall_meta = g_best[3]
                best_genome = g_best[0]
                best_guess_raw = g_best[4]

            print(f"Gen {gen+1:02d}/{num_gens:02d} | Champion Cost: {g_best[1]:.3f} (Dur: {g_best[3]['duration']:.3f}s, Err: {g_best[3]['tracking_error_cm']:.2f}cm, Iter: {g_best[3]['iter']}) | Mean: {g_mean:.3f} | Wall: {wall_t:.2f}s")

            if gen == num_gens - 1:
                break

            # Selection: Elitism (Top 3) + Tournament Crossover + Mutated Offspring
            new_pop = [gen_evals[0][0], gen_evals[1][0], gen_evals[2][0]]
            while len(new_pop) < pop_size:
                c1, c2 = np.random.choice(len(gen_evals[:6]), size=2, replace=False)
                child = RacingGenome.crossover(gen_evals[c1][0], gen_evals[c2][0])
                child = child.mutate(p_mut=0.35, sigma=sigma_decay)
                new_pop.append(child)
            pop = new_pop

    print("\n================ FINAL 20-GENERATION SUMMARY ================")
    print(f"Stock Best Baseline:         Cost = {cost_stock:.3f}")
    print(f"20-Gen Racing Seed Champion: Cost = {best_overall_cost:.3f} | Dur = {best_overall_meta['duration']:.3f}s | Err = {best_overall_meta['tracking_error_cm']:.2f}cm | Iter = {best_overall_meta['iter']}")
    total_impr = (cost_stock - best_overall_cost) / cost_stock * 100.0
    print(f"Total Cost Improvement:      {total_impr:+.2f}%")

    # -------------------------------------------------------------
    # Re-render Figures
    # -------------------------------------------------------------
    print("\nRe-rendering publication plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. 2D Spatial Paths
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    wx = [w[0] for w in waypoints]
    wy = [w[1] for w in waypoints]
    ax.scatter(wx, wy, c="#111111", s=95, zorder=5, label="Waypoints")
    for i, (x, y, _) in enumerate(waypoints):
        is_stop = i in stops
        lbl = f"W{i}" + (" (STOP)" if is_stop else "")
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, weight="bold", color="#8B0000" if is_stop else "#111111")

    if s_stock:
        rx = [s["x"] for s in s_stock]
        ry = [s["y"] for s in s_stock]
        ax.plot(rx, ry, "-.", color="#9467bd", lw=2.0, alpha=0.8, label=f"Stock Rolling ({s_stock[-1]['t']:.2f}s, cost {cost_stock:.2f})")

    if best_guess_raw is not None:
        N = sum(segment_samples) + 1
        guess_states = best_guess_raw[num_segments:].reshape((N, 5))
        ax.plot(guess_states[:, 0], guess_states[:, 1], ":", color="#ff7f0e", lw=2.0, label="20-Gen Racing Guess $z_0$")

    if best_overall_samples:
        cx = [s["x"] for s in best_overall_samples]
        cy = [s["y"] for s in best_overall_samples]
        ax.plot(cx, cy, "-", color="#2ca02c", lw=2.8,
                label=f"20-Gen Racing Champion $z^*$ ({best_overall_meta['duration']:.2f}s, err {best_overall_meta['tracking_error_cm']:.2f}cm, cost {best_overall_cost:.2f})")

    ax.set_title("2D Mat Path: 20-Generation Deep Racing-Line Optimization", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("X Field Coordinate (m)", fontsize=11)
    ax.set_ylabel("Y Field Coordinate (m)", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "plot_seed_spatial_paths.png"), dpi=200)
    plt.close()

    # 2. 20-Generation Convergence
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    gens = [h["gen"] for h in history]
    bests = [h["best_cost"] for h in history]
    means = [h["mean_cost"] for h in history]

    ax.plot(gens, bests, "o-", color="#2ca02c", lw=2.6, markersize=6, label="Gen Champion Cost")
    ax.plot(gens, means, "s--", color="#1f77b4", lw=1.6, markersize=4, alpha=0.7, label="Pop Mean Cost")
    ax.axhline(cost_stock, color="#9467bd", linestyle="-.", lw=1.8, label=f"Stock Rolling ({cost_stock:.3f})")

    ax.set_title("20-Generation Evolutionary Trajectory Convergence", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Evolutionary Generation", fontsize=11)
    ax.set_ylabel("Composite Cost Metric ($J$)", fontsize=11)
    ax.set_xticks(range(1, num_gens + 1, 2))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "plot_seed_convergence.png"), dpi=200)
    plt.close()

    # 3. Kinematic Profiles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)
    if s_stock:
        tr = [s["t"] for s in s_stock]
        vr = [0.5 * (s["vl"] + s["vr"]) for s in s_stock]
        ax1.plot(tr, vr, "-.", color="#9467bd", lw=1.8, alpha=0.8, label="Stock Rolling")

    if best_overall_samples:
        tc = [s["t"] for s in best_overall_samples]
        vc = [0.5 * (s["vl"] + s["vr"]) for s in best_overall_samples]
        ax1.plot(tc, vc, "-", color="#2ca02c", lw=2.4, label="20-Gen Champion")

    ax1.set_title("Linear Velocity Profiles: Agile Racing Acceleration", fontsize=12, weight="bold")
    ax1.set_ylabel("Linear Speed $v$ (m/s)", fontsize=11)
    ax1.set_xlabel("Trajectory Time (s)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    if best_overall_samples:
        v_l = [s["vl"] for s in best_overall_samples]
        v_r = [s["vr"] for s in best_overall_samples]
        ax2.plot(tc, v_l, color="#1f77b4", lw=1.8, label="Left Wheel $v_l$")
        ax2.plot(tc, v_r, color="#ff7f0e", lw=1.8, label="Right Wheel $v_r$")
        ax2.axhline(0.88, color="black", linestyle=":", lw=1.2, label="Motor Limit ($0.88\\text{ m/s}$)")
        ax2.axhline(-0.88, color="black", linestyle=":", lw=1.2)
        ax2.set_title("Differential Wheel Velocities (Traction Preserved)", fontsize=12, weight="bold")
        ax2.set_ylabel("Wheel Velocity (m/s)", fontsize=11)
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "plot_seed_velocity_profiles.png"), dpi=200)
    plt.close()

    # 4. Chromosome Map
    fig, (ax_dt, ax_apex, ax_peak, ax_spd) = plt.subplots(4, 1, figsize=(10, 10), dpi=200, sharex=True)
    segs = np.arange(1, num_segments + 1)
    if best_genome is not None:
        ax_dt.bar(segs - 0.2, best_genome.dt_mults, width=0.4, color="#1f77b4", edgecolor="black", label="Duration Multiplier")
        ax_dt.axhline(1.0, color="gray", linestyle="--", lw=1.2, label="Nominal (1.0x)")
        ax_dt.set_ylabel("dt Mult", fontsize=9, weight="bold")
        ax_dt.set_title("Anatomy of the 20-Generation Evolved Racing Genome (44 Genes)", fontsize=13, weight="bold")
        ax_dt.legend(loc="upper right", frameon=True, fontsize=8)
        ax_dt.grid(True, linestyle="--", alpha=0.5)

        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in best_genome.apex_offsets * 100.0]
        ax_apex.bar(segs, best_genome.apex_offsets * 100.0, width=0.4, color=colors, edgecolor="black", label="Apex Offset (cm)")
        ax_apex.axhline(0.0, color="black", lw=1.0)
        ax_apex.set_ylabel("Apex Shift (cm)", fontsize=9, weight="bold")
        ax_apex.legend(loc="upper right", frameon=True, fontsize=8)
        ax_apex.grid(True, linestyle="--", alpha=0.5)

        peak_colors = ["#9467bd" if p > 0.52 else "#17becf" if p < 0.48 else "gray" for p in best_genome.apex_peaks]
        ax_peak.bar(segs, best_genome.apex_peaks, width=0.4, color=peak_colors, edgecolor="black", label="Apex Peak Location (Early < 0.5 < Late)")
        ax_peak.axhline(0.5, color="black", linestyle="--", lw=1.2, label="Midpoint (0.50)")
        ax_peak.set_ylabel("Apex Frac", fontsize=9, weight="bold")
        ax_peak.set_ylim(0.15, 0.85)
        ax_peak.legend(loc="upper right", frameon=True, fontsize=8)
        ax_peak.grid(True, linestyle="--", alpha=0.5)

        ax_spd.bar(segs + 0.2, best_genome.speed_scales, width=0.4, color="#ff7f0e", edgecolor="black", label="Speed Fraction ($v / v_{max}$)")
        ax_spd.axhline(0.65, color="gray", linestyle="--", lw=1.2, label="Default (0.65)")
        ax_spd.set_ylabel("Speed Frac", fontsize=9, weight="bold")
        ax_spd.set_xlabel("Mission Segment Index", fontsize=10, weight="bold")
        ax_spd.set_xticks(segs)
        ax_spd.legend(loc="upper right", frameon=True, fontsize=8)
        ax_spd.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "plot_seed_chromosome_map.png"), dpi=200)
    plt.close()
    print("All 4 figures updated successfully!")


if __name__ == "__main__":
    run_deep_racing_evolution()
