"""
Deep Multi-Generation Seed / Starting-Value Evolution (Parallel Execution)
==========================================================================
Runs 8 generations of population-based starting-parameter evolution using
ProcessPoolExecutor on multi-core CPU.
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
from scratch.test_seed_evolution import (
    InitialGuessGenome,
    build_guess_from_genome,
)

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def _solve_worker(task: Tuple[Any, ...]) -> Tuple[int, Optional[List[Dict[str, Any]]], Dict[str, Any], float]:
    """Worker process to solve IPOPT from an initial guess without GIL/MUMPS conflict."""
    idx, guess, waypoints, stops, cfg_dict, max_iter, acc_weight = task
    t0 = time.time()
    try:
        config = RobotConfig(cfg_dict)
        optimizer = TrajectoryOptimizer(config)
        solver_options = {
            "max_iter": max_iter,
            "print_level": 0,
        }
        samples, stats = optimizer.solve(
            waypoints=waypoints,
            stop_waypoint_indices=stops,
            guess_style="baseline",
            initial_guess=guess,
            per_segment_dt=True,
            accuracy_weight=acc_weight,
            solver_options=solver_options,
            verbose=False,
        )
        return idx, samples, stats, time.time() - t0
    except Exception as e:
        return idx, None, {"error": str(e)}, time.time() - t0


def run_deep_evolution():
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

    print(f"Mission: {len(waypoints)} waypoints, {num_segments} segments, {unpinned_count} unpinned intermediate headings")
    print(f"Starting 8-Generation Evolution with Population Size 10 (Parallel Workers)")

    # 1. Evaluate stock baselines
    print("\n--- 1. Stock Baselines ---")
    stock_seeds = ["baseline", "bow_left", "bow_right", "rolling"]
    stock_results = {}
    for style in stock_seeds:
        g = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style=style, stop_waypoint_indices=stops)
        _, samples, stats, solve_t = _solve_worker((0, g, waypoints, stops, cfg_dict, 150, 0.002))
        states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
        rep = engine.evaluate_states(states, dts, sample_to_segment)
        T = float(samples[-1]["t"])
        e_track = float(rep["max_tracking_error_m"])
        tube = float(rep["max_envelope_radius_m"])
        cost = T + 120.0 * max(0.0, e_track - 0.005) + 60.0 * max(0.0, tube - 0.020)
        stock_results[style] = {
            "samples": samples,
            "cost": cost,
            "duration": T,
            "tracking_error_cm": e_track * 100.0,
            "tube_cm": tube * 100.0,
            "iter": stats.get("iter_count", 150),
        }
        print(f"  Stock {style:10s} -> Dur: {T:.3f}s | Err: {e_track*100:.2f}cm | Cost: {cost:.3f} | Iter: {stock_results[style]['iter']} ({solve_t:.2f}s)")

    # 2. Multi-generation evolution
    num_gens = 8
    pop_size = 10
    num_workers = 6

    # Initialize diverse population
    pop: List[InitialGuessGenome] = [InitialGuessGenome.create_default(num_segments, unpinned_count)]
    for _ in range(pop_size - 1):
        pop.append(InitialGuessGenome.create_random(num_segments, unpinned_count))

    best_overall_cost = float("inf")
    best_overall_samples = None
    best_overall_meta = None
    best_genome = None
    best_guess_raw = None

    history = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for gen in range(num_gens):
            gen_start_t = time.time()
            sigma_decay = 0.18 * (0.88 ** gen)  # Temperature cooling schedule

            # Prepare guesses
            tasks = []
            guesses_raw = []
            for idx, ind in enumerate(pop):
                raw_g = build_guess_from_genome(ind, waypoints, segment_samples, config, stops)
                guesses_raw.append(raw_g)
                tasks.append((idx, raw_g, waypoints, stops, cfg_dict, 120, 0.002))

            # Run parallel solves
            results = list(executor.map(_solve_worker, tasks))
            results.sort(key=lambda x: x[0])  # Restore index order

            # Evaluate each individual
            gen_evals = []
            for idx, samples, stats, solve_t in results:
                ind = pop[idx]
                raw_g = guesses_raw[idx]
                if not samples:
                    gen_evals.append((ind, 999.0, None, {"error": "solve_failed"}, raw_g))
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
                    "iter": stats.get("iter_count", 120),
                    "solve_t": solve_t,
                }
                gen_evals.append((ind, cost, samples, meta, raw_g))

            gen_evals.sort(key=lambda x: x[1])
            g_best = gen_evals[0]
            valid_costs = [e[1] for e in gen_evals if e[1] < 900]
            g_mean = float(np.mean(valid_costs)) if valid_costs else 999.0
            gen_wall_t = time.time() - gen_start_t

            history.append({
                "gen": gen + 1,
                "best_cost": g_best[1],
                "mean_cost": g_mean,
                "best_meta": g_best[3],
                "wall_time": gen_wall_t,
            })

            if g_best[1] < best_overall_cost:
                best_overall_cost = g_best[1]
                best_overall_samples = g_best[2]
                best_overall_meta = g_best[3]
                best_genome = g_best[0]
                best_guess_raw = g_best[4]

            print(f"Gen {gen+1:02d}/{num_gens:02d} | Champion Cost: {g_best[1]:.3f} (Dur: {g_best[3].get('duration', 99):.3f}s, Err: {g_best[3].get('tracking_error_cm', 99):.2f}cm, Iter: {g_best[3].get('iter', 0)}) | Pop Mean: {g_mean:.3f} | Wall: {gen_wall_t:.2f}s")

            if gen == num_gens - 1:
                break

            # Next generation breeding: Top 2 Elitism + Tournament Selection + Segment Crossover + Annealed Mutation
            new_pop = [gen_evals[0][0], gen_evals[1][0]]
            while len(new_pop) < pop_size:
                c1, c2 = np.random.choice(len(gen_evals[:5]), size=2, replace=False)
                parent_a = gen_evals[c1][0]
                parent_b = gen_evals[c2][0]
                child = InitialGuessGenome.crossover(parent_a, parent_b)
                child = child.mutate(p_mut=0.35, sigma=sigma_decay)
                new_pop.append(child)
            pop = new_pop

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    best_stock = min(stock_results.items(), key=lambda x: x[1]["cost"])
    print(f"Best Stock Seed ({best_stock[0]}): Cost = {best_stock[1]['cost']:.3f} | Dur = {best_stock[1]['duration']:.3f}s | Err = {best_stock[1]['tracking_error_cm']:.2f}cm | Iter = {best_stock[1]['iter']}")
    print(f"8-Gen Evolved Champion:      Cost = {best_overall_cost:.3f} | Dur = {best_overall_meta['duration']:.3f}s | Err = {best_overall_meta['tracking_error_cm']:.2f}cm | Iter = {best_overall_meta['iter']}")
    cost_impr = (best_stock[1]["cost"] - best_overall_cost) / best_stock[1]["cost"] * 100.0
    print(f"Cost Improvement:            {cost_impr:+.2f}%")

    # -------------------------------------------------------------
    # Generate Updated Plots
    # -------------------------------------------------------------
    print("\nGenerating updated publication plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. 2D Mat Field Paths
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    wx = [w[0] for w in waypoints]
    wy = [w[1] for w in waypoints]
    ax.scatter(wx, wy, c="#222222", s=90, zorder=5, label="Waypoints")
    for i, (x, y, _) in enumerate(waypoints):
        is_stop = i in stops
        lbl = f"W{i}" + (" (STOP)" if is_stop else "")
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, weight="bold", color="#8B0000" if is_stop else "#111111")

    # Baseline & Rolling
    if stock_results["baseline"]["samples"]:
        bx = [s["x"] for s in stock_results["baseline"]["samples"]]
        by = [s["y"] for s in stock_results["baseline"]["samples"]]
        ax.plot(bx, by, "--", color="#d62728", lw=1.8, alpha=0.75,
                label=f"Stock Baseline ({stock_results['baseline']['duration']:.2f}s, cost {stock_results['baseline']['cost']:.2f})")

    if stock_results["rolling"]["samples"]:
        rx = [s["x"] for s in stock_results["rolling"]["samples"]]
        ry = [s["y"] for s in stock_results["rolling"]["samples"]]
        ax.plot(rx, ry, "-.", color="#9467bd", lw=2.0, alpha=0.8,
                label=f"Stock Rolling ({stock_results['rolling']['duration']:.2f}s, cost {stock_results['rolling']['cost']:.2f})")

    # Evolved Guess Genome z0
    if best_guess_raw is not None:
        N = sum(segment_samples) + 1
        guess_states = best_guess_raw[num_segments:].reshape((N, 5))
        ax.plot(guess_states[:, 0], guess_states[:, 1], ":", color="#ff7f0e", lw=2.0,
                label="Evolved Seed Genome $z_0$ (Pre-IPOPT Guess)")

    # Evolved Champion Solved Trajectory
    if best_overall_samples:
        cx = [s["x"] for s in best_overall_samples]
        cy = [s["y"] for s in best_overall_samples]
        ax.plot(cx, cy, "-", color="#2ca02c", lw=2.8,
                label=f"8-Gen Evolved Champion $z^*$ ({best_overall_meta['duration']:.2f}s, err {best_overall_meta['tracking_error_cm']:.2f}cm, cost {best_overall_cost:.2f})")

    ax.set_title("2D Mat Paths: 8-Generation Starting-Value Evolution vs. Stock Seeds", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("X Field Coordinate (m)", fontsize=11)
    ax.set_ylabel("Y Field Coordinate (m)", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    p1_path = os.path.join(ARTIFACT_DIR, "plot_seed_spatial_paths.png")
    plt.savefig(p1_path, dpi=200)
    plt.close()

    # 2. Multi-Generation Convergence Dynamics
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    gens = [h["gen"] for h in history]
    bests = [h["best_cost"] for h in history]
    means = [h["mean_cost"] for h in history]

    ax.plot(gens, bests, "o-", color="#2ca02c", lw=2.6, markersize=8, label="Generation Champion Cost")
    ax.plot(gens, means, "s--", color="#1f77b4", lw=2.0, markersize=6, alpha=0.75, label="Population Mean Cost")

    ax.axhline(stock_results["baseline"]["cost"], color="#d62728", linestyle=":", lw=1.8, label=f"Stock Baseline ({stock_results['baseline']['cost']:.3f})")
    ax.axhline(stock_results["rolling"]["cost"], color="#9467bd", linestyle="-.", lw=1.8, label=f"Stock Rolling ({stock_results['rolling']['cost']:.3f})")

    ax.set_title("8-Generation Evolutionary Convergence of Starting Values", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Evolutionary Generation", fontsize=11)
    ax.set_ylabel("Composite Cost Metric ($J$)", fontsize=11)
    ax.set_xticks(gens)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    p2_path = os.path.join(ARTIFACT_DIR, "plot_seed_convergence.png")
    plt.savefig(p2_path, dpi=200)
    plt.close()

    # 3. Kinematic Profiles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)
    if stock_results["baseline"]["samples"]:
        tb = [s["t"] for s in stock_results["baseline"]["samples"]]
        vb = [0.5 * (s["vl"] + s["vr"]) for s in stock_results["baseline"]["samples"]]
        ax1.plot(tb, vb, "--", color="#d62728", lw=1.8, alpha=0.75, label="Stock Baseline")

    if stock_results["rolling"]["samples"]:
        tr = [s["t"] for s in stock_results["rolling"]["samples"]]
        vr = [0.5 * (s["vl"] + s["vr"]) for s in stock_results["rolling"]["samples"]]
        ax1.plot(tr, vr, "-.", color="#9467bd", lw=1.8, alpha=0.8, label="Stock Rolling")

    if best_overall_samples:
        tc = [s["t"] for s in best_overall_samples]
        vc = [0.5 * (s["vl"] + s["vr"]) for s in best_overall_samples]
        ax1.plot(tc, vc, "-", color="#2ca02c", lw=2.4, label="8-Gen Champion")

    ax1.set_title("Linear Velocity Profiles: Agile Cornering and Acceleration", fontsize=12, weight="bold")
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
        ax2.set_title("8-Gen Champion Wheel Velocities", fontsize=12, weight="bold")
        ax2.set_ylabel("Wheel Velocity (m/s)", fontsize=11)
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    p3_path = os.path.join(ARTIFACT_DIR, "plot_seed_velocity_profiles.png")
    plt.savefig(p3_path, dpi=200)
    plt.close()

    # 4. Chromosome Map
    fig, (ax_dt, ax_apex, ax_spd) = plt.subplots(3, 1, figsize=(10, 8), dpi=200, sharex=True)
    segs = np.arange(1, num_segments + 1)
    if best_genome is not None:
        ax_dt.bar(segs - 0.2, best_genome.dt_mults, width=0.4, color="#1f77b4", edgecolor="black", label="Evolved $dt$ Multiplier")
        ax_dt.axhline(1.0, color="gray", linestyle="--", lw=1.2, label="Nominal (1.0x)")
        ax_dt.set_ylabel("Duration Mult", fontsize=10, weight="bold")
        ax_dt.set_title("8-Generation Evolved Chromosome Anatomy (31 Genes)", fontsize=13, weight="bold")
        ax_dt.legend(loc="upper right", frameon=True, fontsize=9)
        ax_dt.grid(True, linestyle="--", alpha=0.5)

        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in best_genome.apex_offsets * 100.0]
        ax_apex.bar(segs, best_genome.apex_offsets * 100.0, width=0.4, color=colors, edgecolor="black", label="Apex Offset (cm)")
        ax_apex.axhline(0.0, color="black", lw=1.0)
        ax_apex.set_ylabel("Apex Shift (cm)", fontsize=10, weight="bold")
        ax_apex.legend(loc="upper right", frameon=True, fontsize=9)
        ax_apex.grid(True, linestyle="--", alpha=0.5)

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

    print("All plots updated successfully in:", ARTIFACT_DIR)


if __name__ == "__main__":
    run_deep_evolution()
