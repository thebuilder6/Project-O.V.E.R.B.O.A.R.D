"""
Advanced 44-Gene "Racing-Line" Starting-Value Evolutionary Pipeline
===================================================================
Incorporates:
1. Asymmetric Apex Racing Lines (Early vs. Late Apex parameterization).
2. Segment Entry/Exit Passing Velocities.
3. Adaptive Mutation Annealing across 12 Generations.
4. Parallel ProcessPool execution across CPU cores.
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

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


class RacingGenome:
    def __init__(
        self,
        dt_mults: np.ndarray,      # [S]
        apex_offsets: np.ndarray,  # [S] lateral offset in meters
        apex_peaks: np.ndarray,    # [S] fraction f in [0.25, 0.75] where apex peaks
        theta_biases: np.ndarray,  # [num_unpinned]
        speed_scales: np.ndarray,  # [S]
        wp_speeds: np.ndarray,     # [num_unpinned]
    ):
        self.dt_mults = np.array(dt_mults, dtype=float)
        self.apex_offsets = np.array(apex_offsets, dtype=float)
        self.apex_peaks = np.array(apex_peaks, dtype=float)
        self.theta_biases = np.array(theta_biases, dtype=float)
        self.speed_scales = np.array(speed_scales, dtype=float)
        self.wp_speeds = np.array(wp_speeds, dtype=float)

    @classmethod
    def create_default(cls, num_segments: int, num_unpinned: int):
        return cls(
            dt_mults=np.ones(num_segments),
            apex_offsets=np.zeros(num_segments),
            apex_peaks=np.full(num_segments, 0.50),  # symmetric apex
            theta_biases=np.zeros(num_unpinned),
            speed_scales=np.full(num_segments, 0.65),
            wp_speeds=np.full(num_unpinned, 0.50),
        )

    @classmethod
    def create_random(cls, num_segments: int, num_unpinned: int, rng=np.random):
        dt_mults = rng.uniform(0.65, 1.45, size=num_segments)
        apex_offsets = rng.uniform(-0.15, 0.15, size=num_segments)
        apex_peaks = rng.uniform(0.30, 0.70, size=num_segments)  # early to late apex
        theta_biases = rng.uniform(-0.40, 0.40, size=num_unpinned) if num_unpinned > 0 else np.array([])
        speed_scales = rng.uniform(0.35, 0.95, size=num_segments)
        wp_speeds = rng.uniform(0.30, 0.85, size=num_unpinned) if num_unpinned > 0 else np.array([])
        return cls(dt_mults, apex_offsets, apex_peaks, theta_biases, speed_scales, wp_speeds)

    def mutate(self, p_mut: float = 0.35, sigma: float = 0.15, rng=np.random):
        # 1. dt
        dt_m = self.dt_mults.copy()
        mask = rng.rand(len(dt_m)) < p_mut
        dt_m[mask] += rng.normal(0, sigma, size=np.sum(mask))
        dt_m = np.clip(dt_m, 0.35, 2.3)

        # 2. apex offset
        apex = self.apex_offsets.copy()
        mask = rng.rand(len(apex)) < p_mut
        apex[mask] += rng.normal(0, 0.04 * (sigma / 0.15), size=np.sum(mask))
        apex = np.clip(apex, -0.22, 0.22)

        # 3. apex peak fraction
        peaks = self.apex_peaks.copy()
        mask = rng.rand(len(peaks)) < p_mut
        peaks[mask] += rng.normal(0, 0.08 * (sigma / 0.15), size=np.sum(mask))
        peaks = np.clip(peaks, 0.20, 0.80)

        # 4. theta bias
        thetas = self.theta_biases.copy()
        if len(thetas) > 0:
            mask = rng.rand(len(thetas)) < p_mut
            thetas[mask] += rng.normal(0, 0.15 * (sigma / 0.15), size=np.sum(mask))
            thetas = np.clip(thetas, -math.pi / 2, math.pi / 2)

        # 5. speed scales
        spd = self.speed_scales.copy()
        mask = rng.rand(len(spd)) < p_mut
        spd[mask] += rng.normal(0, sigma, size=np.sum(mask))
        spd = np.clip(spd, 0.1, 1.15)

        # 6. waypoint passing speeds
        wspd = self.wp_speeds.copy()
        if len(wspd) > 0:
            mask = rng.rand(len(wspd)) < p_mut
            wspd[mask] += rng.normal(0, sigma, size=np.sum(mask))
            wspd = np.clip(wspd, 0.15, 1.0)

        return RacingGenome(dt_m, apex, peaks, thetas, spd, wspd)

    @classmethod
    def crossover(cls, parent_a: 'RacingGenome', parent_b: 'RacingGenome', rng=np.random):
        num_segments = len(parent_a.dt_mults)
        dt_m = np.zeros(num_segments)
        apex = np.zeros(num_segments)
        peaks = np.zeros(num_segments)
        spd = np.zeros(num_segments)

        for s in range(num_segments):
            if rng.rand() < 0.5:
                dt_m[s] = parent_a.dt_mults[s]
                apex[s] = parent_a.apex_offsets[s]
                peaks[s] = parent_a.apex_peaks[s]
                spd[s] = parent_a.speed_scales[s]
            else:
                dt_m[s] = parent_b.dt_mults[s]
                apex[s] = parent_b.apex_offsets[s]
                peaks[s] = parent_b.apex_peaks[s]
                spd[s] = parent_b.speed_scales[s]

        num_unpinned = len(parent_a.theta_biases)
        if num_unpinned > 0:
            alpha = rng.uniform(0.0, 1.0, size=num_unpinned)
            thetas = alpha * parent_a.theta_biases + (1 - alpha) * parent_b.theta_biases
            wspd = alpha * parent_a.wp_speeds + (1 - alpha) * parent_b.wp_speeds
        else:
            thetas = np.array([])
            wspd = np.array([])

        return cls(dt_m, apex, peaks, thetas, spd, wspd)


def build_racing_guess(
    genome: RacingGenome,
    waypoints: List[Tuple[float, float, Optional[float]]],
    segment_samples: List[int],
    config: RobotConfig,
    stop_waypoint_indices: Optional[List[int]] = None,
) -> np.ndarray:
    num_segments = len(waypoints) - 1
    N = sum(segment_samples) + 1
    stops = set(stop_waypoint_indices or [])

    lengths = []
    for i in range(num_segments):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        lengths.append(float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])))
    total_length = max(sum(lengths), 1e-6)

    total_intervals = sum(segment_samples)
    total_time_budget = 0.1 * total_intervals
    initial_dts = np.zeros(num_segments, dtype=float)
    for i in range(num_segments):
        seg_time = total_time_budget * (lengths[i] / total_length)
        base_dt = seg_time / max(1, segment_samples[i])
        initial_dts[i] = float(np.clip(base_dt * genome.dt_mults[i], 0.01, 1.0))

    wp_headings = []
    wp_speeds_mps = []
    v_max = config.v_max_rad_s * config.wheel_radius / max(config.gearing, 1e-9)
    unpinned_idx = 0

    for i, wp in enumerate(waypoints):
        if wp[2] is not None:
            wp_headings.append(float(wp[2]))
        else:
            if i == 0:
                p_next = waypoints[1]
                base_th = math.atan2(p_next[1] - wp[1], p_next[0] - wp[0])
            elif i == len(waypoints) - 1:
                p_prev = waypoints[-2]
                base_th = math.atan2(wp[1] - p_prev[1], wp[0] - p_prev[0])
            else:
                p_prev = waypoints[i - 1]
                p_next = waypoints[i + 1]
                base_th = math.atan2(p_next[1] - p_prev[1], p_next[0] - p_prev[0])

            bias = genome.theta_biases[unpinned_idx] if unpinned_idx < len(genome.theta_biases) else 0.0
            wp_headings.append(base_th + bias)

        # Waypoint speeds
        if i in stops or i == 0 or i == len(waypoints) - 1:
            wp_speeds_mps.append(0.0)
        else:
            spd_frac = genome.wp_speeds[unpinned_idx] if unpinned_idx < len(genome.wp_speeds) else 0.5
            wp_speeds_mps.append(float(spd_frac * v_max))
            unpinned_idx += 1

    states = []
    for i in range(num_segments):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        s_i = segment_samples[i]
        count = s_i if i < num_segments - 1 else s_i + 1

        th_start = wp_headings[i]
        th_end = wp_headings[i + 1]
        dth = (th_end - th_start + np.pi) % (2 * np.pi) - np.pi

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        L = max(lengths[i], 1e-6)
        nx = -dy / L
        ny = dx / L
        apex_amp = genome.apex_offsets[i]
        f_apex = float(np.clip(genome.apex_peaks[i], 0.15, 0.85))

        v_start = wp_speeds_mps[i]
        v_end = wp_speeds_mps[i + 1]
        seg_cruise = v_max * np.clip(genome.speed_scales[i], 0.1, 1.0)

        for j in range(count):
            u = j / s_i
            # Asymmetric late/early apex piecewise quadratic curve
            if u <= f_apex:
                bump = 1.0 - ((u - f_apex) / max(f_apex, 1e-3)) ** 2
            else:
                bump = 1.0 - ((u - f_apex) / max(1.0 - f_apex, 1e-3)) ** 2
            bump = max(0.0, bump)

            x = p1[0] + dx * u + nx * apex_amp * bump
            y = p1[1] + dy * u + ny * apex_amp * bump
            theta = th_start + dth * u

            # Velocity profile: smooth blending between entry, cruise, and exit
            if u < 0.5:
                speed = v_start + (seg_cruise - v_start) * (u / 0.5)
            else:
                speed = seg_cruise + (v_end - seg_cruise) * ((u - 0.5) / 0.5)

            vl = speed
            vr = speed
            states.append([x, y, theta, vl, vr])

    states = np.array(states, dtype=float)
    guess = np.concatenate([initial_dts, states.flatten()])
    return guess


def _solve_task(task: Tuple[Any, ...]) -> Tuple[int, Optional[List[Dict[str, Any]]], Dict[str, Any], float]:
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


def run_racing_evolution():
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

    print(f"================ ADVANCED RACING-LINE SEED EVOLUTION ================")
    print(f"Mission: {len(waypoints)} waypoints, {num_segments} segments, {unpinned_count} unpinned headings")
    print(f"Chromosome: 44 continuous genes (Early/Late Apex Fractions + Waypoint Speeds)")
    print(f"Execution: 12 Generations x 10 Genomes on 8 Parallel CPU Workers")

    # Reference Stock Seeds
    stock_seeds = ["baseline", "bow_left", "bow_right", "rolling"]
    stock_results = {}
    for style in stock_seeds:
        g = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style=style, stop_waypoint_indices=stops)
        _, samples, stats, _ = _solve_task((0, g, waypoints, stops, cfg_dict, 150, 0.002))
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

    best_stock_cost = min(s["cost"] for s in stock_results.values())
    print(f"Best Stock Baseline Cost: {best_stock_cost:.3f}")

    num_gens = 12
    pop_size = 10
    num_workers = 8

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
            sigma_decay = 0.18 * (0.86 ** gen)

            tasks = []
            guesses_raw = []
            for idx, ind in enumerate(pop):
                raw_g = build_racing_guess(ind, waypoints, segment_samples, config, stops)
                guesses_raw.append(raw_g)
                tasks.append((idx, raw_g, waypoints, stops, cfg_dict, 120, 0.002))

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
                    "iter": stats.get("iter_count", 120),
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

            print(f"Gen {gen+1:02d}/{num_gens:02d} | Champion Cost: {g_best[1]:.3f} (Dur: {g_best[3]['duration']:.3f}s, Err: {g_best[3]['tracking_error_cm']:.2f}cm, Iter: {g_best[3]['iter']}) | Pop Mean: {g_mean:.3f} | Wall: {wall_t:.2f}s")

            if gen == num_gens - 1:
                break

            # Selection: Elitism (Top 2) + Tournament Crossover + Annealed Mutation
            new_pop = [gen_evals[0][0], gen_evals[1][0]]
            while len(new_pop) < pop_size:
                c1, c2 = np.random.choice(len(gen_evals[:5]), size=2, replace=False)
                child = RacingGenome.crossover(gen_evals[c1][0], gen_evals[c2][0])
                child = child.mutate(p_mut=0.35, sigma=sigma_decay)
                new_pop.append(child)
            pop = new_pop

    print("\n================ RACING-LINE BENCHMARK SUMMARY ================")
    print(f"Stock Best Seed:              Cost = {best_stock_cost:.3f}")
    print(f"12-Gen Racing Seed Champion:  Cost = {best_overall_cost:.3f} | Dur = {best_overall_meta['duration']:.3f}s | Err = {best_overall_meta['tracking_error_cm']:.2f}cm | Iter = {best_overall_meta['iter']}")
    impr = (best_stock_cost - best_overall_cost) / best_stock_cost * 100.0
    print(f"Total Cost Improvement:       {impr:+.2f}%")

    # -------------------------------------------------------------
    # Plotting 12-Gen Evolution Figures
    # -------------------------------------------------------------
    print("\nRendering high-res publication figures...")
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

    if stock_results["baseline"]["samples"]:
        bx = [s["x"] for s in stock_results["baseline"]["samples"]]
        by = [s["y"] for s in stock_results["baseline"]["samples"]]
        ax.plot(bx, by, "--", color="#d62728", lw=1.8, alpha=0.75, label=f"Stock Baseline ({stock_results['baseline']['duration']:.2f}s)")

    if stock_results["rolling"]["samples"]:
        rx = [s["x"] for s in stock_results["rolling"]["samples"]]
        ry = [s["y"] for s in stock_results["rolling"]["samples"]]
        ax.plot(rx, ry, "-.", color="#9467bd", lw=2.0, alpha=0.8, label=f"Stock Rolling ({stock_results['rolling']['duration']:.2f}s)")

    if best_guess_raw is not None:
        N = sum(segment_samples) + 1
        guess_states = best_guess_raw[num_segments:].reshape((N, 5))
        ax.plot(guess_states[:, 0], guess_states[:, 1], ":", color="#ff7f0e", lw=2.0, label="12-Gen Racing Guess $z_0$")

    if best_overall_samples:
        cx = [s["x"] for s in best_overall_samples]
        cy = [s["y"] for s in best_overall_samples]
        ax.plot(cx, cy, "-", color="#2ca02c", lw=2.8,
                label=f"12-Gen Racing Champion $z^*$ ({best_overall_meta['duration']:.2f}s, err {best_overall_meta['tracking_error_cm']:.2f}cm, cost {best_overall_cost:.2f})")

    ax.set_title("2D Mat Path: 12-Generation Racing-Line Starting Seed vs. Stock", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("X Field Coordinate (m)", fontsize=11)
    ax.set_ylabel("Y Field Coordinate (m)", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "plot_seed_spatial_paths.png"), dpi=200)
    plt.close()

    # 2. 12-Generation Convergence
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    gens = [h["gen"] for h in history]
    bests = [h["best_cost"] for h in history]
    means = [h["mean_cost"] for h in history]

    ax.plot(gens, bests, "o-", color="#2ca02c", lw=2.6, markersize=7, label="Gen Champion Cost")
    ax.plot(gens, means, "s--", color="#1f77b4", lw=1.8, markersize=5, alpha=0.7, label="Pop Mean Cost")
    ax.axhline(stock_results["baseline"]["cost"], color="#d62728", linestyle=":", lw=1.8, label=f"Stock Baseline ({stock_results['baseline']['cost']:.3f})")
    ax.axhline(stock_results["rolling"]["cost"], color="#9467bd", linestyle="-.", lw=1.8, label=f"Stock Rolling ({stock_results['rolling']['cost']:.3f})")

    ax.set_title("12-Generation Convergence Dynamics (44-Gene Racing Landscape)", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Evolutionary Generation", fontsize=11)
    ax.set_ylabel("Composite Cost Metric ($J$)", fontsize=11)
    ax.set_xticks(gens)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "plot_seed_convergence.png"), dpi=200)
    plt.close()

    # 3. Kinematic Profiles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=False)
    if stock_results["baseline"]["samples"]:
        tb = [s["t"] for s in stock_results["baseline"]["samples"]]
        vb = [0.5 * (s["vl"] + s["vr"]) for s in stock_results["baseline"]["samples"]]
        ax1.plot(tb, vb, "--", color="#d62728", lw=1.8, alpha=0.75, label="Stock Baseline")

    if best_overall_samples:
        tc = [s["t"] for s in best_overall_samples]
        vc = [0.5 * (s["vl"] + s["vr"]) for s in best_overall_samples]
        ax1.plot(tc, vc, "-", color="#2ca02c", lw=2.4, label="12-Gen Racing Champion")

    ax1.set_title("Linear Velocity Profiles: Racing-Line Entry & Exit Acceleration", fontsize=12, weight="bold")
    ax1.set_ylabel("Linear Speed $v$ (m/s)", fontsize=11)
    ax1.set_xlabel("Trajectory Time (s)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    if best_overall_samples:
        v_l = [s["vl"] for s in best_overall_samples]
        v_r = [s["vr"] for s in best_overall_samples]
        ax2.plot(tc, v_l, color="#1f77b4", lw=1.8, label="Left Wheel $v_l$")
        ax2.plot(tc, v_r, color="#ff7f0e", lw=1.8, label="Right Wheel $v_r$")
        ax2.axhline(0.88, color="black", linestyle=":", lw=1.2, label="Motor Saturation Limit ($0.88\\text{ m/s}$)")
        ax2.axhline(-0.88, color="black", linestyle=":", lw=1.2)
        ax2.set_title("Differential Wheel Velocities (Traction Preserved)", fontsize=12, weight="bold")
        ax2.set_ylabel("Wheel Velocity (m/s)", fontsize=11)
        ax2.set_xlabel("Time (s)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "plot_seed_velocity_profiles.png"), dpi=200)
    plt.close()

    # 4. Racing Chromosome Anatomy (4 Subplots)
    fig, (ax_dt, ax_apex, ax_peak, ax_spd) = plt.subplots(4, 1, figsize=(10, 10), dpi=200, sharex=True)
    segs = np.arange(1, num_segments + 1)
    if best_genome is not None:
        ax_dt.bar(segs - 0.2, best_genome.dt_mults, width=0.4, color="#1f77b4", edgecolor="black", label="Duration Multiplier")
        ax_dt.axhline(1.0, color="gray", linestyle="--", lw=1.2, label="Nominal (1.0x)")
        ax_dt.set_ylabel("dt Mult", fontsize=9, weight="bold")
        ax_dt.set_title("Anatomy of the 44-Gene Racing Starting Solution", fontsize=13, weight="bold")
        ax_dt.legend(loc="upper right", frameon=True, fontsize=8)
        ax_dt.grid(True, linestyle="--", alpha=0.5)

        colors = ["#2ca02c" if x >= 0 else "#d62728" for x in best_genome.apex_offsets * 100.0]
        ax_apex.bar(segs, best_genome.apex_offsets * 100.0, width=0.4, color=colors, edgecolor="black", label="Apex Offset (cm)")
        ax_apex.axhline(0.0, color="black", lw=1.0)
        ax_apex.set_ylabel("Apex Shift (cm)", fontsize=9, weight="bold")
        ax_apex.legend(loc="upper right", frameon=True, fontsize=8)
        ax_apex.grid(True, linestyle="--", alpha=0.5)

        # Early vs Late Apex
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
    print("All figures successfully rendered to:", ARTIFACT_DIR)


if __name__ == "__main__":
    run_racing_evolution()
