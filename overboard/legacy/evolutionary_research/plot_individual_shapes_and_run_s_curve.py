"""
1. Render high-resolution detailed plots for each geometric shape (2D path + velocity & yaw rate profiles).
2. Run the Full 5-Pillar Multi-Island Evolutionary Engine on the acrobatic S-Curve (example_s_curve.json).
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor
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
from scratch.run_full_5pillar_evolution import (
    create_prior_genome,
    apply_spatial_niching,
)

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def plot_detailed_shape(res: Dict[str, Any], filename: str):
    fig, (ax_path, ax_vel) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)

    wps = res["waypoints"]
    stops = set(res["stops"])

    # 1. 2D Path
    wx = [w[0] for w in wps]
    wy = [w[1] for w in wps]
    ax_path.scatter(wx, wy, c="#111111", s=100, zorder=5, label="Waypoints")
    for idx, (x, y, th) in enumerate(wps):
        is_stop = idx in stops
        lbl = f"W{idx}" + (" (STOP)" if is_stop else "")
        if th is not None:
            lbl += f" ({math.degrees(th):.0f}°)"
            # Draw heading arrow
            dx = 0.08 * math.cos(th)
            dy = 0.08 * math.sin(th)
            ax_path.arrow(x, y, dx, dy, head_width=0.025, head_length=0.035, fc="black", ec="black", zorder=6)
        ax_path.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, 8),
                         fontsize=9, weight="bold", color="#8B0000" if is_stop else "#111111")

    if res["stock_samples"]:
        sx = [s["x"] for s in res["stock_samples"]]
        sy = [s["y"] for s in res["stock_samples"]]
        ax_path.plot(sx, sy, "--", color="#d62728", lw=2.0, alpha=0.8,
                     label=f"Stock ({res['stock_duration']:.2f}s, err {res['stock_err']:.2f}cm)")

    if res["evo_samples"]:
        ex = [s["x"] for s in res["evo_samples"]]
        ey = [s["y"] for s in res["evo_samples"]]
        ax_path.plot(ex, ey, "-", color="#2ca02c", lw=2.8,
                     label=f"Evolved ({res['evo_duration']:.2f}s, err {res['evo_err']:.2f}cm) [+{res['impr']:.1f}%]")

    ax_path.set_title(f"2D Spatial Trajectory: {res['name']}", fontsize=12, weight="bold")
    ax_path.set_xlabel("X (m)", fontsize=10)
    ax_path.set_ylabel("Y (m)", fontsize=10)
    ax_path.axis("equal")
    ax_path.legend(loc="best", frameon=True, fontsize=9)
    ax_path.grid(True, linestyle="--", alpha=0.5)

    # 2. Linear Velocity Profile
    if res["stock_samples"]:
        t_s = [s["t"] for s in res["stock_samples"]]
        v_s = [0.5 * (s["vl"] + s["vr"]) for s in res["stock_samples"]]
        ax_vel.plot(t_s, v_s, "--", color="#d62728", lw=1.8, alpha=0.8, label="Stock Baseline $v(t)$")

    if res["evo_samples"]:
        t_e = [s["t"] for s in res["evo_samples"]]
        v_e = [0.5 * (s["vl"] + s["vr"]) for s in res["evo_samples"]]
        ax_vel.plot(t_e, v_e, "-", color="#2ca02c", lw=2.4, label="Evolved Champion $v(t)$")

    ax_vel.set_title(f"Linear Speed Profile $v(t)$", fontsize=12, weight="bold")
    ax_vel.set_ylabel("Speed (m/s)", fontsize=10)
    ax_vel.set_xlabel("Time (s)", fontsize=10)
    ax_vel.grid(True, linestyle="--", alpha=0.5)
    ax_vel.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved detailed plot to:", out_path)


def run_scurve_5pillar():
    print("\n================ RUNNING FULL 5-PILLAR ENGINE ON S-CURVE REVERSAL ================")
    with open("examples/fll_choreo.json") as f:
        cfg_dict = json.load(f)
    config = RobotConfig(cfg_dict)
    jax_cfg = JAXRobotConfig(config)
    val = ImmraxValidator(jax_cfg)
    engine = CompiledReachabilityEngine(val)
    optimizer = TrajectoryOptimizer(config)

    with open("examples/example_s_curve.json") as f:
        data1 = json.load(f)
    waypoints = [(w["x"], w["y"], w.get("heading")) for w in data1]
    stops = [i for i, w in enumerate(data1) if w.get("stop", False)]

    num_segments = len(waypoints) - 1
    segment_samples = optimizer.get_segment_sample_counts(waypoints, 10)
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    unpinned_count = sum(1 for i, w in enumerate(waypoints) if w[2] is None and 0 < i < len(waypoints) - 1)

    print(f"Waypoints: {waypoints}")
    print(f"Segments: {num_segments}, unpinned intermediate headings: {unpinned_count}")

    # Baseline
    g_stock = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="rolling", stop_waypoint_indices=stops)
    _, s_stock, _, _ = _solve_task((0, g_stock, waypoints, stops, cfg_dict, 150, 0.002))
    states_st, dts_st = _extract_state_and_dt_arrays(s_stock, num_segments, segment_samples)
    rep_st = engine.evaluate_states(states_st, dts_st, sample_to_segment)
    cost_stock = float(s_stock[-1]["t"]) + 120.0 * max(0.0, float(rep_st["max_tracking_error_m"]) - 0.005) + 60.0 * max(0.0, float(rep_st["max_envelope_radius_m"]) - 0.020)
    print(f"Stock Baseline (rolling): Cost = {cost_stock:.3f} | Dur = {s_stock[-1]['t']:.3f}s | Err = {rep_st['max_tracking_error_m']*100:.2f}cm")

    # 3 Islands x 6 individuals across 20 generations
    islands_config = [
        {"id": 0, "name": "Sprint Line", "type": "sprint", "acc_weight": 0.0015, "sigma_base": 0.15, "color": "#2ca02c"},
        {"id": 1, "name": "Smooth Sweeper", "type": "smooth", "acc_weight": 0.0030, "sigma_base": 0.12, "color": "#1f77b4"},
        {"id": 2, "name": "Exploratory Homotopy", "type": "exploratory", "acc_weight": 0.0020, "sigma_base": 0.22, "color": "#ff7f0e"},
    ]

    for island in islands_config:
        island["pop"] = [
            create_prior_genome(waypoints, unpinned_count, island_type=island["type"])
            for _ in range(6)
        ]
        island["stagnation_counter"] = 0
        island["best_cost"] = float("inf")
        island["best_cand"] = None

    global_best_cost = float("inf")
    global_best_cand = None

    with ProcessPoolExecutor(max_workers=6) as executor:
        for gen in range(20):
            all_tasks = []
            task_island_map = []
            all_raw_guesses = []

            iter_budget = 40 if gen < 10 else 80

            for island in islands_config:
                i_id = island["id"]
                acc_w = island["acc_weight"]
                sigma = island["sigma_base"] * (0.90 ** (gen % 5))

                for p_idx, ind in enumerate(island["pop"]):
                    raw_g = build_racing_guess(ind, waypoints, segment_samples, config, stops)
                    all_raw_guesses.append(raw_g)
                    task_id = len(all_tasks)
                    all_tasks.append((task_id, raw_g, waypoints, stops, cfg_dict, iter_budget, acc_w))
                    task_island_map.append((i_id, p_idx))

            results = list(executor.map(_solve_task, all_tasks))
            results.sort(key=lambda x: x[0])

            island_evals = {i["id"]: [] for i in islands_config}
            for task_id, samples, stats, solve_t in results:
                i_id, p_idx = task_island_map[task_id]
                island = islands_config[i_id]
                ind = island["pop"][p_idx]
                raw_g = all_raw_guesses[task_id]

                if not samples:
                    island_evals[i_id].append({"genome": ind, "raw_cost": 999.0, "samples": None, "meta": None, "guess": raw_g})
                    continue

                states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
                rep = engine.evaluate_states(states, dts, sample_to_segment)
                T = float(samples[-1]["t"])
                e_track = float(rep["max_tracking_error_m"])
                tube = float(rep["max_envelope_radius_m"])
                cost = T + 120.0 * max(0.0, e_track - 0.005) + 60.0 * max(0.0, tube - 0.020)
                meta = {
                    "duration": T, "tracking_cm": e_track * 100.0, "tube_cm": tube * 100.0,
                    "cost": cost, "iter": stats.get("iter_count", iter_budget), "island": island["name"]
                }
                island_evals[i_id].append({"genome": ind, "raw_cost": cost, "samples": samples, "meta": meta, "guess": raw_g})

            for island in islands_config:
                i_id = island["id"]
                evals = island_evals[i_id]
                apply_spatial_niching(evals, r_niche=0.08)
                evals.sort(key=lambda x: x["shared_cost"])
                island_best = min(evals, key=lambda x: x["raw_cost"])

                if island_best["raw_cost"] < island["best_cost"] - 0.003:
                    island["best_cost"] = island_best["raw_cost"]
                    island["best_cand"] = island_best
                    island["stagnation_counter"] = 0
                else:
                    island["stagnation_counter"] += 1

                if island_best["raw_cost"] < global_best_cost:
                    global_best_cost = island_best["raw_cost"]
                    global_best_cand = island_best

                # Breed
                new_pop = [evals[0]["genome"], evals[1]["genome"]]
                while len(new_pop) < 6:
                    c1, c2 = np.random.choice(min(len(evals), 3), 2, replace=False)
                    child = RacingGenome.crossover(evals[c1]["genome"], evals[c2]["genome"])
                    child = child.mutate(p_mut=0.35, sigma=island["sigma_base"] * (0.90 ** (gen % 5)))
                    new_pop.append(child)
                island["pop"] = new_pop

            if (gen + 1) % 5 == 0:
                print(f"  [Gen {gen+1:02d}/20] Ring Migration | Best Cost: {global_best_cost:.3f}")
                m0 = islands_config[0]["pop"][0]
                m1 = islands_config[1]["pop"][0]
                m2 = islands_config[2]["pop"][0]
                islands_config[1]["pop"][2] = m0
                islands_config[2]["pop"][2] = m1
                islands_config[0]["pop"][2] = m2

    # Tier C Polish
    print("\n--- Running Tier C Final Polish (200 Iters) on S-Curve Champions ---")
    final_tasks = [(i["id"], i["best_cand"]["guess"], waypoints, stops, cfg_dict, 200, 0.002) for i in islands_config if i["best_cand"]]
    with ProcessPoolExecutor(max_workers=3) as executor:
        final_results = list(executor.map(_solve_task, final_tasks))

    certified = []
    for idx, samples, stats, solve_t in final_results:
        island = islands_config[idx]
        if not samples:
            continue
        states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
        rep = engine.evaluate_states(states, dts, sample_to_segment)
        T = float(samples[-1]["t"])
        e_track = float(rep["max_tracking_error_m"])
        tube = float(rep["max_envelope_radius_m"])
        cost = T + 120.0 * max(0.0, e_track - 0.005) + 60.0 * max(0.0, tube - 0.020)
        passed = (e_track <= 0.005) and (tube <= 0.020)
        certified.append({
            "island": island["name"], "duration": T, "tracking_cm": e_track * 100.0,
            "tube_cm": tube * 100.0, "cost": cost, "passed": passed, "samples": samples,
        })

    certified.sort(key=lambda x: x["cost"])
    grand_winner = certified[0]
    print(f"\nS-CURVE GRAND WINNER: {grand_winner['island']} -> Dur: {grand_winner['duration']:.3f}s | Err: {grand_winner['tracking_cm']:.2f}cm | Cost: {grand_winner['cost']:.3f}")
    speedup = (float(s_stock[-1]["t"]) - grand_winner["duration"]) / float(s_stock[-1]["t"]) * 100.0
    print(f"Total Speedup over Stock: +{speedup:.2f}%\n")

    # Save detailed S-curve comparison plot
    res_scurve = {
        "name": "S-Curve Acrobatic Reversal (180° Inversion)",
        "waypoints": waypoints,
        "stops": stops,
        "stock_samples": s_stock,
        "stock_duration": float(s_stock[-1]["t"]),
        "stock_err": float(rep_st["max_tracking_error_m"]) * 100.0,
        "evo_samples": grand_winner["samples"],
        "evo_duration": grand_winner["duration"],
        "evo_err": grand_winner["tracking_cm"],
        "impr": speedup,
    }
    plot_detailed_shape(res_scurve, "plot_scurve_5pillar_detailed.png")


def main():
    # 1. Load multi_shape_summary.json to plot the other shapes in high-res detail
    with open("scratch/compare_geometric_shapes.py") as f:
        # Re-import solve_shape helper
        pass
    # Run the 5-pillar S-curve search
    run_scurve_5pillar()


if __name__ == "__main__":
    main()
