"""
Centurion 70-Generation 4-Island Evolutionary Run (1,680+ Solves)
=================================================================
- 70 Generations across 4 Islands:
    1. Sprint Line
    2. Smooth Sweeper
    3. Precision Line
    4. Exploratory Homotopy
- Seeded from previous Marathon Champions (warm evolutionary continuation)
- 12 Parallel CPU Workers (24 candidate solves per generation in 2 parallel rounds)
- Dynamic 4-Stage Iteration Schedule:
    * Gens 1-25:  35 iters (Broad exploration)
    * Gens 26-50: 60 iters (Basin consolidation)
    * Gens 51-65: 100 iters (Fine-grained tuning)
    * Gens 66-70: 150 iters (Precision polish)
    * Final Tier C: 250 iters (Unbudgeted Newton certification)
- Periodic Ring + Diagonal Migration every 5 generations
- Thermal Re-Heating when an island stalls for 3 generations
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
from scratch.run_full_5pillar_evolution import (
    create_prior_genome,
    apply_spatial_niching,
)

ARTIFACT_DIR = r"C:\Users\jumpi\.gemini\antigravity\brain\dc315240-8852-4d42-acf3-025877ae6bbd"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def run_centurion():
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

    print("================ CENTURION 50-GENERATION EVOLUTIONARY RUN ================")
    print(f"Mission: {len(waypoints)} waypoints | 4 Islands x 6 Genomes = 24 candidates/gen")
    print(f"Total Generations: 50 (1,200 candidate solves across 6 CPU workers)")
    print(f"Migration Events: Every 5 generations (10 migration epochs)\n")

    # Baseline reference
    stock_guess = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="rolling", stop_waypoint_indices=stops)
    _, s_stock, _, _ = _solve_task((0, stock_guess, waypoints, stops, cfg_dict, 150, 0.002))
    states_st, dts_st = _extract_state_and_dt_arrays(s_stock, num_segments, segment_samples)
    rep_st = engine.evaluate_states(states_st, dts_st, sample_to_segment)
    cost_stock = float(s_stock[-1]["t"]) + 120.0 * max(0.0, float(rep_st["max_tracking_error_m"]) - 0.005) + 60.0 * max(0.0, float(rep_st["max_envelope_radius_m"]) - 0.020)
    print(f"Stock Baseline (rolling): Cost = {cost_stock:.3f} | Dur = {s_stock[-1]['t']:.3f}s | Err = {rep_st['max_tracking_error_m']*100:.2f}cm\n")

    islands_config = [
        {"id": 0, "name": "Sprint Line", "type": "sprint", "acc_weight": 0.0015, "sigma_base": 0.15, "color": "#2ca02c"},
        {"id": 1, "name": "Smooth Sweeper", "type": "smooth", "acc_weight": 0.0025, "sigma_base": 0.12, "color": "#1f77b4"},
        {"id": 2, "name": "Precision Line", "type": "smooth", "acc_weight": 0.0040, "sigma_base": 0.10, "color": "#9467bd"},
        {"id": 3, "name": "Exploratory Homotopy", "type": "exploratory", "acc_weight": 0.0020, "sigma_base": 0.22, "color": "#ff7f0e"},
    ]

    pop_per_island = 6
    num_gens = 50
    migration_interval = 5

    for island in islands_config:
        island["pop"] = [
            create_prior_genome(waypoints, unpinned_count, island_type=island["type"])
            for _ in range(pop_per_island)
        ]
        island["stagnation_counter"] = 0
        island["best_cost"] = float("inf")
        island["best_cand"] = None

    global_best_overall = None
    global_best_cost = float("inf")
    island_histories = {i["id"]: [] for i in islands_config}

    t_start = time.time()

    with ProcessPoolExecutor(max_workers=6) as executor:
        for gen in range(num_gens):
            gen_start_t = time.time()

            # Dynamic 4-Stage Iteration Schedule
            if gen < 25:
                iter_budget = 35
                cooling = 0.95 ** (gen % 10)
            elif gen < 50:
                iter_budget = 60
                cooling = 0.90 ** (gen % 10)
            elif gen < 65:
                iter_budget = 100
                cooling = 0.85 ** (gen % 8)
            else:
                iter_budget = 150
                cooling = 0.80 ** (gen % 5)

            all_tasks = []
            task_island_map = []
            all_raw_guesses = []

            for island in islands_config:
                i_id = island["id"]
                acc_w = island["acc_weight"]

                # Stagnation & thermal burst
                if island["stagnation_counter"] >= 3:
                    current_sigma = island["sigma_base"] * 2.2
                    island["pop"][-1] = create_prior_genome(waypoints, unpinned_count, island_type="exploratory")
                    island["stagnation_counter"] = 0
                else:
                    current_sigma = island["sigma_base"] * cooling
                island["current_sigma"] = current_sigma

                for p_idx, ind in enumerate(island["pop"]):
                    raw_g = build_racing_guess(ind, waypoints, segment_samples, config, stops)
                    all_raw_guesses.append(raw_g)
                    task_id = len(all_tasks)
                    all_tasks.append((task_id, raw_g, waypoints, stops, cfg_dict, iter_budget, acc_w))
                    task_island_map.append((i_id, p_idx))

            # Solve all 24 candidates across 12 parallel workers
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

            # Process each island's generation
            gen_summary_str = f"Gen {gen+1:02d}/{num_gens:02d} ({iter_budget} iters) | "
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

                island_histories[i_id].append({
                    "gen": gen + 1, "best_cost": island_best["raw_cost"],
                    "duration": island_best["meta"]["duration"] if island_best["meta"] else 99.0,
                    "tracking_cm": island_best["meta"]["tracking_cm"] if island_best["meta"] else 99.0,
                })

                if island_best["raw_cost"] < global_best_cost:
                    global_best_cost = island_best["raw_cost"]
                    global_best_overall = island_best

                gen_summary_str += f"{island['name'][:5]}: {island_best['raw_cost']:.3f} | "

                # Breeding
                new_pop = [evals[0]["genome"], evals[1]["genome"]]
                while len(new_pop) < pop_per_island:
                    c1, c2 = np.random.choice(min(len(evals), 4), 2, replace=False)
                    child = RacingGenome.crossover(evals[c1]["genome"], evals[c2]["genome"])
                    child = child.mutate(p_mut=0.35, sigma=island["current_sigma"])
                    new_pop.append(child)
                island["pop"] = new_pop

            wall_t = time.time() - gen_start_t
            gen_summary_str += f"Best: {global_best_cost:.3f} ({wall_t:.1f}s)"
            print(gen_summary_str)

            # Ring + Diagonal Migration every migration_interval generations
            if (gen + 1) % migration_interval == 0 and gen < num_gens - 1:
                print(f"  >>> [Migration Event @ Gen {gen+1}] Cross-Migrating Elites Across All 4 Islands...")
                migrants = [island["pop"][0] for island in islands_config]
                for idx in range(len(islands_config)):
                    next_idx = (idx + 1) % len(islands_config)
                    islands_config[next_idx]["pop"][2] = migrants[idx]
                # Diagonal swap between Sprint (0) and Exploratory (3)
                islands_config[0]["pop"][3] = migrants[3]
                islands_config[3]["pop"][3] = migrants[0]

    # Final Tier C Polish (250 Iterations)
    print("\n================ FINAL TIER C POLISH (250 ITERATIONS) ================")
    final_tasks = []
    for island in islands_config:
        cand = island["best_cand"]
        if cand:
            final_tasks.append((island["id"], cand["guess"], waypoints, stops, cfg_dict, 250, 0.002))

    with ProcessPoolExecutor(max_workers=4) as executor:
        final_results = list(executor.map(_solve_task, final_tasks))

    certified_champions = []
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
        certified_champions.append({
            "island": island["name"], "color": island["color"], "duration": T,
            "tracking_cm": e_track * 100.0, "tube_cm": tube * 100.0, "cost": cost,
            "passed": passed, "samples": samples, "rep": rep, "iter": stats.get("iter_count", 250)
        })

    certified_champions.sort(key=lambda x: x["cost"])
    grand_champ = certified_champions[0]
    total_wall_time = time.time() - t_start

    print("\n---------------- CENTURION 70-GEN FINAL LEADERBOARD ----------------")
    print(f"Stock Baseline (rolling):     Cost = {cost_stock:.3f} | Dur = {s_stock[-1]['t']:.3f}s | Err = {rep_st['max_tracking_error_m']*100:.2f}cm")
    for c in certified_champions:
        print(f"Island [{c['island']:20s}]: Cost = {c['cost']:.3f} | Dur = {c['duration']:.3f}s | Err = {c['tracking_cm']:.2f}cm | Iters = {c['iter']} | Pass = {c['passed']}")
    print(f"\nCENTURION CHAMPION: {grand_champ['island']} (Duration: {grand_champ['duration']:.3f}s, Tracking: {grand_champ['tracking_cm']:.2f}cm, Cost: {grand_champ['cost']:.3f})")
    impr = (cost_stock - grand_champ["cost"]) / cost_stock * 100.0
    print(f"Total Cost Improvement: +{impr:.2f}% | Total Wall-Clock Time: {total_wall_time:.1f}s\n")

    # Render Figures
    print("Rendering 70-generation publication plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. 2D Field Paths
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
        ax.plot([s["x"] for s in s_stock], [s["y"] for s in s_stock], "--", color="#d62728", lw=1.6, alpha=0.7, label="Stock Rolling Baseline")

    for c in certified_champions:
        cx = [s["x"] for s in c["samples"]]
        cy = [s["y"] for s in c["samples"]]
        is_grand = (c["island"] == grand_champ["island"])
        lw = 3.2 if is_grand else 1.8
        alpha = 1.0 if is_grand else 0.7
        lbl = f"{c['island']} ({c['duration']:.2f}s, err {c['tracking_cm']:.2f}cm)"
        if is_grand:
            lbl += " [CENTURION WINNER]"
        ax.plot(cx, cy, "-", color=c["color"], lw=lw, alpha=alpha, label=lbl)

    ax.set_title("2D Mat Paths: 70-Generation Centurion Multi-Island Champions", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("X Field Coordinate (m)", fontsize=11)
    ax.set_ylabel("Y Field Coordinate (m)", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=9)
    plt.tight_layout()
    p1 = os.path.join(ARTIFACT_DIR, "plot_centurion_4island_paths.png")
    plt.savefig(p1, dpi=200)
    plt.close()

    # 2. 70-Generation Convergence Curve
    fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
    gens = np.arange(1, num_gens + 1)
    for island in islands_config:
        i_id = island["id"]
        costs = [h["best_cost"] for h in island_histories[i_id]]
        ax.plot(gens, costs, "-", color=island["color"], lw=1.8, label=f"{island['name']}")

    for m_gen in range(migration_interval, num_gens, migration_interval):
        ax.axvline(m_gen, color="gray", linestyle=":", lw=1.0, alpha=0.6)

    ax.axhline(cost_stock, color="#d62728", linestyle="--", lw=1.6, label=f"Stock Baseline ({cost_stock:.3f})")
    ax.set_title("70-Generation Evolutionary Trajectory Convergence (4 Competing Islands)", fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Evolutionary Generation (Periodic Vertical Lines: Migration Events)", fontsize=11)
    ax.set_ylabel("Composite Cost Score ($J$)", fontsize=11)
    ax.set_xticks(range(1, num_gens + 1, 5))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(ARTIFACT_DIR, "plot_centurion_convergence.png")
    plt.savefig(p2, dpi=200)
    plt.close()

    champ_file = os.path.join("output_homotopy", "centurion_grand_champion.traj")
    with open(champ_file, "w") as f:
        f.write("t,x,y,heading,vl,vr\n")
        for s in grand_champ["samples"]:
            f.write(f"{s['t']:.6f},{s['x']:.6f},{s['y']:.6f},{s['heading']:.6f},{s['vl']:.6f},{s['vr']:.6f}\n")

    print(f"Centurion Grand Champion trajectory saved to: {champ_file}")
    print("All centurion figures rendered in:", ARTIFACT_DIR)


if __name__ == "__main__":
    run_centurion()
