"""
Multi-Shape Benchmark & Optimal Control Discovery
=================================================
Compares Stock Baseline vs. Evolutionary Optimal Control across 4 distinct shapes:
1. S-Curve Inflection & 180-deg Heading Reversal (example_s_curve.json)
2. 90-Degree Box with Pinpoint Stops (example_with_stops.json)
3. Alternating Slalom (example_with_events.json)
4. Hairpin 180-Degree U-Turn
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from typing import Any, Dict, List, Optional, Tuple
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


def solve_shape(
    shape_name: str,
    waypoints: List[Tuple[float, float, Optional[float]]],
    stops: List[int],
    config: RobotConfig,
    optimizer: TrajectoryOptimizer,
    engine: CompiledReachabilityEngine,
    cfg_dict: Dict[str, Any],
) -> Dict[str, Any]:
    print(f"\n--- Solving Shape: {shape_name} ({len(waypoints)} waypoints) ---")
    num_segments = len(waypoints) - 1
    segment_samples = optimizer.get_segment_sample_counts(waypoints, 10)
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    unpinned_count = sum(1 for i, w in enumerate(waypoints) if w[2] is None and 0 < i < len(waypoints) - 1)

    # 1. Stock Baseline Solve
    g_stock = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style="rolling", stop_waypoint_indices=stops)
    _, s_stock, st_stock, _ = _solve_task((0, g_stock, waypoints, stops, cfg_dict, 150, 0.002))

    t_stock = float(s_stock[-1]["t"]) if s_stock else 99.0
    err_stock = 0.0
    tube_stock = 0.0
    if s_stock:
        states_st, dts_st = _extract_state_and_dt_arrays(s_stock, num_segments, segment_samples)
        rep_st = engine.evaluate_states(states_st, dts_st, sample_to_segment)
        err_stock = float(rep_st["max_tracking_error_m"]) * 100.0
        tube_stock = float(rep_st["max_envelope_radius_m"]) * 100.0
    print(f"  Stock Rolling Baseline -> Dur: {t_stock:.3f}s | Err: {err_stock:.2f}cm")

    # 2. Evolutionary Racing Search (8 Genomes x 12 Generations)
    pop = [RacingGenome.create_default(num_segments, unpinned_count)]
    for _ in range(7):
        pop.append(RacingGenome.create_random(num_segments, unpinned_count))

    best_evo_cost = float("inf")
    best_evo_samples = None
    best_evo_meta = None

    for gen in range(12):
        sigma = 0.16 * (0.88 ** gen)
        gen_evals = []
        for ind in pop:
            raw_g = build_racing_guess(ind, waypoints, segment_samples, config, stops)
            _, samples, stats, _ = _solve_task((0, raw_g, waypoints, stops, cfg_dict, 100, 0.002))
            if not samples:
                continue
            states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
            rep = engine.evaluate_states(states, dts, sample_to_segment)
            T = float(samples[-1]["t"])
            e_track = float(rep["max_tracking_error_m"])
            tube = float(rep["max_envelope_radius_m"])
            cost = T + 120.0 * max(0.0, e_track - 0.005) + 60.0 * max(0.0, tube - 0.020)
            gen_evals.append((ind, cost, samples, {"duration": T, "tracking_cm": e_track * 100.0, "tube_cm": tube * 100.0, "cost": cost}))

        if not gen_evals:
            continue
        gen_evals.sort(key=lambda x: x[1])
        g_best = gen_evals[0]
        if g_best[1] < best_evo_cost:
            best_evo_cost = g_best[1]
            best_evo_samples = g_best[2]
            best_evo_meta = g_best[3]

        # Breed
        new_pop = [gen_evals[0][0], gen_evals[1][0]]
        while len(new_pop) < 8:
            p1, p2 = np.random.choice(min(len(gen_evals), 4), 2, replace=False)
            child = RacingGenome.crossover(gen_evals[p1][0], gen_evals[p2][0])
            child = child.mutate(p_mut=0.35, sigma=sigma)
            new_pop.append(child)
        pop = new_pop

    t_evo = best_evo_meta["duration"] if best_evo_meta else 99.0
    err_evo = best_evo_meta["tracking_cm"] if best_evo_meta else 99.0
    print(f"  Evolved Racing Line    -> Dur: {t_evo:.3f}s | Err: {err_evo:.2f}cm (Cost: {best_evo_cost:.3f})")
    impr = (t_stock - t_evo) / t_stock * 100.0 if t_stock > 0 else 0.0
    print(f"  Speedup: +{impr:.1f}%")

    return {
        "name": shape_name,
        "waypoints": waypoints,
        "stops": stops,
        "stock_samples": s_stock,
        "stock_duration": t_stock,
        "stock_err": err_stock,
        "evo_samples": best_evo_samples,
        "evo_duration": t_evo,
        "evo_err": err_evo,
        "evo_cost": best_evo_cost,
        "impr": impr,
    }


def main():
    with open("examples/fll_choreo.json") as f:
        cfg_dict = json.load(f)
    config = RobotConfig(cfg_dict)
    jax_cfg = JAXRobotConfig(config)
    val = ImmraxValidator(jax_cfg)
    engine = CompiledReachabilityEngine(val)
    optimizer = TrajectoryOptimizer(config)

    # Define the 4 shapes
    shapes = []

    # Shape 1: S-Curve Heading Reversal
    with open("examples/example_s_curve.json") as f:
        data1 = json.load(f)
    wps1 = [(w["x"], w["y"], w.get("heading")) for w in data1]
    stops1 = [i for i, w in enumerate(data1) if w.get("stop", False)]
    shapes.append(("S-Curve Reversal (180° Inflection)", wps1, stops1))

    # Shape 2: 90° Box with Pinpoint Stops
    with open("examples/example_with_stops.json") as f:
        data2 = json.load(f)
    wps2 = [(w["x"], w["y"], w.get("heading")) for w in data2]
    stops2 = [i for i, w in enumerate(data2) if w.get("stop", False)]
    shapes.append(("90° Box Loop with Stops", wps2, stops2))

    # Shape 3: Alternating Slalom
    with open("examples/example_with_events.json") as f:
        data3 = json.load(f)
    wps3 = [(w["x"], w["y"], w.get("heading")) for w in data3]
    stops3 = [i for i, w in enumerate(data3) if w.get("stop", False)]
    shapes.append(("Alternating Slalom", wps3, stops3))

    # Shape 4: Hairpin 180° U-Turn
    wps4 = [
        (0.0, 0.0, 0.0),
        (0.6, 0.15, None),
        (0.7, 0.45, math.pi / 2),
        (0.6, 0.75, None),
        (0.0, 0.90, math.pi),
    ]
    stops4 = [0, 4]
    shapes.append(("180° Hairpin U-Turn", wps4, stops4))

    # Solve all shapes
    results = []
    for name, wps, stops in shapes:
        res = solve_shape(name, wps, stops, config, optimizer, engine, cfg_dict)
        results.append(res)

    # -------------------------------------------------------------
    # Render 4-Panel Multi-Shape Visual Comparison
    # -------------------------------------------------------------
    print("\nRendering 4-Panel Multi-Shape Figure...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=200)
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        wps = res["waypoints"]
        stops = set(res["stops"])

        # Plot waypoints
        wx = [w[0] for w in wps]
        wy = [w[1] for w in wps]
        ax.scatter(wx, wy, c="#111111", s=90, zorder=5, label="Waypoints")
        for idx, (x, y, th) in enumerate(wps):
            is_stop = idx in stops
            lbl = f"W{idx}" + (" (STOP)" if is_stop else "")
            ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=8, weight="bold", color="#8B0000" if is_stop else "#111111")

        # Plot Stock Baseline
        if res["stock_samples"]:
            sx = [s["x"] for s in res["stock_samples"]]
            sy = [s["y"] for s in res["stock_samples"]]
            ax.plot(sx, sy, "--", color="#d62728", lw=2.0, alpha=0.75,
                    label=f"Stock ({res['stock_duration']:.2f}s, err {res['stock_err']:.2f}cm)")

        # Plot Evolved Racing Line
        if res["evo_samples"]:
            ex = [s["x"] for s in res["evo_samples"]]
            ey = [s["y"] for s in res["evo_samples"]]
            ax.plot(ex, ey, "-", color="#2ca02c", lw=2.8,
                    label=f"Evolved ({res['evo_duration']:.2f}s, err {res['evo_err']:.2f}cm) [+{res['impr']:.1f}%]")

        ax.set_title(f"Shape {i+1}: {res['name']}", fontsize=12, weight="bold", pad=10)
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.axis("equal")
        ax.legend(loc="best", frameon=True, framealpha=0.95, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    p_shapes = os.path.join(ARTIFACT_DIR, "plot_multi_shape_comparison.png")
    plt.savefig(p_shapes, dpi=200)
    plt.close()

    # Save summary JSON
    summary_data = [
        {
            "name": r["name"],
            "stock_duration": r["stock_duration"],
            "stock_err": r["stock_err"],
            "evo_duration": r["evo_duration"],
            "evo_err": r["evo_err"],
            "speedup_percent": r["impr"],
        }
        for r in results
    ]
    with open(os.path.join(ARTIFACT_DIR, "multi_shape_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nSaved Multi-Shape comparison plot to:", p_shapes)
    print("Multi-Shape summary JSON saved successfully!")


if __name__ == "__main__":
    main()
