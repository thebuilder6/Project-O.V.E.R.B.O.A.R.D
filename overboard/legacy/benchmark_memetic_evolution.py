"""
Memetic Evolution Benchmark & Visualization Script
=================================================
Compares stock candidate trajectories against Memetic Evolution on the
10-waypoint complex mission (examples/example_complex_mission.json).
Generates high-resolution publication plots and compiles a score leaderboard.
"""

import json
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from immrax_validator import ImmraxValidator, CONTROL_DT
from optimizer import TrajectoryOptimizer
from surrogate_polish import (
    CompiledReachabilityEngine,
    generate_homotopy_bank,
    screen_candidates_jax,
    select_topological_representatives,
    solve_pyramid_candidates,
    _extract_state_and_dt_arrays,
)
try:
    from legacy.evolutionary_research.genetic_algorithm import (
        recombine_candidate_segments,
        run_memetic_evolution,
    )
except ImportError:
    from evolutionary_research.genetic_algorithm import (
        recombine_candidate_segments,
        run_memetic_evolution,
    )
from main import _load_waypoints


def run_benchmark():
    os.makedirs("output_memetic", exist_ok=True)
    with open("examples/fll_choreo.json", "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    robot_cfg = RobotConfig(cfg_data)
    jax_cfg = JAXRobotConfig(robot_cfg)
    validator = ImmraxValidator(jax_cfg, control_dt=0.02, propagation="plant")
    engine = CompiledReachabilityEngine(validator)

    wps, events, thresholds, all_stop_indices = _load_waypoints("examples/example_complex_mission.json")

    num_segments = len(wps) - 1
    opt = TrajectoryOptimizer(robot_cfg)

    # 1. Coarse Mesh (37 samples) Baseline
    coarse_samples = opt.get_segment_sample_counts(wps, 4, proportional=True)
    coarse_s2s = list(np.repeat(np.arange(num_segments), coarse_samples))

    # 2. h-Adaptive Mesh (Curvature Refined)
    segment_samples = opt.get_adaptive_mesh_sample_counts(wps, apex_samples=8, straight_samples=4)
    segment_samples = [int(s) for s in segment_samples]
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))

    print(f"Loaded mission with {len(wps)} waypoints ({num_segments} segments).")
    print(f"  Coarse Mesh:      {sum(coarse_samples)+1} samples {coarse_samples}")
    print(f"  h-Adaptive Mesh:  {sum(segment_samples)+1} samples {segment_samples}")

    # 1. Stock Candidate 1: Equal-Duration Baseline (Coarse)
    print("\n--- Solving Stock Candidate 1: Equal-Duration Baseline (Coarse) ---")
    t0 = time.time()
    s_eq, st_eq = opt.solve(
        wps,
        num_samples_per_segment=coarse_samples,
        per_segment_dt=False,
        accuracy_weight=0.0,
        verbose=False,
    )
    t_eq = time.time() - t0
    eq_states, eq_dts = _extract_state_and_dt_arrays(s_eq, num_segments, coarse_samples)
    rep_eq = engine.evaluate_states(eq_states, eq_dts, coarse_s2s)
    eq_dur = float(s_eq[-1]["t"])
    eq_track = float(rep_eq["max_tracking_error_m"])
    eq_tube = float(rep_eq["max_envelope_radius_m"])
    eq_score = eq_dur + 120.0 * max(0.0, eq_track - 0.005) + 60.0 * max(0.0, eq_tube - 0.020)
    print(f"  Stock Equal-Dur: {eq_dur:.3f} s, Track: {eq_track*100:.2f} cm, Tube: {eq_tube*100:.2f} cm, Score: {eq_score:.3f} (Wall: {t_eq:.2f} s)")

    # 2. Stock Candidate 2: Per-Segment Baseline (h-Adaptive Mesh)
    print("\n--- Solving Stock Candidate 2: Per-Segment Baseline (h-Adaptive Mesh) ---")
    t0 = time.time()
    s_ps, st_ps = opt.solve(
        wps,
        num_samples_per_segment=segment_samples,
        per_segment_dt=True,
        accuracy_weight=0.0,
        verbose=False,
    )
    t_ps = time.time() - t0
    ps_states, ps_dts = _extract_state_and_dt_arrays(s_ps, num_segments, segment_samples)
    rep_ps = engine.evaluate_states(ps_states, ps_dts, sample_to_segment)
    ps_dur = float(s_ps[-1]["t"])
    ps_track = float(rep_ps["max_tracking_error_m"])
    ps_tube = float(rep_ps["max_envelope_radius_m"])
    ps_score = ps_dur + 120.0 * max(0.0, ps_track - 0.005) + 60.0 * max(0.0, ps_tube - 0.020)
    print(f"  h-Adaptive Per-Seg: {ps_dur:.3f} s, Track: {ps_track*100:.2f} cm, Tube: {ps_tube*100:.2f} cm, Score: {ps_score:.3f} (Wall: {t_ps:.2f} s)")
    ps_dur = float(s_ps[-1]["t"])
    ps_track = float(rep_ps["max_tracking_error_m"])
    ps_tube = float(rep_ps["max_envelope_radius_m"])
    ps_score = ps_dur + 120.0 * max(0.0, ps_track - 0.005) + 60.0 * max(0.0, ps_tube - 0.020)
    print(f"  Stock Per-Seg:   {ps_dur:.3f} s, Track: {ps_track*100:.2f} cm, Tube: {ps_tube*100:.2f} cm, Score: {ps_score:.3f} (Wall: {t_ps:.2f} s)")

    # 3. Generate 10 screen-then-refine pyramid finalists to seed evolution
    print("\n--- Generating & Solving 10 Population Seeds via Pyramid ---")
    t0 = time.time()
    cand_states, cand_dts, _ = generate_homotopy_bank(
        wps, segment_samples,
        track_width=float(robot_cfg.track_width),
        v_max=float(robot_cfg.max_linear_speed()),
    )
    surr_costs = screen_candidates_jax(engine, cand_states, cand_dts, sample_to_segment)
    selected_indices = select_topological_representatives(cand_states, surr_costs, k_clusters=10, min_separation_m=0.10)
    survivor_guesses = [np.concatenate([cand_dts[i], cand_states[i].flatten()]) for i in selected_indices]

    base_kw = {
        "num_samples_per_segment": segment_samples,
        "per_segment_dt": True,
        "cost_metric": "discretization",
        "epigraph_constraints": True,
        "accuracy_weight": 0.0,
        "force_weight": 0.0,
        "verbose": False,
        "self_consistent": False,
        "solver_options": {"max_iter": 200},
    }
    initial_pop_solved = solve_pyramid_candidates(robot_cfg, wps, survivor_guesses, base_kw, max_workers=10)
    t_pyramid = time.time() - t0
    converged_seeds = sum(1 for _, st in initial_pop_solved if st.get("converged", False))
    print(f"  Pyramid stage solved {converged_seeds}/10 seeds in {t_pyramid:.2f} s")

    # 4. Single-Pass Chimera Recombination
    print("\n--- Running Single-Pass Chimera Recombination ---")
    t0 = time.time()
    chim_samples, chim_stats, chim_rep = recombine_candidate_segments(
        initial_pop_solved, wps, robot_cfg, engine, solve_kwargs=base_kw, max_polish_iters=250
    )
    t_chim = time.time() - t0
    if chim_samples:
        ch_dur = chim_rep.get("duration_s", float(chim_samples[-1]["t"]))
        ch_track = chim_rep.get("tracking_m", 0.0)
        ch_tube = chim_rep.get("envelope_m", 0.0)
        ch_score = ch_dur + 120.0 * max(0.0, ch_track - 0.005) + 60.0 * max(0.0, ch_tube - 0.020)
        print(f"  Single Chimera:  {ch_dur:.3f} s, Track: {ch_track*100:.2f} cm, Tube: {ch_tube*100:.2f} cm, Score: {ch_score:.3f} (Wall: {t_chim:.2f} s)")
    else:
        ch_dur, ch_track, ch_tube, ch_score = None, None, None, None

    # 5. Memetic Evolution (5 Generations, Pop=10, Truncate Worst 3)
    print("\n--- Running Memetic Evolution (5 Generations, Pop=10) ---")
    t0 = time.time()
    mem_samples, mem_stats, evo_report = run_memetic_evolution(
        initial_pop_solved,
        wps,
        robot_cfg,
        engine,
        num_generations=5,
        population_size=10,
        discard_worst_count=3,
        solve_kwargs=base_kw,
        max_polish_iters=150,
        max_workers=10,
        verbose=True,
    )
    t_mem = time.time() - t0
    mem_dur = evo_report["champion_duration"]
    mem_score = evo_report["champion_score"]
    mem_champ_ind = evo_report["population_leaderboard"][0]
    mem_track = mem_champ_ind["tracking_cm"] / 100.0
    mem_tube = mem_champ_ind["envelope_cm"] / 100.0
    print(f"  Memetic Champion: {mem_dur:.3f} s, Track: {mem_track*100:.2f} cm, Tube: {mem_tube*100:.2f} cm, Score: {mem_score:.3f} (Total Evo Wall: {t_mem:.2f} s)")

    # Save trajectories and scores
    def _save_traj(samples, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(f"{s['t']:.4f},{s['x']:.4f},{s['y']:.4f},{s['heading']:.4f},{s['vl']:.4f},{s['vr']:.4f}\n")

    _save_traj(s_eq, "output_memetic/stock_equal_dur.traj")
    _save_traj(s_ps, "output_memetic/stock_per_seg.traj")
    if chim_samples:
        _save_traj(chim_samples, "output_memetic/chimera_single_pass.traj")
    _save_traj(mem_samples, "output_memetic/memetic_champion.traj")

    summary_scores = {
        "stock_equal_duration": {
            "duration_s": eq_dur,
            "tracking_error_cm": eq_track * 100.0,
            "envelope_radius_cm": eq_tube * 100.0,
            "composite_score": eq_score,
            "wall_time_s": t_eq,
        },
        "stock_per_segment": {
            "duration_s": ps_dur,
            "tracking_error_cm": ps_track * 100.0,
            "envelope_radius_cm": ps_tube * 100.0,
            "composite_score": ps_score,
            "wall_time_s": t_ps,
        },
        "single_pass_chimera": {
            "duration_s": ch_dur,
            "tracking_error_cm": ch_track * 100.0 if ch_track else None,
            "envelope_radius_cm": ch_tube * 100.0 if ch_tube else None,
            "composite_score": ch_score,
            "wall_time_s": t_chim,
        },
        "memetic_evolution_champion": {
            "duration_s": mem_dur,
            "tracking_error_cm": mem_track * 100.0,
            "envelope_radius_cm": mem_tube * 100.0,
            "composite_score": mem_score,
            "wall_time_s": t_mem,
            "champion_tag": mem_champ_ind["tag"],
            "generation": mem_champ_ind["generation"],
        },
        "generations_history": evo_report["generations"],
        "final_population_leaderboard": evo_report["population_leaderboard"],
    }
    with open("output_memetic/memetic_scores.json", "w", encoding="utf-8") as f:
        json.dump(summary_scores, f, indent=2)

    # --------------------------------------------------------------------------
    # VISUALIZATION SUITE (4 High-Resolution Plots)
    # --------------------------------------------------------------------------
    print("\n--- Generating High-Resolution Comparative Visuals ---")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # PLOT 1: 2D Spatial Mat Trajectories
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    # Draw field boundary
    ax.plot([0, 2.36, 2.36, 0, 0], [0, 0, 1.14, 1.14, 0], "k--", lw=1.5, alpha=0.5, label="Field Boundary (2.36m x 1.14m)")

    # Draw Waypoints
    wp_xs = [wp[0] for wp in wps]
    wp_ys = [wp[1] for wp in wps]
    ax.scatter(wp_xs, wp_ys, color="black", s=70, zorder=6, label="Mission Waypoints (10)")
    for i, (wx, wy, wh) in enumerate(wps):
        ax.annotate(f"W{i}", (wx, wy), textcoords="offset points", xytext=(6, 6), fontweight="bold", fontsize=9)
        if wh is not None:
            dx = 0.05 * np.cos(wh)
            dy = 0.05 * np.sin(wh)
            ax.arrow(wx, wy, dx, dy, head_width=0.02, head_length=0.025, fc="black", ec="black", zorder=7)

    # Trajectories
    ax.plot([s["x"] for s in s_eq], [s["y"] for s in s_eq], "r--", lw=2.0, alpha=0.8,
            label=f"Stock Eq-Dur: {eq_dur:.3f} s (Track {eq_track*100:.1f} cm)")
    ax.plot([s["x"] for s in s_ps], [s["y"] for s in s_ps], "m-.", lw=2.0, alpha=0.8,
            label=f"Stock Per-Seg: {ps_dur:.3f} s (Track {ps_track*100:.1f} cm)")

    # Generation snapshots
    snaps = evo_report["snapshots"]
    if 0 in snaps:
        ax.plot([s["x"] for s in snaps[0]], [s["y"] for s in snaps[0]], color="orange", lw=1.8, linestyle=":", alpha=0.75,
                label=f"Gen 0 Leader: {evo_report['generations'][0]['best_duration']:.3f} s")
    if 2 in snaps:
        ax.plot([s["x"] for s in snaps[2]], [s["y"] for s in snaps[2]], color="deepskyblue", lw=1.8, linestyle="-.", alpha=0.75,
                label=f"Gen 2 Leader: {evo_report['generations'][2]['best_duration']:.3f} s")
    ax.plot([s["x"] for s in mem_samples], [s["y"] for s in mem_samples], color="#008000", lw=3.0, zorder=5,
            label=f"Gen 5 Champion: {mem_dur:.3f} s (Track {mem_track*100:.1f} cm, Tube {mem_tube*100:.1f} cm)")

    ax.set_title("2D Trajectory Comparison: Stock Baseline vs. Memetic Evolution Generations", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X Position [m]", fontsize=11)
    ax.set_ylabel("Y Position [m]", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 2.40)
    ax.set_ylim(-0.05, 1.20)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9)
    plt.tight_layout()
    plt.savefig("output_memetic/plot_memetic_2d_paths.png", dpi=300)
    plt.close()

    # PLOT 2: Generation Convergence Curve (Best vs Mean Fitness)
    gens = [g["generation"] for g in evo_report["generations"]]
    b_scores = [g["best_score"] for g in evo_report["generations"]]
    m_scores = [g["mean_score"] for g in evo_report["generations"]]
    b_durs = [g["best_duration"] for g in evo_report["generations"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    ax1.plot(gens, m_scores, "o--", color="orange", lw=2.2, markersize=8, label="Population Mean Score")
    ax1.plot(gens, b_scores, "s-", color="#1f77b4", lw=2.5, markersize=9, label="Population Best Score (Elite)")
    ax1.axhline(eq_score, color="red", linestyle=":", lw=1.8, label=f"Stock Eq-Dur Score ({eq_score:.2f})")
    ax1.set_title("Fitness Score Convergence Across Generations", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Evolution Generation", fontsize=10)
    ax1.set_ylabel("Composite Cost (Lower is Better)", fontsize=10)
    ax1.set_xticks(gens)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    ax2.plot(gens, b_durs, "d-", color="#2ca02c", lw=2.5, markersize=9, label="Best Trajectory Duration [s]")
    ax2.axhline(eq_dur, color="red", linestyle=":", lw=1.8, label=f"Stock Eq-Dur Duration ({eq_dur:.3f} s)")
    ax2.axhline(ps_dur, color="purple", linestyle=":", lw=1.8, label=f"Stock Per-Seg Duration ({ps_dur:.3f} s)")
    ax2.set_title("Trajectory Duration Improvement", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Evolution Generation", fontsize=10)
    ax2.set_ylabel("Duration [s]", fontsize=10)
    ax2.set_xticks(gens)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Memetic Evolution Convergence Dynamics (Pop=10, 30% Truncation, Chimera Crossover)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("output_memetic/plot_memetic_convergence.png", dpi=300)
    plt.close()

    # PLOT 3: Segment Durations / Gene Breakdown
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    eq_seg_times = [float(eq_dts[i]) * segment_samples[i] for i in range(num_segments)]
    ps_seg_times = [float(ps_dts[i]) * segment_samples[i] for i in range(num_segments)]
    mem_seg_dts = mem_stats.get("segment_dt", [0.0]*num_segments)
    mem_seg_times = [float(mem_seg_dts[i]) * segment_samples[i] for i in range(num_segments)]

    x_segs = np.arange(num_segments)
    width = 0.28
    ax.bar(x_segs - width, eq_seg_times, width, label="Stock Equal-Duration", color="#e74c3c", alpha=0.85)
    ax.bar(x_segs, ps_seg_times, width, label="Stock Per-Segment", color="#9b59b6", alpha=0.85)
    ax.bar(x_segs + width, mem_seg_times, width, label="Memetic Champion", color="#2ecc71", alpha=0.95)

    ax.set_title("Segment-Wise Gene Analysis: Allocation of Time per Mission Segment", fontsize=12, fontweight="bold")
    ax.set_xlabel("Segment Index (W_i -> W_{i+1})", fontsize=11)
    ax.set_ylabel("Segment Time [s]", fontsize=11)
    ax.set_xticks(x_segs)
    ax.set_xticklabels([f"Seg {i}\n(W{i}->W{i+1})" for i in range(num_segments)], fontsize=9)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig("output_memetic/plot_memetic_segment_dna.png", dpi=300)
    plt.close()

    # PLOT 4: Velocity and Wheel Accelerations
    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=300)
    # Stock Baseline velocities
    t_eq_arr = [s["t"] for s in s_eq]
    v_eq_arr = [0.5 * (s["vl"] + s["vr"]) for s in s_eq]
    t_mem_arr = [s["t"] for s in mem_samples]
    v_mem_arr = [0.5 * (s["vl"] + s["vr"]) for s in mem_samples]

    ax_v.plot(t_eq_arr, v_eq_arr, "r--", lw=1.8, label=f"Stock Eq-Dur (Total {eq_dur:.2f} s)")
    ax_v.plot(t_mem_arr, v_mem_arr, color="#008000", lw=2.2, label=f"Memetic Champion (Total {mem_dur:.2f} s)")
    ax_v.axhline(float(robot_cfg.max_linear_speed()), color="black", linestyle=":", lw=1.5, label="Max Linear Speed Limit")
    ax_v.set_ylabel("Forward Velocity [m/s]", fontsize=10)
    ax_v.set_title("Kinematic Speed Profiles: Stock Baseline vs. Memetic Champion", fontsize=11, fontweight="bold")
    ax_v.legend(loc="upper right", frameon=True, fontsize=9)
    ax_v.grid(True, alpha=0.3)

    # Wheel velocities for Memetic Champion
    ax_a.plot(t_mem_arr, [s["vl"] for s in mem_samples], color="#3498db", lw=1.8, label="Left Wheel Velocity (Memetic)")
    ax_a.plot(t_mem_arr, [s["vr"] for s in mem_samples], color="#e67e22", lw=1.8, label="Right Wheel Velocity (Memetic)")
    ax_a.axhline(0, color="gray", lw=0.8)
    ax_a.set_ylabel("Wheel Speed [m/s]", fontsize=10)
    ax_a.set_xlabel("Time [s]", fontsize=10)
    ax_a.set_title("Individual Wheel Velocities (Left & Right Differential Drive)", fontsize=11, fontweight="bold")
    ax_a.legend(loc="upper right", frameon=True, fontsize=9)
    ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output_memetic/plot_memetic_profiles.png", dpi=300)
    plt.close()

    print("All plots generated and saved to output_memetic/.")
    return summary_scores


if __name__ == "__main__":
    run_benchmark()
