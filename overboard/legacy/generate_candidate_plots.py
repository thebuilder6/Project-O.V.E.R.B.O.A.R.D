"""
Generates and keeps all 3 candidate trajectories, computes scores,
and outputs a comprehensive set of plots.
"""

import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_model import RobotConfig
from optimizer import TrajectoryOptimizer
from main import _load_waypoints
from jax_robot_model import JAXRobotConfig
from immrax_validator import ImmraxValidator
from surrogate_polish import (
    CompiledReachabilityEngine,
    generate_homotopy_bank,
    screen_candidates_jax,
    select_topological_representatives,
    solve_pyramid_candidates,
    _extract_state_and_dt_arrays,
)
from validator import compute_metrics, validate_samples


def main():
    os.makedirs("output_candidates", exist_ok=True)
    mission_path = "examples/example_complex_mission.json"
    config_path = "examples/fll_choreo.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    robot_cfg = RobotConfig(config_data)

    wps, waypoint_events, waypoint_thresholds, json_stop_indices = _load_waypoints(mission_path)
    all_stop_indices = sorted(set(json_stop_indices))

    opt = TrajectoryOptimizer(robot_cfg)
    num_samples = 10
    segment_samples = opt.get_segment_sample_counts(wps, num_samples, proportional=True)
    segment_samples = [int(s) for s in segment_samples]
    num_segments = len(wps) - 1
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))

    print("Stage 1: Generating homotopy bank...")
    cand_states, cand_dts, metadata = generate_homotopy_bank(
        wps,
        segment_samples,
        track_width=float(robot_cfg.track_width),
        v_max=float(robot_cfg.max_linear_speed()),
        stop_waypoint_indices=all_stop_indices,
    )
    print(f"  Generated {len(cand_states)} hypotheses.")

    print("Stage 2: Screening via compiled JAX sieve...")
    jax_cfg = JAXRobotConfig(robot_cfg)
    validator = ImmraxValidator(jax_cfg)
    engine = CompiledReachabilityEngine(validator)
    surrogate_costs = screen_candidates_jax(
        engine,
        cand_states,
        cand_dts,
        sample_to_segment,
    )

    print("Stage 3: Topological clustering into Top 3 winners...")
    selected_indices = select_topological_representatives(
        cand_states,
        surrogate_costs,
        k_clusters=3,
        min_separation_m=0.08,
    )
    print(f"  Selected candidate indices: {selected_indices}")

    survivor_guesses = [
        np.concatenate([cand_dts[idx], cand_states[idx].flatten()])
        for idx in selected_indices
    ]

    print("Stage 4: Solving 3 candidates in parallel via ProcessPoolExecutor...")
    t0 = time.time()
    solve_kwargs = {
        "num_samples_per_segment": segment_samples,
        "accuracy_weight": 0.0,
        "stop_waypoint_indices": all_stop_indices,
        "waypoint_events": waypoint_events,
        "verbose": False,
        "capture_iterations": False,
        "per_segment_dt": True,
        "self_consistent": True,
    }
    solved_pairs = solve_pyramid_candidates(
        robot_cfg, wps, survivor_guesses, solve_kwargs, max_workers=3
    )
    solve_elapsed = time.time() - t0
    print(f"  Solved all 3 candidates in {solve_elapsed:.2f} s.")

    # Process and evaluate each candidate
    candidate_records = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, ((samples_data, stats), idx) in enumerate(zip(solved_pairs, selected_indices)):
        c_name = f"candidate_{i + 1}"
        traj_file = os.path.join("output_candidates", f"{c_name}.traj")
        
        # Save .traj file
        result = {
            "name": c_name,
            "version": 3,
            "trajectory": {
                "config": config_data.get("robot", config_data.get("config", config_data)),
                "samples": samples_data,
            },
        }
        with open(traj_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)

        # Evaluate metrics & Immrax
        metrics = compute_metrics(samples_data, robot_cfg)
        cpu_audit = validate_samples(samples_data, robot_cfg)

        c_states, c_dts = _extract_state_and_dt_arrays(samples_data, num_segments, segment_samples)
        reach_out = engine.evaluate_states(c_states, c_dts, sample_to_segment)
        tracking_err_cm = float(reach_out["max_tracking_error_m"]) * 100.0
        envelope_cm = float(reach_out["max_envelope_radius_m"]) * 100.0
        duration_s = float(samples_data[-1]["t"]) if samples_data else 0.0
        path_length_m = float(metrics.get("path_length_m", 0.0))

        # Fitness score formulation matching main.py
        fitness = (
            float(duration_s)
            + 0.5 * (envelope_cm / 2.0)
            + 1.0 * (tracking_err_cm / 0.5)
            + 0.1 * float(metrics.get("rms_wheel_force_n", 0.0))
        )

        rec = {
            "name": c_name,
            "cluster_index": int(idx),
            "duration_s": round(duration_s, 3),
            "path_length_m": round(path_length_m, 3),
            "peak_wheel_speed_m_s": round(float(metrics.get("max_wheel_speed_m_s", 0.0)), 3),
            "avg_linear_speed_m_s": round(float(metrics.get("avg_linear_speed_m_s", 0.0)), 3),
            "max_accel_m_s2": round(float(metrics.get("max_accel_m_s2", 0.0)), 3),
            "max_jerk_m_s3": round(float(metrics.get("max_jerk_m_s3", 0.0)), 3),
            "rms_force_n": round(float(metrics.get("rms_wheel_force_n", 0.0)), 3),
            "tracking_error_cm": round(tracking_err_cm, 3),
            "envelope_radius_cm": round(envelope_cm, 3),
            "fitness_score": round(fitness, 3),
            "ipopt_iters": int(stats.get("iter_count", 0)),
            "cost": round(float(stats.get("final_cost", 0.0)), 4),
            "file": traj_file,
            "samples": samples_data,
            "reach_out": reach_out,
        }
        candidate_records.append(rec)
        print(f"  {c_name}: Duration={duration_s:.3f}s, Path={path_length_m:.3f}m, "
              f"Tracking={tracking_err_cm:.2f}cm, Envelope={envelope_cm:.2f}cm, Fitness={fitness:.3f}")

    # Save scores summary
    scores_file = os.path.join("output_candidates", "scores.json")
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump([
            {k: v for k, v in rec.items() if k not in ("samples", "reach_out")}
            for rec in candidate_records
        ], f, indent=2)
    print(f"Saved scores to {scores_file}")

    # =========================================================================
    # PLOT 1: 2D Spatial Paths Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    ax.set_facecolor("#fcfcfc")
    ax.grid(True, linestyle="--", alpha=0.5, color="#ccc")

    # Plot waypoints
    wp_x = [p[0] for p in wps]
    wp_y = [p[1] for p in wps]
    ax.plot(wp_x, wp_y, "k--", alpha=0.35, label="Waypoints Polyline", linewidth=1.5)
    for k, p in enumerate(wps):
        is_stop = k in all_stop_indices or k == 0 or k == len(wps) - 1
        color = "red" if is_stop else "#333333"
        ax.scatter(p[0], p[1], s=80 if is_stop else 50, color=color, zorder=5)
        ax.text(p[0] + 0.015, p[1] + 0.015, f"WP{k}{' (Stop)' if is_stop else ''}",
                fontsize=8, fontweight="bold", color=color, zorder=6)
        if p[2] is not None:
            dx_arr = 0.04 * np.cos(p[2])
            dy_arr = 0.04 * np.sin(p[2])
            ax.arrow(p[0], p[1], dx_arr, dy_arr, head_width=0.015, head_length=0.015,
                     fc=color, ec=color, zorder=5)

    # Plot all 3 candidates
    for i, rec in enumerate(candidate_records):
        xs = [s["x"] for s in rec["samples"]]
        ys = [s["y"] for s in rec["samples"]]
        lbl = f"{rec['name'].replace('_', ' ').title()} (T={rec['duration_s']}s, L={rec['path_length_m']}m, Score={rec['fitness_score']})"
        ax.plot(xs, ys, color=colors[i], linewidth=2.5, label=lbl, zorder=4)
        ax.scatter(xs, ys, color=colors[i], s=12, alpha=0.8, zorder=4)

    ax.set_title("2D Path Comparison: Multi-Fidelity Homotopy Candidates", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("X Position (meters)", fontsize=11)
    ax.set_ylabel("Y Position (meters)", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    p1_path = os.path.join("output_candidates", "plot_2d_trajectories.png")
    plt.savefig(p1_path)
    plt.close()
    print(f"Saved {p1_path}")

    # =========================================================================
    # PLOT 2: Velocity Profiles (Linear, Angular, Wheels)
    # =========================================================================
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), dpi=200, sharex=True)
    for i, rec in enumerate(candidate_records):
        ts = [s["t"] for s in rec["samples"]]
        vl = np.array([s["vl"] for s in rec["samples"]])
        vr = np.array([s["vr"] for s in rec["samples"]])
        v_lin = 0.5 * (vl + vr)
        w_rad = (vr - vl) / robot_cfg.track_width

        c_lbl = rec['name'].replace('_', ' ').title()
        axs[0].plot(ts, v_lin, color=colors[i], linewidth=2, label=c_lbl)
        axs[1].plot(ts, w_rad, color=colors[i], linewidth=2, label=c_lbl)
        axs[2].plot(ts, vl, color=colors[i], linestyle="-", linewidth=1.8, label=f"{c_lbl} VL")
        axs[2].plot(ts, vr, color=colors[i], linestyle="--", linewidth=1.8, label=f"{c_lbl} VR")

    axs[0].set_ylabel("Linear Speed (m/s)", fontsize=10, fontweight="bold")
    axs[0].axhline(robot_cfg.max_linear_speed(), color="black", linestyle=":", label="Speed Limit")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].set_title("Kinematic Trajectory Comparison: Velocity & Heading Rate", fontsize=12, fontweight="bold")

    axs[1].set_ylabel("Angular Vel (rad/s)", fontsize=10, fontweight="bold")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="upper right", fontsize=8)

    axs[2].set_ylabel("Wheel Speeds (m/s)", fontsize=10, fontweight="bold")
    axs[2].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend(loc="upper right", fontsize=7, ncol=3)

    plt.tight_layout()
    p2_path = os.path.join("output_candidates", "plot_velocity_profiles.png")
    plt.savefig(p2_path)
    plt.close()
    print(f"Saved {p2_path}")

    # =========================================================================
    # PLOT 3: Accelerations & Actuator Effort
    # =========================================================================
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), dpi=200, sharex=True)
    for i, rec in enumerate(candidate_records):
        ts = [s["t"] for s in rec["samples"]]
        al = np.array([s["al"] for s in rec["samples"]])
        ar = np.array([s["ar"] for s in rec["samples"]])
        a_lin = 0.5 * (al + ar)
        f_lin = robot_cfg.mass * np.abs(a_lin)
        c_lbl = rec['name'].replace('_', ' ').title()

        axs[0].plot(ts, a_lin, color=colors[i], linewidth=2, label=c_lbl)
        axs[1].plot(ts, f_lin, color=colors[i], linewidth=2, label=c_lbl)

    axs[0].set_ylabel("Linear Accel (m/s²)", fontsize=10, fontweight="bold")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].set_title("Actuator & Dynamic Loading Profiles", fontsize=12, fontweight="bold")

    axs[1].set_ylabel("Total Wheel Force (N)", fontsize=10, fontweight="bold")
    axs[1].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    p3_path = os.path.join("output_candidates", "plot_accelerations_forces.png")
    plt.savefig(p3_path)
    plt.close()
    print(f"Saved {p3_path}")

    # =========================================================================
    # PLOT 4: Tracking Error & Envelope Radius vs Sample
    # =========================================================================
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), dpi=200, sharex=True)
    for i, rec in enumerate(candidate_records):
        c_lbl = rec['name'].replace('_', ' ').title()
        track_errs = np.asarray(rec["reach_out"]["tracking_errors"]) * 100.0  # cm
        radii = np.asarray(rec["reach_out"]["radii"]) * 100.0  # cm
        intervals = np.arange(len(track_errs))

        axs[0].plot(intervals, track_errs, color=colors[i], linewidth=2, marker="o", markersize=4, label=c_lbl)
        axs[1].plot(intervals, radii, color=colors[i], linewidth=2, marker="s", markersize=4, label=c_lbl)

    axs[0].axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Safety Gate Threshold (0.50 cm)")
    axs[0].set_ylabel("Tracking Error (cm)", fontsize=10, fontweight="bold")
    axs[0].set_title("Immrax Closed-Loop Tracking & Reachability Envelope Profiles", fontsize=12, fontweight="bold")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper right", fontsize=8)

    axs[1].axhline(2.0, color="orange", linestyle="--", linewidth=1.5, label="Envelope Gate Threshold (2.0 cm)")
    axs[1].set_ylabel("Envelope Radius (cm)", fontsize=10, fontweight="bold")
    axs[1].set_xlabel("Sample Interval Index", fontsize=10, fontweight="bold")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    p4_path = os.path.join("output_candidates", "plot_tracking_reachability.png")
    plt.savefig(p4_path)
    plt.close()
    print(f"Saved {p4_path}")

    # =========================================================================
    # PLOT 5: Global Sieve Landscape (All 54 Hypotheses vs 3 Finalists)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    ax.set_facecolor("#f9f9fb")
    ax.grid(True, linestyle="--", alpha=0.4, color="#bbb")

    # Plot all screened hypotheses in faint grey/blue
    for k in range(len(cand_states)):
        ax.plot(cand_states[k, :, 0], cand_states[k, :, 1], color="#a0aec0", alpha=0.35, linewidth=1.0)

    # Highlight 3 selected survivor seeds
    for i, idx in enumerate(selected_indices):
        ax.plot(cand_states[idx, :, 0], cand_states[idx, :, 1], color=colors[i],
                linewidth=2.8, linestyle="--", label=f"Homotopy Winner {i+1} (Seed #{idx})")

    ax.scatter(wp_x, wp_y, s=70, color="black", zorder=6, label="Mission Waypoints")
    ax.set_title(f"Stage 1 & 2 Global Exploration: {len(cand_states)} Hypotheses → Top 3 Homotopy Winners",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("X Position (meters)", fontsize=11)
    ax.set_ylabel("Y Position (meters)", fontsize=11)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    p5_path = os.path.join("output_candidates", "plot_homotopy_bank_sieve.png")
    plt.savefig(p5_path)
    plt.close()
    print(f"Saved {p5_path}")

    print("\nAll 3 candidate trajectories, scores, and plots generated successfully!")


if __name__ == "__main__":
    main()
