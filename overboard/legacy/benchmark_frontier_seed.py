"""
Benchmark Kinetic Frontier Seeds against Stock & Evolved Champions
==================================================================
Runs on examples/example_complex_mission.json and evaluates:
1. Stock Baseline
2. Stock Rolling
3. Kinetic Frontier Racing Line (Analytical)
4. Kinetic Frontier Direct Chord (Analytical)
"""

import json
import time
import numpy as np

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from immrax_validator import ImmraxValidator
from optimizer import TrajectoryOptimizer
from surrogate_polish import CompiledReachabilityEngine, _extract_state_and_dt_arrays
from main import _load_waypoints
from frontier_seed import generate_frontier_seed, generate_canonical_seeds


def run_benchmark():
    # 1. Setup
    with open("examples/fll_choreo.json") as f:
        cfg_data = json.load(f)
    robot_cfg = RobotConfig(cfg_data)
    jax_cfg = JAXRobotConfig(robot_cfg)
    validator = ImmraxValidator(jax_cfg, control_dt=0.02, propagation="plant")
    engine = CompiledReachabilityEngine(validator)

    wps, events, thresholds, all_stop_indices = _load_waypoints("examples/example_complex_mission.json")
    num_segments = len(wps) - 1

    opt = TrajectoryOptimizer(robot_cfg)
    segment_samples = opt.get_adaptive_mesh_sample_counts(wps, apex_samples=8, straight_samples=4)
    segment_samples = [int(s) for s in segment_samples]
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    N = sum(segment_samples) + 1

    print("=" * 80)
    print(f"BENCHMARK: 10-Waypoint Complex Mission ({N} samples, {num_segments} segments)")
    print("=" * 80)

    # Base solve settings matching surrogate_polish discretization mode
    solve_kwargs = {
        "num_samples_per_segment": segment_samples,
        "per_segment_dt": True,
        "cost_metric": "discretization",
        "epigraph_constraints": True,
        "accuracy_weight": 0.0,
        "force_weight": 0.0,
        "verbose": False,
        "self_consistent": True,
        "solver_options": {"max_iter": 200},
    }

    results = []

    # --- Test 1: Stock Baseline ---
    print("\n1. Solving Stock Baseline...")
    guess_base = opt._build_initial_guess(wps, segment_samples, N, style="baseline", stop_waypoint_indices=all_stop_indices)
    t0 = time.time()
    s_base, st_base = opt.solve(wps, initial_guess=guess_base, **solve_kwargs)
    t_base = time.time() - t0
    states_b, dts_b = _extract_state_and_dt_arrays(s_base, num_segments, segment_samples)
    rep_b = engine.evaluate_states(states_b, dts_b, sample_to_segment)
    dur_b = float(s_base[-1]["t"])
    track_b = float(rep_b["max_tracking_error_m"]) * 100.0
    tube_b = float(rep_b["max_envelope_radius_m"]) * 100.0
    iters_b = st_base.get("iter_count", len(st_base.get("iterations", {}).get("obj", [])))
    score_b = dur_b + 120.0 * max(0.0, (track_b / 100.0) - 0.005) + 60.0 * max(0.0, (tube_b / 100.0) - 0.020)
    results.append(("Stock Baseline", dur_b, track_b, tube_b, iters_b, score_b, t_base))
    print(f"   -> Duration: {dur_b:.3f} s | Track: {track_b:.2f} cm | Tube: {tube_b:.2f} cm | Iters: {iters_b} | Score: {score_b:.3f} | Wall: {t_base:.2f} s")

    # --- Test 2: Stock Rolling ---
    print("\n2. Solving Stock Rolling...")
    guess_roll = opt._build_initial_guess(wps, segment_samples, N, style="rolling", stop_waypoint_indices=all_stop_indices)
    t0 = time.time()
    s_roll, st_roll = opt.solve(wps, initial_guess=guess_roll, **solve_kwargs)
    t_roll = time.time() - t0
    states_r, dts_r = _extract_state_and_dt_arrays(s_roll, num_segments, segment_samples)
    rep_r = engine.evaluate_states(states_r, dts_r, sample_to_segment)
    dur_r = float(s_roll[-1]["t"])
    track_r = float(rep_r["max_tracking_error_m"]) * 100.0
    tube_r = float(rep_r["max_envelope_radius_m"]) * 100.0
    iters_r = st_roll.get("iter_count", len(st_roll.get("iterations", {}).get("obj", [])))
    score_r = dur_r + 120.0 * max(0.0, (track_r / 100.0) - 0.005) + 60.0 * max(0.0, (tube_r / 100.0) - 0.020)
    results.append(("Stock Rolling", dur_r, track_r, tube_r, iters_r, score_r, t_roll))
    print(f"   -> Duration: {dur_r:.3f} s | Track: {track_r:.2f} cm | Tube: {tube_r:.2f} cm | Iters: {iters_r} | Score: {score_r:.3f} | Wall: {t_roll:.2f} s")

    # --- Test 3: Kinetic Frontier - Racing Line ---
    print("\n3. Solving Kinetic Frontier (Racing Line)...")
    t_gen0 = time.time()
    guess_frontier_racing = generate_frontier_seed(
        wps, segment_samples, robot_cfg, stop_waypoint_indices=all_stop_indices, style="racing_line"
    )
    t_gen_racing = (time.time() - t_gen0) * 1000.0
    print(f"   [Analytical Seed Generated in {t_gen_racing:.2f} ms]")

    t0 = time.time()
    s_fracing, st_fracing = opt.solve(wps, initial_guess=guess_frontier_racing, **solve_kwargs)
    t_fracing = time.time() - t0
    states_fr, dts_fr = _extract_state_and_dt_arrays(s_fracing, num_segments, segment_samples)
    rep_fr = engine.evaluate_states(states_fr, dts_fr, sample_to_segment)
    dur_fr = float(s_fracing[-1]["t"])
    track_fr = float(rep_fr["max_tracking_error_m"]) * 100.0
    tube_fr = float(rep_fr["max_envelope_radius_m"]) * 100.0
    iters_fr = st_fracing.get("iter_count", len(st_fracing.get("iterations", {}).get("obj", [])))
    score_fr = dur_fr + 120.0 * max(0.0, (track_fr / 100.0) - 0.005) + 60.0 * max(0.0, (tube_fr / 100.0) - 0.020)
    results.append(("Frontier Racing Line", dur_fr, track_fr, tube_fr, iters_fr, score_fr, t_fracing))
    print(f"   -> Duration: {dur_fr:.3f} s | Track: {track_fr:.2f} cm | Tube: {tube_fr:.2f} cm | Iters: {iters_fr} | Score: {score_fr:.3f} | Wall: {t_fracing:.2f} s")

    # --- Test 4: Kinetic Frontier - Direct Chord ---
    print("\n4. Solving Kinetic Frontier (Direct Chord)...")
    t_gen0 = time.time()
    guess_frontier_direct = generate_frontier_seed(
        wps, segment_samples, robot_cfg, stop_waypoint_indices=all_stop_indices, style="direct_chord"
    )
    t_gen_direct = (time.time() - t_gen0) * 1000.0
    print(f"   [Analytical Seed Generated in {t_gen_direct:.2f} ms]")

    t0 = time.time()
    s_fdirect, st_fdirect = opt.solve(wps, initial_guess=guess_frontier_direct, **solve_kwargs)
    t_fdirect = time.time() - t0
    states_fd, dts_fd = _extract_state_and_dt_arrays(s_fdirect, num_segments, segment_samples)
    rep_fd = engine.evaluate_states(states_fd, dts_fd, sample_to_segment)
    dur_fd = float(s_fdirect[-1]["t"])
    track_fd = float(rep_fd["max_tracking_error_m"]) * 100.0
    tube_fd = float(rep_fd["max_envelope_radius_m"]) * 100.0
    iters_fd = st_fdirect.get("iter_count", len(st_fdirect.get("iterations", {}).get("obj", [])))
    score_fd = dur_fd + 120.0 * max(0.0, (track_fd / 100.0) - 0.005) + 60.0 * max(0.0, (tube_fd / 100.0) - 0.020)
    results.append(("Frontier Direct Chord", dur_fd, track_fd, tube_fd, iters_fd, score_fd, t_fdirect))
    print(f"   -> Duration: {dur_fd:.3f} s | Track: {track_fd:.2f} cm | Tube: {tube_fd:.2f} cm | Iters: {iters_fd} | Score: {score_fd:.3f} | Wall: {t_fdirect:.2f} s")

    # Print Final Comparison Table
    print("\n" + "=" * 95)
    print(f"{'Method / Candidate':<25} | {'Duration (s)':<12} | {'Track Err':<10} | {'Tube':<8} | {'Iters':<6} | {'Score':<8} | {'Wall (s)'}")
    print("-" * 95)
    for name, dur, trk, tub, it, sc, wt in results:
        gate_str = "PASS" if trk <= 0.50 and tub <= 2.0 else "FAIL"
        print(f"{name:<25} | {dur:<12.3f} | {trk:<7.2f} cm | {tub:<5.2f} cm | {it:<6} | {sc:<8.3f} | {wt:.2f} s ({gate_str})")
    print("=" * 95)
    print("Historical Reference:")
    print("44-Gene GA Champion (12 Gens): 5.417 s | 0.49 cm | 120 iters | Score 5.417 | Wall ~25 s")
    print("=" * 95)


if __name__ == "__main__":
    run_benchmark()
