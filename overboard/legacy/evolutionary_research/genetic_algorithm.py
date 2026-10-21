"""
Experimental Genetic Algorithm & Segment Recombination Trajectory Optimization
==============================================================================
Archived from surrogate_polish.py to isolate exploratory stochastic evolutionary
research from deterministic production solvers (superseded by analytical frontier seeds).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from reachability_engine import (
    CompiledReachabilityEngine,
    JAX_COMPILE_LOCK,
    _extract_state_and_dt_arrays,
)
from robot_model import RobotConfig, diff_drive_fk, diff_drive_ik, wrap_angle
from optimizer import TrajectoryOptimizer
from surrogate_polish import solve_pyramid_candidates


# ==============================================================================
# SEGMENT-WISE RECOMBINATION CROSSOVER ("CHIMERA SPLICING")
# ==============================================================================

def recombine_candidate_segments(
    candidates_data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    waypoints: List[Tuple[float, float, Optional[float]]],
    robot_cfg: RobotConfig,
    engine: CompiledReachabilityEngine,
    stop_waypoint_indices: Optional[List[int]] = None,
    accuracy_weight: float = 0.0,
    force_weight: float = 0.0,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    max_polish_iters: int = 250,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Advanced Optimization: Segment-Wise Recombination Splicer.
    Evaluates candidate solutions segment-by-segment, extracts the best local
    segment from each candidate, blends transition knots, and rapidly warm-polishes
    the recombined Chimera trajectory.
    """
    valid_cands = [
        (samples, stats) for samples, stats in candidates_data
        if stats.get("converged", False) and samples
    ]
    if len(valid_cands) < 2:
        return None, None, {"recombined": False, "reason": "Fewer than 2 valid converged candidates"}

    num_segments = len(waypoints) - 1
    sample_0, stats_0 = valid_cands[0]
    segment_samples = stats_0.get("segment_samples")
    if not segment_samples or len(segment_samples) != num_segments:
        opt = TrajectoryOptimizer(robot_cfg)
        segment_samples = opt.get_segment_sample_counts(
            waypoints, len(sample_0) // max(1, num_segments), proportional=True
        )
    segment_samples = [int(s) for s in segment_samples]
    N = sum(segment_samples) + 1
    wp_indices = [0] + list(np.cumsum(segment_samples))
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    stops = set(stop_waypoint_indices or [])

    # Evaluate segment-by-segment score for each candidate
    candidate_states_list = []
    candidate_dts_list = []
    candidate_segment_scores = []  # shape: [num_candidates, num_segments]

    for c_idx, (samples, stats) in enumerate(valid_cands):
        states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
        candidate_states_list.append(states)
        candidate_dts_list.append(dts)

        with JAX_COMPILE_LOCK:
            rep = engine.evaluate_states(states, dts, sample_to_segment)
        track_errs = np.asarray(rep["tracking_errors"])
        radii = np.asarray(rep["radii"])

        seg_scores = []
        for s in range(num_segments):
            k_start = wp_indices[s]
            k_end = wp_indices[s + 1]
            seg_dt = float(dts[s])
            seg_time = seg_dt * segment_samples[s]
            seg_track_err = float(np.max(track_errs[k_start:k_end]))
            seg_tube = float(np.max(radii[k_start:k_end]))

            score = (
                seg_time
                + 120.0 * max(0.0, seg_track_err - 0.005)
                + 60.0 * max(0.0, seg_tube - 0.020)
            )
            seg_scores.append(score)
        candidate_segment_scores.append(seg_scores)

    # Determine segment winners
    winners_per_segment = []
    for s in range(num_segments):
        best_cand = min(range(len(valid_cands)), key=lambda c: candidate_segment_scores[c][s])
        winners_per_segment.append(best_cand)

    unique_winners = set(winners_per_segment)
    if len(unique_winners) <= 1:
        # One candidate dominated all segments, no chimera splicing needed
        dominant_cand = list(unique_winners)[0]
        return (
            valid_cands[dominant_cand][0],
            valid_cands[dominant_cand][1],
            {"recombined": False, "reason": "Single candidate won all segments", "dominant_candidate": dominant_cand},
        )

    # Build Chimera States and DTs
    chimera_states = np.zeros((N, 5), dtype=np.float64)
    chimera_dts = np.zeros(num_segments, dtype=np.float64)

    for s in range(num_segments):
        k_win = winners_per_segment[s]
        win_states = candidate_states_list[k_win]
        win_dts = candidate_dts_list[k_win]
        k_start = wp_indices[s]
        k_end = wp_indices[s + 1]

        chimera_dts[s] = win_dts[s]
        chimera_states[k_start : k_end + 1] = win_states[k_start : k_end + 1]

    # Knot Alignment & Transition Smoothing at Waypoints
    for s in range(1, num_segments):
        idx = wp_indices[s]
        if s in stops:
            chimera_states[idx, 3:5] = 0.0
        else:
            k_prev = winners_per_segment[s - 1]
            k_next = winners_per_segment[s]
            if k_prev != k_next:
                prev_v = candidate_states_list[k_prev][idx, 3:5]
                next_v = candidate_states_list[k_next][idx, 3:5]
                chimera_states[idx, 3:5] = 0.5 * (prev_v + next_v)

    # Rapid Warm-Start IPOPT Polish
    chimera_guess = np.concatenate([chimera_dts, chimera_states.flatten()])
    pol_kw = dict(solve_kwargs or {})
    pol_kw["initial_guess"] = chimera_guess
    pol_kw["num_samples_per_segment"] = segment_samples
    pol_kw["per_segment_dt"] = True
    pol_kw["cost_metric"] = "discretization"
    pol_kw["epigraph_constraints"] = True
    pol_kw["accuracy_weight"] = accuracy_weight
    pol_kw["force_weight"] = force_weight
    pol_kw["verbose"] = False
    pol_kw["self_consistent"] = False
    pol_kw["solver_options"] = {"max_iter": max_polish_iters}

    pol_opt = TrajectoryOptimizer(robot_cfg)
    try:
        pol_samples, pol_stats = pol_opt.solve(waypoints, **pol_kw)
    except Exception as exc:
        return None, None, {"recombined": False, "reason": f"Polish solve failed: {exc}"}

    if not pol_stats.get("converged", False) or not pol_samples:
        return None, None, {"recombined": False, "reason": "Polish solve did not converge"}

    c_states, c_dts = _extract_state_and_dt_arrays(pol_samples, num_segments, segment_samples)
    with JAX_COMPILE_LOCK:
        rep = engine.evaluate_states(c_states, c_dts, sample_to_segment)

    recomb_info = {
        "recombined": True,
        "segment_winners": winners_per_segment,
        "unique_contributors": sorted(list(unique_winners)),
        "polish_iters": pol_stats.get("iter_count", 0),
        "tracking_m": float(rep["max_tracking_error_m"]),
        "envelope_m": float(rep["max_envelope_radius_m"]),
        "duration_s": float(pol_samples[-1]["t"]) if pol_samples else 0.0,
    }
    pol_stats["chimera_recombination"] = recomb_info
    return pol_samples, pol_stats, recomb_info


# ==============================================================================
# MEMETIC EVOLUTION MODULE (GENETIC SPLICING + CREEP MUTATION + LOCAL SEARCH)
# ==============================================================================

def _compute_hierarchical_weights(num_survivors: int, power: float = 1.6) -> np.ndarray:
    """
    Computes monotonically increasing element weights across survivors:
    - 4th-worst (index num_survivors - 1): small element (~0.05 - 0.08)
    - 5th best (index 4): bigger element
    - 4th best (index 3): even bigger element
    - 3rd, 2nd best: increasingly larger elements
    - Best (index 0): 100% baseline weight (1.0)
    """
    ranks_from_bottom = np.arange(1, num_survivors + 1, dtype=np.float64)
    weights = (ranks_from_bottom / float(num_survivors)) ** power
    return weights[::-1]


def _compute_shared_fitness(
    population: List[Dict[str, Any]],
    d_niche: float = 0.08,
    sharing_power: float = 0.5,
) -> np.ndarray:
    """
    Computes niche-discounted shared fitness to prevent premature diversity collapse.
    Individuals clustered in the same spatial basin receive a density penalty,
    preserving structurally distinct alternative homotopies in the surviving pool.
    """
    n = len(population)
    if n <= 1:
        return np.array([p["score"] for p in population])

    positions = np.array([p["states"][:, 0:2] for p in population])  # [n, N, 2]
    diff = positions[:, None, :, :] - positions[None, :, :, :]        # [n, n, N, 2]
    dist_matrix = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)    # [n, n]

    sharing = np.maximum(0.0, 1.0 - dist_matrix / max(d_niche, 1e-4))
    m = np.sum(sharing, axis=1)  # niche count

    raw_scores = np.array([p["score"] for p in population])
    return raw_scores * (m ** sharing_power)


def run_memetic_evolution(
    candidates_data: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    waypoints: List[Tuple[float, float, Optional[float]]],
    robot_cfg: RobotConfig,
    engine: CompiledReachabilityEngine,
    num_generations: int = 3,
    population_size: int = 10,
    discard_worst_count: int = 3,
    stop_waypoint_indices: Optional[List[int]] = None,
    accuracy_weight: float = 0.0,
    force_weight: float = 0.0,
    solve_kwargs: Optional[Dict[str, Any]] = None,
    max_polish_iters: int = 150,
    max_workers: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Memetic Evolution Trajectory Optimizer:
    Iterative population-based genetic algorithm that models robot trajectories as
    discrete chromosomes of segment timings and spatial state sequences.

    Lifecycle per generation:
    1. Evaluate & Rank: Rank population by multi-objective fitness (duration + tracking error + envelope radius).
    2. Truncation Selection: Discard worst 30% candidates (lowest 3 scoring paths).
    3. Elitism: Keep #1 overall best path intact and unaltered (100% of the best).
    4. Hierarchical Recombination: Combine a small element from the 4th-worst, a bigger element
       from the 5th best, an even bigger element from the 4th best, etc., and 100% of the best,
       incorporating a calibrated degree of randomness.
    5. Creep Mutation: Apply Gaussian perturbation to segment dts and lateral path coordinates.
    6. Memetic Local Search: Fast warm-start IPOPT polish on newly bred offspring in parallel.
    """
    valid_cands = [
        (samples, stats) for samples, stats in candidates_data
        if stats.get("converged", False) and samples
    ]
    if not valid_cands:
        return None, None, {"converged": False, "reason": "No valid converged candidates to seed evolution"}

    rng = np.random.default_rng(seed)
    num_segments = len(waypoints) - 1
    sample_0, stats_0 = valid_cands[0]
    segment_samples = stats_0.get("segment_samples")
    if not segment_samples or len(segment_samples) != num_segments:
        opt = TrajectoryOptimizer(robot_cfg)
        segment_samples = opt.get_segment_sample_counts(
            waypoints, len(sample_0) // max(1, num_segments), proportional=True
        )
    segment_samples = [int(s) for s in segment_samples]
    N = sum(segment_samples) + 1
    wp_indices = [0] + list(np.cumsum(segment_samples))
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))
    stops = set(stop_waypoint_indices or [])

    def _eval_individual(samples, stats, tag="seed", gen=0):
        states, dts = _extract_state_and_dt_arrays(samples, num_segments, segment_samples)
        with JAX_COMPILE_LOCK:
            rep = engine.evaluate_states(states, dts, sample_to_segment)
        track_errs = np.asarray(rep["tracking_errors"])
        radii = np.asarray(rep["radii"])

        seg_scores = []
        for s in range(num_segments):
            k_start = wp_indices[s]
            k_end = wp_indices[s + 1]
            seg_dt = float(dts[s])
            seg_time = seg_dt * segment_samples[s]
            seg_track = float(np.max(track_errs[k_start:k_end]))
            seg_tube = float(np.max(radii[k_start:k_end]))
            seg_score = (
                seg_time
                + 120.0 * max(0.0, seg_track - 0.005)
                + 60.0 * max(0.0, seg_tube - 0.020)
            )
            seg_scores.append(seg_score)

        duration = float(samples[-1]["t"]) if samples else float(np.sum(dts * np.array(segment_samples)))
        max_track = float(rep["max_tracking_error_m"])
        max_tube = float(rep["max_envelope_radius_m"])
        total_score = duration + 120.0 * max(0.0, max_track - 0.005) + 60.0 * max(0.0, max_tube - 0.020)

        return {
            "samples": samples,
            "stats": stats,
            "states": np.asarray(states),
            "dts": np.asarray(dts),
            "score": total_score,
            "duration": duration,
            "tracking_m": max_track,
            "envelope_m": max_tube,
            "segment_scores": seg_scores,
            "tag": tag,
            "generation": gen,
        }

    # Initialize population from valid candidates
    population: List[Dict[str, Any]] = []
    for idx, (samples, stats) in enumerate(valid_cands):
        pop_ind = _eval_individual(samples, stats, tag=f"seed_{idx+1}", gen=0)
        population.append(pop_ind)

    # If population is smaller than target population_size, fill with mutated seeds
    while len(population) < population_size:
        parent = population[len(population) % len(valid_cands)]
        mut_dts = parent["dts"] * (1.0 + rng.normal(0.0, 0.05, size=num_segments))
        mut_dts = np.clip(mut_dts, 0.005, 0.25)
        mut_states = np.copy(parent["states"])
        # Subtle Gaussian lateral jitter on intermediate states
        for s in range(num_segments):
            k_start = wp_indices[s]
            k_end = wp_indices[s + 1]
            if k_end - k_start > 2:
                noise = rng.normal(0.0, 0.01, size=(k_end - k_start - 1, 2))
                mut_states[k_start + 1 : k_end, 0:2] += noise
        mut_guess = np.concatenate([mut_dts, mut_states.flatten()])

        pol_kw = dict(solve_kwargs or {})
        pol_kw["initial_guess"] = mut_guess
        pol_kw["num_samples_per_segment"] = segment_samples
        pol_kw["per_segment_dt"] = True
        pol_kw["cost_metric"] = "discretization"
        pol_kw["epigraph_constraints"] = True
        pol_kw["accuracy_weight"] = accuracy_weight
        pol_kw["force_weight"] = force_weight
        pol_kw["verbose"] = False
        pol_kw["self_consistent"] = False
        pol_kw["solver_options"] = {"max_iter": max_polish_iters}

        opt_mut = TrajectoryOptimizer(robot_cfg)
        try:
            m_samples, m_stats = opt_mut.solve(waypoints, **pol_kw)
            if m_stats.get("converged", False) and m_samples:
                population.append(_eval_individual(m_samples, m_stats, tag=f"seed_mut_{len(population)}", gen=0))
            else:
                population.append(dict(parent))
        except Exception:
            population.append(dict(parent))

    # Clamp population to population_size
    population = population[:population_size]

    history: List[Dict[str, Any]] = []
    snapshots: Dict[int, List[Dict[str, Any]]] = {}

    # Sort initial population
    population.sort(key=lambda ind: ind["score"])
    snapshots[0] = population[0]["samples"]
    init_best = population[0]
    init_mean_score = float(np.mean([ind["score"] for ind in population]))
    history.append({
        "generation": 0,
        "best_duration": init_best["duration"],
        "best_score": init_best["score"],
        "mean_score": init_mean_score,
        "best_tracking_m": init_best["tracking_m"],
        "best_envelope_m": init_best["envelope_m"],
        "best_tag": init_best["tag"],
    })

    if verbose:
        print(f"  [Gen 0] Leader: {init_best['duration']:.3f} s (score {init_best['score']:.3f}, "
              f"track {init_best['tracking_m']*100:.2f} cm, tube {init_best['envelope_m']*100:.2f} cm) | "
              f"Pop mean score: {init_mean_score:.3f}")

    # Evolutionary Loop with Persistent ProcessPoolExecutor
    num_w = max(1, min(discard_worst_count, max_workers or discard_worst_count, os.cpu_count() or 8))
    persistent_pool = None
    if num_w > 1:
        try:
            persistent_pool = ProcessPoolExecutor(max_workers=num_w)
        except Exception:
            persistent_pool = None

    track_w = float(robot_cfg.track_width)
    v_max = float(robot_cfg.max_linear_speed())

    try:
        for gen in range(1, num_generations + 1):
            # 1. Truncation Selection with Spatial Niching (Diversity Preservation)
            k_discard = min(discard_worst_count, len(population) - 2)
            shared_costs = _compute_shared_fitness(population, d_niche=0.08, sharing_power=0.5)

            # Elite is strictly the individual with lowest raw objective score (100% preserved)
            elite_idx = int(np.argmin([p["score"] for p in population]))
            elite = population[elite_idx]

            # Rank the remaining candidates by shared fitness to penalize redundant clones
            other_indices = [i for i in range(len(population)) if i != elite_idx]
            other_indices.sort(key=lambda i: shared_costs[i])

            num_surv_needed = len(population) - k_discard - 1
            survivor_indices = [elite_idx] + other_indices[:num_surv_needed]
            survivors = [population[i] for i in survivor_indices]

            num_surv = len(survivors)
            hier_weights = _compute_hierarchical_weights(num_surv, power=1.6)

            offspring_guesses = []
            offspring_tags = []
            offspring_kws = []

            base_kw = dict(solve_kwargs or {})
            base_kw["num_samples_per_segment"] = segment_samples
            base_kw["per_segment_dt"] = True
            base_kw["cost_metric"] = "discretization"
            base_kw["epigraph_constraints"] = True
            base_kw["accuracy_weight"] = accuracy_weight
            base_kw["force_weight"] = force_weight
            base_kw["verbose"] = False
            base_kw["self_consistent"] = False
            base_kw["solver_options"] = {"max_iter": max_polish_iters}

            # 2. Hierarchical Recombination Breeding
            for off_idx in range(k_discard):
                sigma_rand = 0.20 + 0.15 * off_idx
                off_dts = np.zeros(num_segments, dtype=np.float64)
                off_states = np.zeros((N, 5), dtype=np.float64)
                donor_trail = []

                for s in range(num_segments):
                    seg_perf = np.array([p["segment_scores"][s] for p in survivors], dtype=np.float64)
                    min_p = float(np.min(seg_perf))
                    seg_gap = (seg_perf - min_p) / (min_p + 1e-4)
                    rand_jitter = rng.normal(0.0, sigma_rand, size=num_surv)

                    log_priorities = np.log(np.maximum(hier_weights, 1e-6)) - 1.5 * seg_gap + rand_jitter
                    max_log = np.max(log_priorities)
                    probs = np.exp(log_priorities - max_log)
                    probs = probs / np.sum(probs)

                    chosen_r = int(rng.choice(num_surv, p=probs))
                    donor_trail.append(chosen_r)
                    donor = survivors[chosen_r]

                    k_start = wp_indices[s]
                    k_end = wp_indices[s + 1]

                    seg_states_donor = np.copy(donor["states"][k_start : k_end + 1])
                    seg_dts_donor = float(donor["dts"][s])

                    v_scale = 1.0 + rng.normal(0.0, 0.02 * sigma_rand)
                    kappa_jitter = rng.normal(0.0, 0.04 * sigma_rand)
                    v_lin = diff_drive_fk(seg_states_donor[:, 3], seg_states_donor[:, 4], track_w)[0] * v_scale
                    vl_jit, vr_jit = diff_drive_ik(v_lin, v_lin * kappa_jitter, track_w)
                    seg_states_donor[:, 3] = np.clip(vl_jit, -v_max, v_max)
                    seg_states_donor[:, 4] = np.clip(vr_jit, -v_max, v_max)

                    timing_creep = 1.0 + rng.normal(0.0, 0.02 * sigma_rand)
                    off_dts[s] = np.clip(seg_dts_donor * timing_creep, 0.005, 0.25)
                    off_states[k_start : k_end + 1] = seg_states_donor

                if off_idx >= 1:
                    mut_segs = rng.choice(num_segments, size=min(2, num_segments), replace=False)
                    for ms in mut_segs:
                        off_dts[ms] = np.clip(off_dts[ms] * rng.uniform(0.80, 1.25), 0.005, 0.25)

                    disp_seg = int(rng.integers(0, num_segments))
                    ks = wp_indices[disp_seg]
                    ke = wp_indices[disp_seg + 1]
                    if ke - ks > 2:
                        p1 = off_states[ks, 0:2]
                        p2 = off_states[ke, 0:2]
                        chord = p2 - p1
                        chord_len = float(np.linalg.norm(chord))
                        if chord_len > 0.08:
                            norm_vec = np.array([-chord[1], chord[0]]) / chord_len
                            shift_mag = float(rng.uniform(-0.06, 0.06))
                            tau = np.linspace(0.0, 1.0, ke - ks + 1)[:, None]
                            bump = 4.0 * tau * (1.0 - tau)
                            off_states[ks : ke + 1, 0:2] += bump * shift_mag * norm_vec

                # SE(2) Continuous Phase Unwrapping across spliced knot boundaries
                for s in range(1, num_segments):
                    idx = wp_indices[s]
                    dth = off_states[idx, 2] - off_states[idx - 1, 2]
                    wrapped_dth = wrap_angle(dth)
                    off_states[idx:, 2] += (wrapped_dth - dth)
                    if s in stops:
                        off_states[idx, 3:5] = 0.0
                    else:
                        off_states[idx, 3:5] = 0.5 * (off_states[idx - 1, 3:5] + off_states[min(idx + 1, N - 1), 3:5])

                # Layer 4 Directed Memetic Mutation (Gradient Guidance on Exploratory Offspring)
                if gen > 1 and off_idx == 0:
                    try:
                        grad_fn = engine._get_grad_fn(safety_gate_m=0.005, tube_gate_m=0.020)
                        seg_idx_arr = jnp.asarray(sample_to_segment, dtype=jnp.int32)
                        active_mask = jnp.ones(off_states.shape[0] - 1, dtype=bool)
                        w_lower, w_upper = engine._uncertainty_bounds()
                        with JAX_COMPILE_LOCK:
                            (_, _), (g_states, g_dts) = grad_fn(
                                jnp.asarray(off_states), jnp.asarray(off_dts), seg_idx_arr, active_mask, w_lower, w_upper
                            )
                        g_s = np.asarray(g_states)
                        g_d = np.asarray(g_dts)
                        if np.all(np.isfinite(g_s)) and np.all(np.isfinite(g_d)):
                            off_states[:, 0:2] -= np.clip(0.004 * g_s[:, 0:2], -0.010, 0.010)
                            off_dts = np.clip(off_dts - np.clip(0.004 * g_d, -0.010, 0.010), 0.005, 0.25)
                    except Exception:
                        pass

                offspring_guesses.append(np.concatenate([off_dts, off_states.flatten()]))
                offspring_tags.append(f"hier_mix_g{gen}_off{off_idx+1}")

                max_donor_track = max(survivors[r]["tracking_m"] for r in donor_trail)
                excess_err = max(0.0, max_donor_track - 0.005)
                adaptive_acc = max(accuracy_weight, float(np.clip(0.002 + 0.06 * (excess_err / 0.02), 0.0, 0.012)))

                off_kw = dict(base_kw)
                off_kw["accuracy_weight"] = adaptive_acc
                offspring_kws.append(off_kw)

            # 3. Memetic Local Search: Fast Parallel IPOPT Polish on Offspring via Persistent Pool
            max_w = min(len(offspring_guesses), max_workers or len(offspring_guesses))
            polished_results = solve_pyramid_candidates(
                robot_cfg, waypoints, offspring_guesses, offspring_kws, max_workers=max_w, pool=persistent_pool
            )

            new_offspring_inds = []
            for o_idx, (o_samples, o_stats) in enumerate(polished_results):
                tag = offspring_tags[o_idx]
                if o_stats.get("converged", False) and o_samples:
                    ind = _eval_individual(o_samples, o_stats, tag=tag, gen=gen)
                    new_offspring_inds.append(ind)
                else:
                    g = offspring_guesses[o_idx]
                    dts_guess = g[:num_segments]
                    states_guess = g[num_segments:].reshape((N, 5))
                    mock_samples = []
                    cur_t = 0.0
                    for i in range(N):
                        s_idx = sample_to_segment[min(i, len(sample_to_segment) - 1)]
                        mock_samples.append({
                            "t": cur_t,
                            "x": float(states_guess[i, 0]),
                            "y": float(states_guess[i, 1]),
                            "heading": float(states_guess[i, 2]),
                            "vl": float(states_guess[i, 3]),
                            "vr": float(states_guess[i, 4]),
                        })
                        cur_t += float(dts_guess[s_idx])
                    mock_stats = {"converged": False, "segment_dt": list(dts_guess)}
                    ind = _eval_individual(mock_samples, mock_stats, tag=f"{tag}_unpolished", gen=gen)
                    ind["score"] += 1.0
                    new_offspring_inds.append(ind)

            # 4. Integrate survivors + new offspring into population
            population = survivors + new_offspring_inds
            population.sort(key=lambda ind: ind["score"])

            current_best = population[0]
            current_mean = float(np.mean([ind["score"] for ind in population]))
            snapshots[gen] = current_best["samples"]
            history.append({
                "generation": gen,
                "best_duration": current_best["duration"],
                "best_score": current_best["score"],
                "mean_score": current_mean,
                "best_tracking_m": current_best["tracking_m"],
                "best_envelope_m": current_best["envelope_m"],
                "best_tag": current_best["tag"],
            })

            if verbose:
                print(f"  [Gen {gen}] Leader: {current_best['duration']:.3f} s (score {current_best['score']:.3f}, "
                      f"track {current_best['tracking_m']*100:.2f} cm, tube {current_best['envelope_m']*100:.2f} cm, "
                      f"origin '{current_best['tag']}') | Pop mean: {current_mean:.3f}")
    finally:
        if persistent_pool is not None:
            try:
                persistent_pool.shutdown(wait=False)
            except Exception:
                pass

    champion = population[0]
    champ_stats = dict(champion["stats"])
    champ_stats["converged"] = True
    champ_stats["segment_dt"] = list(champion["dts"])

    evolution_report = {
        "generations": history,
        "snapshots": snapshots,
        "population_leaderboard": [
            {
                "rank": i + 1,
                "tag": ind["tag"],
                "duration_s": ind["duration"],
                "score": ind["score"],
                "tracking_cm": ind["tracking_m"] * 100.0,
                "envelope_cm": ind["envelope_m"] * 100.0,
                "generation": ind["generation"],
            }
            for i, ind in enumerate(population)
        ],
        "champion_duration": champion["duration"],
        "champion_score": champion["score"],
        "champion_tag": champion["tag"],
    }
    champ_stats["memetic_evolution"] = evolution_report

    return champion["samples"], champ_stats, evolution_report
