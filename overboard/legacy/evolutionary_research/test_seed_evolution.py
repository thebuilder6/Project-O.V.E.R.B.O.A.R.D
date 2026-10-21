"""
Seed-Level / Starting-Value Evolutionary Algorithm
Evolving the parameters that generate IPOPT's initial guess.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from typing import Any, List, Dict, Tuple, Optional
import numpy as np

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from optimizer import TrajectoryOptimizer
from immrax_validator import ImmraxValidator
from surrogate_polish import CompiledReachabilityEngine, _extract_state_and_dt_arrays


class InitialGuessGenome:
    def __init__(
        self,
        dt_mults: np.ndarray,      # [S]
        apex_offsets: np.ndarray,  # [S]
        theta_biases: np.ndarray,  # [S-1 unpinned]
        speed_scales: np.ndarray,  # [S]
    ):
        self.dt_mults = np.array(dt_mults, dtype=float)
        self.apex_offsets = np.array(apex_offsets, dtype=float)
        self.theta_biases = np.array(theta_biases, dtype=float)
        self.speed_scales = np.array(speed_scales, dtype=float)

    @classmethod
    def create_default(cls, num_segments: int, num_unpinned: int):
        return cls(
            dt_mults=np.ones(num_segments),
            apex_offsets=np.zeros(num_segments),
            theta_biases=np.zeros(num_unpinned),
            speed_scales=np.full(num_segments, 0.6),
        )

    @classmethod
    def create_random(cls, num_segments: int, num_unpinned: int, rng=np.random):
        dt_mults = rng.uniform(0.7, 1.4, size=num_segments)
        apex_offsets = rng.uniform(-0.12, 0.12, size=num_segments)
        theta_biases = rng.uniform(-0.35, 0.35, size=num_unpinned) if num_unpinned > 0 else np.array([])
        speed_scales = rng.uniform(0.4, 0.9, size=num_segments)
        return cls(dt_mults, apex_offsets, theta_biases, speed_scales)

    def mutate(self, p_mut: float = 0.35, sigma: float = 0.15, rng=np.random):
        dt_m = self.dt_mults.copy()
        mask = rng.rand(len(dt_m)) < p_mut
        dt_m[mask] += rng.normal(0, sigma, size=np.sum(mask))
        dt_m = np.clip(dt_m, 0.4, 2.2)

        apex = self.apex_offsets.copy()
        mask = rng.rand(len(apex)) < p_mut
        apex[mask] += rng.normal(0, 0.04, size=np.sum(mask))
        apex = np.clip(apex, -0.20, 0.20)

        thetas = self.theta_biases.copy()
        if len(thetas) > 0:
            mask = rng.rand(len(thetas)) < p_mut
            thetas[mask] += rng.normal(0, 0.15, size=np.sum(mask))
            thetas = np.clip(thetas, -math.pi / 2, math.pi / 2)

        spd = self.speed_scales.copy()
        mask = rng.rand(len(spd)) < p_mut
        spd[mask] += rng.normal(0, sigma, size=np.sum(mask))
        spd = np.clip(spd, 0.1, 1.1)

        return InitialGuessGenome(dt_m, apex, thetas, spd)

    @classmethod
    def crossover(cls, parent_a: 'InitialGuessGenome', parent_b: 'InitialGuessGenome', rng=np.random):
        num_segments = len(parent_a.dt_mults)
        dt_m = np.zeros(num_segments)
        apex = np.zeros(num_segments)
        spd = np.zeros(num_segments)

        for s in range(num_segments):
            if rng.rand() < 0.5:
                dt_m[s] = parent_a.dt_mults[s]
                apex[s] = parent_a.apex_offsets[s]
                spd[s] = parent_a.speed_scales[s]
            else:
                dt_m[s] = parent_b.dt_mults[s]
                apex[s] = parent_b.apex_offsets[s]
                spd[s] = parent_b.speed_scales[s]

        num_unpinned = len(parent_a.theta_biases)
        if num_unpinned > 0:
            alpha = rng.uniform(0.0, 1.0, size=num_unpinned)
            thetas = alpha * parent_a.theta_biases + (1 - alpha) * parent_b.theta_biases
        else:
            thetas = np.array([])

        return cls(dt_m, apex, thetas, spd)


def build_guess_from_genome(
    genome: InitialGuessGenome,
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
            unpinned_idx += 1
            wp_headings.append(base_th + bias)

    states = []
    v_max = config.v_max_rad_s * config.wheel_radius / max(config.gearing, 1e-9)

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

        seg_speed = v_max * np.clip(genome.speed_scales[i], 0.1, 1.0)
        is_stop_start = i in stops
        is_stop_end = (i + 1) in stops

        for j in range(count):
            frac = j / s_i
            bump = 4.0 * frac * (1.0 - frac)
            x = p1[0] + dx * frac + nx * apex_amp * bump
            y = p1[1] + dy * frac + ny * apex_amp * bump
            theta = th_start + dth * frac

            speed = seg_speed
            if is_stop_start and frac < 0.2:
                speed *= (frac / 0.2)
            if is_stop_end and frac > 0.8:
                speed *= ((1.0 - frac) / 0.2)

            vl = speed
            vr = speed
            states.append([x, y, theta, vl, vr])

    states = np.array(states, dtype=float)
    guess = np.concatenate([initial_dts, states.flatten()])
    return guess


def evaluate_guess(
    guess: np.ndarray,
    waypoints: List[Tuple[float, float, Optional[float]]],
    config: RobotConfig,
    optimizer: TrajectoryOptimizer,
    engine: CompiledReachabilityEngine,
    stop_waypoint_indices: Optional[List[int]] = None,
    max_iter: int = 120,
) -> Tuple[Optional[List[Dict[str, Any]]], float, Dict[str, Any]]:
    """Solve via IPOPT from guess and evaluate composite fitness."""
    solver_options = {
        "max_iter": max_iter,
        "print_level": 0,
    }

    try:
        samples, stats = optimizer.solve(
            waypoints=waypoints,
            stop_waypoint_indices=stop_waypoint_indices,
            guess_style="baseline",
            initial_guess=guess,
            per_segment_dt=True,
            accuracy_weight=0.002,
            solver_options=solver_options,
            verbose=False,
        )
    except Exception as e:
        return None, 999.0, {"error": str(e)}

    if not samples or len(samples) == 0:
        return None, 999.0, stats

    num_segments = len(waypoints) - 1
    segment_samples = optimizer.get_segment_sample_counts(waypoints, 10)
    sample_to_segment = list(np.repeat(np.arange(num_segments), segment_samples))

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
        "ipopt_iter": stats.get("iter_count", max_iter),
        "converged": stats.get("converged", False),
    }
    return samples, cost, meta


def run_seed_evolution_benchmark():
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
    unpinned_count = sum(1 for i, w in enumerate(waypoints) if w[2] is None and 0 < i < len(waypoints) - 1)

    print(f"Mission: {len(waypoints)} waypoints, {num_segments} segments, {unpinned_count} unpinned intermediate headings")

    # 1. Evaluate standard default seeds for baseline
    print("\n--- 1. Evaluating Standard Hardcoded Seeds ---")
    standard_seeds = ["baseline", "bow_left", "bow_right", "rolling"]
    std_results = {}
    for style in standard_seeds:
        t0 = time.time()
        guess = optimizer._build_initial_guess(waypoints, 10, sum(segment_samples) + 1, style=style, stop_waypoint_indices=stops)
        samples, cost, meta = evaluate_guess(guess, waypoints, config, optimizer, engine, stops, max_iter=150)
        dt = time.time() - t0
        std_results[style] = meta
        print(f"  Seed {style:10s} -> Time: {meta.get('duration', 99):.3f}s | Err: {meta.get('tracking_error_cm', 99):.2f}cm | Cost: {meta.get('cost', 999):.3f} | Iter: {meta.get('ipopt_iter', 0)} ({dt:.2f}s)")

    # 2. Run Initial-Guess Evolutionary Algorithm
    print("\n--- 2. Running Initial-Guess Parameter Evolution ---")
    pop_size = 6
    num_gens = 3

    # Seed 0: Default starting parameters
    pop: List[InitialGuessGenome] = [InitialGuessGenome.create_default(num_segments, unpinned_count)]
    # Seed 1..pop_size-1: Diverse starting parameter variations
    for _ in range(pop_size - 1):
        pop.append(InitialGuessGenome.create_random(num_segments, unpinned_count))

    best_overall_cost = float("inf")
    best_overall_samples = None
    best_overall_meta = None
    best_genome = None

    history = []

    for gen in range(num_gens):
        print(f"\n>> Generation {gen + 1}/{num_gens} (Evaluating {len(pop)} starting parameter genomes)...")
        gen_evals = []
        for idx, ind in enumerate(pop):
            guess = build_guess_from_genome(ind, waypoints, segment_samples, config, stops)
            t0 = time.time()
            samples, cost, meta = evaluate_guess(guess, waypoints, config, optimizer, engine, stops, max_iter=120)
            wall_t = time.time() - t0
            gen_evals.append((ind, cost, samples, meta))
            print(f"   Ind {idx:2d}: Cost = {cost:.3f} | Dur = {meta.get('duration', 99):.3f}s | Err = {meta.get('tracking_error_cm', 99):.2f}cm | Iter = {meta.get('ipopt_iter', 0)} ({wall_t:.2f}s)")

        gen_evals.sort(key=lambda x: x[1])
        gen_best = gen_evals[0]
        history.append({
            "gen": gen + 1,
            "best_cost": gen_best[1],
            "best_meta": gen_best[3],
        })

        if gen_best[1] < best_overall_cost:
            best_overall_cost = gen_best[1]
            best_overall_samples = gen_best[2]
            best_overall_meta = gen_best[3]
            best_genome = gen_best[0]

        print(f"   => Gen {gen + 1} Champion Cost: {gen_best[1]:.3f} (Overall Best: {best_overall_cost:.3f})")

        if gen == num_gens - 1:
            break

        # Breed next generation: Elitism (top 2) + Tournament Crossover + Mutation
        new_pop = [gen_evals[0][0], gen_evals[1][0]]
        while len(new_pop) < pop_size:
            c1, c2 = np.random.choice(len(gen_evals[:4]), size=2, replace=False)
            parent_a = gen_evals[c1][0]
            parent_b = gen_evals[c2][0]
            offspring = InitialGuessGenome.crossover(parent_a, parent_b)
            offspring = offspring.mutate(p_mut=0.35, sigma=0.15)
            new_pop.append(offspring)

        pop = new_pop

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    best_std_seed = min(std_results.items(), key=lambda x: x[1].get('cost', 999))
    print(f"Stock Best Seed ({best_std_seed[0]}): Cost = {best_std_seed[1]['cost']:.3f} | Dur = {best_std_seed[1]['duration']:.3f}s | Err = {best_std_seed[1]['tracking_error_cm']:.2f}cm | Iter = {best_std_seed[1]['ipopt_iter']}")
    print(f"Evolved Initial Guess Seed:  Cost = {best_overall_cost:.3f} | Dur = {best_overall_meta['duration']:.3f}s | Err = {best_overall_meta['tracking_error_cm']:.2f}cm | Iter = {best_overall_meta['ipopt_iter']}")
    if best_std_seed[1]['cost'] > 0:
        cost_impr = (best_std_seed[1]['cost'] - best_overall_cost) / best_std_seed[1]['cost'] * 100.0
        print(f"Cost Improvement:            {cost_impr:+.2f}%")


if __name__ == "__main__":
    run_seed_evolution_benchmark()
