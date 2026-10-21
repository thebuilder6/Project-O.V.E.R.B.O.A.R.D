"""
Tests for Archived Genetic Algorithm & Chimera Splicing Routines
================================================================
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from immrax_validator import ImmraxValidator
from optimizer import TrajectoryOptimizer
from reachability_engine import CompiledReachabilityEngine
from legacy.evolutionary_research.genetic_algorithm import (
    recombine_candidate_segments,
    run_memetic_evolution,
)


class TestGeneticAlgorithm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("examples/fll_choreo.json", "r", encoding="utf-8") as f:
            cls.cfg_data = json.load(f)
        cls.config = RobotConfig(cls.cfg_data)
        cls.jax_cfg = JAXRobotConfig(cls.config)
        cls.validator = ImmraxValidator(cls.jax_cfg)
        cls.engine = CompiledReachabilityEngine(cls.validator)

    def test_chimera_segment_recombination(self):
        """Verify segment-wise recombination (Chimera crossover) on multi-segment candidates."""
        wps = [(0.0, 0.0, 0.0), (0.3, 0.1, 0.2), (0.6, 0.0, 0.0)]
        seg_samples = [4, 4]
        opt = TrajectoryOptimizer(self.config)
        s1, st1 = opt.solve(
            wps, num_samples_per_segment=seg_samples, accuracy_weight=0.0,
            guess_style="baseline", per_segment_dt=True, verbose=False
        )
        s2, st2 = opt.solve(
            wps, num_samples_per_segment=seg_samples, accuracy_weight=0.01,
            guess_style="rolling", per_segment_dt=True, verbose=False
        )
        cands = [(s1, st1), (s2, st2)]

        chim_samples, chim_stats, recomb_rep = recombine_candidate_segments(
            cands, wps, self.config, self.engine, max_polish_iters=150
        )
        if chim_samples:
            self.assertTrue(chim_stats.get("converged", False))
            self.assertIn("chimera_recombination", chim_stats)
            self.assertEqual(len(chim_samples), len(s1))
            self.assertEqual(len(recomb_rep["segment_winners"]), 2)

    def test_memetic_evolution_loop(self):
        """Verify Memetic Evolution algorithm loop, truncation selection, and lineage tracking."""
        wps = [(0.0, 0.0, 0.0), (0.3, 0.1, 0.2), (0.6, 0.0, 0.0)]
        seg_samples = [4, 4]
        opt = TrajectoryOptimizer(self.config)
        s1, st1 = opt.solve(
            wps, num_samples_per_segment=seg_samples, accuracy_weight=0.0,
            guess_style="baseline", per_segment_dt=True, verbose=False
        )
        s2, st2 = opt.solve(
            wps, num_samples_per_segment=seg_samples, accuracy_weight=0.01,
            guess_style="rolling", per_segment_dt=True, verbose=False
        )
        cands = [(s1, st1), (s2, st2)]

        champ_samples, champ_stats, evo_report = run_memetic_evolution(
            cands,
            wps,
            self.config,
            self.engine,
            num_generations=2,
            population_size=4,
            discard_worst_count=2,
            max_polish_iters=100,
            verbose=False,
        )
        self.assertIsNotNone(champ_samples)
        self.assertTrue(champ_stats.get("converged", False))
        self.assertIn("generations", evo_report)
        self.assertEqual(len(evo_report["generations"]), 3)
        self.assertLessEqual(evo_report["champion_score"], evo_report["generations"][0]["best_score"] + 1e-6)
        self.assertIn("population_leaderboard", evo_report)
        self.assertEqual(len(evo_report["population_leaderboard"]), 4)


if __name__ == "__main__":
    unittest.main()
