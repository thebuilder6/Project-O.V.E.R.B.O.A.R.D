"""
Tests for the planner's candidate set and the optional gradient polish.

The 5-Layer Surrogate-to-Truth pipeline is gone. What it had that was worth
keeping is here: the compiled reachability engine (float64, and in parity
with the validator), and the differentiable polish that uses it. The
planner owns both now.
"""

import json
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from overboard.core.optimizer import TrajectoryOptimizer
from overboard.core.unified_planner import UnifiedPathEngine
from overboard.models.jax_robot_model import JAXRobotConfig
from overboard.models.robot_model import RobotConfig
from overboard.verification.immrax_validator import ImmraxValidator
from overboard.verification.reachability_engine import (
    CompiledReachabilityEngine,
    _extract_state_and_dt_arrays,
)


class TestPlannerCandidateSet(unittest.TestCase):
    """The planner owns the candidate set, and the set is the homotopy triad."""

    def setUp(self):
        with open("examples/fll_choreo.json", "r", encoding="utf-8") as handle:
            self.config = RobotConfig(json.load(handle))

    def test_the_set_is_the_triad(self):
        planner = UnifiedPathEngine(self.config)

        specs = planner.candidate_specs()

        self.assertEqual([s["guess_style"] for s in specs],
                         ["frontier_sprint", "frontier_sweeper",
                          "frontier_exploratory"])
        for spec in specs:
            # The per-segment model is the general one (KI-022).
            self.assertTrue(spec["per_segment_dt"])
            self.assertEqual(spec["cost_metric"], "discretization")
            self.assertTrue(spec["epigraph_constraints"])

    def test_the_set_can_be_cut_short(self):
        planner = UnifiedPathEngine(self.config)

        self.assertEqual(len(planner.candidate_specs(count=2)), 2)
        self.assertEqual(len(planner.candidate_specs(count=1)), 1)

    def test_one_weight_for_the_whole_set(self):
        planner = UnifiedPathEngine(self.config)

        specs = planner.candidate_specs(accuracy_weight=0.007)

        self.assertEqual([s["accuracy_weight"] for s in specs], [0.007] * 3)


class TestReachabilityEngine(unittest.TestCase):
    """The compiled engine: float64, and in parity with the validator."""

    @classmethod
    def setUpClass(cls):
        with open("examples/fll_choreo.json", "r", encoding="utf-8") as handle:
            cls.config = RobotConfig(json.load(handle))
        cls.jax_cfg = JAXRobotConfig(cls.config)
        cls.validator = ImmraxValidator(cls.jax_cfg)
        cls.engine = CompiledReachabilityEngine(cls.validator)

    def test_float64_and_parity_with_the_validator(self):
        self.assertTrue(jax.config.read("jax_enable_x64"))

        wps = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)]
        optimizer = TrajectoryOptimizer(self.config)
        samples, stats = optimizer.solve(
            wps, num_samples_per_segment=5, verbose=False,
            per_segment_dt=True, cost_metric="discretization",
            epigraph_constraints=True)
        self.assertTrue(stats["converged"])

        seg_samples = stats["segment_samples"]
        num_segments = len(wps) - 1
        sample_to_segment = list(
            np.repeat(np.arange(num_segments), seg_samples))
        states, dts = _extract_state_and_dt_arrays(
            samples, num_segments, seg_samples)
        self.assertEqual(states.dtype, jnp.float64)
        self.assertEqual(dts.dtype, jnp.float64)

        scan_out = self.engine.evaluate_states(states, dts, sample_to_segment)
        compiled = self.engine.build_immrax_report(samples, scan_out)

        u = self.config.uncertainties
        reference = self.validator.compute_reachability(
            samples, u["gear_backlash_m"], u["track_width_range"],
            u["wheel_radius_range"], u["cof"], u["torque_headroom"],
            u["inertia_kg_m2"])

        self.assertAlmostEqual(compiled["max_tracking_error_m"],
                               reference["max_tracking_error_m"], places=6)
        self.assertAlmostEqual(compiled["max_envelope_radius_m"],
                               reference["max_envelope_radius_m"], places=6)
        self.assertTrue(compiled["passed"])

    def test_the_batch_evaluation_answers_for_every_candidate(self):
        wps = [(0.0, 0.0, 0.0), (0.8, 0.0, 0.0)]
        optimizer = TrajectoryOptimizer(self.config)
        first, first_stats = optimizer.solve(
            wps, verbose=False, cost_metric="discretization",
            epigraph_constraints=True)
        second, _ = optimizer.solve(
            wps, accuracy_weight=0.01, verbose=False,
            cost_metric="discretization", epigraph_constraints=True)

        seg_samples = first_stats["segment_samples"]
        sample_to_segment = list(np.repeat(np.arange(1), seg_samples))
        states1, dts1 = _extract_state_and_dt_arrays(first, 1, seg_samples)
        states2, dts2 = _extract_state_and_dt_arrays(second, 1, seg_samples)

        batch = self.engine.evaluate_batch_vmap(
            jnp.stack([states1, states2], axis=0),
            jnp.stack([dts1, dts2], axis=0), sample_to_segment)

        self.assertEqual(batch["max_tracking_error_m"].shape, (2,))
        self.assertEqual(batch["max_envelope_radius_m"].shape, (2,))


class TestGradientPolish(unittest.TestCase):
    """The polish: the reachability says which samples to fix, and they are fixed."""

    @classmethod
    def setUpClass(cls):
        with open("examples/fll_choreo.json", "r", encoding="utf-8") as handle:
            cls.config = RobotConfig(json.load(handle))
        cls.planner = UnifiedPathEngine(cls.config)

    def _plan(self):
        wps = [(0.0, 0.0, 0.0), (0.3, 0.10, 0.25), (0.6, 0.20, 0.0)]
        samples, stats = self.planner.optimizer.solve(
            wps, num_samples_per_segment=6, verbose=False,
            per_segment_dt=True, cost_metric="discretization",
            epigraph_constraints=True)
        return wps, samples, stats

    def _key(self, samples, stats):
        """The pair the polish compares on: tracking error, then the tube."""
        num_segments = len(stats["segment_samples"])
        sample_to_segment = list(
            np.repeat(np.arange(num_segments), stats["segment_samples"]))
        states, dts = _extract_state_and_dt_arrays(
            samples, num_segments, stats["segment_samples"])
        report = self.planner._ensure_validator().evaluate_states(
            states, dts, sample_to_segment)
        return (float(report["max_tracking_error_m"]),
                float(report["max_envelope_radius_m"]))

    def test_the_polish_never_makes_the_plan_worse(self):
        wps, samples, stats = self._plan()

        polished, polished_stats = self.planner.gradient_polish(
            samples, stats, wps, max_rounds=1)

        self.assertTrue(polished)
        self.assertIn("polish", polished_stats)
        self.assertLessEqual(self._key(polished, polished_stats),
                             self._key(samples, stats))

    def test_the_polish_never_makes_the_plan_slower(self):
        """A time-optimal planner must not trade time for a smaller tube."""
        wps, samples, stats = self._plan()

        polished, _ = self.planner.gradient_polish(
            samples, stats, wps, max_rounds=1)

        self.assertLessEqual(float(polished[-1]["t"]),
                             float(samples[-1]["t"]) + 1e-9)

    def test_a_plan_that_does_not_converge_is_left_alone(self):
        wps, samples, stats = self._plan()
        stats = dict(stats, converged=False)

        polished, polished_stats = self.planner.gradient_polish(
            samples, stats, wps)

        self.assertEqual(polished, samples)
        self.assertEqual(polished_stats, stats)


class TestCandidatePolish(unittest.TestCase):
    """The engine's own entry point: the polish on a candidate result."""

    @classmethod
    def setUpClass(cls):
        with open("examples/fll_choreo.json", "r", encoding="utf-8") as handle:
            cls.config = RobotConfig(json.load(handle))
        cls.planner = UnifiedPathEngine(cls.config)

    def _candidate(self):
        wps = [(0.0, 0.0, 0.0), (0.3, 0.10, 0.25), (0.6, 0.20, 0.0)]
        seg_samples = [6, 6]
        seed = self.planner.generate_seed(wps, seg_samples, "sprint")
        candidate = self.planner.solve_candidate(
            waypoints=wps, seg_samples=seg_samples, name="sprint",
            guess=seed, stop_indices=[], accuracy_weight=0.002)
        self.assertIsNotNone(candidate)
        return wps, seg_samples, candidate

    def test_the_polish_never_makes_the_candidate_slower(self):
        wps, seg_samples, candidate = self._candidate()

        polished = self.planner.polish_candidate_sensitivities(
            candidate, wps, seg_samples, [])

        self.assertLessEqual(polished.duration_s, candidate.duration_s + 1e-9)
        if polished is not candidate:
            self.assertLessEqual(polished.tracking_error_cm,
                                 candidate.tracking_error_cm)
            self.assertLessEqual(polished.envelope_radius_cm,
                                 candidate.envelope_radius_cm)

    def test_the_polish_keeps_the_candidate_when_it_cannot_help(self):
        """A candidate that does not converge is left alone."""
        wps, seg_samples, candidate = self._candidate()
        candidate.stats = dict(candidate.stats, converged=False)

        polished = self.planner.polish_candidate_sensitivities(
            candidate, wps, seg_samples, [])

        self.assertIs(polished, candidate)

    def _plan_with_gate(self, tracking_gate_m):
        """Run plan() and record whether it asked for a polish.

        The polish itself is stubbed: this test is about *when* the engine
        asks for it, and the real pass costs a compiled rollout and a solve.
        """
        wps = [(0.0, 0.0, 0.0), (0.3, 0.10, 0.25), (0.6, 0.20, 0.0)]
        planner = UnifiedPathEngine(self.config, tracking_gate_m=tracking_gate_m)
        calls = []

        def spy(candidate, *args, **kwargs):
            calls.append(candidate)
            return candidate

        planner.polish_candidate_sensitivities = spy
        planner.plan(waypoints=wps, mesh_samples=[6, 6], verbose=False)
        return calls

    def test_a_champion_near_the_gate_is_polished(self):
        """The engine self-heals a plan that rides close to the gate."""
        # A tight gate puts the champion next to it, so the pass runs.
        calls = self._plan_with_gate(tracking_gate_m=0.0005)
        self.assertTrue(calls, "the champion near the gate was not polished")

    def test_a_champion_inside_the_gate_is_left_alone(self):
        """A plan with room to spare does not pay for a polish."""
        calls = self._plan_with_gate(tracking_gate_m=0.5)
        self.assertFalse(calls, "a plan inside the gate was polished anyway")
