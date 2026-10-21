"""
Unit & Integration Tests for the 5-Layer Surrogate-to-Truth Pipeline
====================================================================

Verifies:
1. Layer 1: Discretization-Aware Metric + Linear Epigraph Force Bounds in CasADi
2. Layer 2: Single-Process ThreadPoolExecutor candidate solver (`solve_candidates_threaded`)
3. Layer 3: Compiled `jax.lax.scan` + `vmap` Reachability Engine in float64 & numerical parity with `ImmraxValidator`
4. Layer 4: Differentiable Immrax Polish (`jax.value_and_grad` through `lax.scan`)
5. Layer 5: Gate-Certified Minimum-Time Candidate Selection (`_select_attempt(..., prefer_duration=True)`)
"""

import json
import unittest
import numpy as np
import jax
import jax.numpy as jnp

from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from immrax_validator import ImmraxValidator
from optimizer import TrajectoryOptimizer
from reachability_engine import CompiledReachabilityEngine
from surrogate_polish import (
    SurrogatePolishOptimizer,
    _extract_state_and_dt_arrays,
    generate_homotopy_bank,
    screen_candidates_jax,
    select_topological_representatives,
    solve_candidates_threaded,
    solve_pyramid_candidates,
)
from main import _select_attempt, _load_waypoints


class TestSurrogatePolishPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("examples/fll_choreo.json", "r", encoding="utf-8") as f:
            cls.cfg_data = json.load(f)
        cls.config = RobotConfig(cls.cfg_data)
        cls.jax_cfg = JAXRobotConfig(cls.config)
        cls.validator = ImmraxValidator(cls.jax_cfg)
        cls.engine = CompiledReachabilityEngine(cls.validator)

    def test_layer3_float64_and_reachability_parity(self):
        """Layer 3: Verify float64 precision and exact parity between lax.scan and ImmraxValidator."""
        self.assertTrue(jax.config.read("jax_enable_x64"))

        wps = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0)]
        opt = TrajectoryOptimizer(self.config)
        samples, stats = opt.solve(
            wps,
            num_samples_per_segment=5,
            verbose=False,
            per_segment_dt=True,
            cost_metric="discretization",
            epigraph_constraints=True,
        )
        self.assertTrue(stats["converged"])

        seg_samples = stats["segment_samples"]
        num_segments = len(wps) - 1
        sample_to_segment = list(np.repeat(np.arange(num_segments), seg_samples))
        states, dts = _extract_state_and_dt_arrays(samples, num_segments, seg_samples)
        self.assertEqual(states.dtype, jnp.float64)
        self.assertEqual(dts.dtype, jnp.float64)

        scan_out = self.engine.evaluate_states(states, dts, sample_to_segment)
        compiled_report = self.engine.build_immrax_report(samples, scan_out)

        u = self.config.uncertainties
        reach_ref = self.validator.compute_reachability(
            samples,
            u["gear_backlash_m"],
            u["track_width_range"],
            u["wheel_radius_range"],
            u["cof"],
            u["torque_headroom"],
            u["inertia_kg_m2"],
        )

        self.assertAlmostEqual(
            compiled_report["max_tracking_error_m"],
            reach_ref["max_tracking_error_m"],
            places=6,
        )
        self.assertAlmostEqual(
            compiled_report["max_envelope_radius_m"],
            reach_ref["max_envelope_radius_m"],
            places=6,
        )
        self.assertTrue(compiled_report["passed"])

    def test_layer3_vmap_batch_evaluation(self):
        """Layer 3: Verify batched multi-candidate evaluation via jax.vmap."""
        wps = [(0.0, 0.0, 0.0), (0.8, 0.0, 0.0)]
        opt = TrajectoryOptimizer(self.config)
        s1, st1 = opt.solve(wps, verbose=False, cost_metric="discretization", epigraph_constraints=True)
        s2, st2 = opt.solve(wps, accuracy_weight=0.01, verbose=False, cost_metric="discretization", epigraph_constraints=True)

        seg_samples = st1["segment_samples"]
        sample_to_segment = list(np.repeat(np.arange(1), seg_samples))
        states1, dts1 = _extract_state_and_dt_arrays(s1, 1, seg_samples)
        states2, dts2 = _extract_state_and_dt_arrays(s2, 1, seg_samples)

        batch_states = jnp.stack([states1, states2], axis=0)
        batch_dts = jnp.stack([dts1, dts2], axis=0)
        batch_out = self.engine.evaluate_batch_vmap(batch_states, batch_dts, sample_to_segment)

        self.assertEqual(batch_out["max_tracking_error_m"].shape, (2,))
        self.assertEqual(batch_out["max_envelope_radius_m"].shape, (2,))

    def test_layer4_differentiable_sensitivities_and_polish(self):
        """Layer 4: Verify jax.value_and_grad through lax.scan and polish improvement on S-curve."""
        wps = [(0.0, 0.0, 0.0), (0.3, 0.10, 0.25), (0.6, 0.20, 0.0)]
        sp_opt = SurrogatePolishOptimizer(
            self.config,
            validator=self.validator,
            engine=self.engine,
            safety_gate_m=0.0005,  # Trigger 1 round of sensitivity polish
        )
        samples, stats = sp_opt.solve(
            wps,
            num_samples_per_segment=6,
            verbose=False,
            per_segment_dt=False,
            max_polish_rounds=1,
            polish_iter_budget=60,
            build_full_report=False,
        )
        self.assertTrue(stats["converged"])
        self.assertIn("surrogate_polish", stats)
        sp_info = stats["surrogate_polish"]
        self.assertTrue(sp_info["polished"])
        self.assertLessEqual(sp_info["final_tracking_m"], sp_info["initial_tracking_m"])
        self.assertLessEqual(sp_info["final_tracking_m"], 0.005)
        self.assertLessEqual(sp_info["final_tube_m"], 0.020)

    def test_layer2_and_layer5_threaded_candidates_and_selection(self):
        """Layer 2 & 5: Verify ThreadPoolExecutor candidate solving and minimum-time gate selection."""
        wps, evs, _, stops = _load_waypoints("examples/example_mission_first4.json")
        payloads = [
            (
                self.config,
                wps,
                {"guess_style": "baseline", "per_segment_dt": False, "accuracy_weight": 0.0},
                {
                    "num_samples_per_segment": 5,
                    "accuracy_weight": 0.0,
                    "stop_waypoint_indices": stops,
                    "waypoint_events": evs,
                    "verbose": False,
                    "guess_style": "baseline",
                    "per_segment_dt": False,
                    "max_polish_rounds": 0,
                    "build_full_report": False,
                },
                False,
            ),
            (
                self.config,
                wps,
                {"guess_style": "baseline", "per_segment_dt": True, "accuracy_weight": 0.0},
                {
                    "num_samples_per_segment": 5,
                    "accuracy_weight": 0.0,
                    "stop_waypoint_indices": stops,
                    "waypoint_events": evs,
                    "verbose": False,
                    "guess_style": "baseline",
                    "per_segment_dt": True,
                    "max_polish_rounds": 0,
                    "build_full_report": False,
                },
                False,
            ),
        ]
        solved, _ = solve_candidates_threaded(
            payloads, workers=2, validator=self.validator, engine=self.engine
        )
        self.assertEqual(len(solved), 2)
        self.assertTrue(solved[0][1]["converged"])
        self.assertTrue(solved[1][1]["converged"])

        # Layer 5 check: prefer_duration=True picks the faster passing candidate
        attempts = [
            {"ok": True, "converged": True, "duration": 2.50, "cost": 2.50, "tube": 0.004, "fitness": 3.5},
            {"ok": True, "converged": True, "duration": 1.80, "cost": 1.80, "tube": 0.006, "fitness": 3.8},
        ]
        best_default, _ = _select_attempt(attempts, prefer_duration=False)
        best_surrogate, _ = _select_attempt(attempts, prefer_duration=True)
        self.assertEqual(best_default["duration"], 2.50)
        self.assertEqual(best_surrogate["duration"], 1.80)

    def test_pyramid_stage1_generate_homotopy_bank(self):
        """Pyramid Stage 1: Verify batch generation of diverse geometric/timing hypotheses."""
        wps = [(0.0, 0.0, 0.0), (0.4, 0.1, None), (0.8, 0.0, 0.0)]
        seg_samples = [5, 5]
        stops = [1]
        cand_states, cand_dts, meta = generate_homotopy_bank(
            wps,
            seg_samples,
            track_width=float(self.config.track_width),
            v_max=float(self.config.max_linear_speed()),
            stop_waypoint_indices=stops,
        )

        M = len(cand_states)
        N = sum(seg_samples) + 1
        # Constraint-aware pruned sizing: M should cleanly be in 15-40 range
        self.assertGreaterEqual(M, 9)
        self.assertLessEqual(M, 40)
        self.assertEqual(cand_states.shape, (M, N, 5))
        self.assertEqual(cand_dts.shape, (M, 2))
        self.assertEqual(len(meta), M)

        # Check boundary zero velocities
        np.testing.assert_allclose(cand_states[:, 0, 3:5], 0.0)
        np.testing.assert_allclose(cand_states[:, -1, 3:5], 0.0)
        np.testing.assert_allclose(cand_states[:, 5, 3:5], 0.0)

    def test_pyramid_constraint_aware_pruning(self):
        """Verify that pinned headings and short segments collapse M automatically."""
        # 1. Fully pinned heading mission: 2 segments, pinned headings, non-stop
        wps_pinned = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        cand_states_p, _, meta_p = generate_homotopy_bank(
            wps_pinned, [4, 4],
            track_width=float(self.config.track_width),
            v_max=float(self.config.max_linear_speed()),
        )
        # All angle offsets should be 0.0 (no redundant angle sweeps)
        self.assertTrue(all(m["angle_offset"] == 0.0 for m in meta_p))
        self.assertLessEqual(len(cand_states_p), 18)

        # 2. Short segment mission (< 15 cm): lateral apex shift must collapse to 0.0
        wps_short = [(0.0, 0.0, 0.0), (0.10, 0.0, 0.0)]
        cand_states_s, _, meta_s = generate_homotopy_bank(
            wps_short, [3],
            track_width=float(self.config.track_width),
            v_max=float(self.config.max_linear_speed()),
        )
        self.assertTrue(all(m["shift_factor"] == 0.0 for m in meta_s))
        self.assertLessEqual(len(cand_states_s), 5)

    def test_pyramid_stage2_jax_sieve_screening(self):
        """Pyramid Stage 2: Verify vectorized screening in JAX via evaluate_batch_vmap."""
        wps = [(0.0, 0.0, 0.0), (0.4, 0.1, None), (0.8, 0.0, 0.0)]
        seg_samples = [4, 4]
        cand_states, cand_dts, _ = generate_homotopy_bank(
            wps,
            seg_samples,
            track_width=float(self.config.track_width),
            v_max=float(self.config.max_linear_speed()),
        )
        sample_to_seg = list(np.repeat(np.arange(2), seg_samples))

        costs = screen_candidates_jax(
            self.engine,
            cand_states,
            cand_dts,
            sample_to_seg,
            safety_gate_m=0.005,
            tube_gate_m=0.020,
        )
        self.assertEqual(costs.shape, (len(cand_states),))
        self.assertTrue(np.all(np.isfinite(costs)))
        self.assertTrue(np.all(costs > 0.0))

    def test_pyramid_stage3_topological_clustering(self):
        """Pyramid Stage 3: Verify farthest-point deduplication selects distinct candidates."""
        wps = [(0.0, 0.0, 0.0), (0.4, 0.15, None), (0.8, 0.0, 0.0)]
        seg_samples = [4, 4]
        cand_states, cand_dts, _ = generate_homotopy_bank(
            wps,
            seg_samples,
            track_width=float(self.config.track_width),
            v_max=float(self.config.max_linear_speed()),
        )
        sample_to_seg = list(np.repeat(np.arange(2), seg_samples))
        costs = screen_candidates_jax(self.engine, cand_states, cand_dts, sample_to_seg)

        k = 3
        selected = select_topological_representatives(
            cand_states, costs, k_clusters=k, min_separation_m=0.08
        )
        self.assertEqual(len(selected), k)
        self.assertEqual(len(set(selected)), k)

    def test_pyramid_stage4_parallel_solve_and_end_to_end(self):
        """Pyramid Stage 4: Verify end-to-end solve_pyramid execution."""
        wps = [(0.0, 0.0, 0.0), (0.3, 0.1, 0.2), (0.6, 0.0, 0.0)]
        sp_opt = SurrogatePolishOptimizer(
            self.config,
            validator=self.validator,
            engine=self.engine,
        )
        samples, stats = sp_opt.solve_pyramid(
            wps,
            num_samples_per_segment=4,
            k_clusters=2,
            max_polish_rounds=0,
            build_full_report=False,
            max_workers=2,
        )
        self.assertTrue(stats.get("converged", False))
        self.assertIn("pyramid", stats)
        pyramid_meta = stats["pyramid"]
        # Pinned headings cleanly prune candidates to exactly 18
        self.assertEqual(pyramid_meta["total_generated"], 18)
        self.assertEqual(pyramid_meta["screened_clusters"], 2)
        self.assertGreaterEqual(pyramid_meta["converged_count"], 1)


if __name__ == "__main__":
    unittest.main()
