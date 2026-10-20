"""
Regression tests for the audit fixes.

Covers:
- exported Python config matches the robot (audit finding 1)
- attempt selection never prefers a failed attempt over a passing one (finding 2)
- reachability envelope is aligned to samples and flags divergence (finding 3)
- upper uncertainty bounds widen the reachability envelope (KI-003)
- benchmark validation reads the config that was used (finding 4)
- failed solves and failed validation never write output (KI-001/KI-002)
"""

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from robot_model import RobotConfig
from optimizer import GUESS_STYLES
from export import write_python_file
from benchmark import RandomBenchmark
from validator import check_final_waypoint, forward_integrate, validate_samples
import main

try:
    from immrax_validator import ImmraxValidator
    HAS_IMMRAX = True
except ImportError:
    HAS_IMMRAX = False


def _consistent_samples(n=11, dt=0.1, speed=0.1):
    """A constant-speed straight line that the CPU validator accepts."""
    return [{
        "t": i * dt, "x": speed * i * dt, "y": 0.0, "heading": 0.0,
        "vl": speed, "vr": speed, "omega": 0.0,
        "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
    } for i in range(n)]


_TEST_ROBOT = {"robot": {
    "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
    "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
    "gearing": 1.0, "cof": 1.37,
}}


class TestExportConfig(unittest.TestCase):
    """Finding 1: write_python_file must export the robot's real parameters."""

    def _export(self, config):
        with tempfile.TemporaryDirectory() as tmp:
            traj = os.path.join(tmp, "traj.json")
            out = os.path.join(tmp, "traj.py")
            with open(traj, "w") as f:
                json.dump({
                    "trajectory": {
                        "config": config,
                        "samples": [{
                            "t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0,
                            "vl": 0.0, "vr": 0.0, "omega": 0.0,
                        }],
                    }
                }, f)
            with contextlib.redirect_stdout(io.StringIO()):
                write_python_file(traj, out)
            namespace = {}
            with open(out) as f:
                exec(f.read(), namespace)
            return namespace["config"]

    def test_canonical_config_is_exported(self):
        config = self._export({
            "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
            "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
            "gearing": 1.0, "cof": 1.37,
        })
        self.assertAlmostEqual(config["v_max_rad_s"], 18.94)
        self.assertAlmostEqual(config["t_max_nm"], 0.138)
        self.assertAlmostEqual(config["cof"], 1.37)
        self.assertAlmostEqual(config["track_width"], 0.0965)
        self.assertAlmostEqual(config["wheel_radius"], 0.028)

    def test_legacy_config_keys_are_exported(self):
        config = self._export({
            "mass": 0.9, "inertia": 0.002, "vmax": 12.0, "tmax": 0.05,
            "cof": 0.8, "differentialTrackWidth": 0.1, "radius": 0.03,
            "gearing": 2.0,
        })
        self.assertAlmostEqual(config["v_max_rad_s"], 12.0)
        self.assertAlmostEqual(config["t_max_nm"], 0.05)
        self.assertAlmostEqual(config["track_width"], 0.1)
        self.assertAlmostEqual(config["wheel_radius"], 0.03)
        self.assertAlmostEqual(config["gearing"], 2.0)


class TestAttemptSelection(unittest.TestCase):
    """Finding 2: a failed attempt must never be preferred over a passing one."""

    @staticmethod
    def attempt(name, cost, ok=False, violation=float("inf"), converged=True,
                tube=float("inf"), duration=float("inf")):
        return {"samples": name, "stats": {}, "cost": cost,
                "ok": ok, "violation": violation, "converged": converged,
                "tube": tube, "duration": duration}

    def setUp(self):
        from main import _select_attempt
        self.select = _select_attempt

    def test_passing_attempt_beats_cheaper_failed_one(self):
        failed = self.attempt("failed", 10.0, violation=0.07)
        passed = self.attempt("passed", 20.0, ok=True)
        best, passed_any = self.select([failed, passed])
        self.assertTrue(passed_any)
        self.assertEqual(best["samples"], "passed")

    def test_smallest_tracking_error_wins_when_none_pass(self):
        worse = self.attempt("worse", 10.0, violation=0.08)
        better = self.attempt("better", 20.0, violation=0.05)
        best, passed_any = self.select([worse, better])
        self.assertFalse(passed_any)
        self.assertEqual(best["samples"], "better")

    def test_smallest_envelope_wins_among_passing(self):
        fast = self.attempt("fast", 10.0, ok=True, tube=0.012, duration=6.0)
        safe = self.attempt("safe", 20.0, ok=True, tube=0.008, duration=9.0)
        best, passed_any = self.select([fast, safe])
        self.assertTrue(passed_any)
        self.assertEqual(best["samples"], "safe")

    def test_shortest_wins_when_envelopes_tie(self):
        slow = self.attempt("slow", 10.0, ok=True, tube=0.008, duration=9.0)
        quick = self.attempt("quick", 20.0, ok=True, tube=0.008, duration=6.0)
        best, _ = self.select([slow, quick])
        self.assertEqual(best["samples"], "quick")

    def test_tiny_envelope_difference_beats_cheaper_candidate(self):
        cheap = self.attempt("cheap", 10.0, ok=True, tube=0.010, duration=6.0)
        safer = self.attempt("safer", 20.0, ok=True,
                             tube=0.010 - 1e-6, duration=6.0)
        best, _ = self.select([cheap, safer])
        self.assertEqual(best["samples"], "safer")

    def test_quantified_failure_beats_unquantified(self):
        physical = self.attempt("physical", 5.0)  # violation=inf
        tracking = self.attempt("tracking", 20.0, violation=0.09)
        best, _ = self.select([physical, tracking])
        self.assertEqual(best["samples"], "tracking")

    def test_cheapest_wins_when_all_failed_physically(self):
        expensive = self.attempt("expensive", 20.0)
        cheap = self.attempt("cheap", 10.0)
        best, _ = self.select([expensive, cheap])
        self.assertEqual(best["samples"], "cheap")

    def test_non_converged_attempt_is_never_selected(self):
        non_converged = self.attempt(
            "nonconverged", 1.0, violation=0.0, converged=False)
        converged = self.attempt("converged", 20.0, violation=0.05)
        best, passed_any = self.select([non_converged, converged])
        self.assertFalse(passed_any)
        self.assertEqual(best["samples"], "converged")

    def test_no_converged_attempts_returns_none(self):
        best, passed_any = self.select([
            self.attempt("bad", 1.0, converged=False),
        ])
        self.assertIsNone(best)
        self.assertFalse(passed_any)


class TestCandidateSpecs(unittest.TestCase):
    """KI-020: candidates are an ordered, scored set, not silent escalation."""

    def setUp(self):
        from main import _candidate_specs
        self.specs = _candidate_specs

    def test_auto_alternates_timing_modes_per_seed(self):
        specs = self.specs(0.0, 5, use_styles=True)
        self.assertEqual([s["guess_style"] for s in specs],
                         ["baseline", "baseline", "rolling", "rolling", "bow_left"])
        self.assertEqual([s["per_segment_dt"] for s in specs],
                         [False, True, False, True, False])
        self.assertEqual([s["accuracy_weight"] for s in specs], [0.0] * 5)

    def test_uniform_mode_uses_seeds_without_per_segment_timing(self):
        specs = self.specs(0.0, 3, use_styles=True, dt_mode="uniform")
        self.assertEqual([s["guess_style"] for s in specs],
                         ["baseline", "rolling", "bow_left"])
        self.assertTrue(all(not s["per_segment_dt"] for s in specs))

    def test_per_segment_mode_uses_seeds_with_per_segment_timing(self):
        specs = self.specs(0.0, 3, use_styles=True, dt_mode="per-segment")
        self.assertEqual([s["guess_style"] for s in specs],
                         ["baseline", "rolling", "bow_left"])
        self.assertTrue(all(s["per_segment_dt"] for s in specs))

    def test_unknown_dt_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.specs(0.0, 2, use_styles=True, dt_mode="nope")

    def test_single_candidate_is_the_uniform_baseline(self):
        specs = self.specs(1.5, 1, use_styles=True)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["guess_style"], "baseline")
        self.assertEqual(specs[0]["accuracy_weight"], 1.5)
        self.assertFalse(specs[0]["per_segment_dt"])

    def test_count_is_clamped_to_at_least_one(self):
        specs = self.specs(0.0, 0, use_styles=True)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["guess_style"], "baseline")
        self.assertFalse(specs[0]["per_segment_dt"])

    def test_multiverse_keeps_accuracy_weight_ladder(self):
        specs = self.specs(0.0, 3, use_styles=False)
        self.assertEqual([s["accuracy_weight"] for s in specs], [0.0, 2.0, 4.0])
        self.assertTrue(all(s["guess_style"] == "baseline" for s in specs))
        self.assertTrue(all(not s["per_segment_dt"] for s in specs))

    def test_refine_adds_the_pipeline_trajectory_without_replacing_any(self):
        """KI-004: the pipeline is a seed provider, not a second solver."""
        plain = self.specs(0.0, 2, use_styles=True)
        refined = self.specs(0.0, 2, use_styles=True, refined=True)
        self.assertEqual(plain, refined[:len(plain)])
        added = refined[len(plain):]
        self.assertTrue(added, "the refined candidate was not added")
        for spec in added:
            self.assertEqual(spec.get("guess"), "refined")
            # The refined trajectory carries a per-segment interval, so only
            # the model that reads those intervals can use it.
            self.assertTrue(spec["per_segment_dt"])

    def test_refine_is_off_by_default(self):
        specs = self.specs(0.0, 2, use_styles=True)
        self.assertFalse(any(s.get("guess") == "refined" for s in specs))

    def test_refine_covers_the_accuracy_weight_grid(self):
        """The weight is what buys the tracking error, so the seed needs it."""
        specs = self.specs(0.0, 1, use_styles=True, refined=True)
        weights = [s["accuracy_weight"] for s in specs if s.get("guess") == "refined"]
        self.assertIn(0.0, weights)
        self.assertIn(0.001, weights)
        self.assertIn(0.005, weights)


class TestSweepSpecs(unittest.TestCase):
    """The sweep adds the timing-mode x accuracy-weight grid to the set."""

    def setUp(self):
        from main import _candidate_specs, SWEEP_WEIGHTS
        self.specs = _candidate_specs
        self.weights = SWEEP_WEIGHTS

    def test_sweep_keeps_the_default_candidates(self):
        plain = self.specs(0.0, 2, use_styles=True)
        swept = self.specs(0.0, 2, use_styles=True, sweep=True)
        for spec in plain:
            self.assertIn(spec, swept)

    def test_sweep_covers_both_timing_modes_at_every_weight(self):
        swept = self.specs(0.0, 2, use_styles=True, sweep=True)
        for weight in self.weights:
            for per_segment_dt in (False, True):
                self.assertIn(
                    {"accuracy_weight": weight, "guess_style": "baseline",
                     "per_segment_dt": per_segment_dt},
                    swept)

    def test_sweep_does_not_solve_the_same_combination_twice(self):
        swept = self.specs(0.0, 2, use_styles=True, sweep=True)
        keys = [(s["accuracy_weight"], s["guess_style"], s["per_segment_dt"])
                for s in swept]
        self.assertEqual(len(keys), len(set(keys)))

    def test_sweep_includes_a_non_zero_base_weight(self):
        swept = self.specs(0.005, 2, use_styles=True, sweep=True)
        self.assertIn(0.005, [s["accuracy_weight"] for s in swept])
        self.assertIn(0.0, [s["accuracy_weight"] for s in swept])

    def test_sweep_is_off_by_default(self):
        plain = self.specs(0.0, 2, use_styles=True)
        self.assertEqual(len(plain), 2)


@unittest.skipUnless(HAS_IMMRAX, "jax/immrax not installed")
class TestExportGate(unittest.TestCase):
    """KI-001/KI-002: the write gate refuses failed solves and validation."""

    def setUp(self):
        from main import _export_block_reason
        self.gate = _export_block_reason

    def test_non_converged_solve_is_blocked(self):
        self.assertIsNotNone(
            self.gate({"converged": False}, validate=False, attempt_passed=True))

    def test_missing_converged_key_is_blocked(self):
        self.assertIsNotNone(
            self.gate({}, validate=False, attempt_passed=True))

    def test_failed_validation_is_blocked(self):
        self.assertEqual(
            self.gate({"converged": True}, validate=True, attempt_passed=False),
            "validation failed")

    def test_converged_validation_pass_is_allowed(self):
        self.assertIsNone(
            self.gate({"converged": True}, validate=True, attempt_passed=True))

    def test_converged_no_validation_is_allowed(self):
        self.assertIsNone(
            self.gate({"converged": True}, validate=False, attempt_passed=False))


class TestValidatorGates(unittest.TestCase):
    """The in-memory checks and the final-waypoint check used before writing."""

    def setUp(self):
        self.cfg = RobotConfig(_TEST_ROBOT)

    def test_validate_samples_accepts_consistent_trajectory(self):
        summary = validate_samples(_consistent_samples(), self.cfg)
        self.assertTrue(summary["passed"])
        self.assertLess(summary["errors"]["max_pos_error_m"], 0.01)

    def test_validate_samples_rejects_wrong_final_position(self):
        samples = _consistent_samples()
        samples[-1]["x"] = 1.0
        summary = validate_samples(samples, self.cfg)
        self.assertFalse(summary["passed"])
        self.assertGreater(summary["errors"]["final_pos_error_m"], 0.01)

    def test_validate_samples_rejects_overspeed(self):
        samples = _consistent_samples(speed=1.0)
        summary = validate_samples(samples, self.cfg)
        self.assertFalse(summary["passed"])
        self.assertGreater(summary["audit"]["speed_violation_m_s"], 0.0)

    def test_forward_integrate_rejects_negative_time(self):
        samples = _consistent_samples()
        for i, s in enumerate(samples):
            s["t"] = -0.1 * i
        integrated, errors = forward_integrate(samples, self.cfg)
        self.assertEqual(integrated, [])
        self.assertTrue(math.isinf(errors["max_pos_error_m"]))

        summary = validate_samples(samples, self.cfg)
        self.assertFalse(summary["passed"])

    def test_final_waypoint_matches(self):
        check = check_final_waypoint(_consistent_samples(), (0.1, 0.0, 0.0))
        self.assertTrue(check["passed"])
        self.assertLess(check["pos_error_m"], 1e-9)

    def test_final_waypoint_far_away_fails(self):
        check = check_final_waypoint(_consistent_samples(), (1.1, 0.0, 0.0))
        self.assertFalse(check["passed"])
        self.assertAlmostEqual(check["pos_error_m"], 1.0, places=6)

    def test_final_waypoint_heading_wraps_across_pi(self):
        samples = _consistent_samples()
        samples[-1]["heading"] = 3.13
        check = check_final_waypoint(
            samples, (0.1, 0.0, -3.13), heading_tol=0.05)
        self.assertTrue(check["passed"])
        self.assertLess(check["heading_error_rad"], 0.05)


@unittest.skipUnless(HAS_IMMRAX, "jax/immrax not installed")
class TestWriteGateIntegration(unittest.TestCase):
    """End-to-end: a failed run must exit non-zero and leave no output file."""

    class _Optimizer:
        converged = True

        def __init__(self, config):
            pass

        def solve(self, *args, **kwargs):
            return _consistent_samples(), {
                "converged": self.converged, "final_cost": 1.0,
            }

    class _NonConvergedOptimizer(_Optimizer):
        converged = False
        calls = 0

        def solve(self, *args, **kwargs):
            type(self).calls += 1
            return super().solve(*args, **kwargs)

    class _BaselineSeedFailsOptimizer(_Optimizer):
        """The baseline seed stops at its budget; the other seeds do not.

        KI-048: the default candidate set is two candidates of the *same*
        seed, so a path that beats the baseline seed used to end with no
        trajectory at all.
        """

        styles = []

        def solve(self, *args, **kwargs):
            style = kwargs.get("guess_style", "baseline")
            type(self).styles.append(style)
            if style == "baseline":
                return [], {"converged": False, "final_cost": float("inf")}
            return super().solve(*args, **kwargs)

    class _WrongEndpointOptimizer(_Optimizer):
        def solve(self, *args, **kwargs):
            samples = _consistent_samples()
            samples[-1]["x"] = 1.0
            return samples, {"converged": True, "final_cost": 1.0}

    class _CpuFailingPerSegmentOptimizer(_Optimizer):
        """Per-segment candidate is shorter but violates the CPU force limit."""

        def solve(self, *args, **kwargs):
            if kwargs.get("per_segment_dt"):
                samples = _consistent_samples(n=11, dt=0.05, speed=0.2)
                for sample in samples:
                    sample["al"] = 100.0
                    sample["ar"] = 100.0
                return samples, {"converged": True, "final_cost": 0.5}
            return super().solve(*args, **kwargs)

    class _SeedFailsGateOptimizer(_Optimizer):
        """The baseline seed breaks the force limit; the other seeds do not.

        KI-048: the candidate set's two candidates share one seed, so a set
        that fails every gate is rescued by the other seeds.
        """

        styles = []

        def solve(self, *args, **kwargs):
            style = kwargs.get("guess_style", "baseline")
            type(self).styles.append(style)
            if style == "baseline":
                samples = _consistent_samples(n=11, dt=0.05, speed=0.2)
                for sample in samples:
                    sample["al"] = 100.0
                    sample["ar"] = 100.0
                return samples, {"converged": True, "final_cost": 0.5}
            return super().solve(*args, **kwargs)

    class _FailingImmrax:
        def __init__(self, config, digital_twin=None, **kwargs):
            self.digital_twin = digital_twin

        def validate_trajectory(self, samples, cof_range, torque_range,
                                backlash_range, *args, **kwargs):
            envelope = [{
                "x_min": s["x"] - 0.001, "x_max": s["x"] + 0.001,
                "y_min": s["y"] - 0.001, "y_max": s["y"] + 0.001,
            } for s in samples]
            return {
                "passed": False,
                "max_motor_violation_N": 0.079,
                "max_traction_violation_N": 0.0,
                "max_slip_l_N": 0.0,
                "max_slip_r_N": 0.0,
                "max_tracking_error_m": 0.158,
                "reachability": {
                    "max_error_m": 0.158,
                    "envelope": envelope,
                    "diverged": False,
                },
            }

    class _PassingImmrax(_FailingImmrax):
        def validate_trajectory(self, samples, cof_range, torque_range,
                                backlash_range, *args, **kwargs):
            report = super().validate_trajectory(
                samples, cof_range, torque_range, backlash_range)
            report["passed"] = True
            report["max_motor_violation_N"] = 0.0
            report["max_tracking_error_m"] = 0.002
            return report

    class _CapturingImmrax(_PassingImmrax):
        last_ranges = None

        def validate_trajectory(self, samples, *ranges, **kwargs):
            type(self).last_ranges = ranges
            return super().validate_trajectory(samples, *ranges, **kwargs)

    class _CapturingTwinImmrax(_PassingImmrax):
        last_twin = None

        def __init__(self, config, digital_twin=None, **kwargs):
            super().__init__(config, digital_twin=digital_twin)
            type(self).last_twin = digital_twin

    def setUp(self):
        import main as main_module
        self.main = main_module
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config = os.path.join(repo, "examples", "fll_choreo.json")

    def _run(self, optimizer_cls, extra_args=(), immrax_cls=None, fit_data=None):
        with tempfile.TemporaryDirectory() as tmp:
            wp = os.path.join(tmp, "wp.json")
            out = os.path.join(tmp, "out.traj")
            with open(wp, "w") as f:
                json.dump([
                    {"x": 0.0, "y": 0.0, "heading": 0.0},
                    {"x": 0.1, "y": 0.0, "heading": 0.0},
                ], f)

            args = list(extra_args)
            if fit_data is not None:
                fit_path = os.path.join(tmp, "sysid_fit.json")
                with open(fit_path, "w", encoding="utf-8") as handle:
                    json.dump(fit_data, handle)
                args.extend(["--sysid-fit", fit_path])

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.object(self.main, "TrajectoryOptimizer", optimizer_cls))
                if immrax_cls is not None:
                    stack.enter_context(
                        patch.object(self.main, "ImmraxValidator", immrax_cls))
                result = CliRunner().invoke(self.main.main, [
                    "-c", self.config, "-w", wp, "-o", out,
                    "--simple", "-q", *args,
                ])
            return result, os.path.exists(out)

    def test_non_converged_solve_writes_nothing(self):
        result, exists = self._run(self._NonConvergedOptimizer)
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(exists)
        self.assertIn("did not converge", result.output.lower())

    def test_another_seed_is_tried_when_the_first_one_stops(self):
        """KI-048: one seed that does not converge must not end the mission."""
        self._BaselineSeedFailsOptimizer.styles = []
        result, exists = self._run(self._BaselineSeedFailsOptimizer)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(exists)
        styles = self._BaselineSeedFailsOptimizer.styles
        self.assertIn("baseline", styles)
        self.assertTrue(any(style != "baseline" for style in styles),
                        f"no other seed was tried: {styles}")

    def test_a_set_that_fails_every_gate_is_rescued(self):
        """KI-048: the other seeds are tried when the set passes nothing."""
        self._SeedFailsGateOptimizer.styles = []
        result, exists = self._run(self._SeedFailsGateOptimizer,
                                   ("--validate",), self._PassingImmrax)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(exists)
        styles = self._SeedFailsGateOptimizer.styles
        self.assertTrue(any(style != "baseline" for style in styles),
                        f"no rescue seed was tried: {styles}")

    def test_the_rescue_is_bounded(self):
        """KI-048: the rescue must end, not loop."""
        self._NonConvergedOptimizer.calls = 0
        result, exists = self._run(self._NonConvergedOptimizer)
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(exists)
        # The candidate set, then one rescue per seed the set did not use.
        self.assertLessEqual(self._NonConvergedOptimizer.calls, len(GUESS_STYLES))

    def test_failed_validation_writes_nothing(self):
        result, exists = self._run(
            self._Optimizer, ("--validate",), self._FailingImmrax)
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(exists)
        self.assertIn("validation failed", result.output.lower())
        self.assertIn("tracking error", result.output.lower())

    def test_non_converged_solve_with_passing_immrax_writes_nothing(self):
        result, exists = self._run(
            self._NonConvergedOptimizer, ("--validate",), self._PassingImmrax)
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(exists)
        self.assertIn("did not converge", result.output.lower())

    def test_converged_solve_that_misses_waypoint_writes_nothing(self):
        result, exists = self._run(self._WrongEndpointOptimizer)
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(exists)
        self.assertIn("final waypoint", result.output.lower())

    def test_successful_validation_writes_trajectory(self):
        result, exists = self._run(
            self._Optimizer, ("--validate",), self._PassingImmrax)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(exists)

    def test_cpu_failure_in_one_candidate_does_not_block_a_later_pass(self):
        # Candidate 2 is shorter and passes Immrax, but its forces fail the CPU
        # audit. It must not win over candidate 1, which passes every gate.
        result, exists = self._run(
            self._CpuFailingPerSegmentOptimizer,
            ("--validate", "--candidates", "2"),
            self._PassingImmrax,
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(exists)

    def test_validation_uses_config_uncertainties(self):
        """KI-009: Immrax must receive all ranges declared by the config."""
        self._CapturingImmrax.last_ranges = None
        result, exists = self._run(
            self._Optimizer, ("--validate",), self._CapturingImmrax)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(exists)
        with open(self.config, "r", encoding="utf-8") as handle:
            cfg = RobotConfig(json.load(handle))
        self.assertEqual(self._CapturingImmrax.last_ranges, (
            tuple(cfg.uncertainties["cof"]),
            tuple(cfg.uncertainties["torque_headroom"]),
            tuple(cfg.uncertainties["gear_backlash_m"]),
            tuple(cfg.uncertainties["track_width_range"]),
            tuple(cfg.uncertainties["wheel_radius_range"]),
            tuple(cfg.uncertainties["inertia_kg_m2"]),
        ))

    def test_sysid_fit_is_passed_to_validator(self):
        self._CapturingTwinImmrax.last_twin = None
        result, exists = self._run(
            self._Optimizer,
            ("--validate",),
            self._CapturingTwinImmrax,
            fit_data={
                "actuator": {"straight": {
                    "calibrated": True,
                    "gain_min": 0.94,
                    "gain_max": 0.98,
                    "tau_s": 0.08,
                    "delay_s": 0.02,
                }},
            },
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(exists)
        twin = self._CapturingTwinImmrax.last_twin
        self.assertIsNotNone(twin)
        self.assertTrue(twin.enabled)
        self.assertEqual(twin.actuator_gain_range(), (0.94, 0.98))

    def test_sysid_fit_requires_validation_or_live(self):
        result, exists = self._run(
            self._Optimizer,
            (),
            self._PassingImmrax,
            fit_data={"odometry": {"calibrated": True}},
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(exists)
        self.assertIn("requires --validate or --live", result.output)


class _ConstantDynamics:
    """Fake embedding system returning a fixed flat derivative."""

    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def f(self, t, x_flat, u, w):
        return self.value


class _FeedForwardDynamics:
    """Fake embedding system that integrates the reference wheel speed."""

    def f(self, t, x_flat, u, w):
        v = u[3]
        return np.array([v, 0.0, 0.0, v, 0.0, 0.0])


class _TrackWidthDynamics:
    """Fake embedding system whose output width follows w[3]."""

    def f(self, t, x_flat, u, w):
        width = w[3]
        if hasattr(width, "lower"):
            lower, upper = width.lower, width.upper
        else:
            lower = upper = width
        return np.array([lower, 0.0, 0.0, upper, 0.0, 0.0])


def _samples(n=5, dt=0.1):
    return [{
        "t": i * dt, "x": 0.1 * i, "y": 0.0, "heading": 0.0,
        "vl": 0.1, "vr": 0.1,
    } for i in range(n)]


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestReachabilityEnvelope(unittest.TestCase):
    """Finding 3: envelope[i] is the box at sample i; divergence is not clamped."""

    def _validator(self, dynamics):
        # Bypass __init__ (it compiles the real embedding system)
        validator = ImmraxValidator.__new__(ImmraxValidator)
        validator.emb_sys = dynamics
        validator.cfg = SimpleNamespace(track_width=0.0965)
        return validator

    def _reach(self, dynamics):
        return self._validator(dynamics).compute_reachability(
            _samples(),
            (0.0, 0.0),            # backlash
            (0.0965, 0.0965),      # track width
            (0.028, 0.028),        # wheel radius
            (0.4, 0.4),            # cof
            (0.8, 1.0),            # torque margin
            (0.001, 0.002),        # inertia
        )

    def test_envelope_is_aligned_with_samples(self):
        report = self._reach(_ConstantDynamics(np.zeros(6)))
        self.assertEqual(len(report["envelope"]), len(_samples()))
        self.assertFalse(report["diverged"])
        self.assertFalse(any(e.get("unknown") for e in report["envelope"]))
        # First box is the 1 mm start uncertainty around the initial pose
        first = report["envelope"][0]
        self.assertAlmostEqual(first["x_min"], -0.001, places=6)
        self.assertAlmostEqual(first["x_max"], 0.001, places=6)

    def test_tracking_and_envelope_are_reported_separately(self):
        """KI-019: a small tube must not hide a large tracking error."""
        report = self._reach(_ConstantDynamics(np.zeros(6)))
        self.assertGreater(report["max_tracking_error_m"], 0.05)
        self.assertLess(report["max_envelope_radius_m"], 0.01)
        self.assertAlmostEqual(
            report["max_error_m"],
            report["max_tracking_error_m"] + report["max_envelope_radius_m"],
            places=6)

    def test_interval_reference_is_not_one_sample_behind(self):
        """KI-019: a ramp from rest must integrate over the full interval."""
        samples = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0,
             "vl": 0.0, "vr": 0.0},
            {"t": 0.1, "x": 0.025, "y": 0.0, "heading": 0.0,
             "vl": 0.5, "vr": 0.5},
            {"t": 0.2, "x": 0.075, "y": 0.0, "heading": 0.0,
             "vl": 0.5, "vr": 0.5},
            {"t": 0.3, "x": 0.125, "y": 0.0, "heading": 0.0,
             "vl": 0.5, "vr": 0.5},
        ]
        report = self._validator(_FeedForwardDynamics()).compute_reachability(
            samples,
            (0.0, 0.0),            # backlash
            (0.0965, 0.0965),      # track width
            (0.028, 0.028),        # wheel radius
            (1.37, 1.37),          # cof
            (1.0, 1.0),            # torque margin
            (0.001, 0.001),        # inertia
        )
        self.assertFalse(report["diverged"])
        self.assertLess(report["max_tracking_error_m"], 1e-6)

    def test_upper_uncertainty_bounds_widen_the_envelope(self):
        """KI-003: changing only w_upper must change the reachable set."""
        def reach(upper_track_width):
            return self._validator(_TrackWidthDynamics()).compute_reachability(
                _samples(),
                (0.0, 0.0),                    # backlash
                (0.08, upper_track_width),     # track width
                (0.028, 0.028),                # wheel radius
                (0.4, 0.4),                    # cof
                (0.8, 1.0),                    # torque margin
                (0.001, 0.002),                # inertia
            )

        narrow = reach(0.08)
        wide = reach(0.12)
        narrow_width = (narrow["envelope"][-1]["x_max"]
                        - narrow["envelope"][-1]["x_min"])
        wide_width = (wide["envelope"][-1]["x_max"]
                      - wide["envelope"][-1]["x_min"])

        self.assertGreater(wide_width, narrow_width + 0.01)

    def test_divergence_is_marked_unknown_instead_of_clamped(self):
        # Lower bound runs negative, upper positive -> width explodes after one step
        report = self._reach(_ConstantDynamics(
            [-5000.0, 0.0, 0.0, 5000.0, 0.0, 0.0]
        ))
        self.assertTrue(report["diverged"])
        envelope = report["envelope"]
        self.assertEqual(len(envelope), len(_samples()))
        self.assertFalse(envelope[0].get("unknown"))
        self.assertTrue(all(e.get("unknown") for e in envelope[1:]))
        # The true box is reported, not the old 2.5 cm clamp
        width = envelope[1]["x_max"] - envelope[1]["x_min"]
        self.assertGreater(width, 0.05)


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestOpenLoopModel(unittest.TestCase):
    """KI-019: do not slow the reference with a soft tanh speed limit."""

    def test_no_soft_speed_saturation(self):
        import jax.numpy as jnp
        from jax_robot_model import JAXRobotConfig
        from immrax_validator import RobotOpenLoop

        config = {"robot": {
            "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
            "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
            "gearing": 1.0, "cof": 1.37,
        }}
        plant = RobotOpenLoop(JAXRobotConfig(config))
        w = jnp.array([0.0, 0.0, 0.0, 0.0965, 0.028, 0.0])

        dx = plant.f(0.0, jnp.zeros(3), jnp.array([0.5, 0.5]), w)

        self.assertAlmostEqual(float(dx[0]), 0.5, places=6)
        self.assertAlmostEqual(float(dx[1]), 0.0, places=6)
        self.assertAlmostEqual(float(dx[2]), 0.0, places=6)


class TestBenchmarkValidationConfig(unittest.TestCase):
    """Finding 4: random-benchmark validation must read the real config file."""

    def test_validate_passes_config_file_to_validator(self):
        config = {"robot": {
            "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
            "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
            "gearing": 1.0, "cof": 1.37,
        }}
        with tempfile.TemporaryDirectory() as tmp:
            rb = RandomBenchmark(config, RobotConfig(config), output_dir=tmp)
            self.assertEqual(
                os.path.normpath(str(rb.config_file)),
                os.path.normpath(os.path.join(tmp, "config_used.json")),
            )
            with patch("benchmark.validate_trajectory",
                       return_value=({}, {}, {})) as mocked:
                result = rb._validate(
                    _samples(), os.path.join(tmp, "traj.traj")
                )
        self.assertEqual(result, {"metrics": {}, "audit": {}, "errors": {}})
        mocked.assert_called_once()
        args, kwargs = mocked.call_args
        self.assertEqual(os.path.normpath(args[1]),
                         os.path.normpath(rb.config_file))
        self.assertFalse(kwargs["apply_headroom"])


class TestSegmentLimits(unittest.TestCase):
    """The local envelope limit belongs to the waypoint the segment ends at.

    A waypoint's limit is the limit for the way *into* it: the robot has
    to be tight where it arrives. The old rule applied the limit of
    waypoint j to the segment that *started* at j, so a precision waypoint
    was not held to its tight limit on the approach, and the limit of the
    final waypoint was never read at all (KI-034).
    """

    def test_the_default_is_two_centimetres(self):
        self.assertAlmostEqual(main._segment_limit({}, 0), 0.02)
        self.assertAlmostEqual(main._segment_limit({}, 3), 0.02)

    def test_the_last_waypoint_limit_governs_the_last_segment(self):
        self.assertAlmostEqual(main._segment_limit({3: 0.001}, 2), 0.001)

    def test_a_precision_waypoint_limit_governs_the_approach(self):
        # The segment that ends at waypoint 1.
        self.assertAlmostEqual(main._segment_limit({1: 0.005}, 0), 0.005)

    def test_the_start_waypoint_limit_is_the_fallback(self):
        self.assertAlmostEqual(main._segment_limit({0: 0.005}, 0), 0.005)

    def test_the_report_gives_every_segment_a_limit(self):
        samples_per_segment = 2
        count = 3 * samples_per_segment + 1
        samples = [{"t": i * 0.1, "x": 0.0, "y": 0.0, "heading": 0.0,
                    "vl": 0.0, "vr": 0.0, "al": 0.0, "ar": 0.0}
                   for i in range(count)]
        report = main._segment_report(samples, samples_per_segment,
                                      {0: 0.005, 3: 0.001}, None, None)
        self.assertEqual(len(report), 3)
        self.assertAlmostEqual(report[0]["limit_m"], 0.005)
        self.assertAlmostEqual(report[1]["limit_m"], 0.02)
        # The final waypoint's limit is used, not dropped.
        self.assertAlmostEqual(report[2]["limit_m"], 0.001)


class TestGearedSpeedLimit(unittest.TestCase):
    """The wheel-speed limit must know the gearing (KI-035).

    The motor turns ``gearing`` times per wheel revolution, so the wheel's
    no-load speed is the motor's divided by it. Leaving the gearing out
    made the optimizer plan above the motor's real top speed, and every
    such plan failed the motor gate with a confusing reason.
    """

    def _config(self, gearing):
        return RobotConfig({"robot": dict(_TEST_ROBOT["robot"], gearing=gearing)})

    def test_the_wheel_speed_limit_divides_by_the_gearing(self):
        direct = self._config(1.0)
        geared = self._config(2.0)
        self.assertAlmostEqual(geared.max_linear_speed(),
                               direct.max_linear_speed() / 2.0)

    def test_the_cpu_audit_uses_the_same_limit(self):
        direct = self._config(1.0)
        geared = self._config(2.0)
        self.assertAlmostEqual(geared.no_load_wheel_speed_worst_case(),
                               direct.no_load_wheel_speed_worst_case() / 2.0)

    def test_a_geared_robot_is_not_planned_above_its_top_speed(self):
        geared = self._config(2.0)
        bound = 0.99 * geared.max_linear_speed()
        motor_omega = (bound / geared.wheel_radius) * geared.gearing
        self.assertLess(motor_omega, geared.v_max_rad_s)

    def test_a_direct_drive_robot_is_unchanged(self):
        config = self._config(1.0)
        self.assertAlmostEqual(config.max_linear_speed(),
                               config.v_max_rad_s * config.wheel_radius
                               * config.speed_headroom)


class TestEventNameEscaping(unittest.TestCase):
    """An event name is free text and must not break the generated file.

    The name was pasted into the generated Python source unescaped, so a
    name with a quote in it was a syntax error on the hub, and it could
    inject keys into the sample dict (KI-038).
    """

    def _export(self, name):
        with tempfile.TemporaryDirectory() as tmp:
            traj = os.path.join(tmp, "a.traj")
            out = os.path.join(tmp, "a.py")
            samples = [
                {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0, "vl": 0.0,
                 "vr": 0.0, "omega": 0.0, "event": name},
                {"t": 0.1, "x": 0.1, "y": 0.0, "heading": 0.0, "vl": 0.1,
                 "vr": 0.1, "omega": 0.0},
            ]
            with open(traj, "w", encoding="utf-8") as handle:
                json.dump({"trajectory": {"config": _TEST_ROBOT["robot"],
                                          "samples": samples}}, handle)
            write_python_file(traj, out)
            with open(out, "r", encoding="utf-8") as handle:
                source = handle.read()
        return source

    def test_a_quote_in_an_event_name_still_produces_valid_python(self):
        source = self._export('raise_arm" , "x": 99')
        compile(source, "a.py", "exec")

    def test_the_event_name_stays_one_string(self):
        source = self._export('raise_arm" , "x": 99')
        namespace = {}
        exec(compile(source, "a.py", "exec"), namespace)
        sample = namespace["samples"][0]
        self.assertEqual(sample["event"], 'raise_arm" , "x": 99')
        # The injected text did not become a key of its own.
        self.assertEqual(sample["x"], 0.0)

    def test_a_plain_name_is_unchanged(self):
        source = self._export("lower_arm")
        namespace = {}
        exec(compile(source, "a.py", "exec"), namespace)
        self.assertEqual(namespace["samples"][0]["event"], "lower_arm")


if __name__ == "__main__":
    unittest.main()
