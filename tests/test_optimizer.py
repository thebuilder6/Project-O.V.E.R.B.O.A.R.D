"""
Unit tests for optimizer.py and benchmark dry-run smoke tests.
"""

import unittest
import numpy as np
from overboard.core.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer


class TestTrajectoryOptimizer(unittest.TestCase):
    """Test cases for TrajectoryOptimizer class."""

    def test_init(self):
        """Test optimizer initialization."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        self.assertEqual(optimizer.config, config)
        self.assertIsNotNone(optimizer.model)
        self.assertEqual(optimizer.iteration_history, [])

    def test_solve_simple_straight_line(self):
        """Test optimization for simple straight line trajectory."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Simple straight line: (0, 0, 0) to (1, 0, 0)
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        samples, stats = optimizer.solve(waypoints, num_samples_per_segment=5, 
                                  accuracy_weight=0.0, verbose=False)
        
        self.assertGreater(len(samples), 0)
        self.assertAlmostEqual(samples[0]["x"], 0.0, places=6)
        self.assertAlmostEqual(samples[0]["y"], 0.0, places=6)
        self.assertAlmostEqual(samples[-1]["x"], 1.0, places=6)
        self.assertAlmostEqual(samples[-1]["y"], 0.0, places=6)
        # Start and end at rest
        self.assertAlmostEqual(samples[0]["vl"], 0.0, places=6)
        self.assertAlmostEqual(samples[0]["vr"], 0.0, places=6)
        self.assertAlmostEqual(samples[-1]["vl"], 0.0, places=6)
        self.assertAlmostEqual(samples[-1]["vr"], 0.0, places=6)

    def test_solve_with_heading_constraint(self):
        """Test optimization with heading constraints."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Path with heading constraints
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, np.pi/4)]
        
        samples, stats = optimizer.solve(waypoints, num_samples_per_segment=5,
                                  accuracy_weight=0.0, verbose=False)
        
        self.assertGreater(len(samples), 0)
        self.assertAlmostEqual(samples[0]["heading"], 0.0, places=6)
        # Check that final heading is close to constraint
        final_heading = (samples[-1]["heading"] + np.pi) % (2 * np.pi) - np.pi
        expected_heading = (np.pi/4 + np.pi) % (2 * np.pi) - np.pi
        self.assertAlmostEqual(final_heading, expected_heading, places=2)

    def test_solve_without_heading_constraint(self):
        """Test optimization without heading constraints."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Path with unconstrained headings
        waypoints = [(0.0, 0.0, None), (1.0, 0.5, None)]
        
        samples, stats = optimizer.solve(waypoints, num_samples_per_segment=5,
                                  accuracy_weight=0.0, verbose=False)
        
        self.assertGreater(len(samples), 0)
        self.assertAlmostEqual(samples[0]["x"], 0.0, places=6)
        self.assertAlmostEqual(samples[-1]["x"], 1.0, places=6)

    def test_solve_with_stop_waypoint(self):
        """Test optimization with stop waypoint constraint."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Path with stop at middle waypoint
        waypoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        samples, stats = optimizer.solve(waypoints, num_samples_per_segment=5,
                                  accuracy_weight=0.0, stop_waypoint_indices=[1],
                                  verbose=False)
        
        self.assertGreater(len(samples), 0)
        # Check that robot stops at waypoint index 1 (sample index 5)
        stop_idx = 5
        self.assertAlmostEqual(samples[stop_idx]["vl"], 0.0, places=3)
        self.assertAlmostEqual(samples[stop_idx]["vr"], 0.0, places=3)

    def test_solve_with_event_markers(self):
        """Test optimization with event markers."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Path with event marker
        waypoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        waypoint_events = {1: "lower_arm"}
        
        samples, stats = optimizer.solve(waypoints, num_samples_per_segment=5,
                                  accuracy_weight=0.0, waypoint_events=waypoint_events,
                                  verbose=False)
        
        self.assertGreater(len(samples), 0)
        # Check that event is present at waypoint index 1 (sample index 5)
        event_idx = 5
        self.assertIn("event", samples[event_idx])
        self.assertEqual(samples[event_idx]["event"], "lower_arm")

    def test_solve_with_accuracy_weight(self):
        """Test optimization with accuracy/smoothness weight."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        # Solve with accuracy weight
        samples_smooth, _ = optimizer.solve(waypoints, num_samples_per_segment=5,
                                         accuracy_weight=1.0, verbose=False)
        
        # Solve without accuracy weight
        samples_fast, _ = optimizer.solve(waypoints, num_samples_per_segment=5,
                                       accuracy_weight=0.0, verbose=False)
        
        self.assertGreater(len(samples_smooth), 0)
        self.assertGreater(len(samples_fast), 0)
        # Smooth trajectory should have different characteristics
        # (we can't easily compare jerk without more analysis)

    def test_compute_cost(self):
        """Test cost computation."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Create simple trajectory parameters
        N = 11
        dt = 0.1
        states = np.zeros((N, 5))
        states[:, 0] = np.linspace(0, 1, N)  # x
        states[:, 3] = 0.5  # vl
        states[:, 4] = 0.5  # vr
        params = np.concatenate([[dt], states.flatten()])
        
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        cost = optimizer._compute_cost(params, N, 10, 0.0, waypoints)
        
        # Cost should be positive (time cost)
        self.assertGreater(cost, 0)

    def test_build_initial_guess(self):
        """Test initial guess generation."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        guess = optimizer._build_initial_guess(waypoints, num_samples_per_segment=5, N=6)
        
        self.assertEqual(len(guess), 1 + 6 * 5)  # dt + N * 5 states
        self.assertGreater(guess[0], 0)  # dt should be positive
        # Check that start and end positions match waypoints
        self.assertAlmostEqual(guess[1], 0.0, places=6)  # x[0]
        self.assertAlmostEqual(guess[2], 0.0, places=6)  # y[0]
        self.assertAlmostEqual(guess[1 + 5 * 5], 1.0, places=6)  # x[5]
        self.assertAlmostEqual(guess[2 + 5 * 5], 0.0, places=6)  # y[5]

    def test_format_output(self):
        """Test output formatting."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Create simple trajectory parameters
        N = 11
        dt = 0.1
        states = np.zeros((N, 5))
        states[:, 0] = np.linspace(0, 1, N)  # x
        states[:, 3] = 0.5  # vl
        states[:, 4] = 0.5  # vr
        params = np.concatenate([[dt], states.flatten()])
        
        samples = optimizer.format_output(params, N, num_samples_per_segment=10)
        
        self.assertEqual(len(samples), N)
        # Check required fields
        self.assertIn("t", samples[0])
        self.assertIn("x", samples[0])
        self.assertIn("y", samples[0])
        self.assertIn("heading", samples[0])
        self.assertIn("vl", samples[0])
        self.assertIn("vr", samples[0])
        self.assertIn("omega", samples[0])
        self.assertIn("al", samples[0])
        self.assertIn("ar", samples[0])
        self.assertIn("fl", samples[0])
        self.assertIn("fr", samples[0])

    def test_iteration_history_capture(self):
        """Test iteration history capture."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        samples, stats = optimizer.solve(waypoints, num_samples_per_segment=5,
                                  accuracy_weight=0.0, capture_iterations=True,
                                  verbose=False)
        
        # Should have captured initial guess and final solution
        self.assertGreaterEqual(len(optimizer.iteration_history), 2)
        self.assertEqual(optimizer.iteration_history[0]["phase"], "initial_guess")
        # Note: may be 'failed_solution' if optimization doesn't converge
        self.assertIn(optimizer.iteration_history[-1]["phase"], ["final_solution", "failed_solution"])

    def test_solve_multiple_segments(self):
        """Test optimization with multiple segments."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5
            }
        }
        config = RobotConfig(config_dict)
        optimizer = TrajectoryOptimizer(config)
        
        # Multi-segment path
        waypoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.5, np.pi/4)]
        
        samples, stats = optimizer.solve(waypoints, num_samples_per_segment=5,
                                  accuracy_weight=0.0, verbose=False)
        
        self.assertGreater(len(samples), 0)
        # Check waypoint constraints
        self.assertAlmostEqual(samples[0]["x"], 0.0, places=6)
        self.assertAlmostEqual(samples[5]["x"], 0.5, places=6)
        self.assertAlmostEqual(samples[10]["x"], 1.0, places=6)
        self.assertAlmostEqual(samples[10]["y"], 0.5, places=6)


# ---------------------------------------------------------------------------
# Benchmark dry-run smoke tests
# ---------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.benchmark import (
    STANDARD_CONFIG,
    CLIBenchmarkRunner,
    RandomBenchmark,
    run_pipeline_benchmark,
)


class TestBenchmarkDryRun(unittest.TestCase):
    """Smoke tests for benchmark.py that run fully in-process (no subprocess)."""

    # Shared config so we only build it once per test class
    @classmethod
    def setUpClass(cls):
        from overboard.core.robot_model import RobotConfig
        cls.config_data = STANDARD_CONFIG
        cls.cfg = RobotConfig(STANDARD_CONFIG)

    # ------------------------------------------------------------------
    def test_pipeline_benchmark_simple_only(self):
        """
        Pipeline benchmark must return a result dict with a 'simple' key
        containing positive solve_time and traj_time.
        """
        result = run_pipeline_benchmark(self.config_data, self.cfg)

        self.assertIn("simple", result)
        self.assertGreater(result["simple"]["solve_time"], 0)
        self.assertGreater(result["simple"]["traj_time"], 0)

    # ------------------------------------------------------------------
    def test_pipeline_benchmark_result_keys(self):
        """Pipeline result must include 'waypoints' and 'multiverse' keys."""
        result = run_pipeline_benchmark(self.config_data, self.cfg)

        self.assertIn("waypoints", result)
        # 'multiverse' key must exist (value may be None if JAX absent)
        self.assertIn("multiverse", result)

    # ------------------------------------------------------------------
    def test_cli_aggregate_validation_all_pass(self):
        """
        _aggregate_validation with all-passing runs should return pass_rate=1.0
        and averages close to zero for a near-perfect trajectory.
        """
        perfect_v = {
            "errors": {
                "max_pos_error_m": 0.0001,
                "rms_pos_error_m": 0.00005,
                "max_heading_error_rad": 0.0001,
                "final_pos_error_m": 0.0001,
            },
            "audit": {
                "num_violating_samples": 0,
                "num_slip_points": 0,
                "left_wheel_slip": 0.0,
                "right_wheel_slip": 0.0,
            },
        }
        results = [perfect_v, perfect_v]
        agg = CLIBenchmarkRunner._aggregate_validation(results)

        self.assertIsNotNone(agg)
        self.assertAlmostEqual(agg["pass_rate"], 1.0)
        self.assertAlmostEqual(agg["avg_num_violating_samples"], 0.0)
        self.assertAlmostEqual(agg["avg_num_slip_points"], 0.0)

    # ------------------------------------------------------------------
    def test_cli_aggregate_validation_mixed(self):
        """
        _aggregate_validation with one pass / one fail should return pass_rate=0.5.
        """
        passing = {
            "errors": {"max_pos_error_m": 0.001, "rms_pos_error_m": 0.0005,
                       "max_heading_error_rad": 0.001, "final_pos_error_m": 0.001},
            "audit": {"num_violating_samples": 0, "num_slip_points": 0,
                      "left_wheel_slip": 0.0, "right_wheel_slip": 0.0},
        }
        failing = {
            "errors": {"max_pos_error_m": 0.05, "rms_pos_error_m": 0.03,
                       "max_heading_error_rad": 0.1, "final_pos_error_m": 0.05},
            "audit": {"num_violating_samples": 2, "num_slip_points": 1,
                      "left_wheel_slip": 0.5, "right_wheel_slip": 0.5},
        }
        agg = CLIBenchmarkRunner._aggregate_validation([passing, failing])

        self.assertAlmostEqual(agg["pass_rate"], 0.5)
        self.assertGreater(agg["avg_max_pos_error_m"], 0)

    # ------------------------------------------------------------------
    def test_cli_aggregate_validation_none_entries(self):
        """_aggregate_validation must return None when all entries are None."""
        result = CLIBenchmarkRunner._aggregate_validation([None, None])
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    def test_random_benchmark_generate_waypoints_short_sprint(self):
        """short_sprint profile must always produce exactly 3 waypoints."""
        rb = RandomBenchmark(self.config_data, self.cfg)
        for _ in range(5):
            wps_json, wps_list, stops = rb.generate_waypoints("short_sprint")
            self.assertEqual(len(wps_json), 3)
            self.assertEqual(len(wps_list), 3)
            # First and last must be stops
            self.assertIn(0, stops)
            self.assertIn(2, stops)

    # ------------------------------------------------------------------
    def test_random_benchmark_generate_waypoints_complex(self):
        """complex profile must produce between 5 and 8 waypoints."""
        rb = RandomBenchmark(self.config_data, self.cfg)
        for _ in range(5):
            wps_json, wps_list, stops = rb.generate_waypoints("complex")
            self.assertGreaterEqual(len(wps_json), 5)
            self.assertLessEqual(len(wps_json), 8)
            self.assertEqual(len(wps_list), len(wps_json))

    # ------------------------------------------------------------------
    def test_random_benchmark_generate_waypoints_stress_test(self):
        """stress_test profile must produce between 4 and 6 waypoints."""
        rb = RandomBenchmark(self.config_data, self.cfg)
        for _ in range(5):
            wps_json, wps_list, stops = rb.generate_waypoints("stress_test")
            self.assertGreaterEqual(len(wps_json), 4)
            self.assertLessEqual(len(wps_json), 6)

    # ------------------------------------------------------------------
    def test_benchmark_simple_solver_produces_valid_trajectory(self):
        """
        In-process simple solve (same fixture as pipeline) must produce samples
        that start and end at rest and have monotonically increasing time.
        """
        from overboard.core.optimizer import TrajectoryOptimizer
        opt = TrajectoryOptimizer(self.cfg)
        wps = [(0, 0, 0), (1, 0, 0), (2, 1, np.pi / 2)]
        samples, stats = opt.solve(wps, num_samples_per_segment=5, verbose=False)

        self.assertGreater(len(samples), 0)
        # Monotone time
        times = [s["t"] for s in samples]
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i - 1])
        # Start / end at rest
        self.assertAlmostEqual(samples[0]["vl"], 0.0, places=3)
        self.assertAlmostEqual(samples[0]["vr"], 0.0, places=3)
        self.assertAlmostEqual(samples[-1]["vl"], 0.0, places=3)
        self.assertAlmostEqual(samples[-1]["vr"], 0.0, places=3)


if __name__ == "__main__":
    unittest.main()

