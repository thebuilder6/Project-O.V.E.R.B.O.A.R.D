"""
Unit tests for optimizer.py and benchmark dry-run smoke tests.
"""

import json
import math
import os
import time
import unittest
import numpy as np
from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer, GUESS_STYLES


def _plan_deviation(samples, track_width, steps=400):
    """How far the plan is from the path its own (v, omega) profile traces.

    The profile inside one sample interval is the linear wheel speeds and the
    linear turn rate, so the heading is
    theta_k + omega_k*s + (omega_{k+1} - omega_k)*s^2/(2*dt) and the position
    is its integral. The deviation is the largest gap between the integrated
    profile and the plan's own sample positions.
    """
    x, y = samples[0]["x"], samples[0]["y"]
    worst = 0.0
    for k in range(len(samples) - 1):
        s, nxt = samples[k], samples[k + 1]
        dt = nxt["t"] - s["t"]
        if dt <= 0:
            continue
        v0 = 0.5 * (s["vl"] + s["vr"])
        v1 = 0.5 * (nxt["vl"] + nxt["vr"])
        w0 = (s["vr"] - s["vl"]) / track_width
        w1 = (nxt["vr"] - nxt["vl"]) / track_width
        sub = dt / steps
        for i in range(steps):
            f = (i + 0.5) / steps
            v = v0 + (v1 - v0) * f
            w = w0 + (w1 - w0) * f
            # the heading is the integral of the linear turn rate
            theta = s["heading"] + w0 * (f * dt) \
                + (w1 - w0) * (f * dt) ** 2 / (2.0 * dt)
            x += v * math.cos(theta) * sub
            y += v * math.sin(theta) * sub
        worst = max(worst, math.hypot(x - nxt["x"], y - nxt["y"]))
    return worst


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
        
        # Create a two-segment trajectory with different interval lengths
        N = 21
        dt_values = np.array([0.1, 0.2])
        states = np.zeros((N, 5))
        states[:, 0] = np.linspace(0, 1, N)  # x
        states[:, 3] = 0.5  # vl
        states[:, 4] = 0.5  # vr
        params = np.concatenate([dt_values, states.flatten()])
        
        waypoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        cost = optimizer._compute_cost(params, N, 10, 0.0, waypoints)
        
        # 10 intervals per segment: 10 * (0.1 + 0.2) = 3.0
        self.assertAlmostEqual(cost, 3.0)

    def test_sample_times_are_per_segment(self):
        """Timestamps accumulate each segment's own interval length."""
        times = TrajectoryOptimizer._sample_times(np.array([0.1, 0.2]), 10, 21)
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[10], 1.0)
        self.assertAlmostEqual(times[11], 1.2)
        self.assertAlmostEqual(times[20], 3.0)

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
        
        waypoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        guess = optimizer._build_initial_guess(waypoints, num_samples_per_segment=5, N=11)
        
        self.assertEqual(len(guess), 2 + 11 * 5)  # one dt per segment + N * 5 states
        self.assertTrue(np.all(guess[:2] > 0))  # segment intervals are positive
        states = guess[2:].reshape((11, 5))
        # Check that start, midpoint, and end positions match waypoints
        self.assertAlmostEqual(states[0, 0], 0.0, places=6)
        self.assertAlmostEqual(states[0, 1], 0.0, places=6)
        self.assertAlmostEqual(states[5, 0], 0.5, places=6)
        self.assertAlmostEqual(states[5, 1], 0.0, places=6)
        self.assertAlmostEqual(states[10, 0], 1.0, places=6)
        self.assertAlmostEqual(states[10, 1], 0.0, places=6)

    def test_build_initial_guess_styles_are_distinct_and_keep_waypoints(self):
        """Seed styles change the interior guess but never the waypoints."""
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

        waypoints = [(0.0, 0.0, 0.0), (0.6, 0.0, 1.2), (1.0, 0.5, None)]
        samples = 5
        N = 2 * samples + 1
        stops = [0, 1]
        guesses = {}

        for style in GUESS_STYLES:
            guess = optimizer._build_initial_guess(
                waypoints, num_samples_per_segment=samples, N=N,
                style=style, stop_waypoint_indices=stops)
            self.assertEqual(len(guess), 2 + N * 5)
            states = guess[2:].reshape((N, 5))
            self.assertTrue(np.all(np.isfinite(states)))

            # Waypoint positions are exact for every seed.
            for i, (x, y, heading) in enumerate(waypoints):
                self.assertAlmostEqual(states[i * samples, 0], x, places=6)
                self.assertAlmostEqual(states[i * samples, 1], y, places=6)
                if heading is not None:
                    self.assertAlmostEqual(states[i * samples, 2], heading, places=6)

            # Rest is required at the endpoints and at declared stops.
            for k in (0, samples, N - 1):
                self.assertEqual(states[k, 3], 0.0)
                self.assertEqual(states[k, 4], 0.0)
            guesses[style] = guess

        # Each seed must start IPOPT somewhere different.
        for i, style_a in enumerate(GUESS_STYLES):
            for style_b in GUESS_STYLES[i + 1:]:
                self.assertFalse(np.allclose(guesses[style_a], guesses[style_b]),
                                 f"{style_a} and {style_b} produced the same guess")

    def test_build_initial_guess_rejects_unknown_style(self):
        config = RobotConfig({"robot": {"mass": 0.8, "inertia": 0.002,
                                        "track_width": 0.1, "wheel_radius": 0.03,
                                        "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                                        "gearing": 1.0, "cof": 0.5}})
        optimizer = TrajectoryOptimizer(config)
        with self.assertRaises(ValueError):
            optimizer._build_initial_guess(
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 5, 6, style="nope")

    def test_turn_first_guess_with_one_sample_per_segment_keeps_endpoint(self):
        config = RobotConfig({"robot": {"mass": 0.8, "inertia": 0.002,
                                        "track_width": 0.1, "wheel_radius": 0.03,
                                        "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                                        "gearing": 1.0, "cof": 0.5}})
        optimizer = TrajectoryOptimizer(config)
        waypoints = [(0.0, 0.0, 0.0), (0.5, 0.0, 1.2), (1.0, 0.5, None)]
        guess = optimizer._build_initial_guess(
            waypoints, num_samples_per_segment=1, N=3, style="turn_first")
        states = guess[2:].reshape((3, 5))
        self.assertAlmostEqual(states[-1, 0], waypoints[-1][0], places=6)
        self.assertAlmostEqual(states[-1, 1], waypoints[-1][1], places=6)

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
        
        # Create a two-segment trajectory with different interval lengths
        N = 21
        dt_values = np.array([0.1, 0.2])
        states = np.zeros((N, 5))
        states[:, 0] = np.linspace(0, 1, N)  # x
        states[:, 3] = 0.5 + 0.01 * np.arange(N)  # vl
        states[:, 4] = 0.5 + 0.01 * np.arange(N)  # vr
        params = np.concatenate([dt_values, states.flatten()])
        
        samples = optimizer.format_output(params, N, num_samples_per_segment=10)
        
        self.assertEqual(len(samples), N)
        self.assertAlmostEqual(samples[0]["t"], 0.0)
        self.assertAlmostEqual(samples[10]["t"], 1.0)
        self.assertAlmostEqual(samples[11]["t"], 1.2)
        self.assertAlmostEqual(samples[20]["t"], 3.0)
        # Accelerations use the interval length of their own segment. Interval
        # 9 is the last of segment 0; interval 10 is the first of segment 1.
        self.assertAlmostEqual(samples[9]["al"], 0.01 / 0.1)
        self.assertAlmostEqual(samples[10]["al"], 0.01 / 0.2)
        self.assertAlmostEqual(samples[20]["al"], 0.0)
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

    def test_uniform_dt_keeps_one_interval_for_all_segments(self):
        """Default mode ties every segment to the same interval length."""
        config = RobotConfig({"robot": {
            "mass": 0.8, "inertia": 0.002, "track_width": 0.1,
            "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
            "gearing": 1.0, "cof": 0.5,
        }})
        optimizer = TrajectoryOptimizer(config)
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)]

        samples, stats = optimizer.solve(
            waypoints, num_samples_per_segment=5, accuracy_weight=0.0,
            per_segment_dt=False, verbose=False)

        self.assertEqual(len(stats["segment_dt"]), 2)
        self.assertAlmostEqual(stats["segment_dt"][0], stats["segment_dt"][1], places=9)
        self.assertAlmostEqual(samples[-1]["t"], 10 * stats["segment_dt"][0], places=6)

    def test_per_segment_dt_allocates_time_by_segment(self):
        """Opt-in per-segment timing gives each segment its own interval."""
        config = RobotConfig({"robot": {
            "mass": 0.8, "inertia": 0.002, "track_width": 0.1,
            "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
            "gearing": 1.0, "cof": 0.5,
        }})
        optimizer = TrajectoryOptimizer(config)
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)]

        samples, stats = optimizer.solve(
            waypoints, num_samples_per_segment=5, accuracy_weight=0.0,
            per_segment_dt=True, verbose=False)

        self.assertEqual(len(stats["segment_dt"]), 2)
        self.assertGreater(max(stats["segment_dt"]) / min(stats["segment_dt"]), 1.2)
        expected_t = sum(s * dt for s, dt in zip(stats["segment_samples"], stats["segment_dt"]))
        self.assertAlmostEqual(samples[-1]["t"], expected_t, places=6)

    def test_uniform_dt_forces_every_segment_to_take_the_same_time(self):
        """KI-022: one shared dt also ties the SEGMENT DURATIONS together.

        Every segment has the same sample count, so sharing one dt forces
        every segment to take the same time. On a path with unequal segment
        lengths the short segment inherits the long segment's duration. This
        is the artificial penalty that makes the equal-duration mode slow, and
        it is why the per-segment mode is the general model.
        """
        config = RobotConfig({"robot": {
            "mass": 0.8, "inertia": 0.002, "track_width": 0.1,
            "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
            "gearing": 1.0, "cof": 0.5,
        }})
        optimizer = TrajectoryOptimizer(config)
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)]

        uniform, uniform_stats = optimizer.solve(
            waypoints, num_samples_per_segment=5, accuracy_weight=0.0,
            per_segment_dt=False, verbose=False)
        per_seg, per_seg_stats = optimizer.solve(
            waypoints, num_samples_per_segment=5, accuracy_weight=0.0,
            per_segment_dt=True, verbose=False)

        # The equal-duration mode spends the same time on the 1.0 m segment
        # as on the 0.2 m segment.
        uniform_durations = [uniform[5]["t"] - uniform[0]["t"],
                             uniform[-1]["t"] - uniform[5]["t"]]
        self.assertAlmostEqual(uniform_durations[0], uniform_durations[1], places=9)

        # The per-segment mode is free to spend less time on the short segment,
        # which is why it finishes sooner on an uneven path.
        per_seg_durations = [per_seg[5]["t"] - per_seg[0]["t"],
                             per_seg[-1]["t"] - per_seg[5]["t"]]
        self.assertLess(per_seg_durations[1], per_seg_durations[0])
        self.assertLess(per_seg[-1]["t"], uniform[-1]["t"])

    def test_smoothness_cost_charges_physical_jerk_in_both_modes(self):
        """KI-022: the same physical jerk must cost the same in both modes.

        The smoothness cost used to be the raw acceleration jump, which is
        proportional to jerk*dt. With per-segment dt a short segment was
        therefore charged dt^2 less per unit of physical jerk, so the
        per-segment model could buy violent wheel motion almost for free
        (137.8 m/s^3 against the uniform model's 9.5 m/s^3 at the same
        weight). The cost is now the path-wide reference interval squared
        times the sum of the squared physical jerks, so every layout prices
        physical jerk the same way.
        """
        config = RobotConfig({"robot": {
            "mass": 0.8, "inertia": 0.002, "track_width": 0.1,
            "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
            "gearing": 1.0, "cof": 0.5,
        }})
        optimizer = TrajectoryOptimizer(config)
        num_segments, num_samples = 2, 5
        N = num_segments * num_samples + 1

        states = np.zeros((N, 5))
        for k in range(N):
            states[k, 0] = 0.1 * k
            states[k, 3] = 0.2  # vl
            states[k, 4] = 0.2  # vr
        states[1, 3] = 0.4
        states[1, 4] = 0.4  # one acceleration step, at sample 0

        def cost_for(dt_values):
            params = np.concatenate([np.asarray(dt_values, dtype=float),
                                      states.flatten()])
            return optimizer._compute_cost(params, N, num_samples, 1.0,
                                           [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

        def expected(dt_values):
            """time cost + dt_ref^2 * sum of squared physical jerks."""
            dt_ref = float(num_samples * np.sum(dt_values)) / (N - 1)
            time_cost = float(num_samples * np.sum(dt_values))
            total = 0.0
            for k in range(N - 2):
                dt_k = dt_values[k // num_samples]
                dt_k1 = dt_values[(k + 1) // num_samples]
                al_k = (states[k + 1, 3] - states[k, 3]) / dt_k
                al_k1 = (states[k + 2, 3] - states[k + 1, 3]) / dt_k1
                ar_k = (states[k + 1, 4] - states[k, 4]) / dt_k
                ar_k1 = (states[k + 2, 4] - states[k + 1, 4]) / dt_k1
                span = 0.5 * (dt_k + dt_k1)
                total += ((al_k1 - al_k) / span) ** 2 + ((ar_k1 - ar_k) / span) ** 2
            return time_cost + dt_ref ** 2 * total

        for dt_values in ([0.2, 0.2], [0.35, 0.05], [0.05, 0.35]):
            self.assertAlmostEqual(cost_for(dt_values), expected(dt_values),
                                   places=9)

        # The old cost was the raw sum of squared acceleration jumps, which
        # weighs a boundary by its own dt and so charged a short segment
        # dt^2 less per unit of physical jerk. The identity above only holds
        # with the new, layout-independent price, so it is the regression
        # guard: reverting to the raw sum breaks it for an uneven layout.
        def old_raw_cost(dt_values):
            total = 0.0
            for k in range(N - 2):
                dt_k = dt_values[k // num_samples]
                dt_k1 = dt_values[(k + 1) // num_samples]
                al_k = (states[k + 1, 3] - states[k, 3]) / dt_k
                al_k1 = (states[k + 2, 3] - states[k + 1, 3]) / dt_k1
                ar_k = (states[k + 1, 4] - states[k, 4]) / dt_k
                ar_k1 = (states[k + 2, 4] - states[k + 1, 4]) / dt_k1
                total += (al_k1 - al_k) ** 2 + (ar_k1 - ar_k) ** 2
            return total

        self.assertNotAlmostEqual(cost_for([0.35, 0.05]),
                                  old_raw_cost([0.35, 0.05]), places=6)

    def test_per_segment_objective_never_worse_than_uniform(self):
        """KI-022: per-segment dt is a strictly larger feasible set.

        Setting every dt equal recovers the uniform model exactly, so the
        per-segment optimum can never have a worse penalised objective than
        the uniform one at the same weight.
        """
        config = RobotConfig({"robot": {
            "mass": 0.8, "inertia": 0.002, "track_width": 0.1,
            "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
            "gearing": 1.0, "cof": 0.5,
        }})
        optimizer = TrajectoryOptimizer(config)
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)]
        weight = 0.02

        def objective(per_segment):
            samples, stats = optimizer.solve(
                waypoints, num_samples_per_segment=5, accuracy_weight=weight,
                per_segment_dt=per_segment, verbose=False)
            N = len(samples)
            params = np.concatenate([
                np.asarray(stats["segment_dt"], dtype=float),
                np.array([[s["x"], s["y"], s["heading"], s["vl"], s["vr"]]
                          for s in samples]).flatten()])
            cost = optimizer._compute_cost(params, N, 5, weight, waypoints)
            return samples[-1]["t"] + cost

        uniform_obj = objective(False)
        per_seg_obj = objective(True)
        self.assertLessEqual(per_seg_obj, uniform_obj + 1e-6)


class TestMergeConditions(unittest.TestCase):
    """KI-023: the boundary semantics of the waypoint pieces.

    A waypoint is a legitimate timing boundary only when the robot stops
    there AND the heading is pinned: the state there is then fully
    determined, so the pieces on either side need agree on nothing. At every
    other waypoint the pieces must stay linked, and the wheel force profile
    must run through the point without a step.
    """

    def setUp(self):
        self.config = RobotConfig({"robot": {
            "mass": 0.8, "inertia": 0.002, "track_width": 0.1,
            "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
            "gearing": 1.0, "cof": 0.5,
        }})
        self.optimizer = TrajectoryOptimizer(self.config)

    def test_merge_conditions_cover_the_four_cases(self):
        waypoints = [
            (0.0, 0.0, 0.0),    # 0: stop + pinned heading -> boundary
            (1.0, 0.0, 0.0),    # 1: pinned heading, no stop
            (2.0, 0.0, None),   # 2: stop, no pinned heading
            (3.0, 0.0, None),   # 3: neither
        ]
        stops = [0, 2]
        conds = self.optimizer.merge_conditions(waypoints, stops)

        self.assertTrue(conds[0]["boundary"])
        self.assertFalse(conds[0]["agree_velocity"])
        self.assertFalse(conds[0]["agree_heading"])

        self.assertFalse(conds[1]["boundary"])
        self.assertTrue(conds[1]["agree_velocity"])
        self.assertTrue(conds[1]["continuous_force"])

        self.assertFalse(conds[2]["boundary"])
        self.assertTrue(conds[2]["agree_heading"])
        # The wheel force has to reverse sign at a stop, so it cannot be
        # continuous there.
        self.assertFalse(conds[2]["continuous_force"])

        self.assertFalse(conds[3]["boundary"])
        self.assertTrue(conds[3]["agree_velocity"])
        self.assertTrue(conds[3]["agree_heading"])
        self.assertTrue(conds[3]["continuous_force"])

    def test_per_segment_model_keeps_the_force_continuous_through_a_linked_point(self):
        """The wheel acceleration must not step at a waypoint the robot drives
        through. With a dt change on either side the same velocity ramp is
        read as two different accelerations, which puts a step in the force
        profile exactly at the point."""
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)]
        samples, stats = self.optimizer.solve(
            waypoints, num_samples_per_segment=5, accuracy_weight=0.005,
            per_segment_dt=True, verbose=False)

        # The per-segment model does re-time across the interior waypoint.
        self.assertGreater(max(stats["segment_dt"]) / min(stats["segment_dt"]), 1.2)

        k = stats["segment_samples"][0]  # the interior waypoint sits on sample k
        self.assertLessEqual(abs(samples[k]["al"] - samples[k - 1]["al"]), 1e-6)
        self.assertLessEqual(abs(samples[k]["ar"] - samples[k - 1]["ar"]), 1e-6)

    def test_equal_duration_model_is_untouched_by_the_merge_conditions(self):
        """The equal-duration model never changes dt, so it never re-reads the
        ramp and it keeps its original formulation and result."""
        waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.0, 0.0)]
        samples, stats = self.optimizer.solve(
            waypoints, num_samples_per_segment=5, accuracy_weight=0.0,
            per_segment_dt=False, verbose=False)

        self.assertEqual(len(set(stats["segment_dt"])), 1)
        self.assertAlmostEqual(samples[-1]["t"], 10 * stats["segment_dt"][0],
                               places=9)


class TestFastPolish(unittest.TestCase):
    """The fast solve path must not change a mission that solves today."""

    def setUp(self):
        self.config = RobotConfig({"robot": {
            "mass": 0.8, "inertia": 0.002, "track_width": 0.1,
            "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
            "gearing": 1.0, "cof": 0.5,
        }})
        self.waypoints = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.4, 0.0)]

    def _solve(self, **kwargs):
        optimizer = TrajectoryOptimizer(self.config)
        return optimizer.solve(self.waypoints, num_samples_per_segment=5,
                               accuracy_weight=0.0, verbose=False, **kwargs)

    def test_fast_polish_matches_the_single_pass(self):
        """A mission that converges inside the budget keeps its trajectory."""
        _, plain = self._solve()
        samples_fast, fast = self._solve(fast_polish=True)

        self.assertTrue(plain["converged"])
        self.assertTrue(fast["converged"])
        self.assertTrue(fast["fast_polish"])
        self.assertEqual(fast["stage_count"], 1)
        self.assertAlmostEqual(fast["final_cost"], plain["final_cost"], places=6)

    def test_fast_polish_keeps_the_merge_conditions(self):
        """The two-stage path must not drop the merge conditions (KI-024)."""
        optimizer = TrajectoryOptimizer(self.config)
        samples, stats = optimizer.solve(
            self.waypoints, num_samples_per_segment=5, accuracy_weight=0.0,
            per_segment_dt=True, fast_polish=True, verbose=False)

        self.assertTrue(stats["converged"])
        k = stats["segment_samples"][0]  # the interior waypoint
        # The budgeted pass stops at IPOPT's tolerance, so the step is small
        # rather than exactly zero. The equal-duration model steps by m/s^2
        # here, so four orders of magnitude is the bar.
        self.assertLessEqual(abs(samples[k]["al"] - samples[k - 1]["al"]), 1e-4)
        self.assertLessEqual(abs(samples[k]["ar"] - samples[k - 1]["ar"]), 1e-4)

    def test_fast_polish_finishes_when_the_budget_runs_out(self):
        """With a tiny budget the cold pass takes over and still converges."""
        _, plain = self._solve()
        samples, stats = self._solve(fast_polish=True, polish_iter_budget=2)

        self.assertTrue(stats["converged"])
        self.assertGreaterEqual(stats["stage_count"], 2)
        # A two-iteration budget leaves a poor point to start from, so the
        # result may be worse than the single pass, but it must stay in the
        # same order of magnitude.
        self.assertLess(stats["final_cost"], plain["final_cost"] * 2.0)

    def test_initial_guess_is_used(self):
        """An explicit guess starts the solve from that point."""
        optimizer = TrajectoryOptimizer(self.config)
        samples, stats = optimizer.solve(
            self.waypoints, num_samples_per_segment=5, verbose=False)
        guess = TrajectoryOptimizer.guess_from_samples(samples, 2, 5)

        # Two segment intervals plus eleven states of five values.
        self.assertEqual(len(guess), 2 + 11 * 5)

        optimizer = TrajectoryOptimizer(self.config)
        _, stats = optimizer.solve(
            self.waypoints, num_samples_per_segment=5, verbose=False,
            initial_guess=guess)
        self.assertTrue(stats["converged"])

    def test_initial_guess_with_wrong_size_is_rejected(self):
        optimizer = TrajectoryOptimizer(self.config)
        with self.assertRaises(ValueError):
            optimizer.solve(self.waypoints, num_samples_per_segment=5,
                            verbose=False, initial_guess=np.zeros(3))

    def test_guess_from_samples_round_trip(self):
        """The rebuilt guess keeps the intervals and the states."""
        optimizer = TrajectoryOptimizer(self.config)
        samples, stats = optimizer.solve(
            self.waypoints, num_samples_per_segment=5, verbose=False)
        guess = TrajectoryOptimizer.guess_from_samples(samples, 2, 5)

        self.assertAlmostEqual(guess[0], stats["segment_dt"][0], places=6)
        self.assertAlmostEqual(guess[1], stats["segment_dt"][1], places=6)
        # The states follow the intervals.
        self.assertAlmostEqual(guess[2], samples[0]["x"], places=9)
        self.assertAlmostEqual(guess[2 + 4 * 5 + 3], samples[4]["vl"], places=9)

    def test_guess_from_samples_rejects_an_empty_trajectory(self):
        with self.assertRaises(ValueError):
            TrajectoryOptimizer.guess_from_samples([], 1, 5)


class TestSolverIterationBudget(unittest.TestCase):
    """Guard against the warm-start crawl coming back.

    The warm-start pass treats the initial point as a previous iterate. On a
    mission this size that makes IPOPT walk through thousands of iterations
    for an answer a cold pass reaches in a fraction of the time. Nothing in
    the interface would show that regression: the trajectory would still be
    correct, only the solve would be slow. The iteration count is the
    symptom, so it is what this test bounds.
    """

    # The reference mission: the 10-waypoint complex mission with four stops.
    MISSION = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples", "example_complex_mission.json")
    REFERENCE_COST = 6.068      # seconds, the documented trajectory time
    MAX_ITERATIONS = 1200       # measured: 354 (200 budgeted + 154 cold)
    MAX_WALL_SECONDS = 60.0     # measured: ~7 s; a smoke bound, not a benchmark

    @classmethod
    def setUpClass(cls):
        with open(cls.MISSION, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        waypoints = raw["waypoints"] if isinstance(raw, dict) and "waypoints" in raw else raw
        cls.waypoints = []
        cls.stops = []
        for i, item in enumerate(waypoints):
            if isinstance(item, dict):
                cls.waypoints.append((item["x"], item["y"], item.get("heading")))
                if item.get("stop"):
                    cls.stops.append(i)
            else:
                cls.waypoints.append((item[0], item[1], item[2] if len(item) > 2 else None))
        with open(os.path.join(os.path.dirname(cls.MISSION), "fll_choreo.json"),
                  "r", encoding="utf-8") as handle:
            cls.config = RobotConfig(json.load(handle))

    def test_fast_polish_stays_inside_the_iteration_budget(self):
        """The fast path must keep the heavy mission inside its budget."""
        optimizer = TrajectoryOptimizer(self.config)
        start = time.time()
        samples, stats = optimizer.solve(
            self.waypoints, num_samples_per_segment=10,
            stop_waypoint_indices=self.stops, verbose=False, fast_polish=True,
            self_consistent=False, proportional_samples=False, split_at_boundaries=False)
        wall = time.time() - start

        self.assertTrue(stats["converged"])
        iterations = sum(i or 0 for i in stats["stage_iterations"])
        self.assertLessEqual(
            iterations, self.MAX_ITERATIONS,
            f"the fast path used {iterations} iterations "
            f"(budget {self.MAX_ITERATIONS}); the crawl may be back")
        self.assertLess(wall, self.MAX_WALL_SECONDS,
                        f"the fast path took {wall:.1f}s")
        # The trajectory must still be as good as the one we validated. The
        # fast path finishes cold from the original guess when the budget
        # runs out (KI-047), so it converges to a different optimum: on this
        # tree it is 6.124 s against the plain pass's 6.068 s, 0.9% over.
        # It is no longer the same trajectory, so the bound is a percent and
        # not the old exact match; more than that is a regression.
        self.assertLessEqual(
            stats["final_cost"], self.REFERENCE_COST * 1.01,
            f"the fast path returned {stats['final_cost']:.4f} s, more than "
            f"1% over the reference {self.REFERENCE_COST} s")
        self.assertEqual(len(samples), 91)


# ---------------------------------------------------------------------------
# Benchmark dry-run smoke tests
# ---------------------------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark import (
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
        from overboard.models.robot_model import RobotConfig
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
        """Pipeline result must include the waypoint count and the simple solve."""
        result = run_pipeline_benchmark(self.config_data, self.cfg)

        self.assertIn("waypoints", result)
        self.assertIn("simple", result)

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


class TestSelfConsistentPlan(unittest.TestCase):
    """The plan must lie on the path its own (v, omega) profile traces.

    The trapezoidal collocation places the samples off that path, and the
    deployed controller is then told to be somewhere its own profile never
    goes. The self-consistent collocation integrates the profile instead, so
    the samples lie on the path and the headings are its tangents.
    """

    CONFIG = {
        "robot": {
            "mass": 0.8,
            "inertia": 0.002,
            "track_width": 0.1,
            "wheel_radius": 0.03,
            "v_max_rad_s": 15.0,
            "t_max_nm": 0.05,
            "gearing": 1.0,
            "cof": 0.5,
        }
    }

    WAYPOINTS = [(0.0, 0.0, 0.0), (0.5, 0.3, 0.4), (1.0, 0.0, 0.0)]

    def _solve(self, self_consistent):
        config = RobotConfig(self.CONFIG)
        optimizer = TrajectoryOptimizer(config)
        samples, _ = optimizer.solve(
            self.WAYPOINTS, num_samples_per_segment=6, verbose=False,
            self_consistent=self_consistent)
        return samples

    def test_self_consistent_plan_lies_on_its_own_profile(self):
        samples = self._solve(True)
        deviation = _plan_deviation(samples, self.CONFIG["robot"]["track_width"])
        self.assertLess(deviation, 1e-4,
                        msg="plan deviates from its own profile by "
                            f"{deviation * 1000:.3f} mm")

    def test_trapezoidal_plan_does_not(self):
        """The control: the default collocation really is inconsistent."""
        samples = self._solve(False)
        deviation = _plan_deviation(samples, self.CONFIG["robot"]["track_width"])
        self.assertGreater(deviation, 1e-3,
                           msg="expected the trapezoidal plan to deviate, got "
                               f"{deviation * 1000:.3f} mm")

    def test_both_plans_reach_the_waypoints(self):
        for flag in (False, True):
            samples = self._solve(flag)
            self.assertAlmostEqual(samples[-1]["x"], 1.0, places=2)
            self.assertAlmostEqual(samples[-1]["y"], 0.0, places=2)
            self.assertAlmostEqual(samples[0]["vl"], 0.0, places=3)
            self.assertAlmostEqual(samples[-1]["vr"], 0.0, places=3)


class TestSoftWheelForceCost(unittest.TestCase):
    """Stage 3: Soft wheel-force and traction-utilization objective penalty."""

    CONFIG = {
        "robot": {
            "mass": 0.8,
            "inertia": 0.002,
            "track_width": 0.1,
            "wheel_radius": 0.03,
            "v_max_rad_s": 15.0,
            "t_max_nm": 0.05,
            "gearing": 1.0,
            "cof": 0.5,
        }
    }

    def test_force_weight_reduces_rms_wheel_force_and_peak_traction(self):
        from overboard.verification.validator import compute_metrics
        config = RobotConfig(self.CONFIG)
        optimizer = TrajectoryOptimizer(config)
        waypoints = [(0.0, 0.0, 0.0), (0.35, 0.0, 0.0)]

        samples_zero, stats_zero = optimizer.solve(
            waypoints, num_samples_per_segment=8,
            accuracy_weight=0.0, force_weight=0.0, verbose=False,
        )
        samples_soft, stats_soft = optimizer.solve(
            waypoints, num_samples_per_segment=8,
            accuracy_weight=0.0, force_weight=2.0, verbose=False,
        )
        self.assertTrue(stats_zero["converged"])
        self.assertTrue(stats_soft["converged"])

        m_zero = compute_metrics(samples_zero, config)
        m_soft = compute_metrics(samples_soft, config)

        self.assertLess(m_soft["rms_wheel_force_n"], m_zero["rms_wheel_force_n"])
        self.assertLessEqual(
            m_soft["peak_traction_utilization"],
            m_zero["peak_traction_utilization"] + 1e-6,
        )
        self.assertGreater(m_soft["total_time_s"], m_zero["total_time_s"])


class TestPathFitnessScoring(unittest.TestCase):
    """Stage 3: 5-part time-equivalent path fitness score prioritizing precision."""

    CONFIG = {
        "robot": {
            "mass": 0.8,
            "inertia": 0.002,
            "track_width": 0.1,
            "wheel_radius": 0.03,
            "v_max_rad_s": 15.0,
            "t_max_nm": 0.05,
            "gearing": 1.0,
            "cof": 0.5,
        }
    }

    def setUp(self):
        from overboard.verification.validator import compute_path_fitness
        from overboard.main import _select_attempt
        self.compute_path_fitness = compute_path_fitness
        self.select = _select_attempt
        self.robot_cfg = RobotConfig(self.CONFIG)

    def _make_samples(self, f_wheel=0.5, duration=2.0, n=11):
        samples = []
        dt = duration / (n - 1)
        for k in range(n):
            t = k * dt
            sign = 1.0 if k < n // 2 else -1.0
            a = sign * f_wheel / (0.5 * self.robot_cfg.mass)
            v = 0.2 if 0 < k < n - 1 else 0.0
            samples.append({
                "t": t,
                "x": 0.5 * (k / (n - 1)),
                "y": 0.0,
                "heading": 0.0,
                "vl": v,
                "vr": v,
                "al": a,
                "ar": a,
                "fl": sign * f_wheel,
                "fr": sign * f_wheel,
                "reachability_envelope": {
                    "x_min": -0.002, "x_max": 0.002,
                    "y_min": -0.002, "y_max": 0.002,
                    "theta_min": -0.01, "theta_max": 0.01,
                    "tracking_error_m": 0.001,
                    "heading_error_rad": 0.005,
                },
            })
        return samples

    def test_fitness_computes_all_five_parts(self):
        samples = self._make_samples(f_wheel=0.5, duration=2.0)
        res = self.compute_path_fitness(
            samples, self.robot_cfg, stop_sample_indices=[5, 10]
        )
        self.assertIn("fitness_score", res)
        self.assertGreater(res["fitness_score"], 2.0)
        self.assertGreater(res["stop_precision_spread_m"], 0.0)
        self.assertEqual(res["slip_risk_penalty"], 0.0)  # well below 70% dead-zone
        self.assertGreater(res["rms_force_utilization"], 0.0)
        self.assertGreater(res["tracking_error_m"], 0.0)
        self.assertAlmostEqual(res["travel_time_s"], 2.0, places=6)

    def test_slip_risk_deadzone_activates_above_70_percent(self):
        # f_stall_robust = 0.05 * 0.7 / 0.031 = 1.129 N
        low_force = self._make_samples(f_wheel=0.5, duration=2.0)   # ~57% motor limit
        high_force = self._make_samples(f_wheel=0.95, duration=2.0) # ~90% motor limit
        res_low = self.compute_path_fitness(low_force, self.robot_cfg)
        res_high = self.compute_path_fitness(high_force, self.robot_cfg)
        self.assertEqual(res_low["slip_risk_penalty"], 0.0)
        self.assertGreater(res_high["slip_risk_penalty"], 0.0)
        self.assertGreater(res_high["fitness_score"], res_low["fitness_score"])

    def test_stop_precision_prioritizes_stops_over_midpath_spread(self):
        # Candidate A: wide mid-path envelope (1.5 cm), tight stop envelope (0.2 cm)
        cand_a = self._make_samples(f_wheel=0.5, duration=2.0)
        for k in range(1, len(cand_a) - 1):
            cand_a[k]["reachability_envelope"] = {
                "x_min": -0.015, "x_max": 0.015,
                "y_min": 0.0, "y_max": 0.0,
                "theta_min": -0.01, "theta_max": 0.01,
            }
        # Candidate B: uniform 0.8 cm envelope everywhere including final stop
        cand_b = self._make_samples(f_wheel=0.5, duration=2.0)
        for k in range(len(cand_b)):
            cand_b[k]["reachability_envelope"] = {
                "x_min": -0.008, "x_max": 0.008,
                "y_min": 0.0, "y_max": 0.0,
                "theta_min": -0.05, "theta_max": 0.05,
            }
        fit_a = self.compute_path_fitness(cand_a, self.robot_cfg)["fitness_score"]
        fit_b = self.compute_path_fitness(cand_b, self.robot_cfg)["fitness_score"]
        # Even though A has larger max tube (1.5 cm vs 0.8 cm), A has much tighter
        # precision at the final stop and therefore wins on fitness!
        self.assertLess(fit_a, fit_b)

        attempts = [
            {"samples": "B", "stats": {}, "cost": 2.0, "ok": True,
             "converged": True, "tube": 0.008, "duration": 2.0, "fitness": fit_b},
            {"samples": "A", "stats": {}, "cost": 2.0, "ok": True,
             "converged": True, "tube": 0.015, "duration": 2.0, "fitness": fit_a},
        ]
        best, passed = self.select(attempts)
        self.assertTrue(passed)
        # The two plans take the same time, so the envelope decides - the
        # fitness score is reported, but it does not choose.
        self.assertEqual(best["samples"], "B")



# ---------------------------------------------------------------------------
# The golden plan
#
# The numbers below are the plan the solver produced, sample by sample. They
# are the reference the strategy ledger was measured against.
#
# They are here because a rewrite of the NLP can keep the same objective and
# the same feasible set and still land in a different local optimum. The
# `opti.bounded` encoding did exactly that: the complex mission went from
# 4.96 s to 5.78 s and its trajectory by 118 mm, and every scalar check in
# the tree still passed - the fast-path test's 1% band was not tight enough
# to see it. A golden plan is checked against the whole trajectory, so a
# change of encoding shows up as a millimetre.
#
# To re-base after a deliberate change:
#   python -m tests.test_optimizer --rebase
# and say in the commit why the plan moved.
# ---------------------------------------------------------------------------

GOLDEN_MISSIONS = (
    "example_straight.json",        # pure longitudinal optimality
    "example_s_curve.json",         # the inflections and the clothoid transitions
    "example_complex_mission.json",  # the non-convex multi-waypoint case
)

# The unified path's first candidate: the settings the product plans with.
GOLDEN_SOLVE = dict(
    num_samples_per_segment=10,
    per_segment_dt=True,
    cost_metric="discretization",
    epigraph_constraints=True,
    accuracy_weight=0.001,
    guess_style="frontier_sprint",
)

# GOLDEN-BEGIN
GOLDEN_PLANS = {
    "example_straight.json": {
        "duration_s": 2.376607,
        "n": 11,
        "path_length_m": 1.000000,
        "max_force_utilization": 0.801252,
        "rms_wheel_force_n": 0.307282,
        "max_jerk_m_s3": 8.365638,
        "samples": [
            (0.0000, -0.0000, 0.0000), (0.0561, 0.0000, 0.0000), (0.1684, 0.0000, 0.0000), (0.2807, 0.0000, 0.0000),
            (0.3930, 0.0000, 0.0000), (0.5053, 0.0000, 0.0000), (0.6176, 0.0000, 0.0000), (0.7299, 0.0000, 0.0000),
            (0.8422, 0.0000, -0.0000), (0.9492, 0.0000, -0.0000), (1.0000, -0.0000, 0.0000),
        ],
    },
    "example_s_curve.json": {
        "duration_s": 3.771355,
        "n": 21,
        "path_length_m": 1.188641,
        "max_force_utilization": 0.798424,
        "rms_wheel_force_n": 0.351310,
        "max_jerk_m_s3": 25.494515,
        "samples": [
            (0.0000, 0.0000, -0.0000), (0.0327, 0.0033, 0.2036), (0.1002, 0.0292, 0.4730), (0.1715, 0.0698, 0.5411),
            (0.2446, 0.1139, 0.5436), (0.3178, 0.1581, 0.5438), (0.3909, 0.2023, 0.5440), (0.4564, 0.2447, 0.6364),
            (0.4939, 0.2834, 1.1129), (0.4966, 0.3008, 2.2999), (0.5000, 0.3000, 3.1400), (0.5071, 0.2978, 2.3195),
            (0.5080, 0.2935, 0.6734), (0.5402, 0.2899, -0.3674), (0.6079, 0.2487, -0.6280), (0.6825, 0.1941, -0.6330),
            (0.7573, 0.1393, -0.6330), (0.8321, 0.0844, -0.6328), (0.9051, 0.0337, -0.5543), (0.9705, 0.0035, -0.2380),
            (1.0000, 0.0000, 0.0000),
        ],
    },
    "example_complex_mission.json": {
        "duration_s": 5.375203,
        "n": 91,
        "path_length_m": 1.409060,
        "max_force_utilization": 0.805736,
        "rms_wheel_force_n": 0.473536,
        "max_jerk_m_s3": 102.024742,
        "samples": [
            (-0.0000, 0.0000, 0.0000), (0.0043, 0.0001, 0.0601), (0.0157, 0.0016, 0.1946), (0.0312, 0.0059, 0.3284),
            (0.0481, 0.0127, 0.4306), (0.0658, 0.0216, 0.5012), (0.0844, 0.0324, 0.5420), (0.1037, 0.0443, 0.5570),
            (0.1235, 0.0566, 0.5579), (0.1432, 0.0688, 0.5500), (0.1623, 0.0803, 0.5257), (0.1811, 0.0907, 0.4831),
            (0.2000, 0.1000, 0.4337), (0.2182, 0.1080, 0.3899), (0.2366, 0.1152, 0.3612), (0.2542, 0.1218, 0.3707),
            (0.2697, 0.1283, 0.4339), (0.2825, 0.1351, 0.5510), (0.2922, 0.1420, 0.7011), (0.2981, 0.1477, 0.8401),
            (0.3000, 0.1500, 0.9000), (0.3000, 0.1500, 0.4587), (0.3048, 0.1498, -0.2635), (0.3223, 0.1395, -0.6719),
            (0.3489, 0.1138, -0.8238), (0.3742, 0.0863, -0.8087), (0.3914, 0.0714, -0.5004), (0.3983, 0.0695, 0.2529),
            (0.3988, 0.0698, 1.0945), (0.4000, 0.0800, 1.5700), (0.3996, 0.0899, 1.6266), (0.3989, 0.1030, 1.6219),
            (0.3983, 0.1187, 1.5890), (0.3983, 0.1361, 1.5571), (0.3988, 0.1544, 1.5392), (0.3994, 0.1732, 1.5330),
            (0.4002, 0.1923, 1.5320), (0.4009, 0.2114, 1.5332), (0.4016, 0.2303, 1.5395), (0.4020, 0.2487, 1.5562),
            (0.4020, 0.2664, 1.5864), (0.4014, 0.2834, 1.6297), (0.4000, 0.3000, 1.6824), (0.3982, 0.3134, 1.7304),
            (0.3957, 0.3266, 1.7803), (0.3926, 0.3397, 1.8279), (0.3888, 0.3530, 1.8702), (0.3844, 0.3664, 1.9051),
            (0.3794, 0.3801, 1.9319), (0.3740, 0.3940, 1.9513), (0.3683, 0.4080, 1.9650), (0.3623, 0.4221, 1.9755),
            (0.3562, 0.4362, 1.9860), (0.3500, 0.4500, 2.0000), (0.3391, 0.4727, 2.0360), (0.3278, 0.4939, 2.0960),
            (0.3159, 0.5125, 2.1887), (0.3034, 0.5280, 2.3191), (0.2905, 0.5398, 2.4961), (0.2774, 0.5476, 2.7251),
            (0.2651, 0.5513, 2.9957), (0.2556, 0.5514, 3.2743), (0.2509, 0.5504, 3.4829), (0.2500, 0.5500, 3.5560),
            (0.2465, 0.5484, 3.5868), (0.2356, 0.5426, 3.6739), (0.2194, 0.5319, 3.7640), (0.2022, 0.5192, 3.7712),
            (0.1863, 0.5087, 3.6522), (0.1719, 0.5024, 3.4352), (0.1597, 0.5002, 3.2141), (0.1500, 0.5000, 3.1400),
            (0.1435, 0.4997, 3.2743), (0.1381, 0.4982, 3.5515), (0.1326, 0.4946, 3.8698), (0.1270, 0.4879, 4.1445),
            (0.1219, 0.4775, 4.3441), (0.1174, 0.4628, 4.4711), (0.1135, 0.4439, 4.5411), (0.1101, 0.4215, 4.5720),
            (0.1068, 0.3974, 4.5794), (0.1035, 0.3732, 4.5733), (0.1000, 0.3500, 4.5477), (0.0885, 0.3015, 4.3827),
            (0.0713, 0.2660, 4.1191), (0.0557, 0.2505, 3.5248), (0.0501, 0.2502, 2.4730), (0.0501, 0.2501, 1.2719),
            (0.0500, 0.2500, 0.0709), (0.0500, 0.2500, -1.0504), (0.0500, 0.2500, -1.5700),
        ],
    },
}
# GOLDEN-END


def _golden_literal():
    """The golden plans as source, for a re-base."""
    import json
    import os
    from overboard.main import _load_waypoints
    from overboard.verification.validator import compute_metrics

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "examples", "fll_choreo.json"), encoding="utf-8") as handle:
        robot = RobotConfig(json.load(handle))

    lines = ["GOLDEN_PLANS = {"]
    for mission in GOLDEN_MISSIONS:
        path = os.path.join(root, "examples", mission)
        waypoints, events, thresholds, stops = _load_waypoints(path)
        optimizer = TrajectoryOptimizer(robot)
        samples, _ = optimizer.solve(
            waypoints, stop_waypoint_indices=stops, waypoint_events=events,
            verbose=False, **GOLDEN_SOLVE)
        metrics = compute_metrics(samples, robot)
        lines.append(f'    "{mission}": {{')
        lines.append(f'        "duration_s": {samples[-1]["t"]:.6f},')
        lines.append(f'        "n": {len(samples)},')
        lines.append(f'        "path_length_m": {metrics["path_length_m"]:.6f},')
        lines.append(f'        "max_force_utilization": {metrics["max_force_utilization"]:.6f},')
        lines.append(f'        "rms_wheel_force_n": {metrics["rms_wheel_force_n"]:.6f},')
        lines.append(f'        "max_jerk_m_s3": {metrics["max_jerk_m_s3"]:.6f},')
        lines.append('        "samples": [')
        row = []
        for sample in samples:
            row.append(f'({sample["x"]:.4f}, {sample["y"]:.4f}, {sample["heading"]:.4f})')
            if len(row) == 3:
                lines.append("            " + ", ".join(row) + ",")
                row = []
        if row:
            lines.append("            " + ", ".join(row) + ",")
        lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _rebase_golden():
    """Rewrite the golden block in this file from the current solver."""
    import os
    import re

    here = os.path.abspath(__file__)
    with open(here, encoding="utf-8") as handle:
        text = handle.read()
    new_block = _golden_literal()
    pattern = re.compile(r"# GOLDEN-BEGIN\n.*?# GOLDEN-END\n", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit("the GOLDEN-BEGIN/GOLDEN-END markers are not in the file")
    text = pattern.sub(
        "# GOLDEN-BEGIN\n" + new_block + "# GOLDEN-END\n", text, count=1)
    with open(here, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("golden plans re-based in", here)


class TestGoldenPlan(unittest.TestCase):
    """The plan the solver produces must not move on its own.

    A rewrite of the NLP can keep the same objective and the same feasible set
    and still land in a different local optimum: the `opti.bounded` encoding
    moved the complex mission from 4.96 s to 5.78 s and its trajectory by
    118 mm, and every scalar check in the tree still passed. These are the
    shipped solver's plans, checked against the whole trajectory.
    """

    # 5 ms is a hundredth of the fast path's own 1% band, and the rewrite
    # that this test was written for moved the complex mission by 1.4 ms.
    DURATION_TOLERANCE_S = 0.005
    # 1 mm on the path. The rewrite moved it by 0.2 mm; the bounded
    # encoding moved it by 118 mm.
    POSITION_TOLERANCE_M = 0.001
    # 20 mrad is about 1 degree. The rewrite moved the complex mission's
    # heading by 9 mrad; the bounded encoding moved it far more.
    HEADING_TOLERANCE_RAD = 0.02
    # One percent of the motor's limit, on the force-utilisation scale.
    FORCE_UTILIZATION_TOLERANCE = 0.01

    @classmethod
    def setUpClass(cls):
        import json
        import os

        from overboard.main import _load_waypoints

        cls.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(cls.root, "examples", "fll_choreo.json"), encoding="utf-8") as handle:
            cls.config = RobotConfig(json.load(handle))
        cls._load_waypoints = staticmethod(_load_waypoints)

    def _solve(self, mission):
        path = os.path.join(self.root, "examples", mission)
        waypoints, events, thresholds, stops = self._load_waypoints(path)
        optimizer = TrajectoryOptimizer(self.config)
        samples, stats = optimizer.solve(
            waypoints, stop_waypoint_indices=stops, waypoint_events=events,
            verbose=False, **GOLDEN_SOLVE)
        self.assertTrue(stats.get("converged"), f"{mission} did not converge")
        return samples

    def test_the_plans_match_the_golden_ones(self):
        """Duration, path and demand: the plan must not have moved."""
        from overboard.verification.validator import compute_metrics

        for mission in GOLDEN_MISSIONS:
            with self.subTest(mission=mission):
                golden = GOLDEN_PLANS[mission]
                samples = self._solve(mission)

                self.assertEqual(len(samples), golden["n"],
                                 f"{mission} has {len(samples)} samples, "
                                 f"the golden plan has {golden['n']}")

                duration = samples[-1]["t"]
                self.assertLessEqual(
                    abs(duration - golden["duration_s"]), self.DURATION_TOLERANCE_S,
                    f"{mission} takes {duration:.4f} s, the golden plan takes "
                    f"{golden['duration_s']:.4f} s")

                worst_position = 0.0
                worst_heading = 0.0
                for sample, (gx, gy, gth) in zip(samples, golden["samples"]):
                    worst_position = max(
                        worst_position, math.hypot(sample["x"] - gx, sample["y"] - gy))
                    worst_heading = max(
                        worst_heading, abs(sample["heading"] - gth))
                self.assertLessEqual(
                    worst_position, self.POSITION_TOLERANCE_M,
                    f"{mission} is {worst_position * 1000:.3f} mm off the golden path")
                self.assertLessEqual(
                    worst_heading, self.HEADING_TOLERANCE_RAD,
                    f"{mission} is {worst_heading * 1000:.2f} mrad off the golden heading")

                metrics = compute_metrics(samples, self.config)
                self.assertLessEqual(
                    abs(metrics["max_force_utilization"] - golden["max_force_utilization"]),
                    self.FORCE_UTILIZATION_TOLERANCE,
                    f"{mission} peaks at {metrics['max_force_utilization']:.4f} of the "
                    f"motor limit, the golden plan at {golden['max_force_utilization']:.4f}")
                self.assertLessEqual(
                    abs(metrics["rms_wheel_force_n"] - golden["rms_wheel_force_n"]),
                    0.05,
                    f"{mission} has an RMS wheel force of "
                    f"{metrics['rms_wheel_force_n']:.4f} N, the golden plan "
                    f"{golden['rms_wheel_force_n']:.4f} N")

    def test_the_golden_plans_violate_nothing(self):
        """A golden plan that breaks the drivebase is not a plan."""
        from overboard.verification.validator import audit_constraints

        for mission in GOLDEN_MISSIONS:
            with self.subTest(mission=mission):
                samples = self._solve(mission)
                audit = audit_constraints(samples, self.config)
                for key in ("left_motor_force", "right_motor_force",
                            "left_wheel_slip", "right_wheel_slip",
                            "traction_total", "speed_violation_m_s"):
                    self.assertLessEqual(
                        audit[key], 1e-9,
                        f"{mission} violates {key}: {audit[key]}")
                self.assertEqual(audit["num_violating_samples"], 0)

if __name__ == "__main__":
    import sys

    if "--rebase" in sys.argv:
        _rebase_golden()
    else:
        unittest.main()

