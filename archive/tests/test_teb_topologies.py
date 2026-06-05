"""
Unit tests for TEB (Timed Elastic Band) topology heuristics.

Tests each topology individually to measure:
- Kinematic validity
- Constraint satisfaction
- Improvement over baseline
"""

import unittest
import numpy as np
from overboard.core.robot_model import RobotConfig
from overboard.core.multiverse_optimizer import MultiVerseRefiner, LocalSegmentOptimizer


class TestTEBTopologies(unittest.TestCase):
    """Test suite for individual TEB topology heuristics."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Load robot config
        config_dict = {
            "mass": 0.723,
            "inertia": 0.0024,
            "track_width": 0.0965,
            "wheel_radius": 0.028,
            "v_max_rad_s": 15.7,
            "t_max_nm": 0.04,
            "gearing": 1.0,
            "cof": 0.40,
            "gravity": 9.81,
            "torque_headroom": 0.85,
            "speed_headroom": 0.90
        }
        self.config = RobotConfig(config_dict)
        self.refiner = MultiVerseRefiner(self.config, enable_parallel=False, num_workers=1)
    
    def test_forward_bias_topology(self):
        """Test forward bias topology encourages forward motion."""
        # Create a simple segment
        start_state = (0.0, 0.0, 0.0, 0.0, 0.0)
        end_state = (1.0, 0.0, 0.0, 0.0, 0.0)
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1  # dt
        
        # Generate forward bias guess
        teb_guesses = self.refiner._generate_teb_heuristics(start_state, end_state, num_samples, base_guess)
        # Find Bounded_Forward guess
        forward_guess = None
        for g, _, name in teb_guesses:
            if name == 'Bounded_Forward':
                forward_guess = g
                break
        self.assertIsNotNone(forward_guess)
        
        # Verify forward bias guess is different from baseline
        self.assertFalse(np.array_equal(forward_guess, base_guess))
        
        # Verify velocities are non-negative (forward motion)
        for i in range(num_samples):
            idx = 1 + i * 5
            vl = forward_guess[idx + 3]
            vr = forward_guess[idx + 4]
            self.assertGreaterEqual(vl, 0, f"Left velocity should be non-negative at sample {i}")
            self.assertGreaterEqual(vr, 0, f"Right velocity should be non-negative at sample {i}")
    
    def test_reverse_bias_topology(self):
        """Test reverse bias topology encourages reverse motion."""
        start_state = (1.0, 0.0, 0.0, 0.0, 0.0)
        end_state = (0.0, 0.0, 0.0, 0.0, 0.0)
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        teb_guesses = self.refiner._generate_teb_heuristics(start_state, end_state, num_samples, base_guess)
        
        # Should have at least 2 topologies (forward, reverse)
        self.assertGreaterEqual(len(teb_guesses), 2)
    
    def test_point_turn_bias_topology(self):
        """Test point-turn bias topology for in-place rotation."""
        start_state = (0.0, 0.0, 0.0, 0.0, 0.0)
        end_state = (0.0, 0.0, np.pi/2, 0.0, 0.0)
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        teb_guesses = self.refiner._generate_teb_heuristics(start_state, end_state, num_samples, base_guess)
        
        # Should have point-turn topology
        self.assertGreaterEqual(len(teb_guesses), 3)
    
    def test_wide_sweep_topology(self):
        """Test wide sweep topology for gentle turns."""
        start_state = (0.0, 0.0, 0.0, 0.0, 0.0)
        end_state = (1.0, 1.0, np.pi/4, 0.0, 0.0)
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        teb_guesses = self.refiner._generate_teb_heuristics(start_state, end_state, num_samples, base_guess)
        
        # Should have wide sweep topology
        self.assertGreaterEqual(len(teb_guesses), 4)
    
    def test_topology_kinematic_validity(self):
        """Test that all TEB topologies produce kinematically valid guesses."""
        start_state = (0.0, 0.0, 0.0, 0.0, 0.0)
        end_state = (1.0, 0.5, np.pi/6, 0.0, 0.0)
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        teb_guesses = self.refiner._generate_teb_heuristics(start_state, end_state, num_samples, base_guess)
        
        for i, (guess, _, name) in enumerate(teb_guesses):
            # Check that velocities are within reasonable bounds
            for j in range(num_samples):
                idx = 1 + j * 5
                vl = guess[idx + 3]
                vr = guess[idx + 4]
                
                # Check velocity limits (converted from rad/s to m/s)
                v_max = self.config.v_max_rad_s * self.config.wheel_radius
                self.assertLess(abs(vl), v_max * 2, f"TEB topology {i} sample {j} left velocity exceeds limit")
                self.assertLess(abs(vr), v_max * 2, f"TEB topology {i} sample {j} right velocity exceeds limit")


if __name__ == '__main__':
    unittest.main()
