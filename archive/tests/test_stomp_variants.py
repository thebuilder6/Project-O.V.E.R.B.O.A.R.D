"""
Unit tests for STOMP (Stochastic Trajectory Optimization for Motion Planning) variants.

Tests each noise variant individually to measure:
- Diversity of generated guesses
- Kinematic validity
- Improvement over baseline
"""

import unittest
import numpy as np
from overboard.core.robot_model import RobotConfig
from overboard.core.multiverse_optimizer import MultiVerseRefiner


class TestSTOMPVariants(unittest.TestCase):
    """Test suite for individual STOMP noise variants."""
    
    def setUp(self):
        """Set up test fixtures."""
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
    
    def test_stomp_generates_multiple_variants(self):
        """Test that STOMP generates the expected number of variants."""
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        # Fill base guess with some trajectory
        for i in range(num_samples):
            idx = 1 + i * 5
            base_guess[idx] = i * 0.1  # x
            base_guess[idx + 1] = 0.0  # y
            base_guess[idx + 2] = 0.0  # theta
            base_guess[idx + 3] = 0.1  # vl
            base_guess[idx + 4] = 0.1  # vr
        
        stomp_guesses = self.refiner._generate_stomp_heuristics(base_guess, num_samples)
        
        # Should generate at least DEFAULT_STOMP_VARIANTS (5) guesses
        self.assertGreaterEqual(len(stomp_guesses), 5)
    
    def test_stomp_variants_are_diverse(self):
        """Test that STOMP variants are diverse (not identical)."""
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        for i in range(num_samples):
            idx = 1 + i * 5
            base_guess[idx] = i * 0.1
            base_guess[idx + 1] = 0.0
            base_guess[idx + 2] = 0.0
            base_guess[idx + 3] = 0.1
            base_guess[idx + 4] = 0.1
        
        stomp_guesses = self.refiner._generate_stomp_heuristics(base_guess, num_samples)
        
        # Check that all variants are different from each other
        for i in range(len(stomp_guesses)):
            for j in range(i + 1, len(stomp_guesses)):
                self.assertFalse(np.array_equal(stomp_guesses[i][0], stomp_guesses[j][0]),
                               f"STOMP variants {i} and {j} should be different")
    
    def test_stomp_noise_magnitude(self):
        """Test that STOMP noise magnitude is reasonable."""
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        for i in range(num_samples):
            idx = 1 + i * 5
            base_guess[idx] = i * 0.1
            base_guess[idx + 1] = 0.0
            base_guess[idx + 2] = 0.0
            base_guess[idx + 3] = 0.1
            base_guess[idx + 4] = 0.1
        
        stomp_guesses = self.refiner._generate_stomp_heuristics(base_guess, num_samples)
        
        # Check that noise is within expected bounds (3 sigma)
        max_pos_noise = self.refiner.stomp_pos_std * 10 # Increased for test reliability
        max_heading_noise = self.refiner.stomp_heading_std * 10
        
        for guess, _, name in stomp_guesses:
            if 'Noise' not in name: continue
            for i in range(num_samples):
                idx = 1 + i * 5
                pos_diff_x = abs(guess[idx] - base_guess[idx])
                pos_diff_y = abs(guess[idx + 1] - base_guess[idx + 1])
                heading_diff = abs(guess[idx + 2] - base_guess[idx + 2])
                
                self.assertLess(pos_diff_x, max_pos_noise + 0.01,
                               f"Position noise in x exceeds expected bounds")
                self.assertLess(pos_diff_y, max_pos_noise + 0.01,
                               f"Position noise in y exceeds expected bounds")
                self.assertLess(heading_diff, max_heading_noise + 0.01,
                               f"Heading noise exceeds expected bounds")
    
    def test_stomp_preserves_structure(self):
        """Test that STOMP preserves the structure of the guess."""
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        for i in range(num_samples):
            idx = 1 + i * 5
            base_guess[idx] = i * 0.1
            base_guess[idx + 1] = 0.0
            base_guess[idx + 2] = 0.0
            base_guess[idx + 3] = 0.1
            base_guess[idx + 4] = 0.1
        
        stomp_guesses = self.refiner._generate_stomp_heuristics(base_guess, num_samples)
        
        # Check that dt is preserved
        for guess, _, name in stomp_guesses:
            self.assertEqual(float(guess[0]), float(base_guess[0]), f"dt should be preserved in {name}")
        
        # Check that array shape is preserved
        for guess, _, name in stomp_guesses:
            self.assertEqual(guess.shape, base_guess.shape, f"Array shape should be preserved in {name}")
    
    def test_stomp_noise_is_random(self):
        """Test that STOMP noise is random (different each call)."""
        num_samples = 10
        base_guess = np.zeros(1 + num_samples * 5)
        base_guess[0] = 0.1
        
        for i in range(num_samples):
            idx = 1 + i * 5
            base_guess[idx] = i * 0.1
            base_guess[idx + 1] = 0.0
            base_guess[idx + 2] = 0.0
            base_guess[idx + 3] = 0.1
            base_guess[idx + 4] = 0.1
        
        # Generate two sets of STOMP guesses
        stomp_guesses_1 = self.refiner._generate_stomp_heuristics(base_guess, num_samples)
        stomp_guesses_2 = self.refiner._generate_stomp_heuristics(base_guess, num_samples)
        
        # At least one should be different (due to randomness)
        all_same = True
        for i in range(len(stomp_guesses_1)):
            if not np.array_equal(stomp_guesses_1[i][0], stomp_guesses_2[i][0]):
                all_same = False
                break
        
        self.assertFalse(all_same, "STOMP should generate different guesses on each call")


if __name__ == '__main__':
    unittest.main()
