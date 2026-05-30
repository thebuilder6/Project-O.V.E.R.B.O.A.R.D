"""
Updated Unit tests for JAX-based Multi-Verse heuristics.
"""

import unittest
import jax
import jax.numpy as jnp
import numpy as np
from robot_model import RobotConfig
from jax_robot_model import JAXRobotConfig
from multiverse_optimizer import MasterTrajectoryOptimizer
from jax_optimizer import generate_candidates_jax

class TestJAXHeuristics(unittest.TestCase):
    def setUp(self):
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
        self.jax_config = JAXRobotConfig(config_dict)
        self.optimizer = MasterTrajectoryOptimizer(self.config, enable_parallel=False)

    def test_jax_candidate_generation(self):
        """Test that JAX generates diverse candidates in parallel."""
        s_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        e_state = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        key = jax.random.PRNGKey(42)

        guesses, costs, biases = generate_candidates_jax(
            s_state, e_state, 10, 50, self.jax_config, key
        )

        # (4 compass seeds * 3 velocity modes) + 50 STOMP perturbations = 62
        # 1 (dt) + 10 (samples) * 5 (states) = 51
        self.assertEqual(guesses.shape, (62, 51))
        self.assertEqual(costs.shape, (62,))

        # Check for diversity in costs
        self.assertGreater(jnp.std(costs), 0.0)

    def test_stomp_noise_jax(self):
        """Verify that STOMP noise is correctly applied in JAX."""
        s_state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        e_state = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        key = jax.random.PRNGKey(123)

        guesses, _, _ = generate_candidates_jax(
            s_state, e_state, 10, 10, self.jax_config, key
        )

        # Guesses should not be identical
        self.assertFalse(jnp.allclose(guesses[0], guesses[1]))

if __name__ == '__main__':
    unittest.main()
