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

    def test_jax_config_matches_legacy_robot_config(self):
        """KI-008/KI-009: JAX must parse the same robot and ranges."""
        legacy = {
            "config": {
                "mass": {"val": 1.23},
                "inertia": {"val": 0.004},
                "differentialTrackWidth": {"val": 0.2},
                "radius": {"val": 0.04},
                "vmax": {"val": 10.0},
                "tmax": {"val": 0.08},
                "gearing": {"val": 1.0},
                "cof": {"val": 0.9},
                "gravity": {"val": 9.81},
                "torqueHeadroom": {"val": 0.8},
                "speedHeadroom": {"val": 0.85},
                "uncertainties": {
                    "cof": {"val": [0.8, 1.2]},
                    "track_width_range": [0.19, 0.21],
                    "wheel_radius_range": [0.039, 0.041],
                    "inertia_kg_m2": [0.003, 0.005],
                    "torque_headroom": [0.7, 0.9],
                    "gear_backlash_m": [0.0002, 0.0003],
                },
            }
        }
        rc = RobotConfig(legacy)
        jc = JAXRobotConfig(legacy)

        for attr in ("mass", "inertia", "track_width", "wheel_radius",
                     "v_max_rad_s", "t_max_nm", "gearing", "cof", "g",
                     "torque_headroom", "speed_headroom"):
            self.assertEqual(getattr(jc, attr), getattr(rc, attr), attr)

        self.assertEqual(jc.track_width_range, (0.19, 0.21))
        self.assertEqual(jc.wheel_radius_range, (0.039, 0.041))
        self.assertEqual(jc.inertia_range, (0.003, 0.005))
        self.assertEqual(jc.cof_range, (0.8, 1.2))
        self.assertEqual(jc.torque_headroom_range, (0.7, 0.9))
        self.assertEqual(jc.backlash_range, (0.0002, 0.0003))

        # A parsed RobotConfig is accepted directly, avoiding double parsing.
        self.assertEqual(JAXRobotConfig(rc).track_width, rc.track_width)

    def test_jax_config_reads_nested_uncertainties(self):
        """KI-009: canonical robot.uncertainties must reach the JAX config."""
        config = {"robot": {
            "cof": 1.37,
            "inertia": 0.001,
            "track_width": 0.0965,
            "wheel_radius": 0.028,
            "uncertainties": {
                "cof": [1.24, 1.51],
                "torque_headroom": [0.75, 0.9],
                "gear_backlash_m": [0.0002, 0.0007],
                "track_width_range": [0.096, 0.097],
                "wheel_radius_range": [0.0275, 0.0285],
                "inertia_kg_m2": [0.0008, 0.0011],
            },
        }}
        jc = JAXRobotConfig(config)

        self.assertEqual(jc.cof_range, (1.24, 1.51))
        self.assertEqual(jc.torque_headroom_range, (0.75, 0.9))
        self.assertEqual(jc.backlash_range, (0.0002, 0.0007))
        self.assertEqual(jc.track_width_range, (0.096, 0.097))
        self.assertEqual(jc.wheel_radius_range, (0.0275, 0.0285))
        self.assertEqual(jc.inertia_range, (0.0008, 0.0011))

if __name__ == '__main__':
    unittest.main()
