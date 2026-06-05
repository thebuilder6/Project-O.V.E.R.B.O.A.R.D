"""
Unit tests for TrajectoryCritic metrics.

Tests each anomaly detection metric individually:
- Tortuosity (path length / straight-line distance)
- Yaw excess (heading changes)
- Velocity chattering (rapid velocity changes)
"""

import unittest
import numpy as np
from overboard.core.robot_model import RobotConfig
from overboard.core.multiverse_optimizer import TrajectoryCritic


class TestCriticMetrics(unittest.TestCase):
    """Test suite for individual critic metrics."""
    
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
        self.critic = TrajectoryCritic(self.config)
    
    def test_tortuosity_straight_line(self):
        """Test tortuosity metric for a straight line (should be ~1.0)."""
        trajectory = []
        for i in range(10):
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        tortuosity = self.critic._compute_tortuosity(trajectory)
        
        # Straight line should have tortuosity close to 1.0
        self.assertAlmostEqual(tortuosity, 1.0, places=2,
                           msg="Straight line should have tortuosity ~1.0")
    
    def test_tortuosity_zigzag(self):
        """Test tortuosity metric for a zigzag path (should be > 1.0)."""
        trajectory = []
        for i in range(10):
            y = 0.1 if i % 2 == 0 else -0.1
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': y,
                'heading': 0.0,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        tortuosity = self.critic._compute_tortuosity(trajectory)
        
        # Zigzag should have tortuosity > 1.0
        self.assertGreater(tortuosity, 1.0,
                          msg="Zigzag path should have tortuosity > 1.0")
    
    def test_yaw_excess_constant_heading(self):
        """Test yaw excess for constant heading (should be 0)."""
        trajectory = []
        for i in range(10):
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        yaw_excess = self.critic._compute_yaw_excess(trajectory)
        
        # Constant heading should have zero yaw excess
        self.assertEqual(yaw_excess, 0.0,
                        msg="Constant heading should have zero yaw excess")
    
    def test_yaw_excess_turning(self):
        """Test yaw excess for turning path (should be > 0)."""
        trajectory = []
        for i in range(10):
            heading = i * 0.1  # Gradual turn
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': heading,
                'vl': 0.1,
                'vr': 0.2,
                'omega': (0.2 - 0.1) / self.config.track_width
            })
        
        yaw_excess = self.critic._compute_yaw_excess(trajectory)
        
        # Turning should have positive yaw excess
        self.assertGreater(yaw_excess, 0.0,
                          msg="Turning path should have positive yaw excess")
    
    def test_velocity_chattering_constant_velocity(self):
        """Test velocity chattering for constant velocity (should be 0)."""
        trajectory = []
        for i in range(10):
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        chattering = self.critic._compute_velocity_chattering(trajectory)
        
        # Constant velocity should have zero chattering
        self.assertEqual(chattering, 0,
                        msg="Constant velocity should have zero chattering")
    
    def test_velocity_chattering_oscillating(self):
        """Test velocity chattering for oscillating velocity (should be > 0)."""
        trajectory = []
        for i in range(10):
            vl = 0.1 if i % 2 == 0 else -0.1
            vr = 0.1 if i % 2 == 0 else -0.1
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': vl,
                'vr': vr,
                'omega': 0.0
            })
        
        chattering = self.critic._compute_velocity_chattering(trajectory)
        
        # Oscillating velocity should have positive chattering
        self.assertGreater(chattering, 0,
                          msg="Oscillating velocity should have positive chattering")
    
    def test_evaluate_identifies_no_problems(self):
        """Test that evaluate returns empty list for good trajectory."""
        trajectory = []
        for i in range(10):
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        bad_windows = self.critic.evaluate(trajectory, num_samples_per_segment=5)
        
        # Good trajectory should have no problematic windows
        self.assertEqual(len(bad_windows), 0,
                        msg="Good trajectory should have no problematic windows")
    
    def test_evaluate_identifies_problems(self):
        """Test that evaluate identifies problems in bad trajectory."""
        trajectory = []
        for i in range(10):
            # Create a trajectory with sudden changes
            y = 0.0 if i < 5 else 0.5
            heading = 0.0 if i < 5 else np.pi/2
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': y,
                'heading': heading,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        bad_windows = self.critic.evaluate(trajectory, num_samples_per_segment=5)
        
        # Bad trajectory should have problematic windows
        self.assertGreater(len(bad_windows), 0,
                          msg="Bad trajectory should have problematic windows")
    
    def test_thresholds_are_respected(self):
        """Test that thresholds are respected when flagging problems."""
        # Create a trajectory that's just below the threshold
        trajectory = []
        for i in range(10):
            # Small tortuosity (close to straight line)
            y = 0.01 * np.sin(i * 0.1)  # Very small oscillation
            trajectory.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': y,
                'heading': 0.0,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        bad_windows = self.critic.evaluate(trajectory, num_samples_per_segment=5)
        
        # Small deviations should not trigger problems
        self.assertEqual(len(bad_windows), 0,
                        msg="Small deviations should not trigger problems")


if __name__ == '__main__':
    unittest.main()
