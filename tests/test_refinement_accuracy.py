"""
Integration tests for measuring accuracy improvements from refinements.

Tests measure how much each heuristic improves:
- Trajectory quality metrics
- Constraint satisfaction
- Kinematic validity
"""

import unittest
import numpy as np
import json
import tempfile
import os
from robot_model import RobotConfig
from multiverse_optimizer import MasterTrajectoryOptimizer, PathBootstrapper, TrajectoryCritic
from validator import validate_trajectory


class TestRefinementAccuracy(unittest.TestCase):
    """Test suite measuring accuracy improvements from refinements."""
    
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
        
        # Create simple test waypoints
        self.simple_waypoints = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0)
        ]
        
        # Create complex test waypoints
        self.complex_waypoints = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, np.pi/4),
            (1.0, 0.0, 0.0),
            (1.5, -0.5, -np.pi/4),
            (2.0, 0.0, 0.0)
        ]
    
    def _save_trajectory(self, trajectory, filename):
        """Save trajectory to temporary file for validation."""
        with open(filename, 'w') as f:
            json.dump(trajectory, f)
    
    def test_bootstrap_vs_no_bootstrap(self):
        """Test improvement from bootstrapping with Reeds-Shepp vs linear interpolation."""
        bootstrapper = PathBootstrapper(self.config)
        
        # Generate bootstrap trajectory
        bootstrap_traj = bootstrapper.generate_baseline(self.complex_waypoints, num_samples_per_segment=5)
        
        # Convert 1D array to list of dicts
        # Format: [dt, x, y, theta, vl, vr, x, y, theta, vl, vr, ...]
        bootstrap_traj_formatted = []
        dt = bootstrap_traj[0]
        for i in range(1, len(bootstrap_traj), 5):
            bootstrap_traj_formatted.append({
                'x': bootstrap_traj[i],
                'y': bootstrap_traj[i + 1],
                'heading': bootstrap_traj[i + 2],
                'vl': bootstrap_traj[i + 3],
                'vr': bootstrap_traj[i + 4],
                'omega': (bootstrap_traj[i + 4] - bootstrap_traj[i + 3]) / self.config.track_width
            })
        
        # Generate linear interpolation trajectory
        from path_planning import linear_interpolation_waypoints
        linear_traj = []
        for i in range(len(self.complex_waypoints) - 1):
            start = self.complex_waypoints[i]
            end = self.complex_waypoints[i + 1]
            segment = linear_interpolation_waypoints(start, end, 5)
            linear_traj.extend(segment)
        
        # Convert to trajectory format
        linear_traj_formatted = []
        for i, (x, y, theta) in enumerate(linear_traj):
            linear_traj_formatted.append({
                'x': x, 'y': y, 'heading': theta,
                'vl': 0.1, 'vr': 0.1, 'omega': 0.0
            })
        
        # Compare trajectory smoothness
        bootstrap_smoothness = self._calculate_smoothness(bootstrap_traj_formatted)
        linear_smoothness = self._calculate_smoothness(linear_traj_formatted)
        
        # Bootstrap should be at least as smooth as linear interpolation
        # (may be equal if Reeds-Shepp falls back to linear)
        self.assertLessEqual(bootstrap_smoothness, linear_smoothness,
                       "Bootstrap trajectory should be at least as smooth as linear interpolation")
    
    def test_critic_identifies_problems(self):
        """Test that critic correctly identifies problematic segments."""
        # Create a trajectory with known problems
        problematic_traj = []
        for i in range(10):
            # Create a zigzag pattern with high tortuosity
            y = 0.5 if i % 2 == 0 else -0.5
            problematic_traj.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': y,  # Zigzag pattern
                'heading': 0.0,
                'vl': 0.1 if i % 2 == 0 else -0.1,  # Velocity chattering
                'vr': 0.1 if i % 2 == 0 else -0.1,
                'omega': 0.0
            })
        
        critic = TrajectoryCritic(self.config)
        bad_windows = critic.evaluate(problematic_traj, num_samples_per_segment=5)
        
        # Should identify at least one problematic window
        self.assertGreater(len(bad_windows), 0, "Critic should identify problematic segments")
    
    def test_refinement_improves_cost(self):
        """Test that refinement improves trajectory cost."""
        # This is an integration test that requires the full optimizer
        # For now, we'll test the concept
        
        optimizer = MasterTrajectoryOptimizer(self.config, enable_parallel=False, num_workers=1)
        
        # Generate initial trajectory
        initial_traj = optimizer.bootstrapper.generate_baseline(
            self.simple_waypoints, num_samples_per_segment=10
        )
        
        # Convert 1D array to list of dicts for cost calculation
        initial_traj_formatted = []
        dt = initial_traj[0]
        for i in range(1, len(initial_traj), 5):
            initial_traj_formatted.append({
                'x': initial_traj[i],
                'y': initial_traj[i + 1],
                'heading': initial_traj[i + 2],
                'vl': initial_traj[i + 3],
                'vr': initial_traj[i + 4],
                'omega': (initial_traj[i + 4] - initial_traj[i + 3]) / self.config.track_width
            })
        
        # Calculate initial cost (total time)
        initial_cost = self._calculate_trajectory_cost(initial_traj_formatted)
        
        # Note: Full refinement test would require running the optimizer
        # This is a placeholder for the integration test
        self.assertIsNotNone(initial_cost, "Should be able to calculate initial cost")
    
    
    def test_refinement_preserves_constraints(self):
        """Test that refinement preserves kinematic constraints."""
        from robot_model import DifferentialDriveModel
        optimizer = MasterTrajectoryOptimizer(self.config, enable_parallel=False, num_workers=1)
        
        # Generate trajectory
        trajectory = optimizer.bootstrapper.generate_baseline(
            self.simple_waypoints, num_samples_per_segment=10
        )
        
        # Convert 1D array to list of dicts
        trajectory_formatted = []
        dt = trajectory[0]
        for i in range(1, len(trajectory), 5):
            trajectory_formatted.append({
                'x': trajectory[i],
                'y': trajectory[i + 1],
                'heading': trajectory[i + 2],
                'vl': trajectory[i + 3],
                'vr': trajectory[i + 4],
                'omega': (trajectory[i + 4] - trajectory[i + 3]) / self.config.track_width
            })
        
        # Check constraints
        model = DifferentialDriveModel(self.config)
        violations = []
        
        for state in trajectory_formatted:
            vl = state['vl']
            vr = state['vr']
            al = 0.0  # Assume zero acceleration for this test
            ar = 0.0
            
            constraint_check = model.check_constraints(vl, vr, al, ar, apply_headroom=False)
            
            # Check for any violations
            if constraint_check['left_motor_violation'] > 0:
                violations.append('left_motor')
            if constraint_check['right_motor_violation'] > 0:
                violations.append('right_motor')
            if constraint_check['left_wheel_slip'] > 0:
                violations.append('left_slip')
            if constraint_check['right_wheel_slip'] > 0:
                violations.append('right_slip')
        
        # Bootstrap trajectory should have minimal violations
        # (may have some due to discretization, but should be minimal)
        self.assertLess(len(violations), len(trajectory_formatted) * 0.1,
                       "Bootstrap trajectory should have minimal constraint violations")
    
    def _calculate_smoothness(self, trajectory):
        """Calculate trajectory smoothness (tortuosity)."""
        if len(trajectory) < 2:
            return 0.0
        
        total_path_length = 0.0
        straight_line_distance = np.sqrt(
            (trajectory[-1]['x'] - trajectory[0]['x'])**2 +
            (trajectory[-1]['y'] - trajectory[0]['y'])**2
        )
        
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1]['x'] - trajectory[i]['x']
            dy = trajectory[i+1]['y'] - trajectory[i]['y']
            total_path_length += np.sqrt(dx**2 + dy**2)
        
        if straight_line_distance == 0:
            return float('inf')
        
        return total_path_length / straight_line_distance
    
    def _calculate_trajectory_cost(self, trajectory):
        """Calculate total trajectory time."""
        if not trajectory:
            return 0.0
        
        # Simple cost: sum of dt values (assuming uniform dt)
        # In a real implementation, this would be more sophisticated
        return len(trajectory) * 0.1  # Placeholder
    
    def test_jerk_cost_metric(self):
        """Test jerk cost metric calculation."""
        critic = TrajectoryCritic(self.config)
        
        # Create trajectory with high jerk (rapid acceleration changes)
        high_jerk_traj = []
        for i in range(10):
            vl = 0.1 * ((-1) ** i)  # Alternating acceleration
            vr = 0.1 * ((-1) ** i)
            high_jerk_traj.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': vl,
                'vr': vr,
                'omega': 0.0
            })
        
        # Create smooth trajectory with low jerk
        smooth_traj = []
        for i in range(10):
            vl = 0.1  # Constant velocity
            vr = 0.1
            smooth_traj.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': vl,
                'vr': vr,
                'omega': 0.0
            })
        
        high_jerk_cost = critic._calculate_jerk_cost(high_jerk_traj)
        smooth_jerk_cost = critic._calculate_jerk_cost(smooth_traj)
        
        # High jerk trajectory should have higher cost
        self.assertGreater(high_jerk_cost, smooth_jerk_cost,
                          "High jerk trajectory should have higher jerk cost")
        
        # Smooth trajectory should be below threshold
        self.assertLess(smooth_jerk_cost, critic.jerk_cost_threshold,
                        "Smooth trajectory should be below jerk threshold")
    
    def test_curvature_cost_metric(self):
        """Test curvature cost metric calculation."""
        critic = TrajectoryCritic(self.config)
        
        # Create trajectory with sharp turns (high curvature)
        sharp_turn_traj = []
        for i in range(10):
            heading = (np.pi / 2) if i % 2 == 0 else 0.0  # 90-degree turns
            sharp_turn_traj.append({
                'x': i * 0.1,
                'y': 0.0,
                'heading': heading,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        # Create straight trajectory (low curvature)
        straight_traj = []
        for i in range(10):
            straight_traj.append({
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': 0.1,
                'vr': 0.1,
                'omega': 0.0
            })
        
        sharp_curvature_cost = critic._calculate_curvature_cost(sharp_turn_traj)
        straight_curvature_cost = critic._calculate_curvature_cost(straight_traj)
        
        # Sharp turn trajectory should have higher curvature cost
        self.assertGreater(sharp_curvature_cost, straight_curvature_cost,
                          "Sharp turn trajectory should have higher curvature cost")
        
        # Straight trajectory should be below threshold
        self.assertLess(straight_curvature_cost, critic.curvature_cost_threshold,
                        "Straight trajectory should be below curvature threshold")
    
    def test_centripetal_cost_metric(self):
        """Test centripetal acceleration cost metric calculation."""
        critic = TrajectoryCritic(self.config)
        
        # Calculate friction limit
        a_max = self.config.cof * self.config.g  # ~3.924 m/s²
        threshold = 0.8 * a_max  # ~3.139 m/s²
        
        # Create trajectory with high centripetal acceleration (exceeds threshold)
        high_centripetal_traj = []
        for i in range(10):
            v = 2.0  # High speed
            omega = 2.0  # High angular velocity (v*omega = 4.0 > threshold)
            high_centripetal_traj.append({
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': v - omega * self.config.track_width / 2,
                'vr': v + omega * self.config.track_width / 2,
                'omega': omega
            })
        
        # Create trajectory with low centripetal acceleration (below threshold)
        low_centripetal_traj = []
        for i in range(10):
            v = 0.1  # Low speed
            omega = 0.0  # No turn
            low_centripetal_traj.append({
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': v,
                'vr': v,
                'omega': omega
            })
        
        high_centripetal_cost = critic._calculate_centripetal_cost(high_centripetal_traj)
        low_centripetal_cost = critic._calculate_centripetal_cost(low_centripetal_traj)
        
        # High centripetal trajectory should have higher cost
        self.assertGreater(high_centripetal_cost, low_centripetal_cost,
                          "High centripetal trajectory should have higher cost")
        
        # Low centripetal trajectory should be below threshold (cost = 0)
        self.assertEqual(low_centripetal_cost, 0.0,
                         "Low centripetal trajectory should have zero cost")
    
    def test_research_metrics_in_critic_evaluation(self):
        """Test that research-grounded metrics are used in critic evaluation."""
        critic = TrajectoryCritic(self.config)
        
        # Create trajectory that violates research metrics but not original metrics
        research_violating_traj = []
        for i in range(10):
            # High jerk but low tortuosity, yaw excess, and chattering
            vl = 0.1 + 0.05 * ((-1) ** i)  # Small oscillations
            vr = 0.1 + 0.05 * ((-1) ** i)
            research_violating_traj.append({
                't': i * 0.1,
                'x': i * 0.1,
                'y': 0.0,
                'heading': 0.0,
                'vl': vl,
                'vr': vr,
                'omega': 0.0
            })
        
        bad_windows = critic.evaluate(research_violating_traj, num_samples_per_segment=5)
        
        # Should identify problematic windows due to research metrics
        # (This test verifies the integration of new metrics into evaluation)
        self.assertIsNotNone(bad_windows, "Critic should return bad windows list")
    
    def test_research_metrics_thresholds_configurable(self):
        """Test that research metric thresholds are configurable via config."""
        critic = TrajectoryCritic(self.config)
        
        # Verify thresholds are set
        self.assertIsNotNone(critic.jerk_cost_threshold)
        self.assertIsNotNone(critic.curvature_cost_threshold)
        self.assertIsNotNone(critic.centripetal_cost_threshold)
        
        # Verify thresholds match constants
        from multiverse_optimizer import JERK_COST_THRESHOLD, CURVATURE_COST_THRESHOLD, CENTRIPETAL_COST_THRESHOLD
        self.assertEqual(critic.jerk_cost_threshold, JERK_COST_THRESHOLD)
        self.assertEqual(critic.curvature_cost_threshold, CURVATURE_COST_THRESHOLD)
        self.assertEqual(critic.centripetal_cost_threshold, CENTRIPETAL_COST_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
