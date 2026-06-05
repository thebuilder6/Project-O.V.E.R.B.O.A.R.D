"""
Unit tests for validator.py
"""

import unittest
import numpy as np
import json
import tempfile
import os
from overboard.core.robot_model import RobotConfig
from overboard.validation.validator import forward_integrate, audit_constraints, compute_metrics


class TestForwardIntegrate(unittest.TestCase):
    """Test cases for forward_integrate function."""

    def test_forward_integrate_empty_samples(self):
        """Test forward integration with empty samples."""
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
        
        integrated, errors = forward_integrate([], config, fine_dt=0.001)
        
        self.assertEqual(integrated, [])
        self.assertEqual(errors, {})

    def test_forward_integrate_straight_line(self):
        """Test forward integration for straight line motion."""
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
        
        # Create simple trajectory: constant velocity straight line
        samples = []
        for i in range(11):
            t = i * 0.1
            samples.append({
                "t": t,
                "x": 0.5 * t,
                "y": 0.0,
                "heading": 0.0,
                "vl": 0.5,
                "vr": 0.5,
                "omega": 0.0,
                "al": 0.0,
                "ar": 0.0,
                "fl": 0.0,
                "fr": 0.0
            })
        
        integrated, errors = forward_integrate(samples, config, fine_dt=0.001)
        
        self.assertGreater(len(integrated), 0)
        self.assertIn("max_pos_error_m", errors)
        self.assertIn("rms_pos_error_m", errors)
        self.assertIn("max_heading_error_rad", errors)
        # For constant velocity, error should be small
        self.assertLess(errors["max_pos_error_m"], 0.01)

    def test_forward_integrate_with_turning(self):
        """Test forward integration with turning motion."""
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
        
        # Create trajectory with turning
        samples = []
        for i in range(11):
            t = i * 0.1
            # Differential velocities cause turning
            vl = 0.5
            vr = 0.6
            omega = (vr - vl) / config.track_width
            v = (vl + vr) / 2.0
            if t == 0:
                x = 0.0
                y = 0.0
            else:
                x = (v / omega) * np.sin(omega * t)
                y = (v / omega) * (1.0 - np.cos(omega * t))
            samples.append({
                "t": t,
                "x": x,
                "y": y,
                "heading": omega * t,
                "vl": vl,
                "vr": vr,
                "omega": omega,
                "al": 0.0,
                "ar": 0.0,
                "fl": 0.0,
                "fr": 0.0
            })
        
        integrated, errors = forward_integrate(samples, config, fine_dt=0.001)
        
        self.assertGreater(len(integrated), 0)
        self.assertLess(errors["max_pos_error_m"], 0.05)  # Allow some error for turning


class TestAuditConstraints(unittest.TestCase):
    """Test cases for audit_constraints function."""

    def test_audit_constraints_empty_samples(self):
        """Test constraint audit with empty samples."""
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
        
        audit = audit_constraints([], config, apply_headroom=True)
        
        self.assertEqual(audit["num_violating_samples"], 0)
        self.assertEqual(audit["num_slip_points"], 0)
        self.assertEqual(audit["left_motor_force"], 0.0)
        self.assertEqual(audit["right_motor_force"], 0.0)

    def test_audit_constraints_no_violation(self):
        """Test constraint audit with no violations."""
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
        
        # Create trajectory with low acceleration (no violations)
        samples = []
        for i in range(11):
            t = i * 0.1
            samples.append({
                "t": t,
                "x": 0.3 * t,
                "y": 0.0,
                "heading": 0.0,
                "vl": 0.3,
                "vr": 0.3,
                "omega": 0.0,
                "al": 0.1,
                "ar": 0.1,
                "fl": 0.0,
                "fr": 0.0
            })
        
        audit = audit_constraints(samples, config, apply_headroom=True)
        
        self.assertEqual(audit["num_violating_samples"], 0)
        self.assertEqual(audit["num_slip_points"], 0)

    def test_audit_constraints_with_motor_violation(self):
        """Test constraint audit with motor limit violation."""
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
        
        # Create trajectory with high acceleration (violates motor limits)
        samples = []
        for i in range(11):
            t = i * 0.1
            samples.append({
                "t": t,
                "x": 0.5 * t,
                "y": 0.0,
                "heading": 0.0,
                "vl": 0.5,
                "vr": 0.5,
                "omega": 0.0,
                "al": 100.0,  # Very high acceleration
                "ar": 100.0,
                "fl": 0.0,
                "fr": 0.0
            })
        
        audit = audit_constraints(samples, config, apply_headroom=False)
        
        # Should have motor violations
        self.assertGreater(audit["num_violating_samples"], 0)

    def test_audit_constraints_with_slip(self):
        """Test constraint audit with wheel slip."""
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
        
        # Create trajectory with high forces (causes slip)
        samples = []
        for i in range(11):
            t = i * 0.1
            samples.append({
                "t": t,
                "x": 0.5 * t,
                "y": 0.0,
                "heading": 0.0,
                "vl": 0.5,
                "vr": 0.5,
                "omega": 0.0,
                "al": 10.0,
                "ar": 10.0,
                "fl": 10.0,  # High force
                "fr": 10.0
            })
        
        audit = audit_constraints(samples, config, apply_headroom=False)
        
        # Should have slip points
        self.assertGreater(audit["num_slip_points"], 0)

    def test_audit_constraints_normal_forces(self):
        """Test that normal forces are computed correctly."""
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
        
        # Static case
        samples = [{
            "t": 0.0,
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
            "vl": 0.0,
            "vr": 0.0,
            "omega": 0.0,
            "al": 0.0,
            "ar": 0.0,
            "fl": 0.0,
            "fr": 0.0
        }]
        
        audit = audit_constraints(samples, config, apply_headroom=True)
        
        # Check that normal forces are computed
        self.assertIn("left_normal_force", audit)
        self.assertIn("right_normal_force", audit)
        # Should be approximately 50/50 weight distribution
        expected = (0.8 * 9.81) / 2.0
        self.assertAlmostEqual(audit["left_normal_force"], expected, delta=0.1)
        self.assertAlmostEqual(audit["right_normal_force"], expected, delta=0.1)


class TestComputeMetrics(unittest.TestCase):
    """Test cases for compute_metrics function."""

    def test_compute_metrics_empty_samples(self):
        """Test metrics computation with empty samples."""
        metrics = compute_metrics([])
        
        self.assertEqual(metrics, {})

    def test_compute_metrics_basic(self):
        """Test basic metrics computation."""
        samples = []
        for i in range(11):
            t = i * 0.1
            samples.append({
                "t": t,
                "x": 0.5 * t,
                "y": 0.0,
                "heading": 0.0,
                "vl": 0.5,
                "vr": 0.5,
                "omega": 0.0,
                "al": 0.0,
                "ar": 0.0,
                "fl": 0.0,
                "fr": 0.0
            })
        
        metrics = compute_metrics(samples)
        
        self.assertIn("total_time_s", metrics)
        self.assertIn("path_length_m", metrics)
        self.assertIn("max_linear_speed_m_s", metrics)
        self.assertIn("max_wheel_speed_m_s", metrics)
        self.assertIn("max_accel_m_s2", metrics)
        self.assertIn("max_jerk_m_s3", metrics)
        
        self.assertAlmostEqual(metrics["total_time_s"], 1.0, places=6)
        self.assertAlmostEqual(metrics["path_length_m"], 0.5, places=6)
        self.assertAlmostEqual(metrics["max_linear_speed_m_s"], 0.5, places=6)

    def test_compute_metrics_with_acceleration(self):
        """Test metrics computation with acceleration."""
        samples = []
        for i in range(11):
            t = i * 0.1
            samples.append({
                "t": t,
                "x": 0.25 * t * t,  # x = 0.5 * a * t^2 with a=0.5
                "y": 0.0,
                "heading": 0.0,
                "vl": 0.5 * t,  # v = a * t
                "vr": 0.5 * t,
                "omega": 0.0,
                "al": 0.5,
                "ar": 0.5,
                "fl": 0.0,
                "fr": 0.0
            })
        
        metrics = compute_metrics(samples)
        
        self.assertGreater(metrics["max_accel_m_s2"], 0)
        self.assertAlmostEqual(metrics["max_accel_m_s2"], 0.5, places=6)

    def test_compute_metrics_with_jerk(self):
        """Test metrics computation with jerk."""
        samples = []
        for i in range(11):
            t = i * 0.1
            samples.append({
                "t": t,
                "x": 0.0,
                "y": 0.0,
                "heading": 0.0,
                "vl": 0.5,
                "vr": 0.5,
                "omega": 0.0,
                "al": 0.1 * t,  # Linearly increasing acceleration
                "ar": 0.1 * t,
                "fl": 0.0,
                "fr": 0.0
            })
        
        metrics = compute_metrics(samples)
        
        # Should have non-zero jerk
        self.assertGreater(metrics["max_jerk_m_s3"], 0)


class TestValidateTrajectory(unittest.TestCase):
    """Test cases for validate_trajectory function (integration test)."""

    def test_validate_trajectory_integration(self):
        """Test full validation workflow."""
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
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.traj', delete=False) as traj_file:
            traj_data = {
                "name": "test",
                "version": 3,
                "trajectory": {
                    "config": config_dict,
                    "samples": []
                }
            }
            # Add simple trajectory samples
            for i in range(11):
                t = i * 0.1
                traj_data["trajectory"]["samples"].append({
                    "t": t,
                    "x": 0.5 * t,
                    "y": 0.0,
                    "heading": 0.0,
                    "vl": 0.5,
                    "vr": 0.5,
                    "omega": 0.0,
                    "al": 0.0,
                    "ar": 0.0,
                    "fl": 0.0,
                    "fr": 0.0
                })
            json.dump(traj_data, traj_file)
            traj_path = traj_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            config_data = config_dict
            json.dump(config_data, config_file)
            config_path = config_file.name
        
        try:
            from overboard.validation.validator import validate_trajectory
            metrics, audit, errors = validate_trajectory(traj_path, config_path, apply_headroom=True)
            
            # Check that all components are returned
            self.assertIsInstance(metrics, dict)
            self.assertIsInstance(audit, dict)
            self.assertIsInstance(errors, dict)
            
            # Check metrics
            self.assertIn("total_time_s", metrics)
            self.assertIn("path_length_m", metrics)
            
            # Check audit
            self.assertIn("num_violating_samples", audit)
            self.assertIn("num_slip_points", audit)
            
            # Check errors
            self.assertIn("max_pos_error_m", errors)
            self.assertIn("rms_pos_error_m", errors)
            
        finally:
            # Clean up temporary files
            os.unlink(traj_path)
            os.unlink(config_path)
