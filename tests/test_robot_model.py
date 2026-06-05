"""
Unit tests for robot_model.py
"""

import unittest
import numpy as np
from overboard.core.robot_model import RobotConfig, DifferentialDriveModel


class TestRobotConfig(unittest.TestCase):
    """Test cases for RobotConfig class."""

    def test_init_with_new_json_format(self):
        """Test initialization with new JSON format."""
        config_dict = {
            "robot": {
                "mass": 0.8,
                "inertia": 0.002,
                "track_width": 0.1,
                "wheel_radius": 0.03,
                "v_max_rad_s": 15.0,
                "t_max_nm": 0.05,
                "gearing": 1.0,
                "cof": 0.5,
                "gravity": 9.81,
                "torque_headroom": 0.9,
                "speed_headroom": 0.95
            },
            "multiverse": {
                "bootstrap": {
                    "turning_radius": 0.3
                }
            }
        }
        config = RobotConfig(config_dict)
        
        self.assertEqual(config.mass, 0.8)
        self.assertEqual(config.inertia, 0.002)
        self.assertEqual(config.track_width, 0.1)
        self.assertEqual(config.wheel_radius, 0.03)
        self.assertEqual(config.v_max_rad_s, 15.0)
        self.assertEqual(config.t_max_nm, 0.05)
        self.assertEqual(config.gearing, 1.0)
        self.assertEqual(config.cof, 0.5)
        self.assertEqual(config.g, 9.81)
        self.assertEqual(config.torque_headroom, 0.9)
        self.assertEqual(config.speed_headroom, 0.95)
        self.assertEqual(config.multiverse_config, {"bootstrap": {"turning_radius": 0.3}})

    def test_init_with_old_choreo_format(self):
        """Test initialization with old Choreo format."""
        config_dict = {
            "config": {
                "mass": {"val": 0.7},
                "inertia": {"val": 0.001},
                "differentialTrackWidth": {"val": 0.095},
                "radius": {"val": 0.028},
                "vmax": {"val": 15.7},
                "tmax": {"val": 0.04},
                "gearing": {"val": 1.0},
                "cof": {"val": 0.45},
                "torqueHeadroom": {"val": 0.85},
                "speedHeadroom": {"val": 0.90}
            }
        }
        config = RobotConfig(config_dict)
        
        self.assertEqual(config.mass, 0.7)
        self.assertEqual(config.inertia, 0.001)
        self.assertEqual(config.track_width, 0.095)
        self.assertEqual(config.wheel_radius, 0.028)
        self.assertEqual(config.v_max_rad_s, 15.7)
        self.assertEqual(config.t_max_nm, 0.04)
        self.assertEqual(config.gearing, 1.0)
        self.assertEqual(config.cof, 0.45)
        self.assertEqual(config.g, 9.81)  # Default gravity
        self.assertEqual(config.torque_headroom, 0.85)
        self.assertEqual(config.speed_headroom, 0.90)
        self.assertEqual(config.multiverse_config, {})

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config_dict = {}
        config = RobotConfig(config_dict)
        
        self.assertEqual(config.mass, RobotConfig.DEFAULT_MASS)
        self.assertEqual(config.inertia, RobotConfig.DEFAULT_INERTIA)
        self.assertEqual(config.track_width, RobotConfig.DEFAULT_TRACK_WIDTH)
        self.assertEqual(config.wheel_radius, RobotConfig.DEFAULT_WHEEL_RADIUS)
        self.assertEqual(config.v_max_rad_s, RobotConfig.DEFAULT_VMAX)
        self.assertEqual(config.t_max_nm, RobotConfig.DEFAULT_TMAX)
        self.assertEqual(config.gearing, RobotConfig.DEFAULT_GEARING)
        self.assertEqual(config.cof, RobotConfig.DEFAULT_COF)
        self.assertEqual(config.g, RobotConfig.GRAVITY)

    def test_get_max_force_at_velocity_zero(self):
        """Test max force calculation at zero velocity."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1, 
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        
        force = config.get_max_force_at_velocity(0.0, apply_headroom=False)
        expected = 0.05 / 0.03  # t_max / wheel_radius
        self.assertAlmostEqual(force, expected, places=6)

    def test_get_max_force_at_velocity_max(self):
        """Test max force calculation at max velocity."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        
        force = config.get_max_force_at_velocity(0.03 * 15.0, apply_headroom=False)
        self.assertAlmostEqual(force, 0.0, places=6)  # No force at max velocity

    def test_get_max_force_with_headroom(self):
        """Test max force calculation with headroom."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5, "torque_headroom": 0.8}}
        config = RobotConfig(config_dict)
        
        force_no_headroom = config.get_max_force_at_velocity(0.0, apply_headroom=False)
        force_with_headroom = config.get_max_force_at_velocity(0.0, apply_headroom=True)
        
        self.assertEqual(force_with_headroom, force_no_headroom * 0.8)

    def test_max_linear_speed(self):
        """Test max linear speed calculation."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        
        speed = config.max_linear_speed(apply_headroom=False)
        expected = 15.0 * 0.03  # v_max * wheel_radius
        self.assertAlmostEqual(speed, expected, places=6)

    def test_max_linear_speed_with_headroom(self):
        """Test max linear speed with headroom."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5, "speed_headroom": 0.9}}
        config = RobotConfig(config_dict)
        
        speed_no_headroom = config.max_linear_speed(apply_headroom=False)
        speed_with_headroom = config.max_linear_speed(apply_headroom=True)
        
        self.assertEqual(speed_with_headroom, speed_no_headroom * 0.9)


class TestDifferentialDriveModel(unittest.TestCase):
    """Test cases for DifferentialDriveModel class."""

    def test_get_dynamics_straight_line(self):
        """Test dynamics calculation for straight line motion."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)
        
        # Straight line: both wheels at same velocity and acceleration
        fl, fr = model.get_dynamics(vl=0.5, vr=0.5, al=1.0, ar=1.0)
        
        # Forces should be equal for straight line motion
        self.assertAlmostEqual(fl, fr, places=6)

    def test_get_dynamics_turning(self):
        """Test dynamics calculation for turning motion."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)
        
        # Turning: different wheel accelerations
        fl, fr = model.get_dynamics(vl=0.5, vr=0.5, al=0.0, ar=2.0)
        
        # Forces should be different for turning
        self.assertGreater(abs(fl - fr), 1e-6)

    def test_check_constraints_no_violation(self):
        """Test constraint checking with no violations."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)
        
        # Low acceleration should not violate constraints
        violations = model.check_constraints(vl=0.1, vr=0.1, al=0.1, ar=0.1, apply_headroom=True)
        
        self.assertEqual(violations["left_motor_violation"], 0.0)
        self.assertEqual(violations["right_motor_violation"], 0.0)
        self.assertEqual(violations["traction_violation"], 0.0)

    def test_check_constraints_motor_limit(self):
        """Test constraint checking with motor limit violation."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)
        
        # Very high acceleration should violate motor limits
        violations = model.check_constraints(vl=0.1, vr=0.1, al=100.0, ar=100.0, apply_headroom=False)
        
        # Should have motor violations
        self.assertTrue(violations["left_motor_violation"] > 0.0 or violations["right_motor_violation"] > 0.0)

    def test_get_wheel_normal_forces_static(self):
        """Test normal force calculation for static case."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)
        
        # Static case: no acceleration
        nl, nr = model.get_wheel_normal_forces(vl=0.0, vr=0.0, al=0.0, ar=0.0)
        
        # Should be 50/50 weight distribution
        expected = (0.8 * 9.81) / 2.0
        self.assertAlmostEqual(nl, expected, places=6)
        self.assertAlmostEqual(nr, expected, places=6)

    def test_get_wheel_normal_forces_acceleration(self):
        """Test normal force calculation with acceleration."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)
        
        # Accelerating forward
        nl, nr = model.get_wheel_normal_forces(vl=0.5, vr=0.5, al=1.0, ar=1.0)
        
        # Weight should shift (longitudinal transfer)
        base_normal = (0.8 * 9.81) / 2.0
        # nl and nr should differ from base_normal due to weight transfer
        self.assertTrue(abs(nl - base_normal) > 1e-6 or abs(nr - base_normal) > 1e-6)
