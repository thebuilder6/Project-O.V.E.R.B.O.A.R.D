"""
Unit tests for robot_model.py
"""

import unittest
import numpy as np
from overboard.models.robot_model import RobotConfig, DifferentialDriveModel


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

    def test_robust_force_uses_declared_worst_case(self):
        """KI-009/KI-010: use the declared torque headroom and radius ranges."""
        config_dict = {
            "robot": {
                "mass": 0.723,
                "inertia": 0.001,
                "track_width": 0.0965,
                "wheel_radius": 0.028,
                "v_max_rad_s": 18.94,
                "t_max_nm": 0.138,
                "gearing": 1.0,
                "cof": 1.37,
                "uncertainties": {
                    "torque_headroom": [0.75, 0.9],
                    "wheel_radius_range": [0.0275, 0.0285],
                },
            }
        }
        config = RobotConfig(config_dict)

        v = 0.3771
        speed_radius, force_radius, headroom = config.get_robust_force_params()
        omega = v / speed_radius
        torque = max(0.0, config.t_max_nm * (1.0 - omega / config.v_max_rad_s))
        expected = (torque / force_radius) * config.gearing * headroom

        self.assertAlmostEqual(
            config.get_max_force_at_velocity_robust(v), expected, places=9)
        self.assertLess(
            config.get_max_force_at_velocity_robust(v),
            config.get_max_force_at_velocity(v, apply_headroom=True))

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

    def test_canonical_uncertainties_are_parsed(self):
        """KI-009: robot.uncertainties must be normalized for validation."""
        config_dict = {
            "robot": {
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
            }
        }
        config = RobotConfig(config_dict)

        self.assertEqual(config.uncertainties["cof"], (1.24, 1.51))
        self.assertEqual(config.uncertainties["torque_headroom"], (0.75, 0.9))
        self.assertEqual(config.uncertainties["gear_backlash_m"], (0.0002, 0.0007))
        self.assertEqual(config.uncertainties["track_width_range"], (0.096, 0.097))
        self.assertEqual(config.uncertainties["wheel_radius_range"], (0.0275, 0.0285))
        self.assertEqual(config.uncertainties["inertia_kg_m2"], (0.0008, 0.0011))

    def test_legacy_uncertainties_are_normalized(self):
        """KI-008/KI-009: legacy {val} ranges must work too."""
        config_dict = {
            "config": {
                "mass": {"val": 1.23},
                "differentialTrackWidth": {"val": 0.2},
                "uncertainties": {
                    "cof": {"val": [0.8, 1.2]},
                    "track_width_range": [0.19, 0.21],
                },
            }
        }
        config = RobotConfig(config_dict)

        self.assertEqual(config.track_width, 0.2)
        self.assertEqual(config.uncertainties["cof"], (0.8, 1.2))
        self.assertEqual(config.uncertainties["track_width_range"], (0.19, 0.21))
        # Missing declarations keep the previous defaults.
        self.assertEqual(config.uncertainties["gear_backlash_m"], (0.0001, 0.0006))


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

    def test_check_constraints_headroom_uses_declared_worst_case(self):
        """KI-010: CPU and Immrax use the same per-wheel force limit."""
        config_dict = {
            "robot": {
                "mass": 0.723,
                "inertia": 0.001,
                "track_width": 0.0965,
                "wheel_radius": 0.028,
                "v_max_rad_s": 18.94,
                "t_max_nm": 0.138,
                "gearing": 1.0,
                "cof": 1.37,
                "uncertainties": {
                    "torque_headroom": [0.75, 0.9],
                    "wheel_radius_range": [0.0275, 0.0285],
                },
            }
        }
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)

        v = 0.3771
        a = 3.2
        self.assertLess(abs(model.get_dynamics(v, v, a, a)[0]),
                        config.get_max_force_at_velocity(v, apply_headroom=True))
        violations = model.check_constraints(
            vl=v, vr=v, al=a, ar=a, apply_headroom=True)

        self.assertGreater(violations["left_motor_violation"], 0.0)
        self.assertAlmostEqual(
            violations["left_motor_violation"],
            violations["right_motor_violation"], places=9)

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
        """KI-010: straight acceleration must not split load left/right."""
        config_dict = {"robot": {"mass": 0.8, "inertia": 0.002, "track_width": 0.1,
                               "wheel_radius": 0.03, "v_max_rad_s": 15.0, "t_max_nm": 0.05,
                               "gearing": 1.0, "cof": 0.5}}
        config = RobotConfig(config_dict)
        model = DifferentialDriveModel(config)
        base_normal = (0.8 * 9.81) / 2.0

        # Accelerating straight: no left/right weight transfer (no front axle)
        nl, nr = model.get_wheel_normal_forces(vl=0.5, vr=0.5, al=1.0, ar=1.0)
        self.assertAlmostEqual(nl, base_normal, places=6)
        self.assertAlmostEqual(nr, base_normal, places=6)

        # Turning: the outer wheel gets more load than the inner wheel
        nl, nr = model.get_wheel_normal_forces(vl=0.3, vr=0.6, al=0.0, ar=0.0)
        self.assertLess(nl, base_normal)
        self.assertGreater(nr, base_normal)
        self.assertLess(nl + nr, 2.0 * base_normal + 1e-9)


class TestMeasuredMotorConfig(unittest.TestCase):
    """A measured motor replaces the declared one in the config (KI-010)."""

    def _config(self, **extra):
        robot = {"mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
                 "wheel_radius": 0.028, "v_max_rad_s": 18.94,
                 "t_max_nm": 0.138, "gearing": 1.0, "cof": 1.37}
        robot.update(extra)
        return RobotConfig({"robot": robot})

    def test_the_declared_motor_is_untouched_until_it_is_measured(self):
        config = self._config()
        self.assertAlmostEqual(config.t_max_nm, 0.138)
        self.assertAlmostEqual(config.v_max_rad_s, 18.94)

    def test_a_measured_motor_replaces_the_declared_numbers(self):
        config = self._config()
        changes = config.apply_measured_motor(
            t_max_nm=0.120, no_load_speed_rad_s=22.5)
        self.assertAlmostEqual(config.t_max_nm, 0.120)
        self.assertAlmostEqual(config.v_max_rad_s, 22.5)
        self.assertAlmostEqual(changes["t_max_nm"]["measured"], 0.120)
        self.assertAlmostEqual(changes["t_max_nm"]["declared"], 0.138)

    def test_the_derived_torque_range_follows_the_measured_number(self):
        config = self._config()
        declared = config.uncertainties["t_max_range"]
        self.assertAlmostEqual(declared[1], 0.138 * 0.95)
        config.apply_measured_motor(t_max_nm=0.120)
        measured = config.uncertainties["t_max_range"]
        self.assertAlmostEqual(measured[1], 0.120 * 0.95)

    def test_a_declared_range_is_kept(self):
        config = self._config(uncertainties={"t_max_range": [0.10, 0.13]})
        config.apply_measured_motor(t_max_nm=0.120)
        self.assertEqual(config.uncertainties["t_max_range"], (0.10, 0.13))

    def test_a_measured_range_wins_over_the_derived_one(self):
        config = self._config()
        config.apply_measured_motor(t_max_nm=0.120,
                                    t_max_range=(0.115, 0.125))
        self.assertEqual(config.uncertainties["t_max_range"], (0.115, 0.125))

    def test_the_force_limit_moves_with_the_measured_motor(self):
        config = self._config()
        before = config.get_max_force_at_velocity(0.1, apply_headroom=False)
        config.apply_measured_motor(t_max_nm=0.138, no_load_speed_rad_s=22.5)
        after = config.get_max_force_at_velocity(0.1, apply_headroom=False)
        self.assertGreater(after, before)

    def test_nothing_is_applied_when_nothing_was_measured(self):
        config = self._config()
        changes = config.apply_measured_motor()
        self.assertEqual(changes, {})
        self.assertAlmostEqual(config.t_max_nm, 0.138)


class TestUniversalRobotConfig(unittest.TestCase):
    """Test consolidated RobotConfig: geometry, footprint, hardware, sysid merge, pybricks sync."""

    def test_consolidated_parameters_defaults(self):
        cfg = RobotConfig({})
        self.assertEqual(cfg.rolling_resistance, RobotConfig.DEFAULT_ROLLING_RESISTANCE)
        self.assertEqual(cfg.gearbox_coulomb_nm, RobotConfig.DEFAULT_GEARBOX_COULOMB_NM)
        self.assertEqual(cfg.gearbox_viscous_nm_s, RobotConfig.DEFAULT_GEARBOX_VISCOUS_NM_S)
        self.assertEqual(cfg.reflected_inertia_kg_m2, RobotConfig.DEFAULT_REFLECTED_INERTIA)

        self.assertEqual(cfg.width_m, RobotConfig.DEFAULT_WIDTH_M)
        self.assertEqual(cfg.length_m, RobotConfig.DEFAULT_LENGTH_M)
        self.assertEqual(cfg.front_bumper_m, RobotConfig.DEFAULT_FRONT_BUMPER_M)
        self.assertEqual(cfg.back_bumper_m, RobotConfig.DEFAULT_BACK_BUMPER_M)
        self.assertEqual(cfg.lever_arm_m, RobotConfig.DEFAULT_LEVER_ARM_M)
        self.assertEqual(cfg.color_sensor_left_offset_m, RobotConfig.DEFAULT_COLOR_SENSOR_LEFT_OFFSET)
        self.assertEqual(cfg.color_sensor_right_offset_m, RobotConfig.DEFAULT_COLOR_SENSOR_RIGHT_OFFSET)

        self.assertEqual(cfg.hub_top_side, "Axis.Z")
        self.assertEqual(cfg.hub_front_side, "Axis.Y")
        self.assertEqual(cfg.motor_left_port, "Port.F")
        self.assertEqual(cfg.motor_left_dir, "Direction.COUNTERCLOCKWISE")
        self.assertEqual(cfg.motor_right_port, "Port.B")
        self.assertEqual(cfg.motor_right_dir, "Direction.CLOCKWISE")

        self.assertIn("initial_x_m", cfg.uncertainties)
        self.assertIn("initial_y_m", cfg.uncertainties)
        self.assertIn("initial_theta_rad", cfg.uncertainties)

    def test_custom_footprint_and_hardware(self):
        data = {
            "robot": {
                "width_m": 0.17,
                "length_m": 0.20,
                "front_bumper_m": 0.12,
                "back_bumper_m": 0.08,
                "lever_arm_m": 0.11,
                "rolling_resistance": 0.015,
                "gearbox_coulomb_nm": 0.02,
                "gearbox_viscous_nm_s": 0.0003,
                "reflected_inertia_kg_m2": 5e-5,
                "color_sensors": {
                    "left": [0.07, -0.04],
                    "right": [0.07, 0.04],
                },
                "hardware": {
                    "hub_top_side": "Axis.X",
                    "hub_front_side": "Axis.Z",
                    "motor_left_port": "Port.A",
                    "motor_left_dir": "Direction.CLOCKWISE",
                    "motor_right_port": "Port.B",
                    "motor_right_dir": "Direction.COUNTERCLOCKWISE",
                },
                "uncertainties": {
                    "initial_x_m": [-0.003, 0.003],
                    "initial_y_m": [-0.003, 0.003],
                    "initial_theta_rad": [-0.02, 0.02],
                },
            }
        }
        cfg = RobotConfig(data)
        self.assertEqual(cfg.width_m, 0.17)
        self.assertEqual(cfg.front_bumper_m, 0.12)
        self.assertEqual(cfg.rolling_resistance, 0.015)
        self.assertEqual(cfg.color_sensor_left_offset_m, (0.07, -0.04))
        self.assertEqual(cfg.hub_top_side, "Axis.X")
        self.assertEqual(cfg.motor_left_port, "Port.A")
        self.assertEqual(cfg.uncertainties["initial_x_m"], (-0.003, 0.003))
        self.assertGreater(cfg.chassis_radius_m(), 0.12)

    def test_apply_sysid_fit_complete_merge(self):
        from overboard.sysid.digital_twin import DigitalTwinConfig, OdometryFit, StallTorqueFit, DrivetrainFit, ActuatorFit, SlipFit
        cfg = RobotConfig({"robot": {"wheel_radius": 0.028, "track_width": 0.0965, "t_max_nm": 0.04}})
        twin = DigitalTwinConfig(
            drivebase={"wheel_radius_m": 0.0275, "track_width_m": 0.0960},
            odometry=OdometryFit(calibrated=True, distance_scale=1.01, heading_scale=0.99),
            stall_torque=StallTorqueFit(calibrated=True, t_max_nm=0.125, t_max_range=(0.115, 0.130)),
            drivetrain=DrivetrainFit(calibrated=True, no_load_speed_rad_s=21.0),
            actuator_straight=ActuatorFit(calibrated=True, gain_range=(0.98, 1.02)),
            slip=SlipFit(calibrated=True, brake_ratio_min=0.010, brake_ratio_median=0.012, brake_ratio_max=0.015),
            backlash={"calibrated": True, "backlash_m": 0.0003},
        )
        changes = cfg.apply_sysid_fit(twin)
        self.assertIn("wheel_radius", changes)
        self.assertIn("track_width", changes)
        self.assertIn("t_max_nm", changes)
        self.assertIn("gear_backlash_m", changes)
        self.assertIn("rolling_resistance", changes)
        self.assertIn("actuator_gain_range", changes)

        self.assertAlmostEqual(cfg.wheel_radius, 0.0275 * 1.01)
        self.assertAlmostEqual(cfg.track_width, 0.0960 / 0.99)
        self.assertEqual(cfg.uncertainties["gear_backlash_m"], (0.0003 * 0.8, 0.0003 * 1.2))
        self.assertEqual(cfg.rolling_resistance, 0.012)
        self.assertEqual(cfg.uncertainties["actuator_gain_range"], (0.98, 1.02))

    def test_sync_pybricks_robot_config(self):
        import tempfile
        from pathlib import Path
        cfg = RobotConfig({
            "robot": {
                "wheel_radius": 0.030,
                "track_width": 0.100,
                "hardware": {
                    "motor_left_port": "Port.C",
                    "motor_right_port": "Port.D",
                }
            }
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "robot_config.py"
            target.write_text(
                "WHEEL_DIAMETER_MM = 56\nAXLE_TRACK_MM = 96.5\n"
                "motorLeft = Motor(Port.F, Direction.COUNTERCLOCKWISE)\n"
                "motorRight = Motor(Port.B, Direction.CLOCKWISE)\n"
                "hub = PrimeHub(top_side=Axis.Z, front_side=Axis.Y)\n"
            )
            cfg.sync_pybricks_robot_config(target)
            synced = target.read_text()
            self.assertIn("WHEEL_DIAMETER_MM = 60", synced)
            self.assertIn("AXLE_TRACK_MM = 100", synced)
            self.assertIn("motorLeft = Motor(Port.C, Direction.COUNTERCLOCKWISE)", synced)
            self.assertIn("motorRight = Motor(Port.D, Direction.CLOCKWISE)", synced)


class TestUnifiedMathPrimitives(unittest.TestCase):
    """Test cases for unified analytical kinematics and dynamics primitives."""

    def test_diff_drive_fk_ik_roundtrip_scalar(self):
        from overboard.models.robot_model import diff_drive_fk, diff_drive_ik
        track_width = 0.100
        v_in, omega_in = 0.45, 2.5
        vl, vr = diff_drive_ik(v_in, omega_in, track_width)
        v_out, omega_out = diff_drive_fk(vl, vr, track_width)
        self.assertAlmostEqual(v_in, v_out)
        self.assertAlmostEqual(omega_in, omega_out)

    def test_diff_drive_fk_ik_roundtrip_numpy(self):
        from overboard.models.robot_model import diff_drive_fk, diff_drive_ik
        track_width = 0.0965
        v_in = np.array([0.0, 0.2, 0.5, -0.3])
        omega_in = np.array([0.0, 1.2, -3.4, 0.8])
        vl, vr = diff_drive_ik(v_in, omega_in, track_width)
        v_out, omega_out = diff_drive_fk(vl, vr, track_width)
        np.testing.assert_allclose(v_in, v_out, atol=1e-12)
        np.testing.assert_allclose(omega_in, omega_out, atol=1e-12)

    def test_wrap_angle_and_angle_diff(self):
        from overboard.models.robot_model import wrap_angle, angle_diff
        # Wrapping tests
        self.assertAlmostEqual(wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle(np.pi / 2), np.pi / 2)
        self.assertAlmostEqual(wrap_angle(3 * np.pi), -np.pi)
        self.assertAlmostEqual(wrap_angle(-3 * np.pi), -np.pi)

        # Difference tests
        self.assertAlmostEqual(angle_diff(np.pi / 2, 0.0), np.pi / 2)
        self.assertAlmostEqual(angle_diff(0.0, np.pi / 2), -np.pi / 2)
        # Across the [-pi, pi] branch seam
        self.assertAlmostEqual(angle_diff(-3.10, 3.10), 2 * np.pi - 6.20, delta=1e-5)

    def test_diff_drive_wheel_forces_decomposition(self):
        from overboard.models.robot_model import diff_drive_wheel_forces
        mass = 0.8
        inertia = 0.002
        track_width = 0.10
        # Pure straight acceleration: ar = al = 2.0
        fl_s, fr_s = diff_drive_wheel_forces(2.0, 2.0, mass, inertia, track_width)
        self.assertAlmostEqual(fl_s, fr_s)
        self.assertAlmostEqual(fl_s + fr_s, mass * 2.0)

        # Pure angular spin: al = -2.0, ar = 2.0
        fl_t, fr_t = diff_drive_wheel_forces(-2.0, 2.0, mass, inertia, track_width)
        self.assertAlmostEqual(fl_t + fr_t, 0.0)
        alpha = 4.0 / track_width
        expected_ang_force = (inertia * alpha) / track_width
        self.assertAlmostEqual(fr_t, expected_ang_force)
        self.assertAlmostEqual(fl_t, -expected_ang_force)

    def test_kinematic_cornering_ceiling(self):
        from overboard.models.robot_model import kinematic_cornering_ceiling
        track_width = 0.10
        v_wheel_max = 0.80
        # Straight path: kappa = 0 -> ceiling is v_wheel_max
        v_straight = kinematic_cornering_ceiling(0.0, track_width, v_wheel_max)
        self.assertAlmostEqual(float(v_straight), v_wheel_max)

        # Sharp turn: kappa = 10.0 -> ceiling is reduced
        v_curve = kinematic_cornering_ceiling(10.0, track_width, v_wheel_max)
        self.assertLess(float(v_curve), v_wheel_max)

    def test_get_motor_linear_limits(self):
        from overboard.models.robot_model import RobotConfig
        cfg = RobotConfig({"robot": {"t_max_nm": 0.04, "wheel_radius": 0.028, "v_max_rad_s": 15.7, "gearing": 1.0}})
        f_stall, v_noload = cfg.get_motor_linear_limits(apply_headroom=False)
        self.assertAlmostEqual(f_stall, 0.04 / 0.028)
        self.assertAlmostEqual(v_noload, 15.7 * 0.028)

    def test_drivetrain_physics_duck_typing(self):
        """Verify drivetrain_physics operates seamlessly across all 5 backends."""
        from overboard.models.drivetrain_physics import motor_force_at_velocity, diff_drive_wheel_forces, check_drivetrain_constraints
        import casadi as ca
        import jax.numpy as jnp
        from immrax import Interval

        # 1. Float scalar
        f_float = motor_force_at_velocity(0.1, 0.028, 1.0, 0.04, 15.7, 0.85)
        self.assertIsInstance(f_float, float)
        self.assertGreater(f_float, 0.5)

        # 2. NumPy array
        v_np = np.array([0.0, 0.1, 0.2])
        f_np = motor_force_at_velocity(v_np, 0.028, 1.0, 0.04, 15.7, 0.85)
        self.assertEqual(f_np.shape, (3,))

        # 3. JAX array
        v_jax = jnp.array([0.0, 0.1, 0.2])
        f_jax = motor_force_at_velocity(v_jax, 0.028, 1.0, 0.04, 15.7, 0.85)
        np.testing.assert_allclose(np.array(f_np), np.array(f_jax), rtol=1e-5)

        # 4. CasADi symbolic MX
        v_mx = ca.MX.sym("v")
        f_mx = motor_force_at_velocity(v_mx, 0.028, 1.0, 0.04, 15.7, 0.85, smooth=True)
        self.assertTrue(isinstance(f_mx, ca.MX))

        # 5. Immrax Interval
        r_iv = Interval(0.027, 0.029)
        t_iv = Interval(0.035, 0.045)
        v_iv = Interval(0.1, 0.1)
        f_iv = motor_force_at_velocity(v_iv, r_iv, 1.0, t_iv, 15.7, 0.85)
        self.assertTrue(isinstance(f_iv, Interval))
        self.assertLess(f_iv.lower, f_float)
        self.assertGreater(f_iv.upper, f_float)

        # Interval diff_drive_wheel_forces
        L_iv = Interval(0.09, 0.10)
        I_iv = Interval(0.002, 0.003)
        fl_iv, fr_iv = diff_drive_wheel_forces(Interval(0.5, 0.5), Interval(0.5, 0.5), 0.723, I_iv, L_iv)
        self.assertTrue(isinstance(fl_iv, Interval))
        self.assertTrue(isinstance(fr_iv, Interval))

    def test_body_frame_error(self):
        from overboard.models.robot_model import body_frame_error
        # Pure forward displacement along x with 0 heading
        ex, ey = body_frame_error(1.0, 0.0, 0.5, 0.0, 0.0)
        self.assertAlmostEqual(ex, 0.5)
        self.assertAlmostEqual(ey, 0.0)

        # 90 deg rotation: x global displacement becomes -y in body frame
        ex90, ey90 = body_frame_error(1.0, 0.0, 0.0, 0.0, np.pi / 2)
        self.assertAlmostEqual(ex90, 0.0)
        self.assertAlmostEqual(ey90, -1.0)

    def test_asymmetric_late_apex_bump(self):
        from overboard.models.robot_model import asymmetric_late_apex_bump
        # Boundaries must evaluate to 0.0
        self.assertEqual(asymmetric_late_apex_bump(0.0), 0.0)
        self.assertEqual(asymmetric_late_apex_bump(1.0), 0.0)
        # Peak at u_peak=0.68 must be exactly 1.0
        self.assertAlmostEqual(asymmetric_late_apex_bump(0.68, u_peak=0.68), 1.0)
        # Asymmetry check: bump at 0.5 < bump at 0.68
        self.assertLess(asymmetric_late_apex_bump(0.5, u_peak=0.68), 1.0)

    def test_ramsete_gain(self):
        from overboard.models.robot_model import ramsete_gain
        # k1 = 2 * zeta * sqrt(omega^2 + b * v^2)
        k1 = ramsete_gain(v=0.5, omega=1.0, b=2.0, zeta=0.7)
        expected = 2.0 * 0.7 * np.sqrt(1.0**2 + 2.0 * 0.5**2)
        self.assertAlmostEqual(k1, expected)
