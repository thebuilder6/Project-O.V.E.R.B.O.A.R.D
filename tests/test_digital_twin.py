"""Tests for the optional sysid_fit.json digital-twin integration."""

import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from robot_model import RobotConfig

from digital_twin import (
    ActuatorFit,
    DigitalTwinConfig,
    OdometryFit,
    SlipFit,
    load_sysid_fit,
)

try:
    import jax.numpy as jnp
    from immrax import Interval
    from immrax_validator import ImmraxValidator
    from jax_robot_model import JAXRobotConfig
    HAS_IMMRAX = True
except ImportError:
    HAS_IMMRAX = False


_TEST_ROBOT = {"robot": {
    "mass": 0.723, "inertia": 0.001, "track_width": 0.0965,
    "wheel_radius": 0.028, "v_max_rad_s": 18.94, "t_max_nm": 0.138,
    "gearing": 1.0, "cof": 1.37,
}}

_FIT = {
    "actuator": {
        "straight": {
            "calibrated": True,
            "gain": 0.95,
            "gain_min": 0.93,
            "gain_max": 0.97,
            "tau_s": 0.08,
            "delay_s": 0.02,
            "accel_limit": 1.2,
            "rmse": 0.01,
            "n_steps": 2,
        },
        "turn": {
            "calibrated": True,
            "gain": 0.94,
            "tau_s": 0.07,
            "delay_s": 0.01,
            "n_steps": 2,
        },
    },
    "odometry": {
        "calibrated": True,
        "distance_scale": 0.98,
        "heading_scale": 1.01,
        "n_distance_sources": 1,
        "n_heading_sources": 1,
    },
    "slip": {
        "calibrated": True,
        "brake_ratio_median": 0.03,
        "brake_ratio_min": 0.02,
        "brake_ratio_max": 0.04,
    },
    "sensors": {
        "calibrated": True,
        "gyro_z_bias_deg_s": 0.1,
        "gyro_z_std_deg_s": 0.01,
    },
    "drift": {
        "calibrated": True,
        "drift_deg_per_m": 2.0,
        "drift_deg_per_m_ci95": [1.5, 2.5],
        "n_sources": 3,
    },
    "stall_torque": {
        "calibrated": True,
        "t_max_nm": 0.13,
        "t_max_nm_ci95": [0.12, 0.14],
        "n_sources": 3,
    },
    "drivebase": {
        "name": "test drivebase",
        "wheel_radius_m": 0.028,
        "track_width_m": 0.0965,
    },
}


class TestDigitalTwinConfig(unittest.TestCase):
    def test_loads_expected_fields(self):
        twin = DigitalTwinConfig.from_dict(_FIT)
        self.assertTrue(twin.enabled)
        self.assertEqual(twin.actuator_gain_range(), (0.93, 0.97))
        self.assertAlmostEqual(twin.actuator_tau_s(), 0.08)
        self.assertAlmostEqual(twin.actuator_delay_s(), 0.02)
        self.assertAlmostEqual(twin.odometry.distance_scale, 0.98)
        self.assertAlmostEqual(twin.odometry.heading_scale, 1.01)
        self.assertAlmostEqual(twin.slip.brake_ratio_median, 0.03)
        self.assertEqual(twin.drivebase["name"], "test drivebase")
        self.assertAlmostEqual(twin.drivebase["wheel_radius_m"], 0.028)
        self.assertEqual(twin.summary()["drivebase"]["name"], "test drivebase")

    def test_drivebase_geometry_can_come_from_top_level(self):
        twin = DigitalTwinConfig.from_dict({
            "wheel_radius_m": 0.03,
            "track_width_m": 0.11,
            "odometry": {"calibrated": True, "distance_scale": 0.99},
        })
        self.assertAlmostEqual(twin.drivebase["wheel_radius_m"], 0.03)
        self.assertAlmostEqual(twin.drivebase["track_width_m"], 0.11)

    def test_rejects_non_positive_drivebase_geometry(self):
        with self.assertRaises(ValueError):
            DigitalTwinConfig.from_dict({
                "drivebase": {"wheel_radius_m": 0.0},
            })

    def test_uncalibrated_fit_is_disabled(self):
        twin = DigitalTwinConfig.from_dict({
            "actuator": {"straight": {"calibrated": False}},
            "odometry": {"calibrated": False},
            "slip": {"calibrated": False},
        })
        self.assertFalse(twin.enabled)
        self.assertEqual(twin.actuator_gain_range(), (1.0, 1.0))
        self.assertEqual(twin.actuator_tau_s(), 0.0)
        self.assertEqual(twin.actuator_delay_s(), 0.0)
        self.assertFalse(twin.summary()["applied"]["slip_model"])
        self.assertFalse(twin.summary()["applied"]["integrated_odometry"])
        self.assertIsNone(twin.summary()["applied"]["actuator_accel_limit"])

    def test_odometry_only_fit_enables_identity_actuator(self):
        twin = DigitalTwinConfig.from_dict({
            "odometry": {"calibrated": True, "distance_scale": 0.99},
        })
        self.assertTrue(twin.enabled)
        self.assertEqual(twin.actuator_gain_range(), (1.0, 1.0))
        self.assertEqual(twin.actuator_tau_s(), 0.0)

    def test_loads_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sysid_fit.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(_FIT, handle)
            twin = load_sysid_fit(path)
        self.assertTrue(twin.enabled)
        self.assertEqual(twin.actuator_gain_range(), (0.93, 0.97))

    def test_rejects_inverted_gain_range(self):
        with self.assertRaises(ValueError):
            ActuatorFit(calibrated=True, gain_range=(1.1, 0.9))

    def test_rejects_non_positive_odometry_scale(self):
        with self.assertRaises(ValueError):
            OdometryFit(calibrated=True, distance_scale=0.0)

    def test_rejects_inverted_slip_range(self):
        with self.assertRaises(ValueError):
            SlipFit(
                calibrated=True,
                brake_ratio_min=0.04,
                brake_ratio_median=0.03,
                brake_ratio_max=0.02,
            )

    def test_rejects_bad_range_in_fit_dict(self):
        with self.assertRaises(ValueError):
            DigitalTwinConfig.from_dict({
                "actuator": {"straight": {
                    "calibrated": True,
                    "gain_min": 1.1,
                    "gain_max": 0.9,
                }},
            })


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestDigitalTwinValidator(unittest.TestCase):
    def _twin(self):
        return DigitalTwinConfig.from_dict(_FIT)

    def test_enabled_twin_uses_actuator_states(self):
        validator = ImmraxValidator(
            JAXRobotConfig(_TEST_ROBOT), digital_twin=self._twin())
        self.assertTrue(validator._digital_twin_active)
        self.assertEqual(validator.plant.xlen, 5)
        self.assertEqual(validator.digital_twin_gain_range, (0.93, 0.97))
        self.assertAlmostEqual(validator.digital_twin_delay_s, 0.02)

    def test_enabled_twin_produces_a_bounded_envelope(self):
        validator = ImmraxValidator(
            JAXRobotConfig(_TEST_ROBOT), digital_twin=self._twin())
        samples = [
            {"t": 0.0, "x": 0.0, "y": 0.0, "heading": 0.0,
             "vl": 0.0, "vr": 0.0},
            {"t": 0.1, "x": 0.02, "y": 0.0, "heading": 0.0,
             "vl": 0.2, "vr": 0.2},
            {"t": 0.2, "x": 0.05, "y": 0.0, "heading": 0.0,
             "vl": 0.3, "vr": 0.3},
        ]
        report = validator.compute_reachability(
            samples,
            (0.0, 0.0),
            (0.0965, 0.0965),
            (0.028, 0.028),
            (1.37, 1.37),
            (1.0, 1.0),
            (0.001, 0.001),
        )
        self.assertFalse(report["diverged"])
        self.assertEqual(len(report["envelope"]), len(samples))
        self.assertTrue(np.isfinite(report["max_tracking_error_m"]))
        self.assertGreater(report["max_tracking_error_m"], 0.0)

    def test_odometry_scales_nominal_and_interval_poses(self):
        validator = ImmraxValidator.__new__(ImmraxValidator)
        validator.digital_twin = self._twin()
        start = jnp.array([0.0, 0.0, 0.0])
        pose = jnp.array([1.0, 0.5, 0.2])
        scaled = validator._odometry_pose(pose, start)
        self.assertAlmostEqual(float(scaled[0]), 0.98, places=6)
        self.assertAlmostEqual(float(scaled[1]), 0.49, places=6)
        self.assertAlmostEqual(float(scaled[2]), 0.202, places=6)

        interval = Interval(jnp.zeros(3), jnp.array([1.0, 0.5, 0.2]))
        scaled_interval = validator._odometry_pose(interval, start)
        self.assertAlmostEqual(float(scaled_interval.lower[0]), 0.0, places=6)
        self.assertAlmostEqual(float(scaled_interval.upper[0]), 0.98, places=6)
        self.assertAlmostEqual(float(scaled_interval.upper[1]), 0.49, places=6)

    def test_zero_delay_uses_current_command(self):
        command = jnp.array([0.2, 0.2])
        history = [(0.0, command), (0.02, command)]
        selected = ImmraxValidator._delayed_command(history, 0.01)
        self.assertTrue(np.array_equal(np.asarray(selected), np.asarray(command)))

    def test_positive_delay_starts_from_zero(self):
        command = jnp.array([0.2, 0.2])
        history = [(0.0, command)]
        selected = ImmraxValidator._delayed_command(history, -0.05)
        self.assertTrue(np.array_equal(np.asarray(selected), np.zeros(2)))


@unittest.skipUnless(HAS_IMMRAX, "immrax not installed")
class TestDigitalTwinMechanisms(unittest.TestCase):
    """The measured behaviours the plant and the belief now model (KI-020)."""

    def _twin(self, **overrides):
        payload = json.loads(json.dumps(_FIT))
        for key, value in overrides.items():
            payload[key] = value
        return DigitalTwinConfig.from_dict(payload)

    def _validator(self, twin):
        return ImmraxValidator(JAXRobotConfig(_TEST_ROBOT), digital_twin=twin)

    def _flat(self, state_len, vl=0.0, vr=0.0):
        """A flat state array with the given wheel speeds in both bounds."""
        lower = np.zeros(state_len)
        upper = np.zeros(state_len)
        lower[3] = upper[3] = vl
        lower[4] = upper[4] = vr
        return jnp.concatenate([jnp.array(lower), jnp.array(upper)])

    def test_belief_uses_the_odometry_scale_and_gyro_bias(self):
        validator = self._validator(self._twin())
        belief = (0.0, 0.0, 0.0)
        x_flat = self._flat(5, vl=0.2, vr=0.2)

        belief = validator._advance_belief(belief, x_flat, 5, 1.0)

        # The wheel speed 0.2 m/s over one second, scaled by 0.98. The
        # heading drifts while the step runs, so the path curves a little.
        drift = math.radians(0.1)
        self.assertAlmostEqual(belief[0], 0.2 * 0.98 * math.cos(drift),
                               places=6)
        self.assertAlmostEqual(belief[1], 0.2 * 0.98 * math.sin(drift),
                               places=6)
        # The heading drifts with the measured gyro bias: 0.1 deg/s.
        self.assertAlmostEqual(belief[2], drift, places=6)

    def test_belief_turns_with_the_wheel_difference(self):
        validator = self._validator(self._twin())
        belief = (0.0, 0.0, 0.0)
        x_flat = self._flat(5, vl=0.1, vr=0.3)

        belief = validator._advance_belief(belief, x_flat, 5, 1.0)

        # (0.3 - 0.1) / 0.0965 rad/s, scaled by the heading scale 1.01.
        expected = (0.3 - 0.1) / 0.0965 * 1.01 + math.radians(0.1)
        self.assertAlmostEqual(belief[2], expected, places=6)

    def test_no_fit_means_no_belief_parameters(self):
        validator = ImmraxValidator(JAXRobotConfig(_TEST_ROBOT))

        self.assertEqual(validator.digital_twin_distance_scale, 1.0)
        self.assertEqual(validator.digital_twin_heading_scale, 1.0)
        self.assertEqual(validator.digital_twin_gyro_bias_rad_s, 0.0)

    def test_the_measured_drift_curves_the_path(self):
        """A robot whose wheels are not parallel leaves a straight plan."""
        validator = self._validator(self._twin())
        w = jnp.array([1.37, 0.85, 0.0004, 0.0965, 0.028, 0.001, 1.0])
        state = jnp.array([0.0, 0.0, 0.0, 0.4, 0.4])
        command = jnp.array([0.4, 0.4])

        curving = validator.plant.f(0.0, state, command, w)
        straight = self._validator(
            self._twin(drift={"calibrated": False})).plant.f(
                0.0, state, command, w)

        # The wheels are commanded equal, so without the drift the heading
        # does not change. With it, the heading creeps at 2 deg per metre.
        self.assertAlmostEqual(float(straight[2]), 0.0, places=9)
        self.assertAlmostEqual(
            float(curving[2]), math.radians(2.0) * 0.4, places=6)

    def test_no_drift_means_no_plant_term(self):
        validator = self._validator(self._twin(drift={"calibrated": False}))

        self.assertEqual(validator.plant.drift_rad_per_m, 0.0)

    def test_the_measured_motor_is_carried_by_the_config(self):
        """One motor, one number, everywhere.

        The wall test measures the stall torque and the fit hands it over.
        The robot config is where it lives, so the optimizer, the CPU
        audit, the validator and the exported controller all plan against
        the same limit. A second limit in the validator would check the
        plan against a different number than the one it was planned with.
        """
        twin = self._twin()
        config = RobotConfig(_TEST_ROBOT)

        changes = config.apply_measured_motor(**twin.measured_motor())

        self.assertAlmostEqual(config.t_max_nm, 0.13, places=9)
        validator = ImmraxValidator(JAXRobotConfig(config), digital_twin=twin)
        self.assertAlmostEqual(validator.cfg.t_max_nm, 0.13, places=9)
        # The interval follows the measured number, in the config.
        self.assertEqual(tuple(config.uncertainties["t_max_range"]),
                         (0.12, 0.14))

    def test_the_validator_has_no_second_motor_limit(self):
        validator = self._validator(self._twin())

        self.assertFalse(hasattr(validator, "stall_torque_range_nm"))

    def test_braking_slip_shortens_the_ground_covered(self):
        w = jnp.array([1.37, 0.85, 0.0004, 0.0965, 0.028, 0.001, 1.0])
        state = jnp.array([0.0, 0.0, 0.0, 0.01, 0.01])
        command = jnp.array([0.01, 0.01])

        slipping = self._validator(self._twin()).plant.f(0.0, state, command, w)
        gripping = self._validator(
            self._twin(slip={"calibrated": False})).plant.f(
                0.0, state, command, w)

        # The wheels turn the same amount, but the robot covers less ground.
        self.assertLess(float(slipping[0]), float(gripping[0]))
        self.assertAlmostEqual(
            float(slipping[0]) / float(gripping[0]), 1.0 - 0.03, places=2)

    def test_no_slip_when_the_robot_is_at_speed(self):
        w = jnp.array([1.37, 0.85, 0.0004, 0.0965, 0.028, 0.001, 1.0])
        state = jnp.array([0.0, 0.0, 0.0, 0.4, 0.4])
        command = jnp.array([0.4, 0.4])

        slipping = self._validator(self._twin()).plant.f(0.0, state, command, w)
        gripping = self._validator(
            self._twin(slip={"calibrated": False})).plant.f(
                0.0, state, command, w)

        self.assertAlmostEqual(float(slipping[0]), float(gripping[0]), places=6)

    def test_accel_limit_caps_the_wheel_rate(self):
        validator = self._validator(self._twin())
        w = jnp.array([1.37, 0.85, 0.0004, 0.0965, 0.028, 0.001, 1.0])

        # The measured limit is 1.2 m/s^2; ask for far more.
        dx = validator.plant.f(0.0, jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                                jnp.array([5.0, 5.0]), w)

        self.assertLessEqual(abs(float(dx[3])), 1.2 + 1e-6)
        self.assertGreater(float(dx[3]), 0.0)

    def test_summary_reports_what_is_applied(self):
        twin = self._twin()
        summary = twin.summary()

        self.assertAlmostEqual(summary["odometry"]["distance_scale"], 0.98)
        self.assertAlmostEqual(summary["sensors"]["gyro_z_bias_deg_s"], 0.1)
        self.assertTrue(summary["slip"]["calibrated"])
        self.assertTrue(summary["applied"]["slip_model"])
        self.assertTrue(summary["applied"]["integrated_odometry"])
        self.assertAlmostEqual(
            summary["applied"]["actuator_accel_limit"], 1.2)


if __name__ == "__main__":
    unittest.main()


class TestMeasuredMotor(unittest.TestCase):
    """The measured motor numbers reach the config (KI-010)."""

    def _fit(self, **drivetrain):
        payload = dict(_FIT)
        payload["drivetrain"] = drivetrain or {
            "calibrated": True,
            "no_load_speed_rad_s": 22.5,
            "no_load_speed_rad_s_ci95": [22.0, 23.0],
            "n_sources": 3,
        }
        return DigitalTwinConfig.from_dict(payload)

    def test_the_no_load_speed_is_read(self):
        twin = self._fit()
        self.assertAlmostEqual(twin.no_load_speed_rad_s(), 22.5)

    def test_the_measured_motor_carries_both_ends_of_the_curve(self):
        twin = self._fit()
        measured = twin.measured_motor()
        self.assertAlmostEqual(measured["t_max_nm"], 0.13)
        self.assertEqual(measured["t_max_range"], (0.12, 0.14))
        self.assertAlmostEqual(measured["no_load_speed_rad_s"], 22.5)

    def test_the_gearing_moves_the_speed_to_the_motor(self):
        twin = self._fit()
        measured = twin.measured_motor(gearing=2.0)
        self.assertAlmostEqual(measured["no_load_speed_rad_s"], 45.0)

    def test_an_uncalibrated_fit_keeps_the_declared_motor(self):
        payload = dict(_FIT)
        twin = DigitalTwinConfig.from_dict(payload)
        self.assertIsNone(twin.no_load_speed_rad_s())
        self.assertNotIn("no_load_speed_rad_s", twin.measured_motor())

    def test_the_summary_reports_the_drivetrain(self):
        summary = self._fit().summary()
        self.assertTrue(summary["drivetrain"]["calibrated"])
        self.assertAlmostEqual(
            summary["drivetrain"]["no_load_speed_rad_s"], 22.5)
        self.assertTrue(summary["applied"]["measured_motor"])


class TestMeasuredMotorEndToEnd(unittest.TestCase):
    """--sysid-fit puts the measured motor into the solver's config.

    Before this, the optimizer planned against the declared motor while the
    validator checked the measured one, so a plan could satisfy its own
    constraint and still trip the validator (KI-010).
    """

    def test_the_solver_gets_the_measured_motor(self):
        try:
            import main  # noqa: F401
            from click.testing import CliRunner
        except ImportError:
            self.skipTest("click is not installed")
        if not HAS_IMMRAX:
            self.skipTest("JAX and Immrax are not installed")

        recorded = {}

        class StubOptimizer:
            def __init__(self, config):
                recorded["config"] = config

            def solve(self, waypoints, **kwargs):
                samples = [{
                    "t": i * 0.1, "x": 0.02 * i, "y": 0.0, "heading": 0.0,
                    "vl": 0.2, "vr": 0.2, "omega": 0.0,
                    "al": 0.0, "ar": 0.0, "fl": 0.0, "fr": 0.0,
                } for i in range(11)]
                return samples, {"converged": True, "final_cost": 1.0}

        fit = {
            "actuator": {"straight": {"calibrated": True, "gain": 0.95}},
            "stall_torque": {
                "calibrated": True,
                "t_max_nm": 0.110,
                "t_max_nm_ci95": [0.105, 0.115],
                "n_sources": 3,
            },
            "drivetrain": {
                "calibrated": True,
                "no_load_speed_rad_s": 21.5,
                "no_load_speed_rad_s_ci95": [21.0, 22.0],
                "n_sources": 3,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.traj")
            wp = os.path.join(tmp, "wp.json")
            fit_path = os.path.join(tmp, "fit.json")
            with open(wp, "w", encoding="utf-8") as handle:
                json.dump([{"x": 0.0, "y": 0.0}, {"x": 0.2, "y": 0.0}], handle)
            with open(fit_path, "w", encoding="utf-8") as handle:
                json.dump(fit, handle)
            import main
            with patch.object(main, "TrajectoryOptimizer", StubOptimizer):
                result = CliRunner().invoke(
                    main.main,
                    ["-w", wp, "-o", out, "--sysid-fit", fit_path, "--validate"])

        self.assertIn("config", recorded, result.output)
        config = recorded["config"]
        self.assertAlmostEqual(config.t_max_nm, 0.110)
        self.assertAlmostEqual(config.v_max_rad_s, 21.5)
        # The measured stall torque's interval replaces the derived one.
        self.assertEqual(config.uncertainties["t_max_range"], (0.105, 0.115))
        self.assertIn("Measured motor", result.output)
