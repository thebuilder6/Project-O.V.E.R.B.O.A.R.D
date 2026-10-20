"""Tests for the optional sysid_fit.json digital-twin integration."""

import json
import os
import tempfile
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
