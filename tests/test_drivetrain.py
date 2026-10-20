"""Tests for the drivetrain model."""

import math
import unittest

from drivetrain import (
    DrivetrainModel,
    MotorSpec,
    SPIKE_PRIME_MEDIUM,
    compare_declared_motor,
    gearbox_drag,
    motor_torque,
)


def _model(spec=SPIKE_PRIME_MEDIUM, **kwargs):
    values = {
        "spec": spec,
        "wheel_radius_m": 0.028,
        "mass_kg": 0.723,
        "gearing": 1.0,
        "track_width_m": 0.0965,
        "coefficient_of_friction": 1.37,
    }
    values.update(kwargs)
    return DrivetrainModel(**values)


class TestMotorLaw(unittest.TestCase):
    def test_torque_is_the_stall_torque_at_standstill(self):
        torque = motor_torque(SPIKE_PRIME_MEDIUM, 1.0, 0.0)
        self.assertAlmostEqual(torque, SPIKE_PRIME_MEDIUM.stall_torque_nm)

    def test_torque_is_zero_at_the_no_load_speed(self):
        omega = SPIKE_PRIME_MEDIUM.no_load_speed_rad_s
        torque = motor_torque(SPIKE_PRIME_MEDIUM, 1.0, omega)
        self.assertAlmostEqual(torque, 0.0, places=12)

    def test_torque_falls_linearly_with_the_speed(self):
        half = SPIKE_PRIME_MEDIUM.no_load_speed_rad_s / 2.0
        torque = motor_torque(SPIKE_PRIME_MEDIUM, 1.0, half)
        self.assertAlmostEqual(
            torque, 0.5 * SPIKE_PRIME_MEDIUM.stall_torque_nm, places=12)

    def test_the_motor_brakes_against_its_own_back_emf(self):
        """A motor driven against its back EMF is a generator."""
        omega = 10.0
        braking = motor_torque(SPIKE_PRIME_MEDIUM, 0.0, omega)
        self.assertLess(braking, 0.0)
        # The braking torque grows with the speed.
        self.assertLess(
            motor_torque(SPIKE_PRIME_MEDIUM, 0.0, 2.0 * omega), braking)

    def test_the_drag_opposes_the_motion_and_is_symmetric(self):
        forward = gearbox_drag(SPIKE_PRIME_MEDIUM, 5.0)
        backward = gearbox_drag(SPIKE_PRIME_MEDIUM, -5.0)
        self.assertGreater(forward, 0.0)
        self.assertAlmostEqual(forward, backward)


class TestDrivetrainModel(unittest.TestCase):
    def test_the_drive_force_falls_to_zero_at_the_no_load_speed(self):
        model = _model()
        speed = model.spec.no_load_speed_rad_s * model.wheel_radius_m
        self.assertAlmostEqual(model.drive_force(speed), 0.0, places=9)
        self.assertGreater(model.drive_force(0.0), 0.0)

    def test_the_drive_force_falls_with_the_speed(self):
        """That is the whole point of the curve: less push as it speeds up."""
        model = _model()
        self.assertGreater(model.drive_force(0.05), model.drive_force(0.1))

    def test_the_braking_force_is_limited_by_the_tyres(self):
        model = _model()
        # A fast wheel: the motor could brake harder than the mat can hold.
        decel = model.brake_deceleration(10.0)
        self.assertLessEqual(decel, model.traction_limit() / model.mass_kg + 1e-9)

    def test_the_coast_is_slower_than_the_brake(self):
        model = _model()
        speed = 0.3
        self.assertLess(model.coast_deceleration(speed),
                        model.brake_deceleration(speed))

    def test_the_step_response_reaches_the_commanded_speed(self):
        model = _model()
        rows = model.step_response(lambda t: 0.2, duration_s=2.0, dt_s=0.002)
        final = rows[-1]
        self.assertAlmostEqual(final[1], 0.2, delta=0.02)
        self.assertGreater(final[2], 0.3)

    def test_the_step_response_stops_when_the_command_stops(self):
        model = _model()
        rows = model.step_response(
            lambda t: 0.2 if t < 1.0 else 0.0, duration_s=2.0, dt_s=0.002)
        self.assertGreater(rows[-1][1], 0.0)
        self.assertLess(rows[-1][1], rows[len(rows) // 2][1])

    def test_a_bad_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            MotorSpec("x", 0.0, 10.0, 9.0)
        with self.assertRaises(ValueError):
            MotorSpec("x", 0.1, 0.0, 9.0)
        with self.assertRaises(ValueError):
            _model(wheel_radius_m=0.0)


class TestDeclaredMotorCheck(unittest.TestCase):
    def test_the_reference_robot_matches_the_medium_motor(self):
        check = compare_declared_motor(
            stall_torque_nm=0.138,
            no_load_speed_rad_s=18.94,
            wheel_radius_m=0.028,
        )
        self.assertFalse(check["suspicious"])
        self.assertIn("medium", check["closest"])
        self.assertIn("medium", check["note"])

    def test_a_guess_that_is_far_off_is_flagged(self):
        check = compare_declared_motor(
            stall_torque_nm=5.0,
            no_load_speed_rad_s=200.0,
            wheel_radius_m=0.028,
        )
        self.assertTrue(check["suspicious"])
        self.assertIn("guess", check["note"])

    def test_a_non_positive_motor_is_flagged(self):
        check = compare_declared_motor(0.0, 18.94, 0.028)
        self.assertTrue(check["suspicious"])


if __name__ == "__main__":
    unittest.main()
