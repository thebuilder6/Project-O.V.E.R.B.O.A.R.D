"""Robot kinematics, dynamics, physical models, and tracking controllers."""

__all__ = [
    "DrivetrainModel",
    "MotorSpec",
    "CG_HEIGHT_M",
    "diff_drive_fk",
    "diff_drive_ik",
    "diff_drive_wheel_forces",
    "wheel_normal_forces",
    "motor_force_at_velocity",
    "motor_linear_limits",
    "traction_limits",
    "total_traction_limit",
    "check_drivetrain_constraints",
    "kinematic_cornering_ceiling",
    "wrap_angle",
    "angle_diff",
    "body_frame_error",
    "asymmetric_late_apex_bump",
    "ramsete_gain",
    "RobotConfig",
    "MotorConfig",
    "DifferentialDriveModel",
]


def __getattr__(name: str):
    if name in ("DrivetrainModel", "MotorSpec"):
        from overboard.models import drivetrain
        return getattr(drivetrain, name)
    if name in (
        "CG_HEIGHT_M",
        "diff_drive_fk",
        "diff_drive_ik",
        "diff_drive_wheel_forces",
        "wheel_normal_forces",
        "motor_force_at_velocity",
        "motor_linear_limits",
        "traction_limits",
        "total_traction_limit",
        "check_drivetrain_constraints",
        "kinematic_cornering_ceiling",
        "wrap_angle",
        "angle_diff",
        "body_frame_error",
        "asymmetric_late_apex_bump",
        "ramsete_gain",
    ):
        from overboard.models import drivetrain_physics
        return getattr(drivetrain_physics, name)
    if name in ("RobotConfig", "MotorConfig", "DifferentialDriveModel"):
        from overboard.models import robot_model
        return getattr(robot_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
