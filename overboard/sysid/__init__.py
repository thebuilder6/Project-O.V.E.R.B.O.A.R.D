"""System identification, hardware parameter fitting, and digital twin."""

__all__ = [
    "ActuatorFit",
    "ConfidenceInterval",
    "DigitalTwin",
    "DigitalTwinConfig",
    "DrivetrainFit",
    "MotorFit",
    "OdometryFit",
    "SlipFit",
    "StallTorqueFit",
    "WheelFit",
    "load_sysid_fit",
    "fit_session",
    "load_poses",
    "load_runs",
    "parse_session",
    "poses_from_hub_session",
    "read_drivebase_profile",
    "read_sheet",
    "repeat_statistics",
    "session_tests",
    "sheet_rows",
    "sheet_to_poses",
    "write_poses_csv",
    "write_sheet",
    "SHEET_FIELDS",
    "write_form",
    "compare",
]


def __getattr__(name: str):
    if name in (
        "ActuatorFit",
        "ConfidenceInterval",
        "DigitalTwin",
        "DigitalTwinConfig",
        "DrivetrainFit",
        "MotorFit",
        "OdometryFit",
        "SlipFit",
        "StallTorqueFit",
        "WheelFit",
        "load_sysid_fit",
    ):
        from overboard.sysid import digital_twin
        return getattr(digital_twin, name)
    if name in (
        "fit_session",
        "load_poses",
        "load_runs",
        "parse_session",
        "poses_from_hub_session",
        "read_drivebase_profile",
        "read_sheet",
        "repeat_statistics",
        "session_tests",
        "sheet_rows",
        "sheet_to_poses",
        "write_poses_csv",
        "write_sheet",
    ):
        from overboard.sysid import fit
        return getattr(fit, name)
    if name in ("SHEET_FIELDS", "write_form"):
        from overboard.sysid import sheet_form
        return getattr(sheet_form, name)
    if name == "compare":
        from overboard.sysid import compare
        return getattr(compare, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
