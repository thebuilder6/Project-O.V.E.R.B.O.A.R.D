"""Robot deployment bridges, BLE communication, and trajectory code exporters."""

__all__ = [
    "HubNotFound",
    "describe_device",
    "device_names",
    "find_hub",
    "list_hubs",
    "COMPACT_HEADER",
    "COMPACT_VERSION",
    "encode_compact_trajectory",
    "export_controller_json",
    "resample_to_fixed_dt",
    "write_compact_python_file",
    "write_controller_file",
    "write_python_file",
    "HUB_CODE_DIR",
    "HUB_MISSION_MODULE",
    "HUB_RUNNER",
    "HubLinkError",
    "deploy",
    "push_trajectory",
    "write_session",
]


def __getattr__(name: str):
    if name in (
        "HubNotFound",
        "describe_device",
        "device_names",
        "find_hub",
        "list_hubs",
    ):
        from overboard.deploy import hub_ble
        return getattr(hub_ble, name)
    if name in (
        "COMPACT_HEADER",
        "COMPACT_VERSION",
        "encode_compact_trajectory",
        "export_controller_json",
        "resample_to_fixed_dt",
        "write_compact_python_file",
        "write_controller_file",
        "write_python_file",
    ):
        from overboard.deploy import export
        return getattr(export, name)
    if name in (
        "HUB_CODE_DIR",
        "HUB_MISSION_MODULE",
        "HUB_RUNNER",
        "HubLinkError",
        "deploy",
        "push_trajectory",
        "write_session",
    ):
        from overboard.deploy import hub_deploy
        return getattr(hub_deploy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
