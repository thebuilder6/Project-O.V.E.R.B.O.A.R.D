"""
Push a program to a Pybricks hub and capture what it prints.

The Pybricks IDE (code.pybricks.com) is one client of the documented Pybricks
BLE/USB profile. This module uses the official client library for that
profile, so a hub can be driven from a script instead of a browser tab:

    pip install pybricksdev

    python hub_deploy.py pybricks_code/robot.py
    python hub_deploy.py pybricks_code/robot.py --conn usb --out session.csv

The captured console output is written verbatim, so a session logged by
``pybricks_code/sysid_digital_twin.py`` can be read by ``sysid_fit.py``
without hand-editing.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

HUB_HINT = "Install the hub tooling with: pip install pybricksdev"

# The hub side of a push: the runner travels with the packed mission.
HUB_CODE_DIR = Path(__file__).resolve().parent / "pybricks_code"
HUB_MISSION_MODULE = HUB_CODE_DIR / "hub_mission.py"
HUB_RUNNER = HUB_CODE_DIR / "hub_runner.py"

# The hub needs a moment to push its last console lines out before the
# connection goes away.
_DRAIN_SECONDS = 0.3


class HubLinkError(RuntimeError):
    """The link to the hub dropped while the program was running.

    The output captured so far is still on disk, so a caller that can work
    with a partial session should catch this and carry on.
    """


def _usb_hub():
    """Find a Pybricks hub on USB."""
    from usb.core import find as find_usb
    from pybricksdev.usb import (
        LEGO_USB_VID,
        MINDSTORMS_INVENTOR_USB_PID,
        SPIKE_ESSENTIAL_USB_PID,
        SPIKE_PRIME_USB_PID,
    )

    pids = (SPIKE_PRIME_USB_PID, SPIKE_ESSENTIAL_USB_PID, MINDSTORMS_INVENTOR_USB_PID)

    def is_pybricks(device):
        product = getattr(device, "product", None)
        return (
            device.idVendor == LEGO_USB_VID
            and device.idProduct in pids
            and product is not None
            and product.endswith("Pybricks")
        )

    device = find_usb(custom_match=is_pybricks)
    if device is None:
        raise RuntimeError(
            "No Pybricks hub found on USB. On Windows the hub needs a WinUSB "
            "driver (Zadig); over BLE no driver is needed."
        )
    return device


async def _connect(connection, name):
    """Return an unconnected pybricksdev hub for the requested transport."""
    from pybricksdev.connections.pybricks import PybricksHubBLE, PybricksHubUSB

    if connection == "usb":
        return PybricksHubUSB(_usb_hub())

    from pybricksdev.ble import find_device

    try:
        device = await find_device(name)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "No Pybricks hub found over BLE. Turn it on, close Pybricks Code "
            "(only one BLE central can connect), and pair it once if Windows "
            "asks."
        ) from exc
    return PybricksHubBLE(device)


async def _deploy_async(script_path, connection, name, wait, print_output,
                        stream_path):
    from pybricksdev.connections.pybricks import HubDisconnectError

    hub = await _connect(connection, name)
    stream = None
    subscription = None
    if stream_path:
        stream = open(stream_path, "wb")
    await hub.connect()
    try:
        if stream is not None:
            # Write the console output as it arrives, so a dropped link
            # costs the last test instead of the whole session.
            subscription = hub.stdout_observable.subscribe(stream.write)

        # A running program blocks the download, so stop it first.
        try:
            await hub.stop_user_program()
        except Exception:
            pass  # nothing was running

        try:
            await hub.run(script_path, wait=wait, print_output=print_output)
        except HubDisconnectError as exc:
            raise HubLinkError(str(exc)) from exc
        await asyncio.sleep(_DRAIN_SECONDS)
        return [line.decode("utf-8", "replace") for line in hub.output]
    finally:
        if subscription is not None:
            subscription.dispose()
        if stream is not None:
            stream.close()
        try:
            await hub.disconnect()
        except Exception:
            pass  # the hub is already gone


def deploy(script_path, connection="ble", name=None, wait=True, print_output=True,
           stream_path=None):
    """Push a script to the hub, run it, and return the lines it printed.

    connection: "ble" or "usb".
    name: Bluetooth name or address of the hub (BLE only).
    wait: keep the connection until the program stops.
    stream_path: write the console output there as it arrives, so a dropped
    link does not lose the session.
    """
    try:
        return asyncio.run(
            _deploy_async(script_path, connection, name, wait, print_output,
                          stream_path)
        )
    except ImportError as exc:
        raise RuntimeError(HUB_HINT) from exc


def push_trajectory(traj_file, connection="ble", name=None, log_path=None,
                    mission_module=None, runner=None):
    """Write the packed mission and push it to the hub.

    The hub entry point is pybricks_code/hub_runner.py; it imports the
    trajectory from pybricks_code/hub_mission.py, so both files travel with
    the push. Returns the console lines the hub printed.

    log_path, if given, receives the output as it arrives, so a dropped link
    costs the last test instead of the whole session.
    """
    from export import write_compact_python_file

    mission_module = mission_module or HUB_MISSION_MODULE
    runner = runner or HUB_RUNNER
    write_compact_python_file(traj_file, str(mission_module))
    if log_path:
        return deploy(str(runner), connection=connection, name=name,
                      stream_path=log_path)
    return deploy(str(runner), connection=connection, name=name)


def write_session(lines, path):
    """Write the hub console output verbatim.

    The sysid logger keeps its TEST_START / META / DATA structure in stdout,
    so the file can be read by sysid_fit.py as it is.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Push a program to a Pybricks hub and capture its output."
    )
    parser.add_argument("script", nargs="?",
                        help="Python script to push to the hub.")
    parser.add_argument("--trajectory",
                        help="Trajectory to push instead of a script: the "
                             "packed mission is written and hub_runner.py runs "
                             "it on the hub.")
    parser.add_argument("--conn", choices=("ble", "usb"), default="ble",
                        help="How to reach the hub (default: ble).")
    parser.add_argument("--name", default=None,
                        help="Bluetooth name or address of the hub.")
    parser.add_argument("--out", default=None,
                        help="Write the hub console output to this file as it "
                             "arrives.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Return as soon as the program starts, instead of "
                             "waiting for it to finish.")
    args = parser.parse_args(argv)

    if args.trajectory and args.script:
        parser.error("give a script or --trajectory, not both")
    if not args.trajectory and not args.script:
        parser.error("give a script to push, or --trajectory")

    try:
        if args.trajectory:
            if not os.path.exists(args.trajectory):
                print(f"error: trajectory '{args.trajectory}' not found",
                      file=sys.stderr)
                return 1
            lines = push_trajectory(args.trajectory, connection=args.conn,
                                    name=args.name, log_path=args.out)
        else:
            lines = deploy(args.script, connection=args.conn, name=args.name,
                           wait=not args.no_wait, stream_path=args.out)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out:
        print(f"Wrote {len(lines)} lines of hub output to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
