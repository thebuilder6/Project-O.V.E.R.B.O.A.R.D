"""
Push a program to a Pybricks hub and capture what it prints.

The Pybricks IDE (code.pybricks.com) is one client of the documented Pybricks
BLE/USB profile. This module uses the official client library for that
profile, so a hub can be driven from a script instead of a browser tab:

    pip install pybricksdev

    python hub_deploy.py pybricks_code/robot.py
    python hub_deploy.py pybricks_code/robot.py --conn usb --out session.csv

To see what the Bluetooth adapter can see, and which of the devices is a
Pybricks hub, use the IDE's device picker:

    python hub_deploy.py --list-hubs

The captured console output is written verbatim, so a session logged by
``pybricks_code/sysid_digital_twin.py`` can be read by ``sysid_fit.py``
without hand-editing.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys

HUB_HINT = "Install the hub tooling with: pip install pybricksdev"

from overboard.paths import HUB_MISSION_MODULE, HUB_RUNNER

# The decision-score contract: a trajectory may only drive the hub when its
# validation record approves it (score_version 1).
SCORE_VERSION = 1

# The hub needs a moment to push its last console lines out before the
# connection goes away.
_DRAIN_SECONDS = 0.3


class HubLinkError(RuntimeError):
    """The link to the hub dropped while the program was running.

    The output captured so far is still on disk, so a caller that can work
    with a partial session should catch this and carry on.
    """


class _LineFeed:
    """Turn the hub's output chunks into complete lines for ``on_line``."""

    def __init__(self, on_line):
        self._on_line = on_line
        self._pending = b""

    def __call__(self, chunk):
        self._pending += chunk
        while b"\n" in self._pending:
            line, self._pending = self._pending.split(b"\n", 1)
            self._on_line(line.decode("utf-8", "replace").rstrip("\r"))

    def flush(self):
        if self._pending:
            self._on_line(self._pending.decode("utf-8", "replace").rstrip("\r"))
            self._pending = b""


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

    from overboard.deploy.hub_ble import find_hub

    # The hub is found by probing what is nearby, the way the Pybricks IDE
    # finds it. pybricksdev's own discovery only accepts an advertisement
    # that already carries the Pybricks service UUID and a local name, which
    # Windows does not usually report.
    device = await find_hub(name)
    return PybricksHubBLE(device)


async def _deploy_async(script_path, connection, name, wait, print_output,
                        stream_path, on_line):
    from pybricksdev.connections.pybricks import HubDisconnectError

    hub = await _connect(connection, name)
    try:
        await hub.connect()
    except Exception as exc:
        raise RuntimeError(
            f"The hub would not take the connection: {exc}\n"
            "Close Pybricks Code (code.pybricks.com) if it is open: it holds "
            "the hub's single Bluetooth connection, and only one central can "
            "be connected at a time. If the hub was just found, try again."
        ) from exc

    stream = None
    subscription = None
    feed = _LineFeed(on_line) if on_line is not None else None
    line_subscription = None
    if stream_path:
        stream = open(stream_path, "wb")
    try:
        if stream is not None:
            # Write the console output as it arrives, so a dropped link
            # costs the last test instead of the whole session.
            subscription = hub.stdout_observable.subscribe(stream.write)
        if feed is not None:
            line_subscription = hub.stdout_observable.subscribe(feed)

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
        if feed is not None:
            feed.flush()
        if line_subscription is not None:
            line_subscription.dispose()
        if subscription is not None:
            subscription.dispose()
        if stream is not None:
            stream.close()
        try:
            await hub.disconnect()
        except Exception:
            pass  # the hub is already gone


def deploy(script_path, connection="ble", name=None, wait=True, print_output=True,
           stream_path=None, on_line=None):
    """Push a script to the hub, run it, and return the lines it printed.

    connection: "ble" or "usb".
    name: Bluetooth name or address of the hub (BLE only).
    wait: keep the connection until the program stops.
    stream_path: write the console output there as it arrives, so a dropped
    link does not lose the session.
    on_line: called with each console line as it arrives, so a caller can
    follow the run live (the session wire format is untouched).
    """
    try:
        return asyncio.run(
            _deploy_async(script_path, connection, name, wait, print_output,
                          stream_path, on_line)
        )
    except ImportError as exc:
        raise RuntimeError(HUB_HINT) from exc


def trajectory_sha256(config, samples):
    """SHA-256 of the canonical JSON for {"config": ..., "samples": ...}.

    Both the writer and the pusher call this helper, so the hash recorded at
    export and the hash checked at push cannot drift apart.
    """
    payload = {"config": config, "samples": samples}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_trajectory_approval(doc):
    """Return the trajectory's validation record, or raise ValueError.

    A hub run needs an ordinary validator pass or an explicitly accepted
    score (score_version 1), and the recorded hash must still match the
    trajectory in the file. An unvalidated, best-effort, stale or edited
    file is refused here - before anything is written to disk or connected.
    """
    validation = doc.get("validation") if isinstance(doc, dict) else None
    if not isinstance(validation, dict):
        raise ValueError(
            "the trajectory carries no validation record, so it is not "
            "approved for the hub")
    if not validation.get("hub_approved"):
        raise ValueError(
            "the trajectory is not approved for the hub: it needs an ordinary "
            "validator pass or an explicitly accepted score (--accept-score)")
    if validation.get("score_version") != SCORE_VERSION:
        raise ValueError(
            "the trajectory's approval uses unsupported score_version "
            f"{validation.get('score_version')!r}, expected {SCORE_VERSION}")
    if validation.get("propagation", "plant") not in ("plant", "tick"):
        raise ValueError(
            "the trajectory's approval uses unsupported propagation "
            f"{validation.get('propagation')!r}, expected 'plant' or 'tick'")
    traj = doc.get("trajectory") or {}
    expected = validation.get("trajectory_sha256")
    actual = trajectory_sha256(traj.get("config"), traj.get("samples"))
    if not expected or expected != actual:
        raise ValueError(
            "the trajectory does not match the hash its approval was issued "
            "for: the file was edited after it was approved")
    return validation


def push_trajectory(traj_file, connection="ble", name=None, log_path=None,
                    mission_module=None, runner=None):
    """Write the packed mission and push it to the hub.

    The hub entry point is pybricks_code/hub_runner.py; it imports the
    trajectory from pybricks_code/hub_mission.py, so both files travel with
    the push. Returns the console lines the hub printed.

    The trajectory must carry an approval its own hash still matches
    (`check_trajectory_approval`), and that is checked before the mission is
    written, the robot config is synced, or a connection is opened.

    log_path, if given, receives the output as it arrives, so a dropped link
    costs the last test instead of the whole session.
    """
    from overboard.deploy.export import write_compact_python_file

    with open(traj_file, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    check_trajectory_approval(doc)

    try:
        from overboard.models.robot_model import RobotConfig
        cfg_dict = doc.get("config") or doc.get("robot")
        if not cfg_dict:
            traj = doc.get("trajectory") or {}
            cfg_dict = traj.get("config")
        if cfg_dict:
            RobotConfig(cfg_dict).sync_pybricks_robot_config()
    except Exception:
        pass

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


def list_hubs(name=None):
    """Show the Bluetooth devices around, like the IDE's device picker.

    Returns the exit code: 0 when a hub answered, 1 when none did.
    """
    from overboard.deploy.hub_ble import describe_device, device_names, list_hubs as _list

    wanted = (name or "").strip().lower()

    async def run():
        print("Scanning for Bluetooth devices...")
        hubs = []
        async for device, advertisement, is_hub in _list():
            label = describe_device(device, advertisement)
            marks = []
            if is_hub:
                marks.append("Pybricks hub")
            names = [text.lower()
                     for text in device_names(device, advertisement)]
            if wanted and any(wanted in text for text in names):
                marks.append("named")
            suffix = "   <- " + ", ".join(marks) if marks else ""
            print(f"  {label}{suffix}")
            if is_hub:
                hubs.append(label)
        return hubs

    try:
        hubs = asyncio.run(run())
    except ImportError:
        print(f"error: {HUB_HINT}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not hubs:
        print("\nNo Pybricks hub answered. Turn it on, and close Pybricks "
              "Code: only one Bluetooth central can hold the hub's single "
              "connection.")
        return 1
    print(f"\nFound {len(hubs)} Pybricks hub(s).")
    return 0


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
    parser.add_argument("--list-hubs", action="store_true",
                        help="List the Bluetooth devices around and say which "
                             "one is a Pybricks hub.")
    parser.add_argument("--out", default=None,
                        help="Write the hub console output to this file as it "
                             "arrives.")
    parser.add_argument("--no-wait", action="store_true",
                        help="Return as soon as the program starts, instead of "
                             "waiting for it to finish.")
    args = parser.parse_args(argv)

    if args.list_hubs:
        return list_hubs(args.name)

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
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.out:
        print(f"Wrote {len(lines)} lines of hub output to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
