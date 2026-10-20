"""
Tests for the Pybricks event path (KI-012).

The firmware modules cannot run on the host as they are, because they import
``pybricks``, ``umath``, ``ustruct`` and ``micropython``. They are stubbed
here so the deployed control loops can be executed on the host: the event
bookkeeping, the sample advance and the loop timing are the real code, only
the hardware calls are fake.

KI-012 was a silent mission failure: the loop stops before the final sample,
so an event attached to the last waypoint was exported but never fired.
"""

import asyncio
import math
import os
import struct
import sys
import types
import unittest

from tests.firmware_stubs import install_firmware_stubs as _install_firmware_stubs


class _GcStub:
    """MicroPython's gc.threshold() has no CPython equivalent."""

    @staticmethod
    def collect():
        pass

    @staticmethod
    def threshold(_value):
        pass


_install_firmware_stubs()

# The firmware modules import each other by name, so the folder has to be on
# the path. Use an absolute path so the test works from any directory.
_FIRMWARE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pybricks_code")
sys.path.insert(0, _FIRMWARE_DIR)

import ramsete  # noqa: E402  (firmware module, imported after the stubs)
import ramsete_compact_events  # noqa: E402

ramsete.gc = _GcStub()
ramsete_compact_events.gc = _GcStub()

from export import encode_compact_trajectory  # noqa: E402


class _FakeDriveBase:
    """Just enough DriveBase for the controller loop."""

    def __init__(self):
        self.commands = []
        self.stopped = False

    def drive(self, speed, turn_rate):
        self.commands.append((speed, turn_rate))

    def stop(self):
        self.stopped = True

    def distance(self):
        return 0

    def angle(self):
        return 0.0


def _samples(final_event=None):
    """A short straight-line trajectory with an optional final event."""
    samples = []
    for i in range(6):
        t = i * 0.02
        samples.append({
            "t": t,
            "x": 0.1 * t,
            "y": 0.0,
            "heading": 0.0,
            "vl": 0.1,
            "vr": 0.1,
            "omega": 0.0,
        })
    if final_event is not None:
        samples[-1]["event"] = final_event
    return samples


class TestFinalWaypointEvent(unittest.TestCase):
    """KI-012: the event on the final waypoint must fire."""

    def test_final_sample_event_fires(self):
        fired = []

        asyncio.run(ramsete.follow_trajectory(
            _FakeDriveBase(), _samples("release"), event_map={
                "release": lambda: fired.append("release")},
        ))

        self.assertEqual(fired, ["release"])

    def test_mid_trajectory_event_fires_once(self):
        fired = []
        samples = _samples()
        samples[2]["event"] = "lower_arm"

        asyncio.run(ramsete.follow_trajectory(
            _FakeDriveBase(), samples, event_map={
                "lower_arm": lambda: fired.append("lower_arm")},
        ))

        self.assertEqual(fired, ["lower_arm"])

    def test_unknown_event_name_is_ignored(self):
        fired = []

        asyncio.run(ramsete.follow_trajectory(
            _FakeDriveBase(), _samples("release"), event_map={}))

        self.assertEqual(fired, [])

    def test_event_fires_after_the_drivebase_stops(self):
        """The robot must be stopped before the final action runs."""
        order = []
        drivebase = _FakeDriveBase()
        original_stop = drivebase.stop

        def stop():
            order.append("stop")
            original_stop()

        drivebase.stop = stop

        asyncio.run(ramsete.follow_trajectory(
            drivebase, _samples("release"), event_map={
                "release": lambda: order.append("release")},
        ))

        self.assertEqual(order, ["stop", "release"])


class TestFinalWaypointEventCompact(unittest.TestCase):
    """KI-012 for the packed event controller."""

    def test_final_sample_event_fires(self):
        fired = []
        blob, events = encode_compact_trajectory(_samples("release"))

        asyncio.run(ramsete_compact_events.follow_trajectory(
            _FakeDriveBase(), blob, events=events, event_map={
                "release": lambda: fired.append("release")},
        ))

        self.assertEqual(fired, ["release"])
        self.assertEqual(events, ((len(_samples()) - 1, "release"),))

    def test_mid_trajectory_event_fires_once(self):
        fired = []
        samples = _samples()
        samples[2]["event"] = "lower_arm"
        blob, events = encode_compact_trajectory(samples)

        asyncio.run(ramsete_compact_events.follow_trajectory(
            _FakeDriveBase(), blob, events=events, event_map={
                "lower_arm": lambda: fired.append("lower_arm")},
        ))

        self.assertEqual(fired, ["lower_arm"])

    def test_follow_trajectory_reports_the_final_pose(self):
        """The host reads the measured pose back from the hub."""
        samples = _samples()
        samples[0].update({"x": 0.25, "y": -0.1, "heading": 0.3})
        blob, events = encode_compact_trajectory(samples)

        pose = asyncio.run(ramsete_compact_events.follow_trajectory(
            _FakeDriveBase(), blob, events=events,
        ))

        self.assertEqual(len(pose), 3)
        x, y, heading = pose
        self.assertAlmostEqual(x, 0.25, places=5)
        self.assertAlmostEqual(y, -0.1, places=5)
        self.assertAlmostEqual(heading, 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
