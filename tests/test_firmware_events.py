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
import re
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


class TestFirmwareParity(unittest.TestCase):
    """The validator must model the controller that actually runs (KI-028).

    The wheel-force hold error is proportional to how long the drive command
    is held, so a validator that models a different control period from the
    firmware reports a different tracking error - and the gate is decided on
    that number. The controllers that are in use are the two compact ones and
    the list-of-dict reference; they must all hold the command for the same
    time, and the validator must hold it for that time too.
    """

    def _declared_period_ms(self, module_name):
        """The control period a firmware module declares, in milliseconds.

        A module either names it as a constant or sleeps for it directly.
        """
        path = os.path.join(_FIRMWARE_DIR, module_name)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        match = re.search(r"_LOOP_MS\s*=\s*const\((\d+)\)", text)
        if match:
            return int(match.group(1))
        match = re.search(r"CONTROL_MS\s*=\s*(\d+)", text)
        if match:
            return int(match.group(1))
        waits = [int(w) for w in re.findall(r"await wait\((\d+)\)", text)]
        self.assertTrue(waits, f"{module_name} declares no control period")
        return max(waits)

    def test_the_controllers_in_use_hold_the_command_for_the_same_time(self):
        periods = {}
        for module in ("ramsete_compact.py", "ramsete_compact_events.py",
                       "ramsete.py"):
            periods[module] = self._declared_period_ms(module)
        self.assertEqual(len(set(periods.values())), 1,
                         f"the controllers disagree on the hold: {periods}")

    def test_the_validator_holds_the_command_for_the_firmware_period(self):
        from immrax_validator import CONTROL_DT
        period_ms = self._declared_period_ms("ramsete_compact_events.py")
        self.assertAlmostEqual(CONTROL_DT, period_ms / 1000.0, places=9,
                               msg="the validator's control period does not "
                                   "match the firmware's")

    def test_the_two_compact_controllers_share_the_same_parts(self):
        """They must not be able to drift apart.

        Both use the odometry, the controller and the limiter from
        ramsete.py, so a change to the shared behaviour reaches both.
        """
        for module in ("ramsete_compact.py", "ramsete_compact_events.py"):
            path = os.path.join(_FIRMWARE_DIR, module)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            for part in ("Odometry", "RamseteController", "KinematicLimiter"):
                self.assertIn(part, text,
                              f"{module} does not use the shared {part}")

    def test_the_validator_does_not_round_the_command(self):
        """The controllers that are in use pass floats to drive().

        The legacy ramsete_fast.py rounds, and so did the validator; a
        rounded command is a different command, so the model has to agree
        with the firmware that runs.
        """
        from immrax_validator import RamseteController
        import jax.numpy as jnp
        controller = RamseteController(2.0, 0.7, track_width=0.0965)
        pose = jnp.array([0.0, 0.0, 0.0])
        ref = jnp.array([0.0, 0.0, 0.0, 0.1234, 0.0, 0.0])
        w = jnp.array([1.0, 1.0, 0.0, 0.0965, 0.028, 0.001, 1.0])
        rounded = controller(0.0, pose, ref, w, normalize=True, quantize=True)
        raw = controller(0.0, pose, ref, w, normalize=True, quantize=False)
        self.assertNotAlmostEqual(float(rounded[0]), float(raw[0]), places=6)


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


class _ScriptedClock:
    """A clock that jumps forward on every sleep.

    The control loop reads the clock and sleeps until the next tick. A real
    clock rarely lands exactly on time, and one late tick is enough for the
    sample index to jump over the sample an event is attached to. This clock
    makes that happen on purpose, so the event path is tested instead of
    hoped for.
    """

    def __init__(self, step_ms=60):
        self.now_ms = 0
        self.step_ms = step_ms

    def time(self):
        return self.now_ms

    async def wait(self, _ms):
        self.now_ms += self.step_ms


class _JumpyClock:
    """Installs a scripted clock into a firmware module, then puts it back."""

    def __init__(self, module, step_ms=60):
        self.module = module
        self.step_ms = step_ms
        self.clock = None

    def __enter__(self):
        clock = self.clock = _ScriptedClock(self.step_ms)
        self._watch = self.module.StopWatch
        self._wait = self.module.wait
        self.module.StopWatch = lambda: clock
        self.module.wait = clock.wait
        return clock

    def __exit__(self, *exc):
        self.module.StopWatch = self._watch
        self.module.wait = self._wait
        return False


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


class TestLateTickEvents(unittest.TestCase):
    """KI-029: an event must fire even when a late tick skips its sample.

    The control period and the sample spacing are both 20 ms, so a tick
    that arrives late jumps the sample index over one or more samples. The
    event of a sample that was jumped over used to be never looked at, and
    the mission action was silently lost - the same failure KI-012 was
    about, in a different place.
    """

    def _samples_with_a_mid_event(self):
        samples = _samples()
        samples[2]["event"] = "lower_arm"
        return samples

    def test_the_reference_controller_fires_an_event_a_late_tick_skipped(self):
        fired = []
        with _JumpyClock(ramsete, step_ms=60):
            asyncio.run(ramsete.follow_trajectory(
                _FakeDriveBase(), self._samples_with_a_mid_event(), event_map={
                    "lower_arm": lambda: fired.append("lower_arm")},
            ))
        self.assertEqual(fired, ["lower_arm"],
                         "the event on the skipped sample was never fired")

    def test_the_deployed_controller_fires_an_event_a_late_tick_skipped(self):
        """The controller that runs on the hub must not lose one either."""
        fired = []
        blob, events = encode_compact_trajectory(self._samples_with_a_mid_event())
        with _JumpyClock(ramsete_compact_events, step_ms=60):
            asyncio.run(ramsete_compact_events.follow_trajectory(
                _FakeDriveBase(), blob, events=events, event_map={
                    "lower_arm": lambda: fired.append("lower_arm")},
            ))
        self.assertEqual(fired, ["lower_arm"],
                         "the event on the skipped sample was never fired")

    def test_the_event_fires_once_and_not_twice(self):
        fired = []
        with _JumpyClock(ramsete, step_ms=60):
            asyncio.run(ramsete.follow_trajectory(
                _FakeDriveBase(), self._samples_with_a_mid_event(), event_map={
                    "lower_arm": lambda: fired.append("lower_arm")},
            ))
        self.assertEqual(len(fired), 1)


if __name__ == "__main__":
    unittest.main()
