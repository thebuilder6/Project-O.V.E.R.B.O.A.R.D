"""
RAMSETE COMPACT CONTROLLER (Pybricks MicroPython)
-------------------------------------------------
RAM-lean pathing controller for packed trajectories.

This module is the eventless counterpart of ramsete_compact_events.py.
It delegates directly to ramsete_compact_events.follow_trajectory with events=None.

Packed layout, little-endian:
    header: <HH  version, count
    sample: <6f  t, x, y, heading, v, omega
"""

from micropython import const
from ramsete import KinematicLimiter, Odometry, RamseteController
import ramsete_compact_events
from ramsete_compact_events import (
    _VERSION,
    _HEADER_SIZE,
    _RECORD_SIZE,
    _HEADER_FMT,
    _RECORD_FMT,
    _TIME_FMT,
    _check_blob,
)

_LOOP_MS = const(20)


async def follow_trajectory(
    drivebase,
    samples,
    config=None,
    b=2.0,
    zeta=0.7,
    debug=False,
):
    """Follow a packed trajectory without events (delegates to ramsete_compact_events)."""
    return await ramsete_compact_events.follow_trajectory(
        drivebase,
        samples,
        config=config,
        b=b,
        zeta=zeta,
        events=None,
        event_map=None,
        debug=debug,
    )
