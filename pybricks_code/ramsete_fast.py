"""
RAMSETE FAST CONTROLLER (Pybricks MicroPython)
----------------------------------------------
Legacy pathing controller for list-of-dict missions. Delegates to ramsete.py.
"""

from micropython import const
from ramsete import (
    KinematicLimiter,
    Odometry,
    RamseteController,
    follow_trajectory as _follow_trajectory,
)

_LOOP_MS = const(20)


async def follow_trajectory(drivebase, samples, config=None, b=2.0, zeta=0.7):
    """Follow a list-of-dict trajectory using the shared ramsete loop."""
    return await _follow_trajectory(
        drivebase, samples, config=config, b=b, zeta=zeta, event_map=None, debug=False
    )
