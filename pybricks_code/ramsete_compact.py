"""
RAMSETE COMPACT CONTROLLER (Pybricks MicroPython)
-------------------------------------------------
RAM-lean pathing controller for packed trajectories.

This is the compact counterpart of ramsete_fast.py. It expects the bytes blob
written by ``main.py --export-format python-compact`` and never expands it into
a list of dicts, so the trajectory costs 24 bytes per sample instead of several
hundred.

Packed layout, little-endian:
    header: <HH  version, count
    sample: <6f  t, x, y, heading, v, omega

Only ustruct, umath, gc, micropython.const, and pybricks.tools are used. The
array module is disabled on Pybricks, so do not add it here.

Legacy list-of-dict missions should keep using ramsete_fast.py.
"""

import gc
import umath as math
import ustruct
from micropython import const
from pybricks.tools import StopWatch, wait

from ramsete import KinematicLimiter, Odometry, RamseteController

_VERSION = const(1)
_HEADER_SIZE = const(4)
_RECORD_SIZE = const(24)
_HEADER_FMT = "<HH"
_RECORD_FMT = "<6f"
_TIME_FMT = "<f"

_PI = const(3141593) / 1000000.0
_TWO_PI = const(6283185) / 1000000.0
_I1000 = const(1) / 1000.0

# The control period, in milliseconds.
#
# It is the same as the event-capable twin's (ramsete_compact_events.py) and
# the same as ramsete.py's: the drive command is held until the next
# iteration, and the hold is what the wheel-force error is proportional to.
# The validator models it (immrax_validator.py, CONTROL_DT), so the two
# firmwares and the model have to agree on it.
#
# 20 ms is the hub's budget: the loop does the odometry update, the
# controller and the limiter every iteration, and a 50 Hz heartbeat leaves
# the hub room to run everything else. It is also a *deterministic* period,
# so the model can mirror it exactly; a free-running loop would make the
# model's hold a lower bound on the robot's.
_LOOP_MS = const(20)
_GC_EVERY = const(50)      # collect about once per second, not every tick


def _check_blob(samples):
    """Validate the packed header and return the sample count."""
    if not isinstance(samples, (bytes, bytearray)):
        raise ValueError(
            "ramsete_compact needs packed trajectory bytes; "
            "use ramsete_fast for list-of-dict missions"
        )
    if len(samples) < _HEADER_SIZE:
        raise ValueError("packed trajectory is too short")

    version, count = ustruct.unpack_from(_HEADER_FMT, samples, 0)
    if version != _VERSION:
        raise ValueError("unsupported packed trajectory version")
    if len(samples) != _HEADER_SIZE + count * _RECORD_SIZE:
        raise ValueError("packed trajectory length does not match its header")
    if count < 2:
        raise ValueError("packed trajectory needs at least two samples")
    return count


async def follow_trajectory(drivebase, samples, config=None, b=2.0, zeta=0.7):
    """Follow a packed trajectory with the Ramsete controller.

    samples: bytes/bytearray blob from --export-format python-compact.
    config: robot configuration for the KinematicLimiter (the same shape
        ramsete.py takes). It is optional, but without it the limiter falls
        back to its own defaults, which are not this robot's.
    """
    num_pts = _check_blob(samples)

    db_drive = drivebase.drive
    unpack_from = ustruct.unpack_from

    # Initial reference pose. The robot is assumed to start there.
    _, x0, y0, h0, _, _ = unpack_from(_RECORD_FMT, samples, _HEADER_SIZE)
    total_time = unpack_from(
        _TIME_FMT, samples, _HEADER_SIZE + (num_pts - 1) * _RECORD_SIZE
    )[0]

    # The same three pieces the event-capable twin uses, so the two follow a
    # packed trajectory identically and the validator can mirror both.
    odom = Odometry(drivebase, x0, y0, h0)
    controller = RamseteController(b, zeta)
    limiter = KinematicLimiter(config if config else {})

    # Metrics
    loop_count = 0
    total_calc = 0.0
    max_calc = 0.0
    max_dt = 0.0

    s_idx = 0
    last_t = 0.0
    current_v = 0.0
    current_w = 0.0

    watch = StopWatch()
    now_time = watch.time
    next_loop = now_time()

    gc.collect()
    print("Mission Start: compact %dms loop..." % _LOOP_MS)

    try:
        while True:
            now_ms = now_time()
            t = now_ms * _I1000
            if t >= total_time:
                break

            calc_start = now_ms

            # Find the segment that contains t. t < total_time keeps s_idx at
            # most num_pts - 2, so p2 is always a real sample and the final
            # segment is usable.
            while (
                s_idx < num_pts - 1
                and unpack_from(
                    _TIME_FMT,
                    samples,
                    _HEADER_SIZE + (s_idx + 1) * _RECORD_SIZE,
                )[0] <= t
            ):
                s_idx += 1

            p1 = _HEADER_SIZE + s_idx * _RECORD_SIZE
            p2 = p1 + _RECORD_SIZE
            t1, x1, y1, h1, v1, w1 = unpack_from(_RECORD_FMT, samples, p1)
            t2, x2, y2, h2, v2, w2 = unpack_from(_RECORD_FMT, samples, p2)

            dt_traj = t2 - t1
            if dt_traj > 0:
                alpha = (t - t1) / dt_traj
                if alpha < 0.0:
                    alpha = 0.0
                elif alpha > 1.0:
                    alpha = 1.0
            else:
                alpha = 0.0

            xr = x1 + alpha * (x2 - x1)
            yr = y1 + alpha * (y2 - y1)
            dh = (h2 - h1 + _PI) % _TWO_PI - _PI
            tr = (h1 + alpha * dh + _PI) % _TWO_PI - _PI
            vr = v1 + alpha * (v2 - v1)
            # The turn rate is the derivative of the heading, not an
            # independent line: interpolating both separately makes them
            # describe different motions (KI-027).
            wr = dh / dt_traj if dt_traj > 0 else 0.0

            # The same odometry, controller and limiter the event-capable twin
            # uses. The gain is recomputed from the interpolated speed and turn
            # rate inside the controller, so the two agree.
            odom.update()
            v_cmd, w_cmd = controller.calculate(
                odom.x, odom.y, odom.heading, xr, yr, tr, vr, wr,
            )

            dt_loop = t - last_t
            if dt_loop > 0:
                v_cmd, w_cmd = limiter.limit(
                    v_cmd, w_cmd, current_v, current_w, dt_loop
                )
                if dt_loop > max_dt:
                    max_dt = dt_loop
            last_t = t
            current_v, current_w = v_cmd, w_cmd

            # No rounding: the event-capable twin and ramsete.py pass floats
            # straight through, and the validator does the same.
            db_drive(v_cmd * 1000.0, math.degrees(-w_cmd))

            dur = now_time() - calc_start
            total_calc += dur
            if dur > max_calc:
                max_calc = dur
            loop_count += 1

            # The old controller collected every tick. A periodic collect in
            # the slack window keeps the timing profile without the cost.
            next_loop += _LOOP_MS
            if loop_count % _GC_EVERY == 0 and next_loop - now_time() > 3:
                gc.collect()

            slack = next_loop - now_time()
            if slack > 0:
                await wait(slack)
            else:
                next_loop = now_time()
                await wait(1)

    finally:
        db_drive(0, 0)

    if loop_count > 0 and total_time > 0:
        print("--- Compact Controller Metrics ---")
        print("Samples:", num_pts, "Packed bytes:", len(samples))
        print("Total Loops:", loop_count)
        print("Avg Loop Time (ms):", (total_time * 1000.0) / loop_count)
        print("Max Loop Time (ms):", max_dt * 1000.0)
        print("Avg Math Calc Time (ms):", total_calc / loop_count)
        print("Max Math Calc Time (ms):", max_calc)
        print("----------------------------------")
