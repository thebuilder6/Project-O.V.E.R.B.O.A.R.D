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

_VERSION = const(1)
_HEADER_SIZE = const(4)
_RECORD_SIZE = const(24)
_HEADER_FMT = "<HH"
_RECORD_FMT = "<6f"
_TIME_FMT = "<f"

_PI = const(3141593) / 1000000.0
_TWO_PI = const(6283185) / 1000000.0
_R2D = const(5729578) / 100000.0
_D2R = const(17453) / 1000000.0
_I1000 = const(1) / 1000.0

_LOOP_MS = const(20)       # same 50 Hz heartbeat as ramsete_fast.py
_GC_EVERY = const(50)      # collect about once per second, not every tick


def _round(value):
    """Round to the nearest integer; MicroPython's int() truncates."""
    return int(value + 0.5) if value >= 0.0 else -int(-value + 0.5)


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
    config: accepted for drop-in compatibility with ramsete_fast.py; unused.
    """
    num_pts = _check_blob(samples)

    db_distance = drivebase.distance
    db_angle = drivebase.angle
    db_drive = drivebase.drive
    m_cos, m_sin, m_sqrt = math.cos, math.sin, math.sqrt
    unpack_from = ustruct.unpack_from

    # Initial reference pose. The robot is assumed to start there.
    _, x, y, h, _, _ = unpack_from(_RECORD_FMT, samples, _HEADER_SIZE)
    total_time = unpack_from(
        _TIME_FMT, samples, _HEADER_SIZE + (num_pts - 1) * _RECORD_SIZE
    )[0]

    # Initial hardware state. distance() is mm; angle() is float degrees
    # (gyro when use_gyro(True)).
    d_mm = db_distance()
    a_deg = db_angle()
    last_d = d_mm * _I1000
    last_a = -a_deg * _D2R

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
    print("Mission Start: compact 20ms loop...")

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
            wr = w1 + alpha * (w2 - w1)

            # Precompute the gain per sample, then interpolate, matching the
            # ramsete_fast.py / validator reference model for any b.
            kr1 = m_sqrt(w1 * w1 + b * v1 * v1)
            kr2 = m_sqrt(w2 * w2 + b * v2 * v2)
            kr = kr1 + alpha * (kr2 - kr1)

            # Synchronous odometry. Local on the hub; angle() keeps the
            # gyro's fractional degrees.
            d_mm = db_distance()
            a_deg = db_angle()
            d = d_mm * _I1000
            a = -a_deg * _D2R
            dd, da = d - last_d, a - last_a
            avg_h = h + da * 0.5
            x += dd * m_cos(avg_h)
            y += dd * m_sin(avg_h)
            h = (h + da + _PI) % _TWO_PI - _PI
            last_d, last_a = d, a

            # Ramsete
            dx, dy = xr - x, yr - y
            ct, st = m_cos(h), m_sin(h)
            ex = ct * dx + st * dy
            ey = -st * dx + ct * dy
            et = (tr - h + _PI) % _TWO_PI - _PI

            sinc = 1.0 if abs(et) < 1e-6 else m_sin(et) / et
            kg = 2 * zeta * kr

            current_v = vr * m_cos(et) + kg * ex
            current_w = wr + b * vr * sinc * ey + kg * et

            # Round first: Pybricks truncates float arguments toward zero.
            db_drive(_round(current_v * 1000.0), _round(-current_w * _R2D))

            dt_loop = t - last_t
            if dt_loop > max_dt:
                max_dt = dt_loop
            last_t = t

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
