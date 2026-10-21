"""
RAMSETE COMPACT EVENT CONTROLLER (Pybricks MicroPython)
-------------------------------------------------------
Event-capable counterpart of ramsete_compact.py.

It reads the same packed bytes blob and adds the reference controller's
Odometry, RamseteController, and KinematicLimiter. Use this module for
generated event missions; use ramsete_compact.py for plain pathing.

Packed layout, little-endian:
    header: <HH  version, count
    sample: <6f  t, x, y, heading, v, omega

Events are exported separately:
    events = ((sample_index, "name"), ...)

Only ustruct, umath, gc, micropython.const, pybricks.tools, and the three
classes imported from ramsete.py are used. The array module is disabled on
Pybricks, so do not add it here.
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

# The control period, in milliseconds (50 Hz).
_LOOP_MS = const(20)
_GC_EVERY = const(50)      # Target collection interval (~1 second)
_MIN_GC_SLACK_MS = const(8) # Minimum required idle margin to run GC safely


def _check_blob(samples):
    """Validate the packed header and return the sample count."""
    if not isinstance(samples, (bytes, bytearray, memoryview)):
        raise ValueError(
            "ramsete_compact_events needs packed trajectory bytes; "
            "use ramsete.py for list-of-dict missions"
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


async def follow_trajectory(
    drivebase,
    samples,
    config=None,
    b=2.0,
    zeta=0.7,
    events=None,
    event_map=None,
    debug=False,
):
    """Follow a packed trajectory and fire the exported events.

    samples: bytes/bytearray blob from --export-format python-compact.
    config: robot configuration for KinematicLimiter (same shape as ramsete.py).
    events: ((sample_index, name), ...) from the generated module.
    event_map: dict mapping event name to a callable, as in ramsete.py.

    Returns the final odometry pose as (x, y, heading) in metres and radians.
    """
    num_pts = _check_blob(samples)

    # Local variable caching to avoid dict lookups in the inner loop
    unpack_from = ustruct.unpack_from
    rec_fmt = _RECORD_FMT
    rec_size = _RECORD_SIZE
    hdr_size = _HEADER_SIZE
    loop_ms = _LOOP_MS
    gc_every = _GC_EVERY
    min_gc_slack = _MIN_GC_SLACK_MS
    pi_val = _PI
    two_pi_val = _TWO_PI
    i1000 = _I1000
    deg = math.degrees
    wait_coro = wait
    collect = gc.collect

    # Unpack sample 0 directly; avoid redundant unpacking and slicing
    t1, x1, y1, h1, v1, w1 = unpack_from(rec_fmt, samples, hdr_size)

    # Unpack sample 1
    p2 = hdr_size + rec_size
    t2, x2, y2, h2, v2, w2 = unpack_from(rec_fmt, samples, p2)

    total_time = unpack_from(
        _TIME_FMT, samples, hdr_size + (num_pts - 1) * rec_size
    )[0]

    odom = Odometry(drivebase, x1, y1, h1)
    controller = RamseteController(b, zeta)
    limiter = KinematicLimiter(config if config else {})
    n_events = len(events) if events else 0

    watch = StopWatch()
    now_time = watch.time

    # Precalculate first segment geometry and dynamics
    dt_traj = t2 - t1
    inv_dt = (1.0 / dt_traj) if dt_traj > 0.0 else 0.0
    dx = x2 - x1
    dy = y2 - y1
    dh = (h2 - h1 + pi_val) % two_pi_val - pi_val
    vr = 0.5 * (v1 + v2)
    wr = dh * inv_dt

    s_idx = 0
    ev_idx = 0
    last_t = 0.0
    current_v = 0.0
    current_w = 0.0
    next_loop = now_time()

    loop_count = 0
    last_gc_loop = 0
    total_calc = 0.0
    max_calc = 0.0
    max_dt = 0.0

    # Ensure clean heap before the control loop starts
    collect()
    # Disable automatic GC threshold if supported (prevent mid-step GC latency)
    if hasattr(gc, "threshold"):
        gc.threshold(-1)

    print("Mission Start: compact event loop, %dms period..." % loop_ms)

    try:
        while True:
            t = now_time() * i1000
            if t >= total_time:
                break

            calc_start = now_time()

            # Advance waypoint segment only when t >= t2.
            # While t < t2, zero unpacking calls are made.
            while s_idx < num_pts - 2 and t >= t2:
                s_idx += 1
                t1, x1, y1, h1, v1, w1 = t2, x2, y2, h2, v2, w2
                p2 += rec_size
                t2, x2, y2, h2, v2, w2 = unpack_from(rec_fmt, samples, p2)

                dt_traj = t2 - t1
                inv_dt = (1.0 / dt_traj) if dt_traj > 0.0 else 0.0
                dx = x2 - x1
                dy = y2 - y1
                dh = (h2 - h1 + pi_val) % two_pi_val - pi_val
                vr = 0.5 * (v1 + v2)
                wr = dh * inv_dt

            # Fire events attached to samples reached so far
            while ev_idx < n_events and events[ev_idx][0] <= s_idx:
                ev = events[ev_idx]
                ev_idx += 1
                event_name = ev[1]
                if debug:
                    print("[%.2fs] Event triggered: %s" % (t, event_name))
                if event_map:
                    fn = event_map.get(event_name)
                    if fn:
                        fn()

            # Interpolate state along current segment
            alpha = (t - t1) * inv_dt if dt_traj > 0.0 else 0.0
            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0

            xr = x1 + alpha * dx
            yr = y1 + alpha * dy
            tr = (h1 + alpha * dh + pi_val) % two_pi_val - pi_val

            # Tick-level interval average with T/2 = 10 ms lead cancels ZOH hold lag
            t_mid = t + 0.010
            is_final_stop = (s_idx >= num_pts - 2) and (abs(v2) < 1e-4)
            if is_final_stop and t_mid >= t2:
                vr = 0.0
                wr = 0.0
            elif dt_traj > 0.0:
                alpha_mid = (t_mid - t1) * inv_dt
                if alpha_mid < 0.0:
                    alpha_mid = 0.0
                elif alpha_mid > 1.0:
                    alpha_mid = 1.0
                vr = v1 + alpha_mid * (v2 - v1)
                wr = dh * inv_dt
            else:
                vr = 0.0
                wr = 0.0

            odom.update()
            v_cmd, w_cmd = controller.calculate(
                odom.x, odom.y, odom.heading,
                xr, yr, tr, vr, wr,
            )

            dt = t - last_t
            if dt > 0.0:
                v_cmd, w_cmd = limiter.limit(
                    v_cmd, w_cmd, current_v, current_w, dt
                )
                if dt > max_dt:
                    max_dt = dt

            last_t = t
            current_v = v_cmd
            current_w = w_cmd

            drivebase.drive(v_cmd * 1000.0, deg(-w_cmd))

            dur = now_time() - calc_start
            total_calc += dur
            if dur > max_calc:
                max_calc = dur
            loop_count += 1

            if debug:
                print(
                    "LOG,%.3f,%.4f,%.4f,%.4f,%.4f"
                    % (t, odom.x, odom.y, xr, yr)
                )

            # Fixed-rate heartbeat synchronization
            next_loop += loop_ms
            slack = next_loop - now_time()

            # Deterministic GC: only collect when due AND plenty of slack remains.
            # If slack is tight, defer to the next tick to prevent overrunning.
            if (loop_count - last_gc_loop) >= gc_every and slack > min_gc_slack:
                collect()
                last_gc_loop = loop_count
                slack = next_loop - now_time()

            if slack > 0:
                await wait_coro(slack)
            else:
                next_loop = now_time()
                await wait_coro(1)

    finally:
        drivebase.stop()
        if hasattr(gc, "threshold"):
            gc.threshold(-1)

    # Fire any events attached to the final sample (KI-012)
    while ev_idx < n_events:
        ev = events[ev_idx]
        ev_idx += 1
        event_name = ev[1]
        if debug:
            print("Event triggered: %s" % event_name)
        if event_map:
            fn = event_map.get(event_name)
            if fn:
                fn()

    print("Trajectory complete.")
    if loop_count > 0:
        print("--- Compact Event Metrics ---")
        print("Samples:", num_pts, "Packed bytes:", len(samples))
        print("Total Loops:", loop_count)
        print("Avg Loop Time (ms):", (total_time * 1000.0) / loop_count)
        print("Max Loop Time (ms):", max_dt * 1000.0)
        print("Avg Math Calc Time (ms):", total_calc / loop_count)
        print("Max Math Calc Time (ms):", max_calc)
        print("-----------------------------")

    odom.update()
    return odom.x, odom.y, odom.heading
