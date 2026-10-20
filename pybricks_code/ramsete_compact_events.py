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

# The control period, in milliseconds.
#
# It is the same as the plain twin's (ramsete_compact.py) and the same as
# ramsete.py's: the drive command is held until the next iteration, and the
# hold is what the wheel-force error is proportional to. The validator models
# it (immrax_validator.py, CONTROL_DT), so the two firmwares and the model
# have to agree on it.
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

    x0, y0, h0 = ustruct.unpack_from(_RECORD_FMT, samples, _HEADER_SIZE)[1:4]
    total_time = ustruct.unpack_from(
        _TIME_FMT, samples, _HEADER_SIZE + (num_pts - 1) * _RECORD_SIZE
    )[0]

    odom = Odometry(drivebase, x0, y0, h0)
    controller = RamseteController(b, zeta)
    limiter = KinematicLimiter(config if config else {})
    n_events = len(events) if events else 0

    watch = StopWatch()
    now_time = watch.time
    unpack_from = ustruct.unpack_from

    s_idx = 0
    ev_idx = 0
    last_t = 0.0
    current_v = 0.0
    current_w = 0.0
    next_loop = now_time()

    loop_count = 0
    total_calc = 0.0
    max_calc = 0.0
    max_dt = 0.0

    gc.collect()
    gc.threshold(4096)
    print("Mission Start: compact event loop, %dms period..." % _LOOP_MS)

    try:
        while True:
            t = now_time() * _I1000
            if t >= total_time:
                break

            calc_start = watch.time()

            # t < total_time keeps s_idx at most num_pts - 2, so s2 is real.
            while (
                s_idx < num_pts - 1
                and unpack_from(
                    _TIME_FMT,
                    samples,
                    _HEADER_SIZE + (s_idx + 1) * _RECORD_SIZE,
                )[0] <= t
            ):
                s_idx += 1

            # Fire every event attached to a sample we have reached. The event
            # on the final sample is fired after the loop (KI-012).
            while ev_idx < n_events and events[ev_idx][0] <= s_idx:
                event_name = events[ev_idx][1]
                if debug:
                    print("[%.2fs] Event triggered: %s" % (t, event_name))
                if event_map and event_name in event_map:
                    event_map[event_name]()
                ev_idx += 1

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
            vr = 0.5 * (v1 + v2)
            # The turn rate is the derivative of the heading, not an
            # independent line: interpolating both separately makes them
            # describe different motions (KI-027).
            wr = dh / dt_traj if dt_traj > 0 else 0.0

            odom.update()
            v_cmd, w_cmd = controller.calculate(
                odom.x, odom.y, odom.heading,
                xr, yr, tr, vr, wr,
            )

            dt = t - last_t
            if dt > 0:
                v_cmd, w_cmd = limiter.limit(
                    v_cmd, w_cmd, current_v, current_w, dt
                )
                if dt > max_dt:
                    max_dt = dt

            last_t = t
            current_v, current_w = v_cmd, w_cmd

            drivebase.drive(v_cmd * 1000.0, math.degrees(-w_cmd))

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

            # The 20 ms heartbeat, synced so the period is exact: the model
            # mirrors it, so the hold it assumes has to be the hold the robot
            # really gets.
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
        drivebase.stop()
        gc.threshold(-1)

    # The loop stops before the final sample, so an event attached to the final
    # waypoint never fired (KI-012). Fire whatever is left now: the robot has
    # arrived at the last waypoint.
    while ev_idx < n_events:
        event_name = events[ev_idx][1]
        if debug:
            print("Event triggered: %s" % event_name)
        if event_map and event_name in event_map:
            event_map[event_name]()
        ev_idx += 1

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

    # The robot stopped just above, so one more integration step gives the
    # pose the host reads back.
    odom.update()
    return odom.x, odom.y, odom.heading
