"""
DIGITAL-TWIN SYSTEM IDENTIFICATION DATA COLLECTOR
--------------------------------------------------
Robot-side data collection for the JAX/Immrax digital twin.

This script is Pybricks-only. Run it on the Hub, not with host Python.

It deliberately logs outside the controller so the deployed controller keeps
its memory/timing profile. For replay runs, it wraps DriveBase in a proxy that
records state() and drive() calls while forwarding them unchanged. Replay uses
ramsete_fast.py for list-format modules and ramsete_compact.py for packed
modules.

The PC-side solver consumes the CSV blocks printed between TEST_START and
TEST_END. Keep one console capture per session and save it as a .csv file.
"""

import gc
import ustruct
import umath as math

from pybricks.parameters import Button, Color
from pybricks.tools import StopWatch, wait, run_task

from robot_config import (
    hub,
    motorLeft,
    motorRight,
    drivebase,
    WHEEL_DIAMETER_MM,
    AXLE_TRACK_MM,
)
from odometry import Odometry


# The collector grew test by test and the hub's heap did not. Reclaim the
# import-time garbage before anything waits on a button, or the first poll
# dies with MemoryError while the heap is still full of import fragments.
gc.collect()

# The studio can run a subset of the suite: the collect request carries the
# test ids, and the server writes them here before pushing (pybricksdev
# bundles this file with the collector). No file means the full suite.
# The id is the META test name the session carries. A test that needs the
# robot left somewhere runs its dependency first even when the dependency
# was not picked (wall_stall needs wall_homing's robot-against-the-wall).
try:
    from sysid_selection import SELECTED, AUTO_START
except ImportError:
    SELECTED = None
    AUTO_START = False

# Bench mode (sysid_selection BENCH): the same suite in short runs for a
# small bench (~0.6 x 0.8 m). Speeds are unchanged, so the fitted dynamics
# are the same; only the distances shrink. Read separately so a selection
# file written before the bench mode still loads (it just runs full-size).
try:
    from sysid_selection import BENCH
except ImportError:
    BENCH = False


def _selected(test_id, extra=()):
    """Whether one test runs: everything without a selection file."""
    if SELECTED is None:
        return True
    return test_id in SELECTED or test_id in extra


# Tests that chain without the operator touching anything: the robot stays
# where it is (rate steps, spins, chirps) or is already where the next test
# needs it (the stall, after its homing). Everything else waits for the
# button, because the operator has to place, lift or still the robot first.
AUTO_ADVANCE = {
    "straight_steps_fwd", "straight_steps_rev",
    "turn_steps_cw", "turn_steps_ccw",
    "curvature_cw", "curvature_ccw",
    "reversal_backlash",
    "chirp_straight", "chirp_turn",
    "wall_stall",
}

# Seconds between two chained tests. A press skips the wait.
AUTO_DELAY_S = 5

# The first wait of a push is the operator getting ready, so it always
# blocks - unless the page triggered this one test (AUTO_START), which is
# the trigger already.
_FIRST_WAIT = True


# --- CONFIGURATION ---------------------------------------------------------

CONTROL_MS = 20          # ramsete_fast.py heartbeat
SAMPLE_MS = 10           # raw plant logging; DriveBase commands stay at 20 ms
DEG_TO_RAD = 0.017453292519943295

# How many times each calibration-critical test is repeated. One run cannot
# say how much the robot varies from day to day, and the spread of the steps
# inside one run is not that: the steps are sequential, so a sagging battery
# or a warming motor maps straight onto the step order. The repeats give
# the between-run spread, which is the honest uncertainty of the fit.
REPEATS = 3

# The repeat number of the test that is running. It goes into the META line
# so the host can group the repeats of one test.
_REPEAT = 0

# Bench geometry: short runs for a small bench. Speeds are unchanged, only
# durations and distances shrink, so the fitted dynamics are the same. The
# curvature arcs are skipped in bench mode on top of this: the host fit
# never reads them.
STEP_MS = 500 if BENCH else 900
STEP_SETTLE_MS = 300 if BENCH else 600
ODOMETRY_STRAIGHT_MS = 5000 if BENCH else 10000
MULTI_STOP_TARGETS_MM = (200, 350, 500) if BENCH else (500, 1000, 1500)
BRAKE_CRUISE_MS = 1000 if BENCH else 2000

RATE_HEADER = (
    "t_ms,loop_ms,v_cmd_mm_s,w_cmd_deg_s,l_deg,r_deg,"
    "gyro_z_rad_s,ax_m_s2,ay_m_s2,batt_mv,batt_ma,l_load_mNm,r_load_mNm"
)
RATE_COLS = 13

ODOM_HEADER = (
    "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,"
    "db_a_deg,db_w_deg_s,batt_mv"
)
ODOM_COLS = 8

# The wall homing run: the odometry columns plus the motor load, so the same
# run gives the distance to the wall and the stall torque.
HOMING_HEADER = (
    "t_ms,v_cmd_mm_s,w_cmd_deg_s,db_d_mm,db_v_mm_s,"
    "db_a_deg,db_w_deg_s,batt_mv,l_load_mNm,r_load_mNm"
)
HOMING_COLS = 10

_RUN_ID = 0


# --- SMALL HELPERS ---------------------------------------------------------

def _fmt_tuple(value):
    try:
        return ";".join(str(v) for v in value)
    except TypeError:
        return str(value)


def _read_optional(obj, name, method):
    """Read an advanced control setting if this firmware exposes it."""
    try:
        target = getattr(obj, name)
        return _fmt_tuple(getattr(target, method)())
    except Exception:
        return "unavailable"


def _print_metadata(name, notes="", poses=0, extra=None):
    global _RUN_ID
    _RUN_ID += 1
    # The META lines are "META,key,value": the host parses them that way, so
    # the run id of a test can be matched to its measured poses. "poses" is
    # how many end poses the operator has to measure for this test.
    print("TEST_START", name)
    print("META,run_id,%d" % _RUN_ID)
    print("META,test,%s" % name)
    print("META,repeat,%d" % _REPEAT)
    print("META,notes,%s" % notes.replace(",", ";"))
    print("META,poses,%d" % poses)
    print("META,wheel_diameter_mm,%d" % WHEEL_DIAMETER_MM)
    print("META,axle_track_mm,%d" % AXLE_TRACK_MM)
    if extra:
        for key in sorted(extra):
            print("META,%s,%s" % (key, extra[key]))
    print("META,battery_mv,%d" % hub.battery.voltage())
    print("META,mem_free,%d" % gc.mem_free())
    print("META,settings,%s" % _fmt_tuple(drivebase.settings()))
    print("META,distance_limits,%s" % _read_optional(drivebase, "distance_control", "limits"))
    print("META,distance_pid,%s" % _read_optional(drivebase, "distance_control", "pid"))
    print("META,heading_limits,%s" % _read_optional(drivebase, "heading_control", "limits"))
    print("META,heading_pid,%s" % _read_optional(drivebase, "heading_control", "pid"))
    print("META,left_limits,%s" % _read_optional(motorLeft, "control", "limits"))
    print("META,left_pid,%s" % _read_optional(motorLeft, "control", "pid"))
    print("META,right_limits,%s" % _read_optional(motorRight, "control", "limits"))
    print("META,right_pid,%s" % _read_optional(motorRight, "control", "pid"))
    print("DATA")


def _dump(buf, cols, formatter):
    # The replay logger stores rows as packed <cols>f records. The array
    # module is disabled on Pybricks, so a bytearray + ustruct is used.
    if isinstance(buf, bytearray):
        record_size = cols * 4
        rows = len(buf) // record_size
        fmt = "<%df" % cols
        for i in range(rows):
            values = ustruct.unpack_from(fmt, buf, i * record_size)
            print(formatter(values, 0))
        print("TEST_END", rows)
        return

    for i in range(0, len(buf), cols):
        print(formatter(buf, i))
    print("TEST_END", len(buf) // cols)


def _wait_for_start(test_name, instructions):
    """Hold for the operator, the button, or the countdown.

    The first wait of a push always blocks: the operator is still getting
    ready - unless the page triggered this one test (AUTO_START), which is
    the trigger already. Later, a chained test counts down and starts by
    itself; a press skips the wait. A setup test always blocks.
    """
    global _FIRST_WAIT
    first = _FIRST_WAIT
    _FIRST_WAIT = False
    # One collection per test: each test frees a big log buffer, and the
    # next wait loop is the first thing that allocates after it.
    gc.collect()
    hub.light.on(Color.YELLOW)
    print("")
    print("=" * 50)
    print("READY FOR:", test_name)
    print("SETUP:", instructions)
    if first and AUTO_START:
        print("Started from the page.")
        print("=" * 50)
    elif test_name in AUTO_ADVANCE and not first:
        print("Starting %s in %d s - press CENTER to start now."
              % (test_name, AUTO_DELAY_S))
        for _ in range(AUTO_DELAY_S * 10):
            if Button.CENTER in hub.buttons.pressed():
                break
            wait(100)
        print("=" * 50)
    else:
        print("Press CENTER to start. BLUETOOTH is emergency stop.")
        print("=" * 50)
        while Button.CENTER not in hub.buttons.pressed():
            wait(10)
    hub.display.text("OK")
    while Button.CENTER in hub.buttons.pressed():
        wait(10)
    hub.display.off()
    hub.light.on(Color.RED)
    wait(500)


def _wait_for_center(message):
    print(message)
    hub.light.on(Color.YELLOW)
    while Button.CENTER not in hub.buttons.pressed():
        wait(10)
    while Button.CENTER in hub.buttons.pressed():
        wait(10)
    hub.light.on(Color.RED)


def _pose_line(odom):
    """Report the measured pose in the format the host parses.

    The pose is the hub's own odometry, integrated to 2D. Its heading is
    counter-clockwise positive, the same convention as the planned
    trajectory, so the host can compare the two directly. It is the robot's
    estimate, not a tape measurement: the distance and heading scale of the
    fit need a real measurement to be calibrated.
    """
    print("POSE,%.4f,%.4f,%.2f" % (odom.x, odom.y, math.degrees(odom.heading)))

def _sequence(segments):
    """Return a command function for [(duration_ms, v_mm_s, w_deg_s), ...]."""
    def command(t_ms):
        elapsed = 0
        for duration, v_cmd, w_cmd in segments:
            if t_ms < elapsed + duration:
                return v_cmd, w_cmd
            elapsed += duration
        return 0.0, 0.0
    return command


def _chirp(amplitude, duration_ms, f0=0.2, f1=3.0):
    """Linear frequency sweep with a continuous phase."""
    duration_s = duration_ms / 1000.0
    rate = (f1 - f0) / duration_s

    def command(t_ms):
        t = t_ms / 1000.0
        phase = 2.0 * math.pi * (f0 * t + 0.5 * rate * t * t)
        return amplitude * math.sin(phase), 0.0
    return command


def _turn_chirp(amplitude, duration_ms, f0=0.2, f1=3.0):
    duration_s = duration_ms / 1000.0
    rate = (f1 - f0) / duration_s

    def command(t_ms):
        t = t_ms / 1000.0
        phase = 2.0 * math.pi * (f0 * t + 0.5 * rate * t * t)
        return 0.0, amplitude * math.sin(phase)
    return command


# --- HIGH-RATE PLANT TESTS -------------------------------------------------

def run_rate_test(name, duration_ms, command, instructions, notes="",
                  poses=0, extra=None):
    """Log raw encoders/IMU at ~100 Hz while issuing drive() every 20 ms."""
    _wait_for_start(name, instructions)
    _print_metadata(name, notes, poses=poses, extra=extra)

    buf = bytearray()
    watch = StopWatch()
    watch.reset()
    start = watch.time()
    next_control = start
    next_sample = start
    last_sample = start
    v_cmd = 0.0
    w_cmd = 0.0

    while True:
        now = watch.time()
        t = now - start
        if t >= duration_ms:
            break

        if now >= next_control:
            v_cmd, w_cmd = command(t)
            drivebase.drive(v_cmd, w_cmd)
            next_control += CONTROL_MS
            if next_control <= now:
                next_control = now + CONTROL_MS

        if now >= next_sample:
            gyro = hub.imu.angular_velocity()
            accel = hub.imu.acceleration()
            buf.extend(ustruct.pack(
                "<13f",
                t,
                now - last_sample,
                v_cmd,
                w_cmd,
                motorLeft.angle(),
                motorRight.angle(),
                gyro[2] * DEG_TO_RAD,
                accel[0] / 1000.0,
                accel[1] / 1000.0,
                hub.battery.voltage(),
                hub.battery.current(),
                motorLeft.load(),
                motorRight.load(),
            ))
            last_sample = now
            next_sample += SAMPLE_MS
            if next_sample <= now:
                next_sample = now + SAMPLE_MS

        slack = next_sample - watch.time()
        if slack > 0:
            wait(slack)

    drivebase.drive(0, 0)
    wait(400)
    drivebase.stop()

    def formatter(values, i):
        # MicroPython cannot concatenate two f-strings, so this uses %.
        return (
            "%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,"
            "%.5f,%.3f,%.3f,%.0f,%.0f,%.1f,%.1f"
            % (values[i], values[i + 1], values[i + 2], values[i + 3],
               values[i + 4], values[i + 5], values[i + 6], values[i + 7],
               values[i + 8], values[i + 9], values[i + 10], values[i + 11],
               values[i + 12])
        )

    print(RATE_HEADER)
    _dump(buf, RATE_COLS, formatter)


def test_static():
    _wait_for_start("STATIC / GYRO BIAS",
                    "Robot still and level. Do not touch it.")
    _print_metadata("static", "no drive; gyro/accel bias")

    buf = bytearray()
    watch = StopWatch()
    watch.reset()
    start = watch.time()
    next_sample = start
    last_sample = start

    while watch.time() - start < 5000:
        now = watch.time()
        if now >= next_sample:
            gyro = hub.imu.angular_velocity()
            accel = hub.imu.acceleration()
            buf.extend(ustruct.pack(
                "<12f",
                now - start,
                now - last_sample,
                motorLeft.angle(),
                motorRight.angle(),
                gyro[0], gyro[1], gyro[2],
                accel[0], accel[1], accel[2],
                hub.battery.voltage(),
                hub.battery.current(),
            ))
            last_sample = now
            next_sample += SAMPLE_MS
        slack = next_sample - watch.time()
        if slack > 0:
            wait(slack)

    print("t_ms,loop_ms,l_deg,r_deg,gyro_x_deg_s,gyro_y_deg_s,gyro_z_deg_s,"
          "ax_mm_s2,ay_mm_s2,az_mm_s2,batt_mv,batt_ma")
    _dump(buf, 12, lambda values, i: (
        "%.2f,%.2f,%.1f,%.1f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.0f,%.0f"
        % (values[i], values[i + 1], values[i + 2], values[i + 3],
           values[i + 4], values[i + 5], values[i + 6], values[i + 7],
           values[i + 8], values[i + 9], values[i + 10], values[i + 11])
    ))


# --- TEST PROFILES ---------------------------------------------------------

def _step_values(values):
    """The step sizes, in the order this repeat runs them.

    The order is reversed on every other repeat, so a drift over the run
    (a sagging battery, a warming motor) cannot land on the step index and
    make the steps look like they disagree.
    """
    if _REPEAT % 2:
        return tuple(reversed(values))
    return tuple(values)


def test_straight_steps(direction):
    sign = 1 if direction > 0 else -1
    segments = []
    for value in _step_values((50, 100, 200, 400, 600)):
        segments.append((STEP_MS, sign * value, 0))
        segments.append((STEP_SETTLE_MS, 0, 0))
    name = "straight_steps_fwd" if sign > 0 else "straight_steps_rev"
    run_rate_test(
        name,
        sum(s[0] for s in segments),
        _sequence(segments),
        ("Longest bench diagonal, facing along it. About 0.7 m of travel."
         if BENCH else
         "2+ m clear runway. Robot starts at rest facing along the runway."),
        "forward steps" if sign > 0 else "reverse steps",
    )


def test_turn_steps(direction):
    sign = 1 if direction > 0 else -1
    segments = []
    for value in _step_values((30, 60, 120, 240)):
        segments.append((STEP_MS, 0, sign * value))
        segments.append((STEP_SETTLE_MS, 0, 0))
    name = "turn_steps_cw" if sign > 0 else "turn_steps_ccw"
    run_rate_test(
        name,
        sum(s[0] for s in segments),
        _sequence(segments),
        "Open space. Robot spins in place; keep the hub clear.",
        "clockwise turns" if sign > 0 else "counter-clockwise turns",
    )


def test_curvature(direction):
    sign = 1 if direction > 0 else -1
    segments = []
    for v_cmd, w_cmd in ((100, 30), (200, 60), (300, 90), (400, 120)):
        segments.append((900, v_cmd, sign * w_cmd))
        segments.append((600, 0, 0))
    name = "curvature_cw" if sign > 0 else "curvature_ccw"
    run_rate_test(
        name,
        sum(s[0] for s in segments),
        _sequence(segments),
        "Large open area. Robot follows constant-curvature arcs.",
        "combined v/omega steps",
    )


def test_reversal():
    segments = []
    for _ in range(7):
        segments.append((300, 20, 0))
        segments.append((300, -20, 0))
    for _ in range(4):
        segments.append((800, 100, 0))
        segments.append((800, -100, 0))
    run_rate_test(
        "reversal_backlash",
        sum(s[0] for s in segments),
        _sequence(segments),
        ("Centred with room to spare. Robot oscillates forward/back around "
         "the start." if BENCH else
         "2 m clear runway. Robot oscillates forward/back."),
        "backlash and command deadband",
    )


def test_chirp_straight():
    duration = 10000
    run_rate_test(
        "chirp_straight",
        duration,
        _chirp(250, duration),
        ("Start in the middle. Robot sweeps forward/back around the start."
         if BENCH else
         "2 m clear runway. Robot sweeps forward/back around the start."),
        "linear chirp 0.2-3 Hz, amplitude 250 mm/s",
    )


def test_chirp_turn():
    duration = 10000
    run_rate_test(
        "chirp_turn",
        duration,
        _turn_chirp(180, duration),
        "Open space. Robot spins in place with a sweep.",
        "linear chirp 0.2-3 Hz, amplitude 180 deg/s",
    )


# --- ODOMETRY / STOPPING TESTS --------------------------------------------

def run_odometry_test(name, duration_ms, v_cmd, w_cmd, instructions, notes=""):
    _wait_for_start(name, instructions)
    drivebase.reset(0, 0)
    odom = Odometry(drivebase)
    _print_metadata(name, notes, poses=1)

    buf = bytearray()
    watch = StopWatch()
    watch.reset()
    start = watch.time()
    next_tick = start

    while True:
        now = watch.time()
        t = now - start
        if t >= duration_ms:
            break

        if now >= next_tick:
            state = drivebase.state()
            # Integrate the pose every tick: a turn has to be integrated
            # along the way, not as one chord at the end.
            odom.update()
            drivebase.drive(v_cmd, w_cmd)
            buf.extend(ustruct.pack(
                "<8f",
                t,
                v_cmd,
                w_cmd,
                state[0],
                state[1],
                state[2],
                state[3],
                hub.battery.voltage(),
            ))
            next_tick += CONTROL_MS
            if next_tick <= now:
                next_tick = now + CONTROL_MS

        slack = next_tick - watch.time()
        if slack > 0:
            wait(slack)

    drivebase.drive(0, 0)
    wait(500)
    odom.update()
    _pose_line(odom)
    final_state = drivebase.state()
    buf.extend(ustruct.pack(
        "<8f",
        watch.time() - start,
        0.0,
        0.0,
        final_state[0],
        final_state[1],
        final_state[2],
        final_state[3],
        hub.battery.voltage(),
    ))
    drivebase.stop()

    def formatter(values, i):
        return (
            "%.2f,%.2f,%.2f,%.0f,%.0f,%.2f,%.2f,%.0f"
            % (values[i], values[i + 1], values[i + 2], values[i + 3],
               values[i + 4], values[i + 5], values[i + 6], values[i + 7])
        )

    print(ODOM_HEADER)
    _dump(buf, ODOM_COLS, formatter)


def test_odometry_straight():
    run_odometry_test(
        "odometry_straight",
        ODOMETRY_STRAIGHT_MS,
        100,
        0,
        ("Bench diagonal. Robot drives straight at 100 mm/s for 5 s."
         if BENCH else
         "2 m clear runway. Robot drives straight at 100 mm/s for 10 s."),
        ("measure final distance with a tape; expected about 500 mm"
         if BENCH else
         "measure final distance with a tape; expected about 1000 mm"),
    )


SQUARE_SIDE_MM = 150
SQUARE_SPEED_MM_S = 100
SQUARE_TURN_DEG_S = 50
# The robot pauses briefly at every corner and after every turn, so the
# servo transient of the phase change has decayed before the next phase is
# measured. Without the pause the transient curves into the leg and biases
# the geometry the squares identify.
SQUARE_SETTLE_MS = 500


def test_square(direction):
    """One UMBmark square: four slow legs, four in-place quarter turns.

    Run slowly so slip stays out of the measurement (the UMBmark
    discipline). The operator measures only the end pose; the start mark
    is the reference the fit compares it against.
    """
    name = "square_cw" if direction > 0 else "square_ccw"
    turn = SQUARE_TURN_DEG_S if direction > 0 else -SQUARE_TURN_DEG_S
    leg_ms = SQUARE_SIDE_MM * 1000 // SQUARE_SPEED_MM_S
    turn_ms = 90 * 1000 // SQUARE_TURN_DEG_S
    segments = []
    for _ in range(4):
        segments.append((leg_ms, SQUARE_SPEED_MM_S, 0))
        segments.append((SQUARE_SETTLE_MS, 0, 0))
        segments.append((turn_ms, 0, turn))
        segments.append((SQUARE_SETTLE_MS, 0, 0))
    run_rate_test(
        name,
        sum(s[0] for s in segments),
        _sequence(segments),
        "Clear 400 mm square, e.g. a launch zone. Robot drives a 150 mm "
        "square, then stops.",
        "square, run slowly; end pose is the measurement",
        poses=1,
        extra={"square_side_mm": SQUARE_SIDE_MM,
               "square_direction": "cw" if direction > 0 else "ccw"},
    )


def run_homing_test(name, v_cmd_mm_s, timeout_ms, instructions, notes=""):
    """Drive into a wall and stop as soon as the robot is blocked.

    The distance the robot covers is the distance to the wall, so the host
    can check the robot's odometry against a reference that does not move and
    does not have to be read by hand. That is what makes it possible to
    re-check the calibration unattended, over and over.

    The robot stops when the wheels stop turning while the command is still
    on, so the contact is a stall and not a bounce. A bounce lets the tyres
    slip, and the encoders then read further than the robot actually went,
    which would bias the scale.
    """
    _wait_for_start(name, instructions)
    drivebase.reset(0, 0)
    _print_metadata(name, notes)

    buf = bytearray()
    watch = StopWatch()
    watch.reset()
    start = watch.time()
    next_tick = start
    blocked = 0

    while True:
        now = watch.time()
        t = now - start
        if t >= timeout_ms:
            break

        if now >= next_tick:
            state = drivebase.state()
            # The speed is the second entry of the drivebase state.
            if state[1] < 0.2 * v_cmd_mm_s:
                blocked += 1
            else:
                blocked = 0
            drivebase.drive(v_cmd_mm_s, 0)
            buf.extend(ustruct.pack(
                "<10f",
                t,
                v_cmd_mm_s,
                0.0,
                state[0],
                state[1],
                state[2],
                state[3],
                hub.battery.voltage(),
                motorLeft.load(),
                motorRight.load(),
            ))
            next_tick += CONTROL_MS
            if next_tick <= now:
                next_tick = now + CONTROL_MS
            if blocked >= 5:
                # Blocked for a fifth of a second: it has reached the wall.
                break

        slack = next_tick - watch.time()
        if slack > 0:
            wait(slack)

    drivebase.drive(0, 0)
    wait(300)
    final_state = drivebase.state()
    buf.extend(ustruct.pack(
        "<10f",
        watch.time() - start,
        0.0,
        0.0,
        final_state[0],
        final_state[1],
        final_state[2],
        final_state[3],
        hub.battery.voltage(),
        motorLeft.load(),
        motorRight.load(),
    ))
    drivebase.stop()

    def formatter(values, i):
        return (
            "%.2f,%.2f,%.2f,%.0f,%.2f,%.2f,%.2f,%.0f,%.1f,%.1f"
            % (values[i], values[i + 1], values[i + 2], values[i + 3],
               values[i + 4], values[i + 5], values[i + 6], values[i + 7],
               values[i + 8], values[i + 9])
        )

    print(HOMING_HEADER)
    _dump(buf, 10, formatter)


def test_wall_homing():
    run_homing_test(
        "wall_homing",
        60,
        30000,
        ("Put the robot's centre on the start mark, square to the bench "
         "backstop. Do not touch the robot once it is running."
         if BENCH else
         "Put the robot's centre on the start mark, square to the wall. The "
         "wall is about a metre ahead. Do not touch the robot once it is "
         "running."),
        "drive into the wall and stop when it is blocked; the host checks "
        "the odometry against the marked distance",
    )


def test_odometry_turn():
    run_odometry_test(
        "odometry_turn_cw",
        1800,
        0,
        50,
        "Open space. Robot turns clockwise at 50 deg/s for 1.8 s.",
        "expected about 90 deg; measure final heading",
    )


def test_multi_stop():
    _wait_for_start(
        "MULTI-STOP ODOMETRY",
        ("Bench diagonal. Mark the start pose and a measuring spot at each "
         "stop." if BENCH else
         "2 m runway. Mark the start pose and a measuring spot at each stop."),
    )
    drivebase.reset(0, 0)
    odom = Odometry(drivebase)
    _print_metadata(
        "multi_stop",
        ("measure pose at 200/350/500 mm" if BENCH else
         "measure pose at 500/1000/1500 mm"),
        poses=3,
    )

    buf = bytearray()
    watch = StopWatch()
    watch.reset()
    start = watch.time()
    print("t_ms,target_mm,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv")

    for target_mm in MULTI_STOP_TARGETS_MM:
        while drivebase.distance() < target_mm:
            drivebase.drive(100, 0)
            wait(CONTROL_MS)
        drivebase.drive(0, 0)
        wait(500)
        # The run is straight, so integrating once per stop is exact.
        odom.update()
        state = drivebase.state()
        t = watch.time() - start
        buf.extend(ustruct.pack(
            "<7f",
            t, target_mm, state[0], state[1], state[2], state[3],
            hub.battery.voltage(),
        ))
        print("STOP_AT", target_mm, state)
        _pose_line(odom)
        _wait_for_center("Measure the robot pose now, then press CENTER.")

    drivebase.stop()

    print("t_ms,target_mm,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv")
    _dump(buf, 7, lambda values, i: (
        "%.1f,%.0f,%.0f,%.0f,%.2f,%.2f,%.0f"
        % (values[i], values[i + 1], values[i + 2], values[i + 3],
           values[i + 4], values[i + 5], values[i + 6])
    ))


def test_brake_from_speed(speed_mm_s):
    _wait_for_start(
        "BRAKE FROM " + str(speed_mm_s) + " MM/S",
        ("Mark the start and final pose. Use the bench diagonal."
         if BENCH else
         "Mark the start and final pose. 2 m runway is required."),
    )
    drivebase.reset(0, 0)
    odom = Odometry(drivebase)
    _print_metadata(
        "brake_from_" + str(speed_mm_s),
        "measure final pose after the stop",
        poses=1,
    )

    watch = StopWatch()
    watch.reset()
    start = watch.time()
    while watch.time() - start < BRAKE_CRUISE_MS:
        drivebase.drive(speed_mm_s, 0)
        wait(CONTROL_MS)
    stop_t = watch.time() - start
    drivebase.drive(0, 0)
    wait(1000)
    odom.update()
    state = drivebase.state()

    print("t_ms,stop_t_ms,v_cmd_mm_s,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv")
    print(
        "%.1f,%.1f,%.0f,%.0f,%.0f,%.2f,%.2f,%.0f"
        % (watch.time() - start, stop_t, speed_mm_s, state[0], state[1],
           state[2], state[3], hub.battery.voltage())
    )
    print("FINAL_STATE", state)
    _pose_line(odom)
    _wait_for_center("Measure the final pose, then press CENTER.")
    drivebase.stop()
    print("TEST_END", 1)


def test_wall_stall():
    run_rate_test(
        "wall_stall",
        2500,
        _sequence(((2500, 30, 0),)),
        "Front bumper flat against a rigid wall. Use a low-traction-safe setup.",
        "drive(30, 0) into the wall; measure stall load/battery",
    )


def test_top_speed():
    """The motor's free-running speed, with the wheels off the ground.

    The torque-speed line the force model uses is fixed by its two ends:
    the stall torque, which the wall test measures, and the speed at which
    the motor makes no torque, which is this one. With the wheels lifted the
    motor pushes nothing, so it runs up to that speed and stops there.

    The robot config declares that speed; this measures it. The command is
    deliberately larger than any LEGO motor can reach, so the motor is
    torque-limited rather than speed-limited: if the wheels were still on the
    ground the measurement would be the traction limit instead.
    """
    run_rate_test(
        "top_speed",
        3000,
        _sequence(((3000, 1000, 0),)),
        "Lift the robot so both wheels spin free. They must not touch the mat.",
        "free-running speed; the motor's no-load speed",
    )


# --- REPLAY / HELD-OUT VALIDATION -----------------------------------------

class LoggedDriveBase:
    """Log state()/drive() without changing the selected controller."""

    def __init__(self, real):
        self.real = real
        self.watch = StopWatch()
        self.watch.reset()
        self.start = self.watch.time()
        self.last_state = (0, 0, 0, 0)
        self.last_battery = hub.battery.voltage()
        self.tick = 0
        self.buf = bytearray()

    def state(self):
        state = self.real.state()
        self.last_state = state
        return state

    def drive(self, v_cmd, w_cmd):
        if self.tick % 10 == 0:
            self.last_battery = hub.battery.voltage()
        t = self.watch.time() - self.start
        state = self.last_state
        self.buf.extend(ustruct.pack(
            "<8f",
            t,
            v_cmd,
            w_cmd,
            state[0],
            state[1],
            state[2],
            state[3],
            self.last_battery,
        ))
        self.tick += 1
        self.real.drive(v_cmd, w_cmd)

    def finish(self):
        state = self.real.state()
        t = self.watch.time() - self.start
        self.buf.extend(ustruct.pack(
            "<8f",
            t,
            0,
            0,
            state[0],
            state[1],
            state[2],
            state[3],
            hub.battery.voltage(),
        ))


async def test_replay(module_name):
    module = __import__(module_name)
    samples = module.samples
    config = module.config

    if isinstance(samples, (bytes, bytearray)):
        from ramsete_compact import follow_trajectory
    else:
        from ramsete_fast import follow_trajectory

    _wait_for_start(
        "REPLAY " + module_name,
        "Place robot at the trajectory start pose. Clear the full path.",
    )
    drivebase.reset(0, 0)
    _print_metadata("replay_" + module_name,
                    "held-out run; do not fit this log")

    gc.collect()
    logged = LoggedDriveBase(drivebase)
    await follow_trajectory(logged, samples, config=config)
    logged.finish()
    drivebase.stop()

    def formatter(values, i):
        return (
            "%.2f,%.2f,%.2f,%.0f,%.0f,%.2f,%.2f,%.0f"
            % (values[i], values[i + 1], values[i + 2], values[i + 3],
               values[i + 4], values[i + 5], values[i + 6], values[i + 7])
        )

    print(ODOM_HEADER)
    _dump(logged.buf, ODOM_COLS, formatter)


# --- MAIN ------------------------------------------------------------------

def run_suite():
    """The whole collection protocol, run synchronously at top level.

    Blocking wait() calls inside a run_task coroutine leak a scheduler
    allocation per call (~36 bytes each, measured), which bleeds the heap
    dry during a long button wait - the MemoryError before the first test.
    Only the replay step is async, and it runs its own task and returns.
    """
    global _REPEAT

    hub.system.set_stop_button((Button.BLUETOOTH,))
    hub.light.on(Color.BLUE)
    print("Digital twin SysId collector")
    print("Save the complete console output as one .csv file.")
    print("Calibration tests run %d times each." % REPEATS)
    if SELECTED is None:
        print("Running the full suite.")
    else:
        print("Running %d selected tests: %s." % (len(SELECTED),
                                                  ", ".join(SELECTED)))
    if BENCH:
        print("Bench mode: short runs for a small bench; "
              "curvature is skipped.")

    # The tests that feed the fitted numbers are repeated, so the fit has a
    # between-run spread to report. The rest run once: they are diagnostics
    # or a held-out check, and one run is enough for those.
    if _selected("static"):
        test_static()
    for _REPEAT in range(REPEATS):
        if _selected("straight_steps_fwd"):
            test_straight_steps(+1)
        if _selected("straight_steps_rev"):
            test_straight_steps(-1)
    for _REPEAT in range(REPEATS):
        if _selected("turn_steps_cw"):
            test_turn_steps(+1)
        if _selected("turn_steps_ccw"):
            test_turn_steps(-1)

    # These run once, so they are all repeat zero.
    _REPEAT = 0
    # Bench mode skips the curvature arcs: the host fit never reads them.
    # The _selected() calls stay, so the studio's test list still matches
    # the collector's suite.
    if _selected("curvature_cw") and not BENCH:
        test_curvature(+1)
    if _selected("curvature_ccw") and not BENCH:
        test_curvature(-1)
    if _selected("reversal_backlash"):
        test_reversal()
    if _selected("chirp_straight"):
        test_chirp_straight()
    if _selected("chirp_turn"):
        test_chirp_turn()

    # The tests the operator has to measure, one repeat after another so the
    # measurements are taken in one go. The wall test is in here too: it
    # measures the motor's stall torque, and the operator has to move the
    # robot to the wall anyway, so three pushes cost little.
    for _REPEAT in range(REPEATS):
        if _selected("odometry_straight"):
            test_odometry_straight()
        if _selected("odometry_turn_cw"):
            test_odometry_turn()
        if _selected("multi_stop"):
            test_multi_stop()
        if _selected("brake_from_300"):
            test_brake_from_speed(300)
        if _selected("brake_from_600"):
            test_brake_from_speed(600)
        if _selected("square_cw"):
            test_square(+1)
        if _selected("square_ccw"):
            test_square(-1)

    # The wall tests, from the start mark to the wall and then into it. The
    # homing run leaves the robot against the wall, which is where the stall
    # test wants it, so the operator only sets the robot up once. Picking the
    # stall without the homing still homes first (the extra argument).
    for _REPEAT in range(REPEATS):
        if _selected("wall_homing", ("wall_stall",)):
            test_wall_homing()
        if _selected("wall_stall"):
            test_wall_stall()

    # The robot has to be lifted off the ground for this one, so it runs on
    # its own after the measuring tests are done.
    for _REPEAT in range(REPEATS):
        if _selected("top_speed"):
            test_top_speed()

    _REPEAT = 0
    if _selected("replay_complex_mission"):
        run_task(test_replay("complex_mission"))

    hub.light.on(Color.GREEN)
    print("SysId collection complete.")


# The hub menu runs the suite by importing this module, and hub_deploy runs
# it as __main__. gc.mem_free exists on the hub and never on a host
# interpreter, so a stubbed host import defines everything and starts
# nothing (the tests replace this module's gc after import, so the check
# runs unpatched).
if __name__ == "__main__" or __name__ == "sysid_digital_twin":
    if hasattr(gc, "mem_free"):
        run_suite()
