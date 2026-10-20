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


# --- CONFIGURATION ---------------------------------------------------------

CONTROL_MS = 20          # ramsete_fast.py heartbeat
SAMPLE_MS = 10           # raw plant logging; DriveBase commands stay at 20 ms
DEG_TO_RAD = 0.017453292519943295

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


def _print_metadata(name, notes=""):
    global _RUN_ID
    _RUN_ID += 1
    print("TEST_START", name)
    print("META,run_id", _RUN_ID)
    print("META,test", name)
    print("META,notes", notes.replace(",", ";"))
    print("META,wheel_diameter_mm", WHEEL_DIAMETER_MM)
    print("META,axle_track_mm", AXLE_TRACK_MM)
    print("META,battery_mv", hub.battery.voltage())
    print("META,mem_free", gc.mem_free())
    print("META,settings", _fmt_tuple(drivebase.settings()))
    print("META,distance_limits",
          _read_optional(drivebase, "distance_control", "limits"))
    print("META,distance_pid",
          _read_optional(drivebase, "distance_control", "pid"))
    print("META,heading_limits",
          _read_optional(drivebase, "heading_control", "limits"))
    print("META,heading_pid",
          _read_optional(drivebase, "heading_control", "pid"))
    print("META,left_limits",
          _read_optional(motorLeft, "control", "limits"))
    print("META,left_pid",
          _read_optional(motorLeft, "control", "pid"))
    print("META,right_limits",
          _read_optional(motorRight, "control", "limits"))
    print("META,right_pid",
          _read_optional(motorRight, "control", "pid"))
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
    hub.light.on(Color.YELLOW)
    print("")
    print("=" * 50)
    print("READY FOR:", test_name)
    print("SETUP:", instructions)
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

def run_rate_test(name, duration_ms, command, instructions, notes=""):
    """Log raw encoders/IMU at ~100 Hz while issuing drive() every 20 ms."""
    _wait_for_start(name, instructions)
    _print_metadata(name, notes)

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
        return (
            f"{values[i]:.2f},{values[i+1]:.2f},"
            f"{values[i+2]:.2f},{values[i+3]:.2f},"
            f"{values[i+4]:.1f},{values[i+5]:.1f},"
            f"{values[i+6]:.5f},{values[i+7]:.3f},{values[i+8]:.3f},"
            f"{values[i+9]:.0f},{values[i+10]:.0f},"
            f"{values[i+11]:.1f},{values[i+12]:.1f}"
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
        f"{values[i]:.2f},{values[i+1]:.2f},"
        f"{values[i+2]:.1f},{values[i+3]:.1f},"
        f"{values[i+4]:.3f},{values[i+5]:.3f},{values[i+6]:.3f},"
        f"{values[i+7]:.1f},{values[i+8]:.1f},{values[i+9]:.1f},"
        f"{values[i+10]:.0f},{values[i+11]:.0f}"
    ))


# --- TEST PROFILES ---------------------------------------------------------

def test_straight_steps(direction):
    sign = 1 if direction > 0 else -1
    segments = []
    for value in (50, 100, 200, 400, 600):
        segments.append((900, sign * value, 0))
        segments.append((600, 0, 0))
    name = "straight_steps_fwd" if sign > 0 else "straight_steps_rev"
    run_rate_test(
        name,
        sum(s[0] for s in segments),
        _sequence(segments),
        "2+ m clear runway. Robot starts at rest facing along the runway.",
        "forward steps" if sign > 0 else "reverse steps",
    )


def test_turn_steps(direction):
    sign = 1 if direction > 0 else -1
    segments = []
    for value in (30, 60, 120, 240):
        segments.append((900, 0, sign * value))
        segments.append((600, 0, 0))
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
        "2 m clear runway. Robot oscillates forward/back.",
        "backlash and command deadband",
    )


def test_chirp_straight():
    duration = 10000
    run_rate_test(
        "chirp_straight",
        duration,
        _chirp(250, duration),
        "2 m clear runway. Robot sweeps forward/back around the start.",
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
    _print_metadata(name, notes)

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
            f"{values[i]:.2f},{values[i+1]:.2f},{values[i+2]:.2f},"
            f"{values[i+3]:.0f},{values[i+4]:.0f},"
            f"{values[i+5]:.2f},{values[i+6]:.2f},"
            f"{values[i+7]:.0f}"
        )

    print(ODOM_HEADER)
    _dump(buf, ODOM_COLS, formatter)


def test_odometry_straight():
    run_odometry_test(
        "odometry_straight",
        10000,
        100,
        0,
        "2 m clear runway. Robot drives straight at 100 mm/s for 10 s.",
        "measure final distance with a tape; expected about 1000 mm",
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
        "2 m runway. Mark the start pose and a measuring spot at each stop.",
    )
    drivebase.reset(0, 0)
    _print_metadata("multi_stop", "measure pose at 500/1000/1500 mm")

    buf = bytearray()
    watch = StopWatch()
    watch.reset()
    start = watch.time()
    print("t_ms,target_mm,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv")

    for target_mm in (500, 1000, 1500):
        while drivebase.distance() < target_mm:
            drivebase.drive(100, 0)
            wait(CONTROL_MS)
        drivebase.drive(0, 0)
        wait(500)
        state = drivebase.state()
        t = watch.time() - start
        buf.extend(ustruct.pack(
            "<7f",
            t, target_mm, state[0], state[1], state[2], state[3],
            hub.battery.voltage(),
        ))
        print("STOP_AT", target_mm, state)
        _wait_for_center("Measure the robot pose now, then press CENTER.")

    drivebase.stop()

    print("t_ms,target_mm,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv")
    _dump(buf, 7, lambda values, i: (
        f"{values[i]:.1f},{values[i+1]:.0f},{values[i+2]:.0f},"
        f"{values[i+3]:.0f},{values[i+4]:.2f},{values[i+5]:.2f},"
        f"{values[i+6]:.0f}"
    ))


def test_brake_from_speed(speed_mm_s):
    _wait_for_start(
        "BRAKE FROM " + str(speed_mm_s) + " MM/S",
        "Mark the start and final pose. 2 m runway is required.",
    )
    drivebase.reset(0, 0)
    _print_metadata(
        "brake_from_" + str(speed_mm_s),
        "measure final pose after the stop",
    )

    watch = StopWatch()
    watch.reset()
    start = watch.time()
    while watch.time() - start < 2000:
        drivebase.drive(speed_mm_s, 0)
        wait(CONTROL_MS)
    stop_t = watch.time() - start
    drivebase.drive(0, 0)
    wait(1000)
    state = drivebase.state()

    print("t_ms,stop_t_ms,v_cmd_mm_s,db_d_mm,db_v_mm_s,db_a_deg,db_w_deg_s,batt_mv")
    print(
        f"{watch.time() - start:.1f},{stop_t:.1f},{speed_mm_s:.0f},"
        f"{state[0]:.0f},{state[1]:.0f},{state[2]:.2f},{state[3]:.2f},"
        f"{hub.battery.voltage():.0f}"
    )
    print("FINAL_STATE", state)
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
            f"{values[i]:.2f},{values[i+1]:.2f},{values[i+2]:.2f},"
            f"{values[i+3]:.0f},{values[i+4]:.0f},"
            f"{values[i+5]:.2f},{values[i+6]:.2f},"
            f"{values[i+7]:.0f}"
        )

    print(ODOM_HEADER)
    _dump(logged.buf, ODOM_COLS, formatter)


# --- MAIN ------------------------------------------------------------------

async def main():
    hub.system.set_stop_button((Button.BLUETOOTH,))
    hub.light.on(Color.BLUE)
    print("Digital twin SysId collector")
    print("Save the complete console output as one .csv file.")

    test_static()
    test_straight_steps(+1)
    test_straight_steps(-1)
    test_turn_steps(+1)
    test_turn_steps(-1)
    test_curvature(+1)
    test_curvature(-1)
    test_reversal()
    test_chirp_straight()
    test_chirp_turn()
    test_odometry_straight()
    test_odometry_turn()
    test_multi_stop()
    test_brake_from_speed(300)
    test_brake_from_speed(600)
    test_wall_stall()
    await test_replay("complex_mission")

    hub.light.on(Color.GREEN)
    print("SysId collection complete.")


if __name__ == "__main__" or __name__ == "sysid_digital_twin":
    run_task(main())
