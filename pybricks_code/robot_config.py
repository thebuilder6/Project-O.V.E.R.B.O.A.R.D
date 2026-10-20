"""
ROBOT CONFIGURATION
-------------------
This file defines the physical hardware of the robot. It is the central place
to configure motors, sensors, and the drive base.

Key Concepts:
- Hub Orientation: The `PrimeHub` must be told which way is UP and FORWARD using Axis 
  parameters so the gyro works correctly if mounted sideways.
- Motor Reversal: Left motors are usually mounted mirrored, so we reverse them.
- DriveBase geometry: `WHEEL_DIAMETER_MM` and `AXLE_TRACK_MM` are also logged by
  the SysId collector, so keep them here instead of duplicating literals.
- Controllers are imported by each mission: pathing uses `ramsete_fast`, events
  use `ramsete`.
"""

from pybricks.hubs import PrimeHub
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor
from pybricks.parameters import Button, Color, Direction, Port, Side, Stop, Axis
from pybricks.robotics import DriveBase
from pybricks.tools import wait, StopWatch, multitask, run_task
import umath as math

# Controllers are imported by the mission that uses them. Keeping them out of
# this module avoids loading the event-capable reference controller on memory-
# constrained pathing runs.

# ------------------------------------------------------------------
# HUB & HARDWARE INITIALIZATION
# ------------------------------------------------------------------

# These were the values that we used for our robot - you will need to change these for your robot!

# Specify Hub orientation (Top=Z, Front=Y means mounted flat on the table)
hub = PrimeHub(top_side=Axis.Z, front_side=Axis.Y)

# Drive motors (Left reversed to move forward positively)
motorLeft = Motor(Port.F, Direction.COUNTERCLOCKWISE)
motorRight = Motor(Port.B, Direction.CLOCKWISE)

# Comment out if you are not using the attachment motors or sensors!

# Attachment motors
motorFront = Motor(Port.C, Direction.CLOCKWISE)
motorBack = Motor(Port.D, Direction.CLOCKWISE)

# Sensors
colorSensorLeft = ColorSensor(Port.E)
colorSensorRight = ColorSensor(Port.A)


# ------------------------------------------------------------------
# DRIVEBASE CONFIGURATION
# ------------------------------------------------------------------
# DriveBase simplifies moving the robot. 
# Arguments: left motor, right motor, wheel diameter (mm), axle track (distance between wheels in mm)
# You will need to change these values for your robot!
WHEEL_DIAMETER_MM = 56
AXLE_TRACK_MM = 96.5
drivebase = DriveBase(motorLeft, motorRight,
                      wheel_diameter=WHEEL_DIAMETER_MM,
                      axle_track=AXLE_TRACK_MM)

# Enable the built-in gyro for straight driving and accurate turns
drivebase.use_gyro(True)

# Raise the internal DriveBase trajectory limits. Pybricks defaults to 3/4 of
# the motor's 2000 deg/s^2, which is only ~733 mm/s^2 with 56 mm wheels. The
# optimized trajectories use up to ~6 m/s^2 and ~9k deg/s^2, so the default
# makes the inner reference lag the plan. These are upper limits, not commands;
# traction and the motor controller still bound the actual acceleration.
#
# Pybricks rejects acceleration limits above 20,000 deg/s^2 in motor units, so
# the values are derived from this robot's geometry and kept at 80% of that
# ceiling. A different wheel size or track width cannot make this call fail.
_MAX_MOTOR_ACCEL_DEG_S2 = 20000
_ACCEL_HEADROOM = 0.8
drivebase.settings(
    straight_acceleration=int(_ACCEL_HEADROOM * _MAX_MOTOR_ACCEL_DEG_S2
                              * math.pi * WHEEL_DIAMETER_MM / 360.0),  # mm/s^2
    turn_acceleration=int(_ACCEL_HEADROOM * _MAX_MOTOR_ACCEL_DEG_S2
                          * WHEEL_DIAMETER_MM / AXLE_TRACK_MM),          # deg/s^2
)

# Diagnostics
print("Battery Voltage:", hub.battery.voltage(), "mV")
if hub.battery.voltage() < 8000:
    print("WARNING: Battery is low! Please charge or swap.")

# ------------------------------------------------------------------
# GENERAL FUNCTIONS
# ------------------------------------------------------------------
# Here are some examples of how to make custom functions for your robot:


def wait_for_button():
    """Pauses the program until a button is pressed. Great for debugging."""
    print("PAUSED: Press any button to continue...")
    while not hub.buttons.pressed():
        wait(10)
    while hub.buttons.pressed():
        wait(10)
    print("Resuming!")


# ------------------------------------------------------------------
# ASYNC HELPER FUNCTIONS
# ------------------------------------------------------------------
# These functions are defined with 'async def'. This means they can be run 
# simultaneously alongside other tasks (like driving) using Pybricks multitask().

async def move_front_attachment(speed, angle):
    """Moves front attachment and waits for it to finish."""
    await motorFront.run_angle(speed, angle)

async def move_back_attachment(speed, angle):
    """Moves back attachment and waits for it to finish."""
    await motorBack.run_angle(speed, angle)

async def move_front_attachment_async(speed, angle):
    """Fires the front motor in the background and immediately moves to next line."""
    motorFront.run_angle(speed, angle, wait=False)

async def reset_front_attachment(speed, torque):
    """Runs the motor until it hits a physical stop to find the zero position."""
    await motorFront.run_until_stalled(speed, then=Stop.COAST, duty_limit=torque)


# ------------------------------------------------------------------
# ADVANCED OPTIONS SECTION
# ------------------------------------------------------------------
"""
ADVANCED: TRANSMISSIONS & ABSOLUTE ENCODERS
Our last FLL robots used a "transmission" - a single motor that shifts 
between multiple gears to control many attachments. We had to track the motor's exact absolute position.

How to set it up:
1. When defining the motor, pass `reset_angle=False` so it doesn't lose its position on startup:
   motorShift = Motor(Port.D, reset_angle=False)

2. Create a function that resets the angle to the true absolute value before moving:
   async def shift(gear):
       motorShift.reset_angle(None) # None forces it to read the absolute encoder
       target_angle = gear * 90 + 45
       motorShift.run_target(500, target_angle, wait=False)
"""
