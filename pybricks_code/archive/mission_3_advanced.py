"""
MISSION 3: THE ULTIMATE ADVANCED MISSION
----------------------------------------
This mission combines everything to demonstrate a highly competitive, 
time-saving robotics run.

Key Concepts Demonstrated:
- Stall Detection: Resetting arms blindly by running them until they stall against a physical limit.
- Dynamic Settings: Changing speeds and tolerances inline for speed across the board vs. precision near targets.
- Momentum Preservation: Using `curve()` to turn without stopping.
- Wall Squaring: Disabling the gyro to flatten the robot against a wall to regain perfect 100% accuracy mid-match.
- Fire-and-forget: Using async without `await` to fling arms while driving away.
"""

from robot_config import *
from pybricks.tools import StopWatch, wait
from pybricks.parameters import Stop

async def main():
    print("Starting Mission 3: Advanced Concepts")
    
    # Start a timer to benchmark our time
    timer = StopWatch()
    timer.reset()

    # ARM RESET (STALL DETECTION)
    # At the start of the match, it is important to reset arms to a known position.
    # We can run it backwards until it hits a physical stop to "zero" it out.
    # This will run the motor backwards at 300 deg/s until it stalls against a limit
    await reset_front_attachment(speed=-300, torque=30)

    await move_front_attachment(speed=500, angle=90)

    # FAST DRIVING WITH HIGH TOLERANCES
    # We have a long distance to travel, so we don't care if we are off by a few mm.
    # We increase speed and tolerances to prevent stuttering at the end of the move.
    drivebase.settings(straight_speed=500, straight_acceleration=800)
    # I don't suggest changing tolerances without a specific reason or testing!
    drivebase.distance_control.target_tolerances(speed=100, position=10) 
    
    await drivebase.straight(800)

    # PRECISION APPROACH WITH ASYNC
    # We are near the target. Slow down, tighten tolerances, and lower the arm
    # while we drive the final few inches to save time
    drivebase.settings(straight_speed=100, straight_acceleration=300)
    drivebase.distance_control.target_tolerances(speed=20, position=2)
    
    await multitask(
        drivebase.straight(150),
        move_front_attachment(speed=300, angle=-90)
    )
    
    # Restore normal settings!
    drivebase.settings(straight_speed=500, straight_acceleration=800)
    drivebase.distance_control.target_tolerances(speed=100, position=10)
    
    # FIRE AND FORGET
    # We don't want to wait for it to finish before we start driving home!
    await move_front_attachment_async(speed=800, angle=360)

    # SAVING MOMENTUM WITH CURVES
    # Instead of driving straight, stopping, and turning 90 degrees, we can drive 
    # in a smooth arc! This maintains momentum and saves huge amounts of time.
    # Radius=300mm, turning right by 90 degrees
    await drivebase.curve(radius=300, angle=90)

    # SQUARING ON A WALL (GYRO DISABLED)
    # The gyro can drift slightly over time. To ensure perfect accuracy, 
    # we can turn the gyro OFF and let the robot "feel" the wall.
    drivebase.use_gyro(False)
    
    await multitask(
        motorLeft.run_until_stalled(speed=200, then=Stop.HOLD, duty_limit=30),
        motorRight.run_until_stalled(speed=200, then=Stop.HOLD, duty_limit=30)
    )
    
    # Turn gyro back on to reset heading to exactly 0
    drivebase.use_gyro(True)

    # RUSH HOME

    drivebase.settings(straight_speed=600)
    await drivebase.straight(-950)
    
    # TURN SETTINGS AND TOLERANCES
    # We can also change the turn rate and turn acceleration
    drivebase.settings(turn_rate=300, turn_acceleration=600)
    
    # Relax heading tolerances if it stutters at the end of a fast turn
    drivebase.heading_control.target_tolerances(speed=50, position=5)
    await drivebase.turn(180)
    
    # THE WAIT COMMAND
    print("Pausing for 1 second before finishing...")
    wait(1000)

    print("Mission 3 Complete! Total time:", timer.time(), "ms")

if __name__ == "__main__" or __name__ == "mission_3_advanced":
    run_task(main())

