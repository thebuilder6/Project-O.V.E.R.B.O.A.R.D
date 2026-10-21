"""
MISSION 1: SIMPLE SEQUENTIAL MOVEMENT
-------------------------------------
This mission demonstrates basic, step-by-step movement using the DriveBase.
Each action waits for the previous one to finish before starting (synchronous).

Key Concepts Demonstrated:
- Basic DriveBase movements (straight, turn)
- Using the `wait_for_button()` debugger to pause mid-mission.
- Adjusting speed inline for precise movements.
- Running a blocking attachment motor command.
"""

from robot_config import *

async def main():
    # Drive to target
    await drivebase.straight(300)
    await drivebase.turn(90)
    
    # Pause until a button is pressed to check alignment 
    # make sure to remove this when competing!!!!
    wait_for_button()

    # Slow down for precision approach
    drivebase.settings(straight_speed=100)
    await drivebase.straight(50)
    
    # Move attachment (blocks until finished)
    await move_front_attachment(speed=500, angle=500)
    
    # Restore speed and return to base
    drivebase.settings(straight_speed=400)
    
    await drivebase.straight(-350)
    
    print("Mission 1 Complete!")

if __name__ == "__main__" or __name__ == "mission_1_simple":
    run_task(main())
