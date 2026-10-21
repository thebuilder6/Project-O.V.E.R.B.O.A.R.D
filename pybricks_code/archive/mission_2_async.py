"""
MISSION 2: ADVANCED ASYNC MOVEMENT
----------------------------------
Demonstrates asynchronous programming to save time. 

Key Concepts Demonstrated:
- Using `await multitask()` to perform actions simultaneously.
- Saving time by moving arms *while* driving.
- Mixing sequential setup moves with simultaneous execution.
"""

from robot_config import *

async def main():
    print("Starting Mission 2: Advanced Async Movement")
    
    # Standard sequential setup
    await drivebase.straight(330)
    await drivebase.turn(90)
    
    # Multitask: Drive straight AND move back arm simultaneously
    await multitask(
        drivebase.straight(705),
        move_back_attachment(speed=500, angle=360)
    )
    
    # Reposition
    await drivebase.turn(90)
    await drivebase.straight(175)
    
    # Turn first, then Multitask: Backup AND use front arm simultaneously
    await drivebase.turn(-6)
    await multitask(
        move_front_attachment(speed=700, angle=2000),
        drivebase.straight(-100)
    )
    
    print("Mission 2 Complete!")

# This ensures the code only runs if this file is imported or run directly
if __name__ == "__main__" or __name__ == "mission_2_async":
    run_task(main())
