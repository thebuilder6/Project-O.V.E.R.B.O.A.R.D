"""
MISSION 5: LAMBDA FUNCTIONS
-----------------------------
Runs the optimized path imported from complete_mission.py with lambda functions for events.
"""

from robot_config import *
from ramsete import follow_trajectory
from complete_mission import samples as mission_samples
from complete_mission import config as mission_config

# Define the actions for the events specified in the trajectory
mission_events = {
    # This returns None immediately, but the motor keeps spinning 
    # in the background because wait=False
    "lower_arm": lambda: motorFront.run_angle(500, 90, wait=False),
    "intake":    lambda: motorFront.run(800),
    "stop_all":  lambda: motorFront.stop(),

    # You can also map to complex functions if needed
    "custom":    lambda: print("Doing a complex calculation!")
}

async def main():
    print("Starting Ramsete Mission...")
    
    # Optional: Reset gyro or set initial position if needed
    # initial_h = mission_samples[0]['heading']
    # hub.imu.reset_heading(math.degrees(initial_h))
    
    # Follow the optimized trajectory with physical constraints
    await follow_trajectory(drivebase, mission_samples, config=mission_config, debug=True, b=3.0, zeta=0.8, event_map=mission_events)
    
    print("Mission Complete!")

if __name__ == "__main__" or __name__ == "mission_5_lambdas":
    run_task(main())
