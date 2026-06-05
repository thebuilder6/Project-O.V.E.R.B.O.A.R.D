"""
Path planning utilities for trajectory generation.
Provides Reeds-Shepp path generation for initial trajectory bootstrapping.
"""

from typing import List, Optional, Tuple
import numpy as np
from rsplan import path


class ReedsSheppPath:
    """
    Wrapper for Reeds-Shepp path planning using rsplan library.
    
    Reeds-Shepp paths are the shortest paths for a car-like robot
    that can move both forwards and backwards with a minimum turning radius.
    """
    
    def __init__(self, turning_radius: float = 0.3) -> None:
        """
        Initialize Reeds-Shepp planner.

        Args:
            turning_radius: Minimum turning radius in meters (default: 0.3m for FLL robot)
        """
        self.turning_radius = turning_radius
    
    def plan(self, start_pose: Tuple[float, float, float], goal_pose: Tuple[float, float, float], step_size: float = 0.05) -> Optional[List[Tuple[float, float, float]]]:
        """
        Generate a Reeds-Shepp path from start to goal.

        Args:
            start_pose: (x, y, heading) tuple in meters and radians
            goal_pose: (x, y, heading) tuple in meters and radians
            step_size: Distance between sample points in meters

        Returns:
            List of (x, y, heading) waypoints along the path, or None if planning fails
        """
        try:
            # rsplan path function: path(start_pose, end_pose, turn_radius, runway_length, step_size, length_tolerance)
            result = path(start_pose, goal_pose, self.turning_radius, 0.0, step_size)

            if result is None or len(result) == 0:
                return None

            # Convert path to list of waypoints
            waypoints = []
            for state in result:
                waypoints.append((state[0], state[1], state[2]))

            return waypoints
        except Exception:
            # If Reeds-Shepp fails, return None to fall back to linear interpolation
            return None


def linear_interpolation_waypoints(start_pose: Tuple[float, float, Optional[float]], goal_pose: Tuple[float, float, Optional[float]], num_points: int) -> List[Tuple[float, float, Optional[float]]]:
    """
    Generate linearly interpolated waypoints between two poses.

    Args:
        start_pose: (x, y, heading) tuple
        goal_pose: (x, y, heading) tuple
        num_points: Number of points to generate (including endpoints)

    Returns:
        List of (x, y, heading) tuples
    """
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start_pose[0] + t * (goal_pose[0] - start_pose[0])
        y = start_pose[1] + t * (goal_pose[1] - start_pose[1])

        # Interpolate heading with angle wrapping
        if start_pose[2] is not None and goal_pose[2] is not None:
            diff = (goal_pose[2] - start_pose[2] + np.pi) % (2 * np.pi) - np.pi
            heading = start_pose[2] + diff * t
        elif start_pose[2] is not None:
            heading = start_pose[2]
        elif goal_pose[2] is not None:
            heading = goal_pose[2]
        else:
            # Use direction to goal
            heading = np.arctan2(goal_pose[1] - start_pose[1], goal_pose[0] - start_pose[0])

        waypoints.append((x, y, heading))

    return waypoints
