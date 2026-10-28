"""Wheel odometry: the robot's own pose estimate from the drivebase.

This is the one implementation of the odometry in the firmware (KI-027):
``ramsete.py`` re-exports ``Odometry`` from here, and the SysId collector
imports it from here, so the planner, the validator model and the collector
all read the same pose. Its heading is counter-clockwise positive, the same
convention as the planned trajectory.

It lives on its own so the collector can use it without loading the Ramsete
maths and the trajectory loop: on a small hub that bytecode is the difference
between collecting and a MemoryError before the first test.
"""

import umath as math


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class Odometry:
    """
    Tracks the global position of the robot using Pybricks DriveBase distance and angle.
    Assumes initial pose is (x, y, heading) where heading is in radians.
    """
    def __init__(self, drivebase, initial_x=0.0, initial_y=0.0, initial_heading=0.0):
        self.db = drivebase
        self.x = initial_x
        self.y = initial_y
        self.heading = initial_heading

        # DriveBase metrics
        self.last_dist = self.db.distance() / 1000.0  # Convert mm to meters
        # Pybricks angle is positive-clockwise. We invert it so positive is Counter-Clockwise (standard math)
        self.last_angle = math.radians(-self.db.angle())

    def update(self):
        """Updates the internal (x, y, heading) based on changes since last update."""
        dist = self.db.distance() / 1000.0
        angle = math.radians(-self.db.angle())

        d_dist = dist - self.last_dist
        d_angle = angle - self.last_angle

        # Use average heading during the move for slightly better integration
        avg_angle = self.heading + d_angle / 2.0

        self.x += d_dist * math.cos(avg_angle)
        self.y += d_dist * math.sin(avg_angle)
        self.heading += d_angle
        self.heading = normalize_angle(self.heading)

        self.last_dist = dist
        self.last_angle = angle
