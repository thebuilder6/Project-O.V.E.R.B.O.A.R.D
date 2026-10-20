"""
HUB RUNNER (Pybricks MicroPython)
----------------------------------
Entry point for a trajectory pushed from the host.

The host writes the packed trajectory to hub_mission.py next to this file
(main.py --push-hub) and pushes this script. pybricksdev bundles the local
imports, so robot_config, ramsete_compact_events and hub_mission all travel
with it.

The console output is the result channel: the final pose is printed as
POSE,x_m,y_m,heading_deg and every exported event as EVENT,name, so the host
can turn the session into the poses.csv that sysid_compare.py reads.
"""

import umath as math
from pybricks.tools import run_task

import robot_config
from ramsete_compact_events import follow_trajectory
from hub_mission import samples, config, events

# Actions for the exported events. Fill this in for your robot, the same way
# mission_5_lambdas.py does it. An event with no action here is only
# reported, so the trajectory still runs.
MISSION_EVENTS = {}


def _report(name):
    """Stand-in action for an event the robot has no action for."""

    def action():
        print("EVENT,%s" % name)

    return action


def _event_map():
    """Map exported event names onto actions."""
    actions = {}
    for _, name in events or ():
        actions[name] = MISSION_EVENTS.get(name) or _report(name)
    return actions


async def main():
    print("Hub runner: following the pushed trajectory...")
    x, y, heading = await follow_trajectory(
        robot_config.drivebase,
        samples,
        config=config,
        events=events,
        event_map=_event_map(),
    )
    print("POSE,%.4f,%.4f,%.2f" % (x, y, math.degrees(heading)))
    print("Hub runner: done.")


run_task(main())
