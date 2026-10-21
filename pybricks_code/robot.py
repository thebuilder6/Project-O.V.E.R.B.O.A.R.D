"""
MAIN ENTRY POINT (HUB MENU)
---------------------------
Run this file on the hub to start a supported program from the hub's own
buttons, with no laptop attached:

    1  Follow the trajectory the host pushed last (hub_runner.py)
    2  Run the SysId data-collection protocol (sysid_digital_twin.py)

The host-side tools do the same jobs with the laptop connected:

    python -m overboard.main --push-hub ...   # writes hub_mission.py and runs option 1
    python -m overboard.sysid.collect ...     # the connected version of option 2

The demo missions this menu used to list are archived under archive/.
"""

from pybricks.tools import hub_menu

# The hub comes from the config so the battery can be checked before a run.
from robot_config import hub

print("Starting Robot...")
print("Battery: %dmV <- This value should be at least 8000 to run."
      % hub.battery.voltage())

selection = hub_menu(1, 2)
print("Selected: %d" % selection)

if selection == 1:
    try:
        import hub_mission  # noqa: F401  written by --push-hub
    except ImportError:
        print("No pushed trajectory found. Run --push-hub from the host first.")
    else:
        # Importing the runner follows the trajectory it finds in hub_mission.
        import hub_runner  # noqa: F401
elif selection == 2:
    # Importing the collector starts the SysId protocol (its __main__ guard
    # accepts this module name too).
    import sysid_digital_twin  # noqa: F401
