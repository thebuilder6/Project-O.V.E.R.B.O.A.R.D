"""Where the repository's directories are.

Every module that needs a file from the repository - the hub firmware, the
example missions, the field pictures, the calibration folder - asks here
instead of climbing its own parents. Two reasons:

- The number of parents a file has to climb to reach the root is easy to get
  wrong, and a wrong one is silent: the path simply is not there. The tree has
  the same root spelled as ``parents[2]`` in one module and ``parent.parent``
  in the next, and both are correct today only by coincidence of depth.
- A directory that is resolved once can be checked once. ``tests/test_paths.py``
  pins that every directory this module names is really the one it claims.

The module is stdlib only. ``overboard/diagnostics.py`` has to work on a
machine where nothing is installed yet, so it may not pull in a dependency
just to learn where it is.
"""

from __future__ import annotations

from pathlib import Path

# The ``overboard`` package itself.
PACKAGE_ROOT = Path(__file__).resolve().parent

# The repository root is the parent of the package. Nothing else in the package
# may climb its own parents to find it.
REPO_ROOT = PACKAGE_ROOT.parent

# The firmware the hub runs. The runner and the generated mission live here,
# and pybricksdev bundles them with the robot config when it pushes.
PYBRICKS_DIR = REPO_ROOT / "pybricks_code"
HUB_MISSION_MODULE = PYBRICKS_DIR / "hub_mission.py"
HUB_RUNNER = PYBRICKS_DIR / "hub_runner.py"
ROBOT_CONFIG_PY = PYBRICKS_DIR / "robot_config.py"
# The script the one-command SysId loop pushes to the hub.
COLLECTOR_SCRIPT = PYBRICKS_DIR / "sysid_digital_twin.py"

# The missions the CLI and the studio start from.
EXAMPLES_DIR = REPO_ROOT / "examples"
MISSIONS_DIR = REPO_ROOT / "missions"

# The field pictures the studio draws on.
FIELDS_DIR = REPO_ROOT / "fields"
# The mat artwork. It is not tracked: FIRST's mat artwork is copyrighted, so
# it is downloaded on request and kept here.
MATS_DIR = REPO_ROOT / "mats"

# The robot, measured once: the session, the sheet, the poses and the fit.
CALIBRATION_DIR = REPO_ROOT / "calibration"

# The browser studio's own assets: the page, its script and its style.
VIZ_DIR = PACKAGE_ROOT / "viz"


__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "PYBRICKS_DIR",
    "HUB_MISSION_MODULE",
    "HUB_RUNNER",
    "ROBOT_CONFIG_PY",
    "COLLECTOR_SCRIPT",
    "EXAMPLES_DIR",
    "MISSIONS_DIR",
    "FIELDS_DIR",
    "MATS_DIR",
    "CALIBRATION_DIR",
    "VIZ_DIR",
]
