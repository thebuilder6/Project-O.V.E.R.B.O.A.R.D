"""Where the repository's directories are.

Every module that needs a file from the repository - the hub firmware, the
example missions, the field pictures, the calibration folder - asks here
instead of climbing its own parents. Two reasons:

- The number of parents a file has to climb to reach the root is easy to get
  wrong, and a wrong one is silent: the path simply is not there. The tree has
  the same root spelled as ``parents[2]`` in one module and ``parent.parent``
  in the next, and both are correct today only by coincidence of depth.
- A directory that is resolved once can be checked once. ``tests/test_paths.py``
  pins that every directory this module names is really the one it claims,
  and that no module in the package spells the root out for itself.

The module is stdlib only. ``overboard/diagnostics.py`` has to work on a
machine where nothing is installed yet, so it may not pull in a dependency
just to learn where it is.

The tools under ``research/`` live outside the package and still put the
repository root on ``sys.path`` themselves, one expression each: they have to
import the package before the package can be imported. That is theirs, and
``tests/test_paths.py`` does not look at them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# In the desktop build (PyInstaller) the package is unpacked into a temporary
# asset folder that is born and dies with the run, so the writable directories
# cannot live next to it. ASSET_ROOT is what the build ships, STATE_ROOT is
# where the user's missions, calibration and generated firmware go.
FROZEN = bool(getattr(sys, "frozen", False))


def _user_data_dir():
    """Where the desktop build keeps what the user makes."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "OVERBOARD"


# The ``overboard`` package itself.
PACKAGE_ROOT = Path(__file__).resolve().parent

# The repository root is the parent of the package. Nothing else in the package
# may climb its own parents to find it. In a checkout it is both roots, so
# nothing below this line behaves differently from before.
if FROZEN:
    ASSET_ROOT = Path(getattr(sys, "_MEIPASS", PACKAGE_ROOT.parent))
    STATE_ROOT = Path(os.environ.get("OVERBOARD_HOME") or _user_data_dir())
else:
    ASSET_ROOT = STATE_ROOT = PACKAGE_ROOT.parent

# Kept for the modules that read a shipped file. A bundled build resolves it to
# the unpacked assets, never to a writable folder.
REPO_ROOT = ASSET_ROOT

# The firmware the hub runs. The runner and the generated mission live here,
# and pybricksdev bundles them with the robot config when it pushes. The
# mission is generated and the robot config is calibrated, so this folder is
# writable: the bundled copy is only the seed (overboard_desktop.py).
PYBRICKS_DIR = STATE_ROOT / "pybricks_code"
HUB_MISSION_MODULE = PYBRICKS_DIR / "hub_mission.py"
HUB_RUNNER = PYBRICKS_DIR / "hub_runner.py"
ROBOT_CONFIG_PY = PYBRICKS_DIR / "robot_config.py"
# The script the one-command SysId loop pushes to the hub.
COLLECTOR_SCRIPT = PYBRICKS_DIR / "sysid_digital_twin.py"

# The missions the CLI and the studio start from.
EXAMPLES_DIR = ASSET_ROOT / "examples"
MISSIONS_DIR = STATE_ROOT / "missions"

# The field pictures the studio draws on.
FIELDS_DIR = ASSET_ROOT / "fields"
# The mat artwork. It is not tracked: FIRST's mat artwork is copyrighted, so
# it is downloaded on request and kept here.
MATS_DIR = STATE_ROOT / "mats"

# The robot, measured once: the session, the sheet, the poses and the fit.
CALIBRATION_DIR = STATE_ROOT / "calibration"

# The browser studio's own assets: the page, its script and its style.
VIZ_DIR = PACKAGE_ROOT / "viz"


__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "FROZEN",
    "ASSET_ROOT",
    "STATE_ROOT",
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
