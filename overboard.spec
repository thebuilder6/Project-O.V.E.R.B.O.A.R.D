# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the desktop build: the studio, offline, one folder.

    pyinstaller --noconfirm overboard.spec

One folder, never one file: a one-file build unpacks the whole tree (tens of
thousands of files) before it can show a window. Console stays on because the
studio starts the solver as a child process and reads its stdout.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [
    ("overboard/viz", "overboard/viz"),
    ("examples", "examples"),
    ("fields", "fields"),
    # The manual and the documents it points at. The app has to explain itself
    # on a machine that never saw the repository.
    ("docs", "docs"),
    ("README.md", "."),
    ("CLAIMS.md", "."),
    ("KNOWN_ISSUES.md", "."),
    ("CHANGELOG.md", "."),
]
binaries = []
hiddenimports = ["overboard.deploy.hub_deploy", "overboard.deploy.hub_ble"]

# The firmware is the seed the app copies into its writable folder on first
# run; __pycache__ is the build machine's, archive/ is not copied.
for base, dirs, files in os.walk("pybricks_code"):
    dirs[:] = [d for d in dirs if d not in {"__pycache__", "archive"}]
    for name in files:
        datas.append((os.path.join(base, name), os.path.relpath(base, ".")))

# Packages whose hooks do not cover their data files or submodules.
for package in (
    "casadi",
    "plotly",
    "pypdfium2",
    "jax",
    "jaxlib",
    "jaxopt",
    "jaxtyping",
    "pybricksdev",
    "bleak",
    "mpy_cross_v5",
    "mpy_cross_v6",
    "usb",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# immrax is imported, not collected: collecting it imports every submodule,
# which pulls pypoman -> cvxopt, and that pair hard-crashes the interpreter on
# some Windows installs. The import graph already reaches what the validator uses.
hiddenimports += ["immrax"]

# bleak talks to Windows through winrt, which collect_all cannot see.
try:
    hiddenimports += collect_submodules("winrt")
except Exception:  # noqa: BLE001 - only present on Windows
    pass

a = Analysis(
    ["overboard_desktop.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "IPython", "sphinx"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OVERBOARD",
    debug=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="OVERBOARD",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="OVERBOARD.app",
        bundle_identifier="com.thebuilder6.overboard",
        info_plist={"NSHighResolutionCapable": True},
    )
