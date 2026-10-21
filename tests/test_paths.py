"""The repository's directories live in one module.

``overboard/paths.py`` is the only place that knows where the root, the
firmware, the examples, the fields and the calibration folder are. A module
that resolves them on its own is a second answer to the same question, and
the two answers drift: the root was spelled ``parents[2]`` in one module and
``parent.parent`` in the next, and a file that moved up a directory would
have kept resolving to a path that no longer exists.

The check is mechanical on purpose. It reads the tree rather than the
imports, so a module that rebuilds the root by hand is caught even when it
never mentions ``overboard.paths``.
"""

import ast
import os
import unittest
from pathlib import Path

from overboard import paths

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "overboard"

# The three entry points that may be run as a standalone script, before the
# package is importable. They put the root on sys.path themselves, and that
# one bootstrap is the reason the root is spelled out in them at all. Every
# other module asks overboard.paths.
BOOTSTRAP_MODULES = {
    "main.py",
    "gui_server.py",
    "diagnostics.py",
}


def _live_modules():
    """Every module of the shipped package, the archive left out.

    The archive is allowed to be whatever it was: it is kept for reference,
    it has no CLI entry point, and nothing live imports it.
    """
    for base, dirs, files in os.walk(PACKAGE_DIR):
        dirs[:] = [d for d in dirs if d not in {"legacy", "__pycache__"}]
        for name in files:
            if name.endswith(".py"):
                yield Path(base) / name


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _climbs_to_the_root(tree):
    """The places where a module walks up to the repository root itself.

    Two spellings reach the root from inside the package: ``parents[2]`` and
    a doubled ``.parent.parent``. Both are listed with the line they are on.
    """
    found = []
    for node in ast.walk(tree):
        # Path(...).parents[2]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
            and node.slice.value >= 2
        ):
            found.append(node.lineno)
        # Path(...).parent.parent
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "parent"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parent"
        ):
            found.append(node.lineno)
    return found


class TestCanonicalPaths(unittest.TestCase):
    """The paths module names the directories the tree actually has."""

    def test_the_root_is_the_parent_of_the_package(self):
        self.assertEqual(paths.REPO_ROOT, ROOT)
        self.assertEqual(paths.PACKAGE_ROOT, PACKAGE_DIR)
        self.assertTrue((paths.PACKAGE_ROOT / "__init__.py").is_file())

    def test_the_named_directories_are_the_ones_they_claim(self):
        self.assertEqual(paths.PYBRICKS_DIR, paths.REPO_ROOT / "pybricks_code")
        # The deploy package publishes the firmware folder under this name.
        self.assertEqual(paths.PYBRICKS_DIR, paths.PYBRICKS_DIR)
        self.assertEqual(paths.EXAMPLES_DIR, paths.REPO_ROOT / "examples")
        self.assertEqual(paths.MISSIONS_DIR, paths.REPO_ROOT / "missions")
        self.assertEqual(paths.FIELDS_DIR, paths.REPO_ROOT / "fields")
        self.assertEqual(paths.MATS_DIR, paths.REPO_ROOT / "mats")
        self.assertEqual(paths.CALIBRATION_DIR, paths.REPO_ROOT / "calibration")
        self.assertEqual(paths.VIZ_DIR, paths.PACKAGE_ROOT / "viz")

    def test_the_tracked_directories_are_there(self):
        """A directory the tree ships has to exist.

        ``calibration`` and ``mats`` are the exception: the studio creates
        them when the operator needs them, so they may be absent on a fresh
        checkout.
        """
        for directory in (
            paths.PYBRICKS_DIR,
            paths.EXAMPLES_DIR,
            paths.FIELDS_DIR,
            paths.VIZ_DIR,
        ):
            self.assertTrue(directory.is_dir(), f"{directory} is missing")

    def test_the_firmware_files_are_inside_the_firmware_directory(self):
        for firmware in (
            paths.HUB_MISSION_MODULE,
            paths.HUB_RUNNER,
            paths.ROBOT_CONFIG_PY,
            paths.COLLECTOR_SCRIPT,
        ):
            self.assertEqual(firmware.parent, paths.PYBRICKS_DIR)
        self.assertTrue(paths.HUB_RUNNER.is_file())
        self.assertTrue(paths.ROBOT_CONFIG_PY.is_file())

    def test_no_live_module_resolves_the_repository_root_itself(self):
        """Only the standalone entry points may spell the root out.

        They have to: they run before the package is importable. Everything
        else asks ``overboard.paths``, so there is one answer to where the
        root is.
        """
        offenders = []
        for path in _live_modules():
            if path.name == "paths.py":
                continue
            climbs = _climbs_to_the_root(_tree(path))
            if not climbs:
                continue
            if path.name in BOOTSTRAP_MODULES:
                # The bootstrap is one expression, on the way to sys.path.
                if len(climbs) > 1:
                    offenders.append(
                        f"{path.relative_to(ROOT)} climbs to the root on lines {climbs}")
                continue
            offenders.append(
                f"{path.relative_to(ROOT)} climbs to the root on lines {climbs}")
        self.assertEqual(
            offenders, [],
            "these modules resolve the repository root themselves: " + "; ".join(offenders))

    def test_the_modules_that_need_a_directory_ask_the_paths_module(self):
        """A module that builds a repository path uses the paths module.

        The directory names are the ones the tree is built from, so a literal
        on the right of a path join means somebody rebuilt a path by hand.
        A name that is only a key in a JSON payload is not a path, so only
        the join and the ``Path()`` call are looked at.
        """
        directory_names = {
            "pybricks_code", "examples", "missions", "fields", "mats", "calibration",
        }
        offenders = []
        for path in _live_modules():
            if path.name == "paths.py":
                continue
            tree = _tree(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    right = node.right
                    if (
                        isinstance(right, ast.Constant)
                        and isinstance(right.value, str)
                        and right.value in directory_names
                    ):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} joins {right.value!r}")
                elif isinstance(node, ast.Call):
                    func = node.func
                    name = getattr(func, "id", None) or getattr(func, "attr", None)
                    if name != "Path" or not node.args:
                        continue
                    arg = node.args[0]
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value in directory_names
                    ):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} names {arg.value!r}")
        self.assertEqual(
            offenders, [],
            "these modules spell a repository directory out: " + "; ".join(offenders))


if __name__ == "__main__":
    unittest.main()
