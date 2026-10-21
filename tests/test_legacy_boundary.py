"""The live tree must not depend on the archived implementation.

The archive holds code that has been replaced but is kept for reference: the
old Multi-Verse pipeline, the memetic search and the prototypes around them.
A live module importing from it would resurrect the old behaviour without the
tests and docs that describe it, so the boundary is checked mechanically here.

The archive lives at `overboard/legacy/` (`overboard.legacy` in import terms).
"""

import ast
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The archive itself is excluded: it is allowed to import whatever it likes.
# Generated, virtual-environment and cache trees are not source.
EXCLUDED_DIRS = {
    "legacy",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "build",
    "dist",
    "output",
    "outputs",
}


def _active_python_files():
    # Prune the excluded trees during the walk: rglob still descends into
    # .venv and legacy/, and enumerating them costs more than the check.
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(".py"):
                yield Path(root) / name


_TREES = {}


def _tree(path):
    """Parse a file once: both checks read the same tree."""
    if path not in _TREES:
        _TREES[path] = ast.parse(path.read_text(encoding="utf-8-sig"),
                                 filename=str(path))
    return _TREES[path]


def _is_archive(module):
    """True when `module` names the archive, however it is reached.

    `legacy`, `legacy.x`, `overboard.legacy` and `overboard.legacy.x` all
    import archived code.
    """
    return "legacy" in module.split(".")


def _legacy_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_archive(alias.name):
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_archive(module):
                yield node.lineno, module


def _legacy_sys_path_entries(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in {"append", "extend", "insert"}
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "sys"
        ):
            continue
        for argument in node.args:
            for inner in ast.walk(argument):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    # "legacy", "legacy/", "overboard/legacy" and their
                    # Windows spellings all put the archive on the path.
                    segments = inner.value.replace("\\", "/").split("/")
                    if "legacy" in segments:
                        yield node.lineno, inner.value


class TestLegacyBoundary(unittest.TestCase):
    def test_no_active_module_imports_the_archive(self):
        offenders = []
        for path in _active_python_files():
            tree = _tree(path)
            for line, module in _legacy_imports(tree):
                offenders.append(f"{path.relative_to(ROOT)}:{line} imports {module}")
        self.assertEqual(
            offenders,
            [],
            "Live modules must not import legacy/: " + "; ".join(offenders),
        )

    def test_no_active_module_puts_the_archive_on_sys_path(self):
        offenders = []
        for path in _active_python_files():
            tree = _tree(path)
            for line, entry in _legacy_sys_path_entries(tree):
                offenders.append(f"{path.relative_to(ROOT)}:{line} adds {entry}")
        self.assertEqual(
            offenders,
            [],
            "Live modules must not put legacy/ on sys.path: " + "; ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
