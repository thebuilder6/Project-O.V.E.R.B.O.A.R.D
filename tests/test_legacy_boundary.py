"""The live tree must not depend on or reintroduce the archived legacy tree.

The archive lives in git tag archive/legacy-2026-10. This test guards against
accidental reintroduction of legacy directories or imports.
"""

import ast
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
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
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for name in files:
            if name.endswith(".py"):
                yield Path(root) / name


def _is_archive(module):
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


class TestLegacyBoundary(unittest.TestCase):
    def test_no_legacy_directory_exists(self):
        self.assertFalse((ROOT / "overboard" / "legacy").exists(),
                         "overboard/legacy/ must remain archived in git tag archive/legacy-2026-10")
        self.assertFalse((ROOT / "legacy").exists(),
                         "legacy/ must not exist in repository root")

    def test_no_active_module_imports_the_archive(self):
        offenders = []
        for path in _active_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for line, module in _legacy_imports(tree):
                offenders.append(f"{path.relative_to(ROOT)}:{line} imports {module}")
        self.assertEqual(offenders, [], "Live modules must not import legacy: " + "; ".join(offenders))


if __name__ == "__main__":
    unittest.main()
