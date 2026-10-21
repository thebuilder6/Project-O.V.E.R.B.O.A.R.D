"""The live tree must not depend on the archived implementation.

`legacy/` holds code that has been replaced but is kept for reference.  A live
module importing from it would resurrect the old behaviour without the tests
and docs that describe it, so the boundary is checked mechanically here.
"""

import ast
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
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        yield path


def _legacy_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "legacy" or alias.name.startswith("legacy."):
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "legacy" or module.startswith("legacy."):
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
                    if inner.value == "legacy" or inner.value.startswith("legacy/"):
                        yield node.lineno, inner.value


class TestLegacyBoundary(unittest.TestCase):
    def test_no_active_module_imports_the_archive(self):
        offenders = []
        for path in _active_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
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
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for line, entry in _legacy_sys_path_entries(tree):
                offenders.append(f"{path.relative_to(ROOT)}:{line} adds {entry}")
        self.assertEqual(
            offenders,
            [],
            "Live modules must not put legacy/ on sys.path: " + "; ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
