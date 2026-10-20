"""
The suite keeps an explicit list of the test classes it runs.

``run_tests.py`` imports the classes by name and hands that list to the
runner, so a class that is not in the list is never run. That is a quiet
hole: a new test class that nobody registered looks green, because it never
ran. This module closes it.
"""

import importlib
import os
import pkgutil
import unittest

import tests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _test_modules():
    """Every test module in the package, and the ones that cannot be imported.

    A module that needs an optional package (JAX, for instance) may not
    import here. That is not what this test is about, so it is skipped; the
    runner reports the missing package on its own.
    """
    modules = []
    for info in pkgutil.iter_modules(tests.__path__):
        if not info.name.startswith("test_"):
            continue
        try:
            modules.append(importlib.import_module("tests." + info.name))
        except ImportError:
            continue
    return modules


def _has_own_tests(cls):
    """True when the class has test methods of its own.

    A base class that other classes build on has none, and it is not
    something the runner has to run.
    """
    return any(name.startswith("test") for name in vars(cls))


class TestEveryTestClassRuns(unittest.TestCase):
    """Every test class in the tree is one the runner actually runs."""

    def test_every_test_class_is_registered(self):
        import run_tests

        registered = {cls.__name__ for cls in run_tests.test_classes()}
        unregistered = []
        for module in _test_modules():
            for name, obj in vars(module).items():
                if not isinstance(obj, type):
                    continue
                if not issubclass(obj, unittest.TestCase):
                    continue
                # A class imported from another module is that module's
                # business; only the ones defined here are ours to register.
                if obj.__module__ != module.__name__:
                    continue
                if not _has_own_tests(obj):
                    continue
                if name not in registered:
                    unregistered.append(f"{module.__name__}.{name}")

        self.assertEqual(
            unregistered, [],
            "these test classes are not in run_tests.test_classes(), so the "
            "suite never runs them: " + ", ".join(unregistered))

    def test_the_runner_has_no_class_that_does_not_exist(self):
        """A name in the list that no module defines is a broken list."""
        import run_tests

        defined = set()
        for module in _test_modules():
            for name, obj in vars(module).items():
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                    defined.add(name)

        missing = [cls.__name__ for cls in run_tests.test_classes()
                   if cls.__name__ not in defined]
        self.assertEqual(missing, [])

    def test_no_class_is_in_the_list_twice(self):
        """A class twice in the list runs its tests twice."""
        import run_tests

        names = [cls.__name__ for cls in run_tests.test_classes()]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        self.assertEqual(duplicates, [])


if __name__ == "__main__":
    unittest.main()
