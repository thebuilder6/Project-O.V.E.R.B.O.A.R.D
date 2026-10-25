"""
The test cache must skip a test only when nothing it reads has changed.

The cache is the thing that makes a second run over an unchanged tree cheap,
so the rules it follows are worth pinning down: a test that passed and whose
code and data are byte for byte the same is skipped, and anything else runs.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import testcache

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dummy(name, body):
    """A test class built here, so the loader never collects it."""
    return type(name, (unittest.TestCase,),
                {"test_it": body, "__module__": "dummy"})


def _run(klass, cache):
    """Run a test class under the cache and hand back the result."""
    testcache.instrument([klass], cache)
    try:
        suite = unittest.TestLoader().loadTestsFromTestCase(klass)
        result = unittest.TestResult()
        suite.run(result)
        return result
    finally:
        testcache._ORIGINALS.pop(klass, None)


def _deps_of(cache, test_id):
    """The files a recorded test depends on."""
    entry = cache.entries[test_id]
    return cache.dep_sets[entry["deps"]]


class TestCacheSkipsUnchangedCode(unittest.TestCase):
    """A test that passed runs again only when something it reads changed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = testcache.Cache(path=Path(self._tmp.name) / "cache.json")

    def test_a_passed_test_is_skipped_the_second_time(self):
        klass = _dummy("Passes", lambda self: self.assertTrue(True))
        first = _run(klass, self.cache)
        self.assertEqual(len(first.failures), 0)
        self.assertEqual(len(first.errors), 0)
        second = _run(klass, self.cache)
        self.assertEqual(len(second.skipped), 1)
        self.assertIn("cache:", second.skipped[0][1])

    def test_a_changed_file_makes_the_test_run_again(self):
        klass = _dummy("Passes", lambda self: self.assertTrue(True))
        _run(klass, self.cache)
        real = self.cache.file_hash

        def changed(rel):
            value = real(rel)
            return "0" * 64 if rel == "testcache.py" and value else value

        with patch.object(self.cache, "file_hash", side_effect=changed):
            self.assertIsNone(self.cache.should_skip("dummy.Passes.test_it"))

    def test_a_failed_test_is_never_skipped(self):
        def boom(self):
            self.fail("on purpose")

        klass = _dummy("Fails", boom)
        first = _run(klass, self.cache)
        self.assertEqual(len(first.failures), 1)
        second = _run(klass, self.cache)
        self.assertEqual(len(second.failures), 1)
        self.assertEqual(len(second.skipped), 0)

    def test_a_test_that_wrote_in_the_repository_is_not_skipped(self):
        """Another test may read what this one wrote, so it must run again."""
        probe = REPO_ROOT / ".test_cache_probe.tmp"

        def writer(self):
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("probe")
            self.addCleanup(lambda: probe.unlink(missing_ok=True))

        klass = _dummy("Writes", writer)
        _run(klass, self.cache)
        self.assertIsNone(self.cache.should_skip("dummy.Writes.test_it"))

    def test_a_test_that_reads_a_data_file_depends_on_it(self):
        def reader(self):
            with open(REPO_ROOT / "examples" / "example_straight.json",
                      "r", encoding="utf-8") as handle:
                handle.read()

        klass = _dummy("Reads", reader)
        _run(klass, self.cache)
        deps = _deps_of(self.cache, "dummy.Reads.test_it")
        self.assertIn("examples/example_straight.json", deps)


class TestCacheFingerprints(unittest.TestCase):
    """The fingerprint has to cover the code, the data and the environment."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = testcache.Cache(path=Path(self._tmp.name) / "cache.json")

    def test_the_import_closure_is_part_of_the_fingerprint(self):
        deps = self.cache.static_deps("tests.test_testcache")
        self.assertIn("testcache.py", deps)
        self.assertIn("tests/test_testcache.py", deps)

    def test_the_package_init_is_in_the_closure_of_a_module_inside_it(self):
        """Importing a.b.c runs a/__init__.py and a/b/__init__.py first."""
        deps = self.cache.static_deps("overboard.viz.fields")
        self.assertIn("overboard/__init__.py", deps)
        self.assertIn("overboard/viz/fields.py", deps)

    def test_the_runner_and_the_cache_are_dependencies_of_every_test(self):
        deps = self.cache.static_deps("tests.test_testcache")
        self.assertIn("run_tests.py", deps)
        self.assertIn("testcache.py", deps)

    def test_a_file_that_is_gone_changes_the_fingerprint(self):
        before = self.cache.key(["testcache.py"])
        with patch.object(self.cache, "file_hash", return_value=None):
            after = self.cache.key(["testcache.py"])
        self.assertNotEqual(before, after)

    def test_the_environment_is_part_of_the_fingerprint(self):
        before = self.cache.key(["testcache.py"])
        with patch.object(self.cache, "env", return_value="other"):
            after = self.cache.key(["testcache.py"])
        self.assertNotEqual(before, after)

    def test_the_code_of_a_subprocess_is_a_dependency(self):
        watch = testcache._Watch()
        with watch:
            subprocess.run([sys.executable, "-c", "import testcache"],
                           cwd=str(REPO_ROOT), capture_output=True)
        self.assertIn("testcache", watch.code_imports)

    def test_a_script_on_the_command_line_is_a_dependency(self):
        watch = testcache._Watch()
        with watch:
            subprocess.run([sys.executable, "testcache.py"],
                           cwd=str(REPO_ROOT), capture_output=True)
        self.assertIn("testcache.py", watch.scripts)

    def test_a_directory_the_test_looked_in_is_a_dependency(self):
        watch = testcache._Watch()
        with watch:
            os.listdir(str(REPO_ROOT / "pybricks_code"))
        self.assertIn("pybricks_code", watch.dirs)

    def test_a_directory_has_a_fingerprint_of_its_listing(self):
        value = self.cache.file_hash("pybricks_code")
        self.assertIsNotNone(value)
        self.assertEqual(value, self.cache.file_hash("pybricks_code"))


class TestCacheRecords(unittest.TestCase):
    """The cache is a JSON file that survives a reload."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "cache.json"

    def test_a_recorded_test_survives_a_reload(self):
        cache = testcache.Cache(path=self.path)
        cache.record("some.test", ["testcache.py"], "passed", 1.5)
        self.assertTrue(cache.save())
        again = testcache.Cache(path=self.path)
        entry = again.should_skip("some.test")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["status"], "passed")

    def test_the_file_is_valid_json(self):
        cache = testcache.Cache(path=self.path)
        cache.record("some.test", ["testcache.py"], "passed", 1.5)
        cache.save()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], testcache.SCHEMA)
        self.assertIn("some.test", data["entries"])

    def test_a_cache_from_another_schema_is_ignored(self):
        self.path.write_text(json.dumps({"schema": 999, "entries": {"x": {}}}),
                             encoding="utf-8")
        cache = testcache.Cache(path=self.path)
        self.assertEqual(cache.entries, {})

    def test_a_broken_cache_is_ignored(self):
        self.path.write_text("not json at all", encoding="utf-8")
        cache = testcache.Cache(path=self.path)
        self.assertEqual(cache.entries, {})

    def test_the_changed_files_are_the_ones_that_moved(self):
        cache = testcache.Cache(path=self.path)
        cache.record("some.test", ["testcache.py"], "passed", 1.5)
        with patch.object(cache, "file_hash", return_value="0" * 64):
            self.assertEqual(cache.changed_files(), ["testcache.py"])
