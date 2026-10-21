"""
A test cache: a test that already passed is skipped when nothing it depends
on has changed since it passed.

The cache is a map from a test id to a fingerprint of the files that test
touched. The fingerprint is the hash of the contents of

  - the test module and the project modules it imports, transitively,
  - every file the test opens, every directory it lists, every project script
    it hands to a subprocess and every project module it imports while it
    runs,
  - the environment: the Python version and the versions of the packages that
    change the numbers (numpy, casadi, jax, immrax).

When the fingerprint still matches the one stored by the last successful run,
the test is skipped and its runtime is not paid again.

The rules that keep the cache honest:

  - Only a test that PASSED is skipped. A failure always runs again.
  - A test that wrote a file inside the repository is never skipped, because
    another test may read what it wrote.
  - The cache is a disposable JSON file in the repository root.
    `python run_tests.py --clear-cache` throws it away and
    `python run_tests.py --no-cache` ignores it and runs everything.
  - The cache never decides that a test passed. It only decides that a test
    does not have to run again, because the code it reads is byte for byte
    the code that passed.

Nothing here is imported by the planner: a plain solve never pays for it.
"""

import ast
import builtins
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CACHE_PATH = REPO_ROOT / ".test_cache.json"
SCHEMA = 1

# Directories that hold no project code.
_IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache",
    ".mypy_cache", "mats", "random_suite", "benchmarks", "output", ".roo",
}

# The runner and the cache itself are a dependency of every test: a change in
# either can change what a test does.
_ALWAYS_DEPS = ("run_tests.py", "testcache.py")

# The packages whose version changes the numbers a test asserts on.
_ENV_PACKAGES = ("numpy", "casadi", "jax", "jaxlib", "immrax", "scipy",
                 "Pillow", "pypdfium2", "PyYAML", "click", "matplotlib")

# The installed packages, read once.
_INSTALLED = None

# The project modules, read once.
_MODULES = None


def _rel(path):
    """The path relative to the repository root, or None when it is outside."""
    try:
        raw = os.fsdecode(path)
    except (TypeError, ValueError):
        return None
    # A subprocess sent to os.devnull opens "nul" on Windows, and that name
    # resolves against the working directory like any other relative path.
    # The null device is not a file in the repository, so it must not make
    # the test a writer - and an uncacheable one.
    if raw.lower() == os.devnull.lower():
        return None
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        # A relative path is read from the working directory, which is the
        # repository root when the suite runs, but the suite must not depend
        # on that.
        candidates.append(REPO_ROOT / raw)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except (TypeError, ValueError, OSError):
            continue
        try:
            return resolved.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
    return None


def _module_map():
    """Every project module, by dotted name, with the file behind it."""
    global _MODULES
    if _MODULES is not None:
        return _MODULES
    modules = {}
    for root, dirs, files in os.walk(REPO_ROOT):
        # The virtual environment and the caches are not project code, and
        # walking them costs seconds.
        dirs[:] = [d for d in dirs
                   if d not in _IGNORED_DIRS and not d.startswith(".")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = Path(root) / name
            rel = path.relative_to(REPO_ROOT)
            parts = list(rel.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts.pop()
            if not parts:
                continue
            modules[".".join(parts)] = path
    _MODULES = modules
    return modules


def _import_names(path, module_name):
    """The module names a file imports, however it imports them."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return set()
    names = set()
    package = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[: max(0, len(parts) - (node.level - 1))]
                base = ".".join(parts)
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            if not target:
                continue
            names.add(target)
            for alias in node.names:
                names.add(f"{target}.{alias.name}")
    return names


def _resolve(name, modules):
    """The longest module name in `name` that is a project module."""
    parts = name.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in modules:
            return candidate
        parts.pop()
    return None


def _closure(modules, roots):
    """The files of the project modules the roots reach, transitively."""
    seen = set()
    queue = [r for r in roots if r]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = modules.get(name)
        if path is None:
            continue
        for imported in _import_names(path, name):
            resolved = _resolve(imported, modules)
            if resolved and resolved not in seen:
                queue.append(resolved)
    return {modules[name].relative_to(REPO_ROOT).as_posix()
            for name in seen if name in modules}


def _imports_of_code(code):
    """The module names a snippet of code imports."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def _installed_versions():
    """The version of every installed package, read once."""
    global _INSTALLED
    if _INSTALLED is None:
        versions = {}
        try:
            for dist in importlib.metadata.distributions():
                name = dist.metadata["Name"]
                if name:
                    versions[name.lower()] = dist.version
        except Exception:  # noqa: BLE001 - a broken metadata is not fatal
            versions = {}
        _INSTALLED = versions
    return _INSTALLED


class _Watch:
    """Records the files a test reads, writes and hands to a subprocess."""

    def __init__(self):
        self.reads = set()
        self.writes = set()
        self.scripts = set()
        self.dirs = set()
        self.code_imports = set()
        self.new_modules = set()
        self._saved = {}
        self._before = set(sys.modules)

    def __enter__(self):
        self._saved = {
            "open": builtins.open,
            "io_open": io.open,
            "os_open": os.open,
            "os_listdir": os.listdir,
            "os_scandir": os.scandir,
            "popen": subprocess.Popen,
        }
        builtins.open = self._open(self._saved["open"])
        io.open = builtins.open
        os.open = self._os_open(self._saved["os_open"])
        os.listdir = self._listdir(self._saved["os_listdir"])
        os.scandir = self._scandir(self._saved["os_scandir"])
        subprocess.Popen = self._popen(self._saved["popen"])
        return self

    def __exit__(self, *exc_info):
        builtins.open = self._saved["open"]
        io.open = self._saved["io_open"]
        os.open = self._saved["os_open"]
        os.listdir = self._saved["os_listdir"]
        os.scandir = self._saved["os_scandir"]
        subprocess.Popen = self._saved["popen"]
        for name in set(sys.modules) - self._before:
            module = sys.modules.get(name)
            path = getattr(module, "__file__", None)
            if path and "__pycache__" not in os.fspath(path):
                self.new_modules.add(name)
        return False

    def _open(self, original):
        def wrapper(file, mode="r", *args, **kwargs):
            rel = _rel(file)
            if rel is not None:
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    self.writes.add(rel)
                else:
                    self.reads.add(rel)
            return original(file, mode, *args, **kwargs)
        return wrapper

    def _os_open(self, original):
        def wrapper(path, flags, *args, **kwargs):
            rel = _rel(path)
            if rel is not None:
                if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                    self.writes.add(rel)
                else:
                    self.reads.add(rel)
            return original(path, flags, *args, **kwargs)
        return wrapper

    def _popen(self, original):
        def wrapper(*args, **kwargs):
            argv = kwargs.get("args")
            if argv is None and args:
                argv = args[0]
            self._argv(argv)
            return original(*args, **kwargs)
        return wrapper

    def _listdir(self, original):
        def wrapper(path="."):
            self._directory(path)
            return original(path)
        return wrapper

    def _scandir(self, original):
        def wrapper(path="."):
            self._directory(path)
            return original(path)
        return wrapper

    def _directory(self, path):
        """A directory the test looked in: a file added there matters."""
        rel = _rel(path)
        if rel is not None:
            self.dirs.add(rel)

    def _argv(self, argv):
        if isinstance(argv, (str, bytes)):
            argv = [argv]
        if not isinstance(argv, (list, tuple)):
            return
        items = [a for a in argv if isinstance(a, (str, bytes))]
        for index, item in enumerate(items):
            text = item.decode("utf-8", "replace") if isinstance(item, bytes) else item
            if text == "-c" and index + 1 < len(items):
                code = items[index + 1]
                if isinstance(code, bytes):
                    code = code.decode("utf-8", "replace")
                self.code_imports |= _imports_of_code(code)
                continue
            if text.endswith(".py"):
                rel = _rel(text)
                if rel:
                    self.scripts.add(rel)


class _ResultProxy:
    """Forwards to the real result and remembers what happened to one test."""

    def __init__(self, real):
        self._real = real
        self.status = "passed"

    def __getattr__(self, name):
        return getattr(self._real, name)

    def addSuccess(self, test):
        self.status = "passed"
        self._real.addSuccess(test)

    def addFailure(self, test, err):
        self.status = "failed"
        self._real.addFailure(test, err)

    def addError(self, test, err):
        self.status = "failed"
        self._real.addError(test, err)

    def addSkip(self, test, reason):
        self.status = "skipped"
        self._real.addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self._real.addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.status = "failed"
        self._real.addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, outcome):
        self._real.addSubTest(test, subtest, outcome)


class Cache:
    """The fingerprints of the tests that passed, and when they passed."""

    def __init__(self, path=CACHE_PATH, enabled=True):
        self.path = Path(path)
        self.enabled = enabled
        self.entries = {}
        self.dep_sets = {}
        self.files = {}
        self.skipped = []
        self.ran = 0
        self.saved_seconds = 0.0
        self._modules = None
        self._by_path = {}
        self._closures = {}
        self._hashes = {}
        self._env = None
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if data.get("schema") == SCHEMA:
                self.entries = data.get("entries", {})
                self.dep_sets = data.get("dep_sets", {})
                self.files = data.get("files", {})

    # --- the fingerprint -------------------------------------------------

    def modules(self):
        """Every project module, by dotted name, and the file behind it."""
        if not self._by_path:
            self._modules = _module_map()
            self._by_path = {
                path.relative_to(REPO_ROOT).as_posix(): name
                for name, path in self._modules.items()
            }
        return self._modules

    def file_hash(self, rel):
        """The hash of a file's contents, or None when it is gone.

        A directory is hashed by its listing: a file added to it is a change
        to it, even when nothing in it was read.
        """
        path = REPO_ROOT / rel
        try:
            stat = path.stat()
        except OSError:
            return None
        if path.is_dir():
            stamp = (stat.st_mtime_ns, stat.st_size)
            cached = self._hashes.get(rel)
            if cached and cached[0] == stamp:
                return cached[1]
            value = self._dir_hash(path)
            self._hashes[rel] = (stamp, value)
            return value
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self._hashes.get(rel)
        if cached and cached[0] == stamp:
            return cached[1]
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
        except OSError:
            return None
        value = digest.hexdigest()
        self._hashes[rel] = (stamp, value)
        return value

    def _dir_hash(self, path):
        """The hash of a directory's listing.

        The cache file itself is not part of it: writing the cache must not
        change the fingerprint of the tree it describes.
        """
        try:
            mine = self.path.resolve()
        except OSError:
            mine = None
        try:
            entries = sorted(
                f"{entry.name}:{entry.stat().st_size}"
                for entry in os.scandir(path)
                if mine is None or Path(entry.path).resolve() != mine)
        except OSError:
            return None
        digest = hashlib.sha256()
        for entry in entries:
            digest.update(entry.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def env(self):
        """The fingerprint of the environment the numbers come from."""
        if self._env is not None:
            return self._env
        parts = [f"py{sys.version_info[0]}.{sys.version_info[1]}",
                 f"impl={platform.python_implementation()}",
                 f"os={sys.platform}"]
        for name in _ENV_PACKAGES:
            version = _installed_versions().get(name.lower())
            parts.append(f"{name}={version or 'none'}")
        self._env = "|".join(parts)
        return self._env

    def key(self, deps):
        """The fingerprint of a set of files, in this environment."""
        digest = hashlib.sha256()
        digest.update(f"schema={SCHEMA}\n".encode())
        digest.update(f"env={self.env()}\n".encode())
        for rel in sorted(deps):
            value = self.file_hash(rel)
            digest.update(f"{rel}={value or 'missing'}\n".encode())
        return digest.hexdigest()

    def closure(self, roots):
        """The files of the project modules the roots reach."""
        modules = self.modules()
        resolved = []
        for name in roots:
            found = _resolve(name, modules) if name else None
            if found:
                resolved.append(found)
        return _closure(modules, resolved)

    def static_deps(self, module_name):
        """The files a test module depends on before it runs a single line."""
        if module_name not in self._closures:
            deps = set(self.closure([module_name]))
            deps |= set(_ALWAYS_DEPS)
            self._closures[module_name] = deps
        return self._closures[module_name]

    # --- the entries -----------------------------------------------------

    def should_skip(self, test_id):
        """The entry of a test that may be skipped, or None."""
        if not self.enabled:
            return None
        entry = self.entries.get(test_id)
        if not entry or entry.get("status") != "passed":
            return None
        if entry.get("writes"):
            return None
        deps = self.dep_sets.get(entry.get("deps"))
        if not deps:
            return None
        if self.key(deps) != entry.get("key"):
            return None
        return entry

    def note_skip(self, test_id, entry):
        """Note a test that was skipped, and the time it saved."""
        self.skipped.append(test_id)
        self.saved_seconds += float(entry.get("duration") or 0.0)

    def deps_for(self, module_name, watch):
        """Every file a test depends on: the code and the data it touched."""
        deps = set(self.static_deps(module_name))
        deps |= watch.reads
        deps |= watch.writes
        deps |= watch.scripts
        deps |= watch.dirs
        deps |= self.closure(watch.code_imports)
        deps |= self.closure(watch.new_modules)
        # A script on the command line brings its own imports with it.
        self.modules()
        deps |= self.closure([self._by_path.get(rel) for rel in watch.scripts])
        deps |= set(_ALWAYS_DEPS)
        return {rel for rel in deps if (REPO_ROOT / rel).exists()}

    def record(self, test_id, deps, status, duration, writes=False):
        """Store the fingerprint of a test that just ran."""
        deps = sorted(deps)
        if not deps:
            return
        dep_id = hashlib.sha1("\n".join(deps).encode()).hexdigest()[:16]
        self.dep_sets[dep_id] = deps
        for rel in deps:
            value = self.file_hash(rel)
            if value:
                self.files[rel] = value
        self.entries[test_id] = {
            "deps": dep_id,
            "key": self.key(deps),
            "status": status,
            "duration": round(float(duration), 4),
            "writes": bool(writes),
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def changed_files(self):
        """The files the last run read that have changed since."""
        changed = []
        for rel, value in self.files.items():
            now = self.file_hash(rel)
            if now is None or now != value:
                changed.append(rel)
        return sorted(changed)

    def save(self):
        """Write the cache out. A cache that cannot be written is not fatal."""
        data = {
            "schema": SCHEMA,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "env": self.env(),
            "files": self.files,
            "dep_sets": self.dep_sets,
            "entries": self.entries,
        }
        try:
            self.path.write_text(json.dumps(data, indent=1, sort_keys=True),
                                 encoding="utf-8")
        except OSError:
            return False
        return True


# The `run` of every class the cache has seen, before the cache touched it.
_ORIGINALS = {}


def _original_run(klass):
    """The `run` a class would have used without the cache."""
    for parent in klass.__mro__:
        candidate = vars(parent).get("run")
        if candidate is None:
            continue
        if getattr(candidate, "_testcache_patched", False):
            saved = _ORIGINALS.get(parent)
            if saved is not None:
                return saved
            continue
        return candidate
    return unittest.TestCase.run


def instrument(test_classes, cache):
    """Let every test in the classes ask the cache before it runs."""
    for klass in test_classes:
        if getattr(klass.run, "_testcache_patched", False):
            continue
        _patch_run(klass, cache)


def _patch_run(klass, cache):
    """Give a test class a `run` that consults the cache first."""
    original = _original_run(klass)
    _ORIGINALS[klass] = original

    def run(self, result=None):
        test_id = (f"{type(self).__module__}.{type(self).__qualname__}."
                   f"{self._testMethodName}")
        entry = cache.should_skip(test_id)
        if entry is not None:
            cache.note_skip(test_id, entry)
            if result is None:
                result = self.defaultTestResult()
            result.startTest(self)
            try:
                result.addSkip(self, f"cache: unchanged since {entry.get('when')}")
            finally:
                result.stopTest(self)
            return result
        if result is None:
            result = self.defaultTestResult()
        proxy = _ResultProxy(result)
        watch = _Watch()
        started = time.perf_counter()
        try:
            with watch:
                original(self, proxy)
        except BaseException:  # noqa: BLE001 - the runner reports it
            proxy.status = "failed"
            raise
        finally:
            duration = time.perf_counter() - started
            deps = cache.deps_for(type(self).__module__, watch)
            cache.record(test_id, deps, proxy.status, duration,
                         writes=bool(watch.writes))
            cache.ran += 1
            if cache.ran % 50 == 0:
                # A run that dies half way keeps what it already proved.
                cache.save()
        return proxy

    run._testcache_patched = True
    klass.run = run
