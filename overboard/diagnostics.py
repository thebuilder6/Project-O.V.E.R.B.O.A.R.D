"""
Environment checks for `python -m overboard.main --doctor` and the browser studio.

Everything here is read-only: it reports what is installed, what is missing and
what to do about it, without importing the heavy optional packages.

This module is stdlib only on purpose: it has to work on a machine where
nothing is installed yet, so `main.py` can report the install command instead
of dying with an ImportError.
"""

import importlib.metadata
import importlib.util
import os
import socket
import sys
from pathlib import Path

# Ensure repo root is on sys.path if invoked directly as a script
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

REPO_ROOT = Path(__file__).resolve().parent.parent

# (import name, pip name, required?)
CORE_PACKAGES = [
    ("numpy", "numpy", True),
    ("casadi", "casadi", True),
    ("click", "click", True),
    ("plotly", "plotly", True),
    ("matplotlib", "matplotlib", True),
]

OPTIONAL_PACKAGES = [
    ("jax", "jax", False),
    ("immrax", "immrax", False),
    ("pandas", "pandas", False),
]

EXAMPLE_FILES = [
    "examples/fll_choreo.json",
    "examples/example_straight.json",
    "examples/example_complex_mission.json",
]


def missing_core_packages():
    """The pip names of the required packages that are not installed."""
    return [pip_name for module, pip_name, required in CORE_PACKAGES
            if required and not module_available(module)]


def missing_optional_packages():
    """The pip names of the optional packages that are not installed."""
    return [pip_name for module, pip_name, _ in OPTIONAL_PACKAGES
            if not module_available(module)]


def install_script():
    """The one-click install script for this platform."""
    return "install.bat" if sys.platform.startswith("win") else "install.sh"


def module_available(name):
    """True when a module can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def port_free(port, host="127.0.0.1"):
    """True when nothing is listening on the port."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def check_environment(config_path=None, ports=(8000,)):
    """Return a list of check results: {name, status, detail, hint}.

    status is "ok", "warn" or "error".
    """
    checks = []

    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info >= (3, 9):
        checks.append({"name": "Python", "status": "ok", "detail": version, "hint": ""})
    else:
        checks.append({
            "name": "Python", "status": "warn", "detail": version,
            "hint": "Python 3.9 or newer is recommended.",
        })

    for module, pip_name, required in CORE_PACKAGES:
        if module_available(module):
            found = package_version(pip_name)
            checks.append({
                "name": pip_name, "status": "ok",
                "detail": found or "installed", "hint": "",
            })
        else:
            checks.append({
                "name": pip_name, "status": "error", "detail": "missing",
                "hint": f"pip install {pip_name}",
            })

    for module, pip_name, _ in OPTIONAL_PACKAGES:
        if module_available(module):
            found = package_version(pip_name)
            checks.append({
                "name": pip_name, "status": "ok",
                "detail": found or "installed", "hint": "",
            })
        else:
            checks.append({
                "name": pip_name, "status": "warn", "detail": "not installed",
                "hint": "pip install -r requirements-optional.txt",
            })

    for relative in EXAMPLE_FILES:
        path = REPO_ROOT / relative
        if path.exists():
            checks.append({"name": relative, "status": "ok", "detail": "found", "hint": ""})
        else:
            checks.append({
                "name": relative, "status": "warn", "detail": "missing", "hint": "",
            })

    if config_path:
        if not os.path.isabs(config_path):
            config_path = str(REPO_ROOT / config_path)
        if Path(config_path).exists():
            checks.append({
                "name": f"config {config_path}", "status": "ok", "detail": "found", "hint": "",
            })
        else:
            checks.append({
                "name": f"config {config_path}", "status": "error", "detail": "not found",
                "hint": "Pass -c/--config with a robot config file.",
            })

    try:
        with open(".write_test_tmp", "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(".write_test_tmp")
        checks.append({
            "name": "output directory", "status": "ok",
            "detail": os.getcwd(), "hint": "",
        })
    except OSError as exc:
        checks.append({
            "name": "output directory", "status": "error",
            "detail": str(exc), "hint": "Run from a directory you can write to.",
        })

    for port in ports:
        if port_free(port):
            checks.append({"name": f"port {port}", "status": "ok", "detail": "free", "hint": ""})
        else:
            checks.append({
                "name": f"port {port}", "status": "warn", "detail": "in use",
                "hint": "The studio tries the next free port.",
            })

    return checks


def format_checks(checks):
    """Render check results as plain text lines."""
    marks = {"ok": "[ok]  ", "warn": "[warn]", "error": "[fail]"}
    lines = []
    for check in checks:
        line = f"{marks.get(check['status'], '[?]')} {check['name']}: {check['detail']}"
        if check.get("hint"):
            line += f"  -> {check['hint']}"
        lines.append(line)
    return lines


def summarize(checks):
    """Return (ok, warn, error) counts."""
    ok = sum(1 for c in checks if c["status"] == "ok")
    warn = sum(1 for c in checks if c["status"] == "warn")
    error = sum(1 for c in checks if c["status"] == "error")
    return ok, warn, error


def run_doctor(config_path=None):
    """Print the environment report. Returns the process exit code."""
    checks = check_environment(config_path=config_path)
    print("OVERBOARD environment check")
    print("=" * 70)
    for line in format_checks(checks):
        print(line)
    ok, warn, error = summarize(checks)
    print("=" * 70)
    print(f"{ok} ok, {warn} warning(s), {error} problem(s)")

    missing = missing_core_packages()
    if error:
        print("\nThe core solver is not ready. Fix the [fail] lines above and try again.")
        if missing:
            print(f"  Missing: {', '.join(missing)}")
        print(f"  Easiest: run {install_script()} (it sets up a virtual environment"
              " and installs everything).")
        print("  Or:      pip install -r requirements.txt")
        return 1
    if missing_optional_packages():
        print("\nThe core solver is ready. The optional packages are missing, so "
              "--validate and sysid_fit.py do not work yet.")
        print("  To install them: pip install -r requirements-optional.txt")
    else:
        print("\nEverything is ready.")
    print("\nStart the studio with: python -m overboard.main --gui")
    return 0
