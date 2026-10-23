"""The desktop build's entry point (PyInstaller). Not used by the CLI.

Frozen, ``sys.executable`` is this program, so the studio cannot start the
solver with ``-m overboard.main``. It starts it with ``--run-cli`` instead, and
the hub with ``--run-hub``; both land here and call the same code the CLI does.
"""

import shutil
import sys


def main():
    from overboard import paths

    if paths.FROZEN:
        paths.STATE_ROOT.mkdir(parents=True, exist_ok=True)
        _seed_state_tree(paths)

    argv = sys.argv[1:]

    if argv[:1] == ["--run-cli"]:
        # The studio reads the solve line by line; a pipe is block-buffered and
        # -u is not available to a frozen build.
        sys.stdout.reconfigure(line_buffering=True)
        from overboard.main import main as cli
        return cli(args=argv[1:], prog_name="overboard")

    if argv[:1] == ["--run-hub"]:
        from overboard.deploy.hub_deploy import main as hub
        return hub(argv[1:])

    from overboard.main import main as cli
    cli(args=argv or ["--gui"], prog_name="overboard")


def _seed_state_tree(paths):
    """Copy the bundled firmware and documents into the writable folder, once.

    ``hub_mission.py`` is generated and ``robot_config.py`` is calibrated, so
    the firmware folder cannot be the read-only one the build ships. The
    documents go where the manual says they are, next to the user's files.
    """
    if not paths.PYBRICKS_DIR.is_dir():
        shutil.copytree(
            paths.ASSET_ROOT / "pybricks_code",
            paths.PYBRICKS_DIR,
            ignore=shutil.ignore_patterns("__pycache__", "archive"),
        )
    docs = paths.STATE_ROOT / "docs"
    if not docs.is_dir():
        shutil.copytree(paths.ASSET_ROOT / "docs", docs)
        for name in ("README.md", "CLAIMS.md", "KNOWN_ISSUES.md", "CHANGELOG.md"):
            shutil.copy2(paths.ASSET_ROOT / name, paths.STATE_ROOT / name)
    for directory in (paths.MISSIONS_DIR, paths.CALIBRATION_DIR, paths.MATS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
