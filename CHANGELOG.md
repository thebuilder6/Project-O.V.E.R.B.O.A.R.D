# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions are
[PEP 440](https://peps.python.org/pep-0440/): `0.1.0a1` is the first alpha.

## [Unreleased]

### Added

- Desktop build for Windows, macOS and Linux: `overboard.spec` (PyInstaller, one
  folder), `installer.iss` for the Windows installer, and
  `.github/workflows/desktop.yml` to build all three on a `v*` tag. It opens the
  studio with no Python and no checkout; the local-only studio and the export gate
  are unchanged.
- In-app manual (`overboard/viz/help.html`, **Help** in the studio) and the
  project documents copied to `docs/` in the state folder on first run, so the
  packaged app explains itself offline.
- `overboard/paths.py` separates the read-only bundle (`ASSET_ROOT`) from the
  writable folder (`STATE_ROOT`, moved with `OVERBOARD_HOME`).

### Fixed

- On Windows, cvxopt's bundled OpenBLAS aborted the interpreter after printing
  nothing, so `--validate` and the test suite died without a message. The
  process now pins `OPENBLAS_NUM_THREADS=1` before the import.
- The optional dependency pair is `jax==0.12.0` / `immrax==0.3.8`: the previous
  pair raised `MemoryError` in the tick propagation on this tree.

## [0.1.0a1] - 2026-10-22

First alpha release. The planner, the validator and the controller export work
end to end from the CLI and from the browser studio. The robot side is not yet
validated against a calibrated drivebase.

### Added

- Time-optimal trajectory solve for a LEGO/FLL differential-drive robot with
  CasADi + IPOPT, a per-wheel force and slip model, lateral weight transfer, and
  a per-segment timing model held to the wheel-jerk discipline.
- The 3-Pillar unified planner as the only planner: the three frontier
  candidates (`sprint`, `sweeper`, `exploratory`) and `--candidates N`.
- Reachability validation gate (`--validate`, JAX/Immrax) with the `plant` and
  `tick` propagation models, the run-wide envelope budget and its per-metre
  allowance, and one machine-readable report line per run.
- Choreo v3 `.traj` export, controller JSON, Python and compact-module controller
  exports, and the push-to-hub path (`--push-hub`, `overboard.deploy.hub_deploy`).
- Browser studio (`--gui`): draw a mission, solve, compare candidates, export;
  it never bypasses the export gate.
- SysId loop: session collection over BLE/USB, fitting, and the digital-twin
  compare (`overboard.sysid.*`, `docs/CALIBRATION_PROCEDURE.md`).
- `--doctor` diagnostics, `install.bat` / `install.sh` and the studio launchers.

### Known limitations

- KI-020: the validator is not a full digital twin - motor, odometry, slip and
  DriveBase dynamics are only partly modelled. A passing plan is a prediction,
  not a guarantee on the field; calibrate the drivebase before trusting the
  envelope.
- KI-054: the envelope gate compares an absolute radius against a tube that grows
  with path length, so a long path can fail for being long. Use
  `--envelope-limit-per-metre` to relax it.
- `--best-effort` writes an unapproved artifact. It is marked in the `.traj` and
  in `_stats.json`, and it is not approved for the hub.

### Not included

- No package on PyPI. Install is clone plus `install.bat` / `install.sh`, and
  every command runs as `python -m overboard.…` from the repository root.
- The Multi-Verse pipeline, the rescue round and the retired solve flags are
  removed; the old pipeline is archived in the git tag `archive/legacy-2026-10`.
