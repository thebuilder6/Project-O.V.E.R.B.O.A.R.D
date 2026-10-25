# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions are
[PEP 440](https://peps.python.org/pep-0440/): `0.1.0a1` is the first alpha.

## [0.1.0a2] - 2026-10-26

Second alpha. The desktop build ships for Windows, macOS and Linux, the studio
can find the hub over Bluetooth, and a busy machine no longer loses a run.

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
- **Scan for hubs** in the studio's Run tab lists the devices Bluetooth can see
  and fills the hub-name field.
- The reachability envelope is drawn as its sample boxes on demand, and a
  refused solve keeps its numbers instead of leaving an empty field.
- A candidate whose check was skipped now reads "not checked", not "passed".
- `docs/ControlsDoc.pdf`, the paper on time-optimal generation and the
  reachability gate; `research/cli_matrix.py`, the measurement driver for the
  shipping CLI.

### Changed

- The validator reuses one check per candidate and skips a check a slower
  candidate cannot win. The selection rule is unchanged.
- `README.md` rewritten, with a real mission drawn in the studio guide.

### Fixed

- On Windows, cvxopt's bundled OpenBLAS aborted the interpreter after printing
  nothing, so `--validate` and the test suite died without a message. The
  process now pins one thread per BLAS/OpenMP pool before the import (KI-057).
- A busy machine no longer loses a run: the pools reserve far less memory
  (4.89 GB to 0.55 GB per run) and MUMPS is told its limit, so an exhausted
  machine returns a failed candidate instead of aborting the process (KI-057).
- An empty trajectory is reported as a failed candidate, not as an Immrax
  validation error (KI-056).
- The optional dependency pair is `jax==0.12.0` / `immrax==0.3.8`: the previous
  pair raised `MemoryError` in the tick propagation on this tree.

### Known limitations

- KI-020: the validator is not a full digital twin - motor, odometry, slip and
  DriveBase dynamics are only partly modelled. A passing plan is a prediction,
  not a guarantee on the field; calibrate the drivebase before trusting the
  envelope.
- KI-054: the envelope gate compares an absolute radius against a tube that grows
  with path length, so a long path can fail for being long. Use
  `--envelope-limit-per-metre` to relax it.
- KI-055: the gate returns a verdict, not a score: it refuses a mission it cannot
  approve rather than ranking it. `--best-effort` writes the refused artifact,
  marked as unapproved and not approved for the hub.

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
