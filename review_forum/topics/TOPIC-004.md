# TOPIC-004 Config sync silent

Status: OPEN
Author: robotics | Adversary: architect | Arbiter: verification
Owners: [robotics]
Target-Symbols: [overboard/deploy/hub_deploy.py::push_trajectory:config-sync]
Related-Topics: []
Round: R2

## [GRIEVANCE by robotics]:
`push_trajectory` wraps `RobotConfig.sync_pybricks_robot_config()` in bare `except Exception: pass` — hub keeps old `WHEEL_DIAMETER_MM`/`AXLE_TRACK_MM`, systematic scale error on every tick while approval hash says valid. Fix: fail-loud before write/connect; `--allow-stale-config` opt-out only.

## [REBUTTAL by architect]:
Agree fail-loud on `overboard/deploy/hub_deploy.py::push_trajectory:config-sync` — bare `except Exception: pass` is silent corruption, delete it.
Deletion shape: remove try/except, keep bare `sync_pybricks_robot_config()` call — exception already aborts before write/connect, +0/−3 lines, one-line diff.
Reject `--allow-stale-config`: new CLI flag + `gui_server.py` forward + studio checkbox + inventory/test pins ≈ +30 lines to bless a known-scale-wrong run.
Existing bypass already covers bench-hub: raw-script `hub_deploy` push is unrestricted, no second override needed.

## [AMENDMENT by architect]:
Delete `try/except Exception: pass` around config-sync (3 lines); no flag, no catch-and-raise wrapper.

## [CONCESSION by verification]:
Silent stale config breaks the approval guarantee — a valid hash with wrong `WHEEL_DIAMETER_MM`/`AXLE_TRACK_MM` is worse than a refused push, so fail-loud is required.

## [RESOLUTION]:
Accept architect AMENDMENT, reject `--allow-stale-config`. Owner: robotics. Ticket: delete bare `except Exception: pass` around `sync_pybricks_robot_config()`; exception aborts before write/connect; bench-hub uses unrestricted raw-script push. Status: RESOLVED.
