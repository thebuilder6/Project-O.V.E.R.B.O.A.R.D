# Round 1 v3 — robotics (Solder-Sara)

Scope: `pybricks_code/odometry.py`, `ramsete.py`, `ramsete_fast.py`, `ramsete_compact{,_events}.py`, `hub_runner.py`, `robot_config.py`, `overboard/deploy/hub_deploy.py` + `hub_ble.py`.

Symbolic anchors: `ramsete.py::CONTROL_MS=20`, `ramsete_compact_events.py::_LOOP_MS=20`, single `Odometry` impl re-exported, single `KinematicLimiter` + `RamseteController`, session verbs `POSE,` / `EVENT,` / `TEST_START` / `DATA`.

1. Stale robot-config push is silent. `hub_deploy.py::push_trajectory:config-sync` wraps sync in bare `except Exception: pass`. Hub runs old `WHEEL_DIAMETER_MM / AXLE_TRACK_MM` → systematic scale error. Fix: fail-loud before write/connect.
2. Raised DriveBase accel limits remove last firmware backstop. `robot_config.py::limits` lifts to 80% of 20k deg/s² ceiling. Bad-SysId plan now executes instead of clamped. Combined with (1): full-torque drive into model.
3. 20ms hold can break under overrun. Validator assumes exact 20ms ZOH; on overrun loops reset phase, hold stretches, `max_dt` logged never gated. GC pause mid-mission violates force model with no abort. Fix: consecutive-overrun counter, stop loudly.
4. Low battery + session tail advisory-only. `<8000mV` warns but drives; `_DRAIN_SECONDS=0.3` best-effort disconnect can lose tail lines; no sequence/checksum.

Fail-loud fix with override: remove try/except around sync, `--allow-stale-config` opt-out; overrun counter with `ABORT,OVERRUN`; low-battery abort with opt-out.

**HOLD [VETO: HARDWARE_HAZARD]** — do not push until (1) fail-loud.

Skipped: full collector audit, BLE profile, complex_mission regen. Risk: `--allow-*` overrides re-open hazard — bench-hub only.
