# Round 1 — robotics (Solder-Sara)

**1. [GRIEVANCE] `overboard/deploy/hub_deploy.py:264-266` — validated geometry can silently diverge from the hub.**
`push_trajectory` syncs the robot config to `robot_config.py`, then swallows every failure with bare `except Exception: pass`. Hardware risk: the trajectory was validated against the host config, but the hub still drives on stale `WHEEL_DIAMETER_MM`/`AXLE_TRACK_MM` — a systematic distance/turn scale error that walks the chassis into mission models while the approval hash says "valid." Minimal fix: log the exception and abort the push unless the operator passes an explicit `--keep-hub-config` override.

**2. [GRIEVANCE] `pybricks_code/hub_runner.py:55` — final `POSE` breaks the session contract in AGENTS.md.**
The contract is `TEST_START <name>` / `META,run_id,…` / `DATA` / `TEST_END,<n>`, with a pose taking the run id of the open test. `hub_runner` prints bare `POSE,x,y,heading_deg` with no `TEST_START` and no run id (compare `sysid_digital_twin.py:166,271`, which emits both inside the block). Hardware risk: the host parser cannot attribute the pose to a run, so a pushed mission's measured end pose is silently dropped from `poses.csv` and the operator mistakes "no row" for "on target." Minimal fix: wrap the run (`TEST_START hub_mission`, one `META,run_id`, `DATA`, `TEST_END,1`) or document that `hub_runner` output is deliberately outside the session format and give the parser a branch for it.

**3. [QUESTION] `pybricks_code/hub_runner.py:26` — `MISSION_EVENTS = {}` makes every attachment a silent no-op.**
Unmapped events fall through to `_report`, which only prints `EVENT,name`. Hardware risk: the trajectory still runs to completion, so a missed arm/lift action looks like a clean run while the robot physically drives through the mission model on the next segment. Is that the intent — report-only by default? Minimal fix: fail loud — raise in `_event_map()` when an exported event has no entry in `MISSION_EVENTS`, or require an explicit `ALLOW_NOOP_EVENTS` flag for dry runs.

**4. [SUGGESTION] `pybricks_code/ramsete.py:299` + `ramsete_compact_events.py:223` — limiter uses wall-clock `dt`, validator models an exact 20 ms hold.**
`dt = t - last_t` includes GC pauses and the overrun path (`next_loop` reset + `wait(1)`), so a late tick computes a *smaller* required accel and under-constrains exactly the tick with the longest hold — the one the wheel-force error is proportional to (`immrax_validator.py`, `CONTROL_DT`). Hardware risk: force/traction spikes on overrun ticks pass the gate on paper and slip on the mat. Minimal fix: feed the limiter the nominal `CONTROL_MS` (or clamp `dt` to e.g. 1.5× nominal) instead of the measured wall interval.

Skipped: battery-voltage gating (`robot_config.py:87-88` warns but never aborts) and the archived `sysid_data_logger.py` f-strings — out of the 4-finding budget, no sim re-verification run.
