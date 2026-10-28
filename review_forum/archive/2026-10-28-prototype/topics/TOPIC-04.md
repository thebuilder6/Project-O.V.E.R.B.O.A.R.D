# TOPIC-04 Hub deploy trust — DEBATED

## TOPIC-04 — Hub deploy trust: forum debate (Ref Rina, channeling Coach + Robotics)

Source: `review_forum/synthesis/CLUSTERING.md:14-15` — Status: OPEN. Items from `review_forum/round1/robotics.md:3-10`.

### Item 1 — Config sync swallowed (`overboard/deploy/hub_deploy.py:257-266`)

**[REBUTTAL by coach]** — *Classroom impact.*
Twelve kids, one shared robot, three teams pushing in one period: Team B's hub still carries Team A's `WHEEL_DIAMETER_MM` / `AXLE_TRACK_MM` from last week. With the bare `except Exception: pass` at `hub_deploy.py:265-266`, the push "succeeds," the robot drives long/short all period, and the lesson ends with kids blaming their waypoints instead of a stale config. Silent is the worst pedagogy — nothing to point at. Counter-weight: a hard abort on *any* sync hiccup also kills the lesson — gym with no wifi, old hub firmware, one laptop — where the teacher knowingly wants to run on what's on the robot and sort config tomorrow.

**[REBUTTAL by robotics]** — *Hardware truth.*
The trajectory was validated against the *host* config, but the hub drives on whatever `robot_config.py` it still holds. A stale diameter/track is not noise, it is a systematic distance/turn scale error: the chassis walks into mission models while `check_trajectory_approval` (`hub_deploy.py:205-233`) still says "valid," because the recorded `trajectory_sha256` covers the host file, never hub state. Swallowing the sync failure decouples the approval hash from the hardware that executes it. Minimal hardware-safe behavior: log the exception and abort the push.

**[RESOLUTION]** — Fail-loud by default, explicit override for the classroom.
`push_trajectory` logs the sync exception and refuses the push unless the operator passes an explicit `--keep-hub-config` override, which is recorded in the run log/report alongside the config mismatch. Tradeoff: fail-loud protects the approval invariant (hash ⇒ hardware); the override keeps a wifi-less gym session moving but moves blame explicitly onto the operator, never silently onto the robot.

### Item 2 — Final `POSE` breaks the session contract (`pybricks_code/hub_runner.py:55`)

**[REBUTTAL by coach]** — *Classroom impact.*
The host parser drops unattributed poses, so a pushed mission's measured end pose never lands in `poses.csv` — and to a 12-year-old "no row" reads as "the robot didn't move" or, worse, "we were on target, the sheet just missed it." No `TEST_START` / run id means the teacher cannot match a printout to a team. Counter-weight: forcing kids to hand-maintain a `TEST_START`/`META,run_id`/`DATA`/`TEST_END,1` envelope around every push adds ceremony they *will* break when copy-pasting.

**[REBUTTAL by robotics]** — *Hardware truth.*
The AGENTS.md hub-to-host contract is `TEST_START <name>` / `META,run_id,…` / `DATA` block with `t_ms,…` header / `TEST_END,<n>`, with a pose taking the run id of the open test. `hub_runner.py:46-55` prints a bare `POSE,x,y,heading_deg` with no envelope and no run id — compare `sysid_digital_twin.py:166,271`, which emits pose inside the block. The host parser therefore cannot attribute the pose to a run and silently drops it. One strict writer beats N tolerant readers.

**[RESOLUTION]** — Fix the writer (fail-loud), tolerate the past (override).
Default: `hub_runner.main` emits the session envelope (`TEST_START hub_mission`, one `META,run_id`, `DATA`, `TEST_END,1`) around the run, so the existing parser stays strict and bare `POSE` lines warn/fail loudly. Back-compat override: one documented parser branch for legacy bare-`POSE` output, marked deprecated. Tradeoff: strict-writer keeps the contract load-bearing (future malformed sessions still caught); the reader-side branch is the escape hatch for logs already captured, at the cost of masking exactly one generation of malformed output.

### Item 3 — `MISSION_EVENTS = {}` silent no-op (`pybricks_code/hub_runner.py:26`)

**[REBUTTAL by coach]** — *Classroom impact.*
This is the cruelest classroom failure mode: the trajectory runs to completion, every `EVENT,name` prints via `_report` (`hub_runner.py:29-35`), the run *looks* clean — while the arm never fired and the team celebrates a zero-score run. Kids cannot distinguish "reported" from "acted." Counter-weight: raising on every unmapped event blocks the legitimate dry run — path-only driving practice with the attachment unmounted — which is half of gym time.

**[REBUTTAL by robotics]** — *Hardware truth.*
`MISSION_EVENTS = {}` (`hub_runner.py:26`) plus the `_report` fallback in `_event_map` (`hub_runner.py:38-43`) means an unmapped event is print-only by default, with the comment at `hub_runner.py:23-25` blessing it ("only reported, so the trajectory still runs"). The physical risk is not the print — it is the *next segment*: a missed lift/arm leaves hardware in the path and the robot drives through the mission model. Report-only default converts an actuation fault into a clean-looking run.

**[RESOLUTION]** — Fail-loud for scored runs, named override for practice.
`_event_map()` raises on an exported event with no entry in `MISSION_EVENTS` unless an explicit `ALLOW_NOOP_EVENTS` / dry-run flag is set (hub-side constant + host `--allow-noop-events` forwarded at push). Tradeoff: fail-loud catches the scored-mission disaster at the cost of one flag on practice days; the override preserves the fast drive-test loop but forces the team to *declare* "no attachments today," so the log distinguishes a dry run from a scored miss.

*Skipped: line-level patch wording and parser-branch placement — left to the implementing session. Risk: none of the three resolutions were sim-verified; Item 1's override flag name may collide with an existing CLI surface.*
