# Pushing a Trajectory to the Hub from the Studio

**Status:** 2026-10-20. All four phases are implemented and tested; the push has never been run against a
real hub. The plan is kept because it records why the push is shaped the way it is, and because the
first real push still has to confirm the conventions in [Section 7](#7-gotchas).
**Scope:** the browser studio's "push to hub" feature, and how it ties the mission files, the robot-side code and the sysid loop together.
**Audience:** anyone implementing it, and anyone changing the hub path afterwards.

> **Read this first.** The push path works from the browser now, but no trajectory has ever been pushed to a **real** hub from the studio: the tests use a stand-in for the hub client. The measured pose the hub reports is its **own odometry**, not ground truth: see [Section 7](#7-gotchas).

---

## Table of contents

- [Decisions](#decisions)
- [1. The loop](#1-the-loop)
- [2. What already exists](#2-what-already-exists)
- [3. The gap](#3-the-gap)
- [4. The pieces](#4-the-pieces)
  - [4.1 Pushing an existing trajectory](#41-pushing-an-existing-trajectory)
  - [4.2 The push as a studio job](#42-the-push-as-a-studio-job)
  - [4.3 The studio's UI](#43-the-studios-ui)
- [5. The integrations](#5-the-integrations)
  - [5.1 Missions](#51-missions)
  - [5.2 Robot code](#52-robot-code)
  - [5.3 Sysid](#53-sysid)
- [6. The phases](#6-the-phases)
- [7. Gotchas](#7-gotchas)
- [8. Deferred](#8-deferred)
- [9. Still open](#9-still-open)

---

## Decisions

These are settled; the rest of the document assumes them.

| # | Decision | Why |
|---|---|---|
| D1 | **The push is a separate button on a passed solve**, not a checkbox on Solve. | The plan is already approved and written; re-solving to push it is wasteful and could produce a different trajectory than the one that was validated. |
| D2 | **The session is written next to the mission**, as `missions/<name>.session.csv`. | The plan and what the robot actually did then travel together, and "compare" finds them without a second argument. |
| D3 | **The button-press SysId loop is a later phase**, not part of this feature. | It is a different interaction: the operator is at the robot pressing the hub button between tests, not in a browser. |
| D4 | **A geometry mismatch is a warning**, with a one-click fix, not a blocker. | `pybricks_code/robot_config.py` is often stale, and the plan may still be the right one. The warning says which number differs; the user decides. |

---

## 1. The loop

The studio currently stops at "download the controller and flash it by hand". The feature closes the loop:

```text
   draw the mission
          |
          v
   solve  (validated)  ----------------------------.
          |                                        |
          v                                        |
   [ Push to hub ]  ->  the robot drives          |
          |                                        |
          v                                        |
   session: POSE / EVENT lines                     |
          |                                        |
          v                                        |
   measured pose on the canvas, next to the plan   |
          |                                        |
          v                                        |
   compare (overboard.sysid.compare) / fit (overboard.sysid.fit) ---'
```

The phases in [Section 6](#6-the-phases) build it in that order.

---

## 2. What already exists

Everything below is in the tree today. The feature is mostly a matter of wiring these together in the right order and showing the results.

| Piece | Where | Notes |
|---|---|---|
| Write the packed mission | `overboard/deploy/export.py::write_compact_python_file(traj, path)` | Writes `samples`, `config` and `events` as a MicroPython module. |
| Push a program and capture its output | `overboard/deploy/hub_deploy.py::deploy(script, connection, name, wait, print_output, stream_path)` | `stream_path` writes the console output **as it arrives**, so a dropped link costs the last test. |
| The hub entry point | `pybricks_code/hub_runner.py` | Imports the trajectory from `hub_mission.py` and the hardware from `robot_config.py`; runs `ramsete_compact_events.follow_trajectory`; prints `POSE,x_m,y_m,heading_deg` and `EVENT,name`. |
| The command-line push | `overboard/main.py --push-hub` → `_push_to_hub()` | Writes `pybricks_code/hub_mission.py`, deploys `hub_runner.py`, optionally saves the session with `--hub-log`. |
| Session → measured poses | `overboard/sysid/fit.py::poses_from_hub_session(lines)` → rows + event names | The rows are the `run_id,stop,x_m,y_m,heading_deg` CSV the fitter and the comparator read. |
| Plan vs. measured | `overboard/sysid/compare.py::compare(traj_path, poses_path, ...)` → a report dict | The same rows and the "inside the envelope?" check the command line prints. |
| The studio's job model | `overboard/gui_server.py` (`Job`, `start_job`, `build_command`) | A subprocess, a log tail, artifacts, a cancel, and one job at a time. |

## 3. The gap

Three things are missing.

1. **There is no way to push a trajectory that already exists.** `overboard/main.py --push-hub` re-solves first. The studio has the approved trajectory in the job directory, and re-solving is both slow and not necessarily the same trajectory.
2. **Nothing shows the robot's result against the plan.** The `POSE` line is parsed by the sysid tools, but a person driving the robot cannot see "the plan ended here, the robot ended 4 mm to the left" anywhere.
3. **Nothing checks the plan against the robot's own files** before the robot drives: the event names the robot has actions for, and the wheel size the robot actually has.

---

## 4. The pieces

### 4.1 Pushing an existing trajectory

Move the push out of the CLI so the command line and the studio share one path.

In `overboard/deploy/hub_deploy.py` (`parents[2]` because the module is two levels below the repo
root):

```python
HUB_CODE_DIR = Path(__file__).resolve().parents[2] / "pybricks_code"
HUB_MISSION_MODULE = HUB_CODE_DIR / "hub_mission.py"
HUB_RUNNER = HUB_CODE_DIR / "hub_runner.py"


def push_trajectory(traj_file, connection="ble", name=None, log_path=None):
    """Write the packed mission and push it to the hub.

    Returns the console lines the hub printed. The packed mission is written
    to pybricks_code/hub_mission.py, which is where hub_runner.py imports it
    from; the push carries both files.
    """
    from overboard.deploy.export import write_compact_python_file

    write_compact_python_file(traj_file, str(HUB_MISSION_MODULE))
    lines = deploy(str(HUB_RUNNER), connection=connection, name=name,
                   stream_path=log_path)
    return lines
```

`overboard/deploy/hub_deploy.py` also has a command-line mode so the studio can run it as a
subprocess (which is what makes cancel work):

```bash
python -m overboard.deploy.hub_deploy --trajectory output.traj --conn ble --name HUB --out session.csv
```

`overboard/main.py --push-hub` calls `push_trajectory` instead of doing the two steps
itself, so there is one push path in the tree.

### 4.2 The push as a studio job

The push is a second kind of studio job: it needs an existing trajectory, and it
talks to hardware.

```http
POST /api/push
{ "job_id": "job3", "connection": "ble", "hub_name": "HUB" }
```

The server:

1. Looks up the solve job and its `.traj`. **If the file is not there, the export
   gate did not pass** — the push is unreachable, which is the point.
2. Runs `python -m overboard.deploy.hub_deploy --trajectory <traj> --conn <conn> --name <name>
   --out <job_dir>/session.csv` as a subprocess.
3. Puts the subprocess output in the job log, so the hub's console appears in
   the studio while the robot drives.
4. Copies the session to `missions/<mission>.session.csv` (D2).

The job status is the same shape as a solve job, so the studio's progress, cancel
and log code does not change.

### 4.3 The studio's UI

A **Run** section in the result panel:

- **Connection**: `BLE` / `USB` dropdown, and a hub name field for BLE. Both are
  remembered in the browser session.
- **Push to hub**: enabled only when the last solve passed and its trajectory is
  on disk. It is a separate button, not a Solve option (D1).
- **While pushing**: the header says *"Pushing over BLE — the robot is driving"*,
  not "Solving". The hub's console output streams into the log panel.
- **After the push**: the measured pose, the error, the events, and the session.

The measured pose is drawn on the canvas as a distinct marker with a dashed line
from the planned end pose to the measured one. The numbers come from
`sysid_compare.compare()`, so the studio shows exactly what the command line
would print, in the same units (mm and degrees).

---

## 5. The integrations

### 5.1 Missions

A mission in the studio is a waypoint list. A mission on disk is
`missions/<name>.json`. The push adds a second file next to it:

```text
missions/
  my_mission.json          # the plan
  my_mission.session.csv   # what the hub printed when it ran
```

The studio's mission list shows whether a session exists, so a mission that has
been driven is distinguishable from one that has only been planned. The compare
and fit actions read the pair.

### 5.2 Robot code

Two things can be checked before the robot drives, both by reading files rather
than importing them (`pybricks_code/` is firmware: it imports `pybricks.*` and
cannot be imported on the host).

**Events.** `pybricks_code/hub_runner.py` holds `MISSION_EVENTS`, the map from an
exported event name to the action the robot performs. The studio parses the keys
and shows the mission's events against it:

```text
lower_arm   has an action
release     has an action
wave        no action on the robot - it will be reported only
```

`hub_runner.py` already falls back to reporting an unknown event instead of
failing, so this is a warning, not an error. The point is to see it before the
drive rather than in the log afterwards.

**Geometry.** `pybricks_code/robot_config.py` holds the hardware:
`WHEEL_DIAMETER_MM` and `AXLE_TRACK_MM` (56 mm and 96.5 mm in the current
tree). The packed mission carries the **planned** config to the hub
(`hub_mission.py`), while the drivebase comes from `robot_config.py`. If the two
disagree, the controller is tuned for a wheel the robot does not have.

The studio compares them and, on a mismatch, **warns** (D4) with a one-click
"use the robot's geometry" that sets the plan's wheel radius and track width.

### 5.3 Sysid

The fit is the part of the loop that turns "the robot did not end where the plan
said" into a better plan.

- **Compare**: `overboard/sysid/compare.py::compare(traj, poses.csv)` on the plan and the
  session. The studio shows the same table the command line prints: the measured
  and predicted pose per stop, the position and heading error, and whether the
  measured pose is inside the reachability envelope.
- **Fit**: after a push, "fit this session" runs `overboard/sysid/fit.py` over the session
  and writes `sysid_fit.json`. The next solve passes `--sysid-fit`, so the
  validator plans against the measured model instead of the nominal one.
- **Honesty**: the poses from a pushed run are the hub's own odometry, so the
  distance scale, the heading scale and the brake slip of the fit carry no
  information and stay at their defaults. Only the actuator part (gain, time
  constant, command delay, backlash) is calibrated. The studio must say this, the
  way `overboard/sysid/collect.py` does, or it looks like calibration when it is the robot
  agreeing with itself.

---

## 6. The phases

| Phase | Content | Size | State |
|---|---|---|---|
| **1. The push** | `hub_deploy.push_trajectory` + the `--trajectory` mode; `overboard/main.py --push-hub` uses it; the studio's push job, button and streamed console output. | M | **Done** |
| **2. The result** | Parse the `POSE`/`EVENT` lines; the measured marker on the canvas; the error; the event list; the session written next to the mission. | M | **Done** |
| **3. The robot code** | The event checklist; the geometry warning with its one-click fix. | S–M | **Done** |
| **4. The loop** | The compare report in the studio; the measured model; the calibration wizard. | M | **Done** (designed in [CALIBRATION_STUDIO_PLAN.md](CALIBRATION_STUDIO_PLAN.md)) |

What is built today:

- `hub_deploy.push_trajectory(traj_file, connection, name, log_path)` writes the packed
  mission and pushes the runner. `python -m overboard.deploy.hub_deploy --trajectory out.traj --conn ble
  --out session.csv` does the same from the command line, and `overboard/main.py --push-hub` calls
  the same function, so there is one push path in the tree.
- The studio's **Push to hub** button (`POST /api/push`) runs that command as a
  subprocess against the trajectory of the last **passed** solve, so the export gate
  still holds: a solve that failed has no trajectory to push. The hub's console
  streams into the log panel while the robot drives, and the session is written to
  `missions/<name>.session.csv`.
- The measured pose is drawn on the canvas next to the plan, with the position error,
  the heading error and whether it landed inside the reachability envelope. The robot's
  odometry starts where the robot physically is, which is the plan's first waypoint, so
  the measured pose is rotated by the plan's first heading and moved by its first
  position before it is drawn.
- The robot's own files are read as text (they are firmware and cannot be imported on
  the host): `MISSION_EVENTS` gives the event checklist, and `WHEEL_DIAMETER_MM` /
  `AXLE_TRACK_MM` give the geometry warning with its one-click fix.

Phases 1 and 2 are the feature. Phase 3 is what makes it safe to press the
button. Phase 4 is what makes it a loop instead of a one-off: the Run tab's comparison
against the plan, the Robot tab's measured model, and the calibration wizard
(`POST /api/calibration/collect`, `/api/calibration/sheet`, `/api/calibration/fit`).

---

## 7. Gotchas

- **BLE allows one central.** The Pybricks IDE has to be closed. `hub_deploy`
  already says so in its error message; the studio shows it as the failure.
- **The push blocks for the length of the run.** The studio must not look
  frozen: say that the robot is driving, and stream the hub's console.
- **The measured pose is odometry, not ground truth.** Label it as such
  everywhere it appears.
- **Heading convention.** The `POSE` heading comes from `drivebase.odometry()`,
  which should be counter-clockwise positive like the plan, while the hub's
  `angle()` is clockwise positive. Confirm on the first real push before the
  overlay relies on the heading error.
- **`pybricks_code/hub_mission.py` is shared state.** The push overwrites it.
  The studio should say so, and two pushes in a row are fine.
- **Never push an unvalidated trajectory.** The push is only reachable from a
  job whose `.traj` exists, so the export gate holds without a second check.

---

## 8. Deferred

- **The button-press SysId loop** (`overboard/sysid/collect.py`): the operator presses the
  hub button between tests. It is a different interaction from watching a
  trajectory drive, and it belongs in its own phase (D3).
- **A live pose stream.** The hub prints one pose at the end; a robot that
  streamed its pose as it drove would let the studio draw the actual path, not
  just the end point. That needs a firmware change.
- **Multi-hub fleets.** One hub, one name field.

## 9. Still open

- **Which mission name does the session take?** `missions/<name>.session.csv`
  assumes the pushed job knows the mission name; the studio has it, but a push
  from the command line does not. Probably the trajectory's base name.
- **Should the studio write the session, or the CLI?** The CLI's `--hub-log`
  writes it wherever it is told. The studio needs it next to the mission, so the
  studio writes it; the CLI keeps `--hub-log` as is.
- **Does the fit run in the studio or on the command line?** The studio can call
  `sysid_fit` directly, but the fit also needs the drivebase profile (the wheel
  radius for the actuator fit). The studio's robot config has it, so the studio
  can supply it — that needs a decision.
