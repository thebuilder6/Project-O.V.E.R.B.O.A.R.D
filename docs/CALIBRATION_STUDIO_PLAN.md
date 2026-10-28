# The SysId Loop in the Studio

**Status:** 2026-10-20. All four phases are implemented. The sysid process itself is settled (see [docs/CALIBRATION_PROCEDURE.md](CALIBRATION_PROCEDURE.md)); this document is about putting it in the browser, together with the studio's phase 4 (compare and fit). This is the design KI-059 asks the guided page to follow.
**Scope:** how the calibration, the fit and the mission loop fit together in one place, and what the GUI adds to a procedure that already works from the command line.
**Audience:** anyone implementing it, and anyone changing the calibration procedure afterwards.

> **Read this first.** The calibration procedure is designed and tested on paper: collect, measure, fit, plan with the fit, compare. Nothing here changes the measurements or the mathematics. This is about the same steps, driven from a browser, with the sheet as a form instead of a piece of paper.

---

## Table of contents

- [The two loops](#the-two-loops)
- [What the session already tells the GUI](#what-the-session-already-tells-the-gui)
- [Decisions](#decisions)
- [The pieces](#the-pieces)
  - [A. The measured model belongs to the robot](#a-the-measured-model-belongs-to-the-robot)
  - [B. The comparison table](#b-the-comparison-table)
  - [C. The calibration wizard](#c-the-calibration-wizard)
  - [D. The sheet as a form](#d-the-sheet-as-a-form)
- [Where the files live](#where-the-files-live)
- [The phases](#the-phases)
- [Gotchas](#gotchas)
- [Still open](#still-open)

---

## The two loops

There are two loops, and they meet at one artifact: the **measured model**.

```text
   the robot, measured once                 every mission
   ─────────────────────────                ──────────────
   collect  ──▶  measure  ──▶  fit  ──┐
     ▲                     │          │
     │                     │          ▼
   the hub                └────▶  plan  ──▶  validate  ──▶  push  ──▶  compare
                                                                  │
   the tape measure                                                 │
     (ground truth)  ──────────────────────────────────────────────┘
```

- **The calibration loop** is rare and deliberate: it measures the robot, and it produces `sysid_fit.json`.
- **The mission loop** is what the studio already does, and it becomes better when it runs against the measured model.

The unification in the GUI is therefore one sentence: **the robot panel owns the measured model, and the mission panel uses it.**

There is a second unification, and it is the one that is easy to get wrong:

| Session | Where its poses come from | What it can be used for |
|---|---|---|
| A **pushed trajectory** (`hub_runner.py`) | the robot's **own odometry** | **Comparing** the plan with what the robot did. Not fitting: the robot is only agreeing with itself. |
| A **calibration session** (`sysid_digital_twin.py`) | the **tape measure** | **Fitting** the model, and therefore calibrating the odometry scales. |

So the studio must not offer "fit this session" on a pushed trajectory. It should say why: *"The pushed run's poses are the robot's own odometry. They can be compared with the plan, but they cannot calibrate it - that needs the tape measure."*

---

## What the session already tells the GUI

The collector's session is self-describing, and the GUI needs no new format to use it:

```text
TEST_START odometry_straight
META,run_id,11
META,test,odometry_straight
META,repeat,0
META,notes,...
META,poses,1
META,wheel_diameter_mm,56
META,axle_track_mm,96.5
META,battery_mv,...
DATA
t_ms,...
...
TEST_END,...
```

- `META,test` and `META,repeat` key the measurement sheet.
- `META,poses` says **how many measurements that test needs** - the checklist, live.
- `META,wheel_diameter_mm` and `META,axle_track_mm` are the robot's own geometry, measured by the robot itself. **The fit can take the wheel radius from the session**, which answers the open question in [docs/STUDIO_HUB_PLAN.md](STUDIO_HUB_PLAN.md) §9: no drivebase profile file is needed in the GUI flow.
- `META,run_id` ties a pose to the test that was open when it was printed.

`overboard/sysid/fit.py` already parses all of this (`parse_session`, `session_tests`, `write_sheet`, `sheet_to_poses`, `fit_session`). The GUI reuses those functions rather than re-parsing anything.

---

## Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **The measured model belongs to the robot, not the mission.** It lives in the robot panel and applies to every solve with that robot. | The model describes the robot. A mission that borrowed a fit would be a bug waiting to happen. |
| D2 | **The sheet is stored on the server, next to the calibration session**, and the browser keeps only the editing state. | The sheet is data, not UI state. It has to survive a reload, be downloadable, and be usable in a spreadsheet. |
| D3 | **A pushed trajectory can be compared but not fitted.** The studio says so instead of offering a meaningless fit. | The pushed poses are the robot's own odometry. Fitting them would calibrate the robot against itself. |
| D4 | **The calibration step runs the collector through the same push machinery the studio already has.** | One way to talk to a hub, one place where the "no hub found" message lives. |

---

## The pieces

### A. The measured model belongs to the robot

The robot panel gains a **Measured model** slot:

- the fit file in use (`calibration/<robot>.sysid_fit.json`),
- its summary: the straight and turn gain with their 95% intervals, tau, delay, the gyro bias, the distance and heading scales, the brake slip,
- a **use it for planning** toggle, which adds `--sysid-fit` to every solve with that robot.

The solve then runs against the measured model, and the result panel says so: *"planned against the measured model (gain 0.974-0.988), not the nominal one"*. `overboard/sysid/digital_twin.py` already uses the fit's gain interval as the band it plans against, so the conservative behaviour comes for free.

This is the smallest piece of the whole plan, and it is the one that pays off on every mission after one calibration.

### B. The comparison table

The studio already draws the measured pose next to the plan. The **table** is `overboard/sysid/compare.py::compare(traj, poses_csv)`: one row per stop with the measured and predicted pose, the position and heading error, and whether the measured pose is inside the reachability envelope.

The server calls it and the studio renders the rows under the measured marker. The same renderer serves the pushed trajectory (one row: the final pose) and a calibration session (a row per measured stop).

### C. The calibration wizard

A **Calibrate** block in the robot panel, with three steps:

1. **Collect.** Pushes `pybricks_code/sysid_digital_twin.py` with the studio's existing machinery (D4) and captures the session. While it runs, the studio tails the session and shows the live checklist:

   ```text
   test 12 of 19 - odometry_turn_cw, repeat 2 of 3
   waiting for the centre button on the hub
   14 of 21 measurements still needed
   ```

   The operator still presses the hub's button and still holds the tape measure. The studio cannot do either; it can only say what is happening and what is still missing.

2. **Measure.** The sheet as a form (section D).

3. **Fit.** The studio turns the sheet into the measured poses, runs the fit, and shows the summary with the intervals. The fit file is written next to the session.

The wheel radius for the fit comes from the session's own `META,wheel_diameter_mm`; the studio shows a warning if the robot's `robot_config.py` and the session disagree, because then one of them is stale.

### D. The sheet as a form

The rows are exactly the rows `sysid_fit.write_sheet` writes, and the columns are exactly its `SHEET_FIELDS`:

```text
test, repeat, stop, x_mm, y_mm, turn_right_deg
```

The studio shows them as a table: the first three columns are read-only keys, the last three are number inputs. The operator types what the tape says. The studio shows the count of what is still missing, and the fit button stays disabled until the required measurements are in (with an override, because a partial sheet is still worth fitting - the CLI allows it and says which scales are uncalibrated).

The numbers go straight back into a sheet CSV, so the spreadsheet path and the GUI path produce the same file.

The command line has the same form as a standalone file: `python -m overboard.sysid.fit session.csv --write-form calibration_sheet.html` writes the same rows as a self-contained page, and the page downloads the same sheet CSV. An operator who never opens the studio gets the same sheet, and the fit takes it either way (`--sheet`).

---

## Where the files live

```text
calibration/                     # the robot, measured once
  <robot>.session.csv            # what the hub printed
  <robot>.sheet.csv              # the measurements, as a CSV
  <robot>.poses.csv              # the measured poses the fitter reads
  <robot>.sysid_fit.json         # the fit

missions/                        # a mission and what the robot did with it
  <mission>.json                 # the plan
  <mission>.session.csv          # the pushed run
```

The studio's browser session remembers which calibration and which fit are in use, so the two folders never have to be navigated by hand.

---

## The phases

| Phase | Content | Size | State |
|---|---|---|---|
| **1. The measured model** | The robot panel's fit slot; the summary; `--sysid-fit` on the solves; the result panel's wording. | S-M | **Done** |
| **2. The comparison table** | `overboard/sysid/compare.py` in the server; the rows in the studio. | S | **Done** |
| **3. The collect step** | The collector through the push machinery; the live checklist. | M | **Done** |
| **4. The sheet as a form** | The rows, the inputs, the missing count, the sheet file, the fit. | M | **Done** |

Phase 1 alone is worth doing: it makes every mission after a calibration better, and it needs nothing from the hub.

### What phase 1 found

Loading a fitted model and planning against it works end to end: the studio keeps the fit with the robot
(`calibration/<robot>.sysid_fit.json`), passes `--sysid-fit` with every solve, turns validation on (the
fitted plant is only used by the validator), and shows the model's numbers next to the robot.

**The gain band is wider than the tracking gate.** `overboard/sysid/digital_twin.py` uses the fit's gain interval as the band
the validator plans against, so a robot whose straight gain varies by 1% is planned against as if it could be
1% slow. Over a 1 m path that is 10 mm of position error, and the tracking gate is 5 mm. A measured
straight example therefore **fails** validation with a 5.4 cm tracking error, and nothing is exported.

This is not a bug in the studio; the same thing happens on the command line:

```text
python -m overboard.main -w examples/example_straight.json --validate --sysid-fit fit.json
Digital twin: gain 0.971-0.991, tau 0.081s, delay 0.021s, odometry 0.9980/1.0000
  FAILED: envelope radius 5.59 cm exceeds the local threshold; tracking error 5.41 cm > 0.50 cm.
```

Two things follow. The studio says so before the solve fails - the model panel shows *"The gain band is 1.0%.
Over this 1.00 m path that is about 10 mm of position error, and the tracking gate is 5 mm."* - and the
question of what the gate should measure is already open in
[docs/research/RQ-A_tracking_error.md](research/RQ-A_tracking_error.md).

---

## Gotchas

- **The hub waits for the centre button.** The studio cannot press it. The wizard has to say so, or the operator waits in front of a browser that looks stuck.
- **The Bluetooth button is the emergency stop.** Nothing in the studio may look like a control for the robot. The push and the collect are the only things that move it.
- **A dropped link costs the last test, not the run.** The session is written as it arrives, so the studio can show a partial session and let the operator re-run the missing tests.
- **The fit needs the wheel radius; the session has it.** Prefer the session's `META,wheel_diameter_mm` over a profile file, and warn when the robot's own file disagrees with the session.
- **Three repeats, twenty-one measurements.** The wizard has to make that count visible, because it is the part of the procedure people get wrong.
- **The tape error is systematic.** The intervals cover the run-to-run spread, not the tape. The studio's wording should keep that distinction, the same way the fitter's output does.
- **The pushed session is not a calibration session.** See D3.

---

## Still open

- **Should the studio run the collect step at all, or only read a session the operator collected with the CLI?** Resolved: the studio runs it. The `/sysid` page collects with a pickable test list (subset or full suite), one-by-one step mode, and setup-free tests chaining by themselves; the CLI collect stays for scripting. An "open a session collected elsewhere" action was never built — nobody has asked for it since the page reads any session file dropped into `calibration/`.
- **Where does the fit's summary come from - the fit JSON or the fitter's output?** Resolved: `summarize_fit()` in `overboard/gui_server.py` is the single reader, covering every twin section with its interval, so the before/after table is the same numbers row for row. The page renders its own wording around them.
- **Does the studio write the fit next to the robot or next to the mission?** Resolved: `calibration/<robot>.*`, because the model belongs to the robot (D1).
- **Should a mission record which fit it was planned against?** A mission planned with a fit from March and one planned with no fit are different plans. Writing the fit's summary into the trajectory's stats would record it.
