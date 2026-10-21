# Studio UI Inventory

**Status:** 2026-10-21, after the 3-Pillar alignment of the Solve and Result tabs. A complete list of every control, readout and generated element in the browser studio, so the next change starts from a list rather than from the page.
**Scope:** `overboard/viz/studio.html`, `overboard/viz/studio.js`, `overboard/viz/studio.css`
(served by `overboard/gui_server.py`). The studio is the only browser UI (the WebSocket live view was
removed).

---

## The shape of it

```text
+--------------------------------------------------------------------------+
| OVERBOARD Studio        status line                     [Solve] [Cancel]  |
+-------------------+---------------------------------+---------------------+
| MISSION ROBOT     |  +-------------+                  | RESULT WAYPOINT     |
| FIELD   SOLVE     |  | Legend      |  [Frame all]   | PLOTS   RUN         |
|                   |  +-------------+                  |                     |
|  ...tab content...|                                 |  ...tab content...  |
|                   |        the field                |                     |
|                   |  [Play |-----| 0.00s / 0.00s]  |                     |
+-------------------+---------------------------------+---------------------+
```

**Three columns, two of them tabbed.** The left column is what you set, the canvas is the work, the right column is what you read. Each side column is a strip of tabs over one panel, so a column holds one topic at a time.

The page carries **156 element ids**: **81 controls with an id** (36 buttons, 38 inputs, 7 selects), of which 8 are the tab buttons themselves. Twelve regions are filled in at run time:

| Generated region | Element id | Where |
|---|---|---|
| The robot fields (10) | `robot-fields` | Robot tab, Edit robot |
| The uncertainty ranges (6) | `robot-range-fields` | Robot tab, Edit robot, the nested ranges |
| The measurement sheet rows | `sheet-table` | Robot tab, Calibrate, step 2 |
| The plot tabs | `plot-tabs` | Plots tab |
| The field's legend | `legend-items` | The canvas, top left |
| The candidate chips | `validation-candidates` | Result tab, the validation detail |
| The per-segment table | `validation-table` | Result tab, the validation detail |
| The pick button | `pick-row` | Result tab, the validation detail |
| The fix buttons | `fix-row` | Result tab, the validation detail |
| The download buttons | `downloads` | Result tab |
| The history rows | `history-list` | Mission tab, the History list |
| The waypoint list | `wp-list` | Waypoint tab |

---

## The structure, and why it is this way

**The left column has four tabs** - Mission, Robot, Field, Solve - because the old column stacked five topics, and the robot block alone held four unrelated jobs (the facts, the measured model, the calibration wizard and the editor). A tab is one topic.

**The right column has four tabs** - Result, Waypoint, Plots, Run - because the result used to sit third, below the waypoint editor, so reading a result meant scrolling.

**The header always carries the result.** The status line reads `3.70 s | 1.23 m - validated` after a solve, so the answer is on screen whatever the columns are showing. A tabbed layout can hide things; the header cannot.

**The canvas carries a legend**, built from the same list the drawing uses, so the field's colours have a key that cannot go stale. The key is *live*: an entry for something that is not on the field is hidden, so the list is a key to the picture rather than a list of possibilities.

### The rules the structure follows

- One topic per tab. A tab is a *subject*, not a *section*.
- The result is never hidden: the header carries the numbers, and a solve or a failure brings the Result tab forward by itself.
- The action lives with its subject. The file actions are in the Mission tab, the hub actions in the Run tab, and Solve alone in the header.
- Rare things are one click away, not on the page: the nine `<details>` elements (the mission's files, the history, the measured model, the calibration wizard, the robot editor, the uncertainty ranges, the solve options, the solver log) hold what is used once in a while.
- Every tab is wired by the same controller, so a new tab is a button and a panel and nothing else.

### The tabs and their panels

| Column | Tab | Panel id | Holds |
|---|---|---|---|
| Left | Mission | `panel-mission` | The mission, its actions and its history. |
| Left | Robot | `panel-robot` | The config, the measured model, the calibration and the editor. |
| Left | Field | `panel-field` | The season, the mat and what the field draws. |
| Left | Solve | `panel-solve` | The preset and the solve options. |
| Right | Result | `panel-result` | The status, the numbers, the validation and the downloads. |
| Right | Waypoint | `panel-waypoint` | The selected waypoint's fields and the waypoint list. |
| Right | Plots | `panel-plots` | The plot tabs, the canvas and the note. |
| Right | Run | `panel-run` | The hub, the push and the comparison. |

---

## Header

| Control | What it does | Availability |
|---|---|---|
| **status line** | The state, and after a solve the result: `3.70 s \| 1.23 m - validated`, or `Failed`. | Always |
| **Solve** | Runs the command line with the current options. | Always; disabled while a job runs |
| **Cancel** | Stops the running solve, push or collect. | Only while a job runs |

## Left column - the tabs

### Mission tab

| Control | What it does |
|---|---|
| **Name** (text) | The mission's name; the file is `missions/<slug>.json`. |
| **Load** (select) | The saved and example missions. Picking one replaces the mission. |
| **New** | Empties the mission to one waypoint at the origin. |
| **Undo** / **Redo** | 100 steps. Covers the mission *and* the robot config. |
| **Save** | Writes `missions/<name>.json` and commits it to the missions' git repository. |
| **Files** (details) | The file actions, which are used once in a while. |
| **Download the mission** | Saves the mission JSON to the machine. |
| **Open a mission file...** | Loads a mission JSON from disk. |
| **Download the project** | Saves one file with the mission, the robot, the options, the model and the field. |
| **Open a project file...** | Loads such a project and brings all of it back. |
| *waypoint count* | Readout: "10 waypoints". |
| *path summary* | Readout: "Path 1.36 m \| at least 2.85 s at 0.48 m/s top speed". |
| *mission state* | Readout: "Not saved to missions/ yet" / "Saved as missions/x.json" / "Changed since ...". |
| **History** (details) | The git commits for this mission, each with a **Restore** button. |
| *hint* | "Ctrl+Z undo · Ctrl+S save · Space play · the arrow keys move the selected waypoint". |

### Robot tab

| Control | What it does |
|---|---|
| **Configuration** (select) | The example and saved robot configs. |
| **Open config...** | Loads a robot config from disk. |
| *facts* | Readout: mass, wheel radius, track width, friction. |
| *warning* | "This is the example robot config. Its uncertainty ranges are test values...". |
| *geometry warning* | "the plan assumes a 50.0 mm wheel, the robot has 56.0 mm". |
| **Use the robot's geometry** | Copies the robot's wheel and track into the plan. |
| *event warning* | "The robot has no action for lower_arm. The hub will report the event and carry on." |

#### Measured model (details)

| Control | What it does |
|---|---|
| *status* | "Loaded from calibration/robot.sysid_fit.json" / "No measured model: the plan uses the nominal one." |
| **Choose a file...** | Loads a `sysid_fit.json` and keeps it with the robot. |
| **path** + **Load** | Loads a fit from a path. |
| *list* + **Use** | The measured models the studio knows about. |
| *facts* | Straight and turn gain with their 95% intervals, tau, delay, gyro bias, distance and heading scale, brake slip. |
| *hint* | "The gain band is 1.0% ... the reachable set widens with it and the envelope limit is 20 mm." |
| **Plan against the measured model** (check) | Adds `--sysid-fit` to every solve, and turns validation on. |

#### Calibrate this robot (details)

| Control | What it does |
|---|---|
| **Robot name** (text) | The calibration's name; the files are `calibration/<name>.*`. |
| **1. Collect** + status | Pushes the collector to the hub and captures the session. The status names the running test, the repeat and how many measurements are needed. The button is **Collect on the hub**. |
| **2. Measure** + table + **Save the sheet** | The measurements the session needs, as a form (test, repeat, stop, x, y, turn). |
| **3. Fit** + status | The session and the sheet become the measured model. The status carries the fitter's warnings. The button is **Fit the model**. |

#### Edit robot (details)

| Control | What it does |
|---|---|
| *robot fields* (generated) | Mass, inertia, wheel radius, track width, motor speed, stall torque, gearing, friction, and the two headrooms. |
| *uncertainty ranges* (generated) | The six ranges the validator plans for. |
| **Download config** | Saves the robot config. |

### Field tab

| Control | What it does |
|---|---|
| **Season** (select) | Picks the field's mission models (`fields/<season>.json`). |
| **Mat image** (file) | The operator's own picture of the mat. |
| **Download the mat** | Composes the mat picture from FIRST's own field guide, fetched from the URL the field file records and kept in `mats/` (untracked; the artwork is not ours to ship). |
| **Width** / **Height** (mm) | The mat's size, so the picture and the footprints share one frame. |
| **Show the mat** (check) | Draws the mat under the field. |
| **Earlier runs** (check) | Draws the earlier solves as faint paths. |
| **Actuator saturation** (check) | Colours the trajectory by how much of the actuator diamond it asks for: blue below 0.85, amber from 0.85 to 1.0, red above 1.0. |
| **Late-apex cues** (check) | Draws the late-apex diamond on a turning segment and the momentum bisector at a free-heading waypoint. |
| *mat name* / *credit* / *run history* | Readouts. |

### Solve tab

| Control | What it does |
|---|---|
| **fast / balanced / safe** (radio cards) | A preset fills in the options the user did not set. fast: one candidate, no validation, plant propagation - seconds. balanced: the defaults (a candidate count only compares when validation is on) - seconds. safe: the comparison - three candidates, validation, the closed-loop tick map; the three frontier paths are drawn together and the operator may export the one they pick. safe costs minutes: the reachability check, not the solve, is the time. |
| **Options** (details) | The rest of the solve options, below. |
| **Samples per segment** | 2 to 200. Disabled while the automatic mesh is on, because the number is then not used. |
| **Auto invariant mesh (80 ms)** | Ticked by default: `-n` is left off the command, so the 3-Pillar engine sizes the invariant 80 ms mesh from the path. Unticking it sends the number in **Samples per segment**. |
| **Accuracy weight** | The smoothness weight (`--accuracy-weight`). |
| **Candidates (1-3)** | How many of the planner's three frontier candidates are solved (sprint, sweeper, exploratory); clamped to 1-3 by the server. Disabled unless validation is on. |
| **Reachability propagation** | The model the envelope gate uses (`--propagation`): the open-loop plant (fast) or the closed-loop tick map (feedback-contracted, 2-10x slower). |
| **Soft force weight** | A weight above zero prices wheel-force utilisation (`--force-weight`); zero is the default and sends nothing. |
| **Export format** | None / controller JSON / Python / Python compact. |
| **Controller timestep** | The export's sample interval. |
| **Validate before export** | The gate; needs JAX. |
| **Accuracy limit (cm)** | How far the reachable set may spread around the plan at any sample (`--envelope-limit`, default 2 cm, 0.1-5 cm). A waypoint's error threshold still wins. Disabled unless validation is on, and sent only with `--validate`. The limit is absolute, so a long path can fail for being long; declaring the mission's real tolerance is what it is for (KI-054). |
| **Keep the best path when validation fails** | Ticked by default: adds `--best-effort` next to `--validate`, so a run whose only failure is an accuracy excess (envelope or tracking) exports the best candidate and draws it on the field, marked failed. A force, slip, integration or convergence failure still blocks, and the push stays gated on the verdict. |
| **Fast polish** | The budgeted solve. |
| **Gradient sensitivity polish** | The post-solve gradient polish (`--polish`); needs JAX. |

## Field canvas

| Control | What it does |
|---|---|
| **Legend** (details) | The key to the field's colours, built from the list the drawing uses. Live: an entry for something that is not on the field is hidden. |
| **Frame all** | Fits the view to the mission. |
| **Envelope** (check) | The reachable set around the plan, drawn *over* the trajectory (it is only a few millimetres wide on the tick map, so under it it vanished). There is none on a run that was not validated; the key hides it then. |
| **Path** (check) | The straight-line path between the waypoints. |
| **Play** + slider + time | The playback of the solved trajectory. |
| *hint* | "Click the field to add a waypoint. Drag a waypoint to move it." |

Canvas contents: the mat, the grid, the mission models, the earlier runs, the path, the trajectory (coloured by actuator saturation when that is on), the other candidates' paths from a comparison (one hue per seed, the clicked one stronger), the envelope (whose key names the propagation that produced it), the failing segments (red), the worst-drift segment (amber), the solver's first guess (dashed), the late-apex diamonds and the momentum bisectors, the waypoints with their stop and event badges, the selected waypoint, the measured pose from the hub, and the collisions.

## Right column - the tabs

### Result tab

| Element | What it does |
|---|---|
| *status* | "Validated - the trajectory passed the validator" / "Not validated - this trajectory has not been checked against your robot" / "Failed validation - the best path is shown for inspection ... it is not approved for the hub" (a best-effort export) / the failure. |
| **Check it, then solve again** | The lever the status names: turns validation on, brings the Solve tab forward and solves again. Shown only after a solve that was not validated. |
| *facts* | Duration, path, peak wheel speed, samples, max envelope, solve (wall), the time against the top-speed ideal, the field. |
| *solve diff* | "vs. the previous solve: 0.27 s slower, the path 48 mm longer." |
| *collisions* | Where the plan met a mission model. |
| **Validation detail** (details) | The explainer, the candidate chips, the pick button, the per-segment table, the note, and the **Try** buttons. A chip names its seed by its physical role - Sprint, Sweeper, Exploratory - and a rescue seed is a dashed amber chip (🛟 Rescue). Every solved candidate's path is on the field (`--candidates-out`), the clicked chip's path drawn stronger, so the comparison is visible and not only tabulated. A path appears the moment its solve finishes, before its check and before the rescue round, so a validated run is not a blank field for its minutes; the chips and the table arrive with the report, when the run ends. The pick button (`Solve again and export: ...`) appears when the clicked candidate passed but is not the exported one: it re-solves with that candidate pinned (`--pick-candidate`), and the CLI refuses it if it no longer passes. The table carries, per segment, the envelope, the limit, the drift, and the three discretization mechanisms in mm (sagitta, heading mismatch, 20 ms hold lag). The **Try** buttons are the levers the numbers point at: the closed-loop tick map for an envelope failure, the gradient polish for a tracking error just over the gate, a soft force weight of 0.005 for a force or slip failure, and a weight of 0.001 for a rough profile. Each pulls its lever and solves again. |
| **Mission report** | The report of the last solve, as a file. |
| *downloads* | The trajectory, the controller, the Python module, the stats, the session, and the report. |
| **Mission report** | Writes the one-page report next to the trajectory and opens it: the mission, the robot and the result on one printable page. |
| **Solver log** (details) | The command line's output. |

### Waypoint tab

| Control | What it does |
|---|---|
| **X** / **Y** (m) | The position. |
| **Heading** (deg) + **Heading free** | The heading, or unconstrained. |
| **Stop here** (check) | The robot rests here. |
| **Event name** (text) | The event the robot fires here. |
| **Precision** (check) | A 0.5 cm envelope limit. |
| **Envelope limit** (cm) | A custom limit. |
| **Move up** / **Move down** / **Delete** | The order and the removal. |
| *waypoint list* | The ordered waypoints, each clickable, tagged with its stop and event. |

The tab carries a badge with the selected waypoint's number, and clicking a waypoint on the field brings this tab forward.

### Plots tab

The plot tabs - Speed, Force, Acceleration, Heading, Convergence - are built from the plot registry, over one canvas, with a note under it. Adding a plot is one entry in the registry.

### Run tab

| Control | What it does |
|---|---|
| **Connection** (select) | Bluetooth or USB. |
| **Hub name** (text) | The hub's name or address (BLE only). |
| **Push to hub** | Pushes the approved trajectory and runs it. Enabled only after a solve that **passed the validator**: a trajectory that was never checked is not an approved one, and the status says so. |
| *status* | "Pushing. The robot is driving." / "This trajectory has not been validated against your robot. Turn validation on in the Solve tab and solve again before pushing it." / the hub's error. |
| *facts* | Where the robot ended, the error, the heading error, inside the envelope, the events, the session. |
| **Plan against the robot** (details) | The comparison table: the measured and planned pose per stop, the error, the heading error, inside the envelope. |

---

## Keyboard

| Key | What it does |
|---|---|
| Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y | Undo / redo. |
| Ctrl+S | Save the mission. |
| Space | Play / pause. |
| Arrows | Nudge the selected waypoint (1 cm, Shift 10 cm, Alt 1 mm). |
| Delete / Backspace | Remove the selected waypoint. |
| Left / Right / Home / End | Move between the tabs of a strip. |

The keys are ignored while a text field has the focus, so typing a heading does not move a waypoint.

---

## How to extend it

| To add | Do this |
|---|---|
| A tab | A `<button role="tab" aria-controls="...">` and a `<section class="tabpanel">`. The controller wires it. |
| A plot | One entry in `PLOTS`: a key, a label and a build function. The tab strip and the drawing both come from it. |
| Something on the field | Draw it, then add one entry to `CANVAS_LEGEND` so the key stays right. |
| A control | Markup with an id, and the id in the script. The inventory test fails until this document names it. |
| A section in the report | A block in `overboard/viz/report.js`. The report is built from the numbers the Result tab shows, so it cannot disagree with the studio. |

`tests/test_studio_inventory.py` checks the document against the page: no dangling id lookups, the quoted counts are the real counts, every button in the page has a row here, and every generated region is named. A control added to the studio fails the test until this document is updated.

`tests/test_studio_report.py` covers the report and the project file: the report escapes what the operator typed, it is printable and self-contained, the server writes it next to the trajectory and serves it back, and a file that is not a project is refused with a reason.
