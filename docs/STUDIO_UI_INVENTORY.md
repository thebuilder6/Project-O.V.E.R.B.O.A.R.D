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

The page carries **169 element ids**: **83 controls with an id** (38 buttons, 37 inputs, 8 selects), of which 8 are the tab buttons themselves. Fifteen regions are filled in at run time:

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
| The score categories | `score-categories` | Result tab, score block |
| The lever table | `lever-table` | Result tab, tried levers |
| The history rows | `history-list` | Mission tab, the History list |
| The waypoint list | `wp-list` | Waypoint tab |
| The field limit note | `field-limit-note` | Solve tab, next to the accuracy limit |

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
| **Download the mat** | Composes the mat picture from FIRST's own field guide, fetched from the URL the field file records and kept in `mats/` (untracked; the artwork is not ours to ship). It also runs by itself on load and on a season change, so a fresh studio opens on the field: the server answers from `mats/` after the first time, and a failure is still said out loud in a toast. |
| **Width** / **Height** (mm) | The mat's size, so the picture and the footprints share one frame. |
| *mat name* / *credit* / *run history* | Readouts. |

The overlays on the canvas - mat, earlier runs, actuator saturation, late-apex cues, envelope, path, placement band - are switched from the **legend**, not from here: the key is where the operator looks to see what is drawn.

### Solve tab

The panel reads in the order a command is built: the **preset**, then three groups. **Solved** is what the solver is given, **Checked** is the gate and everything only it uses, **Written** is the export. A flat list of the flags put the gate below the export settings it gates, and left four controls that looked live with validation off. The line at the top of the tab repeats the header's status - "Solving...", the elapsed time and candidate of a running job, the finished run's duration and envelope, or the failure - so the panel the operator just filled in is never silent about how the run it started went.

**Preset**

| Control | What it does |
|---|---|
| **fast / safe** (radio cards) | A preset fills in the options the user did not set. fast: one candidate with the default settings, no validation, plant propagation - seconds. safe: the comparison - three candidates, validation, the closed-loop tick map; the three frontier paths are drawn together and the operator may export the one they pick. safe costs minutes: the reachability check, not the solve, is the time. |

**Solved**

| Control | What it does |
|---|---|
| **Samples per segment** | 2 to 200. Disabled while the automatic mesh is on, because the number is then not used. |
| **Auto invariant mesh (50 ms)** | Ticked by default: `-n` is left off the command, so the 3-Pillar engine sizes the invariant 50 ms mesh from the path. Unticking it sends the number in **Samples per segment**. |
| **Accuracy weight** | The smoothness weight (`--accuracy-weight`). |
| **Soft force weight** | A weight above zero prices wheel-force utilisation (`--force-weight`); zero is the default and sends nothing. |
| **Fast polish** | The budgeted solve. |

**Checked**

Every field below the switch is the gate's: the server passes none of them without `--validate`, so all of them are disabled while it is off, and each says in its tooltip what it does when it is on.

| Control | What it does |
|---|---|
| **Validate before export** | The gate; needs JAX. |
| **Scoring profile** | The decision score's weights: precision, balanced (default), or speed. |
| **Target time** / **Maximum time** | The duration the score prices and the duration it will not exceed. A validated run needs both. |
| **Jerk target** / **Jerk maximum** | The wheel-jerk pair the score prices (defaults 10/20 m/s^3). |
| **Accept failed plan at score** | 1-100. A failed plan is accepted at or above this score and can then be hub approved; it is never called passed. |
| **Candidates (1-3)** | How many of the planner's three frontier candidates are solved (sprint, sweeper, exploratory); clamped to 1-3 by the server. The number is kept while the gate is off, so turning validation back on keeps the count. |
| **Reachability propagation** | The model the envelope gate uses (`--propagation`): the open-loop plant (fast) or the closed-loop tick map (feedback-contracted, 2-10x slower). |
| **Accuracy limit (cm)** | How far the reachable set may spread around the plan at any sample (`--envelope-limit`, default 2 cm, 0.1-5 cm). A waypoint's error threshold still wins. The limit is absolute, so a long path can fail for being long (KI-054); the next field is how to price that distance. A run with `--envelope-from-field` replaces this with the field's per-sample clearance (clamped 0.5-5 cm), and the note beside the field (`field-limit-note`) shows that limit with its ~100 mm placement uncertainty. |
| **Allowance per metre (cm/m)** | How much further the reachable set may spread for every metre travelled, on top of the accuracy limit (`--envelope-limit-per-metre`, default 0, 0-5 cm/m). Zero keeps the gate absolute; a slope stops a long mission failing for being long. A waypoint's error threshold still wins. |
| **Keep the best path when validation fails** | Ticked by default: adds `--best-effort` next to `--validate`, so a run whose only failure is an accuracy excess (envelope or tracking) exports the best candidate and draws it on the field, marked failed. A force, slip, integration or convergence failure still blocks, and the push stays gated on the verdict. |

**Written**

| Control | What it does |
|---|---|
| **Export format** | None / controller JSON / Python / Python compact. |
| **Controller timestep** | The export's sample interval. |

## Field canvas

| Control | What it does |
|---|---|
| **Legend** (details) | The key to the field's colours, built from the list the drawing uses. Live: an entry for something that is not on the field is hidden. It is the single home of the canvas overlays: a row with a box before its label is a switch - mat picture, earlier solves, the actuator-limit colouring (both its rows), late-apex cues and the momentum bisector (both its rows), reachability envelope and its sample-box rendering, path, placement uncertainty. The envelope row is the overlay; the box row swaps its drawing for the union of the per-sample boxes the validator stores, which cannot fold over itself where the path turns (the corner ribbon can, and that fold is a drawing artefact, not the robot). Clicking one hides that overlay; the dimmed row (`aria-pressed="false"`) stays in the key to bring it back, so the key says both what the colours mean and what is drawn. The box is filled when the overlay is on and empty when it is off. A row the run cannot draw (an envelope on a run that was not validated) is dimmed, not clickable, and says why in its tooltip. The switches survive a reload with the mission: they are saved in the same session record as the solver options. |
| **Frame all** | Fits the view to the mission. |
| **Play** + slider + time | The playback of the solved trajectory. |
| *hint* | "Double-click the field to add a waypoint. Drag a waypoint to move it." A single click on the mat is a click, not a waypoint: it used to place one. The pointer agrees with it: a crosshair over the empty field (place), `move` over a waypoint (drag), `pointer` over a mission model (hover). |

Canvas contents: the mat, the grid, the mission models - each with the field's placement-uncertainty band as a dashed amber halo (click its key entry to hide it), the same 20-50 mm the collision report uses to call a hit uncertain - the earlier runs, the path, the trajectory (coloured by actuator saturation when that is on), the other candidates' paths from a comparison (one hue per seed, the clicked one stronger), the envelope (whose key names the propagation that produced it; a second key draws it as the union of the per-sample boxes instead of the corner ribbon), the failing segments (red), the worst-drift segment (amber), the solver's first guess (dashed), the late-apex diamonds and the momentum bisectors, the waypoints with their stop and event badges, the selected waypoint, the measured pose from the hub, and the collisions.

## Right column - the tabs

### Result tab

| Element | What it does |
|---|---|
| *status* | "Validated - the trajectory passed the validator" / "Not validated - this trajectory has not been checked against your robot" / "Failed validation - the best path is shown for inspection ... it is not approved for the hub" (a best-effort export) / the failure. |
| **Check it, then solve again** | The lever the status names: turns validation on, brings the Solve tab forward and solves again. Shown only after a solve that was not validated. |
| *facts* | Duration, path, peak wheel speed, samples, max envelope, solve (wall), the time against the top-speed ideal, the field. |
| *score* | The decision score (v1, profile), the four category scores (Physical, Accuracy, Motion Quality, Duration), and the note that it is not a probability. |
| *solve diff* | "vs. the previous solve: 0.27 s slower, the path 48 mm longer." |
| *tried levers* | The measured score delta, duration delta and measured points/second of the levers tried this mission session, ranked by measured improvement. No predicted estimates. |
| *collisions* | Where the plan met a mission model. |
| **Validation detail** (details) | The explainer, the candidate chips, the pick button, the per-segment table, the note, and the **Try** buttons. A chip names its seed by its physical role - Sprint, Sweeper, Exploratory - and a rescue seed is a dashed amber chip (🛟 Rescue). Every solved candidate's path is on the field (`--candidates-out`), the clicked chip's path drawn stronger, so the comparison is visible and not only tabulated. A path appears the moment its solve finishes, before its check and before the rescue round, so a validated run is not a blank field for its minutes; the chips and the table arrive with the report, when the run ends. The pick button (`Solve again and export: ...`) appears when the clicked candidate passed but is not the exported one: it re-solves with that candidate pinned (`--pick-candidate`), and the CLI refuses it if it no longer passes. The table carries, per segment, the envelope, the limit, the drift, and the three discretization mechanisms in mm (sagitta, heading mismatch, 20 ms hold lag). The **Try** buttons are the levers the numbers point at: the closed-loop tick map for an envelope failure, the gradient polish for a tracking error just over the gate, a soft force weight of 0.005 for a force or slip failure, and a weight of 0.001 for a rough profile. Each pulls its lever and solves again. |
| **Mission report** | The report of the last solve, as a file. |
| *downloads* | The trajectory, the controller, the Python module, the stats, the session, and the report. |
| **Mission report** | Writes the one-page report next to the trajectory and opens it: the mission, the robot and the result on one printable page. |
| **Solver log** (details) | The command line's output. |
| **Copy log** | Puts the whole log on the clipboard, not the selection. Where the browser refuses the clipboard (a plain http page), it selects the text instead so Ctrl+C finishes it. |

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

The plot tabs - Speed, Force, Acceleration, Turn rate, Heading, Envelope, Convergence - are built from the plot registry, over one canvas whose title and legend sit on separate rows, with a note under it. Adding a plot is one entry in the registry. Every plot that carries a limit names it in the legend.

### Run tab

| Control | What it does |
|---|---|
| **Connection** (select) | Bluetooth or USB. |
| **Hub name** (text) | The hub's name or address (BLE only). The field carries the choices a scan found (`#hub-choices`), so picking one fills it in. |
| **Scan for hubs** | Scans Bluetooth the way the IDE's device picker does - every visible device is asked whether it speaks the Pybricks protocol, which takes about a minute - and fills the Hub name field. A single hub is filled in and said out loud; several are offered as choices; none gets the guidance for a hub that is off or held by Pybricks Code. Refused while a job runs, because a push holds the hub's connection. |
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
| Something on the field | Draw it, then add one entry to `CANVAS_LEGEND` so the key stays right. A `toggle: '<state flag>'` on the entry makes it a switch the operator can click off and on - that is the only place an overlay switch goes, so the key and the control cannot disagree. The row shows while the thing exists in the data whether or not it is drawn, so the dimmed row is always the way back. |
| A control | Markup with an id, and the id in the script. The inventory test fails until this document names it. |
| A section in the report | A block in `overboard/viz/report.js`. The report is built from the numbers the Result tab shows, so it cannot disagree with the studio. |

`tests/test_studio_inventory.py` checks the document against the page: no dangling id lookups, the quoted counts are the real counts, every button in the page has a row here, and every generated region is named. A control added to the studio fails the test until this document is updated.

`tests/test_studio_report.py` covers the report and the project file: the report escapes what the operator typed, it is printable and self-contained, the server writes it next to the trajectory and serves it back, and a file that is not a project is refused with a reason.
