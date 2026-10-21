# **O**ptimal **V**elocity **E**ngine for **R**obotic **B**oundary-constrained **O**rchestration and **A**dvanced **R**amsete **D**ynamics

A trajectory optimizer for LEGO differential-drive robots in FIRST LEGO League (FLL). It generates time-optimal, physically constrained trajectories with CasADi + IPOPT, validates them before export, and ships a matching Pybricks Ramsete controller.

*Working with an AI coding agent? `AGENTS.md` is the short index of the rules and commands; the documents below are the long form.*

**📖 Technical documentation:** [docs/WHITEPAPER.md](docs/WHITEPAPER.md)
**🔬 Heuristics and verification:** [docs/HEURISTICS_AND_VERIFICATION.md](docs/HEURISTICS_AND_VERIFICATION.md) — every path-planning heuristic and the full verification chain. Its research parts have their own files: [docs/STRATEGY_LEDGER.md](docs/STRATEGY_LEDGER.md) (the strategies that failed, and the measurement that retired each) and [docs/RESEARCH_AGENDA.md](docs/RESEARCH_AGENDA.md) (the open questions).
**⚡ Optimal control derivation:** [docs/OPTIMAL_CONTROL_DERIVATION.md](docs/OPTIMAL_CONTROL_DERIVATION.md) — 5 invariant differential-drive control laws extracted from 50 generations of evolutionary search and the analytical Kinetic Frontier seeds (`overboard/core/frontier_seed.py`). *Historical: the evolutionary search was removed from the tree - it is in the git tag `archive/legacy-2026-10`.*
**🖼️ Visual gallery:** [docs/VISUAL_GALLERY.md](docs/VISUAL_GALLERY.md) — gallery of state-space phase portraits, multi-shape manifolds, and multi-island convergence figures from the archived search.

The audit items from KI-001 through KI-052 are resolved in the codebase, except the one still open (hardware calibration and digital-twin validation, KI-020); KI-004 is deleted and KI-032 is mitigated (see [KNOWN_ISSUES.md](KNOWN_ISSUES.md)). The 3-Pillar unified planner is the default solve path; `--no-unified` restores the stock candidate set, and the old Multi-Verse pipeline was removed from the tree (git tag `archive/legacy-2026-10`) and is not reachable from the CLI.

## Start here (first time)

Three steps. No Python knowledge needed.

### 1. Get the code

- **With git:**

  ```bash
  git clone https://github.com/thebuilder6/Project-O.V.E.R.B.O.A.R.D.git
  cd Project-O.V.E.R.B.O.A.R.D
  ```

- **Without git:** download the **Code → Download ZIP** from
  [the project page](https://github.com/thebuilder6/Project-O.V.E.R.B.O.A.R.D) and
  unzip it. Everything below works from the unzipped folder.

### 2. Install

Double-click **`install.bat`** (Windows) or run **`sh install.sh`**
(macOS/Linux).

It creates a private virtual environment in `.venv`, installs the solver and the
studio into it, and then checks the result. It is safe to run again: the
existing `.venv` is reused and the packages are updated.

You need **Python 3.9 or newer**. If the script says Python was not found,
install it from [python.org](https://www.python.org/downloads/) and tick
**Add python.exe to PATH** in the installer, then run the script again.

### 3. Run the studio

Double-click **`start_studio.bat`** (Windows) or run **`sh start_studio.sh`**
(macOS/Linux). Your browser opens on the studio: draw a mission on the field,
press **Solve**, and download the trajectory or the controller file.

The studio script installs the virtual environment itself if it is not there
yet, so on a fresh download you can skip step 2 and go straight to this one.

That is the whole loop. The terminal runs the same program:

```bash
.venv\Scripts\python -m overboard.main --gui     # Windows
.venv/bin/python -m overboard.main --gui         # macOS/Linux
```

If anything is missing, `python -m overboard.main --doctor` prints what is wrong and the
exact command that fixes it. The [Installation](#installation) section below
explains the same steps in detail.

## Features

- **Time-optimal trajectories**: Minimizes travel time while respecting actuator and traction limits
- **Per-wheel physics model**: Velocity-dependent motor force, lateral weight transfer, per-wheel slip limits, and declared worst-case uncertainty bounds
- **Accuracy tuning**: Optional smoothness/jerk penalty to improve real-world tracking
- **Fast simple solver**: Typical sub-second CasADi/IPOPT solves on a laptop
- **Browser studio**: Draw a mission in the browser, solve it, download the controller file
- **Trajectory validation**: CPU forward integration, constraint audit, final-waypoint check, and Immrax interval reachability
- **Safety gate**: `--validate` blocks every output and returns exit 1 on failure
- **Controller export**: Fixed-timestep JSON for the Pybricks controller; legacy or packed-compact Python export
- **Event markers and stops**: Trigger robot actions and force rest at waypoints

## Installation

The fast path is the one-click script: **`install.bat`** (Windows) or
**`sh install.sh`** (macOS/Linux). It creates a `.venv` in the project folder,
installs the core packages, installs the optional ones, and runs the check
below. The steps underneath are what it does, if you would rather run them
yourself.

### Prerequisites

- Python 3.9 or newer (the install script and the check both want 3.9)
- pip (it comes with Python)

### Core install (solver and studio)

```bash
pip install -r requirements.txt
```

This installs numpy, CasADi, Plotly, Click, Matplotlib and rsplan. It is enough for the default solver and the browser studio.

A virtual environment is the safer place for it, because it keeps the packages
out of the system Python and makes "uninstall" a matter of deleting the folder:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt       # macOS/Linux
```

### Optional extras (validation, sysid fitting)

`--validate` also needs JAX and Immrax:

```bash
pip install -r requirements-optional.txt
```

JAX 0.6.2 was used for the current tests. A fresh install brings the newest
stack (JAX 0.12.0, Immrax 0.3.8, numpy 2.5.4), and the straight example
solves and validates the same way there: 2.3766 s, 0.30 cm envelope, 0.34 cm
tracking.

Pushing to a hub (`--push-hub`, `overboard/deploy/hub_deploy.py`, `overboard/sysid/collect.py`) needs
`pybricksdev`, which is in the same file.

Check what your machine has at any time:

```bash
python -m overboard.main --doctor
```

It lists the Python version, every package, the example files, the output directory and the ports the studio wants, with the exact install command for anything missing.

### Generate a trajectory

```bash
python -m overboard.main -w examples/example_straight.json
```

The 3-Pillar unified planner is the default, so no extra flag is needed; `--no-unified` (or its alias `--simple`) solves the stock candidate set with CasADi/IPOPT directly.

## Quick Start

### 1. Provide a robot configuration

The recommended format is the canonical `robot` object used by `examples/fll_choreo.json`:

```json
{
  "name": "my_robot",
  "version": 4,
  "type": "Differential",
  "robot": {
    "mass": 0.723,
    "inertia": 0.001,
    "track_width": 0.0965,
    "wheel_radius": 0.028,
    "v_max_rad_s": 18.94,
    "t_max_nm": 0.138,
    "gearing": 1.0,
    "cof": 1.37,
    "gravity": 9.81,
    "torque_headroom": 0.85,
    "speed_headroom": 0.90,
    "uncertainties": {
      "cof": [1.24, 1.51],
      "torque_headroom": [0.75, 0.90],
      "gear_backlash_m": [0.0002, 0.0007],
      "inertia_kg_m2": [0.0008, 0.0011],
      "track_width_range": [0.096, 0.097],
      "wheel_radius_range": [0.0275, 0.0285]
    }
  }
}
```

Legacy Choreo files with a `config` object and `{ "val": ... }` entries, and flat dictionaries, are also accepted. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for every field and the uncertainty model.

**Key parameters:**

- `mass`: Robot mass in kg (0.5-1.5 kg typical)
- `inertia`: Rotational inertia in kg·m²
- `track_width`: Distance between wheel centers in meters
- `wheel_radius`: Wheel radius in meters
- `v_max_rad_s`: Motor no-load speed in rad/s (RPM × 2π/60)
- `t_max_nm`: Motor stall torque in N·m
- `gearing`: Gear ratio (1.0 for direct drive)
- `cof`: Wheel/field friction coefficient
- `torque_headroom`, `speed_headroom`: Safety margins used by the optimizer and validator
- `uncertainties`: Ranges used by the optimizer's robust limits and by Immrax

### 2. Define waypoints

```json
[
  { "x": 0.0, "y": 0.0, "heading": 0.0 },
  { "x": 1.0, "y": 0.5, "heading": 0.5 },
  { "x": 2.0, "y": 1.0, "heading": 1.0 }
]
```

**Waypoint options:**

- `x`, `y`: Position in meters (required)
- `heading`: Heading in radians (optional; `null` means unconstrained)
- `stop`: Force the robot to rest at this waypoint (optional, default `false`)
- `event`: Event name to export at this waypoint (optional, e.g. `"lower_arm"`)
- `error_threshold`: Local reachability-envelope threshold in meters (optional). It governs the way *into* this waypoint: the segment that ends here is held to it.
- `precision_mode`: Shortcut for a 0.005 m envelope threshold (optional)

### 3. Generate a trajectory

```bash
python -m overboard.main -w examples/example_straight.json -o output.traj --plot
```

### 4. Validate and export for the controller

```bash
python -m overboard.main -w examples/example_straight.json -o output.traj \
  --validate --export-format controller --controller-dt 0.02
```

`--validate` requires JAX/Immrax. It runs the CPU audit, Immrax closed-loop reachability, and the final-waypoint check before writing anything.

The `--preset safe` shortcut does the same in one flag:

```bash
python -m overboard.main -w examples/example_straight.json --preset safe -o output.traj
```

If you have run `pybricks_code/sysid_digital_twin.py` and fitted the logs with `python -m overboard.sysid.fit`, add `--sysid-fit sysid_fit.json` to opt into the measured model. The default kinematic model is unchanged when the flag is omitted. The current integration applies the straight-line actuator gain range, first-order time constant, command delay, odometry scale and the measured braking shortfall. A fit also measures the motor: the stall torque (the robot is pushed into a wall) and the free-running speed (the robot is lifted off the ground), which are the two ends of the torque-speed line the planner uses. With the flag, those measured numbers replace the declared ones in the robot config, so the optimizer, the validator and the exported controller all plan against the same motor. The model itself is in `overboard/models/drivetrain.py`; see [docs/research/RQ-B_drivetrain_model.md](docs/research/RQ-B_drivetrain_model.md) for what it predicts and what an independent simulator says about it.

The studio's **Field** block picks the season: the mission models standing on the mat come from `fields/<season>.json`, measured off FIRST's public field wireframe, and the mat's picture is the one you bring (FIRST's artwork is not ours to ship). After a solve the server reports where the planned path ran into a mission model and the studio marks those places on the field and lists them under the result. Nothing is planned around: the collisions are only marked, so you can decide what to do about them. A collision shallower than the field's placement uncertainty (50 mm) is reported as uncertain rather than as a hit, because two independent readings of the same wireframe disagree by up to 100 mm. See `research/field_maps/README.md` for the provenance. The studio can also fetch the season's mat picture for you: FIRST's mat artwork is copyrighted, so it is downloaded on request from the URL the field file records and kept in `mats/`, outside the repository.

The host tools are drivebase-agnostic. A minimal profile only needs wheel radius and, optionally, track width:

```json
{
  "name": "My differential drive",
  "wheel_radius_m": 0.028,
  "track_width_m": 0.0965
}
```

Use it when fitting and comparing:

```bash
python -m overboard.sysid.fit logs/session.csv --poses poses.csv \
  --profile examples/drivebase_profile.example.json --out sysid_fit.json
python -m overboard.sysid.compare output.traj poses.csv --run-id 4 \
  --profile examples/drivebase_profile.example.json
```

`--robot-config` remains supported for a full robot config, and `--wheel-radius` overrides either source. The comparator also accepts a plain predicted-pose CSV (`t,x,y,heading`) instead of a `.traj`, so it is not tied to this optimizer. It warns if the profile geometry differs from the trajectory config.

For the end-to-end profile, fit, validate, and compare walkthrough, see [docs/SYSID_QUICKSTART.md](docs/SYSID_QUICKSTART.md).

After a hardware run, compare the planned and measured stop/final poses:

```bash
python -m overboard.sysid.compare output.traj poses.csv --run-id 4
```

For a multi-stop trajectory, pass the waypoint file used to plan it, or use explicit trajectory sample indices:

```bash
python -m overboard.sysid.compare output.traj poses.csv --run-id 12 \
  --waypoints examples/example_with_stops.json
python -m overboard.sysid.compare output.traj poses.csv --stop-indices 20,40
```

The report prints position and shortest-angle heading errors and says whether each measured pose is inside the trajectory's reachability envelope. It also prints the drivebase profile and warns if its geometry differs from the trajectory config. It is a diagnostic only; it does not change the 5 mm/2 cm gate.

When validation passes, the command writes:

- `output.traj`: Full variable-timestep trajectory
- `output_controller.json`: Fixed 20 ms controller profile
- `output.py` instead of the controller file when `--export-format python` is used
- `output.py` packed trajectory module when `--export-format python-compact` is used; import it with `pybricks_code/ramsete_compact.py` (or `ramsete_compact_events.py` for events)

> **Safety note:** if validation fails, the command prints the reason, exits 1, and writes no trajectory, controller, Python file, or stats. A pre-existing file at the output path is left untouched. Which example missions currently pass is a measurement, not a promise - it changes with the config and the planner - so always check the exit code, and see the ledger in `docs/HEURISTICS_AND_VERIFICATION.md` and the KI-020 entry in `KNOWN_ISSUES.md` for the latest numbers.

### 5. Browser studio (optional)

```bash
python -m overboard.main --gui
```

Or double-click `start_studio.bat` (Windows) / `sh start_studio.sh`
(macOS/Linux), which uses the `.venv` from `install.bat` and opens the same
studio.

The studio opens in your browser: draw the mission on the field, set headings, stops and events, pick a preset, and solve. The result shows the trajectory, the reachability envelope and the numbers, and you can download the trajectory and the controller file.

The page is three columns. The left one is what you set (tabs: **Mission**, **Robot**, **Field**, **Solve**), the canvas in the middle is the work, and the right one is what you read (tabs: **Result**, **Waypoint**, **Plots**, **Run**). The line in the header always carries the result, so a tabbed column can never hide the answer, and the field carries a key to its own colours. Every control is listed in [docs/STUDIO_UI_INVENTORY.md](docs/STUDIO_UI_INVENTORY.md), with the structure and how to extend it.

**Mission report.** One button writes a single printable page - the mission, the robot and the result, with the planned path and the wheel speeds drawn - and opens it in a new tab. It is written next to the trajectory, so the report and the plan travel together, and it is built from the same numbers the studio shows, so it cannot disagree with the screen.

**Project file.** One file with the mission, the robot config, the solve options, the measured model and the field. Send it to a team-mate and they open it and get the whole setup, not just the path.

The studio runs the same command line in the background, so the export gate, the validation rules and the file formats are identical to the terminal path. A failed validation shows the reason and offers no download.

The studio also draws the plots the command line can make, from the same samples: the velocity profiles, the wheel forces against the force limit, the accelerations, the heading over time, and how the solve converged. The solver's first guess is drawn on the field as a dashed path, so you can see what the optimizer changed.

The Advanced section has the same levers the command line has: **fast polish**, **self-consistent** samples and **sweep** (the timing model x weight grid, ten candidates - kept for experiments; the command line marks it deprecated because the 80 ms invariant mesh removed the need for it). The self-consistent switch is ticked by default, like the command line's, and unticking it is `--no-self-consistent`. The candidate chips are the leaderboard, and they say which weight each candidate used.

When a validation fails, the studio shows **where**: a row per path segment with the reachability envelope, the local limit and the drift, the failing segment marked in red on the field, and the reason the gate gave. Under the table it names the lever that might fix it - the other timing model, a smaller accuracy weight, or the sweep - and each one is a button that re-solves. The robot itself is editable in the studio — mass, wheel and track size, motor limits, friction and the uncertainty ranges — so the last JSON file you need to open by hand is gone.

Before you solve, the mission panel shows the path length and the fastest the robot could possibly do it, so a waypoint dragged five metres away is obvious. The Advanced section has the **fast polish** switch, which cuts the ten-waypoint complex mission from 26 s to 15 s for the same trajectory.

When a solve passes, the **Run on the robot** section can push the approved trajectory to a Pybricks hub over Bluetooth or USB. The hub's console streams into the studio while the robot drives, and the pose it reports comes back as a marker on the field next to the plan, with the error in millimetres. The session is written next to the mission as `missions/<name>.session.csv`, so the plan and what the robot actually did travel together. The studio also checks the plan against the robot's own files: the event names the robot has actions for, and the wheel size it really has.

A calibrated robot can be loaded into the studio (**Measured model** in the robot panel). The fit is kept with the robot, every solve then runs against the measured model instead of the nominal one, and the panel shows what was measured: the straight and turn gain with their 95% intervals, the response time and delay, the gyro bias, the odometry scales and the brake slip.

After a push, the studio shows the plan against the robot in the same table the command line prints: the measured and planned pose per stop, the position and heading error, and whether the measured pose landed inside the reachability envelope.

**Calibrate this robot** runs the whole calibration in the browser: push the test suite to the hub and watch the progress, type the tape measurements into a form instead of a sheet of paper, and fit the model from the session and the form. The session, the sheet, the measured poses and the fit are kept in `calibration/`, and a session collected in the studio can also be fitted with `python -m overboard.sysid.fit` on the command line.

While it solves, the header ticks the elapsed time and names the stage (*Solving… - 4.2 s - 3 candidates in parallel*). Stops and events are marked on the field, and when several candidates were compared the studio says which one won and what the others cost: *"equal-duration baseline won (7.21 s). per-segment baseline was 0.48 s faster but failed: tracking error 0.61 cm > 0.50 cm"*.

Your work is never lost: every edit can be undone (Ctrl+Z / Ctrl+Shift+Z) - the robot config included, not just the mission - the mission is kept in the browser and comes back after a reload, and **Download** saves a copy of the mission JSON to your machine. **Save** writes it to `missions/` so it shows up in the mission list.

The missions folder is **its own git repository**: every save is a commit, so the **History** list under the mission name shows every version and can put an earlier one back. The commits need no git configuration on your machine, and your project repository is never touched.

The arrow keys nudge the selected waypoint (1 cm; Shift is 10 cm, Alt is 1 mm), and a held key counts as one undo step.

### 6. Solve in the browser (optional)

```bash
python -m overboard.main --live -w examples/example_complex_mission.json
```

`--live` opens the studio on that mission and solves it there, with the same options the command line would use: the browser shows the solve, and the studio does it properly. The server blocks until you stop it, and it never takes a port from another program.

## CLI Options

```text
Usage: python -m overboard.main [OPTIONS]

  FLL Trajectory Optimizer CLI. Generates time-optimal trajectories for Lego
  robots.

  Examples:
    python -m overboard.main -w examples/example_straight.json
    python -m overboard.main -w mission.json --preset safe -o mission.traj
    python -m overboard.main --gui
    python -m overboard.main --doctor

Options:
  -c, --config FILE               Path to the robot configuration file
                                  (.json).
  -w, --waypoints FILE            Path to waypoints JSON file. Required unless
                                  --gui or --doctor.
  -o, --output TEXT               Output trajectory file path.
  --gui                           Open the browser studio: draw a mission,
                                  solve it, export it.
  --gui-port INTEGER              Port for the browser studio.
  --doctor                        Check the install, dependencies, examples
                                  and ports, then exit.
  --simple                        Alias for --no-unified: solve the stock
                                  candidate set with the CasADi/IPOPT
                                  optimizer directly.
  --preset [fast|balanced|safe]   Named starting point. fast: 1 candidate, no
                                  validation. balanced: the defaults. safe: 3
                                  candidates, --validate and a controller
                                  export. Options you set yourself always win.
  -n, --samples INTEGER RANGE     Samples per segment.  [2<=x<=200]
  -a, --accuracy-weight FLOAT RANGE
                                  Smoothness/accuracy weight (0 = pure time-
                                  optimal).  [x>=0.0]
  --force-weight FLOAT RANGE      Soft wheel-force and traction-utilization
                                  penalty weight in the NLP objective (simple
                                  solver only).  [x>=0.0]
  --candidates INTEGER RANGE      Candidates to solve and compare. Each
                                  candidate combines an initial-guess seed
                                  (baseline, rolling, bow_left, bow_right,
                                  turn_first) with the timing mode: by default
                                  candidate 1 is equal-duration baseline,
                                  candidate 2 is per-segment baseline, then
                                  the remaining seeds alternate. Validation
                                  picks the passing candidate with the best
                                  fitness, then the fastest. Without
                                  --validate only the first candidate is
                                  solved. Use 1 for a single solve.  [x>=1]
  --stop-waypoints TEXT           Comma-separated waypoint indices where robot
                                  must stop (e.g., "2,5,7").
  --events TEXT                   Comma-separated waypoint:event pairs (e.g.,
                                  "2:lower_arm,5:release").
  --per-segment-dt                Force each waypoint segment to have its own
                                  sample interval (simple solver only). This
                                  is the general timing model. By default
                                  candidates mix both modes.
  --uniform-dt                    (Deprecated: equal-duration dt severely
                                  penalizes variable-length segments) Force
                                  one shared sample interval across all
                                  segments.
  --no-parallel                   Solve the candidates in this process instead
                                  of in worker processes.
  --workers INTEGER               Number of worker processes used to solve
                                  candidates in parallel.
  --fast-polish                   Solve large missions faster. The warm-start
                                  solve can crawl through thousands of IPOPT
                                  iterations; this gives it an iteration
                                  budget and finishes from that point with a
                                  cold pass. Missions that solve inside the
                                  budget return the same trajectory as before.
  --sweep                         (Deprecated: 80 ms invariant mesh eliminates
                                  manual grid sweep) Search the timing model
                                  and accuracy weight grid.
  --self-consistent / --no-self-consistent
                                  Plan a self-consistent trajectory: the
                                  samples are placed on the path the
                                  trajectory's own (v, omega) profile traces,
                                  so the headings are its tangents (simple
                                  solver only). On by default (KI-026).
  --unified / --no-unified        Run the 3-Pillar Unified Trajectory Engine
                                  (default: on): Kinetic Frontier Seeds + 80
                                  ms Invariant Mesh + Discretization-
                                  Consistent Collocation.
  --surrogate-polish              Run the 5-Layer Unified Surrogate-to-Truth
                                  parallel pipeline: CasADi Discretization-
                                  Aware Metric + Linear Epigraph Force Bounds
                                  + Single-Process ThreadPoolExecutor +
                                  Compiled float64 jax.lax.scan Reachability +
                                  Differentiable Immrax Polish + Gate-
                                  Certified Min-Time Selection.
  --pyramid                       Multi-Fidelity "Screen-then-Refine" Pyramid:
                                  vectorized JAX sieve over 100-350
                                  hypotheses, topological clustering into
                                  survivor seeds, and process-isolated
                                  parallel IPOPT solves.
  --pyramid-candidates INTEGER RANGE
                                  Number of top distinct homotopy finalists to
                                  solve in parallel during --pyramid mode
                                  (defaults to 10).  [x>=1]
  --h-adaptive                    Adaptive mesh refinement (h-refinement):
                                  selectively allocates higher node density
                                  (8-10 nodes) to curved apex segments,
                                  lowering forward-integration discretization
                                  error to sub-centimeter levels (<1.0 cm).
  --validate / --no-validate      Run validation and block export on failure
                                  (requires JAX/Immrax).
  --sysid-fit FILE                Optional sysid_fit.json digital-twin fit
                                  (requires --validate or --live).
  --export-format [none|controller|python|python-compact]
                                  Export format for controller consumption
                                  (python-compact is the packed Pybricks
                                  format).
  --controller-dt FLOAT RANGE     Fixed timestep for controller export
                                  (seconds).  [0.001<=x<=1.0]
  --control-dt FLOAT RANGE        The control period the validator models, in
                                  seconds. The deployed controllers hold their
                                  command for 20 ms, but a robot that cannot
                                  sustain it runs slower; set this to the
                                  period the robot actually achieves (the
                                  pushed session reports it) so the gate is
                                  measured against the controller that will
                                  run.  [0.001<=x<=0.2]
  --propagation [plant|tick]      Which reachability propagation the envelope
                                  gate uses. "plant" is the default: the plant
                                  is embedded and propagated, and the
                                  controller is evaluated outside the
                                  propagation, so the tube bounds an open-loop
                                  plant driven by a pre-recorded command.
                                  "tick" embeds one control tick of the closed
                                  loop - controller, pose estimate and delayed
                                  command inside the map - and propagates
                                  that, so the feedback contracts the tube the
                                  way the robot's does. It needs a Jacobian
                                  inclusion and costs 7-10x the validation
                                  time. The propagation in force is reported
                                  with every candidate.
  --benchmark                     Collect comprehensive benchmarking data for
                                  whitepaper.
  --push-hub                      Push the trajectory to a Pybricks hub and
                                  run it there (needs pybricksdev and a hub
                                  with Pybricks firmware).
  --hub-conn [ble|usb]            How to reach the hub when pushing.
  --hub-name TEXT                 Bluetooth name or address of the hub to push
                                  to.
  --hub-log FILE                  Write the hub console output to this file.
                                  The runner reports the measured pose as a
                                  POSE line in it.
  --plot                          Plot the resulting trajectory.
  --animate                       Animate the trajectory in real-time.
  --live                          Open the browser studio on this mission and
                                  solve it there (blocking).
  --show-convergence              Show convergence visualization.
  --iteration-history FILE        Write the solver's iteration history to this
                                  JSON file, for the studio to plot how the
                                  trajectory converged.
  --convergence-mode [parallel|best|layered]
                                  Convergence visualization mode.
  -v, --verbose                   Show solver internals.
  -q, --quiet                     Suppress verbose output.
  -h, --help                      Show this message and exit.
```

## Project Structure

```text
Project-O.V.E.R.B.O.A.R.D/
├── install.bat / install.sh     # One-click install into .venv
├── start_studio.bat / .sh       # One-click start of the browser studio
├── benchmark.py                 # The four benchmark modes
├── run_tests.py, testcache.py   # Test runner and its cache
├── overboard/
│   ├── main.py                  # CLI entry point and export gate
│   ├── gui_server.py            # Browser studio backend (stdlib HTTP server)
│   ├── diagnostics.py           # Environment checks for --doctor
│   ├── core/                    # optimizer, frontier seeds, unified planner, surrogate polish
│   ├── models/                  # robot config, drivetrain, JAX models
│   ├── verification/            # CPU validator, Immrax reachability, reachability engine
│   ├── sysid/                   # collect, fit, compare, digital twin, sheet form
│   ├── deploy/                  # controller export, hub push and BLE discovery
│   ├── viz/                     # plotters, fields, studio.html/js/css
│   └── legacy/                  # archived Multi-Verse pipeline and research prototypes
├── tests/                       # unittest suite
├── examples/                    # Example configs and waypoint files
├── missions/                    # Missions saved from the studio
├── pybricks_code/               # Pybricks robot controller code (archive/ holds the demos)
├── research/                    # Measurement harness for the research agenda (not imported by the planner)
└── docs/                        # Configuration, examples, API, whitepaper
```

## How It Works

The simple optimizer solves a direct-collocation optimal-control problem:

1. **Discretization**: Each segment is divided into samples; the decision variables include the timestep and the pose/wheel speeds at every sample. The candidate set mixes a uniform-timestep formulation with a per-segment formulation (`--per-segment-dt` forces the latter, `--uniform-dt` forces the former).
2. **Objective**: Minimize total time, plus an optional jerk/smoothness penalty controlled by `--accuracy-weight`.
3. **Constraints**:
   - Self-consistent position rule by default: four-node Gauss-Legendre integration along the integrated heading, so the samples sit on the path the plan's own profile traces (`--no-self-consistent` restores the trapezoidal rule)
   - Waypoint position constraints and optional heading constraints
   - Start, end, and intermediate stop constraints
   - Per-wheel torque-speed limits
   - Per-wheel normal force with lateral weight transfer
   - Per-wheel traction limits
   - Worst-case force bounds over the declared uncertainty ranges
4. **Solve**: IPOPT finds a locally optimal trajectory. The result is converted to Choreo v3 samples.

The 3-Pillar unified planner is the default path: it supplies the seeds and the collocation mesh, and `--no-unified` solves the stock candidate set directly with the same CasADi/IPOPT backend. The old Multi-Verse pipeline was removed from the tree (git tag `archive/legacy-2026-10`) and has no CLI entry point.

## Common Usage Patterns

### Simple straight line

```bash
python -m overboard.main -w examples/example_straight.json -o straight.traj --plot
```

### Path with turns

```bash
python -m overboard.main -w examples/example_s_curve.json -o s_curve.traj --plot
```

The S-curve is the harder case: run it with `--validate` and check the exit code, because not every path passes the Immrax gate under every config (KI-020).

### Mission with stops and events

```bash
python -m overboard.main -w examples/example_complete.json -o mission.traj \
  --export-format controller --controller-dt 0.02 --plot
```

Add `--validate` if you want the safety gate; check the exit code, because some multi-waypoint paths still fail the Immrax gate (KI-020).

### Smooth trajectory for better tracking

```bash
python -m overboard.main -w examples/example_s_curve.json -o smooth.traj \
  -a 1.0 --plot
```

### Full suite (experimental)

```bash
python -m overboard.main -c examples/fll_choreo.json -w waypoints.json -o mission.traj \
  -a 1.0 --benchmark --validate --export-format controller \
  --show-convergence
```

- `-a 1.0`: Smoothness priority
- `--benchmark`: Quality/telemetry stats after the export gate
- `--validate`: CPU + Immrax gate
- `--show-convergence`: Convergence plot for the simple optimizer
- `--workers`/`--no-parallel` are not quality controls in the current tree

## Accuracy vs. Speed Tradeoff

By default the optimizer is purely time-optimal (`--accuracy-weight 0.0`). This is the fastest solution but can demand rapid acceleration changes. Increasing `--accuracy-weight` adds a jerk penalty and trades time for smoother wheel commands:

- `0.0`: Pure time-optimal
- `0.5-1.0`: Balanced smoothness
- `2.0+`: Very smooth, with a larger time cost

The exact tradeoff depends on the path and robot. Measure it on your own trajectory rather than relying on a fixed percentage.

With `--validate`, the tradeoff is measured, not assumed: `--candidates N` (default 2) solves N candidates, validates and scores every one, and keeps a passing one - the **fastest** on the default `--unified` path (the envelope breaks a duration tie), or the one with the **lowest 5-part fitness score** on the stock set (the envelope and the shortest trajectory break ties). The stock candidate set (`--no-unified`) combines initial-guess seeds with the timing mode: by default candidate 1 is equal-duration `baseline` (linear, zero-speed guess), candidate 2 is per-segment `baseline`, then the remaining seeds (`rolling`, `bow_left`, `bow_right`, `turn_first`) alternate modes. The default `--unified` path replaces that set with three frontier candidates (`frontier_sprint`, `frontier_sweeper`, `frontier_exploratory`, all per-segment at a low accuracy weight), and `--candidates` truncates that list. `--uniform-dt` or `--per-segment-dt` forces one mode. `--candidates 1` disables the comparison. A slower or differently shaped candidate that scores worse is discarded; nothing is escalated silently.

The two timing modes are not peers. The equal-duration mode shares one `dt` across all segments, and because every segment has the same sample count, that also forces every segment to take the **same time**. On a path with unequal segment lengths the short segment inherits the long segment's duration, so the robot crawls through it and has to make the time back up with non-monotone wheel motion. The per-segment mode gives every segment its own duration and is the general model: setting every `dt_i` equal recovers the equal-duration solution exactly, and the per-segment optimum is never worse at the same weight. Both modes are held to the same wheel-jerk discipline, so `--accuracy-weight` means the same thing in either (KI-022).

Where the time allocation may change is a separate question, and the per-segment mode now answers it explicitly (KI-024). A waypoint is a legitimate **timing boundary** only when the robot stops there *and* the heading is pinned: the state there is then fully determined, so the pieces on either side need agree on nothing. At every other waypoint the pieces stay linked:

| waypoint | what the two pieces must agree on |
|---|---|
| stop and pinned heading | nothing — the state is fully determined |
| pinned heading, no stop | the wheel velocities there |
| stop, no pinned heading | the heading there; the velocity is zero on both sides |
| neither | the heading and the wheel velocities |

Because the path is one NLP over one shared variable set, the state agreement is automatic. What the per-segment timing could break is the *reading* of the motion profile: the acceleration before a waypoint is `(vl[k]-vl[k-1])/dt_prev` and the one after it is `(vl[k+1]-vl[k])/dt_next`, so a change of interval across the waypoint re-reads the same velocity ramp as two different accelerations and puts a step in the wheel force profile exactly there. The per-segment model therefore requires the wheel acceleration — and so the wheel force — to be continuous through every waypoint the robot does not stop at. A waypoint where the robot stops is exempt, because the wheel force has to reverse sign when the robot comes to rest and goes again. The equal-duration model needs no counterpart: it never changes `dt`, so it reads the ramp the same way on both sides.

The default candidate set includes both timing modes, so the equal-duration candidate preserves the complex-mission pass while an uneven path can select per-segment. The CPU audit runs on every candidate, not only the winner, so a candidate that passes Immrax but fails the force/slip/integration gate cannot block a later passing candidate. On an uneven 1.0 m + 0.2 m path, per-segment at `--accuracy-weight 0.005` passes all gates at 2.784 s with envelope 0.32 cm, tracking 0.36 cm and peak wheel jerk 9.52 m/s³, versus 4.456 s / 0.37 cm / 0.36 cm / 9.52 m/s³ for the equal-duration mode at the same weight; the mixed loop selects the per-segment candidate. At the default weight 0, per-segment is 2.710 s / 0.32 cm but fails the Immrax and CPU gates, so the mixed loop falls back to the equal-duration candidate (4.463 s / 0.35 cm). On the complex mission at `--accuracy-weight 0`, per-segment is faster (4.650 s, envelope 1.00 cm) but its tracking is 0.84 cm, above the 5 mm gate, and its motor force is 0.074 N over the limit, so it is rejected and the equal-duration 6.068 s / 1.22 cm candidate wins. Forcing one mode is available for experiments and reproducibility, not as a silent default.

### Sweeping for the fastest passing trajectory (deprecated)

> **Deprecated.** `--sweep` still runs, but the 80 ms invariant mesh in the default `--unified` path made the manual grid unnecessary, so the CLI now warns and turns `--unified` off. The section is kept because the measurements below are the record of why the grid no longer pays.

The two timing modes and the accuracy weight are a two-dimensional search, and the best combination depends on the path. `--sweep` searched it: it added the timing-mode × accuracy-weight grid (weights `0`, `0.001`, `0.002`, `0.005`, `0.01`) to the candidate set, validated every combination on the same scale, and let the usual rule pick the winner — the passing candidate with the lowest fitness score (the time its defects are worth), the envelope and the shortest one breaking the tie. The default candidates stayed in the set, so a sweep could only add options, never remove them.

Measured on the example missions:

| mission | default candidates | with `--sweep` | change |
|---|---|---|---|
| `example_mission_first4.json` | **1.833 s** (per-segment, weight 0) | 1.833 s | unchanged |
| `example_with_events.json` | **4.416 s** (per-segment, weight 0) | 4.416 s | unchanged |
| `example_complex_mission.json` | 6.068 s | 6.068 s (equal-duration, weight 0) | unchanged |
| `example_straight.json` | 2.377 s | 2.377 s | unchanged |

With the merge conditions in place (KI-024) the per-segment candidate now passes the gates on the first-4 and events missions, so the default two candidates already find those wins and the sweep no longer changes them. It still matters when the best combination is a weight other than the default, because it is the only way to try the grid.

The sweep is opt-in because it costs more: ten candidates instead of two, so ten solves and ten validation passes. It is the tool to reach for when the robot's time matters more than the solve time, and it is the honest way to see the speed-versus-margin frontier: the rejected candidates are printed with their duration, envelope and tracking error, so the trade is visible instead of hidden behind one number.

## Output Format

```json
{
  "name": "output",
  "version": 3,
  "trajectory": {
    "config": { "mass": 0.723, "track_width": 0.0965, "...": "..." },
    "samples": [
      {
        "t": 0.0,
        "x": 0.0,
        "y": 0.0,
        "heading": 0.0,
        "vl": 0.0,
        "vr": 0.0,
        "omega": 0.0,
        "al": 0.0,
        "ar": 0.0,
        "fl": 0.0,
        "fr": 0.0
      }
    ]
  }
}
```

Samples may also contain `event` and `reachability_envelope`. The validator, exporter, plotters, and Pybricks runtime consume this schema; keep it stable.

## Validation

`--validate` performs three independent checks:

1. **CPU audit**: per-wheel motor force, per-wheel slip, total traction, and wheel-speed limits at every sample.
2. **Forward integration**: RK4 at 1 ms against the planned trajectory; max and final position errors must be under 1 cm.
3. **Immrax reachability**: closed-loop Ramsete with interval parameters, a linear reference, a 20 ms sample-and-hold controller, and 5 ms plant substeps. It reports tracking error and envelope radius separately; transit samples use a 2 cm envelope threshold, and precision waypoints use their local threshold.

With `--sysid-fit`, the Immrax plant switches from the default three-state kinematic model to a five-state model with actual wheel speeds. The fitted straight-line gain is an interval, the time constant is a first-order lag, the delay is applied to the held command, and the odometry distance/heading scales are applied to the feedback pose. The fit is optional and disabled by default; without it, the default path is unchanged.

The final pose is also checked against the last waypoint. If any check fails, the CLI exits 1 and writes nothing.

With `--validate`, `--candidates N` (default 2) solves N trajectories. With the stock candidate set, each candidate combines an initial-guess seed with a timing mode at the same `accuracy_weight`; the default `--unified` path uses its three frontier candidates instead. Every candidate is validated and CPU-audited; the selected one is a passing candidate - the fastest on the default `--unified` path, or the one with the lowest fitness score on the stock set, where the score is a single number in seconds covering the reachability spread at the stops, the slip risk, the RMS wheel force, the tracking error and the run time (the envelope and the shortest trajectory break ties). If none pass, the candidate closest to passing is selected for reporting but still blocked from export. Without `--validate`, only the first candidate is solved.

**Measured current examples** (model predictions, not hardware). These rows are the **stock** set's measurement, taken before the unified engine became the default: two candidates, candidate 1 equal-duration baseline and candidate 2 per-segment baseline. Run the default path and the candidate table names the frontier candidates instead.

| Case | Result |
|---|---|
| Straight, `--validate` | PASS; selected 2.377 s, envelope 0.28 cm, tracking 0.35 cm; both candidates converge to the same trajectory. |
| First 4 mission waypoints, `--validate` | PASS; selected 1.833 s, envelope 0.27 cm, tracking 0.38 cm (per-segment); the equal-duration candidate is 2.176 s, 0.27 cm / 0.39 cm. |
| S-curve | FAIL; selected 3.667 s, envelope 0.72 cm, tracking 1.26 cm (per-segment); equal-duration 3.696 s, 0.77 cm / 1.32 cm, also fails. |
| Stops | FAIL; selected 6.246 s, envelope 0.93 cm, tracking 0.61 cm (equal-duration); per-segment 5.138 s, 0.96 cm / 1.81 cm, also fails. |
| Events | PASS; selected 4.416 s, envelope 0.88 cm, tracking 0.44 cm (per-segment); the equal-duration candidate is 4.637 s, 0.97 cm / 0.52 cm, rejected by the 5 mm tracking gate. |
| Complex mission | PASS; selected 6.068 s, envelope 1.22 cm, tracking 0.49 cm; per-segment 4.650 s, 1.00 cm / 0.84 cm, rejected by the 5 mm tracking gate and a 0.074 N motor-force overshoot. |
| Uneven test path (1.0 + 0.2 m), default mixed `-a 0.005` | PASS; selected per-segment 2.784 s, envelope 0.32 cm, tracking 0.36 cm, peak wheel jerk 9.52 m/s³; equal-duration at the same weight: 4.456 s, 0.37 cm / 0.36 cm / 9.52 m/s³. At `-a 0` per-segment is 2.710 s / 0.32 cm but fails the Immrax and CPU gates, so the equal-duration candidate 4.463 s / 0.35 cm is selected. |

All of the failing commands exit 1 and write no output. This is the gate working, not a crash. These results use the narrowed `track_width_range` and `gear_backlash_m` in `examples/fll_choreo.json`; those are test assumptions, not measured hardware values. Without a hardware fit the validator is still kinematic; `--sysid-fit` adds the measured actuator/odometry model, but the thresholds are unchanged. See KI-020.

The full gate chain, every threshold with its source line, and the measured comparison of all five initial-guess seeds under both timing models (60 candidates, with the raw logs) are in [docs/HEURISTICS_AND_VERIFICATION.md](docs/HEURISTICS_AND_VERIFICATION.md). The ledger of strategies that did not work is in [docs/STRATEGY_LEDGER.md](docs/STRATEGY_LEDGER.md), and the research agenda in [docs/RESEARCH_AGENDA.md](docs/RESEARCH_AGENDA.md).

## Controller Export

`--export-format controller` writes `<output>_controller.json`:

- Fixed timestep, default 20 ms
- Samples contain `t, x, y, heading, vl, vr, v, omega`, plus `event` when one is assigned
- Ready for `pybricks_code/ramsete_fast.py` or another trajectory follower

`--export-format python` writes `<output>.py` containing a `config` dictionary and the `samples` list for direct import by `ramsete_fast.py` or `ramsete.py`.

`--export-format python-compact` writes a Pybricks module with a packed `samples` bytes object (4-byte header plus 24 bytes per sample), a `config` dictionary, and an `events` tuple. Use `pybricks_code/ramsete_compact.py` for pathing or `ramsete_compact_events.py` for event missions. The compact controllers are available but are not wired into the existing `robot.py` menu or missions.

## Performance

Measured on the current working tree with the example config:

- 1 m straight, simple optimizer: about **0.06 s** solve wall time, **2.377 s** trajectory time, **0.000407 m** max forward-integration error.
- 10-waypoint complex mission, simple solver: the IPOPT polish takes **1557 iterations** and about **35 s** on the machine it was measured on, and **669 iterations** and about **3 s** on another - the count depends on the machine's floating point, so IPOPT takes a different path through the same NLP. `--fast-polish` budgets the warm pass at 200 iterations and then solves the mission cold from the original guess: **354 iterations** and **7 s** on the first machine, **708** and **3 s** on the second, for a trajectory within 1 % of the plain one. Every other example mission solves inside the iteration budget and returns a bit-identical trajectory.

`python benchmark.py --mode solver --config examples/fll_choreo.json` prints that table for every example mission; add `--fast-polish` for the fast path.

`--validate` adds the Immrax reachability pass and, by default, solves two candidates, so validation runs take longer than a raw single solve. The candidates are independent, so they are solved in worker processes and then scored one by one in this process, where the JAX validator is already warm. There are no verified hardware timing numbers in this repository yet.

### Why there is no GPU path

The solve is not GPU-shaped, and the measurements are not ambiguous:

- The time is in **IPOPT**, which is CPU-only. CasADi's `ipopt` plugin uses a sparse direct solver (MUMPS in the pip build; MA57/MA27 are not compiled in), and IPOPT has no GPU backend at all. Nothing on a GPU can take part in that loop.
- The **JAX** part is latency-bound, not FLOP-bound. The reachability validation is small-matrix work behind a first-call compilation cost. Moving it to a GPU cannot help.
- The **Immrax** validation marches a 3-to-5 state system through a Python loop of 20 ms control ticks. On a GPU every one of those tiny steps becomes a kernel launch.
- **JAX has no native Windows GPU build.** The support matrix lists Windows x86_64 + NVIDIA as unsupported and WSL2 as experimental, so a GPU path would fork the development environment for every contributor.

The remaining lever was the solver itself, not the hardware: the warm-start solve treats the initial point as a previous iterate, and on a large mission that makes IPOPT crawl for thousands of iterations. `--fast-polish` gives the warm pass an iteration budget and finishes from that point with a cold pass, which reaches the same answer in a fraction of the iterations. A GPU would have addressed about 0.2 % of the runtime.

### Example Trajectory

The example plots that used to sit here were produced by the archived evolutionary search; they now live in [docs/VISUAL_GALLERY.md](docs/VISUAL_GALLERY.md), with a note on how to regenerate them from the `archive/research-2026-10` tag.

## Research & References

See [docs/trajectory_tools_research.md](docs/trajectory_tools_research.md) for comparisons with Choreo, PathPlanner, and CasADi, plus academic references.

## Development Status

- ✅ Phase 1: CasADi + IPOPT solver
- ✅ Phase 2: Time vs. accuracy tradeoff
- ✅ Phase 3: Validation and controller export (with a hard export gate)
- ✅ Phase 5: Events and stops implemented; the final-waypoint event fires (KI-012)
- ⚪ Phase 6: Field geometry/obstacle constraints (optional)
- 🟡 Digital twin: parity, gate, and optional fitted actuator/odometry model implemented; motor/DriveBase/slip dynamics remain (KI-020)

## Additional Documentation

- **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**: Configuration schema, uncertainty ranges, waypoints, CLI
- **[docs/EXAMPLES.md](docs/EXAMPLES.md)**: Usage examples and troubleshooting
- **[docs/API_robot_model.md](docs/API_robot_model.md)**: Robot model and force-model API
- **[docs/API_validator_export.md](docs/API_validator_export.md)**: Validator and export API
- **[docs/WHITEPAPER.md](docs/WHITEPAPER.md)**: Current-state technical description
- **[docs/CALIBRATION_PROCEDURE.md](docs/CALIBRATION_PROCEDURE.md)**: The human procedure for calibrating a drivebase: the setup for each test, the measurement convention, the commands, and the troubleshooting
- **[docs/SYSID_QUICKSTART.md](docs/SYSID_QUICKSTART.md)**: The short path from a session to a fitted model
- **[docs/HEURISTICS_AND_VERIFICATION.md](docs/HEURISTICS_AND_VERIFICATION.md)**: Path-planning heuristics and the verification chain
- **[docs/STRATEGY_LEDGER.md](docs/STRATEGY_LEDGER.md)**: the failed-strategy ledger - every approach that was tried, the measurement that retired it, and what would revive it
- **[docs/RESEARCH_AGENDA.md](docs/RESEARCH_AGENDA.md)**: the research agenda - the open questions, cheapest decisive experiment first
- **[docs/research/pq_b/FINDINGS.md](docs/research/pq_b/FINDINGS.md)**: The research agenda measured on 32 missions instead of six — which of its claims survive, and which do not
- **[docs/ARCHIVE.md](docs/ARCHIVE.md)**: How to read the raw measurements and experiment scripts from the `archive/research-2026-10` tag
- **[docs/research/RQ-A_tracking_error.md](docs/research/RQ-A_tracking_error.md)**: Why the closed-loop tracking gate fails: it is a discretisation mismatch, not a physical tracking error
- **[docs/STUDIO_HUB_PLAN.md](docs/STUDIO_HUB_PLAN.md)**: The "push to hub" feature for the browser studio, and how it ties the missions, the robot code and the sysid loop together
- **[docs/CALIBRATION_STUDIO_PLAN.md](docs/CALIBRATION_STUDIO_PLAN.md)**: How the sysid loop (collect, measure, fit) and the studio's compare step fit together in one browser
- **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)**: Open and fixed issues

## Pybricks Integration

The `pybricks_code/` directory contains the robot-side controller:

- `ramsete.py`: Reference controller (odometry, Ramsete maths, limiter, 20 ms loop)
- `ramsete_fast.py`: List-of-dicts pathing entry point; delegates to `ramsete.py`
- `ramsete_compact.py`, `ramsete_compact_events.py`: Packed-trajectory controllers for `--export-format python-compact`
- `hub_runner.py`: Entry point for a trajectory pushed from the host; it imports the trajectory from `hub_mission.py` and reports the measured pose
- `robot.py`, `robot_config.py`: Hub menu (pushed trajectory / SysId) and hardware definitions
- `sysid_digital_twin.py`: Robot-side calibration collector, with `complex_mission.py` as its replay fixture
- `archive/`: The unsupported demos (`mission_1..5`, `mission_template`, `complete_mission`) and the old telemetry logger

**Note:** an event on the final waypoint *is* fired: the event-capable controllers fire the events that are left after the loop, once the robot has arrived and the drivebase has stopped (KI-012). The controllers that are in use are `ramsete_compact_events.py` (the pushed run) and `ramsete.py` (the IDE missions).

### Pushing to the hub without the IDE

The Pybricks IDE is one client of a documented protocol. `overboard/deploy/hub_deploy.py` and `overboard/main.py --push-hub` are another: they push a program to the hub over
Bluetooth and capture what it prints, so a trajectory can be sent to the
robot and its measured pose read back without a browser.

```bash
python -m overboard.main -w mission.json --push-hub --hub-log session.csv
```

Calibration works the same way. `python -m overboard.sysid.collect` pushes the collector,
captures the session, and writes the measurement sheet; the operator measures
the end pose of the tests that need a tape and fills the sheet in.

```bash
python -m overboard.sysid.collect --conn ble --profile my_drivebase.json \
  --session session.csv --sheet calibration_sheet.csv
```

The sheet is written twice: as the CSV the fitter reads, and as a form in a
browser (`calibration_sheet.html`) that you fill in with the numbers the
tape says. The form downloads the CSV, so the fit takes it as it is:

```bash
python -m overboard.sysid.fit session.csv --sheet calibration_sheet.csv \
  --profile my_drivebase.json --out sysid_fit.json
```

The full human procedure is in
[docs/CALIBRATION_PROCEDURE.md](docs/CALIBRATION_PROCEDURE.md). The fit
reports a 95% interval per parameter, taken over three repeats of each
calibration test, and the validator plans against that interval.

See [pybricks_code/README.md](pybricks_code/README.md) for setup instructions.

## License

This project is for FLL educational use.
