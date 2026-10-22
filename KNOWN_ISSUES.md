# Known Issues

Last reviewed: 2026-10-21

Scope: whole repository, reviewed as a single-user, offline desktop tool for one FLL robot. This is not a multi-user or server-scale review.

Status: KI-001 through KI-003, KI-005 through KI-019, KI-021 through KI-031, KI-033 through KI-044, KI-045, KI-046, KI-047 through KI-053 are **fixed**. KI-032 is **mitigated** (test cache). KI-004 is **deleted** (the Multi-Verse pipeline was removed; archived in git tag `archive/legacy-2026-10`). Open: KI-020 (hardware calibration & digital twin validation), KI-054 (the envelope gate's absolute limit is a duration gate).

How to use this list: refer to an item by its ID, for example "fix KI-003". The **Quick triage** table is the index - read that, then open the entry you are working on. The entries are now short records: what was wrong, what changed, and which files carry the fix. The long narrative each entry used to carry is in the git history - this file before the 2026-10-21 pruning - so `git log -p KNOWN_ISSUES.md` reads any entry in full.

> Raw measurement paths under `docs/measurements/` and `docs/research/rq_a/` are archived in the `archive/research-2026-10` tag. Use `git show archive/research-2026-10:<path>`; see [docs/ARCHIVE.md](docs/ARCHIVE.md).
>
> **Paths.** The one-line records name the files as they are now (`overboard/main.py`, `overboard/verification/immrax_validator.py`, `overboard/sysid/fit.py`, and so on). The two entries kept in full below - KI-020 and KI-032 - were written before the runtime moved into the `overboard/` package, so a bare `optimizer.py`, `immrax_validator.py`, `robot_model.py`, `digital_twin.py`, `main.py` or `sysid_fit.py` in them means the file under `overboard/core/`, `overboard/verification/`, `overboard/models/`, `overboard/sysid/` and `overboard/`. The `pybricks_code/` paths are unchanged. The entries are the record of what was found; the tree is the authority on where the code is.

## Quick triage

| ID | Priority | Area | Issue |
|---|---|---|---|
| KI-001 | P0 | Solver / output | ~~A failed solve is exported as a successful trajectory~~ (fixed) |
| KI-002 | P0 | Validation | ~~`--validate` cannot block output or change the exit code~~ (fixed) |
| KI-003 | P0 | Safety | ~~Immrax ignores upper uncertainty bounds~~ (fixed) |
| KI-004 | P0 | Multi-Verse | ~~Refinement has no effect on the final trajectory~~ (deleted; archived in git tag `archive/legacy-2026-10`) |
| KI-005 | P0 | Live view | ~~The visualizer kills whatever owns port 8765~~ (fixed; the live view is gone) |
| KI-006 | P0 | Install | ~~A fresh install cannot run the default command~~ (fixed) |
| KI-007 | P1 | Tests | ~~The JAX test suite is silently skipped~~ (fixed) |
| KI-008 | P1 | Safety / config | ~~JAX validation reads a different config format~~ (fixed) |
| KI-009 | P1 | Safety / config | ~~Declared uncertainty ranges are ignored~~ (fixed) |
| KI-010 | P1 | Physics | ~~Weight-transfer math is wrong and tools disagree~~ (fixed) |
| KI-011 | P1 | Export | ~~Heading interpolation crosses ±π the long way~~ (fixed) |
| KI-012 | P1 | Robot runtime | ~~An event on the final waypoint never fires~~ (fixed) |
| KI-013 | P1 | Export | ~~Python export loses the robot config~~ (fixed) |
| KI-014 | P1 | Safety model | ~~JAX Ramsete uses a poor large-angle approximation~~ (fixed) |
| KI-015 | P1 | Benchmarks | ~~Random benchmark tools are broken~~ (fixed) |
| KI-016 | P1 | Live view | ~~The first client never receives config or waypoints~~ (fixed; the live view is gone) |
| KI-017 | P2 | CLI | ~~Convergence flags do nothing~~ (fixed; dead flags removed and iteration history forwarded) |
| KI-018 | P2 | Maintenance | ~~Dead code, duplicates, and generated files~~ (fixed) |
| KI-019 | P0 | Safety | ~~The reachability tube lags the reference by one sample~~ (fixed) |
| KI-020 | P1 | Digital twin | Validator/optimizer do not yet model motor, odometry, slip, or reachability-driven optimization |
| KI-021 | P1 | Robot runtime | ~~The DriveBase caps the acceleration the plan asks for~~ (fixed) |
| KI-022 | P1 | Solver / timing | ~~Per-segment timing was not held to the same wheel-jerk discipline~~ (fixed) |
| KI-023 | P1 | Robot runtime | ~~The SysId collector could not be pushed to a hub and its metadata did not parse~~ (fixed) |
| KI-024 | P1 | Solver / timing | ~~The timing model had no rule for where the time allocation may change~~ (fixed) |
| KI-025 | P1 | Safety model | ~~The backlash was subtracted from the commanded wheel speed, and the tracking metric ignored the heading~~ (fixed) |
| KI-026 | P1 | Solver / kinematics | ~~The plan's samples were not on the path its own profile traces~~ (fixed) |
| KI-027 | P1 | Robot runtime / model | ~~The reference interpolated the heading and the turn rate as independent lines~~ (fixed) |
| KI-028 | P1 | Robot runtime / model | ~~The validator modelled a controller that is not the one that runs~~ (fixed) |
| KI-029 | P1 | Robot runtime | ~~A late control tick silently skipped a mission event~~ (fixed) |
| KI-030 | P1 | Studio | ~~The studio told the operator an unchecked trajectory was "approved", and kept the last solve's path on the field~~ (fixed) |
| KI-031 | P1 | Install | ~~A new user could not install the project: the two requirement files contradicted each other, and the first run died with an ImportError~~ (fixed) |
| KI-032 | P1 | Tests | A test cache can hide a regression (mitigated: the fingerprint is wide, only a passed test is skipped, and the summary says what was skipped) |
| KI-033 | P1 | Robot runtime | ~~The host could not find the hub over Bluetooth, while the Pybricks IDE could~~ (fixed) |
| KI-034 | P1 | Solver / validation | ~~The local envelope limit landed on the wrong segment, and the last waypoint's limit was never read~~ (fixed) |
| KI-035 | P1 | Solver / safety model | ~~The wheel-speed limit ignored the gearing~~ (fixed) |
| KI-036 | P1 | Studio / safety | ~~The studio pushed whatever trajectory it was given~~ (fixed) |
| KI-037 | P1 | Studio | ~~The studio kept every job forever, and a cancel left the solver running~~ (fixed) |
| KI-038 | P1 | Export | ~~An event name with a quote in it broke the exported Python file~~ (fixed) |
| KI-039 | P2 | Studio / solver | ~~Two mission names became one file, a restore wrote before it parsed, and the guess averaged two headings the long way~~ (fixed) |
| KI-040 | P1 | Robot runtime | ~~`--push-hub --hub-log` lost the whole hub log when the link dropped~~ (fixed) |
| KI-041 | P1 | SysId | ~~One corrupt cell in a session poisoned the whole calibration~~ (fixed) |
| KI-042 | P1 | SysId | ~~A two-repeat calibration could not be loaded at all~~ (fixed) |
| KI-043 | P1 | SysId | ~~The collector calibrated the odometry from the robot's own odometry~~ (fixed) |
| KI-044 | P1 | Studio | ~~The studio kept the last run's verdict, and a reload could not cancel a solve~~ (fixed) |
| KI-045 | P2 | Cleanup | ~~`sysid_solver.py` is a superseded prototype~~ (fixed; retired prototype and dropped pandas) |
| KI-046 | P2 | Cleanup | ~~Five smaller things the audit left open~~ (fixed; all five resolved) |
| KI-047 | P1 | Solver | ~~The fast solve path made the solver slower, and its test failed on the tree~~ (fixed) |
| KI-048 | P1 | Solver / output | ~~The solver gives up on a path instead of trying another candidate~~ (the rescue round that fixed it is removed: it rescued 0 of 22 failing missions) |
| KI-049 | P1 | Validation | ~~The reachability envelope reported a 54 mm width that was the divergence guard's fingerprint, and the tube behind it was blown up by a dependency the interval arithmetic could not see~~ (fixed) |
| KI-050 | P1 | Validation | ~~The closed-loop tick map could not run at all on the installed JAX: the controller unpacked the pose, and Immrax's natural inclusion has no rule for `unstack`~~ (fixed) |
| KI-051 | P2 | CLI / solver | ~~`--self-consistent` is a no-op: the switch cannot change the plan and the trapezoidal collocation cannot be reached from the CLI~~ (fixed; switch is two-way `--self-consistent/--no-self-consistent` and default True) |
| KI-052 | P2 | CLI / solver | ~~The CLI and the library plan different trajectories for the same mission: the CLI was self-consistent by default, the library was not~~ (fixed; both default to the self-consistent rule now) |
| KI-053 | P2 | Physics / safety model | ~~The interval force model passed an interval through unchanged, so a signed velocity could produce a negative force bound~~ (fixed) |
| KI-054 | P1 | Validation | The envelope gate compares an absolute radius against a tube that grows with path length, so a long path fails for being long and a failure usually has no remedy but a better model |

## Open

### KI-020 — Validator is not yet a full digital twin or an optimizer objective

**Status:** Open, partially addressed in the working tree. The controller parity, linear reference, 20 ms sample-and-hold, and physical backlash model are implemented. The controller split is also done: `robot_config.py` is hardware-only, `mission_4_ramsete.py` imports `ramsete_fast` with its calibrated defaults, and `mission_5_lambdas.py` imports `ramsete` for events. `pybricks_code/sysid_digital_twin.py` and `sysid_fit.py` provide the robot log/capture and host fit path; no hardware logs have been collected yet. The host can now push a script to the hub and capture its console output without the Pybricks IDE (`hub_deploy.py`, `main.py --push-hub`), so the robot log no longer has to be copied out by hand, and a pushed trajectory reports its measured final pose as a `POSE` line. The fitted numbers now carry an uncertainty: the calibration-critical tests run three times each, the fit reports a 95% t-interval per parameter over those repeats, and the validator plans against that interval instead of the spread of the steps inside one run, which was not an uncertainty because the steps are sequential. The interval covers the run-to-run variation and any drift over the session, but not the error of the tape measure: the tape error is in the numerator of every measurement, it is the same every time, and no number of repeats averages it out. Two parameters that the robot config could only declare are now measured as well: the straight-line heading drift, which the plant adds to the heading rate because the plan's straight segments are straight, and the motor's stall torque, which the wall test measures and the validator uses as the motor force limit. **The motor is now measured on both ends and every tool uses the measured pair.** The collector's new `top_speed` test measures the free-running speed with the wheels off the ground, which is the other end of the torque-speed line; the fit rejects the number when the robot reached the commanded speed instead, because then the command was the limit and not the motor. `main.py --sysid-fit` writes the measured pair into the robot config, so the optimizer, the CPU audit, the validator and the exported controller all plan against the same motor (the KI-010 rule). Before this, the optimizer planned against the declared motor while the validator checked the measured one, so a plan could satisfy its own constraint and still trip the validator. The drivetrain model itself is in `drivetrain.py`, and its structure was checked against an independent rigid-body simulator: the top speed agrees within 7%, the coast deceleration is 2.4x higher in the simulator because the model has no rolling resistance, and that simulator's brake mode turns out to reverse the wheels instead of braking them, so its braking distance is not a usable reference. See `docs/research/RQ-B_drivetrain_model.md`. The drift is a slope against the speed, not a subtraction, so the gyro's bias lands in the intercept instead of leaking into the drift; the tape's lateral offset is the independent second estimate. The step fit is a two-stage search, so the reported time constant and delay are no longer quantised to the 10 ms grid of the first pass. The wall is also a reference for the distance scale: the robot drives into it from a marked start and stops when it is blocked, so the scale can be checked against something that does not move and that nobody has to read, which is what makes it possible to re-check the calibration unattended. The wall and the tape are two estimates of the same number and the fit reports them side by side. The wall cannot measure the heading, because a wall does not straighten a crooked robot, so the drift still comes from the speed regression. The host tools accept a minimal drivebase profile (`wheel_radius_m`, optional `track_width_m`) instead of a full robot config. `digital_twin.py` and `main.py --sysid-fit` now load and validate a fit. When enabled, the validator uses a five-state plant with actual wheel speeds, the fitted straight-line gain interval, first-order time constant, command delay, and odometry distance/heading scale in feedback. The nominal feedback path does **not** round the drive command: KI-028 moved the firmware to unrounded floats and the validator to mirror it, so the model and the robot agree (the rounding model is still there behind `quantize=True`, for an A/B measurement). Braking slip is now applied by the plant. Reachability is a post-solve candidate score (`--candidates`), not yet an optimizer term. The firmware's DriveBase trajectory limits are now known (about 7800 mm/s² and 9300 deg/s² for this robot, derived from Pybricks' 20,000 deg/s² ceiling), so the plant's instant-tracking assumption matches the hub; Motor/current transients, DriveBase PID, and slip under load remain unimplemented; the whole-millimeter distance reading was measured rather than assumed and costs 0.75 mm of final drift on the 6.08 s complex mission, so it is not worth modelling. The machinery for the rest is built and inert until the fit provides the numbers: the controller is fed the robot's own integrated odometry belief instead of the truth, the plant applies the measured braking shortfall and the measured acceleration limit, and the belief drifts with the measured gyro bias. `docs/DIGITAL_TWIN_PLAN.md` records the data contract, what is built, and what is still open. The measurement loop around them is now complete on the host: `main.py --push-hub --hub-log session.log` captures the hub console output, `sysid_fit.py --from-hub-session session.log --out-poses poses.csv` turns its `POSE` lines into the measured-pose CSV that `sysid_compare.py` reads, and the sysid battery feeds `sysid_fit.py` the same way. The measured poses are matched to the sysid log by `run_id`, and only `odometry_straight`, `odometry_turn` and `brake_from_*` runs contribute a distance scale, a heading scale or a braking-slip estimate, so a poses file with the wrong run id is silently ignored ("no measured poses supplied"). A documented limit needs no calibration: the default `examples/fll_choreo.json` declares the DriveBase's 7.8 m/s² acceleration ceiling, and a declared limit alone selects the ramped five-state plant, so a plan the hub cannot accelerate along now fails the gate (Q11 step 1, `tests/test_digital_twin.py`).

**Files:** `optimizer.py`, `immrax_validator.py`, `jax_ramsete.py`, `digital_twin.py`, `drivetrain.py`, `robot_model.py`, `main.py`, `multiverse_optimizer.py`, `pybricks_code/ramsete_fast.py`, `pybricks_code/robot_config.py`, `pybricks_code/mission_4_ramsete.py`, `pybricks_code/mission_5_lambdas.py`, `pybricks_code/sysid_digital_twin.py`, `pybricks_code/hub_runner.py`, `pybricks_code/ramsete_compact_events.py`, `sysid_fit.py`, `sysid_compare.py`, `sysid_collect.py`, `hub_deploy.py`, `examples/drivebase_profile.example.json`, `tests/test_sysid_fit.py`, `tests/test_sysid_compare.py`, `tests/test_digital_twin.py`, `tests/test_drivetrain.py`, `tests/test_hub_deploy.py`, `tests/test_sysid_collect.py`

**Problem:** The default validator mirrors the deployed controller's math and timing, but its plant is still kinematic: commanded wheel speeds become motion immediately (apart from wheel backlash). The optional fitted plant adds actuator lag, delay, gain uncertainty, and a static odometry correction, but it still has no motor current/torque transient, DriveBase PID, integrated encoder/gyro odometry, or wheel-slip dynamics. Feedback uses a scaled version of the true pose, not the robot's actual `DriveBase.state()` estimate. Reachability is a post-solve candidate score (`--candidates`, `main.py`), not yet a term or constraint in the optimizer.

**Evidence:** The straight example passes with `max_tracking_error_m≈0.00346 m` and `max_envelope_radius_m≈0.0028 m` under the narrowed test ranges. Representative paths still expose the gap: the S-curve example reports `max_tracking_error_m≈0.0132 m` and `max_envelope_radius_m≈0.0077 m`; the stops example reports 0.0052-0.0060 m / 0.0093-0.0095 m; the events example reports 0.0053-0.0054 m / 0.0097 m; the complex mission now passes at 6.068 s with 0.0049 m / 0.0122 m. These are model predictions, not hardware measurements. There is no real tracking log in the repository yet; the logger and fitter have only been exercised with synthetic data. A synthetic fit with `gain=1.0, tau=0.08 s` already rejects the straight example with 3.44 cm tracking error, which shows why hardware calibration matters before enabling the fitted model. The old `accuracy_weight += 2.0` retry is removed: on the complex mission the slow 17.1 s candidate diverged, while the 6.07 s one stayed bounded. A uniform-timing five-seed sweep (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`) shows that on the complex mission all four non-baseline seeds converge to slower, wider paths (8.1-8.7 s, 1.55-1.76 cm, tracking 0.53-0.72 cm), so the 6.07 s / 1.22 cm baseline still wins. Guess diversity alone does not beat the equal-duration baseline; the default simple-path candidates now mix the two timing modes (KI-022). Per-segment timing is path-dependent: on an uneven 1.0 m + 0.2 m path, `accuracy_weight=0.005` passes all gates at 2.784 s / 0.32 cm envelope / 0.36 cm tracking versus 4.456 s / 0.37 cm / 0.36 cm equal-duration at the same weight, so the mixed loop selects per-segment; at `accuracy_weight=0` per-segment is 2.710 s / 0.32 cm but fails the Immrax and CPU gates, so the mixed loop falls back to the equal-duration candidate (4.463 s / 0.35 cm). On the complex mission at `accuracy_weight=0` per-segment is faster (4.650 s / 1.00 cm) but its tracking is 0.84 cm and its motor force is 0.074 N over the limit; at `accuracy_weight=0.02` it is 7.333 s / 1.58 cm with 0.53 cm tracking, still over the 5 mm gate, and worse than the uniform 6.068 s / 1.22 cm baseline. The candidate loop now mixes both timing modes by default (`--uniform-dt`/`--per-segment-dt` force one) and runs the CPU audit on every candidate, so a partial pass cannot block a later fully passing candidate.

**Fix:** Collect and fit in one command with `python -m overboard.sysid.collect --conn ble --profile examples/drivebase_profile.example.json`: it pushes the logger, captures the session as it arrives, converts the hub's `POSE` lines to `poses.csv`, and writes `sysid_fit.json`. The poses are the hub's own odometry, so the distance and heading scale of the fit have no ground truth and stay at 1.0 until someone measures the robot with a tape and replaces those rows; the actuator part of the fit does not need a measurement. The written procedure is `docs/CALIBRATION_PROCEDURE.md`: the operator drives the robot through the collector tests, measures the end pose of the five that need a tape, and fills in a measurement sheet keyed by test name. `python -m overboard.sysid.fit <session.csv> --from-sheet <sheet> --out-poses poses.csv` turns that sheet into the measured poses, so the order the tests ran in does not matter and a skipped test is reported instead of guessed. The sheet is also fitted directly (`--sheet`), so there is no intermediate pose file, and it is written as a form in a browser (`--write-form`, `sheet_form.py`) so the operator types the numbers instead of editing a spreadsheet; the studio has the same form on the server. The same thing step by step: push the logger and capture its session with `python -m overboard.deploy.hub_deploy pybricks_code/sysid_digital_twin.py --out session.csv` (or run it from the Pybricks IDE), record the stop/final poses in `poses.csv`, fit the logs with `sysid_fit.py --profile examples/drivebase_profile.example.json`, and pass the result with `main.py --sysid-fit sysid_fit.json --validate`. Compare the planned trajectory with the measured poses using `sysid_compare.py`; it prints position/heading errors and whether each measured pose is inside the reachability envelope, and it accepts a plain predicted-pose CSV and a drivebase profile. The pushed run gives the same comparison without hand-editing: `main.py --push-hub --hub-log session.log`, then `sysid_fit.py --from-hub-session session.log --out-poses poses.csv --run-id <id>`, then `sysid_compare.py output.traj poses.csv --run-id <id>`. Extend the plant to integrated odometry and a braking-slip state only after that comparison. The post-solve choice is made: `--candidates` scores several solutions (seed/timing combinations with `--simple`, accuracy-weight variants on the Multi-Verse path) and picks the passing one with the lowest envelope (then the shortest). Extend it to a hard constraint or differentiable proxy only if the candidate set cannot find a passing path. Do not tune the 5 mm/2 cm thresholds or b/zeta to one path.

**Workaround:** Use the straight example as a parity smoke test; treat predictions for curved/high-speed paths as uncalibrated until hardware logs are available. `examples/fll_choreo.json` now declares physical-looking ranges instead of the old clamped ones (track width ±0.5 mm, wheel radius ±0.5 mm, backlash 0.2-0.8 mm, COF 0.95 ±10 %), but they are still assumptions, not measured hardware values - run the SysId loop before calling them calibrated. Because a wide range makes the open-loop `plant` tube exceed the 2 cm gate on a curved path, validation of the examples needs `--propagation tick`; the `safe` preset (CLI and studio) selects it.

### KI-054 — The envelope gate is an absolute radius on a cumulative tube

**Status:** Open. The limit is configurable (`--envelope-limit`) and can grow with distance (`--envelope-limit-per-metre`, default 0, so the absolute verdict is unchanged), and a failing run can be inspected (`--best-effort`); a pass still means "inside the declared uncertainty for the whole run".

**Files:** `overboard/main.py` (`_segment_limit`, `_score_attempt`, `_segment_report`, `_emit_validation_report`), `overboard/gui_server.py`, `overboard/viz/studio.html`, `overboard/viz/studio.js`, `docs/HEURISTICS_AND_VERIFICATION.md`, `tests/test_cli_ux.py`, `tests/test_gui_server.py`

**Problem:** The gate compares each sample's reachability radius with 2 cm (`DEFAULT_ENVELOPE_LIMIT_M`). Under the default `plant` propagation the tube is open-loop, so it widens with elapsed time and travelled distance: the limit is a duration budget in disguise. The same control law that passes at 2 s fails at 4 s, not because it is less accurate per metre but because the tube had more time to spread. The default is also not a mission requirement - it is the number that applies when the operator declared nothing.

**Evidence:** S-curve example with the widened `fll_choreo` ranges, `--validate --candidates 1 -n 10`, `plant`: the first segment reaches 0.99 cm at 1.9 s (passes), the second 4.81 cm at 3.7 s (fails), with the same geometry in both halves; tracking is 1.01 cm and over its 0.5 cm gate too. All six candidates - the front triad and the rescue seeds - failed, so the run exited 1 and wrote nothing. Across the 204-candidate matrix the envelope correlates with duration at r = +0.97 (`docs/RESEARCH_AGENDA.md`, the selection-rule answer).

**Why the default stays absolute:** the error budget is physical, not proportional. 4.8 cm off after 3.7 s is 4.8 cm from the mission model; dividing that by the path length would forgive exactly the error that misses a mission. The per-metre slope is therefore a declaration the operator makes about a long mission ("this much growth is acceptable"), not an automatic rescaling of a measured error. The ranges behind the tube are declared, not measured (KI-020), so a threshold chosen until a path passes is tuned against a model that is not the robot. The propagation also cuts the tube off at 5 cm (`diverged`, KI-049), so no limit above 5 cm can be measured at all - that is why the knobs stop there.

**Fix (what exists now):**

- `--envelope-limit METRES` (default 0.02, range 0.001-0.05) sets the run-wide default. A waypoint's `error_threshold` still overrides it, and `precision_mode` still pins 0.5 cm, so the local declaration always wins.
- `--envelope-limit-per-metre M/M` (default 0, max 0.05) adds that allowance for every metre travelled, on top of the local base. Off by default, so existing verdicts do not change; on, a tube that widens with distance is judged at a rate instead of a fixed radius. The per-sample limit is `base + rate * distance`, and the report judges a segment at the sample with the biggest excess.
- The value in force is recorded, so a verdict is never read without the budget it used: `envelope_limit_m` and `envelope_limit_per_metre` in the `##OVERBOARD-VALIDATION##` payload, `limit_m` and `excess_m` per segment, and `validation: {passed: true, envelope_limit_m, envelope_limit_per_metre}` in the `.traj` root and `_stats.json`. A `--best-effort` file carries `{passed: false, best_effort: true, max_excess_m, reasons, segments}` instead.
- The studio has an **Accuracy limit (cm)** field, disabled unless validation is on; it forwards `--envelope-limit` only with `--validate`, and the server rejects a value outside 0.1-5 cm.
- `--best-effort` shows that path marked failed, so an accuracy excess no longer means an empty field.

**Workaround - what to do when it fails.** There are four levers, and none of them is in the solver:

1. **Set the limit to the mission's real tolerance** - how far the robot may be off before it misses the model it is going to touch. That number comes from the field, not from what makes the run pass. If the tolerance grows along a long run, say so with `--envelope-limit-per-metre` instead of raising the flat limit.
2. **Declare `error_threshold` on the waypoints that need it**, instead of one number for the whole path.
3. **Use `--propagation tick`**: the closed-loop tube contracts where the open-loop one spreads, which is a model fix rather than a budget change (2-10x slower).
4. **Measure the robot** (KI-020): narrower declared ranges shrink the tube everywhere. This is the only lever that makes the prediction truer.

If none of those is available, the run does not know that the plan is accurate enough, and `--best-effort` is for inspection, not approval.

**The smarter rule, not built:** compare the tube against the mission's tolerance per sample - the clearance to the nearest mission model or wall - which needs a field model the planner does not have. Until then a per-waypoint `error_threshold` is the manual form of that rule, and the 2 cm default is a placeholder, not a requirement.

## Mitigated

### KI-032 — A test cache can hide a regression

**Status:** Mitigated in the working tree (uncommitted). The cache is on by default and its rules live in `testcache.py`; `--no-cache` runs everything.

**Files:** `testcache.py`, `run_tests.py`, `tests/test_testcache.py`, `.gitignore`

**Problem:** The suite took about two minutes, and nearly all of it went into tests whose code had not changed since the last run. Re-running everything on every change is the wrong trade for a tree this size.

**The risk:** a cache says "this test does not have to run again". If it says that for a test whose code *did* change, a regression walks past a green suite. The cache is a claim about what a test depends on, and a wrong claim is a hole nobody sees.

**Fix (the rules, all in `testcache.py`):**

- The fingerprint of a test covers the test module, the transitive import closure of the project modules it imports, every file it opens, every directory it lists, every project script it hands to a subprocess, every project module it imports while it runs, and the environment: the Python version and the versions of numpy, casadi, jax, jaxlib, immrax, scipy and the rest.
- **Only a test that passed is skipped.** A failure always runs again.
- **A test that wrote a file inside the repository is never skipped**, because another test may read what it wrote.
- A file that is gone counts as changed, and a directory is fingerprinted by its listing, so a file added to it invalidates the tests that listed it.
- The summary says how many tests the cache skipped and how much runtime that saved, so a green run never hides a skipped test. `--cache-status` prints the plan, `--no-cache` ignores the cache, `--clear-cache` throws it away and `--durations N` lists the slowest tests.

**What it cannot see:** a file a test reads only on a path it did not take in the recorded run, and a file that only a *subprocess* reads, when that subprocess was neither named on the command line nor named in the code the test handed it. Those are the holes. They are the reason the fingerprint is deliberately wide and the rules above are conservative.

**Evidence:** Measured on 2026-10-20, a cold full run took 251 s (`run_tests.py` prints the count; this entry does not quote one). A second run over the same tree takes 5 s: 671 of the tests are skipped, and the 16 that run are the ones that must not be skipped because they write into the repository (the doctor's probe, the deprecated-flag CLI runs, the hub and sysid session writers, and the cache's own tests). The saving is reported in the summary, so the number is never a guess, and a third run takes the same 5 s: the cache does not invalidate itself.

**Workaround:** `python run_tests.py --no-cache` for the full suite. The cache is a development convenience; it is not a gate, and no release decision rests on a cached run.

## Deleted

### KI-004 - Multi-Verse refinement has no effect

**Deleted.** The pipeline was repaired and measured as a **seed provider**, but the measurements showed it rescues nothing and is not faster than the stock seeds. The code was removed from the tree; it is preserved in git tag `archive/legacy-2026-10`.

**Was:** the refined guess always lost - 8.1473 s against the baseline guess's 6.0677 s on the complex mission - and the measurement found four separate causes: the refinement moved the waypoint in the middle of its two-segment window, the cost graded a different motion than IPOPT produces, the intervals were hard-coded, and the refined timing was discarded. Measured over 30 missions as a *seed provider*, the refined seed is **not** better than the stock seeds (faster on 2 of the 9 missions where both pass, slower on 6) and it **rescues nothing**; it is a cheaper way to reach the stock set's speed on some missions.

**Next lever:** the objective, because the pipeline's cost buys speed with tracking accuracy and the 5 mm gate rejects the trade.

**Files:** (removed from the tree; see git tag `archive/legacy-2026-10`)

## Fixed

### KI-001 — A failed solve is exported as a successful trajectory

**Fixed.** Failed solves exit non-zero and write nothing; a pre-existing file at the output path is left untouched.

**Was:** IPOPT failure returned the solver's last debug values with `converged=False`, and the CLI wrote them out and exited 0 - a solve to (100, 0, 0) ended near x=20 and still passed the internal checks, so the validator reported PASS.

**Fix:** the CLI checks `stats["converged"]` before any output, `_select_attempt` excludes non-converged candidates, the final pose is checked against the last waypoint, and `forward_integrate` rejects non-positive timestamps instead of crashing.

**Files:** `overboard/main.py` (candidate loop, export gate), `overboard/verification/validator.py`, `tests/test_regressions.py` (`TestExportGate`, `TestWriteGateIntegration`, `TestValidatorGates`)

### KI-002 — `--validate` cannot block output or change the exit code

**Fixed.** `--validate` gates every output and the exit code; a failed validation writes no trajectory, controller, Python file or stats file.

**Was:** `passed_validation` was assigned and never read, a thrown Immrax error set it to `True`, and the straight example printed "FAILED: Physical constraints" and still wrote the output.

**Fix:** the gate runs on the selected candidate, the in-memory CPU validation and the final waypoint *before* the file is created, prints the specific reason and exits 1. The benchmark stats file is written only after the gate.

**Files:** `overboard/main.py`, `overboard/verification/validator.py`

### KI-003 — Immrax ignores upper uncertainty bounds

**Fixed.** `compute_reachability` passes `Interval(w_lower, w_upper)`, and a regression test confirms that widening only `w_upper` widens the envelope.

**Was:** the uncertainty was handed to the embedding system as a plain array, and Immrax's natural inclusion treats a non-`Interval` argument as a constant, so only the lower bounds reached the reachable set - torque, backlash, track width, wheel radius and inertia were never checked in the upper direction.

**Fix:** pass the interval; if the wheel command should be uncertain too, pass `u` as an `Interval`.

**Files:** `overboard/verification/immrax_validator.py`, `tests/test_regressions.py`

### KI-005 — The live visualizer kills whatever owns port 8765

**Fixed.** Closed by removal: the browser studio is the only browser UI, and it takes the next free port instead of killing anything.

**Was:** `LiveVisualizer.start()` ran `netstat` + `taskkill` on Windows (or `fuser -k` on POSIX) against whatever held port 8765, without checking who owned it.

**Fix:** the port is probed before binding, so a busy port raises a clear "already in use" error. The live view was then retired: `--live` opens the studio instead.

**Files:** `live_visualizer.py` (deleted), `overboard/main.py`, `overboard/gui_server.py`

### KI-006 — A fresh install cannot run the default command

**Fixed.** The CasADi/IPOPT backend needs only `requirements.txt`; `--validate` and the optional stack live in `requirements-optional.txt`.

**Was:** `requirements.txt` omitted jax, jaxopt, immrax and pandas, and the CLI defaulted to the JAX-only Multi-Verse optimizer, so a fresh install died on an ImportError.

**Fix:** `python -m overboard.main --doctor` reports what is installed and prints the exact install command for anything missing.

**Files:** `requirements.txt`, `requirements-optional.txt`, `overboard/main.py`, `overboard/diagnostics.py`, `README.md`

### KI-007 — The JAX test suite is silently skipped

**Fixed.** The suite runs every JAX module. The two stale test modules were deleted rather than the production classes changed to satisfy them.

**Was:** all JAX imports sat in one `try`, and two modules imported classes that no longer exist (`MultiVerseRefiner`, `LocalSegmentOptimizer`), so the suite reported 39 passing tests and printed "jax not installed" while JAX was installed.

**Fix:** `tests/test_stomp_variants.py` and `tests/test_teb_topologies.py` are gone; `run_tests.py` imports the JAX modules in a `try` and says what it skipped.

**Files:** `run_tests.py`, `tests/test_regressions.py`

### KI-008 — JAX validation reads a different config format

**Fixed.** `JAXRobotConfig` delegates parsing to `RobotConfig`, and the CLI passes the already-parsed config, so legacy and canonical configs describe the same robot.

**Was:** the JAX config read only `robot` or a flat dictionary with canonical keys, so the legacy `config` + `{val}` format silently fell back to defaults and Immrax validated a different robot than the optimizer planned for.

**Fix:** one parser for both; a test uses the README sample config.

**Files:** `overboard/models/jax_robot_model.py`, `overboard/models/robot_model.py`, `overboard/main.py`

### KI-009 — Declared uncertainty ranges are ignored

**Fixed.** `RobotConfig` normalizes `robot.uncertainties`, and the CLI passes the declared ranges to Immrax, falling back to the old constants only when a range is absent. The optimizer uses the declared headroom and wheel-radius worst case too.

**Was:** the CLI hardcoded `cof ±20 %`, `torque=(0.8, 1.0)` and `backlash=(0.0001, 0.0006)`, while `examples/fll_choreo.json` declared `cof [1.24, 1.51]`, `torque [0.75, 0.9]` and separate track-width, wheel-radius and inertia ranges. The validator checked different numbers than the robot file said.

**Fix:** read the declared ranges and pass the same ones to Immrax and the report.

**Files:** `overboard/models/robot_model.py`, `overboard/main.py`, `examples/fll_choreo.json`

### KI-010 — Weight-transfer math is wrong and tools disagree

**Fixed.** One shared `wheel_normal_forces` (lateral transfer only, no front/rear split) is used by the CPU model, the JAX model, Immrax and the optimizer, and the optimizer constrains the same worst-case per-wheel force Immrax checks.

**Was:** the longitudinal term was subtracted from *both* wheels, so `nl + nr` fell below `mg` under acceleration, and the optimizer limited the total `μmg` while the validators checked each wheel. On the complex mission the fast path reported motor 0.229 N and slip 1.197 N before the fix, and 0.0000 N after.

**Fix:** `nl = mg/2 - lateral`, `nr = mg/2 + lateral`, and the optimizer's per-wheel bound is `f_bound <= max_fl`, `f_bound <= max_fr`, `f_bound <= cof_lo * (mg/2 - |lateral|_max)`, with `f_bound` the worst-case wheel force over the declared inertia and track-width ranges.

**Files:** `overboard/models/drivetrain_physics.py` (`wheel_normal_forces`), `overboard/models/robot_model.py` (`get_wheel_normal_forces`, `get_max_wheel_force_symbolic`, `get_traction_limit_symbolic`), `overboard/core/optimizer.py`, `overboard/models/jax_robot_model.py`, `overboard/verification/immrax_validator.py`

### KI-011 — Heading interpolation crosses ±π the long way

**Fixed.** `resample_to_fixed_dt` unwraps the source heading before interpolating and wraps the result back to [-π, π].

**Was:** headings 3.13 and -3.13 - a 0.0232 rad turn across ±π - were interpolated as a π turn in the wrong direction. The optimizer's own headings are continuous, so only a hand-edited or external trajectory triggered it.

**Fix:** `np.unwrap` before, wrap after. A consumer that computes heading *differences* must use the shortest angle, as the deployed controllers do.

**Files:** `overboard/deploy/export.py` (`resample_to_fixed_dt`, `_wrap_pi`), `tests/test_export.py`

### KI-012 — An event on the final waypoint never fires

**Fixed.** The event-capable controllers fire the final event after the loop, once the robot has arrived and the drivebase has stopped.

**Was:** the loop broke at `t >= total_time` and events were checked only on the current sample, so the last sample's event was never looked at - and the optimizer puts the final waypoint on the last sample, so a final "release" or "score" action was silently skipped.

**Fix:** one `_fire_event` helper in `pybricks_code/ramsete.py`, shared by the loop and the end of the run; the compact controller fires the events that are left after the loop. Both fire on the normal exit path only.

**Files:** `pybricks_code/ramsete.py` (`_fire_event`, `follow_trajectory`), `pybricks_code/ramsete_compact_events.py`, `tests/test_firmware_events.py`

### KI-013 — Python export loses the robot config

**Fixed.** The generated module carries the canonical robot fields, read through `RobotConfig`.

**Was:** the CLI wrote `config_data.get("robot", config_data.get("config", {}))`, which is `{}` for a flat config, and the exporter read only old key names such as `differentialTrackWidth` and `vmax` - so a flat config exported `"config": {}` and a canonical one exported the default track width, wheel radius, speed and torque.

**Fix:** write the canonical robot dict (or the flat dict when there is no wrapper) and read it through `RobotConfig`. Regenerate the export rather than hand-editing it.

**Files:** `overboard/main.py` (trajectory config), `overboard/deploy/export.py` (`write_python_file`), `tests/test_regressions.py` (`TestExportConfig`)

### KI-014 — JAX Ramsete uses a poor large-angle approximation

**Fixed.** The CPU/parity path normalizes the heading error to [-π, π]; the interval path keeps unrolled headings so the error is already continuous. Both use a degree-6 Chebyshev polynomial for `sinc` with a maximum error below 2.5e-9 on [-π, π], which Immrax can evaluate.

**Was:** the controller replaced `sin(etheta)/etheta` with `1 - etheta²/6 + etheta⁴/120`, which is only accurate near zero: at `etheta = π` the true value is 0 and the approximation 0.167, and at 2π it goes negative. Immrax used that controller inside its closed-loop reachability check.

**Fix:** the firmware's formula is the source of truth (exact sinc, shortest-angle error, the interpolated gain); the interval path uses the polynomial because Immrax's natural inclusion implements no `abs`, `where`, `mod`, `sinc` or `atan2`. A randomized parity test compares the JAX command with a direct transcription of the firmware's maths, including ±π crossings.

**Files:** `overboard/models/jax_ramsete.py`, `tests/test_jax_ramsete.py`

### KI-015 — Random benchmark tools are broken

**Fixed.** All four benchmark modes run, the generator solves every mission, and a test keeps it that way.

**Was:** the random benchmark validated against the trajectory file as if it were a config, so the safety numbers described the default robot; the generator read a `robot_config.json` that is not in the repository and treated `solve()`'s `(samples, stats)` tuple as the samples; and the generator's `samples` parameter was shadowed by the solve result, so the *second* mission solved with the first mission's sample list as the sample count and the optimizer died on `num_samples_per_segment + 1`. A one-mission run never showed it.

**Fix:** the benchmark writes the config it used to `benchmarks/config_used.json` and validates against that file, the generator unpacks the tuple (the result is now called `trajectory`), and the missions are drawn as a random walk inside the profile's field rather than as independent uniform points.

**Files:** `benchmark.py`, `examples/generate_random_suite.py`, `tests/test_random_suite.py`

### KI-016 — The first live-view client never receives config or waypoints

**Fixed.** Closed by removal: the studio loads the mission itself, so there is no first-client race.

**Was:** `send_config` ran immediately after the server started, and `broadcast` returned when there were no clients, so the first browser saw an empty field - and the README told users to open the browser afterwards.

**Fix:** the last config, waypoints and trajectory were replayed to every client that connected, and the live view was then retired with KI-005.

**Files:** `live_visualizer.py` (deleted), `viz/index.html` (deleted), `overboard/main.py`

### KI-017 — Convergence flags do nothing

**Fixed.** The dead flags are removed and the iteration history is captured and forwarded.

**Was:** `--convergence-output` and `--convergence-animate` were parsed and never used, and the optimizer declared `iteration_history` but never filled it.

**Fix:** `--iteration-history` is the live flag; the optimizer fills the history when it is asked.

**Files:** `overboard/main.py`, `overboard/core/optimizer.py`

### KI-018 — Dead code, duplicates, and generated files

**Fixed.** The dead code, the duplicate mission copy and the unused helpers are gone, and the decision on each is written down.

**Was:** a duplicated `examples/complete_mission.py`, a root-level test module, `docs/API_optimizer.md`, generated outputs and unused imports. The two mission copies the firmware imports were five months old and carried the *old* robot (`t_max_nm 0.040` against the measured 0.138), so the robot limited itself against a motor more than three times too weak.

**Fix:** the firmware copies are regenerated from the current planner and config (`complex_mission.py` passes the gates at 6.068 s / 1.19 cm; `complete_mission.py` is a demonstration of the lambda event API, exported without `--validate`); `get_safety_ribbon_data` and three other unused functions are removed; `ramsete_fast.py` and `get_dynamics_symbolic` are kept on purpose, and so are the pytree hooks and the `BaseHTTPRequestHandler` hooks.

**Files:** `pybricks_code/complex_mission.py`, `pybricks_code/complete_mission.py`, `overboard/verification/immrax_validator.py`, `overboard/models/robot_model.py`, `overboard/sysid/digital_twin.py`, `overboard/sysid/fit.py`, `overboard/sysid/compare.py`, `.gitignore`

### KI-019 — The reachability tube lags the reference by one sample

**Fixed.** Each interval is driven by the same linear reference and 20 ms sample-and-hold loop as the firmware, with the plant integrated in 5 ms substeps. Tracking error and envelope radius are reported separately and checked independently; the envelope is checked against the 2 cm transit threshold.

**Was:** the tube was propagated with sample `i`'s reference pose and velocity and compared to sample `i+1`, so it ran one sample behind; with every uncertainty collapsed to nominal the model still reported `max_tracking_error_m = 0.1585` against a 5 mm threshold, while the envelope radius was 1.5-2 mm.

**Fix:** linear pose/velocity interpolation and the precomputed Ramsete gain, the reference and controller updated once per 20 ms tick, the command held, and the plant integrated in 5 ms substeps. The thresholds were not relaxed to hide the model error.

**Files:** `overboard/verification/immrax_validator.py` (`compute_reachability`, `_interpolated_reference`, `RamseteController`), `overboard/models/jax_ramsete.py`

### KI-021 — The DriveBase silently caps the acceleration the plan asks for

**Fixed.** The firmware raises the DriveBase trajectory limits to 80 % of the firmware ceiling for this robot's geometry - about 7800 mm/s² and 9300 deg/s² with 56 mm wheels and a 96.5 mm track.

**Was:** `DriveBase.drive()` starts a timed trajectory for the distance controller and another for the heading controller, and both ramp at `settings.acceleration`, which Pybricks defaults to 1500 deg/s² (about 733 mm/s²). The inner reference could therefore move at most 0.015 m/s per 20 ms tick, while the optimized missions reach 5.38 m/s² and 8.32 k deg/s². On top of that, `drivebase.state()` rounds the heading to whole degrees even with the gyro on, and Pybricks truncates float arguments toward zero, so small steering corrections became zero commands.

**Fix:** `robot_config.py` calls `drivebase.settings(straight_acceleration=..., turn_acceleration=...)` with values derived from the wheel diameter and track width, the controllers read `drivebase.angle()` and round the `drive()` arguments so the truncation bias is gone, and the JAX twin mirrors that rounding.

**Files:** `pybricks_code/robot_config.py`, `pybricks_code/ramsete.py`, `pybricks_code/ramsete_compact.py`, `overboard/models/jax_ramsete.py`, `overboard/verification/immrax_validator.py`

### KI-022 — The per-segment timing model was not held to the same wheel-jerk discipline

**Fixed.** The smoothness cost is the path-wide reference interval squared times the sum of the squared physical jerks, so every boundary pays the same per unit of physical jerk in either mode, and the reported `max_jerk_m_s3` uses the same term set and the same midpoint spacing as the solver.

**Was:** the cost penalised the raw acceleration jump, which is proportional to `jerk * dt`. Under one shared `dt` every boundary was weighted the same, but under per-segment `dt` a short interval was charged `dt²` less per unit of physical jerk and could buy violent wheel motion almost for free: on a 1.0 m + 0.2 m path the per-segment model reached 19.28 m/s³ of peak jerk against the uniform model's 9.52 m/s³, so the same `accuracy_weight` meant two different things.

**Fix:** the term is normalised by the path-wide mean interval and the spacing between the two interval midpoints; the uniform branch keeps its original expression, so its results stay bit-identical. After the fix the same path gives 9.52 m/s³ in both modes, and the per-segment objective is never worse than the uniform one at the same weight.

**Files:** `overboard/core/optimizer.py`, `overboard/verification/validator.py`, `overboard/main.py`, `tests/test_optimizer.py`

### KI-023 — The SysId collector could not be pushed to a hub and its metadata did not parse

**Fixed.** The collector compiles for MicroPython, prints its metadata in the format the host parses, and reports its measured pose. The whole log-and-pose path runs over Bluetooth from the host.

**Was:** three defects, all invisible to the host because the file is valid Python. Adjacent f-strings (which MicroPython cannot concatenate) made `mpy-cross` reject the file outright. The metadata was printed as `META,run_id 12` (space separated) while the parser splits `META,key,value` on commas, so no measured pose could ever be attached to its test. And the fit divided the measured heading by the odometry angle without noticing that the two use opposite signs, so the heading scale came out as -1 and the fit could not be loaded at all.

**Fix:** one `%` format per statement, `META,key,value`, `POSE,x_m,y_m,heading_deg` with the heading counter-clockwise positive, and the sign flipped where the two are divided. `tests/test_hub_deploy.py::TestFirmwareCompiles` compiles every file in `pybricks_code/` with `mpy-cross`, so the compiler defect cannot come back silently.

**Files:** `pybricks_code/sysid_digital_twin.py`, `pybricks_code/hub_runner.py`, `overboard/deploy/hub_deploy.py`, `overboard/sysid/collect.py`, `overboard/sysid/fit.py`, `overboard/main.py`, `tests/test_hub_deploy.py`, `tests/test_sysid_collect.py`

### KI-024 — The timing model had no rule for where the time allocation may change

**Fixed.** A waypoint is classified once, and the per-piece timing model only lets the *reading* of the motion profile change where the state at the point is fully determined. Where it is not, the wheel acceleration - and so the wheel force - must be continuous through the point. The equal-duration model is unchanged and its validated results are bit-identical.

**Was:** the per-piece model gave every waypoint its own interval, so the same velocity ramp was read as two different accelerations across a `dt` change, putting a step in the force profile exactly where the path should run smoothly, and the boundary semantics were never written down. On the complex mission the equal-duration solution stepped the wheel acceleration by 3.65 m/s² at a pinned heading the robot drives through; after the fix the per-piece step is 1.2e-4 m/s².

**Fix:** `TrajectoryOptimizer.merge_conditions` is the single place the rule lives - a waypoint is a timing boundary only when the robot stops there *and* the heading is pinned. The constraints are scaled by the reference interval squared so they read in m/s², which also fixed convergence: the complex mission's per-piece candidate went from 6.836 s in 41 iterations to 4.650 s in 203.

**Files:** `overboard/core/optimizer.py`, `tests/test_optimizer.py`

### KI-025 — The backlash was subtracted from the commanded wheel speed

**Fixed.** The backlash is a position uncertainty in the reachable set, and the closed-loop heading error is reported next to the position error.

**Was:** the plant subtracted a distance (metres) from a velocity (m/s), so the robot ran permanently `backlash/2` slower in both directions and the deficit grew into a pure position lag reported as tracking error - 226 µm on a 1 m straight, and the only uncertainty that reached the tracking error at all (`cof`, `torque_headroom`, `wheel_radius_range` and `inertia_kg_m2` are consumed by the constraint check and never touch the propagated state).

**Fix:** the plant turns the wheel at the commanded speed, and the reachable set is widened symmetrically by `backlash/2 * t`. Over the 60-candidate matrix the tracking error moved by a mean of -0.0008 cm and the envelope grew by +0.0070 cm; **no candidate's pass flag changed** (0 of 60). The heading error is reported, not gated: the heading behaviour is not yet calibrated.

**Files:** `overboard/verification/immrax_validator.py` (`RobotOpenLoop.f`, `DigitalTwinOpenLoop.f`, `compute_reachability`), `overboard/main.py` (the validation report), `tests/test_validator_model.py`, `docs/research/RQ-A_tracking_error.md`

### KI-026 — The plan's samples were not on the path its own profile traces

**Fixed.** The collocation is the default on both sides: the CLI has `--self-consistent/--no-self-consistent` with the default on, and `TrajectoryOptimizer.solve` defaults to it as well.

**Was:** the trapezoidal rule samples only the two endpoints of an interval, so the plan's positions were not where its own (v, ω) profile takes it, and the heading samples are free variables, so they are not the path's tangents either. On the S-curve the plan sat 11.5 mm off its own profile with a 1.31-1.32 cm tracking error.

**Fix:** the position rule integrates the interval's velocity along the heading the robot will hold, with a four-node Gauss-Legendre rule (O(dt⁷) per interval against the trapezoid's O(dt³)); the plan now sits on its own profile (under 0.1 mm) and the S-curve's tracking error falls to 0.55-0.59 cm. It is not a free win: the S-curve still misses the 5 mm gate by 0.05 cm and the stops mission gets slower, because a direction reversal is where a self-consistent plan is hardest.

**Files:** `overboard/core/optimizer.py` (`solve`, `_solve_once`, `_GAUSS_LEGENDRE_4`), `overboard/main.py`, `tests/test_optimizer.py` (`TestSelfConsistentPlan`), `docs/research/RQ-A_tracking_error.md`

### KI-027 — The reference interpolated the heading and the turn rate as independent lines

**Fixed.** The turn rate is the derivative of the interpolated heading, in the firmware and in the validator together.

**Was:** the heading, the speed, the turn rate and the Ramsete gain were four independent lines, but they are coupled: a line for the turn rate means the heading is a parabola, and the two disagree by up to `domega*dt/8` mid-interval - 8.7 degrees of contradictory heading on the S-curve at the default density. The robot was handed a heading target and a turn rate that described different motions, and the validator mirrored the mistake, which is why the model and the robot agreed on the bug.

**Fix:** the turn rate is `dh / dt`, and the heading difference is taken the short way exactly as the firmware does. Measured after the fix: the S-curve at n=40 goes from 0.98 cm to 0.27 cm, the stops mission at n=40 from 1.29 cm to 0.41 cm, the complex mission at n=10 from 0.86 cm to 0.41 cm.

**Files:** `pybricks_code/ramsete.py`, `pybricks_code/ramsete_compact.py`, `pybricks_code/ramsete_compact_events.py`, `overboard/verification/immrax_validator.py` (`_interpolated_reference`), `tests/test_jax_ramsete.py`

### KI-028 — The validator modelled a controller that is not the one that runs

**Fixed.** The validator models the command resolution and the Ramsete gain of the controllers that are in use, and the firmware now has **one** implementation: `pybricks_code/ramsete.py` holds the odometry, the Ramsete maths, the limiter and the 20 ms loop, `ramsete_fast.py` delegates to it, and `ramsete_compact.py` delegates to `ramsete_compact_events.py` with the events off.

**Was:** the validator was written against the legacy list-of-dict controller, and the controllers that actually ran did not all do the same thing - the plain compact one rounded the command and interpolated the gain (the way the validator did), the deployed one and the IDE missions did neither, and the plain one carried its own copy of the odometry and the maths, so it could drift away unnoticed. The control period was never the problem: it was 20 ms in the firmware and in the model all along.

**Fix:** one synced 20 ms heartbeat everywhere, the shared parts in the compact controller, and an unrounded command with a recomputed gain in the validator. The remaining parity fixes are worth about 1 % of the tracking error, so the gate did not move - the numbers that quote it are still valid. `tests/test_firmware_events.py::TestFirmwareParity` pins the parity.

**Files:** `overboard/verification/immrax_validator.py`, `pybricks_code/ramsete.py`, `pybricks_code/ramsete_compact.py`, `tests/test_firmware_events.py`, `tests/test_jax_ramsete.py`, `docs/research/RQ-A_tracking_error.md` (Fix 4)

### KI-029 — A late control tick silently skipped a mission event

**Fixed.** The event of every sample the robot has reached is fired, not only the one it is on.

**Was:** the reference controller advanced the sample index in a loop and then fired the event of the sample it was on, so a late tick - 60 ms instead of 20 ms - jumped over a sample and its action was silently lost. The same class of failure as KI-012, in a different place.

**Fix:** the event is fired for every sample the advance passes over and marked as fired, which is what the deployed controller already did.

**Files:** `pybricks_code/ramsete.py` (the sample advance and the event handling), `tests/test_firmware_events.py` (`TestLateTickEvents`)

### KI-030 — The studio told the operator an unchecked trajectory was "approved"

**Fixed.** The push is gated on the *validator's* verdict, a failed solve clears the field, and the `hidden` attribute works.

**Was:** four defects. The Run tab called the trajectory "approved" and enabled **Push to hub** after a solve that was never validated, so the default preset put an unchecked path one click from the robot. A failed solve left the previous solve's trajectory on the canvas, in the "Planned trajectory" colour, while the Result tab said "Failed". A solve that never started kept the previous verdict. And `[hidden]` did nothing, because `.row { display: flex }` beats the browser's own rule, so three controls that should have been invisible were on screen.

**Fix:** `state.solveValidated` comes from the selected candidate of the validation report, `clearTrajectory()` runs when a solve starts and when it fails, `refuseSolve()` sets the header and the Result tab, and `[hidden] { display: none !important; }` is in the stylesheet. The error the operator reads no longer carries the path of the temporary folder the CLI would have written to.

**Files:** `overboard/viz/studio.js`, `overboard/viz/studio.css`, `overboard/viz/studio.html`, `overboard/gui_server.py`, `tests/test_studio_honesty.py`

### KI-031 — A new user could not install the project

**Fixed.** The two requirement files install together, `--doctor` works on a clean machine, and the `.bat`/`.sh` scripts build the venv and start the studio in one double-click.

**Was:** `requirements.txt` pinned `numpy<2.0.0` while the optional stack needs `numpy>=2.2`, so a fresh install ended on 2.5.4 - outside the project's own declared range. `--doctor` imported click and numpy at the top, so a machine with nothing installed saw an ImportError instead of the install command. And `cmd.exe` could not find the `try_python` label in `install.bat` at all, because the file had LF endings.

**Fix:** `requirements.txt` asks for `numpy>=1.24.0,<3.0.0`, the core packages are checked before click is imported, the scripts find a Python 3.9 or newer, create `.venv` and run the check (a failure in the optional packages is a warning, not a failure), and `.gitattributes` keeps the batch files CRLF and the shell scripts LF.

**Files:** `install.bat`, `install.sh`, `start_studio.bat`, `start_studio.sh`, `requirements.txt`, `overboard/diagnostics.py`, `overboard/main.py`, `.gitattributes`, `README.md`

### KI-033 — The host could not find the hub over Bluetooth

**Fixed.** The hub is found the way the Pybricks IDE finds it: by probing what is nearby.

**Was:** the host used `find_device()`, which accepts an advertisement only when it carries the Pybricks service UUID **and** a local name. Windows does not report them in that shape - bleak's WinRT backend leaves `service_uuids` empty unless the peripheral puts the UUID in the payload, and the local name tends to arrive in a scan response the scanner never joins. Every push ended in "No Pybricks hub found over BLE" while the IDE connected to the same hub.

**Fix:** `overboard/deploy/hub_ble.py` scans everything, ranks the candidates (the hub the operator names first, then anything that advertises the service, then the rest) and probes the ones that do not announce themselves. `python -m overboard.deploy.hub_deploy --list-hubs` is the IDE's device picker.

**Files:** `overboard/deploy/hub_ble.py`, `overboard/deploy/hub_deploy.py`, `overboard/gui_server.py`, `tests/test_hub_ble.py`

### KI-034 — The local envelope limit landed on the wrong segment

**Fixed.** One rule, in `main._segment_limit`: a waypoint's limit governs the segment that **ends** at it, and the segment's start waypoint's limit is the fallback.

**Was:** a waypoint's `error_threshold` (or `precision_mode`) was applied to the samples of the segment that **started** at that waypoint, so the tight limit was enforced on the way *out* and the final waypoint's 1 mm limit was never read at all.

**Fix:** the last waypoint's limit now governs the last segment, and a precision waypoint is held to its tight limit on the approach - so a mission with a precision waypoint can now fail where it passed, and the measurement runs under `archive/research-2026-10:docs/measurements/` predate the change.

**Files:** `overboard/main.py` (`_segment_limit`, `_segment_report`, the gate loop), `tests/test_regressions.py` (`TestSegmentLimits`)

### KI-035 — The wheel-speed limit ignored the gearing

**Fixed.** `max_linear_speed` divides by the gearing, and `no_load_wheel_speed_worst_case` gives the validator the same number for the CPU audit and for Immrax.

**Was:** with `gearing: 2.0` the limit was 0.477 m/s where the real no-load wheel speed is 0.265 m/s, so the optimizer's bound and the audit's limit were twice too large - and at the optimizer's own bound the motor torque is exactly zero, so every plan for a geared robot was guaranteed to fail the motor gate, reported as "motor force over limit" rather than "wheel speed over limit".

**Fix:** the divide is applied the same way the force model applies it; with `gearing: 1.0` every number is unchanged.

**Files:** `overboard/models/robot_model.py` (`max_linear_speed`, `no_load_wheel_speed_worst_case`), `overboard/models/jax_robot_model.py`, `overboard/verification/validator.py` (`audit_constraints`), `overboard/verification/immrax_validator.py`

### KI-036 — The studio pushed whatever trajectory it was given

**Fixed.** The server refuses to push a trajectory whose selected candidate did not pass the validator, and the origin check compares the port.

**Was:** the rule "only a validated trajectory goes to the robot" lived only in the browser. The server's `/api/push` checked that a `.traj` file existed, and the CLI writes that file without `--validate`, so a direct POST pushed an unchecked plan onto the mat. The cross-site guard compared the origin's **hostname** but not its **port**, and `/api/robot/model/load` read any path on the machine.

**Fix:** `start_push` reads the validator's verdict back from the solve's own `##OVERBOARD-VALIDATION##` line and refuses unless the **selected** candidate passed, the origin check includes the port, and the model loader only reads under the project or calibration folder.

**Files:** `overboard/gui_server.py` (`_validation_verdict`, `start_push`, `_origin_allowed`, `_load_model`), `tests/test_gui_server.py` (`TestStudioPush`)

### KI-037 — The studio kept every job forever, and a cancel left the solver running

**Fixed.** The studio keeps the newest 20 finished jobs and removes their folders, a cancel kills the whole process tree, and the job stays "running" until the process is really gone.

**Was:** nothing ever removed a job or its temporary folder, so every solve left an `overboard_studio_*` folder with the config, the waypoints, the trajectory and the stats, downloadable by guessing an id. `Cancel` killed only the CLI parent, while the candidate workers survived and kept solving at full speed as the studio said "Cancelled". And the test suite committed into the operator's own mission history.

**Fix:** the child runs in its own process group and a cancel kills the tree (`taskkill /T /F` / `killpg`); the test writes into a temporary folder.

**Files:** `overboard/gui_server.py` (`MAX_JOBS`, `_evict_old_jobs`, `_cancel_job`, `_process_group_kwargs`, `_terminate_tree`), `tests/test_gui_server.py`

### KI-038 — The exported Python file broke on an event name with a quote in it

**Fixed.** The event name is written with `repr()`, so it is one Python string whatever it contains.

**Was:** the name went into the generated source through an f-string, so an event named `raise_arm"` produced a SyntaxError on the hub, and a name like `x", "y": 1, "z": "` injected extra keys into the sample dict.

**Fix:** `repr()` in the exporter; the compact export already did it.

**Files:** `overboard/deploy/export.py` (`write_python_file`), `tests/test_regressions.py` (`TestEventNameEscaping`)

### KI-039 — Two missions in another script became the same file

**Fixed.** The slug keeps unicode letters and digits, the restore parses before it writes, and the baseline guess blends the two headings the short way round.

**Was:** `slugify` stripped everything outside `[a-z0-9_-]`, so two non-latin names both became `mission` and the second save overwrote the first. The restore wrote the file and only then parsed it, so a blob that is not a mission reached the disk. And the baseline guess averaged two headings plainly, so a path turning through ±180° got the worst possible guess for exactly the missions that need another seed.

**Files:** `overboard/gui_server.py` (`slugify`, `restore_mission`), `overboard/core/optimizer.py` (`_guess_point`)

### KI-040 — `--push-hub --hub-log` lost the whole hub log when the link dropped

**Fixed.** The log path goes into `push_trajectory`, which streams it, so a dropped link leaves what arrived on disk and the error says where it is.

**Was:** the CLI called `push_trajectory` without the `log_path`, so the file was written only after the push returned; a dropped Bluetooth link raised before that, and the measured `POSE` and `EVENT` lines were lost - while `hub_deploy --out` and the collector streamed theirs.

**Fix:** the file is rewritten in the text form the other two entry points write, so one reader serves all of them.

**Files:** `overboard/main.py` (`_push_to_hub`), `overboard/deploy/hub_deploy.py`

### KI-041 — One corrupt cell in a session poisoned the whole calibration

**Fixed.** A row with a non-finite value is dropped and counted, and the fit file is written with `allow_nan=False`, so a poisoned fit fails loudly **at write time** and no file is written.

**Was:** the fitter read a session row with `float(part)`, and `float("nan")` and `float("inf")` are valid Python, so one bad cell became data: the gain came out `nan`, the summary still wrote `"calibrated": true`, and `json.dump` emitted a bare `NaN` token - not valid JSON - so the operator was told the fit was broken and not that the *session* was.

**Fix:** the count is reported in the fit and printed by the CLI, and a step whose command or measured window is not finite is not fitted.

**Files:** `overboard/sysid/fit.py` (`parse_session`, `_step_fits_measured`, `_fit_first_order`, `_summary`, the CLI), `tests/test_sysid_fit.py` (`TestBadRows`)

### KI-042 — A two-repeat calibration could not be loaded at all

**Fixed.** The loader uses the interval only when its lower bound is positive and the band is not inverted, and falls back to the observed `gain_min`/`gain_max` otherwise.

**Was:** the gain's 95 % interval over the repeats is a t-interval, and with two repeats the multiplier is 12.706, so an ordinary 20 % run-to-run spread gave `ci95 = [-0.27, 2.27]`. The loader takes that interval as the validator's gain band and refuses a band whose lower bound is not positive - and two repeats is exactly what a dropped Bluetooth link leaves behind.

**Fix:** the fitter clamps the interval's lower bound to a small positive number and keeps the raw t-interval under `<name>_ci95_raw`, so the file always loads and the operator can still see what the interval really was.

**Files:** `overboard/sysid/digital_twin.py` (`_actuator_from_dict`), `overboard/sysid/fit.py` (`_summary`), `tests/test_digital_twin.py`, `tests/test_sysid_fit.py`

### KI-043 — The collector calibrated the odometry from the robot's own odometry

**Fixed.** The collector no longer passes the hub's poses to the fit; the odometry block stays uncalibrated unless a real measurement is supplied.

**Was:** the hub's distance reading **is** its odometry, so measured/odometry is exactly 1.0 by construction, and the fit wrote `"calibrated": true` for a block that carries no information - while the validator uses that scale to advance the robot's own pose estimate. The fit also pooled the tape samples and the wall-homing samples and took their median, so with an even count the number used was neither estimate.

**Fix:** the scale is the tape estimate when there is one and the wall estimate otherwise, and both are reported with their disagreement, so the operator still sees the cross-check. The wall still calibrates the distance scale, because the wall is an independent reference; the heading scale and the brake slip need a tape.

**Files:** `overboard/sysid/collect.py`, `overboard/sysid/fit.py` (`_fit_odometry`), `tests/test_sysid_collect.py`, `tests/test_sysid_fit.py`

### KI-044 — The studio kept the last run's verdict, and a reload could not cancel a solve

**Fixed.** The validation report has one owner, the push refuses a mission that has changed since the solve, and a reload attaches to the running job.

**Was:** a new solve did not clear the previous run's verdict, and a solve without validation prints none, so the old failing chips, the old per-segment table and the old red segment stayed on the field while the header said "not validated". A validated trajectory stayed pushable after a waypoint was dragged, so the push ran the **old** plan while the canvas showed the new mission. And a reload during a solve had no job id, so the next solve was refused with 409 and Cancel was a no-op.

**Fix:** `renderValidationReport` handles "no report" by clearing the table, the chips, the failing-segment set and the red marks, and the push checks that the mission still matches the solve.

**Files:** `overboard/viz/studio.js`, `tests/test_studio_honesty.py`

### KI-045 — `sysid_solver.py` is a superseded prototype

**Fixed.** The file is deleted and `pandas` is out of the optional requirements.

**Was:** a standalone pandas/JAX fitter with the hub's telemetry pasted in as hard-coded CSV text and the robot's numbers hard-coded as constants; `sysid_fit.py` replaced it, parsing the real session format and writing the format the validator reads.

**Files:** `sysid_solver.py` (deleted), `requirements-optional.txt`, `overboard/sysid/fit.py`

### KI-046 — Five smaller things the audit left open

**Fixed.** All five.

**Was:** the studio flipped a job's status to "ok" before it filled in the result; a device whose name merely *contains* the wanted string was never probed; the hub session parser dropped malformed rows silently; `bleak` and `pypdfium2` were imported but not declared; and some docs still described a state that had moved on.

**Fix:** the result is filled in inside a `try...finally` before the status flips; substring name matches rank into the preferred probe tier; the parser reads the `TEST_END,<n>` row counts and warns on truncated BLE captures; the two packages are declared; the stale docs were updated.

**Files:** `overboard/gui_server.py`, `overboard/deploy/hub_ble.py`, `overboard/sysid/fit.py`, `requirements-optional.txt`, `README.md`, `docs/CONFIGURATION.md`

### KI-047 — The fast solve path made the solver slower, and its test failed on the tree

**Fixed.** The cold pass restarts from the **warm stage's iterate**, not from a warm-started half-finished point and not from the original guess.

**Was:** restarting from a half-finished point *with warm start* turned out to be the slowest thing the path could do. On one machine the plain warm pass took 1557 iterations and 35 s while the fast path took 3112 iterations and 12.7 s - four times slower than the plain pass - and the iteration-budget test (1200) failed there. The count is not a property of the code alone: a different CPU and a different BLAS change the last bits, and IPOPT on this poorly scaled NLP is chaotic enough that a 1-ULP difference early on sends it down a different path.

**Fix:** the cold stage is cold-started (`warm_start_init_point=no`) from the budgeted stage's iterate. The warm stage has usually found the good basin already - on the complex mission its 200-iteration iterate is at 6.08 s - so throwing that away and cold-starting from the original guess lands in a worse local optimum (8.67 s on the Gauss-Legendre collocation). Cold from the warm iterate is 433 iterations and 12.5 s against the plain pass's 1257 iterations and 33 s, and returns 6.096 s against 6.087 s. The budgeted stage reports the iteration count it used, so the stage report is honest.

**Files:** `overboard/core/optimizer.py` (`solve`, `_solve_once`), `tests/test_optimizer.py` (`TestSolverIterationBudget`), `AGENTS.md`, `README.md`, `docs/EXAMPLES.md`

### KI-048 — The solver gives up on a path instead of trying another candidate

**Removed (2026-10-21).** The rescue round is gone: `_run_candidates` solves the planner's candidate set and nothing else. Over the 32-mission suite (`research/missions`, 8 per profile, `--candidates 3 --validate --propagation plant`, records in `research/results/q14_plant.jsonl`) 22 missions failed the gates, every one of them ran the rescue, and **no rescue candidate passed on any mission**. 95 of the 97 rescue candidates that were checked failed on the envelope — the same gate that stopped the set, and not something a seed can move — and the round roughly doubled the wall time of every failing mission. The seeds are still declared in `overboard/core/optimizer.py`; reviving the round means restoring the fallback batch in `_run_candidates` from git history.

**Was (fixed until 2026-10-21).** The CLI solved the candidates in rounds: if the run produced **nothing usable** - no candidate passed the gates, or (without validation) none converged - the seeds the set did not use were tried as a **rescue** before the mission was given up on.

**Was:** IPOPT can exhaust its 5000-iteration budget; the optimizer returns best-effort samples, the candidate is marked not converged, and a non-converged candidate is never eligible for export, so it is thrown away with its wall time spent. The stress run of 2026-10-20 found one candidate in 228 at the limit (36.5 s) - though in all 114 mission-robot pairs at least one candidate converged, so no mission was lost.

**Fix:** on a validated run the rescue uses the per-segment model, because that is where the measured rescues came from. A rescue candidate is insurance: it is only selected when the candidate set itself produced nothing that passed. It costs nothing when the plan works.

**Files:** `overboard/core/optimizer.py` (`solve`), `overboard/main.py` (`_run_candidates`), `tests/test_regressions.py` (`TestWriteGateIntegration.test_a_non_converged_first_candidate_is_not_rescued`, `test_a_set_that_fails_every_gate_is_not_rescued`)

### KI-049 — The reachability envelope reported a truncation as a width, and the tube behind it was blown up by a dependency

**Fixed.** Three exact reformulations, and no threshold moved. The report also carries `envelope_truncated`, so a cut-off propagation says so.

**Was:** the propagation stopped when the tube's half-diagonal passed 50 mm and reported the radius it had at that moment, so the same 54.56 mm came out for a 0.5 m, a 1 m and a 2 m path - a property of the guard, quoted as a measurement. And the natural extension could not see that the turn rate does not depend on the gain, so it reported a phantom turn rate that integrated into a heading drift and then a lateral drift. The reachable set the robot can actually reach is 4.5 mm wide; everything above that was the model's own doing.

**Fix:** the wheel-speed difference is taken from the **commands** and the gain applied afterwards (on a straight the command difference is exactly zero, so the turn rate is exactly zero), and the digital twin carries the wheel speeds as the **mean** and the **difference** instead of the two wheel speeds, because the two are perfectly correlated and an interval box cannot express that. The Jacobian inclusion is 0.00-0.08 mm wider and 5-15 % slower than the natural one, so the default stayed on the natural inclusion.

**Files:** `overboard/verification/immrax_validator.py` (`RobotOpenLoop.f`, `DigitalTwinOpenLoop`, `ImmraxValidator.__init__`, `compute_reachability`), `tests/test_envelope_soundness.py`, `requirements-optional.txt`

### KI-050 — The closed-loop tick map could not run at all on the installed JAX

**Fixed.** The pose is read by indexing, which compiles to `slice`+`squeeze` and is in Immrax's registry, and the rule is now a rule for the whole tree: **no array is unpacked in a path that Immrax embeds.**

**Was:** unpacking the pose is the JAX `unstack` primitive, and Immrax's natural inclusion has no rule for it, so every candidate failed with `NotImplementedError: unstack not in inclusion_registry` and the run reported "Immrax validation error". The tick map had been measured on JAX 0.6.2, where the unpacking compiled to slices; the stack a fresh install brings is 0.12.0, where it does not. The measurements were right; the code did not run on the stack the project installs.

**Fix:** the numbers are unchanged to the last digit (0.3047634911948969 cm envelope and 0.3409694880247116 cm tracking on the straight example, before and after). The wiring came with it: `--propagation [plant|tick]` on the CLI and on `research/harness.py`, and every candidate in the `##OVERBOARD-VALIDATION##` line carries the propagation and the inclusion functions, so an envelope is never read without the model that produced it.

**Files:** `overboard/models/jax_ramsete.py` (`JAXRamseteController.calculate`), `tests/test_jax_ramsete.py`, `overboard/main.py`, `research/harness.py`, `overboard/viz/studio.js`

### KI-051 — `--self-consistent` was a no-op: the switch could not change the plan

**Fixed.** The flag is two-way now (`--self-consistent/--no-self-consistent`, default on) and the CLI passes the value to the solver explicitly, so the trapezoidal collocation is reachable again.

**Was:** the CLI only ever set `solve_kwargs["self_consistent"] = True` and never `False`, so solving the S-curve with and without the flag produced identical samples - 21 samples, 3.725 s, every field equal - and `research/harness.py` had the same one-way switch.

**Fix:** the two plans are measurably different: the S-curve is 3.725 s / 1.250 m self-consistent (0.55 cm tracking) and 3.696 s / 1.233 m trapezoidal (1.31 cm).

**Files:** `overboard/main.py` (`--self-consistent/--no-self-consistent`), `overboard/core/optimizer.py` (`solve`), `research/harness.py`, `overboard/viz/studio.html`, `overboard/viz/studio.js`

### KI-052 — The CLI and the library plan different trajectories for the same mission

**Fixed.** `TrajectoryOptimizer.solve` defaults to the self-consistent rule as well, so the command line, the studio and the library all plan the same trajectory for the same mission.

**Was:** the CLI's default was flipped first and the library's stayed the other way round, so the same waypoints, the same robot and the same solver gave two different plans depending on who asked - 3.725 s / 1.250 m (self-consistent) against 3.696 s / 1.233 m (trapezoidal), with tracking errors of 0.55 cm and 1.31 cm: more than a factor of two apart on the gate that decides the mission.

**Fix:** `--no-self-consistent` is the way to ask for the trapezoidal rule from any of them.

**Files:** `overboard/core/optimizer.py` (`solve`), `overboard/main.py`, `research/harness.py`, `docs/HEURISTICS_AND_VERIFICATION.md` (§9), `docs/research/RQ-A_tracking_error.md`

### KI-053 - The interval force model passed an interval through unchanged

**Fixed.** `_duck_abs` and `_duck_clamp_zero` in `overboard/models/drivetrain_physics.py` now map an Immrax interval instead of returning it: `|x|` becomes `[0, max(|lower|,|upper|)]` when the interval straddles zero and `[min(|lower|,|upper|), max(|lower|,|upper|)]` when it does not, and `max(0, x)` becomes `[max(0,lower), max(0,upper)]`.

**Was:** both helpers had an `if hasattr(x, "lower") and hasattr(x, "upper"): return x` branch, so an interval came out of the magnitude and the clamp exactly as it went in. A signed velocity interval therefore produced a **negative lower bound on the motor force** - a force magnitude cannot be negative - and a torque that had already gone past the no-load speed was not clamped at zero.

**Why it was not caught:** the bug was latent. The only interval callers are `JAXRobotConfig.get_max_force_at_velocity(..., use_intervals=True)` and `JAXDifferentialDriveModel.check_constraints(..., use_intervals=True)`, and no shipped path passes that flag (`docs/API_robot_model.md`). The Immrax validator does not use these helpers either: it carries its own copy of the motor torque curve, so the validation gate was never reading the broken branch. No trajectory, gate verdict or exported controller changes.

**Also fixed here:** `overboard.models` exported a `MotorConfig` that does not exist (an `AttributeError` on access); `overboard/core/frontier_seed.py` had a dead `baseline` branch behind an early return, an unreachable `zero_velocity` flag, and no `generate_canonical_seeds` although the strategy ledger and the archived benchmark both name it; and the repository's directories were resolved in eight modules, each by counting its own parents, which is now `overboard/paths.py` with a test that keeps it that way (`tests/test_paths.py`).

**Files:** `overboard/models/drivetrain_physics.py`, `overboard/models/__init__.py`, `overboard/core/frontier_seed.py`, `overboard/paths.py`, `tests/test_robot_model.py`, `tests/test_frontier_seed.py`, `tests/test_paths.py`

## The rules this list produced

1. KI-001, KI-002, KI-003, KI-005, KI-006, KI-007, KI-008, KI-009, KI-010, KI-011, KI-012, KI-013, KI-014, KI-015, KI-016, KI-019, KI-021, KI-022, KI-023, KI-024, KI-025, KI-026, KI-027, KI-028, and KI-029: done in the working tree; never save or ship a failed or unvalidated run, keep validation on the configured robot, use both uncertainty bounds, keep the reference tube aligned with the samples, keep the optimizer and Immrax on the same per-wheel force model, hold both timing models to the same wheel-jerk discipline, keep the wheel force profile continuous through every waypoint the robot does not stop at, keep the backlash in the reachable set rather than in the wheel speed, derive the turn rate from the interpolated heading in the firmware and the model together, place the plan's samples on the path its own profile traces, model the controller that actually runs, fire the event of every sample the robot has reached, never take a port from another program, and keep the firmware compilable for MicroPython.
2. KI-020: controller split, log/fit tooling, and the optional `--sysid-fit` actuator/odometry path are done, and the log and pose capture now run over Bluetooth from the host; collect hardware data, compare fitted predictions with measured poses, add integrated odometry/slip, then use reachability in optimization.
3. KI-004: deleted - the pipeline was removed from the tree (git tag `archive/legacy-2026-10`) and has no CLI entry point; measured as a seed provider it rescued nothing. The next lever is the objective, because the pipeline's cost buys speed with tracking accuracy and the 5 mm gate rejects the trade.
4. KI-017: done - dead flags removed and iteration history forwarded.

## Not checked

This list does not cover `sysid_solver.py` beyond its imports, the plotting and animation modules in depth, the Pybricks tutorials, or the generated trajectory and benchmark data. The Pybricks code was not run on hardware.
 The motor's numbers are measured now as well: the collector's `top_speed` test lifts the robot so the motor runs free and measures the speed at which it makes no torque, the fit rejects that number when the robot reached the commanded speed instead, and `main.py --sysid-fit` writes the measured stall torque and no-load speed into the robot config. That closes a hole where the optimizer planned against the declared motor and the validator checked the measured one. The drivetrain model is in `drivetrain.py`; the model is checked against an independent simulator in `research/crosscheck/`, and the findings are in `docs/research/RQ-B_drivetrain_model.md`.

The random mission suite and the random robot constants were outside this list's scope until 2026-10-20, when a stress run measured 180 candidates on 45 random paths and 90 random robots against the production gate chain (`archive/research-2026-10:docs/measurements/2026-10-20-stress/`). It found one failure mode that was not in this list (KI-048, since fixed) and confirmed that the tracking and forward-integration gates are dominated by the discretisation mismatch rather than by the physics (KI-026, RQ-A). What that run does **not** cover: the paths in the suite are longer and more turny than a real FLL mission, the robot constants are drawn, not measured (KI-020), and nothing was run on hardware.
