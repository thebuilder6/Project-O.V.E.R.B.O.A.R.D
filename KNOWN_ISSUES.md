# Known Issues

Last reviewed: 2026-10-19

Scope: whole repository, reviewed as a single-user, offline desktop tool for one FLL robot. This is not a multi-user or server-scale review.

Status: KI-001, KI-002, KI-003, KI-005, KI-006, KI-007, KI-008, KI-009, KI-010, KI-011, KI-012, KI-013, KI-014, KI-016, KI-019, KI-021, KI-022, KI-023, KI-024, and KI-025 are **fixed in the working tree** (uncommitted). All other items are open; KI-015 and KI-017 are partially fixed. KI-020 records the remaining digital-twin and optimization work. The user-facing docs were reconciled with this tree on 2026-10-19; the code remains the authority if they diverge again.

How to use this list: refer to an item by its ID, for example “fix KI-003”.

## Quick triage

| ID | Priority | Area | Issue |
|---|---|---|---|
| KI-001 | P0 | Solver / output | ~~A failed solve is exported as a successful trajectory~~ (fixed) |
| KI-002 | P0 | Validation | ~~`--validate` cannot block output or change the exit code~~ (fixed) |
| KI-003 | P0 | Safety | ~~Immrax ignores upper uncertainty bounds~~ (fixed) |
| KI-004 | P0 | Multi-Verse | Refinement has no effect on the final trajectory |
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
| KI-015 | P1 | Benchmarks | Random benchmark tools are broken |
| KI-016 | P1 | Live view | ~~The first client never receives config or waypoints~~ (fixed; the live view is gone) |
| KI-017 | P2 | CLI | Convergence flags do nothing (dead flags removed) |
| KI-018 | P2 | Maintenance | Dead code, duplicates, and generated files |
| KI-019 | P0 | Safety | ~~The reachability tube lags the reference by one sample~~ (fixed) |
| KI-020 | P1 | Digital twin | Validator/optimizer do not yet model motor, odometry, slip, or reachability-driven optimization |
| KI-021 | P1 | Robot runtime | ~~The DriveBase caps the acceleration the plan asks for~~ (fixed) |
| KI-022 | P1 | Solver / timing | ~~Per-segment timing was not held to the same wheel-jerk discipline~~ (fixed) |
| KI-023 | P1 | Robot runtime | ~~The SysId collector could not be pushed to a hub and its metadata did not parse~~ (fixed) |
| KI-024 | P1 | Solver / timing | ~~The timing model had no rule for where the time allocation may change~~ (fixed) |
| KI-025 | P1 | Safety model | ~~The backlash was subtracted from the commanded wheel speed, and the tracking metric ignored the heading~~ (fixed) |

## P0 — fix first

### KI-001 — A failed solve is exported as a successful trajectory

**Status:** Fixed in the working tree (uncommitted). Failed solves now exit non-zero and write nothing.

**Files:** `main.py` (candidate loop and export gate), `validator.py`, `multiverse_optimizer.py`

**Problem:** When IPOPT failed, `TrajectoryOptimizer.solve` caught the exception and returned the solver's last debug values with `stats["converged"] = False`. `main.py` did not check that flag. It wrote the trajectory and exited 0.

**Evidence:** A solve from `(0,0,0)` to `(100,0,0)` returned `converged=False` and a trajectory that ended near `x=20` instead of `x=100`. The result was still written as a normal trajectory. The validator only checked internal consistency, not the requested waypoints, so it could still report PASS. After the fix, the infeasible `(0,0,0) -> (1,0,0) -> (2,1,pi/2)` path exits 1 with `Solver did not converge` and writes nothing.

**Fix (implemented):** `main.py` checks `stats["converged"]` before any output. `_select_attempt` excludes non-converged candidates, so a failed candidate cannot be picked over a converged one. Multi-Verse now propagates the inner solve's `converged` flag. The final pose is checked against the last waypoint, and `forward_integrate` rejects non-positive timestamps instead of crashing. See `tests/test_regressions.py` (`TestExportGate`, `TestWriteGateIntegration`, `TestValidatorGates`).

**Workaround:** None needed. Failed solves exit non-zero and write no trajectory, controller, Python file, or stats. A pre-existing file at the output path is left untouched.

---

### KI-002 — `--validate` cannot block output or change the exit code

**Status:** Fixed in the working tree (uncommitted). `--validate` now gates every output and the exit code.

**Files:** `main.py` (candidate loop and export gate), `validator.py`

**Problem:** `passed_validation` was assigned but never read. A failed validation still wrote the trajectory and could still export a controller or Python file. If Immrax threw, the code set `passed_validation = True` and continued. The final `validate_trajectory` call was advisory only.

**Evidence:** The repo’s own straight example (`examples/fll_choreo.json` + `examples/example_straight.json`) with `--simple --validate` prints “FAILED: Physical constraints” on both attempts, then writes the output anyway. Direct Immrax reported `max_motor_violation_N=0.0792`, `max_tracking_error_m=0.1616`, and `passed=False`. The separate validator then printed PASS. After the fix, the same command exits 1 with `Validation failed: motor force 0.079 N over limit; tracking error 16.16 cm > 0.50 cm` and writes no trajectory, controller, Python file, or stats.

**Fix (implemented):** `--validate` requires JAX/Immrax and gates on the selected candidate result, the in-memory CPU validation (`validate_samples`), and the final waypoint before writing. On failure it prints the specific reason and exits 1; the trajectory, controller, Python export, and `_stats.json` are not written. The final `validate_trajectory` call is no longer advisory: the same checks run before the file is created. If only `--live` is set, the trajectory is still shown, but no “safe” claim is made. The benchmark stats file is written only after the gate.

**Workaround:** None needed for gating. The straight example passes after the KI-019 fix, and the robust check now uses both uncertainty bounds after the KI-003 fix.

---

### KI-003 — Immrax ignores upper uncertainty bounds

**Status:** Fixed in the working tree (uncommitted). `compute_reachability` now passes `Interval(w_lower, w_upper)` to the embedding system, and a regression test confirms that widening only `w_upper` widens the envelope.

**Files:** `immrax_validator.py:286-355`, `tests/test_regressions.py`

**Problem:** `compute_reachability` creates `w_flat` as six lower bounds followed by six upper bounds, then passes it to `emb_sys.f` as a plain JAX array. Immrax’s natural inclusion treats non-`Interval` arguments as constants. Only the lower bounds affect the reachable set.

**Evidence:** Changing only `w_upper` produces identical output. Changing `w_lower` changes the output. Torque, backlash, track width, wheel radius, and inertia are therefore not checked in the upper direction.

**Fix:** Pass `w = Interval(w_lower, w_upper)`. If wheel-command error should be uncertain, pass `u` as an `Interval` too. Add a regression test that widening only `w_upper` widens the envelope.

**Workaround:** None needed.

---

### KI-004 — Multi-Verse refinement has no effect

**Status:** Partially fixed in the working tree (uncommitted).

**Files:** `multiverse_optimizer.py` (`_global_solve`, `_probe_refined_guess`), `optimizer.py` (`initial_guess`)

**Problem:** Phase 4 computes a refined `global_traj`, but `_global_solve` builds `new_guess` and never passes it into `TrajectoryOptimizer.solve`; the local variable is unused. `TrajectoryOptimizer.solve` now exposes `guess_style` seeds, but Multi-Verse does not use them or the refined samples. Phase 5 re-solves from its default baseline guess.

**Evidence:** On `examples/example_complex_mission.json`, the default Multi-Verse run and the simple solver both produce 91 samples and the same 6.0677 s final cost; the default Multi-Verse run took 28.26 s wall time. `--workers`, `--no-parallel`, and `enable_parallel` cannot change the result because the refined guess and the thread pool are not used by the final solve.

**Fix:** `TrajectoryOptimizer.solve` now takes an `initial_guess` array, and the Multi-Verse polish probes the refined trajectory with a bounded cold pass (400 iterations) and keeps it when it converges to a lower cost. The outcome is reported in the run stats as `refine_probe` (`converged`, `cost`, `iterations`, `used`).

**What is left:** The plumbing works, but the current refinement is a worse starting point than the trivial baseline guess: on the example mission the probe converges in 38 iterations to **8.1473 s**, while the default guess reaches **6.0677 s**. The probe therefore never wins, and the refinement still has no effect on the result. The residue is the quality of the JAX refinement, not the connection to the solver.

**Workaround:** Use the simple solver if you want predictable behaviour.

---

### KI-005 — The live visualizer kills whatever owns port 8765

**Status:** Fixed in the working tree (uncommitted), then closed by removal.

**Files:** `live_visualizer.py` (deleted), `main.py`

**Problem:** `LiveVisualizer.start()` called `_kill_process_on_port`, which used `netstat` + `taskkill` on Windows and `fuser -k` on POSIX. It did not check which process owned the port.

**Fix:** `_kill_process_on_port` is gone and the port is probed before binding, so a busy port raises a clear "already in use" error. The live view itself was then retired: the browser studio is the only browser UI, and it never takes a port that is already in use (it tries the next free one). `--live` opens the studio instead.

---

### KI-006 — A fresh install cannot run the default command

**Status:** Fixed in the working tree (uncommitted).

**Files:** `requirements-optional.txt`, `main.py`, `diagnostics.py`, `README.md`

**Problem:** `requirements.txt` omitted `jax`, `jaxopt`, `immrax`, and `pandas`, and `main.py` defaulted to `MasterTrajectoryOptimizer`, which raises `ImportError` without JAX.

**Fix:** The simple CasADi solver is now the default and needs only `requirements.txt`; Multi-Verse, `--validate` and `sysid_solver.py` use the new `requirements-optional.txt`. `python main.py --doctor` reports what is installed and prints the exact install command for anything missing.

## P1 — fix before relying on the result

### KI-007 — The JAX test suite is silently skipped

**Status:** Fixed in the working tree (uncommitted). The stale test modules were deleted and the suite now runs every JAX module.

**Files:** `run_tests.py`, `tests/test_regressions.py`

**Problem:** All JAX test imports are wrapped in one `try`. Two test modules import classes that no longer exist: `MultiVerseRefiner` and `LocalSegmentOptimizer`.

**Evidence:** `python run_tests.py` previously reported 39 passing tests and printed "jax not installed", even though JAX 0.6.2 was installed. The JAX tests never ran under the normal test command. After deleting the stale modules, the suite runs 194 tests with 0 skipped.

**Fix (implemented):** The two stale modules (`tests/test_stomp_variants.py`, `tests/test_teb_topologies.py`) were deleted. The production classes were not changed just to satisfy dead tests.

**Workaround:** None needed; `python run_tests.py` now runs the JAX modules.

---

### KI-008 — JAX validation reads a different config format

**Status:** Fixed in the working tree (uncommitted). `JAXRobotConfig` now delegates parsing to `RobotConfig`, and `main.py` passes the already-parsed config.

**Files:** `jax_robot_model.py:11-38`, `robot_model.py:27-100`, `main.py:245`

**Problem:** `JAXRobotConfig` reads only `config_dict["robot"]` or a flat dictionary, and only canonical keys such as `track_width`. `RobotConfig` also supports the `config` wrapper, `{ "val": ... }`, and old names such as `differentialTrackWidth`, `radius`, `vmax`, and `tmax`.

**Evidence:** The README tells users to write the `config` + `{val}` format. `main.py` passes that raw object to `JAXRobotConfig`, which silently falls back to defaults. Immrax then validates a different robot than the optimizer used.

**Fix:** Build `JAXRobotConfig` from a parsed `RobotConfig`, or copy every supported key and wrapper into it. Add a test using the README sample config.

**Workaround:** None needed; legacy and canonical configs now parse identically.

---

### KI-009 — Declared uncertainty ranges are ignored

**Status:** Fixed in the working tree (uncommitted). `RobotConfig` normalizes `robot.uncertainties`; `main.py` passes all six declared ranges to Immrax, falling back to the old constants only when a range is absent. The optimizer also uses the declared torque-headroom and wheel-radius worst case for its per-wheel force constraints.

**Files:** `robot_model.py:64-100`, `main.py:245-260`, `examples/fll_choreo.json:17-24`

**Problem:** `main.py` hardcodes `cof ±20%`, `torque=(0.8, 1.0)`, and `backlash=(0.0001, 0.0006)`.

**Evidence:** `fll_choreo.json` declares `cof [1.24, 1.51]`, `torque [0.75, 0.9]`, and separate track-width, wheel-radius, and inertia ranges. The validator checks different numbers than the robot file says.

**Fix:** Read `robot.uncertainties` from the config, using the hardcoded values only as a fallback. Pass the same ranges to Immrax and the report.

**Workaround:** None needed; declared ranges are used automatically, with the old constants as fallbacks.

---

### KI-010 — Weight-transfer math is wrong and tools disagree

**Status:** Fixed in the working tree (uncommitted). One shared `wheel_normal_forces` function (lateral transfer only, no front/rear split) is used by the CPU model, the JAX model, the Immrax validator, and the optimizer, and the optimizer constrains the same worst-case per-wheel force Immrax checks.

**Files:** `robot_model.py:7-35` (shared function), `robot_model.py:289` (`get_wheel_normal_forces`), `robot_model.py:312` (`get_max_wheel_force_symbolic`), `robot_model.py:334` (`get_traction_limit_symbolic`), `optimizer.py:112-134`, `jax_robot_model.py:126`, `immrax_validator.py:215-223`

**Problem:** `get_wheel_normal_forces` subtracts the same `longitudinal_transfer` from both `nl` and `nr`. The comment says load shifts to the rear, but a two-wheel differential robot has left and right wheels, not front and rear. During acceleration, `nl + nr` becomes less than `mg`; both wheels lose normal force.

**Evidence:** The formula is repeated in the CPU model, the JAX model, and the Immrax validator. The optimizer enforces only total `μmg` traction, while the validators check each wheel. The two can therefore disagree about whether a trajectory is safe.

**Fix (implemented):** The longitudinal term is gone; `nl = mg/2 - lateral` and `nr = mg/2 + lateral` come from one shared function used by all four consumers. The optimizer replaces the total-`μmg` diamond with `f_bound <= max_fl`, `f_bound <= max_fr`, and `f_bound <= cof_lo*(mg/2 - |lateral|_max)`, where `f_bound` is the worst-case wheel force over the declared inertia and track-width ranges (the same quantity Immrax's interval check bounds). On the `examples/example_complex_mission.json` evidence run, the fast path (6.0677 s) now reports motor/traction/slip `0.0000 N`; before the fix the same fast path failed with motor `0.229 N` and slip `1.197 N`.

**Motor limit (implemented earlier):** `RobotConfig.get_max_force_at_velocity_robust` and `get_max_force_symbolic_robust` use the declared torque-headroom lower bound, the smallest wheel radius in the speed term, and the largest wheel radius in the force term. The optimizer uses these per-wheel limits, and `DifferentialDriveModel.check_constraints` uses the same scalar limit when headroom is enabled. With the per-wheel force bound above, the CPU audit and Immrax now agree on the same limit at every sample.

**Workaround:** None needed.

---

### KI-011 — Heading interpolation crosses ±π the long way

**Status:** Fixed in the working tree (uncommitted). `resample_to_fixed_dt` unwraps the source heading before interpolating and wraps the result back to `[-π, π]`.

**Files:** `export.py` (`resample_to_fixed_dt`, `_wrap_pi`), `tests/test_export.py`

**Problem:** `resample_to_fixed_dt` linearly interpolates heading as a raw number, with no unwrap.

**Evidence:** Source headings `3.13` and `-3.13` represent a 0.0232 rad turn across ±π. Before the fix the interpolation went `3.13 → 1.878 → 0.626 → -0.626 → -1.878 → -3.13`, a π turn in the wrong direction. The optimizer itself emits a continuous heading, so no trajectory in the repository triggered the bug; it needed a hand-edited or external trajectory file.

**Fix (implemented):** `np.unwrap` the heading sequence before interpolating, then wrap each output heading back to `[-π, π]`. The same profile now reads `3.13 → 3.1346 → 3.1393 → -3.1393 → -3.1346 → -3.13`. Consumers that compute heading differences must use the shortest angle, as the deployed controllers do.

**Workaround:** None needed. A regression test covers the ±π case and the continuous-heading case.

---

### KI-012 — An event on the final waypoint never fires

**Status:** Fixed in the working tree (uncommitted). The event-capable controllers fire the final event after the loop, once the robot has arrived and the drivebase has stopped.

**Files:** `pybricks_code/ramsete.py` (`_fire_event`, `follow_trajectory`), `pybricks_code/ramsete_compact_events.py` (`follow_trajectory`), `tests/test_firmware_events.py`

**Problem:** The loop breaks when `t >= total_time`, and the sample advance stops at `num_samples - 2`. Events are checked only on `s1`, so the last sample is never `s1`. The compact event controller keeps the same break condition.

**Evidence:** The last segment of the optimizer allocates one extra sample (`optimizer.py:521-525`), so the final waypoint lands on the *last* sample. Solving `examples/example_with_events.json` puts the “release” event on sample 30 of 30, at `t = 4.637`, which is exactly `total_time`. The deployed loop never reached it, so a final “release” or “score” action was silently skipped — a normal mission pattern.

**Fix (implemented):** `ramsete.py` shares one `_fire_event` helper between the loop and the end of the run, and fires the last sample's event after `drivebase.stop()`. `ramsete_compact_events.py` fires the events that are left after the loop. Both run on the normal exit path only, so a crashed run does not fire the final action.

**Workaround:** None needed. The firmware modules are now exercised on the host with stubbed Pybricks modules, so the final event is covered by a regression test.

---

### KI-013 — Python export loses the robot config

**Status:** Fixed in the working tree (uncommitted).

**Files:** `main.py` (trajectory config), `export.py` (`write_python_file`)

**Problem:** `main.py` writes `config_data.get("robot", config_data.get("config", {}))`, which is `{}` for a flat config. `write_python_file` then reads only old key names such as `differentialTrackWidth`, `radius`, `vmax`, and `tmax`.

**Evidence:** A flat config previously exported `"config": {}`, and a modern `robot` config with canonical keys exported default track width, wheel radius, max speed, and torque. After the fix, `main.py` writes `config_data.get("robot", config_data.get("config", config_data))` and `write_python_file` reads the canonical fields through `RobotConfig`, so the generated Python file uses the same physical parameters as the trajectory. Regression tests cover both canonical and legacy configs.

**Fix (implemented):** `main.py` writes the canonical robot dict (or the flat dict when there is no wrapper), and `write_python_file` uses `RobotConfig` instead of old key names. See `TestExportConfig` in `tests/test_regressions.py`.

**Workaround:** None needed; regenerate the Python export instead of hand-editing it.

---

### KI-014 — JAX Ramsete uses a poor large-angle approximation

**Status:** Fixed in the working tree (uncommitted). The CPU/parity path normalizes the heading error to `[-π, π]`; the interval path keeps unrolled headings so the error is already continuous. Both use a degree-6 Chebyshev polynomial for `sinc` with maximum error below `2.5e-9` on `[-π, π]`, which Immrax can evaluate with interval arithmetic.

**Files:** `jax_ramsete.py`, `tests/test_jax_ramsete.py`

**Problem:** The controller replaced `sin(etheta)/etheta` with `1 - etheta²/6 + etheta⁴/120`. This is only accurate near zero.

**Evidence:** At `etheta = π`, the true value is 0, but the old approximation is 0.167. At `2π`, the old approximation is negative. Immrax used this controller inside its closed-loop reachability check.

**Fix (implemented):** The deployed `ramsete_fast.py` formula is the source of truth: exact sinc, shortest-angle error, and the precomputed/interpolated `sqrt(omega² + b·v²)` gain. `JAXRamseteController.calculate` accepts the precomputed gain and normalizes the error on the CPU path. The interval path uses unrolled headings and an interval-friendly polynomial because Immrax's natural inclusion does not implement `abs`, `where`, `mod`, `sinc`, or `atan2`. A randomized parity test compares the JAX command to a direct transcription of `ramsete_fast.py`, including ±π crossings. The deployed controller also rounds the `DriveBase.drive()` arguments, since Pybricks truncates float arguments toward zero; `jax_ramsete.quantize_drive_command` mirrors that rounding (whole mm/s, whole deg/s) at the drive boundary on the nominal feedback path, while the interval path leaves it off because Immrax cannot include those operations.

**Workaround:** None needed for the parity path. The interval validator still approximates cumulative command feedback with the tube midpoint (see KI-020).

---

### KI-015 — Random benchmark tools are broken

**Status:** Partially fixed in the working tree. `RandomBenchmark` now writes and validates against `config_used.json`; the pipeline benchmark still reports failed solves as normal results, and the random suite generator remains broken.

**Files:** `benchmark.py`, `examples/generate_random_suite.py`

**Problem:** `RandomBenchmark._validate` writes a trajectory with the config nested under `trajectory`, then calls `validate_trajectory(output_file, output_file)`. The second argument is read as a config file, so `RobotConfig` falls back to defaults. The generator reads a missing `robot_config.json` and treats `solve()`’s `(samples, stats)` tuple as the samples.

**Evidence:** Random benchmark safety results use default robot parameters. The random suite generator crashes or writes a malformed trajectory. `benchmark.py` also defaults to a `robot_config.json` that is not in the repo; CLI mode fails when it is missing.

**Fix:** Pass `self.config_data` directly or write a separate config file; unpack `samples, stats`; use an existing config such as `examples/fll_choreo.json`.

**Workaround:** Use the CLI benchmark with an explicit config, or skip random benchmark validation.

---

### KI-016 — The first live-view client never receives config or waypoints

**Status:** Fixed in the working tree (uncommitted).

**Files:** `live_visualizer.py` (deleted), `main.py`, `viz/index.html` (deleted)

**Problem:** `send_config` ran immediately after the server started, but `broadcast` returned when there were no clients. The README told users to open the browser afterward.

**Fix:** The last config, waypoints and trajectory were stored and replayed to every client that connects, so a reload or a late browser still showed the field. The live view was then retired along with KI-005: the studio loads the mission itself, so there is no first-client race to have.

## P2 — cleanup

### KI-017 — Convergence flags do nothing

**Status:** Partly fixed in the working tree (uncommitted).

**Files:** `main.py`, `multiverse_optimizer.py`

**Problem:** `--convergence-output` and `--convergence-animate` were parsed but never used. `MasterTrajectoryOptimizer` declares `iteration_history` but never fills it.

**Fix:** The two dead flags were removed, so the help no longer advertises options that do nothing. `--show-convergence` and `--convergence-mode` still work with the default simple solver, which fills the iteration history. Forwarding the history through the Multi-Verse pipeline is still open.

---

### KI-018 — Dead code, duplicates, and generated files

**Files:** `pybricks_code/ramsete_fast.py` (intentionally kept), `examples/complex_mission.py`, `pybricks_code/complex_mission.py`, `pybricks_code/complete_mission.py`, `immrax_validator.py:295`

**Problem:** Cleanup removed the duplicate `examples/complete_mission.py`, the root `test_jax_ramsete.py` (its boundary test moved to `tests/test_jax_ramsete.py`), `docs/API_optimizer.md`, generated example/controller/benchmark outputs, and unused imports plus `send_candidates`/`send_status`. `ramsete_fast.py` is intentionally kept: it is the production controller used on the real robot. Remaining: generated mission copies in `pybricks_code/` and `examples/complex_mission.py`; `get_safety_ribbon_data` is still unused and is no longer documented in `docs/WHITEPAPER.md`. The live view, its dead regenerate controls and the `live_visualizer.py` module are gone.

**Evidence:** The full suite now runs 194 tests with 0 skipped. `random_suite/run_000/plot.png` is kept because `README.md` embeds it as a historical example.

**Fix:** Decide whether the generated mission copies should be tracked; remove `get_safety_ribbon_data` or document it as a supported API. Keep `ramsete_fast.py`.

**Workaround:** None needed for runtime; this is maintenance load.

---

### KI-019 - The reachability tube lags the reference by one sample

**Status:** Fixed in the working tree (uncommitted). Each interval is now driven by the same linear reference and 20 ms sample-and-hold loop as `pybricks_code/ramsete_fast.py`, with the plant integrated in 5 ms substeps. Tracking error and envelope radius are reported separately and checked independently; the envelope is checked against `main.py`'s 2 cm transit threshold.

**Files:** `immrax_validator.py` (`compute_reachability`, `_interpolated_reference`, `RamseteController`), `jax_ramsete.py`

**Problem:** At step `i`, the interval tube was propagated with sample `i`'s reference pose and velocity over `[t_i, t_{i+1}]`, but the result was compared to sample `i+1`. Sample 0 is at rest, so the tube started one step behind and stayed there. The tracking error was roughly one sample spacing even with zero uncertainty.

**Evidence:** On the straight example with every uncertainty collapsed to nominal (`cof=1.37`, torque margin `1.0`, zero backlash, exact track width/wheel radius/inertia), the old model reported `max_tracking_error_m=0.1585 m` against a 5 mm threshold while the envelope radius was only 1.5-2 mm. After the fix, the same straight example reports `max_tracking_error_m≈0.00346 m` and zero motor, slip, and speed violations; the envelope is 0.0028 m under the current narrowed test ranges. The reference is linear (not Hermite) because that is what `ramsete_fast.py` follows.

**Fix (implemented):** Use linear pose/velocity interpolation and the precomputed Ramsete gain from `ramsete_fast.py`, update the reference and controller once per 20 ms tick, hold the command, and integrate the plant in 5 ms substeps. Do not relax the 5 mm tracking or 2 cm envelope thresholds to hide a model error.

**Workaround:** None needed for the straight example. Some representative paths (S-curve, stops, events) still fail until the optimizer is driven by the same tracking model; the complex mission passes under the narrowed test ranges (KI-020).

### KI-020 — Validator is not yet a full digital twin or an optimizer objective

**Status:** Open, partially addressed in the working tree. The controller parity, linear reference, 20 ms sample-and-hold, and physical backlash model are implemented. The controller split is also done: `robot_config.py` is hardware-only, `mission_4_ramsete.py` imports `ramsete_fast` with its calibrated defaults, and `mission_5_lambdas.py` imports `ramsete` for events. `pybricks_code/sysid_digital_twin.py` and `sysid_fit.py` provide the robot log/capture and host fit path; no hardware logs have been collected yet. The host can now push a script to the hub and capture its console output without the Pybricks IDE (`hub_deploy.py`, `main.py --push-hub`), so the robot log no longer has to be copied out by hand, and a pushed trajectory reports its measured final pose as a `POSE` line. The fitted numbers now carry an uncertainty: the calibration-critical tests run three times each, the fit reports a 95% t-interval per parameter over those repeats, and the validator plans against that interval instead of the spread of the steps inside one run, which was not an uncertainty because the steps are sequential. The interval covers the run-to-run variation and any drift over the session, but not the error of the tape measure: the tape error is in the numerator of every measurement, it is the same every time, and no number of repeats averages it out. The host tools accept a minimal drivebase profile (`wheel_radius_m`, optional `track_width_m`) instead of a full robot config. `digital_twin.py` and `main.py --sysid-fit` now load and validate a fit. When enabled, the validator uses a five-state plant with actual wheel speeds, the fitted straight-line gain interval, first-order time constant, command delay, and odometry distance/heading scale in feedback. The nominal feedback path also mirrors the firmware's rounded `DriveBase.drive()` arguments (whole mm/s, whole deg/s). Braking slip is now applied by the plant. Reachability is a post-solve candidate score (`--candidates`), not yet an optimizer term. The firmware's DriveBase trajectory limits are now known (about 7800 mm/s² and 9300 deg/s² for this robot, derived from Pybricks' 20,000 deg/s² ceiling), so the plant's instant-tracking assumption matches the hub; Motor/current transients, DriveBase PID, and slip under load remain unimplemented; the whole-millimeter distance reading was measured rather than assumed and costs 0.75 mm of final drift on the 6.08 s complex mission, so it is not worth modelling. The machinery for the rest is built and inert until the fit provides the numbers: the controller is fed the robot's own integrated odometry belief instead of the truth, the plant applies the measured braking shortfall and the measured acceleration limit, and the belief drifts with the measured gyro bias. `docs/DIGITAL_TWIN_PLAN.md` records the data contract, what is built, and what is still open. The measurement loop around them is now complete on the host: `main.py --push-hub --hub-log session.log` captures the hub console output, `sysid_fit.py --from-hub-session session.log --out-poses poses.csv` turns its `POSE` lines into the measured-pose CSV that `sysid_compare.py` reads, and the sysid battery feeds `sysid_fit.py` the same way. The measured poses are matched to the sysid log by `run_id`, and only `odometry_straight`, `odometry_turn` and `brake_from_*` runs contribute a distance scale, a heading scale or a braking-slip estimate, so a poses file with the wrong run id is silently ignored ("no measured poses supplied").

**Files:** `optimizer.py`, `immrax_validator.py`, `jax_ramsete.py`, `digital_twin.py`, `main.py`, `multiverse_optimizer.py`, `pybricks_code/ramsete_fast.py`, `pybricks_code/robot_config.py`, `pybricks_code/mission_4_ramsete.py`, `pybricks_code/mission_5_lambdas.py`, `pybricks_code/sysid_digital_twin.py`, `pybricks_code/hub_runner.py`, `pybricks_code/ramsete_compact_events.py`, `sysid_fit.py`, `sysid_compare.py`, `sysid_collect.py`, `hub_deploy.py`, `examples/drivebase_profile.example.json`, `tests/test_sysid_fit.py`, `tests/test_sysid_compare.py`, `tests/test_digital_twin.py`, `tests/test_hub_deploy.py`, `tests/test_sysid_collect.py`

**Problem:** The default validator mirrors the deployed controller's math and timing, but its plant is still kinematic: commanded wheel speeds become motion immediately (apart from wheel backlash). The optional fitted plant adds actuator lag, delay, gain uncertainty, and a static odometry correction, but it still has no motor current/torque transient, DriveBase PID, integrated encoder/gyro odometry, or wheel-slip dynamics. Feedback uses a scaled version of the true pose, not the robot's actual `DriveBase.state()` estimate. Reachability is a post-solve candidate score (`--candidates`, `main.py`), not yet a term or constraint in the optimizer.

**Evidence:** The straight example passes with `max_tracking_error_m≈0.00346 m` and `max_envelope_radius_m≈0.0028 m` under the narrowed test ranges. Representative paths still expose the gap: the S-curve example reports `max_tracking_error_m≈0.0132 m` and `max_envelope_radius_m≈0.0077 m`; the stops example reports 0.0052-0.0060 m / 0.0093-0.0095 m; the events example reports 0.0053-0.0054 m / 0.0097 m; the complex mission now passes at 6.068 s with 0.0049 m / 0.0122 m. These are model predictions, not hardware measurements. There is no real tracking log in the repository yet; the logger and fitter have only been exercised with synthetic data. A synthetic fit with `gain=1.0, tau=0.08 s` already rejects the straight example with 3.44 cm tracking error, which shows why hardware calibration matters before enabling the fitted model. The old `accuracy_weight += 2.0` retry is removed: on the complex mission the slow 17.1 s candidate diverged, while the 6.07 s one stayed bounded. A uniform-timing five-seed sweep (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`) shows that on the complex mission all four non-baseline seeds converge to slower, wider paths (8.1-8.7 s, 1.55-1.76 cm, tracking 0.53-0.72 cm), so the 6.07 s / 1.22 cm baseline still wins. Guess diversity alone does not beat the equal-duration baseline; the default simple-path candidates now mix the two timing modes (KI-022). Per-segment timing is path-dependent: on an uneven 1.0 m + 0.2 m path, `accuracy_weight=0.005` passes all gates at 2.784 s / 0.32 cm envelope / 0.36 cm tracking versus 4.456 s / 0.37 cm / 0.36 cm equal-duration at the same weight, so the mixed loop selects per-segment; at `accuracy_weight=0` per-segment is 2.710 s / 0.32 cm but fails the Immrax and CPU gates, so the mixed loop falls back to the equal-duration candidate (4.463 s / 0.35 cm). On the complex mission at `accuracy_weight=0` per-segment is faster (4.650 s / 1.00 cm) but its tracking is 0.84 cm and its motor force is 0.074 N over the limit; at `accuracy_weight=0.02` it is 7.333 s / 1.58 cm with 0.53 cm tracking, still over the 5 mm gate, and worse than the uniform 6.068 s / 1.22 cm baseline. The candidate loop now mixes both timing modes by default (`--uniform-dt`/`--per-segment-dt` force one) and runs the CPU audit on every candidate, so a partial pass cannot block a later fully passing candidate.

**Fix:** Collect and fit in one command with `python sysid_collect.py --conn ble --profile examples/drivebase_profile.example.json`: it pushes the logger, captures the session as it arrives, converts the hub's `POSE` lines to `poses.csv`, and writes `sysid_fit.json`. The poses are the hub's own odometry, so the distance and heading scale of the fit have no ground truth and stay at 1.0 until someone measures the robot with a tape and replaces those rows; the actuator part of the fit does not need a measurement. The written procedure is `docs/CALIBRATION_PROCEDURE.md`: the operator drives the robot through the collector tests, measures the end pose of the five that need a tape, and fills in a measurement sheet keyed by test name. `python sysid_fit.py <session.csv> --from-sheet <sheet> --out-poses poses.csv` turns that sheet into the measured poses, so the order the tests ran in does not matter and a skipped test is reported instead of guessed. The same thing step by step: push the logger and capture its session with `python hub_deploy.py pybricks_code/sysid_digital_twin.py --out session.csv` (or run it from the Pybricks IDE), record the stop/final poses in `poses.csv`, fit the logs with `sysid_fit.py --profile examples/drivebase_profile.example.json`, and pass the result with `main.py --sysid-fit sysid_fit.json --validate`. Compare the planned trajectory with the measured poses using `sysid_compare.py`; it prints position/heading errors and whether each measured pose is inside the reachability envelope, and it accepts a plain predicted-pose CSV and a drivebase profile. The pushed run gives the same comparison without hand-editing: `main.py --push-hub --hub-log session.log`, then `sysid_fit.py --from-hub-session session.log --out-poses poses.csv --run-id <id>`, then `sysid_compare.py output.traj poses.csv --run-id <id>`. Extend the plant to integrated odometry and a braking-slip state only after that comparison. The post-solve choice is made: `--candidates` scores several solutions (seed/timing combinations with `--simple`, accuracy-weight variants on the Multi-Verse path) and picks the passing one with the lowest envelope (then the shortest). Extend it to a hard constraint or differentiable proxy only if the candidate set cannot find a passing path. Do not tune the 5 mm/2 cm thresholds or b/zeta to one path.

**Workaround:** Use the straight example as a parity smoke test; treat predictions for curved/high-speed paths as uncalibrated until hardware logs are available. For planner testing, `examples/fll_choreo.json` temporarily narrows `track_width_range` to ≈±0.1% around 0.0965 m and `gear_backlash_m` to [0.4, 0.5] mm; these are test assumptions, not measured hardware values, and should be widened again once calibration data exists.

### KI-021 — The DriveBase silently caps the acceleration the plan asks for

**Status:** Fixed in the working tree (uncommitted). `robot_config.py` raises the DriveBase trajectory limits to 80% of the firmware ceiling for this robot's geometry, which is about 7800 mm/s² and 9300 deg/s² with 56 mm wheels and a 96.5 mm track. The pathing controllers read `drivebase.angle()` instead of `drivebase.state()`, round the `DriveBase.drive()` arguments, and collect garbage once per second instead of once per tick. The JAX twin mirrors the rounding.

**Files:** `pybricks_code/robot_config.py`, `pybricks_code/ramsete_fast.py`, `pybricks_code/ramsete_compact.py`, `jax_ramsete.py`, `immrax_validator.py`, `tests/test_jax_ramsete.py`

**Problem:** `DriveBase.drive()` does not command the motors directly. It starts a timed trajectory for the distance controller and another for the heading controller, and those trajectories ramp at `settings.acceleration`. Pybricks defaults that to three quarters of the motor's 2000 deg/s², which is 1500 deg/s²: about 733 mm/s² for 56 mm wheels and 1500 deg/s² of chassis rotation. Every call re-plans the trajectory and ramps at that limit, so the inner reference could move at most about 0.015 m/s per 20 ms tick.

Two smaller losses sat on top of it: `drivebase.state()` returns the heading rounded to whole degrees even with the gyro enabled, and Pybricks truncates float arguments toward zero, so `drive()` turned small steering corrections into zero commands.

**Evidence:** The optimized missions reach 5.38 m/s² and 8.32 k deg/s² (`tmp_mixed_mission.traj`), roughly ten times the DriveBase defaults. Pybricks' own ceiling is 20,000 deg/s² in motor units, which is 9775 mm/s² and 11,607 deg/s² for this geometry, so the default was the only thing in the way. The validator never saw it: `validator.forward_integrate` integrates the planned wheel speeds, so the 1.22 cm figure for the complex mission is reference consistency, not hardware tracking (KI-020).

**Fix (implemented):** `robot_config.py` calls `drivebase.settings(straight_acceleration=..., turn_acceleration=...)`. The values are derived from the wheel diameter and track width at 80% of the firmware ceiling, so a different robot geometry cannot make the call fail at import time. `ramsete_fast.py` and `ramsete_compact.py` use `drivebase.angle()`, which keeps the gyro's fractional degrees, and round the `drive()` arguments so the truncation bias is gone. `jax_ramsete.quantize_drive_command` mirrors that rounding in the validator.

**Workaround:** None needed. These are upper limits, not commands, so raising them does not make the robot jerkier; traction and the motor controller still bound the acceleration. If the robot slips or rings on the first hardware runs, lower `straight_acceleration` / `turn_acceleration` or add `Motor(profile=...)` tuning rather than reverting the whole change. The Pybricks code has still not been run on hardware.

### KI-022 — The per-segment timing model was not held to the same wheel-jerk discipline

**Status:** Fixed in the working tree (uncommitted). Two defects, both in how the timing model treated wheel jerk. The smoothness cost is now the path-wide reference interval squared times the sum of the squared physical jerks, so every boundary pays the same per unit of physical jerk in either mode, and the reported `max_jerk_m_s3` metric uses the same term set and the same midpoint spacing as the solver. The uniform branch keeps its original expression, so the validated uniform results are unchanged.

**Files:** `optimizer.py`, `validator.py`, `main.py`, `tests/test_optimizer.py`

**Problem:** Two separate issues made the per-segment model a loophole.

First, "uniform dt" is not a neutral time base. Every segment has the same sample count, so sharing one `dt` across segments also forces every segment to take the SAME time. On a path with unequal segment lengths the short segment inherits the long segment's duration and the robot crawls through it, then has to make the time back up with wheel motion that is not monotone. The uniform mode is therefore not the general model: per-segment `dt` is, since setting every `dt_i` equal recovers the uniform solution exactly.

Second, and worse, the smoothness cost penalised the raw acceleration jump `(al_k1 - al_k)^2`, which is proportional to `jerk * dt`. Under one shared `dt` every boundary was weighted the same, so the penalty was effectively proportional to the physical jerk. Under per-segment `dt` the weight became `dt_k^2`, so a segment with a short interval was charged `dt^2` less per unit of physical jerk and could buy violent wheel motion almost for free. The same `accuracy_weight` therefore meant two different things in the two modes, which made the candidate comparison invalid.

**Evidence:** On a 1.0 m + 0.2 m path with no intermediate stops at `accuracy_weight=0.005`, the per-segment model reached 19.28 m/s^3 of peak wheel jerk against the uniform model's 9.52 m/s^3, and at `accuracy_weight=0.02` it reached 9.53 against 6.71. The peak jerk was also reported wrongly: `compute_metrics` divided by `dt_i` instead of the spacing between the two interval midpoints, and differenced the last sample's `al`, which the formatter sets to 0 because it has no following interval, so a short final interval reported a large fake jerk (37.22 m/s^3 where the physical jerk was 9.52).

After the fix the same path gives 9.52 m/s^3 in both modes at `accuracy_weight=0.005` (per-segment 2.787 s against the old 2.732 s, uniform unchanged at 4.456 s), and 7.95 against 6.71 at `accuracy_weight=0.02`. The per-segment objective is never worse than the uniform one at the same weight, because per-segment `dt` is a strictly larger feasible set.

**Fix (implemented):** `optimizer.py` normalises the smoothness term by the path-wide mean interval and the spacing between the two interval midpoints, so the cost is `dt_ref^2 * sum(J_k^2)` in both modes and reduces to the uniform expression when every `dt` is equal. The uniform branch keeps its original expression so its results stay bit-identical. `validator.py` uses the same term set and the same midpoint spacing for the reported `max_jerk_m_s3`. `main.py` labels the two modes by what they constrain ("equal-duration" versus "per-seg") and prints each candidate's segment times, so a uniform win or loss is self-explanatory.

**Workaround:** None needed. The jerk discipline is now the same in both modes, so `--accuracy-weight` means the same thing regardless of the timing model. Do not compare candidates across timing modes at different accuracy weights.

### KI-023 — The SysId collector could not be pushed to a hub and its metadata did not parse

**Status:** Fixed in the working tree (uncommitted). The collector now compiles for MicroPython, prints its metadata in the format the host parses, and reports its measured pose. The whole log-and-pose path can be run over Bluetooth from the host (`hub_deploy.py`, `sysid_collect.py`). The heading scale of the fit is now a positive number, so the fit it produces can be loaded.

**Files:** `pybricks_code/sysid_digital_twin.py`, `pybricks_code/hub_runner.py`, `pybricks_code/ramsete_compact_events.py`, `hub_deploy.py`, `sysid_collect.py`, `sysid_fit.py`, `main.py`, `tests/test_hub_deploy.py`, `tests/test_sysid_collect.py`, `tests/firmware_stubs.py`

**Problem:** Three separate defects, all invisible to the host tools because the file is valid Python.

First, the collector could not be compiled for MicroPython at all: it built its CSV rows from several adjacent f-strings, and MicroPython cannot concatenate two f-strings. `mpy-cross` rejects that with a bare "SyntaxError: invalid syntax", so the file could be run from the Pybricks IDE but not pushed to a hub by any tool.

Second, the collector printed its metadata as `META,run_id 12` (space separated) while the host parser splits `META,key,value` on commas. The run id therefore never reached the fit, so no measured pose could ever be attached to its test, and the odometry part of the fit always reported "no measured poses supplied".

Third, the fit divided the measured heading by the odometry angle without noticing that the two use opposite signs. The measured heading is counter-clockwise positive, like the planned trajectory, but `drivebase.angle()` is clockwise positive. On any robot that runs the turn test the ratio came out as -1, and `digital_twin.py` refuses a heading scale of zero or less, so the fit could not be loaded at all. Even without that check the heading feedback would have been inverted.

**Evidence:** `mpy-cross` fails on `pybricks_code/sysid_digital_twin.py` with "SyntaxError: invalid syntax"; the same failure is reproducible with `print(\n f"{x}"\n f"{x}"\n)`. With the META format fixed, `parse_session` reads `run["meta"]["run_id"]`, and `poses_from_hub_session` maps each `POSE` line to the run id of the test that was open when it was printed. A turn test that reads +90 in `db_a_deg` and -90 in the pose produced `heading_scale = -1.0`, and `load_sysid_fit` refused it with "odometry.heading_scale must be > 0, got -1.0"; the same fit loads at +1.0.

**Fix (implemented):** The firmware builds its rows with one `%` format per statement, matching the other firmware modules. The metadata is printed as `META,key,value`. The collector reports its pose with `POSE,x_m,y_m,heading_deg`, integrated from the hub's own distance and angle, with the heading counter-clockwise positive so it lines up with the planned trajectory. The fit flips the sign of the odometry angle when it divides the two, so the scale between them is positive. `tests/test_hub_deploy.py::TestFirmwareCompiles` compiles every file in `pybricks_code/` with `mpy-cross`, so the compiler defect cannot come back silently, and `tests/test_sysid_collect.py::TestFittedScales` checks the fitted scales, so the sign defect cannot either.

**Workaround:** None needed. Note that the reported pose is the robot's own odometry, not a measurement: the distance scale, the heading scale and the brake slip of the fit still need a tape measurement to be calibrated (KI-020). The actuator part of the fit does not. The operator's measurements go in a sheet keyed by test name, which `sysid_fit.py --from-sheet` turns into the measured poses, so the measurement convention and the run-id bookkeeping stay in one place.

### KI-024 — The timing model had no rule for where the time allocation may change

**Status:** Fixed in the working tree (uncommitted). A waypoint is now classified once, and the per-piece timing model only lets the *reading* of the motion profile change where the state at the point is fully determined. Where it is not, the wheel acceleration — and so the wheel force — is now required to be continuous through the point. The equal-duration model is unchanged and its validated results are bit-identical.

**Files:** `optimizer.py`, `tests/test_optimizer.py`

**Problem:** The per-piece timing model gave every waypoint its own sample interval, so the time scale could change anywhere. A change of interval across a waypoint has a physical consequence that the model did not account for: the acceleration on the interval before the waypoint is `(vl[k]-vl[k-1])/dt_prev` and the one after it is `(vl[k+1]-vl[k])/dt_next`. When `dt` changes there, the *same* velocity ramp is read as two different accelerations, so the wheel force profile gets a step exactly at the point the path is supposed to run through smoothly.

The boundary semantics were never written down, so the model could not tell a legitimate timing boundary from an illegitimate one. The four cases are:

| waypoint | what the pieces must agree on | timing |
|---|---|---|
| stop and pinned heading | nothing: position, heading and zero velocity are all determined | free |
| pinned heading, no stop | the wheel velocities there | the force profile must be continuous |
| stop, no pinned heading | the heading there; the velocity is zero on both sides | the force profile must be continuous |
| neither | the heading and the wheel velocities | the force profile must be continuous |

**Evidence:** On the complex mission the equal-duration solution steps the wheel acceleration by 3.65 m/s^2 at the waypoint at (0.320, 0.150) m — a pinned heading the robot drives through — and the per-piece solution stepped it by 2.99 m/s^2 while re-timing there. After the fix the per-piece solution's acceleration step at every waypoint the robot does not stop at is 1.2e-4 m/s^2 (equal-duration, 3.65 m/s^2) at `accuracy_weight=0` and 1.1e-3 m/s^2 at `accuracy_weight=0.005`. The step is not exactly zero because IPOPT's absolute constraint tolerance is 1e-2 and the constraints are scaled to acceleration units; the raw residual was in metres, where the same tolerance would have allowed a centimetre of velocity ramp.

The fix also removes two spurious candidate rejections. On `example_mission_first4.json` the per-piece candidate at `accuracy_weight=0` was 1.661 s but failed the gates, and the equal-duration candidate at 2.176 s won; it is now 1.833 s and passes. On `example_with_events.json` the per-piece candidate at `accuracy_weight=0` was 4.896 s and failed; it is now 4.416 s and passes. The complex mission is unchanged: the equal-duration candidate still wins at 6.068 s with a 1.22 cm envelope.

**Fix (implemented):** `TrajectoryOptimizer.merge_conditions` classifies every waypoint and is the single place the rule lives. In the joint NLP the pieces already agree on the state at a waypoint, because position, heading and the wheel velocities are single shared variables, so only the force continuity had to be added: for every waypoint the robot does not stop at, the acceleration on the interval before it must equal the acceleration on the interval after it. A waypoint where the robot stops is exempt, because the wheel force has to reverse sign when the robot comes to rest and goes again. The constraints are scaled by the reference interval squared so they read in m/s^2, which also fixed the solver convergence: the complex mission's per-piece candidate went from 6.836 s in 41 iterations to 4.650 s in 203.

**Workaround:** None needed. Note the deeper cause of the equal-duration model's penalty is the *sample layout*, not the timing: every piece gets the same number of samples, so one shared `dt` gives every piece the same time regardless of its length. A 1.2 m straight line solved as one piece with 20 samples takes 2.702 s with no path excess, while the same line with an interior waypoint and one shared `dt` takes 4.463 s and weaves 19.9 mm. Allocating the samples along the path by arc length would remove that penalty and let the timing change only at real boundaries; it changes the sample count and the sample-to-waypoint mapping, so it is a separate change and is not done here.

---

### KI-025 - The backlash was subtracted from the commanded wheel speed

**Status:** Fixed in the working tree (uncommitted). The backlash is now a position uncertainty in the reachable set, and the closed-loop heading error is reported next to the position error.

**Files:** `immrax_validator.py` (`RobotOpenLoop.f`, `DigitalTwinOpenLoop.f`, `compute_reachability`), `main.py` (the validation report), `tests/test_validator_model.py`, `docs/research/RQ-A_tracking_error.md`

**Problem:** The plant modelled the gear backlash as

```python
vl_act = vl_cmd - (backlash / 2.0) * jnp.tanh(vl_cmd / epsilon)
```

`gear_backlash_m` is a distance in metres; `vl_cmd` is a velocity in m/s. Subtracting one from the other is a dimensional error. Two things followed from it. The robot ran permanently `backlash/2` slower than commanded, in both directions, because the `tanh` saturates for any command above 1 mm/s. And that deficit grew into a pure position lag of `backlash/2 * t`, which was reported as a tracking error.

A real gear backlash is a *position* dead-band: it is taken up once at a direction reversal, after which the wheel turns at exactly the commanded speed. The old model had the speed wrong and the position wrong, in a way that put a systematic bias into the metric that claims to measure tracking.

**Evidence:** Collapsing the backlash interval to zero changed the tracking error by 0.1 % on the S-curve but by 100 % on a straight plan, where the lag was the entire error (226 um on a 1 m straight, and the backlash was the only uncertainty that reached the tracking error at all: `cof`, `torque_headroom`, `wheel_radius_range` and `inertia_kg_m2` are consumed only by the constraint check and never touch the propagated state). The measured lag was 0.226 mm/s, matching `backlash/2` to three digits. See `docs/research/RQ-A_tracking_error.md` finding F2.

**Fix (implemented):** The plant turns the wheel at the commanded speed. The backlash is carried in the reachable set instead: the reachable set is widened by `backlash/2 * t`, symmetrically, at each sample. The worst-case total uncertainty at each sample is therefore unchanged - the same magnitude the plant used to bias the tracking error by, moved from the tracking metric into the envelope and made symmetric rather than one-sided. The gate is not made more permissive by the change; the tracking criterion stops reporting a systematic lag that was never a tracking error.

**Also fixed:** the tracking metric is position-only, so a trajectory could follow the path while arriving at the wrong heading, and nothing measured it. `compute_reachability` now reports `max_heading_error_rad` and the machine-readable validation report carries `heading_error_deg` per candidate. It is reported, not gated: the heading behaviour is not yet calibrated, so it must not become a pass/fail criterion on its own.

**Measured effect on the example missions.** Over the whole 60-candidate matrix, before and after the fix: the tracking error moved by a mean of −0.0008 cm (range −0.025 to +0.029 cm), the envelope radius grew by a mean of +0.0070 cm (range −0.058 to +0.061 cm), and the sum of the two moved by a mean of +0.0062 cm. **No candidate's pass flag changed** (0 of 60), and no selection changed. The worst-case total per run moved by at most 0.065 cm, against a 5 mm tracking threshold and a 2 cm envelope threshold. The tracking error barely moved because it is dominated by the discretisation mismatch, not by the backlash; what changed is *where* the backlash is accounted for.

**Workaround:** None needed. The physically tighter model would cap the backlash at the dead-band width instead of letting it accumulate, which is a smaller envelope than the one implemented here; that needs hardware data to justify (KI-020), so the conservative form was kept.

---



1. KI-001, KI-002, KI-003, KI-005, KI-006, KI-007, KI-008, KI-009, KI-010, KI-011, KI-012, KI-013, KI-014, KI-016, KI-019, KI-021, KI-022, KI-023, KI-024, and KI-025: done in the working tree; never save or ship a failed or unvalidated run, keep validation on the configured robot, use both uncertainty bounds, keep the reference tube aligned with the samples, keep the optimizer and Immrax on the same per-wheel force model, hold both timing models to the same wheel-jerk discipline, keep the wheel force profile continuous through every waypoint the robot does not stop at, keep the backlash in the reachable set rather than in the wheel speed, never take a port from another program, and keep the firmware compilable for MicroPython.
2. KI-020: controller split, log/fit tooling, and the optional `--sysid-fit` actuator/odometry path are done, and the log and pose capture now run over Bluetooth from the host; collect hardware data, compare fitted predictions with measured poses, add integrated odometry/slip, then use reachability in optimization.
3. KI-004: either make Multi-Verse real or remove it.
4. KI-015: restore trustworthy benchmarks.
5. KI-017 and KI-018: remove or repair the remaining dead code.

## Not checked

This list does not cover `sysid_solver.py` beyond its imports, the plotting and animation modules in depth, the Pybricks tutorials, or the generated trajectory and benchmark data. The Pybricks code was not run on hardware.
