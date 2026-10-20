# Known Issues

Last reviewed: 2026-10-19

Scope: whole repository, reviewed as a single-user, offline desktop tool for one FLL robot. This is not a multi-user or server-scale review.

Status: KI-001, KI-002, KI-003, KI-005, KI-006, KI-007, KI-008, KI-009, KI-010, KI-011, KI-012, KI-013, KI-014, KI-016, KI-019, and KI-021 are **fixed in the working tree** (uncommitted). All other items are open; KI-015 and KI-017 are partially fixed. KI-020 records the remaining digital-twin and optimization work. The user-facing docs were reconciled with this tree on 2026-10-19; the code remains the authority if they diverge again.

How to use this list: refer to an item by its ID, for example “fix KI-003”.

## Quick triage

| ID | Priority | Area | Issue |
|---|---|---|---|
| KI-001 | P0 | Solver / output | ~~A failed solve is exported as a successful trajectory~~ (fixed) |
| KI-002 | P0 | Validation | ~~`--validate` cannot block output or change the exit code~~ (fixed) |
| KI-003 | P0 | Safety | ~~Immrax ignores upper uncertainty bounds~~ (fixed) |
| KI-004 | P0 | Multi-Verse | Refinement has no effect on the final trajectory |
| KI-005 | P0 | Live view | ~~The visualizer kills whatever owns port 8765~~ (fixed) |
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
| KI-016 | P1 | Live view | ~~The first client never receives config or waypoints~~ (fixed) |
| KI-017 | P2 | CLI | Convergence flags do nothing (dead flags removed) |
| KI-018 | P2 | Maintenance | Dead code, duplicates, and generated files |
| KI-019 | P0 | Safety | ~~The reachability tube lags the reference by one sample~~ (fixed) |
| KI-020 | P1 | Digital twin | Validator/optimizer do not yet model motor, odometry, slip, or reachability-driven optimization |
| KI-021 | P1 | Robot runtime | ~~The DriveBase caps the acceleration the plan asks for~~ (fixed) |

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

**Status:** Fixed in the working tree (uncommitted).

**Files:** `live_visualizer.py`, `main.py`

**Problem:** `LiveVisualizer.start()` called `_kill_process_on_port`, which used `netstat` + `taskkill` on Windows and `fuser -k` on POSIX. It did not check which process owned the port.

**Fix:** `_kill_process_on_port` is gone. The port is probed before the server starts; a busy port raises a clear "already in use" error, and `--port` picks a different one. The forced `reuse_address=True` was removed as well, so a busy port fails instead of being taken over. `tests/test_cli_ux.py` covers both paths.

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

**Files:** `live_visualizer.py`, `main.py`, `viz/index.html`

**Problem:** `send_config` ran immediately after the server started, but `broadcast` returned when there were no clients. The README told users to open the browser afterward.

**Fix:** The last config, waypoints and trajectory are stored and replayed to every client that connects, so a reload or a late browser still shows the field. The dead regenerate and candidate panels were removed from `viz/index.html`, and `--live` now opens the browser itself.

## P2 — cleanup

### KI-017 — Convergence flags do nothing

**Status:** Partly fixed in the working tree (uncommitted).

**Files:** `main.py`, `multiverse_optimizer.py`

**Problem:** `--convergence-output` and `--convergence-animate` were parsed but never used. `MasterTrajectoryOptimizer` declares `iteration_history` but never fills it.

**Fix:** The two dead flags were removed, so the help no longer advertises options that do nothing. `--show-convergence` and `--convergence-mode` still work with the default simple solver, which fills the iteration history. Forwarding the history through the Multi-Verse pipeline is still open.

---

### KI-018 — Dead code, duplicates, and generated files

**Files:** `pybricks_code/ramsete_fast.py` (intentionally kept), `examples/complex_mission.py`, `pybricks_code/complex_mission.py`, `pybricks_code/complete_mission.py`, `immrax_validator.py:295`

**Problem:** Cleanup removed the duplicate `examples/complete_mission.py`, the root `test_jax_ramsete.py` (its boundary test moved to `tests/test_jax_ramsete.py`), `docs/API_optimizer.md`, generated example/controller/benchmark outputs, and unused imports plus `send_candidates`/`send_status`. `ramsete_fast.py` is intentionally kept: it is the production controller used on the real robot. Remaining: generated mission copies in `pybricks_code/` and `examples/complex_mission.py`; `get_safety_ribbon_data` is still unused and is no longer documented in `docs/WHITEPAPER.md`; the live-view regenerate controls are still dead.

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

**Status:** Open, partially addressed in the working tree. The controller parity, linear reference, 20 ms sample-and-hold, and physical backlash model are implemented. The controller split is also done: `robot_config.py` is hardware-only, `mission_4_ramsete.py` imports `ramsete_fast` with its calibrated defaults, and `mission_5_lambdas.py` imports `ramsete` for events. `pybricks_code/sysid_digital_twin.py` and `sysid_fit.py` provide the robot log/capture and host fit path; no hardware logs have been collected yet. The host tools accept a minimal drivebase profile (`wheel_radius_m`, optional `track_width_m`) instead of a full robot config. `digital_twin.py` and `main.py --sysid-fit` now load and validate a fit. When enabled, the validator uses a five-state plant with actual wheel speeds, the fitted straight-line gain interval, first-order time constant, command delay, and odometry distance/heading scale in feedback. The nominal feedback path also mirrors the firmware's rounded `DriveBase.drive()` arguments (whole mm/s, whole deg/s). Braking slip is loaded and reported but not applied. Reachability is a post-solve candidate score (`--candidates`), not yet an optimizer term. The firmware's DriveBase trajectory limits are now known (about 7800 mm/s² and 9300 deg/s² for this robot, derived from Pybricks' 20,000 deg/s² ceiling), so the plant's instant-tracking assumption matches the hub; Motor/current transients, DriveBase PID, full encoder odometry, slip dynamics, and the remaining whole-millimeter distance reading remain unimplemented.

**Files:** `optimizer.py`, `immrax_validator.py`, `jax_ramsete.py`, `digital_twin.py`, `main.py`, `multiverse_optimizer.py`, `pybricks_code/ramsete_fast.py`, `pybricks_code/robot_config.py`, `pybricks_code/mission_4_ramsete.py`, `pybricks_code/mission_5_lambdas.py`, `pybricks_code/sysid_digital_twin.py`, `sysid_fit.py`, `sysid_compare.py`, `examples/drivebase_profile.example.json`, `tests/test_sysid_fit.py`, `tests/test_sysid_compare.py`, `tests/test_digital_twin.py`

**Problem:** The default validator mirrors the deployed controller's math and timing, but its plant is still kinematic: commanded wheel speeds become motion immediately (apart from wheel backlash). The optional fitted plant adds actuator lag, delay, gain uncertainty, and a static odometry correction, but it still has no motor current/torque transient, DriveBase PID, integrated encoder/gyro odometry, or wheel-slip dynamics. Feedback uses a scaled version of the true pose, not the robot's actual `DriveBase.state()` estimate. Reachability is a post-solve candidate score (`--candidates`, `main.py`), not yet a term or constraint in the optimizer.

**Evidence:** The straight example passes with `max_tracking_error_m≈0.00346 m` and `max_envelope_radius_m≈0.0028 m` under the narrowed test ranges. Representative paths still expose the gap: the S-curve example reports `max_tracking_error_m≈0.0132 m` and `max_envelope_radius_m≈0.0077 m`; the stops example reports 0.0052-0.0060 m / 0.0093-0.0095 m; the events example reports 0.0053-0.0054 m / 0.0097 m; the complex mission now passes at 6.068 s with 0.0049 m / 0.0122 m. These are model predictions, not hardware measurements. There is no real tracking log in the repository yet; the logger and fitter have only been exercised with synthetic data. A synthetic fit with `gain=1.0, tau=0.08 s` already rejects the straight example with 3.44 cm tracking error, which shows why hardware calibration matters before enabling the fitted model. The old `accuracy_weight += 2.0` retry is removed: on the complex mission the slow 17.1 s candidate diverged, while the 6.07 s one stayed bounded. A uniform-timing five-seed sweep (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`) shows that on the complex mission all four non-baseline seeds converge to slower, wider paths (8.1-8.7 s, 1.55-1.76 cm, tracking 0.53-0.72 cm), so the 6.07 s / 1.22 cm baseline still wins. Guess diversity alone does not beat the uniform baseline; the default simple-path candidates now mix uniform and per-segment timing. Per-segment timing is path-dependent: on an uneven 1.0 m + 0.2 m path, `accuracy_weight=0.005` passes all gates at 2.732 s / 0.32 cm envelope and 0.36 cm tracking versus 4.456 s / 0.37 cm / 0.36 cm uniform at the same weight, so the mixed loop selects per-segment; at `accuracy_weight=0` per-segment is 2.710 s / 0.32 cm but the CPU force audit rejects a 0.00057 N overshoot, so the mixed loop falls back to uniform (4.463 s / 0.35 cm). On the complex mission at `accuracy_weight=0` per-segment is faster (5.447 s / 1.18 cm) but fails the 5 mm tracking gate at 0.86 cm; at `accuracy_weight=0.02` it passes Immrax at 7.312 s / 1.61 cm, worse than the uniform 6.068 s / 1.22 cm baseline. The candidate loop now mixes both timing modes by default (`--uniform-dt`/`--per-segment-dt` force one) and runs the CPU audit on every candidate, so a partial pass cannot block a later fully passing candidate.

**Fix:** Run `pybricks_code/sysid_digital_twin.py` on the robot, record the stop/final poses in `poses.csv`, fit the logs with `sysid_fit.py --profile examples/drivebase_profile.example.json`, and pass the result with `main.py --sysid-fit sysid_fit.json --validate`. Compare the planned trajectory with the measured poses using `sysid_compare.py`; it prints position/heading errors and whether each measured pose is inside the reachability envelope, and it accepts a plain predicted-pose CSV and a drivebase profile. Extend the plant to integrated odometry and a braking-slip state only after that comparison. The post-solve choice is made: `--candidates` scores several solutions (seed/timing combinations with `--simple`, accuracy-weight variants on the Multi-Verse path) and picks the passing one with the lowest envelope (then the shortest). Extend it to a hard constraint or differentiable proxy only if the candidate set cannot find a passing path. Do not tune the 5 mm/2 cm thresholds or b/zeta to one path.

**Workaround:** Use the straight example as a parity smoke test; treat predictions for curved/high-speed paths as uncalibrated until hardware logs are available. For planner testing, `examples/fll_choreo.json` temporarily narrows `track_width_range` to ≈±0.1% around 0.0965 m and `gear_backlash_m` to [0.4, 0.5] mm; these are test assumptions, not measured hardware values, and should be widened again once calibration data exists.

### KI-021 — The DriveBase silently caps the acceleration the plan asks for

**Status:** Fixed in the working tree (uncommitted). `robot_config.py` raises the DriveBase trajectory limits to 80% of the firmware ceiling for this robot's geometry, which is about 7800 mm/s² and 9300 deg/s² with 56 mm wheels and a 96.5 mm track. The pathing controllers read `drivebase.angle()` instead of `drivebase.state()`, round the `DriveBase.drive()` arguments, and collect garbage once per second instead of once per tick. The JAX twin mirrors the rounding.

**Files:** `pybricks_code/robot_config.py`, `pybricks_code/ramsete_fast.py`, `pybricks_code/ramsete_compact.py`, `jax_ramsete.py`, `immrax_validator.py`, `tests/test_jax_ramsete.py`

**Problem:** `DriveBase.drive()` does not command the motors directly. It starts a timed trajectory for the distance controller and another for the heading controller, and those trajectories ramp at `settings.acceleration`. Pybricks defaults that to three quarters of the motor's 2000 deg/s², which is 1500 deg/s²: about 733 mm/s² for 56 mm wheels and 1500 deg/s² of chassis rotation. Every call re-plans the trajectory and ramps at that limit, so the inner reference could move at most about 0.015 m/s per 20 ms tick.

Two smaller losses sat on top of it: `drivebase.state()` returns the heading rounded to whole degrees even with the gyro enabled, and Pybricks truncates float arguments toward zero, so `drive()` turned small steering corrections into zero commands.

**Evidence:** The optimized missions reach 5.38 m/s² and 8.32 k deg/s² (`tmp_mixed_mission.traj`), roughly ten times the DriveBase defaults. Pybricks' own ceiling is 20,000 deg/s² in motor units, which is 9775 mm/s² and 11,607 deg/s² for this geometry, so the default was the only thing in the way. The validator never saw it: `validator.forward_integrate` integrates the planned wheel speeds, so the 1.22 cm figure for the complex mission is reference consistency, not hardware tracking (KI-020).

**Fix (implemented):** `robot_config.py` calls `drivebase.settings(straight_acceleration=..., turn_acceleration=...)`. The values are derived from the wheel diameter and track width at 80% of the firmware ceiling, so a different robot geometry cannot make the call fail at import time. `ramsete_fast.py` and `ramsete_compact.py` use `drivebase.angle()`, which keeps the gyro's fractional degrees, and round the `drive()` arguments so the truncation bias is gone. `jax_ramsete.quantize_drive_command` mirrors that rounding in the validator.

**Workaround:** None needed. These are upper limits, not commands, so raising them does not make the robot jerkier; traction and the motor controller still bound the acceleration. If the robot slips or rings on the first hardware runs, lower `straight_acceleration` / `turn_acceleration` or add `Motor(profile=...)` tuning rather than reverting the whole change. The Pybricks code has still not been run on hardware.

## Suggested fix order

1. KI-001, KI-002, KI-003, KI-005, KI-006, KI-007, KI-008, KI-009, KI-010, KI-011, KI-012, KI-013, KI-014, KI-016, KI-019, and KI-021: done in the working tree; never save or ship a failed or unvalidated run, keep validation on the configured robot, use both uncertainty bounds, keep the reference tube aligned with the samples, keep the optimizer and Immrax on the same per-wheel force model, and never take a port from another program.
2. KI-020: controller split, log/fit tooling, and the optional `--sysid-fit` actuator/odometry path are done; collect hardware data, compare fitted predictions with measured poses, add integrated odometry/slip, then use reachability in optimization.
3. KI-004: either make Multi-Verse real or remove it.
4. KI-015: restore trustworthy benchmarks.
5. KI-017 and KI-018: remove or repair the remaining dead code.

## Not checked

This list does not cover `sysid_solver.py` beyond its imports, the plotting and animation modules in depth, the Pybricks tutorials, or the generated trajectory and benchmark data. The Pybricks code was not run on hardware.
