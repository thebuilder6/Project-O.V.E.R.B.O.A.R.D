# Known Issues

Last reviewed: 2026-10-19

Scope: whole repository, reviewed as a single-user, offline desktop tool for one FLL robot. This is not a multi-user or server-scale review.

Status: KI-001, KI-002, KI-003, KI-007, KI-008, KI-009, KI-010, KI-013, KI-014, and KI-019 are **fixed in the working tree** (uncommitted). All other items are open; KI-015 is partially fixed. KI-020 records the remaining digital-twin and optimization work. The user-facing docs were reconciled with this tree on 2026-10-19; the code remains the authority if they diverge again.

How to use this list: refer to an item by its ID, for example “fix KI-003”.

## Quick triage

| ID | Priority | Area | Issue |
|---|---|---|---|
| KI-001 | P0 | Solver / output | ~~A failed solve is exported as a successful trajectory~~ (fixed) |
| KI-002 | P0 | Validation | ~~`--validate` cannot block output or change the exit code~~ (fixed) |
| KI-003 | P0 | Safety | ~~Immrax ignores upper uncertainty bounds~~ (fixed) |
| KI-004 | P0 | Multi-Verse | Refinement has no effect on the final trajectory |
| KI-005 | P0 | Live view | The visualizer kills whatever owns port 8765 |
| KI-006 | P0 | Install | A fresh install cannot run the default command |
| KI-007 | P1 | Tests | ~~The JAX test suite is silently skipped~~ (fixed) |
| KI-008 | P1 | Safety / config | ~~JAX validation reads a different config format~~ (fixed) |
| KI-009 | P1 | Safety / config | ~~Declared uncertainty ranges are ignored~~ (fixed) |
| KI-010 | P1 | Physics | ~~Weight-transfer math is wrong and tools disagree~~ (fixed) |
| KI-011 | P1 | Export | Heading interpolation crosses ±π the long way |
| KI-012 | P1 | Robot runtime | An event on the final waypoint never fires |
| KI-013 | P1 | Export | ~~Python export loses the robot config~~ (fixed) |
| KI-014 | P1 | Safety model | ~~JAX Ramsete uses a poor large-angle approximation~~ (fixed) |
| KI-015 | P1 | Benchmarks | Random benchmark tools are broken |
| KI-016 | P1 | Live view | The first client never receives config or waypoints |
| KI-017 | P2 | CLI | Convergence flags do nothing |
| KI-018 | P2 | Maintenance | Dead code, duplicates, and generated files |
| KI-019 | P0 | Safety | ~~The reachability tube lags the reference by one sample~~ (fixed) |
| KI-020 | P1 | Digital twin | Validator/optimizer do not yet model motor, odometry, slip, or reachability-driven optimization |

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

**Files:** `multiverse_optimizer.py:428-436`

**Problem:** Phase 4 computes a refined `global_traj`, but `_global_solve` builds `new_guess` and never passes it into `TrajectoryOptimizer.solve`; the local variable is unused. `TrajectoryOptimizer.solve` now exposes `guess_style` seeds, but Multi-Verse does not use them or the refined samples. Phase 5 re-solves from its default baseline guess.

**Evidence:** On `examples/example_complex_mission.json`, the default Multi-Verse run and `--simple` both produce 91 samples and the same 6.0677 s final cost; the default Multi-Verse run took 28.26 s wall time. `--workers`, `--no-parallel`, and `enable_parallel` cannot change the result because the refined guess and the thread pool are not used by the final solve.

**Fix:** Add a way to pass the refined sample matrix into `TrajectoryOptimizer.solve` and set it with `opti.set_initial`; the existing `guess_style` parameter only selects deterministic seeds. Or remove the Multi-Verse path and its CLI flags.

**Workaround:** Use `--simple` if you want predictable behaviour. Do not expect Multi-Verse or `--workers` to improve anything.

---

### KI-005 — The live visualizer kills whatever owns port 8765

**Files:** `live_visualizer.py:26`, `live_visualizer.py:30-53`

**Problem:** `LiveVisualizer.start()` calls `_kill_process_on_port`, which uses `netstat` + `taskkill` on Windows and `fuser -k` on POSIX. It does not check which process owns the port.

**Evidence:** Any process listening on 8765 is terminated when `--live` starts. This can close an unrelated application and lose unsaved work.

**Fix:** Delete `_kill_process_on_port`. Try to bind the port; if it is busy, print a clear message and exit. Add a `--port` option if users need to pick another port.

**Workaround:** Make sure port 8765 is free before using `--live`.

---

### KI-006 — A fresh install cannot run the default command

**Files:** `requirements.txt:1-7`, `README.md:21-31`, `main.py:7`, `multiverse_optimizer.py:242-243`

**Problem:** `requirements.txt` omits `jax`, `jaxopt`, `immrax`, and `pandas`. `main.py` defaults to `MasterTrajectoryOptimizer`, which raises `ImportError` when JAX is not installed. The README install command also omits JAX and Immrax.

**Evidence:** A fresh `pip install -r requirements.txt` followed by the README quick start cannot run the default command or the validation path. `sysid_solver.py` imports pandas, which is also missing.

**Fix:** Add the missing packages to `requirements.txt`, or make `--simple` the default and document JAX/Immrax as optional extras. Update the README install command to match.

**Workaround:** Install `jax`, `jaxopt`, `immrax`, and `pandas` manually, or use `--simple` for trajectory generation.

## P1 — fix before relying on the result

### KI-007 — The JAX test suite is silently skipped

**Status:** Fixed in the working tree (uncommitted). The stale test modules were deleted and the suite now runs every JAX module.

**Files:** `run_tests.py`, `tests/test_regressions.py`

**Problem:** All JAX test imports are wrapped in one `try`. Two test modules import classes that no longer exist: `MultiVerseRefiner` and `LocalSegmentOptimizer`.

**Evidence:** `python run_tests.py` previously reported 39 passing tests and printed "jax not installed", even though JAX 0.6.2 was installed. The JAX tests never ran under the normal test command. After deleting the stale modules, the suite runs 151 tests with 0 skipped.

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

**Files:** `export.py:19`, `export.py:58`

**Problem:** `resample_to_fixed_dt` linearly interpolates heading as a raw number, with no unwrap.

**Evidence:** Source headings `3.13` and `-3.13` represent a 0.02 rad turn across ±π. Interpolation goes through `0.0`, a π turn in the wrong direction.

**Fix:** Run `np.unwrap` on the heading sequence before interpolation, then wrap the result back to `[-π, π]`.

**Workaround:** Inspect exported headings manually when the path crosses ±π.

---

### KI-012 — An event on the final waypoint never fires

**Files:** `pybricks_code/ramsete.py:248-263`, `pybricks_code/ramsete_compact_events.py:112-136`

**Problem:** The loop breaks when `t >= total_time`, and the sample advance stops at `num_samples - 2`. Events are checked only on `s1`, so the last sample is never `s1`. The compact event controller keeps the same break condition.

**Evidence:** A final “release” or “score” event is skipped. This is a normal mission pattern.

**Fix:** Allow the index to reach the last sample, or explicitly fire the final sample’s event before the loop exits.

**Workaround:** Trigger the final action outside `follow_trajectory` after it returns.

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

**Fix (implemented):** The deployed `ramsete_fast.py` formula is the source of truth: exact sinc, shortest-angle error, and the precomputed/interpolated `sqrt(omega² + b·v²)` gain. `JAXRamseteController.calculate` accepts the precomputed gain and normalizes the error on the CPU path. The interval path uses unrolled headings and an interval-friendly polynomial because Immrax's natural inclusion does not implement `abs`, `where`, `mod`, `sinc`, or `atan2`. A randomized parity test compares the JAX command to a direct transcription of `ramsete_fast.py`, including ±π crossings.

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

**Files:** `main.py:148`, `live_visualizer.py:100-103`, `live_visualizer.py:90`, `live_visualizer.py:120-135`

**Problem:** `send_config` runs immediately after the server starts, but `broadcast` returns when there are no clients. The README tells users to open the browser afterward.

**Evidence:** The first client never receives robot config or waypoints. `send_candidates` and `send_status` are never called, and `on_regenerate` is never assigned, so the regenerate controls are dead.

**Fix:** Store the last config and waypoints and replay them when a client connects. Wire `on_regenerate` or remove the UI. Alternatively, wait for a client before sending.

**Workaround:** None reliable; the final trajectory arrives, but the field and robot setup do not.

## P2 — cleanup

### KI-017 — Convergence flags do nothing

**Files:** `main.py:67-69`, `main.py:294-296`, `multiverse_optimizer.py:250`

**Problem:** `--convergence-output` and `--convergence-animate` are parsed but never used. `MasterTrajectoryOptimizer` declares `iteration_history` but never fills it.

**Evidence:** With the default optimizer, `--show-convergence` does nothing. The two output flags do nothing in any mode.

**Fix:** Forward the inner `TrajectoryOptimizer.iteration_history` and pass the output flags to the plot functions, or delete the flags and history code.

**Workaround:** Use `--simple --show-convergence`, which does populate the inner optimizer’s history.

---

### KI-018 — Dead code, duplicates, and generated files

**Files:** `pybricks_code/ramsete_fast.py` (intentionally kept), `examples/complex_mission.py`, `pybricks_code/complex_mission.py`, `pybricks_code/complete_mission.py`, `immrax_validator.py:295`

**Problem:** Cleanup removed the duplicate `examples/complete_mission.py`, the root `test_jax_ramsete.py` (its boundary test moved to `tests/test_jax_ramsete.py`), `docs/API_optimizer.md`, generated example/controller/benchmark outputs, and unused imports plus `send_candidates`/`send_status`. `ramsete_fast.py` is intentionally kept: it is the production controller used on the real robot. Remaining: generated mission copies in `pybricks_code/` and `examples/complex_mission.py`; `get_safety_ribbon_data` is still unused and is no longer documented in `docs/WHITEPAPER.md`; the live-view regenerate controls are still dead.

**Evidence:** The full suite now runs 151 tests with 0 skipped. `random_suite/run_000/plot.png` is kept because `README.md` embeds it as a historical example.

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

**Status:** Open, partially addressed in the working tree. The controller parity, linear reference, 20 ms sample-and-hold, and physical backlash model are implemented. The controller split is also done: `robot_config.py` is hardware-only, `mission_4_ramsete.py` imports `ramsete_fast` with its calibrated defaults, and `mission_5_lambdas.py` imports `ramsete` for events. `pybricks_code/sysid_digital_twin.py` and `sysid_fit.py` provide the robot log/capture and host fit path; no hardware logs have been collected yet. The host tools accept a minimal drivebase profile (`wheel_radius_m`, optional `track_width_m`) instead of a full robot config. `digital_twin.py` and `main.py --sysid-fit` now load and validate a fit. When enabled, the validator uses a five-state plant with actual wheel speeds, the fitted straight-line gain interval, first-order time constant, command delay, and odometry distance/heading scale in feedback. Braking slip is loaded and reported but not applied. Reachability is a post-solve candidate score (`--candidates`), not yet an optimizer term. Motor/current transients, DriveBase PID, full encoder odometry, slip dynamics, and DriveBase command limits remain unimplemented.

**Files:** `optimizer.py`, `immrax_validator.py`, `jax_ramsete.py`, `digital_twin.py`, `main.py`, `multiverse_optimizer.py`, `pybricks_code/ramsete_fast.py`, `pybricks_code/robot_config.py`, `pybricks_code/mission_4_ramsete.py`, `pybricks_code/mission_5_lambdas.py`, `pybricks_code/sysid_digital_twin.py`, `sysid_fit.py`, `sysid_compare.py`, `examples/drivebase_profile.example.json`, `tests/test_sysid_fit.py`, `tests/test_sysid_compare.py`, `tests/test_digital_twin.py`

**Problem:** The default validator mirrors the deployed controller's math and timing, but its plant is still kinematic: commanded wheel speeds become motion immediately (apart from wheel backlash). The optional fitted plant adds actuator lag, delay, gain uncertainty, and a static odometry correction, but it still has no motor current/torque transient, DriveBase PID, integrated encoder/gyro odometry, or wheel-slip dynamics. Feedback uses a scaled version of the true pose, not the robot's actual `DriveBase.state()` estimate. Reachability is a post-solve candidate score (`--candidates`, `main.py`), not yet a term or constraint in the optimizer.

**Evidence:** The straight example passes with `max_tracking_error_m≈0.00346 m` and `max_envelope_radius_m≈0.0028 m` under the narrowed test ranges. Representative paths still expose the gap: the S-curve example reports `max_tracking_error_m≈0.0132 m` and `max_envelope_radius_m≈0.0077 m`; the stops example reports 0.0052-0.0060 m / 0.0093-0.0095 m; the events example reports 0.0053-0.0054 m / 0.0097 m; the complex mission now passes at 6.068 s with 0.0049 m / 0.0122 m. These are model predictions, not hardware measurements. There is no real tracking log in the repository yet; the logger and fitter have only been exercised with synthetic data. A synthetic fit with `gain=1.0, tau=0.08 s` already rejects the straight example with 3.44 cm tracking error, which shows why hardware calibration matters before enabling the fitted model. The old `accuracy_weight += 2.0` retry is removed: on the complex mission the slow 17.1 s candidate diverged, while the 6.07 s one stayed bounded. Candidate generation in the simple path now uses five initial-guess seeds (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`); on the complex mission all four non-baseline seeds converge to slower, wider paths (8.1-8.7 s, 1.55-1.76 cm, tracking 0.53-0.72 cm), so the 6.07 s / 1.22 cm baseline still wins. Guess diversity alone does not beat the current uniform-dt formulation; per-segment timing is the next structural lever.

**Fix:** Run `pybricks_code/sysid_digital_twin.py` on the robot, record the stop/final poses in `poses.csv`, fit the logs with `sysid_fit.py --profile examples/drivebase_profile.example.json`, and pass the result with `main.py --sysid-fit sysid_fit.json --validate`. Compare the planned trajectory with the measured poses using `sysid_compare.py`; it prints position/heading errors and whether each measured pose is inside the reachability envelope, and it accepts a plain predicted-pose CSV and a drivebase profile. Extend the plant to integrated odometry and a braking-slip state only after that comparison. The post-solve choice is made: `--candidates` scores several solutions (distinct initial-guess seeds with `--simple`, accuracy-weight variants on the Multi-Verse path) and picks the passing one with the lowest envelope (then the shortest). Extend it to a hard constraint or differentiable proxy only if the candidate set cannot find a passing path. Do not tune the 5 mm/2 cm thresholds or b/zeta to one path.

**Workaround:** Use the straight example as a parity smoke test; treat predictions for curved/high-speed paths as uncalibrated until hardware logs are available. For planner testing, `examples/fll_choreo.json` temporarily narrows `track_width_range` to ≈±0.1% around 0.0965 m and `gear_backlash_m` to [0.4, 0.5] mm; these are test assumptions, not measured hardware values, and should be widened again once calibration data exists.

## Suggested fix order

1. KI-001, KI-002, KI-003, KI-007, KI-008, KI-009, KI-010, KI-013, KI-014, and KI-019: done in the working tree; never save or ship a failed or unvalidated run, keep validation on the configured robot, use both uncertainty bounds, keep the reference tube aligned with the samples, and keep the optimizer and Immrax on the same per-wheel force model.
2. KI-020: controller split, log/fit tooling, and the optional `--sysid-fit` actuator/odometry path are done; collect hardware data, compare fitted predictions with measured poses, add integrated odometry/slip, then use reachability in optimization.
3. KI-006: make a fresh install work.
4. KI-004: either make Multi-Verse real or remove it.
5. KI-005: stop the live server from killing other processes.
6. KI-011 and KI-012: fix the remaining exporter and runtime edge cases.
7. KI-015: restore trustworthy benchmarks.
8. KI-016 through KI-018: repair or remove the remaining UI and dead code.

## Not checked

This list does not cover `sysid_solver.py` beyond its imports, the plotting and animation modules in depth, the Pybricks tutorials, or the generated trajectory and benchmark data. The Pybricks code was not run on hardware.
