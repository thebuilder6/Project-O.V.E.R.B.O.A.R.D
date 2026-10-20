# Known Issues

Last reviewed: 2026-10-19

Scope: whole repository, reviewed as a single-user, offline desktop tool for one FLL robot. This is not a multi-user or server-scale review.

Status: KI-001 through KI-003, KI-005 through KI-019, KI-021 through KI-031, KI-033 through KI-044, KI-045, KI-046, KI-047 through KI-051 are **fixed**. KI-032 is **mitigated** (test cache). KI-004 is **repaired as seed provider**. Open: KI-020 (hardware calibration & digital twin validation), KI-052 (the CLI and the library plan different trajectories for the same mission).

How to use this list: refer to an item by its ID, for example “fix KI-003”.

## Quick triage

| ID | Priority | Area | Issue |
|---|---|---|---|
| KI-001 | P0 | Solver / output | ~~A failed solve is exported as a successful trajectory~~ (fixed) |
| KI-002 | P0 | Validation | ~~`--validate` cannot block output or change the exit code~~ (fixed) |
| KI-003 | P0 | Safety | ~~Immrax ignores upper uncertainty bounds~~ (fixed) |
| KI-004 | P0 | Multi-Verse | ~~Refinement has no effect on the final trajectory~~ (fixed; the pipeline is now a seed provider) |
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
| KI-044 | P1 | Studio | ~~The studio kept the last run's verdict, and a reload could not cancel a solve~~ (fixed) |
| KI-041 | P1 | SysId | ~~One corrupt cell in a session poisoned the whole calibration~~ (fixed) |
| KI-042 | P1 | SysId | ~~A two-repeat calibration could not be loaded at all~~ (fixed) |
| KI-043 | P1 | SysId | ~~The collector calibrated the odometry from the robot's own odometry~~ (fixed) |
| KI-045 | P2 | Cleanup | ~~`sysid_solver.py` is a superseded prototype~~ (fixed; retired prototype and dropped pandas) |
| KI-047 | P1 | Solver | ~~The fast solve path made the solver slower, and its test failed on the tree~~ (fixed) |
| KI-046 | P2 | Cleanup | ~~Five smaller things the audit left open~~ (fixed; all five resolved) |
| KI-048 | P1 | Solver / output | ~~The solver gives up on a path instead of trying another candidate~~ (fixed; a rescue round tries the other seeds when the run produces nothing) |
| KI-049 | P1 | Validation | ~~The reachability envelope reported a 54 mm width that was the divergence guard's fingerprint, and the tube behind it was blown up by a dependency the interval arithmetic could not see~~ (fixed) |
| KI-050 | P1 | Validation | ~~The closed-loop tick map could not run at all on the installed JAX: the controller unpacked the pose, and Immrax's natural inclusion has no rule for `unstack`~~ (fixed) |
| KI-051 | P2 | CLI / solver | ~~`--self-consistent` is a no-op: the switch cannot change the plan and the trapezoidal collocation cannot be reached from the CLI~~ (fixed; switch is two-way `--self-consistent/--no-self-consistent` and default True) |
| KI-052 | P2 | CLI / solver | The CLI and the library plan different trajectories for the same mission: the CLI is self-consistent by default, the library is not (open) |

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

**Files:** `immrax_validator.py:1143-1200`, `tests/test_regressions.py`

**Problem:** `compute_reachability` creates `w_flat` as six lower bounds followed by six upper bounds, then passes it to `emb_sys.f` as a plain JAX array. Immrax’s natural inclusion treats non-`Interval` arguments as constants. Only the lower bounds affect the reachable set.

**Evidence:** Changing only `w_upper` produces identical output. Changing `w_lower` changes the output. Torque, backlash, track width, wheel radius, and inertia are therefore not checked in the upper direction.

**Fix:** Pass `w = Interval(w_lower, w_upper)`. If wheel-command error should be uncertain, pass `u` as an `Interval` too. Add a regression test that widening only `w_upper` widens the envelope.

**Workaround:** None needed.

---

### KI-004 — Multi-Verse refinement has no effect

**Status:** Fixed in the working tree (uncommitted). The pipeline is repaired and re-envisioned: it is no longer a second solver whose result is discarded, but a **seed provider** for the solver that has the right objective. The repair is four changes, and each of them was necessary.

**Files:** `multiverse_optimizer.py` (`_refine_segments`, `_refine_segment`, `_propagate_timing`, `_refresh_forces`, `_probe_refined_guess`), `jax_optimizer.py` (`local_trajectory_cost`, `generate_candidates_jax`), `path_planning.py` (`ReedsSheppPath.plan`, `plan_with_direction`), `main.py` (`--refine`, `_candidate_specs`, `_build_refined_guess`), `examples/fll_choreo.json` (`multiverse.bootstrap`)

**Problem:** Phase 4 computed a refined `global_traj` that `_global_solve` threw away. The probe mechanism was added (it takes an `initial_guess` array) but the refined guess always lost: on `examples/example_complex_mission.json` it converged to **8.1473 s** against the baseline guess's **6.0677 s**.

**Evidence, and what each defect cost.** The ledger asked for the reason ("find out *why* the refined window is worse"), and the measurement found four separate causes:

1. **The refinement clobbered the waypoints.** The critic reports two-segment windows, and the refinement optimised the whole window, so the waypoint in its middle was moved. A refined trajectory that does not pass through the waypoints is not a starting point for a problem that pins them.
2. **The cost measured a different motion than the solver produces.** `local_trajectory_cost` integrated from the left endpoint; IPOPT uses the trapezoidal rule over the interval. A proxy that grades a different trajectory cannot rank two candidates the way the solver would.
3. **The initial `dt` and the velocities were hard-coded** at 0.1 s and 0.1 m/s, whatever the segment's length or the robot's speed. A segment of 0.2 m and a segment of 2 m were given the same interval.
4. **The refined timing was thrown away, and the guess was read with the wrong model.** The refinement gave every segment its own interval, but the timestamps were not updated, so `guess_from_samples` rebuilt the intervals from the *old* ones; and the probe solved with the equal-duration model, which keeps only the **first** interval and re-reads every sample against it. The guess's states, computed with the refined intervals, were therefore read against a timing they did not belong to.

**Fix (implemented):**

- **Segment-preserving refinement.** The unit of refinement is a single segment, whose two endpoints are waypoints and are preserved exactly. The critic still reports two-segment windows; each window is split into the segments it covers. Nothing between two waypoints can be clobbered any more, because there is nothing between them.
- **The collocation matches.** `local_trajectory_cost` now uses the trapezoidal rule, the same one IPOPT enforces.
- **The physics sets the scale.** For a segment of arc length `L`, the nominal speed is 80 % of the robot's top speed and the interval is `L / (0.8 * v_max)` spread over the window. The candidates start from the segment's own size.
- **The refined timing is propagated.** The segment's timestamps are rebuilt with the refined interval and everything after it shifts by the same amount, so the later segments keep their own. `guess_from_samples` then builds the correct per-segment array.
- **The probe reads the guess with the model that reads its intervals**: the per-segment timing model.
- **The pipeline is reproducible.** The STOMP perturbations were seeded from the clock, so the same mission came out at 4.62 s on one run and 5.50 s on the next. The seed is fixed.
- **The Reeds-Shepp bootstrap works.** `path_planning.py` read rsplan's `Path` object as a list; the `TypeError` was swallowed and every call returned `None`, so the "Reeds-Shepp bootstrap" fell back to the straight line every time. The shipped config compounded it with `turning_radius: 0.0`, which rsplan cannot plan with at all (0 of 12 example segments). With the API fixed and the radius at the robot's 0.05 m, 12 of 12 segments plan.
- **The pipeline is re-envisioned as a seed provider.** `main.py --refine` adds the pipeline's refined trajectory to the candidate set, under the per-segment timing model, over the accuracy-weight grid. The pipeline's phases 1 to 4 cost about 2 s; the solver does the magnitudes.

**Measured.** On `examples/example_complex_mission.json` the probe now **wins** against the primary solve: 4.617 s against 6.049 s. With `--refine` the mission goes from **6.068 s to 4.936 s** against the *default* candidate set (two candidates), the trajectory passes the gates (envelope 1.14 cm, tracking 0.45 cm), and the export is written.

**What the seed provider is and is not worth.** Over 30 missions on one tree, the refined family (150 candidates) and the stock seeds (300 candidates) were compared side by side:

| | stock seeds | refined family |
|---|---:|---:|
| candidates | 300 | 150 |
| passing | 69 (23 %) | 29 (19 %) |
| missions with a passing candidate | **11 of 30** | 9 of 30 |
| missions only this family passes | - | **0** |
| missions only the stock seeds pass | **2** | - |

On the nine missions where both have a passing candidate, the refined family is faster on **two** and slower on six. So the refined seed is **not** a better guess than the stock seeds and it **rescues nothing**: it is a *cheaper* way to reach the stock set's speed on some missions (one candidate instead of ten: 4.936 s against the default set's 6.068 s, where the full ten-candidate stock set reaches 4.877 s), not a faster one.

**The next lever is the objective, not the search.** The pipeline's cost is duration plus a jerk term, so it buys speed with tracking accuracy and the 5 mm gate rejects the trade: the refined candidate passes the complex mission only at a small accuracy weight (0.001), and the S-curve and the stops mission fail for it as they fail for everything else.

**Workaround:** None needed. `--refine` is off by default, because it needs JAX, because it costs a refinement and a solve per candidate, and because the measurement says the stock seeds are the better buy. `--multiverse` still runs the whole pipeline as its own solver.

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

**Files:** `robot_model.py:7-35` (shared function), `robot_model.py:289` (`get_wheel_normal_forces`), `robot_model.py:312` (`get_max_wheel_force_symbolic`), `robot_model.py:334` (`get_traction_limit_symbolic`), `optimizer.py:112-134`, `jax_robot_model.py:126`, `immrax_validator.py:930-940`

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

**Status:** Fixed in the working tree (uncommitted). All three benchmark modes run, the generator solves every mission, and there is a test that keeps it that way.

**Files:** `benchmark.py`, `examples/generate_random_suite.py`, `tests/test_random_suite.py`

**Problem:** `RandomBenchmark._validate` wrote a trajectory with the config nested under `trajectory`, then called `validate_trajectory(output_file, output_file)`. The second argument was read as a config file, so `RobotConfig` fell back to defaults. The generator read a missing `robot_config.json` and treated `solve()`'s `(samples, stats)` tuple as the samples. `benchmark.py` also defaulted to a `robot_config.json` that is not in the repository, so CLI mode failed when it was missing.

**Evidence:** Random benchmark safety results used default robot parameters. The random suite generator crashed or wrote a malformed trajectory. CLI mode failed without an explicit `--config`.

**Fix (implemented):**

- `benchmark.py` writes the config it used to `benchmarks/config_used.json` and validates against that file, so the safety numbers describe the robot the benchmark solved for.
- `benchmark.py` defaults to `examples/fll_choreo.json`, which exists. The pipeline mode reads `stats["converged"]` and prints `FAILED` for a solve that did not converge, instead of reporting it as a normal result.
- The random benchmark draws its waypoints as a random walk inside the profile's field rather than as independent uniform draws, which used to put two waypoints on top of each other often enough to make a mission unsolvable for reasons that had nothing to do with the planner.
- `examples/generate_random_suite.py` unpacks `solve()`'s tuple and uses `examples/fll_choreo.json`.

**A third defect, found while checking:** the generator's parameter `samples` (the samples per segment) was shadowed by the solve result, which was also called `samples`. The **second** mission therefore solved with the first mission's sample *list* as the sample count, and the optimizer died on `num_samples_per_segment + 1` with *"can only concatenate list (not "int") to list"*. A one-mission run never showed it. The solve result is now called `trajectory`.

**Workaround:** None needed. `tests/test_random_suite.py` runs two missions and fails if any of them raises, so the shadowing cannot come back. Note that the mission suite itself is generated by `research/suite.py` and measured by `research/harness.py`; this script only solves the missions and writes them out for a human to look at.

---

### KI-016 — The first live-view client never receives config or waypoints

**Status:** Fixed in the working tree (uncommitted).

**Files:** `live_visualizer.py` (deleted), `main.py`, `viz/index.html` (deleted)

**Problem:** `send_config` ran immediately after the server started, but `broadcast` returned when there were no clients. The README told users to open the browser afterward.

**Fix:** The last config, waypoints and trajectory were stored and replayed to every client that connects, so a reload or a late browser still showed the field. The live view was then retired along with KI-005: the studio loads the mission itself, so there is no first-client race to have.

### KI-030 — The studio told the operator an unchecked trajectory was "approved"

**Status:** Fixed in the working tree (uncommitted).

**Files:** `viz/studio.js`, `viz/studio.css`, `viz/studio.html`, `gui_server.py`, `tests/test_studio_honesty.py`

**Problem:** Four defects, all found by walking the studio as somebody who has never seen it.

1. The Run tab called the trajectory "approved" and enabled **Push to hub** after a solve that was never validated. `updatePushButton()` checked only that a trajectory file existed, so the default preset (no validation) put an unchecked path one click away from the robot. The word "approved" was hard-coded, so it was not a verdict but a label.
2. A failed solve left the *previous* solve's trajectory in `state.trajectory`. The canvas drew it in the bright "Planned trajectory" colour, the Play button played it and the Plots tab charted it, while the Result tab said "Failed". The field showed a path that was not the answer to the mission on screen.
3. A solve that never started (fewer than two waypoints, no robot config) only showed a toast. The header and the Result tab kept the previous solve's verdict, so the operator read "Failed" next to a message about waypoints.
4. The `hidden` attribute did nothing on any `.row`: `.row { display: flex }` is an author rule and beats the browser's own `[hidden]` rule. Three controls that should have been invisible were on screen: the "Use the robot's geometry" button with no warning above it, an empty model dropdown with a "Use" button, and the fix-button row.

**Evidence:** Solving `examples/example_with_events.json` (4.64 s) and then `examples/example_s_curve.json`, whose validation fails, left the events path on the canvas and the playback reading `3.88s / 4.64s`. With the default preset the Run tab read "The approved trajectory is ready to run on the robot." for a trajectory the Result tab had labelled "Not validated".

**Fix:**

- The push is gated on the *validator's* verdict, not on the switch on the Solve tab: `state.solveValidated` comes from the selected candidate of the validation report, so a solve with a measured model counts as validated even when the box was not ticked. An unchecked trajectory leaves the button off and the Run tab says why.
- `clearTrajectory()` empties the path, the playback and the plots, and it runs when a solve starts and when it fails. The last good path stays on the field as an "earlier run", which is what it is.
- `refuseSolve()` sets the header and the Result tab, so a solve that never starts cannot leave an old verdict behind.
- `[hidden] { display: none !important; }` in `viz/studio.css`, so the attribute wins over a class that sets `display`.
- The error the operator reads no longer carries the path of the temporary folder the CLI would have written to; the solver log keeps the full text.

**Workaround:** None needed. `tests/test_studio_honesty.py` pins all four.

### KI-031 — A new user could not install the project

**Status:** Fixed in the working tree (uncommitted).

**Files:** `install.bat`, `install.sh`, `start_studio.bat`, `start_studio.sh`, `requirements.txt`, `diagnostics.py`, `main.py`, `.gitattributes`, `README.md`

**Problem:** Three things stood between a first-time user and a working studio.

1. The two requirement files could not be installed together. `requirements.txt` pinned `numpy<2.0.0` and the optional stack needs `numpy>=2.2` (JAX 0.12), so the two-step install ended with numpy 2.5.4 - outside the range the project's own file declares. A later `pip install -r requirements.txt` would then pull numpy back to 1.26 and break the optional stack.
2. The check that was supposed to help - `python main.py --doctor` - could not run on a machine with nothing installed, because `main.py` imported click and numpy at the top. The first thing a new user saw was `ModuleNotFoundError: No module named 'click'`, not the install command.
3. There was no one-click path at all. The install was a `pip install` line in the README, and the studio was `python main.py --gui`.

**Evidence:** A fresh `install.bat` in an empty `.venv` produced numpy 2.5.4 with `requirements.txt` asking for `<2.0.0`. `python main.py --doctor` with the core packages hidden died with an ImportError traceback instead of a report. And `cmd.exe` could not find the `try_python` label in `install.bat` at all, because the file had LF endings: cmd.exe cannot find a batch label in a file without CRLF.

**Fix:**

- `requirements.txt` asks for `numpy>=1.24.0,<3.0.0`. The project already runs on numpy 2 (the solver, the validator and the suite pass on 2.3.5 and on the 2.5.4 a fresh install brings), and the pin under 2.0 only existed to keep the file tidy.
- `main.py` checks the core packages before it imports click, and prints the environment report with the install command when one is missing. `--doctor` now works on a clean machine, because `diagnostics.py` is stdlib only.
- `install.bat` and `install.sh` find a Python 3.9 or newer, create `.venv`, install the core packages and then the optional ones, and run the check. A failure in the optional packages is a warning, not a failure: the solver and the studio work without them.
- `start_studio.bat` and `start_studio.sh` use that `.venv` and install it first when it is not there, so one double-click is enough on a fresh download.
- `.gitattributes` keeps the `.bat` files out of the line-ending conversion, so their CRLF survives into a ZIP download, and keeps the `.sh` files LF.

**Workaround:** None needed. `tests/test_cli_ux.py::TestFirstRun` pins the report on a clean machine, the install command in it, and the CRLF in the batch files.

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

**Evidence:** On the tree as it stands, the suite is 508 tests and takes 121 s. A second run over the same tree takes 20 s: 491 of the tests are skipped, and the 17 that run are the ones that write into the repository (the studio tests, which write into `missions/`, the doctor, which writes its report, and the cache's own tests, which write a probe file). The saving is reported in the summary, so the number is never a guess, and a third run takes the same 20 s: the cache does not invalidate itself.

**Workaround:** `python run_tests.py --no-cache` for the full suite. The cache is a development convenience; it is not a gate, and no release decision rests on a cached run.

### KI-033 — The host could not find the hub over Bluetooth

**Status:** Fixed in the working tree (uncommitted). The hub is found the way the Pybricks IDE finds it: by probing what is nearby.

**Files:** `hub_ble.py` (new), `hub_deploy.py`, `gui_server.py`, `tests/test_hub_ble.py`

**Problem:** The host found the hub with `pybricksdev.ble.find_device()`, which accepts an advertisement only when it carries the Pybricks service UUID **and** a local name. Windows does not report them in that shape. bleak's WinRT backend leaves `service_uuids` empty unless the peripheral puts the UUID in the advertisement payload, and the local name tends to arrive in a scan response the scanner never joins. The filter therefore never matched, and every push over Bluetooth ended in "No Pybricks hub found over BLE" while the Pybricks IDE connected to the same hub without trouble - the IDE enumerates every nearby device and probes the one you pick.

**Evidence:** On the development machine, bleak reported `local_name = None` and `service_uuids = []` for 8 of the 11 devices it saw, and reported a name for the rest only when the name happened to be in the advertisement payload. The one device whose advertisement carried a service UUID was reported correctly, so the UUID is not always lost - but the *name* usually is, and `find_device` insists on both.

**Fix (implemented):** `hub_ble.py` scans everything (no service filter), ranks the candidates - the hub the operator names first, then anything that advertises the Pybricks service, then the rest - and asks the ones that do not announce themselves by connecting and looking for the Pybricks service. The probe runs a few devices at a time, so the answer comes in seconds rather than in a minute. `hub_deploy.py` uses it for the BLE transport, so the CLI, `main.py --push-hub`, `sysid_collect.py` and the studio's push all get it. `python hub_deploy.py --list-hubs` is the IDE's device picker: it shows what the adapter can see and says which of the devices is a hub.

**Workaround:** None needed. If the hub is still not found, the error says what was tried and lists the devices that were visible, and names the three usual causes: the hub is off, the Pybricks IDE holds the hub's single connection, or the hub was never paired.

---

### KI-034 — The local envelope limit landed on the wrong segment

**Status:** Fixed in the working tree (uncommitted).

**Files:** `main.py` (`_segment_limit`, `_segment_report`, the gate loop), `tests/test_regressions.py`

**Problem:** A waypoint's `error_threshold` (or `precision_mode`) was applied to the samples of the segment that **started** at that waypoint. So the tight limit was enforced on the way *out* of a precision waypoint, and the way *into* it was judged by the previous waypoint's limit, which is the default 2 cm.

**Evidence:** The rule is `wp_idx = i // samples` and `waypoint_thresholds.get(wp_idx, 0.02)`. With four waypoints and thresholds on waypoint 0 (5 mm) and waypoint 3 (1 mm), the three segments got limits of 5 mm, 20 mm and 20 mm: the final waypoint's 1 mm was never read, because there is no segment after the last waypoint. The studio calls the switch "Precision (0.5 cm envelope)" at a waypoint, so the operator asked for the robot to be tight *there*, on the approach.

**Fix (implemented):** One rule, in `main._segment_limit`, used by both the report and the gate: a waypoint's limit governs the segment that **ends** at it, and the segment's start waypoint's limit is the fallback. The last waypoint's limit now governs the last segment, and a precision waypoint is held to its tight limit on the approach. A waypoint with no limit of its own still gets the 2 cm default.

**Workaround:** None needed. `tests/test_regressions.py::TestSegmentLimits` pins the rule, including that the last waypoint's limit is used rather than dropped.

**What it changes:** a mission with a precision waypoint is now held to its tight limit on the approach, so it can fail where it passed. The measurement runs in `docs/measurements/` were taken with the old rule; each records the tree it describes, so they stay valid as records, but a run that depends on a precision waypoint has to be measured again.

---

### KI-035 — The wheel-speed limit ignored the gearing

**Status:** Fixed in the working tree (uncommitted).

**Files:** `robot_model.py` (`max_linear_speed`, `no_load_wheel_speed_worst_case`), `jax_robot_model.py`, `validator.py` (`audit_constraints`), `immrax_validator.py`, `tests/test_regressions.py`

**Problem:** `gearing` is motor revolutions per wheel revolution, so the wheel's no-load speed is the motor's divided by it. `RobotConfig.max_linear_speed` did not divide: it returned `v_max_rad_s * wheel_radius`. That number is the optimizer's wheel-speed bound and the CPU audit's speed limit, so for a geared robot both were `gearing` times too large.

**Evidence:** With `gearing: 2.0` the function returned 0.477 m/s where the real no-load wheel speed is 0.265 m/s. The force model in the same file reads it the other way (`omega = (v / r) * gearing`, compared with `v_max_rad_s`), and `drivetrain.py:401` already had the correct formula. The Immrax motor curve is gearing-aware too, so at the optimizer's own bound the motor torque is exactly zero: every plan for a geared robot was guaranteed to fail the motor gate, reported as "motor force over limit" rather than "wheel speed over limit".

**Fix (implemented):** `max_linear_speed` divides by the gearing, and one new method, `no_load_wheel_speed_worst_case`, gives the validator's limit (the smallest declared wheel radius, the same number for the CPU audit and for Immrax). `JAXRobotConfig` carries the same method, so the two models cannot drift apart again. With `gearing: 1.0` every number is unchanged.

**Workaround:** None needed. `tests/test_regressions.py::TestGearedSpeedLimit` pins the ratio and checks that a geared robot is not planned above its motor's real top speed.

---

### KI-036 — The studio pushed whatever trajectory it was given

**Status:** Fixed in the working tree (uncommitted).

**Files:** `gui_server.py` (`_validation_verdict`, `start_push`, `_origin_allowed`, `_load_model`), `tests/test_gui_server.py`

**Problem:** The rule "only a validated trajectory goes to the robot" lived only in the browser. The server's `/api/push` checked that a `.traj` file existed, and `main.py` writes that file without `--validate` (the export gate only blocks a non-converged solve). So a direct POST pushed an unchecked plan onto the mat. Two things made that reachable: the cross-site guard compared the origin's **hostname** but not its **port**, so any other web app on this machine sent an acceptable `Origin`; and `/api/robot/model/load` read any path on the machine and copied it into `calibration/`.

**Evidence:** `gui_server.py` checked `if traj is None or not traj.is_file()` and nothing else. The browser's own check (`viz/studio.js`) is not a gate: the endpoint that drives the robot has to enforce it.

**Fix (implemented):**

- `start_push` reads the validator's verdict back from the solve's own output (`##OVERBOARD-VALIDATION##`, the machine-readable line `main.py` already prints) and refuses unless the **selected** candidate passed. An unvalidated run has no verdict and is refused.
- The origin check compares the port with the studio's own, so only the studio's own page may drive it.
- `/api/robot/model/load` only reads a file under the project folder or the calibration folder.

**Workaround:** None needed. `tests/test_gui_server.py::TestStudioPush` pins that an unvalidated solve is not pushable.

---

### KI-037 — The studio kept every job forever, and a cancel left the solver running

**Status:** Fixed in the working tree (uncommitted).

**Files:** `gui_server.py` (`MAX_JOBS`, `_evict_old_jobs`, `_cancel_job`, `_process_group_kwargs`, `_terminate_tree`), `tests/test_gui_server.py`

**Problem:** Three things about the studio's job handling.

1. Nothing ever removed a job or its temporary folder. Each solve left an `overboard_studio_*` folder with the config, the waypoints, the trajectory, the controller and the stats, and a job object in memory with up to 800 log lines. A season of solves filled the temp folder and left every old trajectory downloadable by guessing an id.
2. `Cancel` called `proc.terminate()`, which kills only the `main.py` parent. The candidates are solved in `ProcessPoolExecutor` **worker processes**, and they survived: the studio said "Cancelled" while the orphan workers kept solving at full speed. The status was also set to "cancelled" immediately, so the studio accepted a second solve while the first was still running.
3. The test suite committed into the operator's own mission history: `tests/test_gui_server.py` posted to `/api/missions` with the real `MISSIONS_DIR` and deleted the file afterwards, but not the commit.

**Evidence:** `git -C missions log` carried `6f834ad Mission unit-test-mission` and `4e2c94e Mission unit-test-mission`, and the folder's status showed the deleted `unit-test-mission.json`.

**Fix (implemented):**

- The studio keeps the newest 20 finished jobs and removes the folder of each one it drops. The running job is never evicted.
- The child runs in its own process group (`CREATE_NEW_PROCESS_GROUP` / `start_new_session`), a cancel kills the whole tree (`taskkill /T /F` / `killpg`), and the job stays "running" until the process is really gone, so the studio does not take a second solve on top of the first.
- The test writes into a temporary folder now, like its sibling test already did.

**Workaround:** None needed.

---

### KI-038 — The exported Python file broke on an event name with a quote in it

**Status:** Fixed in the working tree (uncommitted).

**Files:** `export.py` (`write_python_file`), `tests/test_regressions.py`

**Problem:** `--export-format python` wrote the event name into the generated source with an f-string: `"event": "{s["event"]}"`. An event name is free text the operator typed.

**Evidence:** An event named `raise_arm"` produced `"event": "raise_arm""`, a **SyntaxError** on the hub. A name like `x", "y": 1, "z": "` also injected extra keys into the sample dict. The compact export already did it correctly with `%r`.

**Fix (implemented):** The name is written with `repr()`, so it is one Python string whatever it contains.

**Workaround:** None needed. `tests/test_regressions.py::TestEventNameEscaping` pins that the generated file compiles and that the name stays one string.

---

### KI-039 — Two missions in another script became the same file

**Status:** Fixed in the working tree (uncommitted).

**Files:** `gui_server.py` (`slugify`, `restore_mission`), `optimizer.py` (`_guess_point`)

**Problem:** Three small defects, all from the same kind of shortcut.

1. `slugify` stripped everything outside `[a-z0-9_-]`, so "任务" and "テスト" both became `mission`. The second save overwrote the first, and the toast said "Saved missions/mission.json".
2. `restore_mission` wrote the file and only then parsed it, so a blob that is not a mission reached the disk and the mission then disappeared from the list.
3. The baseline initial guess averaged two headings plainly, so a path that turns through ±180° got a guess heading near zero in the middle - the worst possible guess for exactly the missions that need another seed.

**Fix (implemented):** The slug keeps unicode letters and digits; the restore parses before it writes; the guess blends the two headings the short way round, like the rest of the file.

**Workaround:** None needed.

---

### KI-040 — `--push-hub --hub-log` lost the whole hub log when the link dropped

**Status:** Fixed in the working tree (uncommitted).

**Files:** `main.py` (`_push_to_hub`), `hub_deploy.py` (unchanged; it already streamed)

**Problem:** `main.py` called `push_trajectory` without the `log_path`, so nothing was written as the run arrived; the file was written only after the push returned, from the lines in memory. A dropped Bluetooth link raised `HubLinkError` before that, so the log file was never created and the measured `POSE` and `EVENT` lines were lost.

**Evidence:** `hub_deploy.py --out` and `sysid_collect.py` both stream the console to disk as it arrives, so a dropped link costs the last test and not the session. The CLI's `--push-hub --hub-log` was the one that did not.

**Fix (implemented):** The log path goes into `push_trajectory`, which streams it. A dropped link now leaves the output that arrived on disk, and the error says where it is. The file is rewritten in the text form the other two entry points write, so one reader serves all of them.

**Workaround:** None needed.

---

### KI-044 — The studio kept the last run's verdict, and a reload could not cancel a solve

**Status:** Fixed in the working tree (uncommitted).

**Files:** `viz/studio.js`, `tests/test_studio_honesty.py`

**Problem:** Three faults in what the studio shows and allows.

1. A new solve did not clear the previous run's validation verdict. `solve()` reset the summary, the collisions and the trajectory, but not the validation report, and the report was only redrawn when a **new** one was parsed. A solve without validation prints no verdict, so nothing was redrawn: the old candidate chips ("failed"), the old per-segment table and the old red failing segment stayed on the field while the header said "not validated".
2. The Push button did not know that the mission had moved on. It asked whether there was a trajectory and whether it had been validated, but not whether the mission on screen still matched, so a validated trajectory stayed pushable after a waypoint was dragged - and the push ran the **old** plan while the canvas showed the new mission.
3. A reload during a solve had no way back. The studio asked `/api/jobs/current` only in the `--live` path, so after a reload in the ordinary studio there was no job id: the server refused the next solve with 409, the Cancel button was hidden, and Cancel was a no-op anyway.

**Fix (implemented):** `renderValidationReport` is the one owner of the verdict and handles "no report" by clearing the table, the chips, the failing-segment set and the red marks on the field; `solve()` calls it with nothing, so a new run starts with no verdict. The Push button and the push itself both refuse a mission that has changed since the solve, and the reason is said plainly. `init()` always asks `/api/jobs/current` and adopts a running solve, so a reload attaches to it and Cancel works.

**Workaround:** None needed. `tests/test_studio_honesty.py` pins the three rules.

**Also fixed:** the download button asked the *solve* job for the *push* job's artifact, so it 404'd after a push. The downloads box now asks for the job that owns the file.

---

### KI-041 — One corrupt cell in a session poisoned the whole calibration

**Status:** Fixed in the working tree (uncommitted).

**Files:** `sysid_fit.py` (`parse_session`, `_step_fits_measured`, `_fit_first_order`, `_summary`, the CLI), `tests/test_sysid_fit.py`

**Problem:** The fitter read a session row with `float(part)`, and `float("nan")` and `float("inf")` are valid Python. A single bad cell therefore became data: `np.median` carried it into that step's command, the gain became `nan`, and the summary still wrote `"calibrated": true`. The fit file was written with `json.dump`, which emitted a bare `NaN` token - not valid JSON - and the loader then refused it with "actuator.gain must be finite, got nan", so the operator was told the fit was broken and not that the *session* was.

**Evidence:** A session with `nan` in the `v_cmd_mm_s` column produced `calibrated: true, gain: nan`, a file that `JSON.parse` cannot read, and a `ValueError` at load time.

**Fix (implemented):** A row with a non-finite value is dropped and counted, the count is reported in the fit and printed by the CLI, and a step whose command or measured window is not finite is not fitted. The fit file is written with `allow_nan=False`, so a poisoned fit fails loudly **at write time** and no file is written, instead of writing a file that fails to load.

**Workaround:** None needed. `tests/test_sysid_fit.py::TestBadRows` pins it.

---

### KI-042 — A two-repeat calibration could not be loaded at all

**Status:** Fixed in the working tree (uncommitted).

**Files:** `digital_twin.py` (`_actuator_from_dict`), `sysid_fit.py` (`_summary`), `tests/test_digital_twin.py`, `tests/test_sysid_fit.py`

**Problem:** The gain's 95 % interval over the repeats is a t-interval, and with two repeats the multiplier is 12.706. An ordinary 20 % run-to-run spread therefore gives `ci95 = [-0.27, 2.27]`. The loader takes that interval as the validator's gain band and refuses a band whose lower bound is not positive, so the whole fit was rejected - and two repeats is exactly what a dropped Bluetooth link leaves behind.

**Evidence:** `repeat_statistics([0.90, 1.10])` returns `ci95 = [-0.2706, 2.2706]`, and `DigitalTwinConfig.from_dict` raises `ValueError: actuator gain range (-0.2706, 2.2706) must be positive`.

**Fix (implemented):** The loader uses the interval only when its lower bound is positive and the band is not inverted, and falls back to the observed `gain_min`/`gain_max` otherwise. The fitter clamps the interval's lower bound to a small positive number and keeps the raw t-interval under `<name>_ci95_raw`, so the file always loads and the operator can still see what the interval really was.

**Workaround:** None needed.

---

### KI-043 — The collector calibrated the odometry from the robot's own odometry

**Status:** Fixed in the working tree (uncommitted).

**Files:** `sysid_collect.py`, `sysid_fit.py` (`_fit_odometry`), `tests/test_sysid_collect.py`, `tests/test_sysid_fit.py`

**Problem:** Two things about the distance scale.

1. `sysid_collect.py` handed the hub's own poses to the fit as measurements. The hub's distance reading **is** its odometry, so measured/odometry is exactly 1.0 by construction, and the fit wrote `"calibrated": true` for a block that carries no information. The validator uses that scale to advance the robot's own pose estimate, so it decides how well the modelled controller tracks the plan. The collector's own docstring promised the opposite.
2. The fit pooled the tape samples and the wall-homing samples and took `np.median` of the union. With an even count that averages the two middle values, so the number used was neither estimate: tape 1.000 and wall 1.050 gave 1.025, while the file separately reported 1.0 and 1.05.

**Fix (implemented):** The collector no longer passes the hub's poses to the fit; they are still written out for the operator, and the odometry block stays uncalibrated unless a real measurement is supplied. The scale is the tape estimate when there is one and the wall estimate otherwise, and both are reported with their disagreement, so the operator still sees the cross-check.

**Workaround:** None needed. The wall still calibrates the distance scale in a collector run, because the wall is an independent reference and not the robot agreeing with itself; the heading scale and the brake slip need a tape measurement and stay uncalibrated.

---

### KI-046 — Five smaller things the audit left open

**Status:** Fixed. All five items have been addressed.

**Files:** `gui_server.py`, `hub_ble.py`, `sysid_fit.py`, `requirements-optional.txt`, `KNOWN_ISSUES.md`, `README.md`, `docs/CONFIGURATION.md`

**The five (all resolved):**

1. **The studio flips a job's status to "ok" before it fills in the result.** `gui_server.py` now populates trajectory limits, collisions, and iteration history inside a `try...finally` block before flipping `job.status` away from `"running"`, eliminating the poll race condition.
2. **A device whose name merely *contains* the wanted string is never probed.** `hub_ble.py` now ranks substring name matches into the preferred probe tier before falling back to unnamed UUID matches.
3. **The hub session parser drops malformed rows silently.** `sysid_fit.py` now parses `TEST_END,<n>` row counts, tracks `missing_rows`, and explicitly warns on truncated BLE captures.
4. **Two optional packages were imported but not declared.** `bleak` and `pypdfium2` were added to `requirements-optional.txt`.
5. **Some docs still describe a state that has moved on.** Stale "uncommitted" status notes and superseded `sysid_solver.py` / `pandas` references across `KNOWN_ISSUES.md`, `README.md`, `AGENTS.md`, and `docs/CONFIGURATION.md` have been updated.

**Workaround:** None needed.

---

### KI-048 — The solver gives up on a path instead of trying another candidate

**Status:** Fixed in the working tree (uncommitted). The rescue round is in.

**Files:** `optimizer.py` (`solve`), `main.py` (`_rescue_specs`, the candidate
loop), `tests/test_regressions.py`

**Problem:** IPOPT can exhaust its iteration budget (5 000 is the configured
limit). The optimizer catches the `Maximum_Iterations_Exceeded` and returns
best-effort samples, but the candidate is marked as not converged, and a
non-converged candidate is never eligible for export (KI-001). The candidate
is therefore thrown away, and its wall time is spent for nothing.

The default candidate set makes the loss worse than it has to be: its two
candidates are the *same* seed (baseline) on the two timing models, so when
the solver stops at its budget, the other seeds (`rolling`, `bow_left`,
`bow_right`, `turn_first`) are never tried.

**Evidence:** the stress run of 2026-10-20
([`docs/measurements/2026-10-20-stress/`](docs/measurements/2026-10-20-stress/)).
**One candidate in 228** hit the limit: `m004#1` with the per-segment timing
model, 36.5 s, `Assertion "return_success(accept_limit)" failed ...
return_status is 'Maximum_Iterations_Exceeded'`. The same path on the other
robot took 0.26 s and 0.30 s to solve; the equal-duration candidate on the
same robot needed 3 898 iterations to converge (27.6 s). So the path is not
the problem - the robot's constants pushed the solver into a hard region
where the baseline seed grinds.

**What the measurement does NOT show:** in all 114 mission-robot pairs, at
least one candidate converged. **No mission was lost to this.** The cost in
the measured case was the 36.5 s and the loss of the *better* candidate (the
per-segment one was faster). The "both candidates stop at their budget" case
- the one that loses the mission - was not observed.

**Fix (implemented):** the CLI now solves the candidates in rounds. The first
round is the candidate set; if the run produced **nothing usable** - no
candidate passed the gates, or (without validation) none converged - the
seeds the set did not use are tried as a **rescue** before the mission is
given up on (`_rescue_specs` and the round loop in `main()`). On a validated
run the rescue uses the per-segment model, because that is where the
measured rescues came from. A rescue candidate is insurance: it is only
selected when the candidate set itself produced nothing that passed.

The rescue costs nothing when the plan works, and one parallel round of
solves when it does not. On a path where nothing passes at all, the run now
takes about twice as long and still refuses - which is the honest answer.

**Evidence that it works:** on the pinned tree of the stress run, the two
pairs the default set lost and the stock set rescued are now trajectories:
`m001#0` is written at 7.746 s with a 1.69 cm envelope (the *rolling* seed,
per-segment) and `m005#1` at 3.702 s with a 0.46 cm envelope (the
*bow_right* seed, per-segment) - the same numbers the harness measured.

**What it does not fix:** the measured case where one candidate converged
and the other did not is unchanged, because the rescue only fires when the
run produced nothing usable, and there the run produced a trajectory (which
the gates then refused).

**Workaround:** `--candidates 5` puts the other seeds in the set, so a path
that grinds on the baseline seed gets four more starting points. It costs
three more solves and three more validation passes per mission, and it is
not the default.

---

---

### KI-047 — The fast solve path made the solver slower, and its test failed on the tree

**Status:** Fixed in the working tree (uncommitted).

**Files:** `optimizer.py` (`solve`, `_solve_once`), `tests/test_optimizer.py`, `AGENTS.md`, `README.md`, `docs/EXAMPLES.md`

**Problem:** `--fast-polish` was designed when the warm-start IPOPT pass crawled through thousands of iterations on the 10-waypoint complex mission: it gave the warm pass a 200-iteration budget and then restarted **cold from the warm pass's half-finished trajectory**, which converged in 677 iterations where the plain pass took 1557.

That premise no longer holds on this tree, and the restart from a half-finished point turned out to be the slowest thing the path could do.

**Evidence.** The complex mission, measured two ways on two machines.

On the machine the fast path was designed on, the plain warm pass crawls: 1557 iterations and about 35 s. The fast path (200 budgeted + 154 cold) is 354 iterations and about 7 s, for a trajectory 0.9 % slower. That is the design working.

On a second machine the same code takes a different path through the same NLP: the plain warm pass is 669 iterations and 2.8 s, and the fast path as it was written - a cold restart from the warm pass's half-finished trajectory - took **3112 iterations and 12.7 s**, four times slower than the plain pass and four times slower than the cold restart from the original guess. The count is not a property of the code alone: a different CPU and a different BLAS change the last bits, and IPOPT on this poorly scaled NLP is chaotic enough that a 1-ULP difference early on sends it down a different path. `tests/test_optimizer.py::TestSolverIterationBudget` pins 1200 iterations, and on the second machine it failed with 3112.

The measurements, both reproducible on their own machine (three runs, and the same with one BLAS thread):

| path | machine A | machine B |
|---|---|---|
| plain (unbudgeted warm) | 1557 iterations, 6.068 s, 35 s | 669 iterations, 6.0715 s, 2.8 s |
| warm pass, budget 200 | 200 (did not converge) | 200 (did not converge) |
| cold from the warm pass's trajectory | 154 iterations, 6.124 s | 3112 iterations, 6.0737 s, 12.7 s |
| cold from the original guess | - | 508 iterations, 6.0694 s, 2.3 s |
| **the fix** (cold from the original guess) | **354 iterations, 7 s** | **708 iterations, 3.1 s** |

So the cold restart is only cheap from a good point: from the warm pass's half-finished trajectory it was the slowest thing the path could do.

**Fix (implemented):** The cold pass restarts from the **original guess**, not from the warm pass's half-finished trajectory. That is the fast path as it now stands on both machines: 354 iterations and 7 s where the plain pass crawls at 1557, and 708 iterations and 3.1 s where the plain pass is already quick. The budgeted stage now reports the iteration count it used (the budget), so the stage report is honest.

**Workaround:** None needed. The flag stays off by default, and the test's 1200-iteration budget holds again.

---

### KI-049 — The reachability envelope reported a truncation as a width, and the tube behind it was blown up by a dependency

**Status:** Fixed in the working tree (uncommitted).

**Files:** `immrax_validator.py` (`RobotOpenLoop.f`, `DigitalTwinOpenLoop`, `ImmraxValidator.__init__`, `compute_reachability`), `docs/research/rq_a/exp19_envelope_scatter.py`, `tests/test_envelope_soundness.py`, `requirements-optional.txt`

**Problem:** Two separate faults, and they hid each other.

The first was in the *number*. The validator stops the propagation when the tube's half-diagonal passes 50 mm and reports the radius it had at that moment, so the reported "width" is a property of the guard and not of the path: the same 54.561 mm came out for a 0.5 m path, a 1 m path and a 2 m path, and 53.081 mm for two other combinations. The reachability envelope was quoted as 54 mm against a 20 mm threshold, as if it were a measurement. It was a truncation.

The second was in the *model*. The plant computes the turn rate as `(vr - vl) / track_width`, and both wheel speeds are the **same** command times the **same** gain, so the true turn rate does not depend on the gain at all. The natural extension cannot see that, so it reported a turn rate of `+/- (the width of the wheel speed interval) / track` where the truth says zero. That phantom rate integrates into a heading drift and then into a lateral drift, and the tube grows with the square of the time until it crosses the guard. The digital twin made it worse: it carries the two wheel speeds as *states*, so the lost correlation was in the state and not in the expression, and no inclusion function could recover it.

**Evidence.** A 1 m straight at 0.5 m/s with a +/-1 % gain band, the reachable set sampled over the declared uncertainty with the corners included (`exp19_envelope_scatter.py`):

| plant and inclusion | reported | truncated | reachable set | contains it |
|---|---:|---|---:|---|
| plain plant, natural inclusion | 54.03 mm | yes | 4.51 mm | yes |
| plain plant, Jacobian inclusion | 11.83 mm | no | 4.51 mm | yes |
| plain plant, natural inclusion, **after the fix** | 11.77 mm | no | 4.51 mm | yes |
| digital twin, wheel speeds as states | 53.61 mm | yes | 4.55 mm | yes |
| digital twin, **mean and difference** | 11.76 mm | no | 4.55 mm | yes |

The reachable set the robot can actually reach is 4.5 mm wide. Everything above that was the model's own doing.

**Fix (implemented):** Three exact reformulations, and no threshold was moved.

1. The wheel-speed difference is taken from the **commands** and the gain is applied afterwards. `(u_r - u_l) * gain` is the same motion as `u_r * gain - u_l * gain`, but the command difference is one number, so one number times the gain is an exact interval, while the difference of two gain-scaled commands is not. On a straight the command difference is exactly zero, so the turn rate is exactly zero.
2. The digital twin carries the wheel speeds as the **mean** and the **difference** instead of the two wheel speeds. The difference is what the turn rate is read from, and the two are perfectly correlated, which an interval box cannot express. The change of variables is a linear bijection and the dynamics are transformed exactly, so the enclosure is still a bound of the same reachable set.
3. The **closed-loop tick map** (`immrax_validator.py`, `ClosedLoopTickMap`) is behind `propagation="tick"`. It embeds one 20 ms tick of the closed loop as a discrete map, with the controller inside it, and propagates the map once per tick, so the map's Jacobian carries the feedback term `dF/dx = dPhi/dx + (dPhi/du)(dN/dx)`. The robot's own pose estimate and the delayed command ride in the map's state, because the feedback term is taken through them. It needs a Jacobian inclusion: with the natural extension the closed-loop map explodes (627 mm on a 1 m straight), because that is the naive interval Ramsete the "interval" feedback mode already avoids.

The report now also carries `envelope_truncated`, so a cut-off propagation says so, and the inclusion functions are behind a switch (`PLANT_INCLUSION`, `CONTROLLER_INCLUSION`) with the Jacobian and the mixed-Jacobian both implemented and checked against the sampled set. The default stayed on the natural inclusion because the measurement says it is now the better one: on the six example missions the Jacobian is 0.00-0.08 mm **wider** and 5-15 % slower, because the reformulation already removed the dependency it would have exploited.

**Workaround:** None needed. The 20 mm envelope threshold and the 5 mm tracking threshold are unchanged (`docs/DIGITAL_TWIN_PLAN.md`). What is left of the calibrated-model problem is **not** physics: on the S-curve the sampled reachable set is 4.09 mm while the tube reports 48 mm and is cut off, so the tube is more than ten times too wide there (`docs/research/rq_a/exp21_curve_geometry.py`). The reason is structural rather than numerical: the validator propagates the **plant** and evaluates the controller *outside* the propagation, at the tube's midpoint, so the feedback contraction is not in the tube at all - the reachable set it bounds is that of an open-loop plant driven by a pre-recorded command. The lever for that is the closed-loop tick map (`exp20_closed_loop_tick.py`), where the Jacobian of the map carries the feedback term `dF/dx = dPhi/dx + (dPhi/du)(dN/dx)`. On a straight that map with the Jacobian gives 6.31 mm against the shipped 11.77 mm and a 4.51 mm reachable set, and with the natural extension it is 627 mm and diverges - so there the Jacobian is not a refinement but the thing that makes the enclosure work at all. On the curve it is worth more than on the straight, because on the curve the shipped propagation does not merely widen - it diverges. On the shipped S-curve with a ±1 % band the reachable set is 3.27 mm, the shipped validator reports 53.64 mm and is cut off, the tick map with the natural extension diverges as well, the tick map with the Jacobian gives 25.28 mm, and the tick map with the **mixed-Jacobian gives 18.80 mm, which passes the 20 mm gate** and still contains the sampled set (`docs/measurements/2026-10-20-envelope/exp20_closed_loop_tick.txt`). The cost is 7-10x the validation time, and the tracking error is unchanged to the last digit - it is the tube's midpoint against the plan, so it is decided by the discretisation and not by the uncertainty. The tick map is now implemented in the validator (`PROPAGATION`, `exp22_twin_propagation.py`): on the default plant it is a clear win - with the mixed-Jacobian the S-curve, the stops and the complex mission stop diverging and pass the 20 mm gate (1.88, 1.30 and 1.83 cm against 5.36, 5.31 and 5.07 cm), and every tube contains the sampled reachable set. With the **digital twin it is worse** (5.8-12.1 cm against 5.1 cm), because the map has to carry the robot's own pose estimate, and an estimate carried in an interval state grows with the wheel-speed uncertainty: the belief's spread is added to the plant's, because the two are perfectly correlated and an interval box cannot say so. **The default therefore stays "plant"**, and making the belief's correlation explicit is the open item - the same class of defect as the phantom turn rate, one level up.

**The tick map is in the tree, behind `propagation="tick"`, and it is not the default.** The default stayed on the plant propagation for the reason above: the reformulations already removed the dependency the Jacobian would have exploited, so the tick map's own gain is on the paths where the plant propagation *diverges* rather than merely widens. The cost is 2-3x the validation time with the Jacobian and 7-10x with the mixed-Jacobian, which is why it is opt-in. Two things are still open on it: the tick box is still two to five times the sampled reachable set on a curve (the box hull of the first-order image re-inflates what the map contracts, so the set's *shape* is the next lever, not a tighter inclusion), and the tick propagation was not wired to the CLI, so the studio and `main.py --validate` still ran the plant propagation. The wiring is done now: `main.py --propagation [plant|tick]` and `research/harness.py --propagation`, and every candidate in the validation report names the model that produced its envelope (KI-050). The tick map also could not run at all on the installed JAX until the controller stopped unpacking the pose; see KI-050.

**With the digital twin the tick map is worse, and it is not the belief.** The twin's version of the same measurement gives 5.8-12.1 cm against the plant propagation's 5.1 cm, where the *default* plant's tick map wins (the S-curve 5.36 -> 1.88 cm, the stops 5.31 -> 1.30, the complex 5.07 -> 1.83). The obvious suspect is the belief - the map has to carry the robot's pose estimate, and an estimate in an interval state grows with the wheel-speed uncertainty - and the control rules it out (`docs/research/rq_a/exp23_twin_tick_belief.py`): the same map with the controller fed the *truth* is not smaller (12.48 cm against 12.08 cm on the S-curve) and is unsound while it is at it (1 mm of overshoot). The reason is the twin's **plant state**: the two wheel speeds are interval coordinates whose spread feeds the command, and the interval arithmetic counts that growth twice. So the calibrated-model case has its own open item, and it is the plant state and not the propagation.

---

### KI-050 — The closed-loop tick map could not run at all on the installed JAX

**Status:** Fixed in the working tree (uncommitted).

**Files:** `jax_ramsete.py` (`JAXRamseteController.calculate`), `tests/test_jax_ramsete.py`, `main.py` (`--propagation`), `research/harness.py` (`--propagation`), `viz/studio.js` (the validation note and the chip tooltip)

**Problem:** The Ramsete controller read the robot's pose by unpacking the array:

```python
x, y, theta = current_pose
xr, yr, thetar = ref_pose
```

Unpacking a traced array is the JAX `unstack` primitive, and **Immrax's natural inclusion has no rule for it** — its registry holds `slice`, `squeeze`, `gather`, `concatenate` and the arithmetic, but not `unstack`. The closed-loop tick map embeds the controller *inside* the map, and a Jacobian inclusion evaluates the map's derivative with the natural inclusion, so the whole tick propagation died with `NotImplementedError: unstack not in inclusion_registry`. Every candidate failed, the export gate closed, and the run reported "Immrax validation error".

**Evidence.** On the installed stack (JAX 0.12.0, Immrax 0.3.8) the three candidate maps all fail the same way: the shipped `ClosedLoopTickMap`, the copy inside `docs/research/rq_a/exp20_closed_loop_tick.py`, and a hand-written map that calls the controller the way exp20 does. The primitive in the jaxpr is the tell: unpacking gives `['add', 'unstack']` and indexing gives `['add', 'slice', 'squeeze']`. The research script carries its own copy of the map, so the measurement never touched the shipped one, and nothing in the tree ran `propagation="tick"` end to end.

**There were two sites.** The controller's pose is one; the tick map's own `_advance_belief` is the other (`x, y, heading = belief`), and the second one only fires with the digital twin, because the belief is a traced array there. So the twin's tick map was broken from the start, and the default plant's worked as soon as the controller was fixed. The rule is now the same in both places, and it is a rule for the whole tree: **no array is unpacked in a path that immrax embeds.**

**Why the measurement did not catch it.** The tick map was measured on **JAX 0.6.2** (the environment the tree's own notes record), where the unpacking compiled to slices. The stack a fresh `install.bat` brings is JAX 0.12.0, where it does not. The measurements were right; the code did not run on the stack the project installs.

**Fix (implemented):** the pose is read by indexing, which compiles to `slice`+`squeeze` and is in the registry. It is the same number: the plant propagation's envelope and tracking error on `examples/example_straight.json` are unchanged to the last digit (0.3047634911948969 cm and 0.3409694880247116 cm before and after), and `tests/test_jax_ramsete.py` pins both the absence of `unstack` in the controller's jaxpr and the parity with the firmware's own math.

**Also fixed (the wiring).** The tick propagation was implemented but unreachable: `main.py` built the validator without a `propagation` argument, `research/harness.py` did the same, and the machine-readable report did not say which model produced the envelope. There is now a `--propagation [plant|tick]` flag on both, and every candidate in the `##OVERBOARD-VALIDATION##` line carries the `propagation` and the `inclusion` functions, so an envelope is never read without the model that produced it (the same rule as `envelope_truncated`). The studio names the model in the validation note and in the chip tooltip.

**Measured (the two propagations, side by side).** The six example missions, the same tree, the same robot, the same candidates (`docs/measurements/2026-10-20-tick/`):

| mission | duration | envelope (plant) | envelope (tick) | tracking | verdict |
|---|---:|---:|---:|---:|---|
| example_straight | 2.377 s | 0.305 cm | **0.292 cm** | 0.341 cm | pass, both |
| example_mission_first4 | 1.833 s | 0.310 cm | **0.195 cm** | 0.235 cm | pass, both |
| example_s_curve | 3.667 s | 0.732 cm | **0.250 cm** | 1.468 cm | fail, both |
| example_with_stops | 6.249 s | 0.915 cm | **0.283 cm** | 0.962 cm | fail, both |
| example_with_events | 4.416 s | 0.816 cm | **0.474 cm** | 0.459 cm | pass, both |
| example_complex_mission | 6.072 s | 1.239 cm | **0.283 cm** | 0.437 cm | pass, both |

The tick map is uniformly tighter and much tighter on the curves (2.9x on the S-curve, 3.2x on the stops, 4.4x on the complex, and 1.04x on the straight, where the feedback has nothing to contract: a straight's command difference is exactly zero). **No verdict changes**: the two failing missions fail on the *tracking* gate, which the tick map does not touch - the tracking error is the tube's midpoint against the plan, so it is decided by the discretisation and not by the uncertainty. The envelope was never the failing gate on this set, because the declared ranges in `examples/fll_choreo.json` are narrow test assumptions; the plant propagation's own numbers are much worse on a *calibrated* robot (the S-curve's plant tube is cut off at the 50 mm guard with a +/-1 % gain band, 53.6 mm against the tick map's 18.8 mm). Tier 1 targeted the envelope, and on the shipped example config the envelope was not what was failing.

**The rule for a calibrated robot: allowed, and it is the conservative direction.** With `--sysid-fit` the tick tube is *wider* than the plant tube (12.1 cm against 5.1 cm on the S-curve), and the reason is the twin's plant state - the two wheel speeds are interval coordinates whose spread feeds the command, and the interval arithmetic counts that growth twice - and not the belief (`docs/research/rq_a/exp23_twin_tick_belief.py`). A wider tube is a *tighter* bound on nothing: it is the same reachable set, enclosed more generously, so the gate is **harder** to pass, not easier. The flag is therefore allowed with a fit, and the report says which model produced the number, so a bigger envelope is read as what it is.

**The obvious repair for that was tried and measured worse.** The estimate and the truth are perfectly correlated - the odometry reads the very wheel speeds the plant carries - so carrying the estimate's **error** instead of the estimate should stop the box from adding the estimate's spread to the truth's. It is implemented (`belief_mode` in `immrax_validator.py`, "error" against "estimate") and it does not pay: the S-curve goes from 5.1 cm with the plant propagation to **42.7 cm** with the error form and the Jacobian, and to 9.5 cm with the mixed-Jacobian. The reason is that the error's update differences two cosines of a heading interval that is itself wide, and the difference of two over-approximations is worse than either - the same shape as the problem it was meant to solve. The estimate therefore stays a state of its own, and the switch is kept so the next person does not have to derive it again.

**Workaround:** None needed. The default stays on the plant propagation, so nothing changes for a run that does not ask for the tick map.

---

### KI-051 - `--self-consistent` was a no-op: the switch could not change the plan

**Status:** Fixed. The flag is two-way now (`--self-consistent/--no-self-consistent`, default on) and the CLI passes the value to the solver explicitly, so the trapezoidal collocation is reachable again.

**Files:** `main.py` (`--self-consistent/--no-self-consistent`), `optimizer.py` (`solve`), `research/harness.py` (`--self-consistent`), `viz/studio.html` (the switch), `viz/studio.js` (the switch), `docs/research/RQ-A_tracking_error.md` (F10)

**Problem:** The self-consistent collocation (KI-026) was an opt-in, and the default was then flipped to `True` without the CLI being changed with it: `main.py` only ever set `solve_kwargs["self_consistent"] = True` and never `False`, so the switch could not change the plan and the trapezoidal rule could not be reached from the command line. `research/harness.py` had the same one-way switch.

**Evidence:** Solving `examples/example_s_curve.json` with and without `--self-consistent` produced identical samples (21 samples, 3.725 s, every field of every sample equal); the only difference between the two files was the trajectory's `name`, which comes from the output file name.

**Fix (implemented):** the flag is a pair, `--self-consistent/--no-self-consistent`, with the default on, and the CLI passes the value through explicitly. The two plans are measurably different: the S-curve is 3.725 s / 1.250 m self-consistent and 3.696 s / 1.233 m trapezoidal, with tracking errors of 0.55 cm and 1.31 cm. The Multi-Verse path uses the trapezoidal rule and refuses the flag when it is given explicitly.

**What is left:** the CLI's default and the **library's** default still disagree - `TrajectoryOptimizer.solve` has `self_consistent=False`, so a library caller gets the trapezoidal plan for the same mission the CLI plans self-consistently. That is KI-052.

### KI-052 - The CLI and the library plan different trajectories for the same mission

**Status:** Open.

**Files:** `optimizer.py` (`solve`), `main.py` (`--self-consistent/--no-self-consistent`), `research/harness.py` (`--self-consistent`), `docs/research/RQ-A_tracking_error.md` (F10), `docs/HEURISTICS_AND_VERIFICATION.md` (§9)

**Problem:** The self-consistent collocation (KI-026) is the **command line's** default: `main.py` has `--self-consistent/--no-self-consistent` with the default on, and it passes the value to the solver explicitly. The **library's** default is the other way round: `TrajectoryOptimizer.solve` has `self_consistent=False`, so the same mission solved through the library is planned with the trapezoidal rule.

So the same waypoints, the same robot and the same solver give two different plans depending on who asked. On `examples/example_s_curve.json` the two are 3.725 s / 1.250 m (self-consistent) and 3.696 s / 1.233 m (trapezoidal), and their tracking errors are 0.55 cm and 1.31 cm - more than a factor of two apart on the gate that decides the mission.

**Why it matters:** the numbers in `docs/HEURISTICS_AND_VERIFICATION.md` §9 were measured with the trapezoidal rule, and the CLI now plans the other one by default. A reader who reproduces a table by hand from the command line gets different trajectories from the ones the table quotes, so the tables stop describing the shipped behaviour until they are re-measured. The tree's own rule - the source is the authority, and a number without the model that produced it cannot be read - applies here as well.

**Workaround:** Pass `self_consistent=True` (or `False`) explicitly when the library is used, and say which collocation a number was measured with. `research/harness.py` follows the library, so its numbers are the trapezoidal ones unless it is told otherwise. **Making the two defaults the same would close it**; whichever way it goes, the flag's help text and the notes have to name the default they use.
---

## P2 — cleanup

### KI-017 — Convergence flags do nothing

**Status:** Fixed.

**Files:** `main.py`, `multiverse_optimizer.py`

**Problem:** `--convergence-output` and `--convergence-animate` were parsed but never used. `MasterTrajectoryOptimizer` declared `iteration_history` but never filled it.

**Fix (implemented):** The dead flags were removed from `main.py`. `MasterTrajectoryOptimizer` now captures and forwards `iteration_history` in `multiverse_optimizer.py` when requested by CLI or API.

---

### KI-018 — Dead code, duplicates, and generated files

**Status:** Fixed in the working tree (uncommitted). The dead code is gone, the duplicate mission copy is gone, the two mission copies the robot needs are regenerated, and the decision on each is written down.

**Files:** `pybricks_code/ramsete_fast.py` (intentionally kept), `pybricks_code/complex_mission.py`, `pybricks_code/complete_mission.py`, `immrax_validator.py`, `robot_model.py`, `digital_twin.py`, `sysid_fit.py`, `sysid_compare.py`, `.gitignore`

**Problem:** Cleanup removed the duplicate `examples/complete_mission.py`, the root `test_jax_ramsete.py` (its boundary test moved to `tests/test_jax_ramsete.py`), `docs/API_optimizer.md`, generated example/controller/benchmark outputs, and unused imports plus `send_candidates`/`send_status`. `ramsete_fast.py` is intentionally kept: it is the production controller used on the real robot. Remaining: generated mission copies in `pybricks_code/` and `examples/complex_mission.py`; `get_safety_ribbon_data` is still unused and is no longer documented in `docs/WHITEPAPER.md`. The live view, its dead regenerate controls and the `live_visualizer.py` module are gone.

**Evidence:** The suite runs 454 tests with 0 skipped. `random_suite/run_000/plot.png` is kept because `README.md` embeds it as a historical example.

**Fix (implemented):**

- **The generated mission copies: decided.** `pybricks_code/complex_mission.py` and `pybricks_code/complete_mission.py` stay tracked, because they are **firmware dependencies**: `mission_4_ramsete.py` and `mission_5_lambdas.py` import their `samples` and `config`, and `sysid_digital_twin.py` imports `complex_mission` for its held-out replay test. `examples/complex_mission.py` was a third copy that nothing imported; it is deleted and the name is now ignored, so a stray export into `examples/` cannot be committed by accident.
- **Both copies were regenerated**, because they were five months old and carried the *old* robot: `t_max_nm 0.040` against the measured `0.138` and `v_max_rad_s 15.7` against `18.94`. `mission_4_ramsete.py` hands that config to the controller's `KinematicLimiter`, so the robot was limiting itself against a motor that is more than three times too weak. The regenerated files come from the current planner and the current robot config. `complex_mission.py` passes the gates (6.068 s, envelope 1.19 cm). **`complete_mission.py` does not**: `examples/example_complete.json` misses the envelope gate on its last two segments (2.82 cm against 2 cm) and the tracking gate (0.68 cm against 0.50 cm), like the stops and events example missions do. It is exported without `--validate` and is a **demonstration of the lambda event API**, not a plan to drive; the command that produced it is in `pybricks_code/README.md`.
- **`get_safety_ribbon_data` is removed.** It was unused and undocumented; the reachability envelope is read from the validator's report.
- **Three more unused functions found by an audit of every definition in the tree**: `sysid_compare.load_trajectory` (a thin wrapper around `load_trajectory_document`), `sysid_fit._read_wheel_radius` (a "backward-compatible wrapper" nothing called) and `digital_twin._non_negative_float` / `_range`. Two unused imports went with them (`jax.jit` in `immrax_validator.py`, `numpy` in `robot_model.py`).
- **Kept on purpose:** `robot_model.get_dynamics_symbolic` (documented in `docs/API_robot_model.md`, and its formula matches `get_dynamics`), `gui_server.log_message` / `do_GET` / `do_POST` (the `BaseHTTPRequestHandler` hooks), `jax_robot_model.tree_flatten` / `tree_unflatten` (the pytree protocol JAX calls by name), `ramsete_fast.py` (the production controller), `sysid_solver.py` (a standalone script, documented as such).

**Workaround:** None needed. `docs/research/rq_a/verify_docs.py` checks the documentation, and the generated files carry a header naming the command's output, so a stale copy is visible in the diff.

---

### KI-019 - The reachability tube lags the reference by one sample

**Status:** Fixed in the working tree (uncommitted). Each interval is now driven by the same linear reference and 20 ms sample-and-hold loop as `pybricks_code/ramsete_fast.py`, with the plant integrated in 5 ms substeps. Tracking error and envelope radius are reported separately and checked independently; the envelope is checked against `main.py`'s 2 cm transit threshold.

**Files:** `immrax_validator.py` (`compute_reachability`, `_interpolated_reference`, `RamseteController`), `jax_ramsete.py`

**Problem:** At step `i`, the interval tube was propagated with sample `i`'s reference pose and velocity over `[t_i, t_{i+1}]`, but the result was compared to sample `i+1`. Sample 0 is at rest, so the tube started one step behind and stayed there. The tracking error was roughly one sample spacing even with zero uncertainty.

**Evidence:** On the straight example with every uncertainty collapsed to nominal (`cof=1.37`, torque margin `1.0`, zero backlash, exact track width/wheel radius/inertia), the old model reported `max_tracking_error_m=0.1585 m` against a 5 mm threshold while the envelope radius was only 1.5-2 mm. After the fix, the same straight example reports `max_tracking_error_m≈0.00346 m` and zero motor, slip, and speed violations; the envelope is 0.0028 m under the current narrowed test ranges. The reference is linear (not Hermite) because that is what `ramsete_fast.py` follows.

**Fix (implemented):** Use linear pose/velocity interpolation and the precomputed Ramsete gain from `ramsete_fast.py`, update the reference and controller once per 20 ms tick, hold the command, and integrate the plant in 5 ms substeps. Do not relax the 5 mm tracking or 2 cm envelope thresholds to hide a model error.

**Workaround:** None needed for the straight example. Some representative paths (S-curve, stops, events) still fail until the optimizer is driven by the same tracking model; the complex mission passes under the narrowed test ranges (KI-020).

### KI-020 — Validator is not yet a full digital twin or an optimizer objective

**Status:** Open, partially addressed in the working tree. The controller parity, linear reference, 20 ms sample-and-hold, and physical backlash model are implemented. The controller split is also done: `robot_config.py` is hardware-only, `mission_4_ramsete.py` imports `ramsete_fast` with its calibrated defaults, and `mission_5_lambdas.py` imports `ramsete` for events. `pybricks_code/sysid_digital_twin.py` and `sysid_fit.py` provide the robot log/capture and host fit path; no hardware logs have been collected yet. The host can now push a script to the hub and capture its console output without the Pybricks IDE (`hub_deploy.py`, `main.py --push-hub`), so the robot log no longer has to be copied out by hand, and a pushed trajectory reports its measured final pose as a `POSE` line. The fitted numbers now carry an uncertainty: the calibration-critical tests run three times each, the fit reports a 95% t-interval per parameter over those repeats, and the validator plans against that interval instead of the spread of the steps inside one run, which was not an uncertainty because the steps are sequential. The interval covers the run-to-run variation and any drift over the session, but not the error of the tape measure: the tape error is in the numerator of every measurement, it is the same every time, and no number of repeats averages it out. Two parameters that the robot config could only declare are now measured as well: the straight-line heading drift, which the plant adds to the heading rate because the plan's straight segments are straight, and the motor's stall torque, which the wall test measures and the validator uses as the motor force limit. **The motor is now measured on both ends and every tool uses the measured pair.** The collector's new `top_speed` test measures the free-running speed with the wheels off the ground, which is the other end of the torque-speed line; the fit rejects the number when the robot reached the commanded speed instead, because then the command was the limit and not the motor. `main.py --sysid-fit` writes the measured pair into the robot config, so the optimizer, the CPU audit, the validator and the exported controller all plan against the same motor (the KI-010 rule). Before this, the optimizer planned against the declared motor while the validator checked the measured one, so a plan could satisfy its own constraint and still trip the validator. The drivetrain model itself is in `drivetrain.py`, and its structure was checked against an independent rigid-body simulator: the top speed agrees within 7%, the coast deceleration is 2.4x higher in the simulator because the model has no rolling resistance, and that simulator's brake mode turns out to reverse the wheels instead of braking them, so its braking distance is not a usable reference. See `docs/research/RQ-B_drivetrain_model.md`. The drift is a slope against the speed, not a subtraction, so the gyro's bias lands in the intercept instead of leaking into the drift; the tape's lateral offset is the independent second estimate. The step fit is a two-stage search, so the reported time constant and delay are no longer quantised to the 10 ms grid of the first pass. The wall is also a reference for the distance scale: the robot drives into it from a marked start and stops when it is blocked, so the scale can be checked against something that does not move and that nobody has to read, which is what makes it possible to re-check the calibration unattended. The wall and the tape are two estimates of the same number and the fit reports them side by side. The wall cannot measure the heading, because a wall does not straighten a crooked robot, so the drift still comes from the speed regression. The host tools accept a minimal drivebase profile (`wheel_radius_m`, optional `track_width_m`) instead of a full robot config. `digital_twin.py` and `main.py --sysid-fit` now load and validate a fit. When enabled, the validator uses a five-state plant with actual wheel speeds, the fitted straight-line gain interval, first-order time constant, command delay, and odometry distance/heading scale in feedback. The nominal feedback path also mirrors the firmware's rounded `DriveBase.drive()` arguments (whole mm/s, whole deg/s). Braking slip is now applied by the plant. Reachability is a post-solve candidate score (`--candidates`), not yet an optimizer term. The firmware's DriveBase trajectory limits are now known (about 7800 mm/s² and 9300 deg/s² for this robot, derived from Pybricks' 20,000 deg/s² ceiling), so the plant's instant-tracking assumption matches the hub; Motor/current transients, DriveBase PID, and slip under load remain unimplemented; the whole-millimeter distance reading was measured rather than assumed and costs 0.75 mm of final drift on the 6.08 s complex mission, so it is not worth modelling. The machinery for the rest is built and inert until the fit provides the numbers: the controller is fed the robot's own integrated odometry belief instead of the truth, the plant applies the measured braking shortfall and the measured acceleration limit, and the belief drifts with the measured gyro bias. `docs/DIGITAL_TWIN_PLAN.md` records the data contract, what is built, and what is still open. The measurement loop around them is now complete on the host: `main.py --push-hub --hub-log session.log` captures the hub console output, `sysid_fit.py --from-hub-session session.log --out-poses poses.csv` turns its `POSE` lines into the measured-pose CSV that `sysid_compare.py` reads, and the sysid battery feeds `sysid_fit.py` the same way. The measured poses are matched to the sysid log by `run_id`, and only `odometry_straight`, `odometry_turn` and `brake_from_*` runs contribute a distance scale, a heading scale or a braking-slip estimate, so a poses file with the wrong run id is silently ignored ("no measured poses supplied").

**Files:** `optimizer.py`, `immrax_validator.py`, `jax_ramsete.py`, `digital_twin.py`, `drivetrain.py`, `robot_model.py`, `main.py`, `multiverse_optimizer.py`, `pybricks_code/ramsete_fast.py`, `pybricks_code/robot_config.py`, `pybricks_code/mission_4_ramsete.py`, `pybricks_code/mission_5_lambdas.py`, `pybricks_code/sysid_digital_twin.py`, `pybricks_code/hub_runner.py`, `pybricks_code/ramsete_compact_events.py`, `sysid_fit.py`, `sysid_compare.py`, `sysid_collect.py`, `hub_deploy.py`, `examples/drivebase_profile.example.json`, `tests/test_sysid_fit.py`, `tests/test_sysid_compare.py`, `tests/test_digital_twin.py`, `tests/test_drivetrain.py`, `tests/test_hub_deploy.py`, `tests/test_sysid_collect.py`

**Problem:** The default validator mirrors the deployed controller's math and timing, but its plant is still kinematic: commanded wheel speeds become motion immediately (apart from wheel backlash). The optional fitted plant adds actuator lag, delay, gain uncertainty, and a static odometry correction, but it still has no motor current/torque transient, DriveBase PID, integrated encoder/gyro odometry, or wheel-slip dynamics. Feedback uses a scaled version of the true pose, not the robot's actual `DriveBase.state()` estimate. Reachability is a post-solve candidate score (`--candidates`, `main.py`), not yet a term or constraint in the optimizer.

**Evidence:** The straight example passes with `max_tracking_error_m≈0.00346 m` and `max_envelope_radius_m≈0.0028 m` under the narrowed test ranges. Representative paths still expose the gap: the S-curve example reports `max_tracking_error_m≈0.0132 m` and `max_envelope_radius_m≈0.0077 m`; the stops example reports 0.0052-0.0060 m / 0.0093-0.0095 m; the events example reports 0.0053-0.0054 m / 0.0097 m; the complex mission now passes at 6.068 s with 0.0049 m / 0.0122 m. These are model predictions, not hardware measurements. There is no real tracking log in the repository yet; the logger and fitter have only been exercised with synthetic data. A synthetic fit with `gain=1.0, tau=0.08 s` already rejects the straight example with 3.44 cm tracking error, which shows why hardware calibration matters before enabling the fitted model. The old `accuracy_weight += 2.0` retry is removed: on the complex mission the slow 17.1 s candidate diverged, while the 6.07 s one stayed bounded. A uniform-timing five-seed sweep (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`) shows that on the complex mission all four non-baseline seeds converge to slower, wider paths (8.1-8.7 s, 1.55-1.76 cm, tracking 0.53-0.72 cm), so the 6.07 s / 1.22 cm baseline still wins. Guess diversity alone does not beat the equal-duration baseline; the default simple-path candidates now mix the two timing modes (KI-022). Per-segment timing is path-dependent: on an uneven 1.0 m + 0.2 m path, `accuracy_weight=0.005` passes all gates at 2.784 s / 0.32 cm envelope / 0.36 cm tracking versus 4.456 s / 0.37 cm / 0.36 cm equal-duration at the same weight, so the mixed loop selects per-segment; at `accuracy_weight=0` per-segment is 2.710 s / 0.32 cm but fails the Immrax and CPU gates, so the mixed loop falls back to the equal-duration candidate (4.463 s / 0.35 cm). On the complex mission at `accuracy_weight=0` per-segment is faster (4.650 s / 1.00 cm) but its tracking is 0.84 cm and its motor force is 0.074 N over the limit; at `accuracy_weight=0.02` it is 7.333 s / 1.58 cm with 0.53 cm tracking, still over the 5 mm gate, and worse than the uniform 6.068 s / 1.22 cm baseline. The candidate loop now mixes both timing modes by default (`--uniform-dt`/`--per-segment-dt` force one) and runs the CPU audit on every candidate, so a partial pass cannot block a later fully passing candidate.

**Fix:** Collect and fit in one command with `python sysid_collect.py --conn ble --profile examples/drivebase_profile.example.json`: it pushes the logger, captures the session as it arrives, converts the hub's `POSE` lines to `poses.csv`, and writes `sysid_fit.json`. The poses are the hub's own odometry, so the distance and heading scale of the fit have no ground truth and stay at 1.0 until someone measures the robot with a tape and replaces those rows; the actuator part of the fit does not need a measurement. The written procedure is `docs/CALIBRATION_PROCEDURE.md`: the operator drives the robot through the collector tests, measures the end pose of the five that need a tape, and fills in a measurement sheet keyed by test name. `python sysid_fit.py <session.csv> --from-sheet <sheet> --out-poses poses.csv` turns that sheet into the measured poses, so the order the tests ran in does not matter and a skipped test is reported instead of guessed. The sheet is also fitted directly (`--sheet`), so there is no intermediate pose file, and it is written as a form in a browser (`--write-form`, `sheet_form.py`) so the operator types the numbers instead of editing a spreadsheet; the studio has the same form on the server. The same thing step by step: push the logger and capture its session with `python hub_deploy.py pybricks_code/sysid_digital_twin.py --out session.csv` (or run it from the Pybricks IDE), record the stop/final poses in `poses.csv`, fit the logs with `sysid_fit.py --profile examples/drivebase_profile.example.json`, and pass the result with `main.py --sysid-fit sysid_fit.json --validate`. Compare the planned trajectory with the measured poses using `sysid_compare.py`; it prints position/heading errors and whether each measured pose is inside the reachability envelope, and it accepts a plain predicted-pose CSV and a drivebase profile. The pushed run gives the same comparison without hand-editing: `main.py --push-hub --hub-log session.log`, then `sysid_fit.py --from-hub-session session.log --out-poses poses.csv --run-id <id>`, then `sysid_compare.py output.traj poses.csv --run-id <id>`. Extend the plant to integrated odometry and a braking-slip state only after that comparison. The post-solve choice is made: `--candidates` scores several solutions (seed/timing combinations with `--simple`, accuracy-weight variants on the Multi-Verse path) and picks the passing one with the lowest envelope (then the shortest). Extend it to a hard constraint or differentiable proxy only if the candidate set cannot find a passing path. Do not tune the 5 mm/2 cm thresholds or b/zeta to one path.

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

### KI-027 - The reference interpolated the heading and the turn rate as independent lines

**Status:** Fixed in the working tree (uncommitted). The turn rate is the derivative of the interpolated heading, in the four firmware controllers and in the validator together.

**Files:** `pybricks_code/ramsete_fast.py`, `pybricks_code/ramsete_compact.py`, `pybricks_code/ramsete_compact_events.py`, `pybricks_code/ramsete.py`, `immrax_validator.py` (`_interpolated_reference`), `tests/test_jax_ramsete.py`

**Problem:** The reference interpolated the heading, the speed, the turn rate and the Ramsete gain as four independent lines between the samples. They are coupled: the turn rate is the derivative of the heading. A line for the turn rate means the heading is a parabola, and the two disagree by up to `domega*dt/8` mid-interval - 8.7 degrees of contradictory heading on the S-curve at the default density. The robot was handed a heading target and a turn rate that described different motions, and the validator modelled the same mistake, which is why the model and the robot agreed on the bug.

**Evidence:** See `docs/research/RQ-A_tracking_error.md` finding F7. Measured after the fix, per-segment timing: the S-curve at n=40 goes from 0.98 cm to 0.27 cm (pass), the stops mission at n=40 from 1.29 cm to 0.41 cm (pass), the complex mission at n=10 from 0.86 cm to 0.41 cm (pass) and at n=20 from 1.05 cm to 0.39 cm (pass). The S-curve at the default density gets worse (1.25 cm to 1.51 cm), which is F7's own finding: the coupling cannot be fixed in isolation, the plan has to be self-consistent too (KI-026), and the two together are what the candidate sweep is for.

**Fix (implemented):** The turn rate is now `dh / dt` - the derivative of the interpolated heading - in `ramsete_fast.py`, `ramsete_compact.py`, `ramsete_compact_events.py`, `ramsete.py` and in the validator's `_interpolated_reference`. The model takes the heading difference the short way, exactly as the firmware does, so the two agree even when a sample interval turns more than 180 degrees. The gain was left alone: F7 measured it as worth nothing.

**Workaround:** None needed. The firmware and the validator changed together, so the parity holds; the parity test in `tests/test_jax_ramsete.py` was updated to the coupled reference and a wrap case was added.

---

### KI-026 - The plan's samples were not on the path its own profile traces

**Status:** Fixed. The CLI plans a self-consistent trajectory by default (`--self-consistent/--no-self-consistent`, default on, KI-026), and `--no-self-consistent` restores the trapezoidal rule. The **library's** default is the other way round (`TrajectoryOptimizer.solve` has `self_consistent=False`), which is KI-052.

**Files:** `optimizer.py` (`solve`, `_solve_once`, `_GAUSS_LEGENDRE_4`), `main.py` (`--self-consistent`), `tests/test_optimizer.py` (`TestSelfConsistentPlan`), `docs/research/RQ-A_tracking_error.md`

**Problem:** The collocation placed the samples with the trapezoidal rule, which samples only the two endpoints of an interval. The result is a plan whose positions are not where the trajectory's own (v, omega) profile takes it, so the deployed controller is told to be somewhere its own profile never goes. The heading samples are free variables, so they are not the path's tangents either. That is the mismatch the tracking gate measures (KI-020, `docs/research/RQ-A_tracking_error.md`).

**Evidence:** On the S-curve at the default sampling the plan sits 11.5 mm off its own profile and the tracking error is 1.31-1.32 cm. With the self-consistent collocation the plan sits on its own profile (under 0.1 mm) and the tracking error falls to 0.55-0.59 cm, a 58 % reduction. The events mission, which failed both candidates, passes both (0.32 cm and 0.40 cm). The validator's own forward-integration check stops reporting its 1.3 cm error, because the plan now agrees with itself. See RQ-A finding F10.

**Fix (implemented):** The position rule integrates the interval's velocity along the heading the robot will hold - the turn rate integrated, not the endpoints - with a four-node Gauss-Legendre rule (O(dt^7) per interval, against the trapezoid's O(dt^3)). The heading rule is unchanged, so the heading samples are the path's tangents by construction. Nothing on the robot changes: the firmware still follows the chord between the samples, and the chord is now within the sagitta of the path.

**Also measured:** The change is not a free win, and it is not the whole fix. The S-curve still misses the 5 mm gate by 0.05 cm. The complex mission's passing candidate moves from 0.49 cm to 0.51 cm, so it now fails by 0.01 cm. The stops mission gets slower (6.2 s to 7.8 s) because a direction reversal is where a self-consistent plan is hardest (RQ-A OQ5). The first-four-waypoints mission is unchanged. These numbers are measured with RQ-A's F7, the coupled reference interpolation, already in the tree; before F7 landed the S-curve's self-consistent candidate measured 0.47 cm and passed, so the two changes interact and have to be tuned together.

**Workaround:** The command line plans the collocation by default, so a run needs no switch; `--no-self-consistent` restores the trapezoidal rule. The candidate set, the sweep and the seed conclusions have to be re-measured under it, and the stops mission needs its own treatment. The tracking gate still measures a discretisation mismatch until the control tick is re-measured (RQ-A OQ1) - which KI-028 has now done, and the numbers above predate it. The **library** still defaults to the trapezoidal rule, so a library caller has to ask for the other one (KI-052).

### KI-028 — The validator modelled a controller that is not the one that runs

**Status:** Fixed in commit 12a365b. The validator now models the command resolution and the Ramsete gain of the controllers that are actually in use, and `pybricks_code/ramsete_compact.py` has been brought into line with its event-capable twin so the two cannot drift apart. The control period was never the problem - it was 20 ms in the firmware and in the model all along - so the tracking gate did not move, and the measurements that quote it are still valid.

**Files:** `immrax_validator.py`, `pybricks_code/ramsete_compact.py`, `tests/test_firmware_events.py` (`TestFirmwareParity`), `tests/test_jax_ramsete.py`, `docs/research/rq_a/exp16_firmware_parity.py`, `docs/research/RQ-A_tracking_error.md` (Fix 4)

**Problem:** The validator was written against `pybricks_code/ramsete_fast.py`, the legacy list-of-dict controller, and the controllers that are actually in use did not all do the same thing:

| | `ramsete_compact_events.py` (the pushed run) | `ramsete_compact.py` (the SysId replay) | `ramsete.py` (the IDE missions) | the validator modelled |
|---|---|---|---|---|
| control period | 20 ms, synced | 20 ms, synced | 20 ms, synced | 20 ms |
| drive command | floats | rounded to mm/s and deg/s | floats | rounded |
| Ramsete gain | recomputed from the interpolated v, w | interpolated per sample | recomputed | interpolated |

So there was no single behaviour for one validator to mirror: the plain compact controller rounded the command and interpolated the gain, the way the validator did, while the deployed one and the IDE missions did neither - and the plain one carried its own copy of the odometry and the Ramsete maths, so it could drift away from its twin unnoticed.

**Evidence:** The command and the gain are worth about 1 % of the tracking error (finding F7 measured the gain as worth nothing on its own), so the numbers barely move: the legacy mirror and the shipped one differ by 0.94-1.02x on every (mission, density) cell. The command hold is the expensive line in principle - it is proportional to the period, so a 5 ms tick and a 20 ms tick differ by a factor of four - but the tree ran 20 ms throughout, so the gate was never decided on a wrong period. `docs/research/rq_a/exp16_firmware_parity.py` has the table; the mechanism is finding F6 in `docs/research/RQ-A_tracking_error.md`.

**Fix (implemented):** One control period for every controller that is in use, and for the model.

- `pybricks_code/ramsete_compact.py`, `pybricks_code/ramsete_compact_events.py` and `pybricks_code/ramsete.py` all run a **synced 20 ms heartbeat**. A synced heartbeat is what makes the period exact: the loop waits for the slack to the next tick, so the hold is 20 ms and not 20 ms plus the loop's own compute time. It is also the hub's budget - the loop does the odometry update, the controller and the limiter every iteration - and 20 ms is the period `ramsete_fast.py` always used.
- `pybricks_code/ramsete_compact.py` now uses the shared `Odometry`, `RamseteController` and `KinematicLimiter` from `pybricks_code/ramsete.py`, the same 20 ms loop as its event-capable twin, and the same unrounded drive command, so the two are the same controller with and without events.
- `immrax_validator.py` holds the command for 20 ms (`CONTROL_DT`), does not round it, and recomputes the Ramsete gain from the interpolated speed and turn rate.
- `tests/test_firmware_events.py::TestFirmwareParity` pins the parity: the controllers in use must declare the same control period, the validator's period must equal it, the two compact controllers must use the shared parts, and the validator must not round the command.

**What it does and does not change.** With the period the same on both sides, the remaining parity fixes are worth about 1 % (the rounding and the gain, which finding F7 measured as worth nothing), so the tracking numbers barely move: the legacy mirror and the deployed mirror now differ by 0.94-1.02x on every (mission, density) cell. What changes is that the model now says what the robot does, and that the two compact controllers can no longer drift apart. The gate is unchanged in its strictness relative to the 20 ms model the tree already used, and stricter than a 5 ms one would be.

**Workaround:** None needed. The two gaps recorded in the RQ-A note are now resolved:
1. `KinematicLimiter` no longer injects an artificial velocity boost on direction reversal; backlash is strictly treated as a position uncertainty in the reachable set (KI-025), and the limiter purely enforces physical torque and traction limits on commanded accelerations. It now also derates the **inner** wheel's traction with the lateral weight transfer - the same `nl = mg/2 - m*h*a_c/b` the host's `wheel_normal_forces` computes (KI-010) - and its fallback numbers are the robot's own (mass 1.5 kg, 19.4 rad/s, 0.138 Nm, `cof` 0.6, `h_cg` 0.05 m), so an empty config no longer plans a weaker robot than the one the host plans for.
2. `pybricks_code/ramsete_fast.py` now recomputes the Ramsete gain from interpolated reference speed and turn rate (`math.sqrt(wr**2 + b*vr**2)`), passes unrounded float commands to `DriveBase.drive()`, and applies `KinematicLimiter` when `config` is provided, bringing all four firmware controllers in `pybricks_code/` into parity with the validator's unquantized command model.

**One implementation, not four.** The duplication KI-028 described is now gone: `pybricks_code/ramsete.py` holds the odometry, the Ramsete maths, the limiter and the 20 ms loop, and the other modules are thin wrappers over it - `ramsete_fast.py` delegates to `ramsete.py`, and `ramsete_compact.py` delegates to `ramsete_compact_events.py` with the events off. A second copy of the controller can no longer drift away from the one that runs, which is what made the table above possible in the first place. `tests/test_firmware_events.py::TestFirmwareParity` pins the result: the two compact controllers must use the shared parts, the limiter's fallback numbers must be the robot's own, the limiter must apply the lateral weight transfer, and the wrapper must still drive a trajectory.

---

### KI-029 — A late control tick silently skipped a mission event

**Status:** Fixed in the working tree (uncommitted). The event of every sample the robot has reached is now fired, not only the one it is on. The deployed controller (`ramsete_compact_events.py`) was never affected; the reference controller for the IDE missions (`ramsete.py`) was.

**Files:** `pybricks_code/ramsete.py` (the sample advance and the event handling), `tests/test_firmware_events.py` (`TestLateTickEvents`)

**Problem:** `ramsete.py` advanced the sample index in a loop and then fired the event of **the sample it was on**:

```python
while sample_idx < num_samples - 2 and samples[sample_idx + 1]['t'] <= t:
    sample_idx += 1
...
_fire_event(s1, event_map, debug, t)          # only the current sample
```

A tick that arrives late jumps the index over one or more samples, and the event of a sample that was jumped over was never looked at. The action was **silently lost** - the same class of failure as KI-012, in a different place.

**Evidence:** The control period and the sample spacing are both 20 ms, so the index advances by exactly one sample per tick in the nominal case and by two or three when the tick is late. A late tick is not exotic: a garbage collection, a scheduling hiccup or the hub's other tasks are enough. The test `TestFinalWaypointEvent.test_mid_trajectory_event_fires_once` failed in the full suite and passed when run alone - it is load dependent, not random.

**Fix (implemented):** The event is fired for every sample the advance passes over, and the sample is marked as fired so it still runs exactly once. `ramsete_compact_events.py`, the controller that runs on the hub, already did this (`while ev_idx < n_events and events[ev_idx][0] <= s_idx`), so the two now behave the same way.

**Workaround:** None needed. `tests/test_firmware_events.py::TestLateTickEvents` drives both controllers with a clock that jumps 60 ms per tick, so the skip is deterministic: the reference controller's test fails without the fix and passes with it, and the deployed controller's test guards the behaviour that was already right.

---

### KI-045 — `sysid_solver.py` is a superseded prototype

**Status:** Fixed.

**Files:** `sysid_solver.py` (deleted), `requirements-optional.txt`

**Problem:** The file was a standalone pandas/JAX fitter with the hub's telemetry pasted in as hard-coded CSV text and the robot's numbers hard-coded as constants. `sysid_fit.py` replaced it: it parses the real session format, writes the format the validator reads, and is what the collector and the studio use.

**Fix (implemented):** Deleted `sysid_solver.py` and dropped `pandas` from `requirements-optional.txt`.

**Workaround:** None needed.

---



1. KI-001, KI-002, KI-003, KI-005, KI-006, KI-007, KI-008, KI-009, KI-010, KI-011, KI-012, KI-013, KI-014, KI-015, KI-016, KI-019, KI-021, KI-022, KI-023, KI-024, KI-025, KI-026, KI-027, KI-028, and KI-029: done in the working tree; never save or ship a failed or unvalidated run, keep validation on the configured robot, use both uncertainty bounds, keep the reference tube aligned with the samples, keep the optimizer and Immrax on the same per-wheel force model, hold both timing models to the same wheel-jerk discipline, keep the wheel force profile continuous through every waypoint the robot does not stop at, keep the backlash in the reachable set rather than in the wheel speed, derive the turn rate from the interpolated heading in the firmware and the model together, place the plan's samples on the path its own profile traces, model the controller that actually runs, fire the event of every sample the robot has reached, never take a port from another program, and keep the firmware compilable for MicroPython.
2. KI-020: controller split, log/fit tooling, and the optional `--sysid-fit` actuator/odometry path are done, and the log and pose capture now run over Bluetooth from the host; collect hardware data, compare fitted predictions with measured poses, add integrated odometry/slip, then use reachability in optimization.
3. KI-004: done — the pipeline is repaired and used as a seed provider (`--refine`); the next lever is the objective, because the pipeline's cost buys speed with tracking accuracy and the 5 mm gate rejects the trade.
4. KI-017: done — dead flags removed and iteration history forwarded.

## Not checked

This list does not cover `sysid_solver.py` beyond its imports, the plotting and animation modules in depth, the Pybricks tutorials, or the generated trajectory and benchmark data. The Pybricks code was not run on hardware.
 The motor's numbers are measured now as well: the collector's `top_speed` test lifts the robot so the motor runs free and measures the speed at which it makes no torque, the fit rejects that number when the robot reached the commanded speed instead, and `main.py --sysid-fit` writes the measured stall torque and no-load speed into the robot config. That closes a hole where the optimizer planned against the declared motor and the validator checked the measured one. The drivetrain model is in `drivetrain.py`; the model is checked against an independent simulator in `research/crosscheck/`, and the findings are in `docs/research/RQ-B_drivetrain_model.md`.

The random mission suite and the random robot constants were outside this list's scope until 2026-10-20, when a stress run measured 180 candidates on 45 random paths and 90 random robots against the production gate chain ([`docs/measurements/2026-10-20-stress/`](docs/measurements/2026-10-20-stress/)). It found one failure mode that was not in this list (KI-048, since fixed) and confirmed that the tracking and forward-integration gates are dominated by the discretisation mismatch rather than by the physics (KI-026, RQ-A). What that run does **not** cover: the paths in the suite are longer and more turny than a real FLL mission, the robot constants are drawn, not measured (KI-020), and nothing was run on hardware.
