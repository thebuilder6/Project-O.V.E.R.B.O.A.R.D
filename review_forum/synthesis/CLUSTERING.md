# Clustering v3 — 12 audits (~48 findings) → 22 canonical topics

Rule: merge only if same target-symbol group + same failure mode. Related-but-distinct stays split with Related-Topics.

## Users / classroom (student + coach)
- T001 Studio kid-strings (propagation jargon, accept-score gamble, venv, doctor port/sysid_fit) | Author student | Adversary architect | Arbiter coach | Symbols: studio.html::#opt-propagation, #opt-accept-score, install.bat::VENV_PY, diagnostics.py::run_doctor
- T002 Calibration lab vs 45-min (full-lab unfit, tape sign trap) | Author coach | Adversary physics | Arbiter student | Symbols: CALIBRATION_PROCEDURE::steps, collect::TestGuide
- T003 M1 budget + push contention (new-mission no-budget, BLE queue, USB demo) | Author coach | Adversary architect | Arbiter robotics | Symbols: EXAMPLES::M1-rule, gui_server.py::Handler:do_POST, hub_ble

## Hardware (robotics)
- T004 Config sync silent | Author robotics | Adversary architect | Arbiter verification | Symbol: hub_deploy.py::push_trajectory:config-sync
- T005 Accel-limit backstop removed | Author robotics | Adversary physics | Arbiter verification | Symbol: robot_config.py::limits
- T006 Overrun hold break + battery/drain advisory | Author robotics | Adversary controls | Arbiter verification | Symbols: ramsete.py::follow_trajectory, robot_config.py::battery

## Physics + controls (Vora + Gould)
- T007 CG height hardcoded | Author physics | Adversary controls | Arbiter verification | Symbol: drivetrain_physics.py::CG_HEIGHT_M
- T008 Lift aliases as slip | Author physics | Adversary verification | Arbiter robotics | Symbol: drivetrain_physics.py::wheel_normal_forces
- T009 Robust-Cof 1/3 + longitudinal/caster | Author physics | Adversary controls | Arbiter verification | Symbols: robot_model.py::get_traction_limit_symbolic, drivetrain_physics.py::total_traction_limit
- T010 KKT acceptable-not-optimal | Author controls | Adversary verification | Arbiter physics | Symbol: optimizer.py::IPOPT-opts
- T011 Transcription vs ZOH (GL4 vs staircase + jerk-weight-0 blind) | Author controls | Adversary physics | Arbiter verification | Symbols: optimizer.py::integrate, reachability_engine.py, jax_ramsete.py
- T012 Traction end-row gap + epigraph second set | Author controls | Adversary physics | Arbiter verification | Symbol: optimizer.py::traction-rows
- T013 Ramsete singularity + epsilon drift + dead merge | Author controls | Adversary robotics | Arbiter physics | Symbols: drivetrain_physics.py::ramsete-gains, optimizer.py::merge_conditions

## Gate (verification)
- T014 Gate-hole: propagation/inclusion not persisted (rest sound) | Author verification | Adversary architect | Arbiter robotics | Symbols: main.py::_build_validation_record, hub_deploy.py::check_trajectory_approval

## Method (Abla)
- T015 N=1 seed-7 coverage | Author research-method | Adversary ledger-guardian | Arbiter academic-lead | Symbol: research/suite.py::suite
- T016 RQ-A N=5 stale | Author research-method | Adversary academic-lead | Arbiter physics | Symbol: docs/research/RQ-A_tracking_error.md::table
- T017 Benchmark + harness provenance gaps | Author research-method | Adversary architect | Arbiter academic-lead | Symbols: benchmark.py::_save_results, research/harness.py::tree_state

## Publication (Lindqvist)
- T018 KI-011 contradiction (:554 stale) | Author academic-lead | Adversary librarian | Arbiter verification | Symbol: docs/WHITEPAPER.md::export
- T019 CFF abstract + §2.3 same-numbers + stale tables | Author academic-lead | Adversary research-method | Arbiter librarian | Symbols: CITATION.cff::abstract, WHITEPAPER.md::tables

## Architecture (Dan)
- T020 Dead code (selector + timing plumbing) | Author architect | Adversary librarian | Arbiter verification | Symbols: main.py::_select_attempt, main.py::_studio_options
- T021 Runner triplication + REPO_ROOT frozen | Author architect | Adversary robotics | Arbiter verification | Symbols: gui_server.py::start_job, paths.py::REPO_ROOT

## Docs (Ingrid)
- T022 Doc drift batch (KI-051/052, KI-020 paths, CHANGELOG, KI-062) | Author librarian | Adversary academic-lead | Arbiter moderator | Symbols: KNOWN_ISSUES.md::KI-051, CHANGELOG.md::limits

## Dreamer (Jet, R2 wave)
- T023 Native hot loop | Author dreamer | Adversary verification | Arbiter controls | Related: T014
- T024 Auto-solve on drop (local only) | Author dreamer | Adversary architect | Arbiter coach | Related: T001, T003
- T025 Basin cache (validation-tagged memo) | Author dreamer | Adversary ledger-guardian | Arbiter research-method | Related: T015

Quarantine pre-filter (Bob): rescue, Q17-tail, polish, multiverse stay buried — see archive/2026-10-28-prototype/synthesis/ + topics/quarantine/.
Counts: 22 active + 4 quarantined. Next: open atomic stubs T001..T025, debate 2 turns each, close-or-carry.
