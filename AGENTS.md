# AGENTS.md

Python CLI/library for time-optimal trajectories of a LEGO/FLL differential-drive robot (CasADi + IPOPT), plus robot-side Pybricks code. Flat modules, no packaging: run every command from the repo root.

## Read first
`KNOWN_ISSUES.md` is current as of 2026-10-19. It is the source of truth for risky paths. Fixed in the working tree (uncommitted): KI-001, KI-002, KI-003, KI-007, KI-008, KI-009, KI-010, KI-013, KI-014, and KI-019. Partially fixed: KI-015 and KI-020. Open: KI-004, KI-005, KI-006, KI-011, KI-012, KI-015 (remaining), KI-016, KI-017, KI-018, and KI-020. Cite KI IDs; do not silently "fix" these unless asked.

The docs describe the current tree, but the source is still the authority. In particular, Multi-Verse refinement is still discarded by the final polish (KI-004), `--validate` is a hard gate, and the validation model is not yet a full digital twin (KI-020).

## Commands
- Install the simple path: `pip install -r requirements.txt`.
- The default (Multi-Verse) and `--validate` paths also need `jax`, `jaxopt`, and `immrax`; `sysid_solver.py` needs `pandas`. `--simple --live` can run without JAX, but live reachability envelopes need it. These packages are not in `requirements.txt` yet (KI-006). Verified with JAX 0.6.2. Or pass `--simple` for the CasADi-only path.
- Quick, predictable run:
  `python main.py --simple -c examples/fll_choreo.json -w examples/example_straight.json -o output.traj`
- `--per-segment-dt` (with `--simple`) gives each waypoint segment its own interval. The default is one shared `dt`; the complex mission passes at 6.068 s / 1.22 cm only in that default mode. Per-segment is path-dependent: it wins on uneven segments but at `accuracy_weight=0` on the complex mission it is faster and fails the 5 mm tracking gate. Do not enable it silently.
- Drop `--simple` for Multi-Verse. It works, but the refined guess is not passed to the final solve, so it currently has no effect on the final trajectory (KI-004). Do not use `--workers`/`--no-parallel` as a quality claim.
- `--validate` gates every output and the exit code. It requires JAX/Immrax. With `--candidates N` (default 2, max 5), `main.py` solves N trajectories. With `--simple` they are distinct initial-guess seeds (`baseline`, `rolling`, `bow_left`, `bow_right`, `turn_first`) at the same base accuracy weight; the Multi-Verse path still uses accuracy weights `base, base+2.0, ...` because seed plumbing is only in `TrajectoryOptimizer`. It validates each candidate and selects a passing one by smallest envelope (shortest breaks ties). Without `--validate`, only the first candidate is solved. If validation fails, no `.traj`, controller, Python file, or `_stats.json` is written.
- `--validate --export-format controller` writes `<output>_controller.json`; `--export-format python` writes `<output>.py` (legacy list format); `--export-format python-compact` writes a packed module for `ramsete_compact.py`/`ramsete_compact_events.py`; `--benchmark` writes `<output>_stats.json` after the export gate.
- Tests: `python run_tests.py` (152 tests with JAX installed). Single test: `python -m unittest tests.test_optimizer.TestTrajectoryOptimizer.test_init -v`
- Benchmarks: `python benchmark.py --mode pipeline --config examples/fll_choreo.json` (also `cli`/`random`; the default `robot_config.json` does not exist). Pipeline mode works but reports failed solves as normal results; random mode is still broken (KI-015).
- Digital-twin fit: `python sysid_fit.py <session.csv> --poses <poses.csv> --profile examples/drivebase_profile.example.json --out sysid_fit.json`. Logs come from running `pybricks_code/sysid_digital_twin.py` on the Hub; no hardware logs are checked in. Compare planned and measured poses with `python sysid_compare.py output.traj poses.csv --run-id <id>`.
- `--plot`, `--animate`, and `--live` are blocking. `--live` serves ws://127.0.0.1:8765, waits for Enter, kills whatever owns the port (KI-005), and does not replay config to the first client (KI-016). Keep automated runs headless.

## Test quirks (verified)
- `run_tests.py` imports the JAX test modules in a `try`; if JAX is missing it prints a warning and skips those modules. With JAX installed the suite is 152 tests, 0 skipped.
- The old `test_stomp_variants.py` and `test_teb_topologies.py` modules are deleted. Do not reintroduce tests for removed `MultiVerseRefiner`/`LocalSegmentOptimizer` classes.
- JAX/refinement modules: `python -m unittest tests.test_jax_heuristics tests.test_jax_ramsete tests.test_refinement_accuracy tests.test_critic_metrics`

## Architecture and contracts
- `main.py` is the only CLI entry point. It defaults to `MasterTrajectoryOptimizer`; `--simple` selects `TrajectoryOptimizer` (CasADi/IPOPT). The Multi-Verse pipeline is bootstrap -> kinematic profile -> critic -> JAX refinement -> CasADi polish.
- Waypoints are dicts or `[x,y,heading]`, with optional `stop`, `event`, `error_threshold`, and `precision_mode`. A waypoint event on the final waypoint is exported but is not fired by the deployed Pybricks loop (KI-012).
- `robot_model.py` is the config abstraction: canonical `{"robot": {...}}` (`examples/fll_choreo.json`), legacy Choreo `{"config": {k: {"val": ...}}}`, or a flat dict. All three parse identically, including `robot.uncertainties`. `JAXRobotConfig` delegates to `RobotConfig` (KI-008).
- The optimizer and validator share one per-wheel force/normal-force model (`wheel_normal_forces`): lateral weight transfer only, no front/rear split (KI-010). The optimizer uses the declared worst-case cof, torque headroom, wheel radius, track width, and inertia; Immrax uses the same ranges (KI-009).
- Output is Choreo v3 (`trajectory.samples`). Keep the sample schema (`t,x,y,heading,vl,vr,omega`, forces, optional `event`/`reachability_envelope`) stable: validator, exporter, plotters, and the Pybricks runtime consume it.
- `pybricks_code/` is firmware code, not CPython (`pybricks.*`, `umath`); do not run it on the host. `complex_mission.py`/`complete_mission.py` are generated by `--export-format python`; regenerate, don't hand-edit. `mission_4_ramsete.py` imports `ramsete_fast.py` with its pathing defaults; `mission_5_lambdas.py` imports the event-capable `ramsete.py` reference. `ramsete_compact.py`/`ramsete_compact_events.py` consume `--export-format python-compact` and are not wired into the existing missions.
- Root-level `.traj`, `*_controller.json`, `*_stats.json`, and `test_output*` are generated artifacts, not source. `sysid_solver.py` is a standalone hardcoded-CSV pandas/JAX script, not wired into the CLI. `sysid_fit.py` is the host-side parser/fitter for `pybricks_code/sysid_digital_twin.py` logs; it does not run on the Hub.
