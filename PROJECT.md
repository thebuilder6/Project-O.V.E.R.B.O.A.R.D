# Project: Project-O.V.E.R.B.O.A.R.D Modular Refactoring

## Architecture
Project-O.V.E.R.B.O.A.R.D is refactored from a flat root layout into the modular `overboard/` package architecture with zero root shims.

### Package Hierarchy
```
overboard/
├── __init__.py
├── __main__.py
├── main.py                     # CLI entrypoint (REPO_ROOT = parent.parent)
├── gui_server.py               # Studio web server (VIZ_DIR = parent / "viz")
├── diagnostics.py              # System diagnostics & dependency checks
├── core/                       # Optimization & planning engines
│   ├── __init__.py
│   ├── optimizer.py            # CasADi/IPOPT optimizer; canonical CASADI_IPOPT_LOCK
│   ├── unified_planner.py      # 3-Pillar Unified Trajectory Engine
│   ├── frontier_seed.py        # Frontier seed generator
│   └── surrogate_polish.py     # Multi-layer surrogate polishing pipeline
├── models/                     # Drivetrain & robot dynamics
│   ├── __init__.py
│   ├── robot_model.py          # Canonical RobotConfig and physical parameter specs
│   ├── jax_robot_model.py      # JAX-differentiable robot dynamics
│   ├── jax_ramsete.py          # JAX Ramsete tracking controller
│   ├── drivetrain.py           # Drivetrain kinematics & constraints
│   └── drivetrain_physics.py   # Wheel force and slip limit modeling
├── verification/               # Reachability and trajectory validation
│   ├── __init__.py
│   ├── validator.py            # Open-loop plant simulator & validation gate
│   ├── immrax_validator.py     # Closed-loop interval reachability validator
│   └── reachability_engine.py  # CompiledReachabilityEngine & JAX reachability math
├── sysid/                      # System identification & digital twin
│   ├── __init__.py
│   ├── digital_twin.py         # Motor & drivetrain parameter validation
│   ├── fit.py                  # SysId fitting algorithms (from sysid_fit.py)
│   ├── collect.py              # Telemetry collection (from sysid_collect.py)
│   ├── compare.py              # Telemetry vs simulation comparison (from sysid_compare.py)
│   └── sheet_form.py           # Form generation & report formatting
├── deploy/                     # Hub communication & code generation
│   ├── __init__.py
│   ├── hub_deploy.py           # Pybricksdev deployment driver
│   ├── hub_ble.py              # BLE discovery & connection handling
│   └── export.py               # Choreo v3 and Pybricks code generators
├── viz/                        # Visualization & web studio
│   ├── __init__.py
│   ├── plotter.py              # Matplotlib trajectory plotting
│   ├── convergence_plotter.py  # Optimization convergence visualizer
│   ├── animated_plotter.py     # Live animated simulation visualizer
│   ├── fields.py               # FLL game field definitions
│   ├── mat_import.py           # Field mat image importer
│   ├── studio.html             # Web Studio frontend UI
│   ├── studio.js               # Studio client JavaScript
│   ├── studio.css              # Studio UI stylesheet
│   └── report.js               # Studio report formatting script
└── legacy/                     # Archived prototypes & research (isolated)
    ├── multiverse_optimizer.py # Multi-Verse pipeline (archived, no CLI entry point)
    ├── jax_optimizer.py        # JAX refinement used by the archived pipeline
    └── path_planning.py        # Reeds-Shepp helpers used by the archived pipeline
```

External Isolated Trees:
- `pybricks_code/`: MicroPython robot-side firmware (strictly isolated, zero host imports).
- `tests/`: 32 test modules, 657 tests (all importing from `overboard.*`).
- Root developer tools: `run_tests.py`, `testcache.py`, `benchmark.py`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Target Package Skeleton | Create `overboard/` directory tree and subpackages (`core`, `models`, `verification`, `sysid`, `deploy`, `viz`, `legacy`) | M1 | Survey 1, R1 |
| 2 | Core Optimizer & Lock Migration | Move core modules; define canonical `CASADI_IPOPT_LOCK = threading.RLock()` in `optimizer.py` and import in `unified_planner.py`, `surrogate_polish.py` | M1 | Survey 1, R1 |
| 3 | Models Migration | Move robot dynamic models to `overboard/models/` and update internal imports | M1 | Survey 1, R1 |
| 4 | Verification & Reachability Engine | Move `validator.py`, `immrax_validator.py`, extract `CompiledReachabilityEngine` to `overboard/verification/reachability_engine.py` | M1 | Survey 1, R1 |
| 5 | SysId Module Migration & Renaming | Move to `overboard/sysid/`: `digital_twin.py`, `sysid_fit.py` -> `fit.py`, `sysid_collect.py` -> `collect.py`, `sysid_compare.py` -> `compare.py`, `sheet_form.py` | M1 | Survey 1, R1 |
| 6 | Deployment Bridge Migration | Move `hub_deploy.py`, `hub_ble.py`, `export.py` to `overboard/deploy/` | M1 | Survey 1, R1 |
| 7 | Viz & Web Studio Assets Migration | Move plotting modules and web assets (`studio.html`, `studio.js`, `studio.css`, `report.js`) to `overboard/viz/` | M1 | Survey 1, 3, R1 |
| 8 | Package Entrypoints Migration | Create `overboard/__init__.py`, `__main__.py`, move `main.py`, `gui_server.py`, `diagnostics.py` to `overboard/` | M1 | Survey 1, 3, R1 |
| 9 | Path Anchors & Subprocess Updates | Update `REPO_ROOT = Path(__file__).resolve().parent.parent` and subprocess invocations (`python -m overboard.main`, `-m overboard.deploy.hub_deploy`) | M1 | Survey 3, R2 |
| 10 | Legacy Tree Relocation | Move legacy research into `overboard/legacy/` | M1 | Survey 1, R1 |
| 11 | Source Internal Hard Imports | Migrate all imports inside `overboard/` to `overboard.*` (zero relative escape, zero root imports) | M1 | Survey 1, R3 |
| 12 | Test Suite Hard Imports Migration | Update all 35 test files in `tests/` to import from `overboard.*` | M2 | Survey 2, R3 |
| 13 | Test Suite Registry Update | Update `tests/test_suite_registry.py` to discover and validate against `overboard` package | M2 | Survey 2, R3 |
| 14 | Test Legacy Boundary Update | Update `tests/test_legacy_boundary.py` to guard `overboard.legacy` and `overboard/legacy` | M2 | Survey 2, R3 |
| 15 | Test Studio Static Asset Paths Update | Update `test_studio_inventory.py`, `test_studio_honesty.py`, `test_studio_report.py` to point to `overboard/viz/` | M2 | Survey 2, 3, R2 |
| 16 | Test Mocks & Subprocesses Update | Update subprocess assertions and mock patches in `test_cli_ux.py`, `test_gui_server.py`, `test_hub_deploy.py` | M2 | Survey 2, 3, R2 |
| 17 | Zero Root Shims Cleanup | Delete root-level source modules, root `viz/`, root `legacy/`; verify zero forwarding shims remain | M3 | Survey 1, R3 |
| 18 | Firmware Isolation Preservation | Verify `pybricks_code/` remains intact without host imports | M3 | Survey 3, R2 |
| 19 | Test Runner & Cache Parity | Update `run_tests.py` and `testcache.py` to resolve and track `overboard` package modules | M4 | Survey 3, R4 |
| 20 | Benchmark Script Parity | Update `benchmark.py` to invoke `overboard.main` | M4 | Survey 3, R4 |
| 21 | Environment Scripts Parity | Update `install.bat`, `install.sh`, `start_studio.bat`, `start_studio.sh` with `-m overboard.main` preserving line endings | M4 | Survey 3, R4 |
| 22 | Package Configuration & pyproject.toml | Create `pyproject.toml` with `overboard` package discovery and `viz` package-data inclusion | M4 | Survey 3, R4 |
| 23 | Documentation Updates (AGENTS.md) | Update CLI commands and path conventions in `AGENTS.md` | M4 | Survey 3, R4 |
| 24 | Full Test Suite Execution | Execute all 657 tests with 0 failures, 0 errors via `python run_tests.py --clear-cache` | M5 | R4, Acceptance |
| 25 | CLI Diagnostics Verification | Verify `python -m overboard.main --doctor` passes with full dependency check | M5 | R2, Acceptance |
| 26 | End-to-End Solve & Validation Gate | Verify `python -m overboard.main -w examples/example_straight.json -o output.traj --validate` converges and writes output | M5 | Acceptance |
| 27 | Forensic Audit & Integrity Check | Independent verification of zero shims, zero dummy logic, authentic execution | M5 | Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Package Skeleton & Source Migration | Create `overboard/` hierarchy, move modules, establish `CASADI_IPOPT_LOCK`, extract `reachability_engine.py`, set path anchors & subprocesses, migrate all internal source imports | none | IN_PROGRESS |
| M2 | Test Suite Migration & Registry Verification | Update all 35 test files to `overboard.*`, update `test_suite_registry.py`, `test_legacy_boundary.py`, studio asset paths, mocks | M1 | PLANNED |
| M3 | Root Cleaning & Firmware Isolation | Remove root-level source modules and dirs; confirm zero root shims remain; verify `pybricks_code/` isolation | M2 | PLANNED |
| M4 | Runners, Scripts, Tooling & Docs Parity | Update `run_tests.py`, `testcache.py`, `benchmark.py`, `.bat`/`.sh`, `pyproject.toml`, `AGENTS.md` | M3 | PLANNED |
| M5 | Final Verification & Forensic Audit | Run all 657 tests, `--doctor`, end-to-end `--validate`, and forensic integrity audit | M4 | PLANNED |

## Interface Contracts
### `overboard.core.optimizer` ↔ `overboard.core.unified_planner` / `overboard.core.surrogate_polish`
- `CASADI_IPOPT_LOCK = threading.RLock()`: Canonically defined in `overboard.core.optimizer`.
- `CASADI_SOLVE_LOCK = CASADI_IPOPT_LOCK`: Backward-compatible alias retained in `optimizer.py`.
- `from overboard.core.optimizer import CASADI_IPOPT_LOCK`: Imported and used by `unified_planner.py` and `surrogate_polish.py`.

### `overboard.verification.reachability_engine` ↔ `overboard.verification.immrax_validator` / `overboard.core.surrogate_polish` / `overboard.core.unified_planner`
- `CompiledReachabilityEngine`: Defined in `overboard.verification.reachability_engine`.
- Public functions: `unroll_trajectory_headings`, `_extract_state_and_dt_arrays`, `_cos_interval`, `_sin_interval`, `_mul_scalar_iv`, `_mul_iv_iv`.
- Constants: `STATIC_MAX_INTERVALS`, `STATIC_MAX_SEGMENTS`, `MAX_TICKS_PER_INTERVAL`, `GRAD_TICKS_PER_INTERVAL`, `PLANT_SUBSTEPS`, `CONTROL_DT`.
- Imported via: `from overboard.verification.reachability_engine import CompiledReachabilityEngine, ...`

### `overboard.main` / `overboard.gui_server` ↔ Project Root
- `REPO_ROOT = Path(__file__).resolve().parent.parent` (for files directly in `overboard/`).
- `REPO_ROOT = Path(__file__).resolve().parents[2]` (for files in `overboard/deploy/` or `overboard/sysid/`).
- `VIZ_DIR = Path(__file__).resolve().parent / "viz"`.
- Subprocess invocations: `[sys.executable, "-u", "-m", "overboard.main", ...]` and `[sys.executable, "-u", "-m", "overboard.deploy.hub_deploy", ...]`.

## Code Layout
- Target application source: `overboard/`
- Tests: `tests/`
- MicroPython firmware: `pybricks_code/`
- Field assets: `mats/`, `fields/`
- Examples & calibration: `examples/`, `calibration/`, `missions/`
- Root runners & tooling: `run_tests.py`, `testcache.py`, `benchmark.py`, `pyproject.toml`, `requirements.txt`
