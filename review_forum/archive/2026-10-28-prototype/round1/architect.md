# Round 1 — architect (Delete-Line Dan)

[GRIEVANCE] `tests/test_suite_registry.py:28-34` — `except ImportError: continue` silently skips any test module with a broken import, reintroducing the exact hole `run_tests.py:266-271` is warned about. Debt/flake risk: a typo'd import in a new `test_*` module reads green because it never loads. Shortest fix: only skip `ModuleNotFoundError` for known-optional deps (`jax`, `immrax`, `yaml`), re-raise the rest (or assert skips against an allowlist).

[GRIEVANCE] `overboard/main.py:273-290` — `_solve_candidates` throttles BLAS by mutating `os.environ` in the parent process with save/restore. Global, thread-unsafe state to control children; an exception between set/restore (outside the `try`) or a second thread leaks it. Perf/debt risk. Shortest fix: delete the save/set/restore block, pass limits via the executor's `initializer`/`initargs` (set env in the worker only).

[QUESTION] `overboard/main.py:327` — `may_run_parallel` gates parallelism on `type(optimizer).__module__ == "overboard.core.optimizer"`. Any subclass, wrapper, or injected test double silently serializes the run with no message — is string identity the intent? Flake/debt risk: behavior changes with an import path. Shortest fix: `isinstance(optimizer, TrajectoryOptimizer)` or an explicit `supports_parallel` attribute, one line.

[SUGGESTION] `overboard_desktop.py:44-55` + `.github/workflows/desktop.yml:70-77` — the state tree (`pybricks_code`, `docs`) is seeded once (`if not is_dir`) and never refreshed, so upgrades ship stale firmware/docs; and frozen `--validate` smoke runs only on Linux, leaving the Windows-only kill path (`gui_server.py:173-174` `taskkill /T /F`) and winrt bundling untested. Flake risk on Windows. Shortest fix: stamp a version file on seed and re-copy on mismatch; add the same `example_straight --validate` smoke on Windows.

Skipped: full `gui_server.py` read past line 1202, studio inventory test, and solver-seed audit (no `random`/seed found in CLI layer; determinism rests on ordered `pool.map` + single-thread BLAS). Risk: finding 2's initializer fix must still cover the sequential fallback path, which inherits the parent env.
