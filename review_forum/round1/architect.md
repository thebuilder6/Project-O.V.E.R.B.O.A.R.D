# Round 1 v3 — architect (Dan)

F1 dead selector `main.py::_select_attempt` no callers; `_choose_attempt` uses scored only. Delete 23 lines.
F2 dead timing plumbing TIMING_MODEL→dt_mode→_studio_options→options[timing] read never used; single engine per-segment always. Delete param+key+local.
F3 triplicated job runner start_job/start_collect/start_push each Popen+log+flip; _read_body-before-busy-check x4. One _run_subprocess + one read-then-busy helper deletes ~60 lines.
F4 REPO_ROOT=ASSET_ROOT wrong when frozen; user-relative resolves should be STATE_ROOT. Works in checkout, breaks frozen. Delete alias or repoint 3 sites.

Debt/flake: registry guard right (keep); cache excluded from verdict (CI --no-cache); flake surface Win/Linux process (retry, killpg/taskkill, evict MAX_JOBS=20); desktop frozen validate Linux-only, Windows path untested.

Shortest diff ~30 lines, zero behavior. Next: unify run() closures. Win+Linux static only. Skipped viz/installer full read, no test run. Risk: F4 needs frozen smoke before deleting alias.
