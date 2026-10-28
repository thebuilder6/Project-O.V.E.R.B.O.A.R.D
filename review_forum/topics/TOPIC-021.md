# TOPIC-021 Runner triplication + frozen root

Status: OPEN
Author: architect | Adversary: robotics | Arbiter: verification
Owners: [architect]
Target-Symbols: [overboard/gui_server.py::start_job, overboard/paths.py::REPO_ROOT]
Related-Topics: []
Round: R2

## [GRIEVANCE by architect]:
Three inline Popen+log runners + 4x read-before-busy-check; REPO_ROOT alias points at bundle root, breaks frozen semantics. Fix: one _run_subprocess helper (~-60 lines); repoint user-relative resolves to STATE_ROOT (needs frozen smoke first).

## [REBUTTAL by robotics]:
Unifying the three runners hides real differences: start_push holds the hub's single BLE connection and scan is refused while a job runs — a shared helper that reorders the read-before-busy check or merges run() closures risks a scan/push racing the connection. Collect owns session-format streaming, push owns the approval gate; one helper must not flatten error paths. Same for REPO_ROOT: ASSET_ROOT is read-only bundle, STATE_ROOT owns seeded config — blind repoint risks stale/frozen-mismatched hub config. Do runners after a Linux+Windows frozen push test, not before.

## [AMENDMENT by robotics]:
Accept only exact-duplicate Popen+log extraction with per-runner busy/lock order pinned by test; defer REPO_ROOT repoint until frozen smoke passes.

## [CONCESSION by verification]:
Accept robotics limit: full unification risks scan/push racing the single BLE connection and flattening collect/push error paths.

## [RESOLUTION]:
Extract exact-duplicate Popen+log helper only with per-runner busy/lock order pinned by test; defer REPO_ROOT repoint until Linux+Windows frozen smoke passes. Owner: architect. Status: RESOLVED.
