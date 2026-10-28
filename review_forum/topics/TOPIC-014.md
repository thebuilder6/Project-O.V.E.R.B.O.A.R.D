# TOPIC-014 Gate-hole: propagation/inclusion not persisted

Status: OPEN
Author: verification | Adversary: architect | Arbiter: robotics
Owners: [verification]
Target-Symbols: [overboard/main.py::_build_validation_record, overboard/deploy/hub_deploy.py::check_trajectory_approval]
Related-Topics: [TOPIC-004]
Round: R2

## [GRIEVANCE by verification]:
Stdout JSON carries propagation+inclusion per candidate with never-read-envelope-without-model contract, but `_build_validation_record` drops them — plant-pass and tick-pass files indistinguishable at push. Fix: copy both fields; assert in `check_trajectory_approval`. No scoring change.
**HOLD [VETO: GATE_SOUNDNESS]**

## [REBUTTAL by architect]:
Concede hole: `main.py::_build_validation_record` drops `propagation`/`inclusion` that `main.py::attempt_record` + `_emit_validation_report` already carry — 2-line `.get` copy, no scoring change.
Deletion-shaped: yes, smallest diff; no helper/abstraction, no seed/planner touch.
Schema risk: additive keys only — `_stats.json`/`.traj` readers ignore unknown keys; studio chips read stdout report, not file.
Caution: strict `assert propagation in (plant,tick)` in `check_trajectory_approval` rejects pre-fix approved files (missing key) — schema break, needs grandfather or score-version bump.

## [AMENDMENT by architect]:
Copy as `(attempt or {}).get("propagation"/"inclusion")`; assert `validation.get("propagation", "plant") in ("plant","tick")` so legacy plant files still push, new tick files labelled.

## [CONCESSION by robotics]:
Concede to verification: a tick file mislabeled as plant is a hub-safety hole — tick's closed-loop tube contracts, so filed as plant it claims an open-loop guarantee it never met and could push a path the plant gate would have failed. Strict assert without grandfather trades that hole for a schema break, rejecting pre-fix plant-approved files at push.

## [RESOLUTION]:
Adopt architect amendment: 2-line `.get` copy of propagation/inclusion in `_build_validation_record`; `check_trajectory_approval` asserts `validation.get("propagation","plant") in ("plant","tick")`. Owner: verification. Status: RESOLVED.
