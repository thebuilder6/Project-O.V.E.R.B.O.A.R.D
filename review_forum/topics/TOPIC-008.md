# TOPIC-008 Lift aliases as slip

Status: OPEN
Author: physics | Adversary: verification | Arbiter: robotics
Owners: [physics]
Target-Symbols: [overboard/models/drivetrain_physics.py::wheel_normal_forces]
Related-Topics: [TOPIC-007]
Round: R2

## [GRIEVANCE by physics]:
No `nl,nr ≥ 0` clamp; negative inner `nl` → `cof·negative` → slip always fires with wrong magnitude (symbolic clamps min at 0, different number). Report says "slip" for wheel-off-ground; `v_tip` advisory only. Fix: explicit `wheel_lift` flag when `min(nl,nr) ≤ 0`, gate as hard/structural, not score-acceptable slip.

## [REBUTTAL by verification]:
ACCEPT the category change: `wheel_lift` (min(nl,nr)≤0) gates as hard/structural, never score-acceptable slip.
A negative `cof·nl` magnitude is model-exit, not a margin — scoring/accepting it via `overboard/main.py::_attempt_accepted` would launder an invalid check into a hub-approvable number.
Nothing in the fence breaks, but the flag must enter it: `::_attempt_accepted`/`::_attempt_eligible` already exclude `structural_failure`, and `::_choose_attempt` best-effort pools on `accuracy_only_failure` — so lift must set `structural_failure=True`, `score=None`, `accuracy_only_failure=False`, with a `wheel_lift` reason distinct from slip, or `--accept-score`/`--best-effort`/`--pick-candidate` still export it.

## [AMENDMENT by verification]:
Name `wheel_lift` in the `::_attempt_accepted` structural docstring list and fold it into the `hard_failure` expression behind `accuracy_only_failure`; no scoring-weight change.

## [CONCESSION by robotics]:
Firmware impact: none. Structural lift is a host-side validation gate — it blocks export/push, so no lifting trajectory ever reaches the hub; no change to odometry, Ramsete, runner, or event behavior.

## [RESOLUTION]:
Adopt `wheel_lift` when `min(nl,nr) ≤ 0`, gated as structural: `structural_failure=True`, `score=None`, `accuracy_only_failure=False`, distinct from slip; name it in the `::_attempt_accepted` structural docstring and fold into the `hard_failure` expression. No scoring-weight change.
Owner: physics. Status: RESOLVED.
