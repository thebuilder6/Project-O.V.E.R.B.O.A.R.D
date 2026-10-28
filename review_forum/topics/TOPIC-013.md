# TOPIC-013 Ramsete singularity + epsilon drift + dead merge

Status: OPEN
Author: controls | Adversary: robotics | Arbiter: physics
Owners: [controls]
Target-Symbols: [overboard/models/drivetrain_physics.py::ramsete-gains, overboard/core/optimizer.py::merge_conditions]
Related-Topics: [TOPIC-010]
Round: R2

## [GRIEVANCE by controls]:
Gain k1→0 at pinned stops (authority loss where gate tightest); floors disagree (+1e-12 vs +1e-16 vs sinc/tanh/fmax) inside tick Jacobian; agree_velocity/heading computed never consumed; continuous force only when per_segment_dt, default uniform zero enforcement with jerk weight 0. Fix: unify floors, consume merge or delete, gate stops explicitly.

## [REBUTTAL by robotics]:
- Dead merge is the hub hazard: unconsumed agree_* + force continuity only under per_segment_dt means the hub sees a velocity/force step via drive(), with the raised DriveBase limits as the only backstop — real slip/torque event, not numerics. Fix first.
- k1→0 at pinned stops has no firmware crash but blinds the gate where it matters: zero authority at the stop means the tick tube understates hold error and a bad stop still passes. Fix second: gate stops explicitly.
- Epsilon floors never reach MicroPython — host-side tick Jacobian only. Causes plant/tick disagreement, not hub motion. Fix last: one shared floor constant.

## [AMENDMENT by robotics]:
Concede dead-merge fix + explicit stop gate as hub-safety; defer floor unification to host-only cleanup, no push block.

## [CONCESSION by physics]:
Concede robotics ordering: dead-merge/force step reaches hub via drive(); k1→0 blinds tick hold-error at stops; epsilon floors stay host-side in tick Jacobian only.

## [RESOLUTION]:
1. Fix dead-merge first — consume or delete agree_velocity/heading, enforce force continuity independent of per_segment_dt. Owner: controls.
2. Gate stops explicitly second — restore k1 authority / explicit hold check at pinned stops so bad stops fail. Owner: controls.
3. Unify epsilon floors last, host-only cleanup — one shared floor constant, no push block. Owner: controls. Status: RESOLVED.
