# TOPIC-009 Robust-Cof + longitudinal/caster

Status: OPEN
Author: physics | Adversary: controls | Arbiter: verification
Owners: [physics]
Target-Symbols: [overboard/models/robot_model.py::get_traction_limit_symbolic, overboard/models/drivetrain_physics.py::total_traction_limit]
Related-Topics: [TOPIC-007, TOPIC-008]
Round: R2

## [GRIEVANCE by physics]:
Symbolic bound uses min(cof/track) ranges; CPU + JAX check nominal — audit plans weaker tire than optimizer. Skid/caster unloads drive wheels under accel/brake; per-wheel sum still totals m·g so gate non-conservative on launch/brake. total_traction_limit redundant. Fix: nominal everywhere + robust check at gate; model caster unload or restrict domain.

## [REBUTTAL by controls]:
Robust Cof belongs at the gate, not in the NLP: min(cof/track) inside the symbolic bound solves a worst-case problem with a nonsmooth min the L-BFGS backend already struggles with — conservative, slow, and the reason CPU/JAX nominal disagree with symbolic. Nominal-in-NLP (smooth, solvable) + robust-interval check at the validation gate is fail-closed where it matters and costs zero solver iterations. Caster/longitudinal unload needs unmeasured params no SysId test identifies — an unidentifiable model in the hot path buys false precision; cheaper to derate Cof at the gate or cap launch accel. Concede `total_traction_limit` is redundant — delete it, keep one per-wheel sum.

## [AMENDMENT by controls]:
Nominal Cof/track in optimizer + symbolic; robust min-range bound enforced only in validator gate; remove `total_traction_limit`; caster handled by gate derate, not a new model.

## [CONCESSION by verification]:
Gate audit is weaker than NLP tire — conceded; robust min() has no place in the smooth symbolic bound.

## [RESOLUTION]:
Nominal Cof/track in NLP + symbolic everywhere; robust min-range bound enforced only at validator gate. Delete `total_traction_limit`, keep per-wheel sum. Caster/longitudinal unload handled by gate Cof derate, not a new unidentifiable model. Owner: physics. Status: RESOLVED.
