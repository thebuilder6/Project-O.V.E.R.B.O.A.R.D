# TOPIC-012 Traction end-row gap + epigraph set

Status: OPEN
Author: controls | Adversary: physics | Arbiter: verification
Owners: [controls]
Target-Symbols: [overboard/core/optimizer.py::traction-rows]
Related-Topics: [TOPIC-010]
Round: R2

## [GRIEVANCE by controls]:
Motor bounded both ends (correct); traction start-only, end fmin-priced with shortcut (112→3145 iters MUMPS crash); validator checks both ends — optimizer exploits gap, validator rejects. Epigraph branch swaps in different smaller set, same gate. Fix: enforce trac_end two-half-space (activate where |v·w| transfer >10% static if blowup recurs).

## [REBUTTAL by physics]:
1. The 10%-transfer activation does NOT preserve gate soundness. Straight-line launch/brake binds traction at near-zero lateral transfer (no longitudinal/caster term), so the conditional leaves exactly the binding case unenforced — optimizer still exploits it, validator still rejects.
2. The 10% trip-point inherits unmeasured-h unsoundness (CG_HEIGHT_M=0.05 hardcoded).
3. Epigraph verdict: per-wheel both-end rows ARE the sound gate — approve that set; the plain-branch fmin-priced trac_end is pricing, not a gate, rejected.

## [AMENDMENT by physics]:
Enforce trac_end as two half-spaces unconditionally, mirroring the motor rows; if blowup recurs, sparsify elsewhere, never the traction gate.

## [CONCESSION by verification]:
Concede: 10%-transfer activation unsound — launch/brake binds traction at ~0 transfer, leaving the binding case unenforced; trip inherits unmeasured-h. Epigraph both-end rows are the sound set.

## [RESOLUTION]:
Enforce trac_end as two half-spaces unconditionally, mirroring motor rows; if blowup recurs, sparsify elsewhere, never the traction gate. Owner: controls. Status: RESOLVED.
