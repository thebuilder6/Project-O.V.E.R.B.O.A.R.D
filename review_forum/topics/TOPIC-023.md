# TOPIC-023 Native hot loop

Status: OPEN
Author: dreamer | Adversary: verification | Arbiter: controls
Owners: [dreamer]
Target-Symbols: [overboard/core/optimizer.py::TrajectoryOptimizer.solve, overboard/verification/immrax_validator.py::PROPAGATION]
Related-Topics: [TOPIC-010, TOPIC-014]
Round: R2

## [SUGGESTION by dreamer]:
Compile per-candidate build + plant/tick rollout (codegen/scan); IPOPT stays; parity pinned by byte-identical .traj + envelope on 32-mission suite. Win is validation-bound (tick/mjac 7-10x), not seeds (1ms).
Invited veto: SAFETY_GATE if native float/Jacobian divergence moves envelope; COGNITIVE_LOAD if compiled step breaks student install.

## [REBUTTAL by verification]:
Parity bar: byte-identical .traj + envelope on 32-mission suite, plant and tick — float "close enough" fails. Natural 627mm vs mjac 18.8mm (~33x) proves the danger: compiled scan on natural extension blows the tube and voids the gate. Must preserve KI-027 coupling + KI-028 20ms plant and mixed-Jacobian indexing per KI-049/050. Any divergence fail-closes to Python path; tags must survive or candidates unscorable.

## [AMENDMENT by verification]:
Opt-in only, mixed-Jacobian only, parity test pinned; natural-extension fast mode rejected outright.

## [CONCESSION by controls]:
Concede opt-in compiled plant/tick rollout, mjac-only; natural-extension fast mode rejected — 627mm vs 18.8mm voids the gate.

## [RESOLUTION]:
Allow codegen/scan with IPOPT unchanged, gated on byte-identical .traj+envelope on 32-mission suite both propagations, KI-027/028/049/050 preserved, divergence fail-closes to Python, tags survive. Owner: dreamer. Status: RESOLVED.
