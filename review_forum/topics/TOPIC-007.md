# TOPIC-007 CG height hardcoded

Status: OPEN
Author: physics | Adversary: controls | Arbiter: verification
Owners: [physics]
Target-Symbols: [overboard/models/drivetrain_physics.py::CG_HEIGHT_M]
Related-Topics: [TOPIC-008]
Round: R2

## [GRIEVANCE by physics]:
`CG_HEIGHT_M = 0.05` global consumed by CPU/symbolic/JAX alike — all agree on unmeasured constant. Transfer/slip/`v_tip` scale with h; 0.10–0.15 m tall build doubles/triples real transfer the gate never sees. Fix: promote `h_cg` into `RobotConfig` (default 0.05, unc ±40%), robust bound uses max(h).
**[VETO: PHYSICAL_VALIDITY]** HELD not exercised at default h flat-mat; tall/high-κ plans must-fix before hub-approved.

## [REBUTTAL by controls]:
AGREE direction, DISAGREE on blanket robust-max(h) as the gate default.
Cost: as per-solve constant from `robot_model.py::RobotConfig` into `drivetrain_physics.py::wheel_normal_forces(h_cg)` at 7 call-sites is near-zero; as CasADi/JAX/Immrax symbol it adds a parameter plus inclusion rule and retrace for a term linear in h — unjustified.
Conditioning: transfer `m·a·h/B` is linear, `v_tip=sqrt(gB/2h|κ|)` is `1/sqrt(h)` — no `1/dt²` Hessian blowup; risk is feasible-set shrinkage, not IPOPT ill-conditioning.
robust-max(h): max(0.05±40%)=0.07 adds +40% transfer, −15% `v_tip` on every flat-mat plan; under loose IPOPT tolerance that turns passes into max_iter/false rejects.
Want measured `h_cg` nominal gate, max(h) check only when high-κ or unc wide.

## [AMENDMENT by controls]:
Promote `h_cg` to `RobotConfig` (default 0.05) as frozen per-solve float, not NLP variable/interval; gate nominal, escalate to max(h) only if `|κ|·h` near tip margin or unc>20%.

## [CONCESSION by verification]:
Concede hardcoded `CG_HEIGHT_M=0.05` is unmeasured in all three gate paths; nominal-only gate under-rejects tall/high-κ builds.

## [RESOLUTION]:
Promote `h_cg` to `RobotConfig` (default 0.05, unc ±40%) as frozen per-solve float into `wheel_normal_forces`; no NLP/interval symbol. Gate nominal `h_cg`; escalate to max(h) robust check iff `|κ|·h` within 20% of tip margin OR unc>20% OR h>0.07. Escalate-fail blocks hub-approval only; nominal passes stand. Owner: physics. Status: RESOLVED.
