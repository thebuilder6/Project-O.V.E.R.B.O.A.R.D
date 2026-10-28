# TOPIC-011 Transcription vs ZOH execution

Status: OPEN
Author: controls | Adversary: physics | Arbiter: verification
Owners: [controls]
Target-Symbols: [overboard/core/optimizer.py::integrate, overboard/verification/reachability_engine.py, overboard/models/jax_ramsete.py]
Related-Topics: [TOPIC-010]
Round: R2

## [GRIEVANCE by controls]:
Optimizer GL4 continuous linear-v/w exact for polynomial model; robot/validator runs 20 ms ZOH staircase + quantize + half-tick lead. ZOH error priced only under cost_metric discretization; default jerk + accuracy_weight=0 sets both to zero — time-optimal solve blind to 5 mm gate; turn-boost patches downstream. Fix: price ZOH error in default path or document certified domain.
**[VETO: FORMULATION_FLAW]**

## [REBUTTAL by physics]:
ZOH-blindness is physical, not numerical: the 20 ms hold + quantize + half-tick lead is what the motors actually do; GL4-continuous is the fiction. Error is boundable physics (~dt² × curvature rate), largest exactly where time-optimal solutions push hardest — high-v, high-ω turns.
Turn-boost therefore masks, not covers: downstream margin without pricing the source lets the optimizer keep exploiting the gap the validator then rejects.
Concede the grievance direction but narrow the fix: certified-domain alternative — publish the (v, ω, curvature) envelope where ZOH error stays < fraction of the 5 mm gate, gate/prune plans outside it, price only the residual in the default path.

## [AMENDMENT by physics]:
Accept as physical omission; fix = certified ZOH-error domain + residual pricing in default jerk path, not discretization-metric-only.

## [CONCESSION by verification]:
Concede controls — default jerk path prices zero ZOH error while validator gates the 20 ms ZOH staircase + quantize + half-tick lead, so the time-optimal solve exploits a gap validation then rejects. Turn-boost masks, not covers.

## [RESOLUTION]:
Publish certified (v, ω, curvature) domain where ZOH error stays < fraction of the 5 mm gate, gate/prune plans outside it, price only the residual in the default jerk path — not discretization-metric-only. Owner: controls. Status: RESOLVED.
