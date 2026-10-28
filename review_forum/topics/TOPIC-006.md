# TOPIC-006 Overrun hold break + battery/drain

Status: OPEN
Author: robotics | Adversary: controls | Arbiter: verification
Owners: [robotics]
Target-Symbols: [pybricks_code/ramsete.py::follow_trajectory, pybricks_code/robot_config.py::battery]
Related-Topics: []
Round: R2

## [GRIEVANCE by robotics]:
Overrun resets phase → hold stretches past 20 ms with max_dt logged never gated; GC pause mid-mission violates force model silently. Low battery warns but drives; drain window can lose tail lines. Fix: consecutive-overrun counter + loud abort; low-battery pre-run abort with opt-out.

## [REBUTTAL by controls]:
Counter accepted, abort-on-first-overrun rejected: a single 25 ms stretch stays inside the force envelope; abort on 3 consecutive overruns >25 ms + `ABORT,OVERRUN` preserves availability without silent violation.
Wall-clock dt must not enter the transcription: plan/validation stay uniform-20 ms ZOH; firmware clamps the applied dt to [15,30] ms for the Ramsete update and logs raw dt, with run gated on max_dt. Feeding measured dt back breaks collocation validity and voids the envelope bound.
GC guard is cheap and mandatory: one `gc.collect()` pre-mission, auto-GC off + preallocated buffers during loop, re-enable post-run.

## [AMENDMENT by controls]:
Accept consecutive-overrun counter + dt clamp + gate on max_dt; battery/drain abort deferred — out of controls scope.

## [CONCESSION by verification]:
Single-overrun abort rejected — one 25ms stretch stays in envelope; 3× consecutive >25ms + `ABORT,OVERRUN` ends silent violation without availability loss.

## [RESOLUTION]:
Adopt counter (abort on 3 consecutive overruns >25ms) + dt clamp [15,30]ms for Ramsete update, raw dt logged, run gated on max_dt + GC guard (pre-mission collect, auto-GC off in loop). Battery/drain split to follow-up. Owner: robotics. Status: RESOLVED.
