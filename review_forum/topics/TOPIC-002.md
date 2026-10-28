# TOPIC-002 Calibration lab vs 45-min

Status: OPEN
Author: coach | Adversary: physics | Arbiter: student
Owners: [coach]
Target-Symbols: [docs/CALIBRATION_PROCEDURE.md::steps, overboard/sysid/collect.py::TestGuide]
Related-Topics: []
Round: R2

## [GRIEVANCE by coach]:
Full 27-measure Lab + tape sign-convention cannot fit a 45-min / 12-kid / no-wifi period; one bad tape cell poisons fit. Fix: pre-calibrate per robot before class, Express/Anchor tier in lesson, pre-marked jig + scribe + hub-odometry cross-check.
**HOLD [VETO: PEDAGOGY_BLOCKER]**

## [REBUTTAL by physics]:
Express skips tape/weigh-in and plans on nominal ranges, so the force/slip/envelope gates audit a weaker tire and a wrong mass, not the robot on the mat. Cost compounds on tall builds: transfer and v_tip scale with h_cg, and the hardcoded 0.05 m default halves-to-thirds the real lateral transfer at 0.10–0.15 m — the gate passes a plan the hardware lifts on. Pre-calibrate per robot before class: accepted, measurement still happens. Express live in class: accepted ONLY as declared-not-measured (KI-020), `safe` preset + short straight missions, never hub-approved for tall/high-curvature builds. One bad tape cell poisoning the fit is real, but the fix is jig + one scribe + hub-odometry cross-check, not dropping measurement.

## [AMENDMENT by physics]:
Express/Anchor tier permitted with `validation.hub_approved: false` unless a measured profile exists; h_cg becomes a measured RobotConfig property before any tall-build approval.

## [CONCESSION by student]:
Coach's 45-min / 12-kid constraint is real, and physics' safety bound holds — Express plans on nominals, so it cannot carry hub approval, least of all on tall builds.

## [RESOLUTION]:
Pre-calibrate per robot before class; Express/Anchor tier permitted in class ONLY as declared-not-measured (KI-020), `safe` preset + short straights, `validation.hub_approved: false`; tall/high-curvature builds require a measured profile with h_cg; lesson adds pre-marked jig + one scribe + hub-odometry cross-check instead of dropping measurement.
Owner: coach. Status: RESOLVED.
