# TOPIC-005 Accel-limit backstop removed

Status: OPEN
Author: robotics | Adversary: physics | Arbiter: verification
Owners: [robotics]
Target-Symbols: [pybricks_code/robot_config.py::limits]
Related-Topics: [TOPIC-004]
Round: R2

## [GRIEVANCE by robotics]:
DriveBase accel raised to 80% of 20k deg/s² ceiling so ~6 m/s² plans pass — but bad-SysId/unvalidated plans now execute instead of clamping at stock ~733 mm/s². With stale config (T004) this is full-torque drive into models. Fix: keep raise + pre-run battery/geometry sanity; no silent clamp removal.
**HOLD [VETO: HARDWARE_HAZARD]**

## [REBUTTAL by physics]:
No — raised limit + stale config does not change the h_cg/tip analysis. Tip margin `v_tip=sqrt(gB/2h|κ|)` and transfer `m·h·v·ω/B` are lateral/curvature terms; a longitudinal clamp never bounded them, so stock ~733 mm/s² was never tip protection, only a straight-line truncate. Stale config is a torque-scale error falsifying gated numbers at ANY clamp height. Firmware must execute the validated plan verbatim; a silent clamp is an unmodeled nonlinearity, not a backstop.
Accept: keep the 80% raise + pre-run sanity — fail-loud config-sha match (per W1-T004, no silent sync) and a battery under-load abort threshold, not advisory. No clamp restoration.

## [AMENDMENT by physics]:
Keep raise; add pre-run assert: config-sha == trajectory record AND Vbatt-loaded above threshold, else refuse drive.

## [CONCESSION by verification]:
Stock ~733 mm/s² clamp was never tip protection — tip/transfer are lateral/curvature terms, and stale config falsifies gated numbers at any clamp height. Silent clamp is an unmodeled nonlinearity, not a backstop.

## [RESOLUTION]:
Accept physics AMENDMENT, keep 80% raise, no clamp restoration. Owner: robotics. Ticket: pre-run assert in firmware — config-sha == trajectory record AND Vbatt-loaded above threshold, else refuse drive. Status: RESOLVED.
