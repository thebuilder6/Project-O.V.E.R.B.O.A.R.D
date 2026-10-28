# Round 1 v3 — physics (Vora)

Anchors checked: force split m·a=fl+fr, I·α=(fr−fl)B/2 correct; lateral nl,nr=m·g/2∓m·h·v·ω/B sign correct; v_tip=√(gB/2h|κ|) correct; layout N=sum+1, indices cumsum, sample_to_segment repeat, >20° boost — right call.

1. CG height hardcoded global not robot property. CG_HEIGHT_M=0.05 consumed by CPU/symbolic/JAX alike — all agree on unmeasured constant. Transfer/slip/v_tip scale with h; 0.10-0.15m doubles/triples real transfer gate never sees.
2. Wheel-lift aliases as slip. No nl,nr≥0 clamp; negative inner nl → cof·negative → slip always fires; symbolic clamps min at 0 with different magnitude; report says slip for wheel-off-ground. v_tip advisory only.
3. Robust-Cof in 1 of 3 places. Symbolic uses min(cof/track) ranges; CPU + JAX nominal. Optimizer plans weaker tire than audit checks — fail-open.
4. Longitudinal/caster ignored. No front/rear axle true left-right, but skid/caster unloads drive wheels under accel/brake; total = cof·m·g sums to m·g so slip gate non-conservative on launch/brake. total_traction_limit redundant.

Fix: promote h_cg into RobotConfig (default 0.05, unc ±40%), thread through callers, robust bound uses max(h), explicit wheel_lift flag when min(nl,nr)≤0. Touch: drivetrain_physics sig, robot_model x3 + symbolic, jax x2, validator branch.

**[VETO: PHYSICAL_VALIDITY]** HELD not exercised at default h flat-mat; tall/high-κ plans unmeasured — must-fix before hub-approved.

Skipped interval soundness, firmware parity, suite run. Risk: slip gate passes plans tall robot would lift on.
