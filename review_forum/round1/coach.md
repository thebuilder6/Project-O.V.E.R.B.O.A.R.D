# Round 1 v3 — coach (Mr. Alvarez)

**Finding 1 — Full calibration does not fit one period**
Doc budgets "about half an hour" for one robot + one operator: 21 tests ×3 repeats, 27 tape poses + 3 weigh-ins. With 12 kids sharing one runway, setup eats period. Fix: pre-calibrate once per robot before class, or use Express/Anchor tier; treat shipped ranges as declared-not-measured (KI-020) — `safe` preset + short missions only.

**Finding 2 — Tape convention is sign-error trap**
`x_mm` forward / `y_mm` left / `turn_right_deg` right-positive from centre-between-wheels, tool flips sign for planner. Kids measure from bumper / swap signs → heading_scale failure or silent drift. Fix: teacher demo + jig, one scribe per team, cross-check vs hub odometry. If in doubt skip tape — Express needs none.

**Finding 3 — M1 measured-budget rule breaks un-prepped missions**
Shipped missions carry `scoring.target_time_s = T_ref`; kid-drawn mission has no budget → `--validate` refuses; without validate only candidate-1 solves and push refused. Fix: pre-ship 2-3 missions with budgets measured, lock lesson to box, pre-solve once.

**Finding 4 — Push gate + hub contention stops finale**
Push allows only validated-pass or score-accepted + `hub_approved` + sha match; BLE single-connection; unvalidated laptop solves but cannot push. Fix: pre-install optional stack + pre-push approved `.traj` per mission on USB; demo-push via USB; skip mat download.

**[VETO: PEDAGOGY_BLOCKER]** — do not run full 27-measurement Lab live in 45-min / 12-kid / no-wifi period.

Skipped: fit.py/hub_deploy code audit. Risk: uncalibrated ranges fail long/curved paths even when solve looks fine — short straights only.
