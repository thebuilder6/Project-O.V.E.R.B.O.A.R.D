# Round 1 v3 — verification (Gus)

Sound: export gate blocks correctly (`main.py::_export_block_reason`, `_attempt_accepted` excludes non-converged + structural); truncation honest (50mm cut, MAX_ENVELOPE == guard, envelope_truncated+diverged travel to JSON); score cannot launder failure (caps + pass-before-accepted + unscoreable stays None); best-effort fenced (accuracy-only, hub_approved excludes best_effort, push re-checks).

GATE HOLE — persisted artifact drops symbolic anchor. Stdout JSON carries propagation+inclusion per candidate with never-read-envelope-without-model contract, attempt_record holds them — but `_build_validation_record` does not copy into .traj/_stats.json, and `check_trajectory_approval` does not check them. Plant-pass and tick-pass files indistinguishable at push.

Fix: add propagation + inclusion in _build_validation_record; assert propagation in (plant,tick) in check_trajectory_approval. No scoring change.

**HOLD [VETO: GATE_SOUNDNESS]**

Skipped suite re-run, twin soundness (KI-020). Risk: tick approval misread as plant — same verdict, different tube.
