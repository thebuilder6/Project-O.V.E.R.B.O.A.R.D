# Round 1 v3 — librarian (Index Ingrid)

1. KI-051/052 Fixed text describes removed switch. `KNOWN_ISSUES.md::KI-051/052` "two-way --self-consistent"; tree: no flag, `TestDeprecatedCliFlags` pins rejected; AGENTS/HEURISTICS/AGENT_OPS say removed. Also stale in RQ-A + research/README. Ticket: rewrite to removed/rejected + sweep.
2. KI-020 Files + help strings use pre-overboard names. KI-020 lists sysid_fit.py etc; tree has `overboard/sysid/fit.py`; live strings print old names (fit/compare/sheet_form/hub_deploy/diagnostics/sysid.js/studio.js). Ticket: update to `python -m overboard.sysid.*`.
3. CHANGELOG a2 limits contradict fixed gates. Says KI-055 verdict-not-score (Fixed: score_version:1 + accept-score); KI-054 text omits --field-clearance fix. Ticket: Unreleased entry.
4. KI-059-068 triage stale: KI-062 shipped. Desktop (overboard.spec, installer.iss, desktop.yml, help.html, paths ASSET/STATE_ROOT) + CHANGELOG a2 + AGENTS document it. Ticket: close KI-062 Fixed, refresh counts; KI-059 /sysid stays Open guided-view.

Doc ticket paste-ready in full audit. Skipped full docs prose. Risk: KI-062 close needs maintainer confirm.
