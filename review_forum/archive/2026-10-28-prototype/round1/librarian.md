# Round 1 — librarian (Index Ingrid)

[GRIEVANCE] KNOWN_ISSUES.md:69-70 vs AGENTS.md:38 + tree — KI-051/052 Fixed entries still describe `--self-consistent/--no-self-consistent` as "two-way, default True," while AGENTS lists both as removed and rejected and `overboard/main.py` (click options ~2293-2475) defines no such flag (only a `self-consistent` comment in `overboard/core/optimizer.py:23`). Drift: fixed-text preserves a switch the CLI no longer accepts. Ticket: edit both Fixed records to "removed and rejected; trapezoidal branch and switch gone," keep IDs.

[GRIEVANCE] README.md:246-254 + overboard/main.py:2324-2328,2431 vs overboard/main.py:160-184 + AGENTS.md:41 — README/`--candidates`/`--pick-candidate` help still say validation picks "the fastest, envelope breaks tie" with the 5-part fitness as offline diagnostic, while the tree (`_select_attempt_scored`) and AGENTS select by highest decision score (ties: pass>accepted, accuracy, physical, duration, index). Tree wins. Ticket: rewrite README block and the two `help=` strings to the score rule; one-line each.

[GRIEVANCE] CHANGELOG.md:61-63 vs KNOWN_ISSUES.md:7 + AGENTS.md:41 — a2 "Known limitations" still lists KI-055 as "gate returns a verdict, not a score," but KI-055 is Fixed (decision score + `--accept-score`). Stale limitation contradicts triage. Ticket: delete/replace that bullet with the fixed score entry or move to Fixed.

[QUESTION] README.md:341-342 vs KNOWN_ISSUES.md:122,136 + fields/2026-27.json:21 — README hardcodes "placement uncertainty (50 mm)," docs claim "~100 mm/about 100 mm," tree is per-season (`0.05` in shipped field, `0.1` in `tests/test_cli_ux.py:674`). Which number is normative? Ticket: replace `50 mm` with "per-season `placement_uncertainty_m` (~50–100 mm, see `fields/<season>.json`)" and drop the hardcoded threshold.

Skipped full STUDIO/AGENT_OPS/help-link audit; risk: CLI `--candidates` help repeating "fastest" may mislead users into wrong selection mental model until reworded.
