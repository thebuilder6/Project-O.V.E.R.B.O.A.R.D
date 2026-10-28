# TOPIC-001 Studio kid-strings

Status: OPEN
Author: student | Adversary: architect | Arbiter: coach
Owners: [student]
Target-Symbols: [overboard/viz/studio.html::#opt-propagation, overboard/viz/studio.html::#opt-accept-score, install.bat::VENV_PY, overboard/diagnostics.py::run_doctor]
Related-Topics: []
Round: R2

## [GRIEVANCE by student]:
Propagation dropdown ("plant" vs "tick"), accept-score ("deliberately accept a risky plan"), venv wording, doctor port/sysid_fit naming — all fail a 14yo first-timer. Fix: kid-words visible strings, engineering terms in tooltips; values/ids frozen; inventory doc updated same diff.
**HOLD [VETO: COGNITIVE_LOAD]**

## [REBUTTAL by architect]:
Safe: visible-label-only renames on `#opt-propagation`, `#opt-accept-score`, install.bat echoes, `run_doctor` messages. Option values `plant`/`tick`, element ids, `VENV_PY`, JSON keys frozen — zero behavior change.
Cheap only if engineering term survives in `title`/tooltip, so studio words don't fork from validation report + machine-line vocabulary.
Cost: studio inventory pins markup; any added hint/sub-text node must update both same diff — so prefer in-place retitle, no new elements.
Refuse: hiding `port 8000` / sysid_fit status or making accept-score blank-default — doctor is operator surface, silent diagnostics lose failures, and blank changes eligibility semantics.

## [AMENDMENT by architect]:
Accept label-only kid-words with ids/values frozen + inventory same diff; reject structural or hiding changes.

## [CONCESSION by coach]:
Student's kid-words case stands for visible strings; architect's freeze keeps it zero-behavior.

## [RESOLUTION]:
Accept label-only renames on #opt-propagation, #opt-accept-score, install.bat echoes, run_doctor messages — ids/values/keys frozen, engineering terms kept in title/tooltip, inventory updated same diff, no new elements. Refuse hiding port 8000/sysid_fit status or blank-default accept-score.
Owner: student. Status: RESOLVED.
