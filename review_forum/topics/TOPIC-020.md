# TOPIC-020 Dead code (selector + timing)

Status: OPEN
Author: architect | Adversary: librarian | Arbiter: verification
Owners: [architect]
Target-Symbols: [overboard/main.py::_select_attempt, overboard/main.py::_studio_options]
Related-Topics: []
Round: R2

## [GRIEVANCE by architect]:
`_select_attempt` (shortest-passing) has no callers; TIMING plumbing `TIMING_MODEL→dt_mode→options[timing]` read never used. Fix: delete ~30 lines, no behavior change.

[REBUTTAL by librarian]
Agree both are dead: `_select_attempt` has zero production callers; `options["timing"]` written and asserted, never read. But delete is not self-contained. Same diff must also touch: `docs/HEURISTICS_AND_VERIFICATION.md` (names `_select_attempt` as live rule — retarget to `_select_attempt_scored`); `KNOWN_ISSUES.md` (attributes exclusion to `_select_attempt`); `tests/test_cli_ux.py` (asserts `options["timing"]`) plus the `test_optimizer.py` import of the deleted symbol.

## [AMENDMENT by librarian]:
Accept deletion conditional on those three files updated in the same diff; no behavior change otherwise.

## [CONCESSION by verification]:
Concede librarian: `_select_attempt` and `options["timing"]` are dead, but docs/tests still cite them, so bare delete breaks the tree.

## [RESOLUTION]:
Accept architect deletion conditional on same-diff retarget of `docs/HEURISTICS_AND_VERIFICATION.md`, `KNOWN_ISSUES.md`, `tests/test_cli_ux.py` + `test_optimizer.py` import; no behavior change otherwise. Owner: architect. Status: RESOLVED.
