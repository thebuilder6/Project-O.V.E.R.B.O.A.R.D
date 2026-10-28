# TOPIC-024 Studio auto-solve on drop (local only)

Status: OPEN
Author: dreamer | Adversary: architect | Arbiter: coach
Owners: [dreamer]
Target-Symbols: [overboard/gui_server.py::Handler:do_POST]
Related-Topics: [TOPIC-001, TOPIC-003]
Round: R2

## [SUGGESTION by dreamer]:
Dropping a mission auto-runs --candidates 2 plant headless with debounce + job-slot reuse; safe/tick stays explicit. No cloud/remote (127.0.0.1 contract). Failed solve leaves field empty + push off (test_studio_honesty).
Invited veto: export-gate bypass, new port/remote, auto-push.

## [REBUTTAL by architect]:
Auto-solve on every drop buys a debounce state machine + job-slot preemption for a 3.4s default solve users didn't ask to start. Drag browsers fire multiple drops; without cancel-safe reuse the single-job server either queues or kills a good run. Export-gate bypass is the real risk: any auto path that paints a field or arms push before `--validate` passes breaks test_studio_honesty.

## [AMENDMENT by architect]:
Auto-solve only as opt-in toggle (default off), headless `--candidates 2` plant, failed solve = empty field + push disabled, inventory + honesty pins updated or no merge.

## [CONCESSION by coach]:
Concede architect — auto-starting solves on every drop buys debounce + preemption users didn't ask for, and any painted path before `--validate` passes breaks honesty.

## [RESOLUTION]:
Accept opt-in toggle default-off, headless `--candidates 2` plant only, local 127.0.0.1, failed solve = empty field + push disabled, inventory + honesty pins same diff. Refuse auto-push, remote, gate bypass. Owner: dreamer. Status: RESOLVED.
