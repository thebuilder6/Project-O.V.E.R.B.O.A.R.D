# TOPIC-003 M1 budget + push contention

Status: OPEN
Author: coach | Adversary: architect | Arbiter: robotics
Owners: [coach]
Target-Symbols: [docs/EXAMPLES.md::M1-rule, overboard/gui_server.py::Handler:do_POST, overboard/deploy/hub_deploy.py::push_trajectory]
Related-Topics: [TOPIC-004]
Round: R2

## [GRIEVANCE by coach]:
Kid-drawn missions have no measured budget → validate refuses; 12 kids queue one BLE hub; unvalidated laptop solves but cannot push. Fix: pre-ship 2-3 measured missions, pre-solve T_ref, pre-push approved .traj on USB, demo-push via USB.
**HOLD [VETO: PEDAGOGY_BLOCKER]**

## [REBUTTAL by architect]:
Pre-ship / pre-solve / USB is process, not code: measured box missions + T_ref + approved .traj on stick runs through current do_POST / push_trajectory unchanged — ship a runbook, not a patch.
No auto-budget: any auto-proposed target_time_s violates the M1 rule (budgets measured, never guessed) and turns the duration score into fiction; validate refusing budget-less missions is the gate working.
No queue/gate code: BLE single-connection is physics — a server queue just moves the 12-kid line and adds state to maintain; softening hub_approved/sha to allow unvalidated push trades a pedagogy blocker for a hardware crash.
Delete, don't add: zero lines in gui_server.py / hub_deploy.py for this topic.

## [AMENDMENT by architect]:
Docs-only: pre-measured budgets for 2-3 box missions + pre-solve/USB checklist in EXAMPLES; no CLI/server/deploy change.

## [CONCESSION by robotics]:
Coach's blocker is real — 12 kids, one BLE hub, no budget means no push — but auto-budget and queue code fix the wrong layer: auto target_time_s breaks M1 (budgets measured, never guessed) and a server queue can't fix BLE single-connection physics.

## [RESOLUTION]:
Docs-only runbook: pre-ship 2-3 box missions with measured T_ref/target/max + pre-solve + USB demo-push checklist in docs/EXAMPLES.md; no auto-budget, no queue, zero code lines in gui_server.py / hub_deploy.py. Validate refusing budget-less missions stands.
Owner: coach. Status: RESOLVED.
