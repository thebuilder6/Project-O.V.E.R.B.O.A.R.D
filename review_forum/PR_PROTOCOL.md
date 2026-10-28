# PR Protocol — forum ticket to merged code (2026-10-28)

Every merged line passes: ticket → branch → builder diff → triad review (+ open floor) → CI gates → merge → librarian sync.

## Builder vs reviewer matrix (adversarial pairing, no self-approval)

| Domain | Builder | Mandatory Adversary | Arbiter |
|---|---|---|---|
| Physics & drivetrain | Vora | Gould | Gus |
| Controls & NLP math | Gould | Vora | Gus |
| Firmware & Pybricks | Sara | Dan | Alvarez |
| Architecture & core | Dan | Ingrid | Gus |
| Classroom & UI copy | Mia / Alvarez | Dan | Ingrid |
| Research & suite | Abla | Bob | Lindqvist |
| Paper & whitepaper | Lindqvist | Ingrid | Abla |
| Moonshots & speed | Jet | Gus | Dan |

Builder implements; adversary reviews; arbiter signs. Open-floor comments welcome
per PROTOCOL-v3-OPEN; all must be answered or ticketed before `[LGTM]`.

## PR schema (branch `agent/<TICKET>-<short>`)

Header: ticket link (`synthesis/tickets.json::<ID>`), triad names,
Open-Contributors list. Body: symbolic anchors touched, minimal diff summary
(deletions preferred, no new flags unless the resolution demands one),
invariant + regression checks run. Template: `.github/PULL_REQUEST_TEMPLATE.md`.

## Review turns (same tags as forum)

Adversary `[REVIEW]` → builder `[CONCESSION]`/`[AMENDMENT]` → arbiter `[LGTM]`
or `[VETO:*]`. Two rounds max, then needs-measurement. Veto locks per Rule 3.

## CI gates (all must pass before arbiter LGTM)

1. `python run_tests.py --no-cache` (or the ticket-scoped subset + registry +
   inventory pins when the full run is impractical; record what ran).
2. Registry: no silent skips (`test_suite_registry`).
3. Studio inventory parity when markup/labels change (`test_studio_inventory`).
4. Ledger quarantine: `python review_forum/orchestrator.py --check <TICKET>` —
   fails if the diff re-introduces quarantined strategies.
5. Ticket-specific measurement (e.g. solver benchmark before/after for T010,
   7/17/27 seeds for T015, parity proof for T023) where the resolution demands it.

## Merge + sync

Squash-merge to main, message `fix(<topic>): <change> (Ticket <ID>)`.
Librarian updates KNOWN_ISSUES/CHANGELOG in the same wave. Forum status flips
to MERGED; regressions spawn new R1 findings. Failed CI → FAILED_CI, rework.
Vetoed → VETOED, back to forum, never merged around the veto.
