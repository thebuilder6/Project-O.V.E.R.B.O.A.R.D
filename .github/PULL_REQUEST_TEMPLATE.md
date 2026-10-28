# PR — [TICKET] short title

**Ticket:** `review_forum/synthesis/tickets.json::<ID>`
**Triad:** Author `<role>` · Lead-Adversary `<role>` · Arbiter `<role>`
**Open-Contributors:** <!-- append any agent/human who commented -->

## Symbolic anchors touched
- `path::Symbol`

## Minimal diff summary
<!-- deletions preferred; no new flags unless the resolution demands one -->

## Invariant + regression checks
- [ ] `python review_forum/orchestrator.py --check <ID>` (ledger quarantine)
- [ ] scoped tests recorded here (full `run_tests.py --no-cache` when practical)
- [ ] ticket-specific measurement (benchmark / seeds / parity) where demanded
- [ ] docs/tests updated in the same diff (inventory, KNOWN_ISSUES, CHANGELOG)

## Review turns
<!-- Adversary [REVIEW] → builder [CONCESSION]/[AMENDMENT] → arbiter [LGTM] or [VETO:*].
     Open-floor comments welcome: typed tag + anchor, zero +1 tolerance.
     No LGTM with open vetoes or unanswered grievances. -->
