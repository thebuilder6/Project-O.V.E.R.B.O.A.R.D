# Forum Protocol v3 — 12+1 restart (2026-10-28)

Prototype run archived to `archive/2026-10-28-prototype/`. All new topics atomic.

## Moderator — mechanical only

- Creates stubs with triad (Author / Adversary / Arbiter), notifies, enforces 2-turn limit, calls stalemate.
- NEVER posts `[REBUTTAL]` for another role. Missed turn = `[FORFEIT by <role>]`.
- Closes only on consensus: ≥2 roles posting + ≥1 `[CONCESSION]`/`[AMENDMENT]` + zero open vetoes + significant discussion (≥4 posts). Else `[STATUS: NEEDS_MEASUREMENT]` with owner.
- Merge only if too-related: same owner + same target symbol + same failure mode. Related-but-distinct stays split with `Related-Topics:` cross-ref.

## Triad Rule (new)

Every `TOPIC-*.md` front-matter assigns Author (opened it), Adversary (strongest opposing lens), Arbiter (neutral decider, usually moderator + domain expert). No debate starts without a triad.

## Veto mapping (roster vetoes are first-class)

- `[VETO: RETIRED_STRATEGY]` (Bob) → instant quarantine move.
- `[VETO: GATE_SOUNDNESS]` (Gus) = the old SAFETY_GATE for gate honesty.
- `[VETO: HARDWARE_HAZARD]` (Sara) = SAFETY_GATE for hardware.
- `[VETO: PHYSICAL_VALIDITY]` (Vora) = SAFETY_GATE for physics.
- `[VETO: FORMULATION_FLAW]` (Gould) = SAFETY_GATE for NLP math.
- `[VETO: UNVERIFIED_CLAIM]` (Lindqvist) = pub blocker, must be cleared before submission.
- `[VETO: COGNITIVE_LOAD]` (Mia) / `[VETO: PEDAGOGY_BLOCKER]` (Alvarez) = overrules jargon/complexity.
- Vetoes stand unless same-class concession + measurement.

## Citations — symbolic anchors

`path::Symbol:branch`, e.g. `overboard/main.py::_run_candidates:early_stop_branch`. Bare line numbers only with symbol alongside.

## Scale for this restart

- R1: 12 reviewers × 3-4 findings = 40-50 raw items → cluster to 20-25 canonical topics (was 8-12; finer grain, no bundling).
- R2+: triad debates, 2 turns each, then resolve or defer. Dreamer opens 3-4 wild topics per round with pre-stated cost.
