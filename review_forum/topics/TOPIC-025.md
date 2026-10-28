# TOPIC-025 Basin cache (validation-tagged memo)

Status: OPEN
Author: dreamer | Adversary: ledger-guardian | Arbiter: research-method
Owners: [dreamer]
Target-Symbols: [overboard/core/unified_planner.py::solve_candidate, overboard/core/frontier_seed.py]
Related-Topics: [TOPIC-015]
Round: R2

## [SUGGESTION by dreamer]:
Content-addressed memo (waypoint-signature + mesh + dt + weights + gate vector) → replay plan+envelope+score; skip IPOPT on hit, always re-validate; plant-pass never reads as tick. Not evolution (ledger #4 stays buried) — memoization only. State cost + gate key upfront.
Invited veto: RETIRED_STRATEGY if collapses to evolution; SAFETY_GATE on any hit skipping re-validation.

[REBUTTAL by ledger-guardian]
Memo ≠ evolution (ledger §19.4 stays buried): cache replays one solved plan, no mutation/crossover/generations, claims no new basin. No re-litigation provided a hit skips only IPOPT, never re-validation.
Key must include: waypoint signature + mesh/dt/weights + full gate vector (propagation+inclusion, envelope limit+per-metre, field season, scoring budgets) + robot-config/uncertainty hash + code/solver version. Plant-pass never reads as tick; mismatch = miss.

## [AMENDMENT by ledger-guardian]:
Allow content-addressed memo only with that key, mandatory re-validate on every hit; any learning across misses = evolution, vetoed.

## [CONCESSION by research-method]:
Memo is memoization, not evolution, iff a hit replays one solved plan with no mutation/crossover/generations and no learning across misses.

## [RESOLUTION]:
ALLOW content-addressed memo iff key = waypoint-signature + mesh/dt/weights + full gate vector (propagation+inclusion, envelope limit+per-metre, field season, scoring budgets) + robot-config/uncertainty hash + code/solver version; hit skips only IPOPT, always re-validates; plant-pass never reads as tick, mismatch = miss. Any learning across misses = evolution, vetoed. Owner: dreamer. Status: RESOLVED.
