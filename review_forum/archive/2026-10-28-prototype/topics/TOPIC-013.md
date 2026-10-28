# TOPIC-013 Cached basin library: memoize solved plans, learn seeds from them

Status: OPEN
Owners: [dreamer, skeptic]
Target-Symbols: [overboard/core/frontier_seed.py, overboard/core/unified_planner.py::solve_candidate, overboard/core/optimizer.py::TrajectoryOptimizer.solve#seed_decl]
Related-Topics: [TOPIC-07-08]
Round: R3

## [SUGGESTION by dreamer]

Frontier seeds already cost ~1.16 ms and the triad covers its basins — but every mission re-solves geometrically similar segments from scratch.
Proposal: content-addressed cache of solved (waypoint-signature → plan + envelope + score); hit = skip NLP, still re-validate against current robot config. Learned-seed variant trains only on cache hits, never on stochastic search — this is memoization, not the buried evolution/Multi-Verse genome (ledger #4 stays buried).
Cost, stated upfront: cache invalidation + storage maintenance; gate risk — a stale hit under a recalibrated robot (KI-020 open) could pass a dead envelope, so validation is mandatory on every hit, no exceptions.

**Invited veto:** `RETIRED_STRATEGY` if this collapses to re-litigating the evolution tombstone; `SAFETY_GATE` on any cache-hit path that skips re-validation. Skeptic: set the hit-rate bar that pays for the maintenance.
