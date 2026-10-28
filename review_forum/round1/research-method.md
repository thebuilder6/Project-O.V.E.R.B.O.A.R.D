# Round 1 v3 — research-method (Abla)

**F1 — All coverage N=1 suite seed 7:** `research/suite.py::suite --seed 7`; measurements README + AGENDA reuse `--count 32 --seed 7` for Q13/Q17/Q15. Headline 6/32 has no variance.

**F2 — RQ-A N=5 cherry-picked + stale:** `docs/research/RQ-A_tracking_error.md::table` 5 examples; F6/F4 claims from S-curve/stops; own caveat results predate KI-025/027/028; 32-mission re-run only on shipping planner.

**F3 — Benchmark means without provenance/layout:** `benchmark.py::_save_results` avg/min/max over runs=3; bench_solver N=1/mission (example_complete 670s dominates); no provenance{}, propagation, or sample layout. Not comparable to harness ledger.

**F4 — Harness provenance partial:** `research/harness.py::tree_state` hashes 5 files only; misses suite/seeds/robots/harness itself, suite seed, layout. Silent generator change invalidates comparison.

Fix: pre-registered no-re-tune `suite.py --count 32 --seed {7,17,27}` + same harness flags; report median/range + Spearman per seed; copy provenance+tree+propagation+samples into every benchmark JSON. Pin OVERBOARD_TREE.

Skipped re-runs (read-only). Risk: cross-seed spread may flip tail/tracking nulls if suite-7 is easy.
