# Round 1 — research-method (Abla)

[GRIEVANCE] `docs/RESEARCH_AGENDA.md:130` + `research/suite.py:239` — all gate-coverage claims (Q13 5-6/32, Q14 10/32, Q17 6/32, Q15 6/32) reuse one suite `suite.py --count 32 --seed 7` (N=1 suite, no cross-seed replicate). Method hole: single-cluster draw, no variance on the headline rate. Minimal fix: re-run `suite.py` with seeds 7,17,27 + `harness.py` same flags, report median/range of pass rate instead of 6/32.

[QUESTION] `docs/research/RQ-A_tracking_error.md:51-58` — short-answer table is N=5 hand-picked examples (S-curve/stops carry the claim, 82% attribution on one mission). Method hole: cherry-pick + N=1-mission generalization for F4/F6 density sweeps. Minimal fix: compute the same `plan-vs-profile deviation vs tracking` pair via `research/harness.py` on the full 32-mission suite, report median Spearman, not 5 rows.

[GRIEVANCE] `AGENTS.md:59` vs `KNOWN_ISSUES.md:158` — testcache quotes disagree (292s vs 251s cold, both →5s warm) with no tree hash, machine, or N. Method hole: N=1 wall-time, unpinned baseline. Minimal fix: quote `run_tests.py --no-cache` median of 3 + warm median with `harness.tree_state()`-style hash, delete the other number.

[SUGGESTION] `benchmark.py:104-147` vs `research/harness.py:65,219` — `benchmark.py --mode cli/pipeline/random` reports `avg/min/max_wall_time` over `runs=3` but records no `provenance{}`, `propagation`, or sample-layout (`arc-length` vs uniform, `waypoint_sample_indices`). Method hole: missing layout/provenance, so benchmark numbers are not comparable to harness Q-ledger. Minimal fix: copy `harness.provenance()` + `propagation` + `samples` layout into benchmark JSON.

Skipped: full `pq_b/FINDINGS.md` re-read and oracle Q25 traces. Risk: seed-replication may move 6/32 by ±few missions, weakening tail/coverage nulls.
