# TOPIC-017 Benchmark + harness provenance gaps

Status: OPEN
Author: research-method | Adversary: architect | Arbiter: academic-lead
Owners: [research-method]
Target-Symbols: [benchmark.py::_save_results, research/harness.py::tree_state]
Related-Topics: []
Round: R2

## [GRIEVANCE by research-method]:
Benchmark means over runs=3 with no provenance/propagation/layout; harness hashes 5 files, misses suite/seeds/self. Fix: copy provenance+tree+propagation+samples into every benchmark JSON.

## [REBUTTAL by architect]:
Copying provenance+tree+propagation+samples into every benchmark JSON is duplication: two writers, guaranteed drift. Benchmark timing vs validation ledger are separate concerns. Deletion-shaped: single owner — harness tree_state (or validation JSON) writes provenance once, benchmark stores pointer (tree hash + suite seed + run id).

## [AMENDMENT by architect]:
Concede pointer only: `_save_results` records tree_hash, suite_seed, propagation passthrough referencing harness output — no second provenance builder, no per-sample copy.

## [CONCESSION by academic-lead]:
Method's gap is real: means over runs=3 without seeds/propagation/layout are irreproducible, and a 5-file hash omitting suite/seeds/self is incomplete provenance.

## [RESOLUTION]:
Pointer-only. `_save_results` records tree_hash + suite_seed + run_id + propagation passthrough; no second provenance builder, no per-sample copy. Harness `tree_state` stays sole provenance owner and must add suite/seeds/self. Owner: research-method. Status: RESOLVED.
