# TOPIC-010 KKT acceptable-not-optimal

Status: OPEN
Author: controls | Adversary: verification | Arbiter: physics
Owners: [controls]
Target-Symbols: [overboard/core/optimizer.py::IPOPT-opts]
Related-Topics: [TOPIC-011, TOPIC-012]
Round: R2

## [GRIEVANCE by controls]:
`tol=1e-2, acceptable_tol=1e-1 x5 iters`, L-BFGS + adaptive mu; `Solved_To_Acceptable` counts converged. 1 cm / 0.01 rad residual accepted — half the 20 mm gate consumed pre-validation.
**[VETO: FORMULATION_FLAW]** — loose barrier + single-dt + missing end-traction = feasible set exceeds gate.
Fix in budget: `acceptable_tol=tol`, enforce `trac_end`, `dt` lower bound 0.02. Re-measure solver benchmark.

## [REBUTTAL by verification]:
Tightening `optimizer.py::IPOPT-opts` (`acceptable_tol=tol` + `trac_end` + `dt>=0.02`) moves failures from false-pass to honest `max_iter`/`infeasible` via `optimizer.py::classify_solver_error` — from the gate that direction is safe: `main.py::_export_block_reason` + `_attempt_accepted` already exclude non-converged, so soundness holds or improves.
Risk as stated in `round1/controls.md:12` is availability, not soundness: on long missions tighter barrier can convert `Solved_To_Acceptable` into `max_iter` with zero gate-coverage gain — fewer files, same 6/32.
Measurement demand before merge: re-run `benchmark.py --mode solver` before/after; report converged-then-rejected count vs new `max_iter` count per mission plus `##OVERBOARD-VALIDATION##` eligibility deltas. No tolerance change lands on reasoning alone.
`Solved_To_Acceptable` counting as converged stays acceptable only while `check_trajectory_approval` re-gates it — confirm no scoring-cap laundering in `_build_validation_record`.

## [AMENDMENT by verification]:
Land `round1/verification.md:5-7` (persist `propagation`+`inclusion`, assert in `check_trajectory_approval`) with this fix, else tighter solves stay push-indistinguishable plant-vs-tick.

## [CONCESSION by physics]:
Concede controls — 1 cm / 0.01 rad acceptable residual pre-consumes half the 20 mm gate; acceptable_tol=tol + trac_end + dt>=0.02 is the physically correct barrier.

## [RESOLUTION]:
measure-first, coupled — no merge on reasoning alone. Re-run benchmark.py --mode solver before/after; report converged-then-rejected vs new max_iter per mission plus eligibility deltas. Land jointly with T014 persistence fix (propagation+inclusion persisted, asserted in check_trajectory_approval), else tighter solves stay push-indistinguishable plant-vs-tick.
Owner: controls. Status: NEEDS-MEASUREMENT.
