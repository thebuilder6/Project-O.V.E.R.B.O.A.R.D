# Claims register

What OVERBOARD claims, and what stands behind each claim. If a claim is not in this
table, treat it as unclaimed. Numbers here are the ones the code and the tests produce;
where a document quotes a different number, this file and the code win.

Status vocabulary:

- **Proven** - a result in the cited literature, or a property pinned by a test against
  the shipped code.
- **Measured** - a number produced by a script or test in this repository, reproducible
  for the missions and configuration named. It says nothing about missions not named.
- **Assumed** - a modelling choice. If it is wrong the result is wrong, and nothing here
  tests it.
- **Predicted** - an expectation that no experiment in this repository establishes yet.
- **Not measured** - explicitly absent; do not claim it.

| Claim | Status | Evidence | Appears in |
| --- | --- | --- | --- |
| The planner produces time-optimal trajectories | **Measured, locally** - the solver minimises the transcribed cost and a converged plan is a local optimum of that NLP; nothing proves it is the optimum of the continuous-time problem | `overboard/core/optimizer.py` (NLP), `benchmark.py --mode solver` | README, `docs/WHITEPAPER.md` |
| Wheel force and slip limits are respected | **Measured in the model**, at both ends of every interval | `validator.audit_constraints`, `tests/test_validator.py::TestAuditConstraints`; first-order motor + lateral weight-transfer force model | README, `docs/WHITEPAPER.md` |
| The plan is safe to drive on the real robot | **Assumed / Predicted** - depends on the model matching the robot, which no hardware campaign has established | KI-020; open item #15 of the audit | README, `docs/WHITEPAPER.md` |
| Tracking error stays under the 5 mm gate | **Measured in the model, mission-dependent** - `example_straight.json` passes (0.03 cm); `example_s_curve.json` fails (~1.49 cm) today; the S-curve and stop missions also fail their envelope gate | `python -m overboard.main -w examples/example_straight.json --validate`; `docs/research/RQ-A_tracking_error.md` | README, `docs/WHITEPAPER.md` |
| The reachability envelope bounds the real trajectory | **Assumed** - interval propagation bounds the declared uncertainty; the model, the inclusion of rounding, and the plant-robot match are not proven | `overboard/verification/immrax_validator.py`; KI-020, KI-049, KI-050 | README, `docs/WHITEPAPER.md`, `docs/STUDIO_3D_SIMULATOR_PLAN.md` |
| A run without a validator can never report CERTIFIED | **Proven** by test - with no engine the gates read infinity | `tests/test_polish.py::TestPlanGates::test_without_the_engine_nothing_is_certified` | `overboard/core/unified_planner.py` |
| An unvalidated export cannot be mistaken for an approved one | **Proven** by test - the `.traj` root carries `validation: {status: "unvalidated", passed: false}` | `tests/test_cli_ux.py::TestCliBasics::test_simple_solve_uses_the_default_config` | `overboard/main.py`, `AGENTS.md` |
| The Ramsete controller converges | **Proven in continuous time**; the shipped 20 ms discrete loop is **Measured only** | Aicardi et al., IEEE RAM 2(1):27-35, 1995; De Luca, Oriolo, Vendittelli, "Control of wheeled mobile robots", 2001 | `docs/WHITEPAPER.md` references |
| The transcription is consistent and converges at the documented order | **Not measured** - no convergence or KKT study exists yet | open item #9 of the audit | `docs/OPTIMAL_CONTROL_DERIVATION.md` |
| The 50 ms invariant mesh is the right discretisation | **Assumed / heuristic** - chosen, not derived; the 5 mm gate decides whether a plan passes | `unified_planner.compute_invariant_mesh_samples`; `tests/test_doc_numbers.py` | README, `docs/AGENT_OPS.md` |
| Five "physical laws" govern time-optimal motion | **Heuristics** - recurring patterns with a good record on the example missions, not theorems | `docs/OPTIMAL_CONTROL_DERIVATION.md` banner | README, `docs/WHITEPAPER.md`, `docs/STRATEGY_LEDGER.md` |
| The time-optimal plan saturates an actuator (PMP) | **Measured observation** on the example missions - consistent with PMP, not a proof | phase portraits in `docs/VISUAL_GALLERY.md` | `docs/WHITEPAPER.md`, `docs/VISUAL_GALLERY.md` |
| Faster than Choreo / PathPlanner / TOPP | **Not measured** - no baseline harness on the same missions and limits yet | open item #11 of the audit | nowhere; do not add it |
| +19.2 % faster than the GA champion; beats the 44-gene champion | **Measured once** (2026-10-18/20) on the archived evolutionary pipeline and the then-current ranges; not reproducible from this tree | git tag `archive/research-2026-10`; `docs/STRATEGY_LEDGER.md` (dated) | `docs/WHITEPAPER.md`, `docs/STRATEGY_LEDGER.md`, `docs/VISUAL_GALLERY.md` |
| The candidate set yields the fastest plan among those solved | **Measured by construction** - the selection rule sorts the candidates it solved; it says nothing about candidates it did not solve | `UnifiedPathEngine.plan` | README, `docs/AGENT_OPS.md` |
| The test suite is green | **Measured** - the runner prints the count in TEST SUMMARY; do not quote a count in prose | `python run_tests.py` | `AGENTS.md`, `.github/workflows/tests.yml` |
| It works for any differential-drive robot | **Predicted** - the model takes any parameter set; a robot with no calibration is planned against defaults | KI-020, `docs/CALIBRATION_PROCEDURE.md` | README |

## Reproducing a row

```bash
python run_tests.py                       # the rows marked "by test"
python -m overboard.main -w examples/example_straight.json --validate
python -m overboard.main -w examples/example_s_curve.json --validate   # fails today
```

Rows marked **Measured once** need the archived tag; the current tree cannot regenerate
them, which is why they are dated wherever they appear.
