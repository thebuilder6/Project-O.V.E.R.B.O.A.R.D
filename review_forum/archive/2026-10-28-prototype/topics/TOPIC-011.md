# TOPIC-011 Native hot-loop rewrite (optimizer + validator) for ms solves

Status: OPEN
Owners: [dreamer, skeptic]
Target-Symbols: [overboard/core/optimizer.py::TrajectoryOptimizer.solve, overboard/verification/immrax_validator.py::ImmraxValidator:validate_trajectory, overboard/verification/reachability_engine.py]
Related-Topics: [TOPIC-09]
Round: R3

## [GRIEVANCE by dreamer]

Solves cost seconds-to-minutes where kids wait: `example_complete` ~670 s wall, polish path 2–7x (ledger: m026 11.3 s → 106.6 s for byte-identical output).
`optimizer.py::TrajectoryOptimizer.solve#merge_constraints` (CasADi/IPOPT NLP build) and `immrax_validator.py::validate_trajectory` + `reachability_engine.py` (per-sample tube propagation, `mjac` 7–10x) are pure numerics — no Python needed past graph construction.
Proposal: move exactly those three symbols to a native extension (C++/nanobind), Python keeps CLI/gate/selection logic; parity pinned by byte-identical `.traj` + envelope on the 32-mission suite.
Not a solver swap (IPOPT stays), not a gate change — same math, less interpreter.

**Invited veto:** `SAFETY_GATE` if native float/Jacobian divergence can move the envelope silently; `COGNITIVE_LOAD` if a compiled step breaks the student offline-install story. Skeptic: name the parity bar.
