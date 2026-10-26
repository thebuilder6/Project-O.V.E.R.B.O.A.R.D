# 2026-10-26 — Comprehensive benchmark run

Post-campaign baseline for the whole product: the solver reference table, the
CLI and pipeline paths, randomised stress, the 32-mission scoring suite on the
shipped default and the ledger protocol, plant-vs-tick validation cost, and the
random-robot harness.

- **Tree:** `main` @ `4001e68` (immediately after the Q17 tail merge), clean, pushed.
- **Environment:** Python 3.12.10, JAX 0.6.2, CasADi 3.7.2, Windows.
- **Config:** `examples/fll_choreo.json` unless a run says otherwise.
- **Protocol:** every phase ran alone on the machine (no concurrent load), in the
  order below. Suite arms are the registered Q13–Q17 protocol
  (`research/suite.py --count 32 --seed 7`, m000–m031, 8 per profile).

## 1. Solver reference table — `bench_solver.json`

`python benchmark.py --mode solver --output docs/measurements/2026-10-26-benchmarks/bench_solver.json`

| mission | wall s | iters | traj s | result |
|---|---:|---:|---:|---|
| example_straight | 0.11 | 19 | 2.4013 | converged |
| example_s_curve | 0.27 | 40 | 3.7461 | converged |
| example_mission_first4 | 3.16 | — | 2.0666 | converged |
| example_with_stops | 2.44 | — | 5.3777 | converged |
| example_with_events | 0.69 | 37 | 4.8379 | converged |
| example_complex_segment | 1.98 | 104 | 5.1574 | converged |
| **example_complete** | **670.57** | — | 12.1650 | **FAILED** |
| example_complex_mission | 10.18 | — | 7.4492 | converged |
| total | 689.41 | | | 7/8 |

`example_complete.json` is the one non-converging example: 670 s of wall and no
converged plan. It is the longest mission in the gallery and dominates the run.

## 2. Pipeline — `bench_pipeline.json`

`python benchmark.py --mode pipeline --runs 3 --output …`

Three runs, 3-waypoint toy path: solve 0.1669 / 0.1604 / 0.1587 s, trajectory
duration 6.1562 s every run (bit-identical), all converged. The planner is
deterministic run-to-run on this machine.

## 3. CLI with validation — `bench_cli.json`

`python benchmark.py --mode cli --validate --output …` (3 example missions × 3 runs)

| mission | subprocess wall s (avg, range) | RK4 max pos err mm | pass |
|---|---|---:|---:|
| example_straight | 0.58 (0.31) | 0.429 | 3/3 |
| example_s_curve | 0.62 (0.01) | 0.398 | 3/3 |
| example_complex_mission | 3.43 (0.19) | 0.442 | 3/3 |

Every run passed with zero constraint, slip or integration violations, and the
validation metrics were bit-identical across the three repeats of each mission.

## 4. Randomised stress — `bench_random.json`

`python benchmark.py --mode random --report --output …` (10 runs, profiles
`short_sprint`, `complex`, `stress_test`)

10/10 PASS. Profile mix: complex ×3, stress_test ×3, short_sprint ×4. Zero
violating samples and zero slip points on every run; RK4 max position error at
worst 0.945 mm. The mode's markdown report (`benchmarks/benchmark_report.md`)
and per-run `.traj` files are generated artifacts and are not tracked.

## 5. The 32-mission suite, three arms — `bench_2026-10-26_c2_plant.jsonl`, `…_c3_plant.jsonl`, `…_c3_tick.jsonl`

Missions regenerated with `python research/suite.py --count 32 --seed 7 --out
docs/measurements/2026-10-26-benchmarks/missions` (tracked here, as in the
2026-10-20 stress record). Arms via `python research/m3_matrix.py <missions>
--out … --candidates N --propagation plant|tick`. Scoring budgets are the
measured per-mission ones the tool derives from the baseline slice (Q17 rule).

| arm | candidates | propagation | gate-ok | mean selected dur s | wall s |
|---|---:|---|---:|---:|---:|
| shipped default | 2 | plant | 6/32 | 2.329 | 340 |
| ledger protocol | 3 | plant | 6/32 | 2.325 | 485 |
| ledger protocol | 3 | tick | 6/32 | 2.325 | 2783 |

- The passing six are **m010, m015, m017, m021, m028, m031** in every arm — the
  same set as Q13/Q15/Q17. Gate coverage is identical at 2 and 3 candidates:
  the third candidate (and, per Q17, the whole tail) moves selection, never
  coverage. The 26 failures are the known tracking/envelope gates (RQ-A, KI-054).
- Zero infeasible, zero over-ramp in all arms; 32/32 missions converge per arm.
- Production score selection vs fastest-passing diverge exactly where designed:
  m031 selects `frontier_sweeper` at 3.462 s over the faster `frontier_sprint`
  in every arm; m010 selects `frontier_sprint` while `frontier_exploratory` is
  fastest on the triad arm. With only two candidates, m017 falls back to
  `frontier_sweeper` (1.876 s); the triad's `frontier_exploratory` is faster (1.85 s).
- **Tick vs plant:** verdicts agree on all 32 missions. Full-mission wall is
  5.7× the plant arm, but this cold ratio is dominated by the per-mission JAX
  compilation of the tick map (shapes differ per mission). The warm steady-state
  validation cost is 2.3–2.4× plant on normal-length missions (M2's benchmark,
  2026-10-25), rising toward 5× on multi-second paths.

## 6. Random-robot harness — `bench_2026-10-26_harness_robots.jsonl`

`python research/harness.py --missions docs/measurements/2026-10-26-benchmarks/missions
--out … --specs default --robot-jitter 1 --robots 2` (plant, 2 candidates, 8
workers; `research/analyze.py errors|robots` read-backs)

128 validations (32 missions × 2 candidates × 2 jittered robots): **25 pass
(19.5 %)**, 7 hard solver errors (all on m002, a CasADi `Opti::solve` failure),
0 candidates empty. 727 s wall.

Pass rate by robot constant (selected rows; bins are small — see the record for
all of them):

| constant | best bin | worst bin |
|---|---|---|
| cof | 0.8 → 40 % (n=20) | 0.9–1.1 → 12 % |
| track_width | 1.1 → 33 % (n=58) | 1.0 → 2 % (n=42) |
| v_max_rad_s | 0.9 → 29 % | 1.1 → 7 % |
| gear_backlash | 1.3–1.4 → 50 % | 0.5, 0.9, 1.5, 1.8–2.0 → 0 % |

No constant decides the verdict monotonically; median envelope excess sits at
2–10 cm on most failing bins, i.e. the gates — not the planner — bind. This
repeats the 2026-10-20 stress finding on the current tree.

## Takeaways

1. The product is healthy on its examples and random missions: everything
   converges except `example_complete`, and every converged plan passes
   validation with margin (RK4 errors ≤ 1 mm).
2. The shipped default (2 candidates) matches the triad's gate coverage on the
   suite; selection differences are the score rule working as intended.
3. Tick propagation is verdict-identical to plant on this suite and is a
   validation-time cost, not a coverage lever. Budget compile cost per mission
   shape when using it interactively.
4. Under robot jitter the gates remain the binding constraint (19.5 % pass);
   `cof`, `track_width` and `v_max_rad_s` move the rate the most.

## Caveats

- Wall times are this machine, sequential runs, cold imports; absolute numbers
  do not transfer, ratios do.
- `example_complete`'s non-convergence is reported as observed — no diagnosis.
- The Q17 experiment records (`research/results/q17_*.jsonl`) were worktree-local
  and were lost when that worktree was removed; they are reproducible from the
  measurement trees `6d1b2f4` (arms) and `1794bc9` (anchored arms).
- Copies of the JSONL records in this directory also live in the gitignored
  `research/results/` under the same `bench_2026-10-26_*` names.
