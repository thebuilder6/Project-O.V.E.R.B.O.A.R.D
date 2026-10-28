# Synthesis Report — 10 reviewers + Moderator (Ref Rina)

10 solo audits (3-4 findings each) -> 10 canonical topics -> 5 debate threads (2 turns each). Ledger vetoes held: rescue / Q17 tail / gradient polish / Multi-Verse stay CLOSED.

## Ranked table (severity x gate-impact)

| Rank | Topic | Verdict | Owner | Minimal diff |
|---|---|---|---|---|
| 1 blocker | T02 field uncertainty optimistic | ACCEPT — additive fix `(err+u)/raw`, nominal-only structural, declared waypoints absolute, initial box param | V&V + physics | `main.py:_field_limits`, `:1666-1684`, `:_local_limit:560`, `reachability_engine:227` |
| 2 blocker | T03 force unconservative (CG-h, lift-as-slip) | ACCEPT fixes 1+2 now, 3-5 follow-ups | Vora (models) + Sara (firmware #5) | `h_cg_m` in RobotConfig; `max(0,raw)` + `lift` flag |
| 3 blocker | T08 publication integrity (6/32 hidden, non-rerunnable) | ACCEPT 1→4 order | Academic-lead | results table + pin fll-sim + heuristics rename + CFF |
| 4 major | T01 selection fastest-vs-score + stale help | ACCEPT delete early_stop, rewrite 3 strings + test | Architect, V&V signs | `main.py:1971-2012`, `test_cli_ux:1703`, README/help |
| 5 major | T04 hub trust (config swallow, POSE, noop events) | ACCEPT fail-loud + explicit overrides | Robotics | `--keep-hub-config`, session envelope, `ALLOW_NOOP_EVENTS` |
| 6 major | T07 method (seed-7 only, N=5, provenance) | ACCEPT seeds 7/17/27 pre-registered, no re-tune | Research-method, Bob gates | suite re-run + harness provenance |
| 7 minor | T06 studio UX (jargon, budget, busy) | ACCEPT rename+tooltip, draft budget, sticky hub | Student/Coach + arch | values frozen, inventory doc same diff |
| 8 minor | T05 classroom (30-min vs 45-min, offline) | ACCEPT 1-pass preset + paper sheet | Coach | `collect --repeats 1`, offline pre-install doc |
| 9 debt | T09 architect (BLAS, registry, gate, desktop) | ACCEPT C+A first, B, D | Architect | isinstance, ModuleNotFoundError, initializer, version stamp |
| 10 closed | T10 already-tried + doc drift | REJECT reopen; ACCEPT doc tickets KI-051/052, KI-055 | Bob + librarian | edit Fixed entries, delete stale limitation |

## Doc-vs-tree drifts (tree wins)

- README/help "fastest" vs score selection (T01)
- KI-051/052 Fixed preserves removed `--self-consistent` switch; KI-055 Fixed vs CHANGELOG a2 stale limitation
- README 50mm vs docs ~100mm vs per-season field value (fold into T02)
- HEURISTICS :684 documents skip rule deleted by T01 fix

## Publication blockers (Reviewer #2 order)

1. 6/32 + FAILED in foreground + wall ratios, by hash
2. Rerunnable artifacts (JSONL, fll-sim pin, container)
3. Heuristics rename + TOPP/Choreo related-work table
4. CITATION.cff fix

## Action tickets (next session)

- A1 (blocker): T02 additive formula + waypoint absolute + initial-box param
- A2 (blocker): T03 CG-height + lift flag
- A3 (blocker): T08 results table + artifact pins
- A4: T01 delete early_stop + help rewrite
- A5: T04 fail-loud hub trio with overrides
- A6: T07 cross-seed 7/17/27 pre-registered
- A7: T05/T06 studio + classroom UX (values frozen)
- A8: T09 C+A debt, then B, D
- D1: librarian doc tickets (KI-051/052, KI-055, testcache numbers)

*All findings read-only; line numbers re-grep before editing. Gate-soundness caveat: do not treat near-obstacle passes or tall/fast validations as decided until A1+A2 land (KI-020 open).*
