# TOPIC-07 + TOPIC-08 — Research method + publication: forum debate

# Forum debate — TOPIC-07 (seed-7 variance) + TOPIC-08 (pub blockers)

*Ref Rina channeling Abla (method) + Prof. Lindqvist (academic) + Been-There Bob (ledger). READ-ONLY — no tree changes.*

## TOPIC-07 — Which missing replicate most threatens the Q17 null (seed 7 only)?

**Context (agreed facts).** All gate-coverage headlines — Q13 5–6/32, Q14 10/32, Q17 6/32, Q15 6/32 — reuse one draw: `research/suite.py --count 32 --seed 7` (`review_forum/round1/research-method.md:3`; `docs/RESEARCH_AGENDA.md:130`; `docs/measurements/2026-10-26-benchmarks/README.md:13`). Q17's registered null: expanded arm (`--candidates 6`) passes exactly the same 6 missions as the triad (`m010, m015, m017, m021, m028, m031`), +0.02% duration, seeds 3.03 cm apart → solved plans 0.0126 cm apart over 288 pairs (`docs/STRATEGY_LEDGER.md:240`; `docs/RESEARCH_AGENDA.md:130`). Status OPEN per `review_forum/synthesis/CLUSTERING.md:23`.

### Abla (method): rank the four missing replicates by threat to Q17

1. **Cross-seed suite replicate (seeds 7, 17, 27) — the direct threat.** The 6/32 headline has zero variance estimate: one 32-mission draw, 8 per profile, generator untouched since Q15. A second draw could move coverage ±few missions (Abla's own risk note, `research-method.md:11`), and a single flip in either arm breaks the "0 flips" null as stated. Minimal fix stands: re-run `suite.py` with seeds 7/17/27 + `harness.py` same flags, report median/range of pass rate instead of 6/32 (`research-method.md:3`).
2. **RQ-A full-suite Spearman (N=5 cherry-pick) — mechanism threat, not count threat.** The short-answer table (`docs/research/RQ-A_tracking_error.md:51-58`, S-curve/stops carry the claim, 82% attribution on one mission) generalises F4/F6 density sweeps from 5 hand-picked examples. If the full-32 `plan-vs-profile deviation vs tracking` Spearman is weak, Q17's *explanation* (IPOPT walks shaped seeds back — F8 mechanism) weakens, but the 6/32 *count* stands. Second priority.
3. **Benchmark provenance/layout (`benchmark.py:104-147` vs `research/harness.py:65,219`) — comparability gap, not null-threatening.** `avg/min/max_wall_time` over `runs=3` with no `provenance{}`, `propagation`, or arc-length vs uniform layout means Q17 wall/iter numbers can't be joined to the harness ledger. Fix by copying `harness.provenance()` + `propagation` + samples layout into benchmark JSON (`research-method.md:9`). Does not move 6/32.
4. **Testcache 292 s vs 251 s (`AGENTS.md:59` vs `KNOWN_ISSUES.md:158`) — irrelevant to Q17.** N=1 wall-time, unpinned tree/machine. Doc-hygiene fix only (median-of-3 + `harness.tree_state()`-style hash, delete the other number). Zero threat to the null.

**Verdict:** replicate #1 is the one that can kill Q17; #2 can demote its mechanism from law to anecdote; #3–#4 cannot.

### Bob (ledger): is re-running re-litigation or required variance?

**Required variance — provided it is pre-registered and re-tunes nothing.** Q17 closed 2026-10-26 (`ledger-guardian.md:11-12`); reopen rule: "a geometric term that *holds* shape at a duration price a scoring budget can pay, or missions where a deliberately slower shaped path passes a gate. Do not re-tune the 2 cm bar, 120° reverse threshold, or apex family" (`ledger-guardian.md:12-13`). Re-running the *same frozen tail* on seeds 17/27 to put error bars on 6/32 is **not** re-litigating the retired geometry (rescue 0/22 in `ledger-guardian.md:5-8`, polish 33/33 identical in `:15-17`, evolution in `:21-22` all stay buried). It is the variance the null policy assumes. Re-tuning thresholds or adding a new shape on the new seeds **is** re-litigation and stays rejected per TOPIC-10 (`CLUSTERING.md:32`).

**Bob's condition:** register seeds, arms (`--candidates 3` vs `6`, `--propagation plant`), and the two conjuncts (coverage ≥2 flips, distinctness >2 cm) *before* running; report median/range; the collapse diagnostic (seed distance → plan distance) must be recomputed on each seed, since that is the part of Q17 that should survive a seed change even if the count wobbles.

## TOPIC-08 — Rank the 4 pub blockers by Reviewer-#2 lethality, minimal fix order

Source: `review_forum/round1/academic-lead.md:3-9`; topic status OPEN (`CLUSTERING.md:27`).

| Lethality rank | Blocker | Why this kills | Minimal fix (in order) |
|---|---|---|---|
| **1 — fatal (integrity)** | Selective reporting: paper foregrounds straight-PASS (2.37 s, 0.28 cm) and superseded 2026-10-19 tables while the current suite is **6/32 gate-ok, 26 failures tracking/envelope**, and `example_complete` FAILED after 670 s wall (`academic-lead.md:9`; `measurements/README.md:29-32,73-78`; `WHITEPAPER.md:566-584,327-363`; `CLAIMS.md:20,28` admit local-optimum/unmeasured transcription) | Reviewer #2's classic desk-reject: hiding the headline negative result. Everything else reads as spin once spotted. | Put 6/32 + FAILED + wall ratios in §10/results foreground, cite `bench_solver.json` / `bench_*_plant.jsonl` by hash, register the missing KKT/convergence-order study (Q9/Q18 regime) as explicit future work instead of implying time-optimality. |
| **2 — fatal (reproducibility)** | Key evidence not rerunnable: Q17 JSONL "lost when that worktree was removed" (`measurements/README.md:138-142`), `research/results/` gitignored, oracle needs WSL2 + Node 22 + unlicensed unpinned `martijas/fll-sim` (`crosscheck/README.md:41-60`), RQ-A scripts need `git checkout archive/research-2026-10` (`RQ-A_tracking_error.md:70-74`) | Fails the reproducibility/artifacts checklist outright; reviewer cannot verify any number in rank 1. | Track the JSONL records; pin `fll-sim` commit hash + `trace.csv` checksums in-repo; one Linux container script for `run_sim.sh`/`run_traj.sh`. |
| **3 — major (contribution)** | Novelty overclaim vs no baseline: "5 Principles/Laws" + PMP saturation as discovery (`OPTIMAL_CONTROL_DERIVATION.md:38-65`; `WHITEPAPER.md:686-695`) while the file banner and `CLAIMS.md:30-31` hedge to "heuristics, not theorems" / "consistent with PMP, not a proof", §13 has 8 refs with zero TOPP/TOPP-RA, Choreo/PathPlanner, or direct-collocation literature, and `CLAIMS.md:32` admits "Faster than Choreo/PathPlanner/TOPP — Not measured" | Contribution collapses to "uncompared heuristic"; reviewer demands the related-work table and baseline or rejects as incremental. | Rename to "heuristics" everywhere; add one related-work table (TOPP, Choreo, collocation) with citations; carry the no-baseline disclaimer into the paper abstract. |
| **4 — minor (desk-reject on technicality)** | `CITATION.cff:12-14` uncitable: author `name: "InfiniteQuery"`, no `family-names`/`given-names`, no ORCID, no DOI; `version: "0.1.0a2"` + `date-released: "2026-10-26"` stale vs 2026-10-27/28 tree | JOSS/software-track automation flags it before review; cheapest to fix, most embarrassing to leave. | Split to `family-names`/`given-names` + `orcid`, mint Zenodo DOI, add `doi:` + `preferred-citation`, bump version/date on release. |

**Lindqvist's fix order = lethality order (1→4):** no point polishing the citation (4) or related work (3) while the results table hides 6/32 (1) and nothing reproduces (2). Total: four small doc/packaging changes, zero solver changes.

**Bob's footnote:** none of the four re-opens rescue/tail/polish/evolution (`ledger-guardian.md:3-23`); all four are variance-and-honesty repairs the ledger already requires (Q20 error-kind labelling, M1 re-measured budgets).

*Skipped: re-reading `pq_b/FINDINGS.md`, full `WHITEPAPER.md` proofread, re-running any suite/benchmark (read-only session). Risk: cross-seed replicate may move 6/32 by ±few missions and weaken the tail/coverage nulls' headline form even while the collapse mechanism holds — report the range, don't re-tune the gate.*
