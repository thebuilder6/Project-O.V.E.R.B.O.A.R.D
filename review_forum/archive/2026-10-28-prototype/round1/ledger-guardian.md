# Round 1 — ledger-guardian (Been-There Bob)

Listen up — I've buried these ideas twice already. Don't dig 'em up without new dirt.

**1. GRIEVANCE: "Just retry the failures with another seed / higher weight" — the rescue round (KI-048)**
Ledger: §18.4 (accuracy-weight ladder) + KI-048 record.
Retiring measurement: 32-mission suite, `--candidates 3 --validate --propagation plant`: 22 missions failed, every one ran the rescue, **0 rescued**. 95 of 97 checked rescue candidates failed the *envelope* — same gate that killed the set. Doubled wall time on every failing mission for nothing. The older ladder variant was worse: slow 17.1 s candidate **diverged** while the 6.07 s one stayed bounded.
Reopen only if: you show a failure population that is *not* envelope-bound — e.g. controller-jerk failures where a higher weight buys tracking without growing the tube. No envelope failures need apply.

**2. QUESTION Q17: "Add clever geometry — mirrored bow, Bezier, reverse segments — for gate coverage" — the tail (§19.7)**
Ledger: §19.7; Agenda §24.0 Q17 closed 2026-10-26.
Retiring measurement: 32 missions m000–m031: triad 6/32, expanded `--candidates 6` 6/32 — **0 flips**, +0.02% duration. Collapse diagnostic fired: seeds 3.03 cm apart mean pairwise, solved plans **0.0126 cm** apart over 288 pairs. Stage 2 anchor (`w_anchor=1000 s/m²`) also null: 0.030 cm deviation, 5/32 both arms, price +33–142% duration, iterations ×4. Term removed again.
Reopen only if: a geometric term that *holds* shape at a duration price a scoring budget can pay, or missions where a deliberately slower shaped path passes a gate. Do not re-tune the 2 cm bar, 120° reverse threshold, or apex family — those are registered definitions.

**3. GRIEVANCE: "Polish the near-miss with the reachability gradient" — the gradient polish (§24.2, Q16, KI-058)**
Ledger: §24.2; Agenda §24.0 Q16 closed, re-confirmed 2026-10-27 on fixed lens.
Retiring measurement: 33/33 candidate polishes across 6 examples + 5 near-miss missions **byte-identical on/off**, for 2–7× wall (m026 11.3 s → 106.6 s). Mechanism measured: polish raises `accuracy_weight` ×3 (smoothness bought with time — the Q6 trade) while the accept rule admits no time increase + strict (tracking, envelope) improvement. That intersection was empty everywhere. KI-058's phantom-error lens is fixed (`GRAD_TICKS` 1→5), verdict unchanged.
Reopen only if: as an M1 in-loop linear/proximal term where time is spent *to reach feasibility*, gated on KI-020 — never again as post-hoc polish. The delete branch stands.

**4. GRIEVANCE: "Bring back evolution / Multi-Verse refinement — it finds better lines" (KI-004 deleted)**
Ledger: §19.4 (44-gene genome), §20.1–20.2 (JAX refinement, refine probe); Agenda §24.0 Q12; archive tag `archive/legacy-2026-10`.
Retiring measurement: over 30 missions as seed provider, refined seed **rescued nothing** — faster on 2 of 9 where both pass, slower on 6. Complex mission: refined 8.1473 s vs baseline 6.0677 s. The 44-gene "win" (5.417 s, 4.9 mm) was actually Generation 0 `seed_4` (5.514 s) from the pyramid bank — 25.8 s of evolution added nothing. Superseded by deterministic frontier triad: 5.233–5.239 s, 4.3–4.5 mm, **1.16 ms** seed cost.
Reopen only if: stochastic search finds a basin the <2 ms deterministic homotopy demonstrably cannot reach on the *shipping* candidate set — not on stock sets, not on one mission. Until then the code stays in the tag.

---
Skipped: Q15 tracking-as-constraint and Q4 envelope-in-objective nulls (same "gate is not a price" shape, see agenda). Risk: all numbers above are pre-hardware predictions (KI-020 open) — a twin-calibrated gate could move every bar.
