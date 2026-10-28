# Round 1 v3 — academic-lead (Lindqvist)

F1. Whitepaper KI-011 self-contradiction. WHITEPAPER :424/:654/:664/:678 say fixed; :554 §9.5 says still wrong. API_validator_export + KNOWN_ISSUES agree fixed → :554 stale — but as written export-correctness claim self-refutes.
F2. CITATION.cff abstract overclaims vs CLAIMS:20/24 (local optimum only; envelope assumed). Metadata carries none of hedging.
F3. §2.3 "same numbers" unmeasured (JAX 0.12.0/Immrax 0.3.8, same missions same numbers) — no harness/log/pin; next to loose tol=1e-2 + Not-measured transcription (CLAIMS:28). Optimality-shaped claim on loose tolerance, no mesh study, no pinned rerun.
F4. Stale-number tables bannered but citable out of context (§5.3/10.3/10.5 + derivation GA figures) — envelopes to 0.001cm with no hash beside table; crosscheck oracle correctly scoped (7%, no RR, brake trace unusable) — tables need same pinning.

Kill shot: central claim (spread first, then time) admitted unmeasured §12.1, KI-020 open, §10.3 no hardware log, loose NLP, no mesh study — solver benchmark with sim gate, not validated robotics. Reject pending repeated hardware spread or retitle.

Fix: 1) delete :554; 2) qualify CFF abstract (local optima + model-only gate, KI-020 open); 3) drop/anchor §2.3 sentence; 4) pin each table (tree date+config+flags) or move superseded to ledger.

**[VETO: UNVERIFIED_CLAIM]** on submission/publication framing until (i) KI-011 fixed, (ii) hardware spread (KI-020), or (iii) downgrade to local-NLP + sim gate.

Skipped re-solves. Risk: F1 must fix before external eyes.
