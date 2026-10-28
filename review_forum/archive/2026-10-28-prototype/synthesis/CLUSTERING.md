# Clustering — 40 raw findings -> 10 canonical topics (Ref Rina)

Ledger pre-filter: rescue-round, Q17 tail, gradient polish, Multi-Verse stay CLOSED (see TOPIC-10). No new evidence offered.

## TOPIC-01 Selection rule confusion [verification G1 + librarian G2]
Fastest-passer early-stop (`main.py:2002-2012`) contradicts score selection (`_select_attempt_scored`), while README + `--candidates` help still say "fastest, envelope tie-break". Status: OPEN.

## TOPIC-02 Field uncertainty not applied [verification G2,Q,S + librarian Q4 + physics risk]
Nominal clearance priced while ~100mm placement recorded but never subtracted; per-metre allowance weakens declared 5mm waypoints; initial tube ±1mm vs hand-place error; 50-vs-100mm drift. Status: OPEN.

## TOPIC-03 Force model unconservative [physics G1,G2,G3,Q + robotics S4]
Negative wheel normal -> slip misreport; CG_HEIGHT constant not in config; merge_constraints only with per_segment_dt; traction end-rows dropped as shortcut; limiter uses wall-clock dt vs 20ms model. Status: OPEN.

## TOPIC-04 Hub deploy trust [robotics G1,G2,Q]
Config sync swallowed (`except: pass`), POSE breaks session contract, MISSION_EVENTS={} silent no-op. Status: OPEN.

## TOPIC-05 Classroom onboarding [coach G1,Q + student G1,Q,S4]
30-min/27-measure cal vs 45-min class; button/CSV offline; 6.28 unexplained; install diary jargon; doctor names missing sysid_fit.py. Status: OPEN.

## TOPIC-06 Studio UX [student G3 + coach G2,S]
Plant/tick/mesh jargon; waypoint drag silently invalidates M1 budget; single-connection hub-busy + wrong-robot risk. Status: OPEN.

## TOPIC-07 Research method [research-method all]
N=1 suite seed 7 for all coverage claims; RQ-A N=5 cherry-pick; testcache 292 vs 251s; benchmark missing provenance/layout. Status: OPEN.

## TOPIC-08 Publication blockers [academic-lead all]
CITATION uncitable; novelty overclaim vs TOPP/Choreo with no baseline; lost JSONL + unpinned fll-sim + gitignored results; 6/32 + FAILED not in paper foreground. Status: OPEN.

## TOPIC-09 Architect debt [architect all]
Registry swallows ImportError; BLAS env mutate in parent; module-string parallel gate; desktop seed-once + Linux-only validate smoke. Status: OPEN.

## TOPIC-10 Already-tried + doc drift [ledger all + librarian G1,G3]
Rescue 0/22, tail 6/32->6/32, polish 33/33 identical, evolution slower — CLOSED unless new evidence. Doc tickets: KI-051/052 removed-flag text, KI-055 stale limitation. Status: DEBATED->RESOLUTION pre-filled (reject reopen, accept doc tickets).
