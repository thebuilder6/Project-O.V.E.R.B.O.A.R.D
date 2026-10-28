# TOPIC-01 + TOPIC-09 — Selection + architect debt: forum debate

# Forum debate — TOPIC-01 + TOPIC-09 (post-round-1 rebuttals)

Filed by Ref Rina (channeling architect + verification + librarian). All citations re-checked against the tree; line numbers below are as-verified, differing slightly from round-1 where noted.

## TOPIC-01 — Selection rule confusion (verification G1 + librarian G2)

**Recap.** `_run_candidates` validates fastest-first and marks slower candidates `validation_skipped` once one passes (`overboard/main.py:2002-2012`), while `_select_attempt_scored` picks highest decision score (`overboard/main.py:160-185`). README (`README.md:246-254`), `--candidates` help (`overboard/main.py:2324-2329`) and `--pick-candidate` help (`overboard/main.py:2430-2433`) still teach "fastest, envelope tie-break".

**New evidence since round-1 (both sides accept):** the sole production caller already passes `early_stop=False` (`overboard/main.py:2852-2861`, with a comment stating the score rationale). The studio shells the CLI, so it inherits the fixed path. The `early_stop=True` default is now exercised only by `tests/test_cli_ux.py:1703-1709`, which pins it. `docs/HEURISTICS_AND_VERIFICATION.md:684` still documents the skip as the rule.

### [REBUTTAL by architect]

Verification's option B (skip only when a fitness upper bound can't beat the best score) is new machinery for a 2–3-candidate set where the validations are already paid on the only path that matters. The production call site proves the team already chose "pay the extra validations". So don't bound it — delete it: remove the `early_stop` parameter (`overboard/main.py:1971-1973`), the skip branch (`overboard/main.py:2002-2012`), and rewrite the test that pins the default (`tests/test_cli_ux.py:1703-1709`) to assert every candidate is checked. Same diff fixes the librarian half: rewrite the three stale strings (`README.md:246-254`, `overboard/main.py:2324-2329`, `overboard/main.py:2430-2433`) to the score rule and correct `HEURISTICS_AND_VERIFICATION.md:684`. One deletion-shaped diff, no new logic, no new flags.

### [REBUTTAL by verification]

Conceded: the CLI production path is sound today — do not relitigate past `--validate` CLI runs that went through `main.py:2852`. Not conceded: the gate is sound. A skipped record carries `score None` and is structurally ineligible in `_select_attempt_scored` (`overboard/main.py:160-185`), so any caller taking the `early_stop=True` default gets "fastest passer" while the JSON claims a score decision over a set that was never fully scored. The default is a trap for the next caller, the pinned test blesses it, the design doc blesses it, and the user-facing strings teach the wrong mental model — an operator reading "fastest" in `--help` will misread a score-selected JSON. Gate-soundness standing: any multi-candidate selection *not* routed through the `early_stop=False` call site must not be treated as correctly decided until the default is gone. Deletion (architect's proposal) satisfies this; bound-logic would too, but deletion is smaller — accept deletion, provided the test is rewritten to fail if a skip branch is ever reintroduced.

### [RESOLUTION] — **accept** (delete, don't bound)

Owner: **architect** (implements); sign-off: **verification** (re-runs `TestScoreSelection`-class tests in `tests/test_cli_ux.py:1548-1625` + the rewritten skip test). Minimal diff, in order: (1) delete `early_stop` param + skip branch (`main.py:1971-2012`); (2) rewrite `test_cli_ux.py:1703-1709` to assert full checking; (3) rewrite the three "fastest" strings + `HEURISTICS_AND_VERIFICATION.md:684` to the score rule. No measurement needed — single-caller evidence is in the tree.

## TOPIC-09 — Architect debt: flake-risk ranking + minimal-diff order

Four items, citations as-verified (round-1 numbers in parens where drifted):

- **A. Registry swallows ImportError** — `tests/test_suite_registry.py:31-34` (`:28-34` in round-1). Any broken import in a new `test_*` module reads green because it never loads.
- **B. BLAS env mutated in parent** — `overboard/main.py:273-290` (`:273-290` confirmed). Save/set/restore around the pool; `finally` covers the pool block, so the live risk is thread-safety (global state leaked to a second thread) and the sequential fallback inheriting the parent env.
- **C. Module-string parallel gate** — `overboard/main.py:320-328` (`:327` in round-1; now `may_run_parallel`). Any subclass/wrapper/test double silently serializes with no message.
- **D. Desktop seed-once + Linux-only validate smoke** — `overboard_desktop.py:43-53` (`:44-55` in round-1) seeds state once and never refreshes; `.github/workflows/desktop.yml:70-77` runs frozen `--validate` smoke on Linux only, leaving the Windows kill path (`overboard/gui_server.py:173-174` `taskkill`) untested.

### Rank by flake-risk (nondeterministic-failure risk, highest first)

1. **B (BLAS env)** — only item with true cross-thread nondeterminism: global mutable state + a second thread or nested run leaks limits one way or the other. Flakes as sporadic oversubscription/slowness.
2. **D (desktop/Windows)** — environment-dependent: stale firmware/docs after upgrade, plus an entirely untested Windows-only kill path. Fails only on some machines/versions — the classic "works on CI Linux" flake profile.
3. **C (module-string gate)** — deterministic silent serialize; flake link is second-order (serialized runs timing out under loaded CI look like flakes). Weak, rank it third.
4. **A (registry swallow)** — zero flake-risk (deterministic), but the worst *silent-pass* risk of the four: a green suite that never ran the broken module. Ranked last here only because the axis is flake-risk; on a correctness axis it would be first.

### Minimal-diff order (land cheapest first — differs from flake rank, deliberately)

1. **C** — one line: `isinstance(optimizer, TrajectoryOptimizer)` (`TrajectoryOptimizer` already imported; used at `main.py:215`). No behavior change for the stock path.
2. **A** — 3–5 lines: catch `ModuleNotFoundError` for known-optional deps (`jax`, `immrax`, `yaml`) only, re-raise the rest; extend `tests/test_suite_registry.py` with a broken-import fixture.
3. **B** — medium: pass limits via the executor's `initializer`/`initargs` (set env in the worker only); keep the sequential fallback covered per the round-1 caveat (`architect.md:11`).
4. **D** — slowest feedback (CI + upgrade path): stamp a version file on seed, re-copy on mismatch; add the `example_straight --validate` smoke on Windows in `desktop.yml`.

Suggested ownership: C + A in one small diff (any owner, no measurement); B needs a before/after wall-time check on a `--candidates 3` run; D needs a Windows CI cycle.

*Skipped: re-reading `gui_server.py` past the cited kill-path lines and re-running any suite — relied on round-1 reads plus targeted re-verification above. Risk: line numbers drift as the tree moves; re-grep before editing.*
