# TOPIC-02 + TOPIC-03 — Field uncertainty + force model: forum debate

# Forum debate — TOPIC-02 + TOPIC-03 (Ref Rina, channeling Vora / Gus / Sara)

*Sources: `review_forum/synthesis/CLUSTERING.md:9-12`, `review_forum/round1/physics.md`, `review_forum/round1/verification.md`, `review_forum/round1/robotics.md`. Read-only; no tree edits.*

## TOPIC-02 — Field uncertainty: physics rebuttal, V&V defense, resolution

### Gus's gate defense (stands)

A pass near a mission model today is **evidence against the drawing, not the field**. Three facts, all in the tree:

1. The physical channel prices nominal geometry: `util = err / max(raw, 1e-6)` (`overboard/main.py:1666-1671`) while `placement_uncertainty_m` is only **recorded** (`overboard/main.py:1689-1691`) and never subtracted. A nominal 10 mm clearance with ~100 mm placement error passes `util > 1` (`overboard/main.py:1684`) while the true clearance may be −90 mm.
2. Same optimism in the accuracy channel: `_field_limits` clamps the **raw** clearance to 5–50 mm (`overboard/main.py:620-635`), so the per-sample limit fed to `_local_limit` (`overboard/main.py:549-564`) is nominal too.
3. The initial tube is ±1 mm (`overboard/verification/reachability_engine.py:227-236`) against a hand-placed start error that can exceed the whole tube — the prior is unjustified before any propagation runs.

Gus's charge holds: **do not trust recent `--validate` passes near obstacles until this is fixed.**

### Vora's rebuttal (upheld in part — naive deflation is the wrong formula)

Gus's proposed minimal fix — "deflate each raw clearance by `placement_uncertainty_m` … before `util`/clamp" (`review_forum/round1/verification.md:5`) — is directionally right but physically sloppy in two places:

1. **Deflating `raw` into the structural check creates false crashes.** `raw <= 0` is structural today (`overboard/main.py:1649-1656`, with its own `shortcut:` comment admitting the ~50 mm figure is unhandled). If deflation means `raw_eff = raw − u` feeds that check, then a nominal 10 mm clearance with u ≈ 100 mm becomes "the body is inside a model" — **structural, unscorable, never accepted, never exported** (`overboard/main.py:1739-1743`). That is the wrong category: the plan does not commit a crash, the gate commits an unmeasurable. A near-miss must stay a **scorable physical/accuracy failure** (acceptable via `--accept-score`), not a structural one.
2. **`err/(raw−u)` and `(err+u)/raw` share a pass/fail boundary but only one is well-behaved.** Collision is `err + e_place > raw`, worst case `err + u > raw`. The additive form `util = (err + u)/raw` is the physics; the deflated-denominator form blows up to `1e-6`-clamped infinities exactly when `raw ≈ u` — i.e. precisely the marginal cases this topic is about — and flooring a negative `raw_eff` at 0 before the 5 mm clamp manufactures a limit out of a deficit.

Notably the field model already knows the right semantics: sub-uncertainty collisions are **reported but marked `uncertain`, not hits** (`overboard/viz/fields.py:647-656`, `471-473`).

### Resolution — adopted formula (one change, in `overboard/main.py`)

Keep the structural check **nominal-only**; put the uncertainty in the utilisation and the accuracy limit:

```python
# per sample, u = ctx.field.placement_uncertainty_m (0.0 when no field)
raw_eff_acc = raw - u                                  # accuracy-channel input
limit_i     = clamp(raw_eff_acc, 0.005, 0.050) if raw_eff_acc > 0 else 0.005
util_i      = (err + u) / max(raw, 1e-6)               # physical channel, replaces err/max(raw,1e-6)
structural  iff raw <= 0                               # unchanged, nominal penetration only (:1649)
```

Concretely:

- **`_field_limits` (`overboard/main.py:620-635`)**: take `u` as a parameter; clamp `raw − u`, not `raw`. `NaN` still means "no field limit" (fallback to the flat limit per `_local_limit`, `:549-564`).
- **Physical channel (`overboard/main.py:1666-1671,1684`)**: `worst_clearance_util` from `(err + u)/max(raw, 1e-6)`; `worst_penetration` from `max(0, err + u − raw)`. The `> 1.0` threshold (`:1684`) is unchanged — the boundary `err ≤ raw − u` is exactly Gus's, with finite utils and no false structurals.
- **Declared waypoints stay absolute** (Gus's Q, `review_forum/round1/verification.md:7`): `_local_limit` (`overboard/main.py:549-564`) must apply `per_metre` only to the run-wide default fallback, never to a waypoint-declared limit — otherwise a 5 mm `precision_mode` waypoint 4 m in is judged at 25 mm and KI-054's control is silently un-declared. One-line fix at `:560`.
- **Initial tube** (Gus suggestion 3, `review_forum/round1/verification.md:9`): expose the ±1 mm half-width (`reachability_engine.py:227-236`) as a validator parameter defaulting to 0.001, sourced from `placement_uncertainty_m` where a field model exists. No model change; report the residual (PID + quantisation, KI-020) alongside the verdict.

**Verdict TOPIC-02: Gus's gate finding stands; Vora's category objection is sustained. Adopt the additive formula above — same pass/fail boundary as deflation, correct failure category, finite utils.**

## TOPIC-03 — Force-model unconservatism ranking, fix order, owners

Ranked by **unconservatism**: how much badness the current gate lets through, not how often the code path runs. (The traction end-row `shortcut:` is folded in — it is half of the wheel-lift story at solve time.)

| Rank | Finding | Why this rank | Fix (minimal) | Owner |
|------|---------|---------------|---------------|-------|
| 1 — systematic, silent | **CG height hardcoded** (`overboard/models/drivetrain_physics.py:22`, consumed `overboard/models/robot_model.py:829`, `kinematic_cornering_ceiling` default `drivetrain_physics.py:294`, seed `overboard/core/frontier_seed.py:404`) | `v_tip = √(g·b/2h·κ)` scales as `1/√h`: h = 0.10 m tips at **0.7×** the modeled speed on **every** turning sample, and optimizer/seed/validator all agree on the wrong number. Largest exposed population. | Add `h_cg_m` (default 0.05) to `RobotConfig`, thread through `get_wheel_normal_forces` / `kinematic_cornering_ceiling` instead of the import. | Vora (models) |
| 2 — miscategorised tip | **Negative normal → slip misreport** (`drivetrain_physics.py:145-166` returns signed; `traction_limits` `:228-230` gives `cof·negative`; `check_drivetrain_constraints` `:270-272`) | A lifting wheel never raises a lift flag; its excess lands in the slip channel, where an operator can `--accept-score` away a **tip-over as "friction"**. Wrong fix signal + wrong acceptability. | `nl,nr = max(0, raw)` inside the shared function (duck-typed `_duck_clamp_zero` already exists at `:61-91`) **plus** a separate `lift = (raw<0)` boolean the validator gates as hard/structural. Cite `stats["segment_samples"]` layout on every slip/tip number (arc-length vs sample-index, cf. physics Q on `:582`/`_compute_cost:1032`). | Vora (shared physics; validator consumes) |
| 3 — invisible at the gate | **Continuity rows only in `per_segment_dt`** (`overboard/core/optimizer.py:556-568` gated on `if per_segment_dt`; rule lives in `merge_conditions` `:430-446`) | Default `solve(per_segment_dt=False)` leaves `(vl[k+1]−vl[k])/dt` free to jump at fly-through pins — discontinuous force at identical state continuity, exactly what KI-024 forbids. Per-sample validator is blind to the inter-sample jump. | Build the scaled accel-continuity rows in **both** dt modes, or force `per_segment_dt=True` whenever `stop_waypoint_indices` is non-empty. | Optimizer owner (Vora prescribes, planner implements) |
| 4 — optimizer-only, gate catches it | **Traction end-rows dropped as `shortcut:`** (`overboard/core/optimizer.py:742-752`; motor bound does both ends at `:739-740`, epigraph branch does both at `:710-725`) | Admits plans the validator refuses **by construction** on high lateral-transfer segments (`v·ω` unloads inner wheel mid-interval). Wastes solves; does not export bad plans — which is why it ranks below 1–3. | Enforce `f_bound <= trac_end` like the motor limit; keep the `shortcut:` comment as the record of the 112→3145-iteration blowup, or gate the claim ("certified domain: low lateral-transfer only"). One constraint row. | Optimizer owner |
| 5 — rare, bounded, firmware-side | **Limiter uses wall-clock `dt`** (`pybricks_code/ramsete.py:299-301` → `KinematicLimiter.limit` `:116-128`) vs validator's exact 20 ms hold (`immrax_validator.py:45`, `CONTROL_DT`; `ramsete.py:18,326-336` heartbeat) | A GC-pause/overrun tick computes a *smaller* required accel and under-constrains exactly the longest-held tick — the one wheel-force error is proportional to. Real hardware slip risk, but bounded, rare, and outside the gate (validator models the nominal correctly). | Feed the limiter nominal `CONTROL_MS` (or clamp measured `dt` to ≤1.5× nominal). Firmware-only; validator unchanged. | Sara (firmware) |

**Fix order (cheapest safety first, respecting rank): 1 → 2 → 4 → 3 → 5.** Rationale: 1 and 2 are a few lines in one shared module and remove the two ways the gate currently blesses a tipper; 4 is one constraint row that ends the optimizer/validator disagreement; 3 needs NLP care (the iteration-blowup history behind the `shortcut:`); 5 is firmware deployment, decoupled from the gate.

**Status asked of the forum: TOPIC-03 stays OPEN until fixes 1+2 land; fixes 3–5 are tracked follow-ups with the owners above.**

*Skipped: re-deriving GL4 quadrature, re-running the 32-mission suite, RQ-A notebooks, studio push path (read-only session). Risk: until TOPIC-02's formula and TOPIC-03 fixes 1–2 land, treat near-obstacle passes and tall/fast turning validations as undecided — the gate is optimistic exactly there.*
