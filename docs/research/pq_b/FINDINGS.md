# PQ-B — the research agenda, measured

**Status:** the measurement campaign is complete. The generated tables are in
[`MEASUREMENTS_matrix.md`](MEASUREMENTS_matrix.md) (and the per-study files
next to it); the raw records are in
[`../../measurements/2026-10-20/`](../../measurements/2026-10-20/).

**Tree under test:** pinned, see [`pq_b_PINNED_TREE.md5`](pq_b_PINNED_TREE.md5).

**Companions:** [`../HEURISTICS_AND_VERIFICATION.md`](../HEURISTICS_AND_VERIFICATION.md)
(the agenda this answers) and [`../RQ-A_tracking_error.md`](../RQ-A_tracking_error.md)
(why the tracking gate fails).

---

## 1. Why this exists

Every claim in the agenda rested on **six hand-written example paths**. The
random benchmark that was supposed to supply a mission suite was broken
(KI-015), so "the baseline seed always wins", "the tracking gate is the only
binding constraint" and "the envelope gate never binds" were statements about
six paths.

This campaign measures the same things on **32 missions** (the 8 example
missions plus 24 generated ones), through the **production** gate chain, with
**one JSON record per candidate**.

## 2. Method

| | |
|---|---|
| Missions | 8 example missions + 24 generated (`research/suite.py`, seed 7), stratified over four profiles: `even`, `uneven`, `turny`, `mixed` |
| Candidates | 21 per mission: 5 seeds x 2 timing models at the base weight, plus the baseline x 2 timing models x 5 accuracy weights, plus the two new guess families (Bezier path family, length-proportional time allocation) |
| Total | 672 candidate solves, each solved and then scored by the full gate chain |
| Scoring | the production `main.py` helpers (`_immrax_failure_reasons`, `_cpu_failure_reasons`) and the production Immrax validator and CPU audit |
| Robot | `examples/fll_choreo.json`, 10 samples per segment |
| Tree | pinned copy, hashes in `pq_b_PINNED_TREE.md5`; every record carries a `tree` token, and every record in this campaign carries the same one |

### The instrument was checked first

`research/check_instrument.py` runs the CLI and the harness on the same
mission, against the same tree, and compares what they report:

```text
CLI:      baseline/equal-duration 3.6965s env 0.7650cm trk 1.3197cm ok=False
Harness:  baseline/equal-duration 3.6965s env 0.7650cm trk 1.3197cm ok=False
```

Every duration, envelope, tracking error and pass flag agrees to six
decimals. The harness is a bookkeeper around the production code, not a
re-implementation of it.

### Why the tree is pinned

The working tree was edited while the campaign ran (KI-024 landed mid-flight
once, and KI-025 after the pin). The first attempt at this work produced a
mixed dataset; it was thrown away and re-run against a pinned copy. The
`tree` token in every record makes that check automatic.

**The pin predates KI-025.** The pinned `immrax_validator.py` still has the
old backlash model (a velocity offset) that KI-025 replaced with a position
uncertainty. RQ-A measured that term at ~1 % of the tracking error on curved
paths and ~226 µm on straights, so the effect on the conclusions below is
small — but the numbers in this document are the pinned tree's numbers, not
the working tree's.

---

## 3. What the 32-mission matrix says

### F1 — The baseline's dominance was an artefact of the six example paths (Q8)

On the six example missions the `baseline` seed produced the fastest
trajectory on five of them, and the agenda recorded that as a property of the
planner. On 32 missions it is not true:

| seed | timing | pass rate (95 % CI) | median duration |
|---|---|---|---|
| baseline | equal-duration | **16 %** (7-32 %) | 6.13 s |
| baseline | per-segment | **19 %** (9-35 %) | 5.00 s |
| bow_left | equal-duration | 25 % (13-42 %) | 6.21 s |
| bow_left | per-segment | 25 % (13-42 %) | 5.02 s |
| bow_right | equal-duration | 25 % (13-42 %) | 6.21 s |
| bow_right | per-segment | 19 % (9-35 %) | 4.87 s |
| rolling | equal-duration | **28 %** (16-45 %) | 6.21 s |
| rolling | per-segment | 19 % (9-35 %) | 5.00 s |
| turn_first | equal-duration | 19 % (9-28 %) | 6.23 s |
| turn_first | per-segment | 12 % (5-28 %) | 5.43 s |

The baseline has the **lowest pass rate of the five seeds**. It is also the
fastest seed more often than any other — but those two facts do not combine.
The fastest candidate per mission:

| seed | times it produced the fastest candidate |
|---|---:|
| baseline (9 per-segment, 2 equal-duration) | 11/32 |
| rolling | 8/32 |
| bow_right | 6/32 |
| bow_left | 4/32 |
| turn_first | 3/32 |

So the baseline is the speed winner *and* the gate loser. It produces the
quickest trajectory on a third of the missions and then fails the gates on
them. On the six example paths, where the gate happened to be forgiving, that
looked like dominance.

**Answer to Q8:** the baseline is not universally dominant; it is *locally*
dominant on the paths it was designed next to. The agenda's "seeds never win
on speed" is wrong in general and right only for the six paths it was
measured on; the sharper statement is that the seeds differ in **which gate
they escape**, not in speed.

### F2 — The envelope gate does bind, on a wider suite

The agenda reported that the 2 cm reachability-envelope gate **never** rejected
a candidate (0 of 37 failures) and concluded that the interval machinery was
over-engineered for the system's constraints. Over 672 candidates:

| failure reason | candidates | share |
|---|---:|---:|
| forward integration above 1 cm | 465 | 69 % |
| tracking error above 5 mm | 458 | 68 % |
| **envelope radius above the local threshold** | **308** | **46 %** |
| CPU constraint violations | 113 | 17 % |
| motor force over the limit | 81 | 12 % |

The largest envelope measured is **5.26 cm** against a 2 cm limit (the six
example missions never exceeded 1.76 cm). Only 411 of 672 candidates stay
inside the envelope budget.

**Correction:** the envelope is not a decorative gate. It binds on nearly half
of the candidates once the mission set is wider. It is still *not* the
binding constraint — the tracking gate is — but the claim "the envelope never
binds" was an artefact of the same six paths.

### F3 — The tracking gate's driver is the discretisation, confirmed at scale (RQ-A)

RQ-A found that the tracking error is a discretisation artefact and that the
heading change per sample is the variable that predicts it. Measured over all
672 candidates:

```text
Spearman(mean heading change per sample, tracking error) = 0.615
```

| mean heading change per sample | candidates | median tracking | pass rate |
|---|---:|---:|---:|
| 0.00-0.02 rad | 42 | 0.37 cm | 83 % |
| 0.02-0.05 rad | 20 | 0.41 cm | 80 % |
| 0.05-0.10 rad | 50 | 0.53 cm | 38 % |
| 0.10-0.20 rad | 278 | 0.56 cm | 24 % |
| > 0.20 rad | 282 | 1.45 cm | **5 %** |

The relationship is monotone and steep: the pass rate falls from 83 % to 5 %
as the path turns harder between samples, and the median tracking error grows
fourfold. This is RQ-A's mechanism on five times as many candidates: **the
tracking gate measures how much the plan turns between samples, not whether a
robot can follow the plan.** Every conclusion in this document that mentions
the tracking gate inherits that caveat.

### F4 — The timing model is a bigger lever than any seed

Per-segment timing is faster than equal-duration timing on **29 of 32**
missions (slower on 2, tied on 1), by a median of **21 %**. On the six example
missions the agenda could only claim "faster on three of six".

The equal-duration model is not a peer; it is a legacy special case that
forces every segment to take the same time.

### F5 — Multi-start is insurance, and its price is now measurable (Q3)

The candidates are added in the order the CLI adds them. The table says how
many of the 32 missions have a passing candidate after each addition:

| candidates | missions with a passing candidate | mean best duration |
|---|---:|---:|
| 1 — baseline / equal-duration | 5/32 | 3.30 s |
| 2 — + baseline / per-segment | 8/32 | 3.03 s |
| 3 — + rolling / equal-duration | **11/32** | 3.39 s |
| 6 — + bow_left / per-segment | 12/32 | 3.13 s |
| 16 — + the whole accuracy-weight grid | 12/32 | 3.13 s |
| 17 — + baseline / equal-duration at a=0.01 | 13/32 | 3.36 s |
| 18 — the full candidate set | 13/32 | 3.36 s |

**One extra seed is worth three missions.** Candidates 4 to 16 buy one more
mission between them; the accuracy-weight grid buys exactly one. The baseline
alone passes on 8 of 32 missions (25 %), and a non-baseline candidate rescues
5 more (16 %).

**19 of 32 missions (59 %) produce no passing trajectory at all**, even with
the full 21-candidate set.

### F6 — The selection rule does not matter (Q10)

Five selection rules were replayed over every mission's candidate list:

| rule | missions with a passing pick | mean duration | mean envelope | mean tracking |
|---|---:|---:|---:|---:|
| current (smallest envelope) | 13/32 | 3.361 s | 0.56 cm | 0.42 cm |
| fastest passing | 13/32 | 3.361 s | 0.56 cm | 0.42 cm |
| smallest tracking margin | 13/32 | 3.361 s | 0.56 cm | 0.42 cm |
| smallest envelope margin | 13/32 | 3.361 s | 0.56 cm | 0.42 cm |
| fewest failure reasons | 13/32 | 3.361 s | 0.56 cm | 0.42 cm |

All five pick the same trajectory. The envelope and the duration are so
strongly correlated among the passing candidates that "smallest envelope" and
"fastest" are the same rule in practice.

**Answer to Q10:** the selection rule is not a lever. Q10's hypothesis — that
minimising the envelope prefers slow, over-cautious trajectories — is not
observable on this suite. A different rule would not make any mission solvable
or any trajectory faster.

### F7 — The critic cannot select anything (Q5)

The Multi-Verse critic flagged **152 of 152** passing candidates and **520 of
520** failing ones. Its precision (77 %) is simply the base rate of failure;
its recall is 100 % because it flags everything.

**Answer to Q5:** feeding the critic back into the solver cannot work with the
thresholds as they are. The critic would send every window back, including the
good ones. Before it can be a filter, its thresholds have to discriminate.

### F8 — A real path family is not a speed lever (Q1)

A cubic Bezier path family was built (`research/seeds.py`) whose endpoint
tangents follow the waypoint headings, with rolling speeds consistent with
the geometry, and handed to the solver through the existing `initial_guess`
argument. Matched against the straight-line seed on the same mission, timing
model and weight (64 pairs):

```text
spline faster on 26, slower on 37, tied on 1
duration delta: median 0.0 %, mean +3.0 %
IPOPT iterations: spline median 33, baseline median 48
spline passes on 15, baseline passes on 11
```

The path family makes the **solve** cheaper (33 iterations against 48) and
produces **more passing trajectories** (15 against 11), but it produces
**slower trajectories** on the majority of missions.

**Answer to Q1:** a curvature-continuous seed is not a speed lever, on this
objective. It is a slightly better gate-insurance ticket. The mechanism is
exactly the one the agenda predicted (lesson 23.1): the objective has no term
that prefers a curve, so the solver walks the curve back towards the straight
line — and it does so from a better starting point, which is why the solve is
cheaper and the result is not faster.

### F9 — A length-proportional time allocation finds better optima, not faster solves (Q2)

The same mechanism, applied to the *time* axis instead of the geometry: the
initial `dt_i` is proportional to the segment length instead of a flat 0.1 s.
Matched against the same geometry with the flat allocation (32 pairs):

```text
proportional faster on 18, slower on 11, tied on 3
duration delta: median -0.0 %, mean -1.8 %
IPOPT iterations: proportional median 40, rolling median 25
solve wall time:  proportional median 0.13 s, rolling median 0.05 s
passes: 6 against 6
```

**Answer to Q2:** the hypothesis is refuted in its own terms — the
proportional allocation does **not** converge faster; it converges *slower*
(40 iterations against 25, 2.6x the wall time) because it starts further from
the flat-`dt` optimum. But it lands on a **faster trajectory on 18 of 32
missions** with no loss of gate coverage. The initial time allocation changes
*which* local optimum is found, not how fast the solver gets there. This is
the only lever in the campaign that changed the objective's outcome rather
than the gate outcome, and it is worth 1.8 % on average for one extra solve.

### F10 — 7 % of the plans ask for more acceleration than the hub can deliver (Q11, partially)

The hub's DriveBase ramps its trajectories at about 7.8 m/s^2 for this robot's
geometry (KI-021). The optimizer does not model that limit, and the validator's
plant assumes the commanded wheel speed becomes motion immediately.

| | |
|---|---|
| peak wheel acceleration, median | 3.66 m/s^2 |
| peak, maximum | 10.70 m/s^2 |
| candidates above the 7.8 m/s^2 limit | 49/672 (7 %) |
| **passing** candidates above the limit | **12/152 (8 %)** |

**8 % of the trajectories this system calls safe are not executable by the
hub as configured.** That is a validation-model gap (the plant's instant
tracking assumption), not a planner bug, and it is exactly the class of thing
Q11 asks about. It is cheap to close: the validator's plant can ramp the
commanded wheel speed at the DriveBase limit.

### F11 — The envelope is a function of the duration (Q4, partially)

Over the 152 passing candidates:

| predictor of the envelope radius | Pearson | Spearman |
|---|---|---|
| trajectory duration | **0.898** | **0.873** |
| path length | 0.736 | 0.768 |

The uncertainty accumulates with time, so the envelope is essentially a
function of how long the trajectory takes. A cheap proxy for the envelope
therefore exists and does not need interval arithmetic: it needs the
duration. **Answer to Q4, half of it:** the proxy is easy; whether adding it
to the objective helps is a separate experiment, and per RQ-A it would not
touch the failures that actually block missions today (they fail on tracking,
which is a discretisation artefact).

---

## 4. The accuracy weight: a real dial with a lottery attached (Q6)

A 9-point weight ladder (0, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
0.05) was run on 12 missions under both timing models (216 candidates).

**Pooled across missions the relationship is invisible** — Spearman(weight,
tracking) = -0.174, Pearson -0.013 — because the missions differ more from
each other than the weight moves a mission.

**Within a mission it is strong and in the expected direction**: the median
per-mission Spearman is **-0.683**, and 20 of 24 missions are negative (a
higher weight gives a lower tracking error).

| weight | equal-duration: pass / duration / tracking | per-segment: pass / duration / tracking |
|---:|---|---|
| 0 | 25 % / 6.19 s / 0.71 cm | 17 % / 5.14 s / 1.14 cm |
| 0.00025 | 25 % / 6.19 s / 0.78 cm | 17 % / 4.76 s / 1.09 cm |
| 0.0005 | 17 % / 6.19 s / 0.79 cm | 17 % / 5.28 s / 1.34 cm |
| 0.001 | 33 % / 7.05 s / 0.74 cm | 17 % / 5.31 s / 0.67 cm |
| 0.002 | 33 % / 7.08 s / 0.72 cm | 33 % / 5.26 s / 0.64 cm |
| 0.005 | 33 % / 7.21 s / 0.69 cm | **42 %** / 5.38 s / 0.77 cm |
| 0.01 | 42 % / 7.92 s / 0.50 cm | 33 % / 5.50 s / 0.70 cm |
| 0.02 | **50 %** / 8.04 s / 0.53 cm | 33 % / 5.66 s / 0.67 cm |
| 0.05 | 42 % / 6.91 s / 0.52 cm | 33 % / 8.34 s / 0.61 cm |

**Answer to Q6: the weight is a real accuracy proxy, but it is a search, not a
dial.**

- *The proxy is real*: more weight buys less tracking error, monotonically in
  the median, inside a mission.
- *The trade is real*: the duration grows with the weight. The best trade in
  the whole table is per-segment at `a = 0.005`: **42 % pass against 17 % at
  weight 0, for 5 % more time** (5.38 s against 5.14 s).
- *The lottery is real*: **21 of 192** steps are non-monotone — a higher
  weight gave a slower **and** less accurate trajectory — and the weight that
  produced the fastest *passing* trajectory differs per mission (0, 0.00025,
  0.0005, 0.002 and 0.01 all occur on the twelve missions).

The CLI's five-point grid (0, 0.001, 0.002, 0.005, 0.01) covers the useful
range: the pass rate peaks around 0.005-0.02 and falls again by 0.05, where
the duration jumps to 8.34 s under per-segment timing. Extending the grid
further is not worth the solves.

Note the interaction: under per-segment timing the weight buys pass rate
almost for free (17 % -> 42 % for 0.24 s), while under equal-duration timing
it costs 2 s for the same gain. **The weight and the timing model are not
independent axes**, which the agenda's candidate set treats them as.

---

## 5. The Multi-Verse pipeline reproduced the simple solver and cost 3-15x the time (Q12) — before it was repaired

The Multi-Verse path was run on 12 missions and compared with the simple
solver's baseline on the same missions and the same tree:

| | |
|---|---|
| missions where the Multi-Verse duration equals the simple solver's | **11 of 12** (to three decimals) |
| missions where Multi-Verse is faster | 1 (m000, -3.2 %) |
| passing | simple 3/12, Multi-Verse 4/12 (the extra one is m000) |
| wall time per solve | 2.5-28 s against 0.07-30 s; typically **3-15x** the simple solve |

At the time, the reason was KI-004: the pipeline's phase 5 re-solved from the
default baseline guess, and the JAX-refined probe never won. The one mission
where it differed was the one where the probe did win.

**That measurement is the record of why the pipeline was worth repairing, and
the verdict it produced has since been overtaken.** The pipeline was not
deleted; it was **repaired** (KI-004): the refinement no longer moves the
waypoints, the JAX cost uses IPOPT's trapezoidal rule, the intervals follow
the segment's own length, the refined timing is propagated into the
timestamps, the probe reads the guess with the per-segment model, and the
Reeds-Shepp bootstrap works (its rsplan API was broken, so the bootstrap had
never run). With those fixed the probe **wins** on the complex mission —
4.617 s against 6.049 s — and `main.py --refine` puts the refined trajectory
into the candidate set: 6.068 s becomes **4.936 s** there, and the trajectory
passes the gates (envelope 1.14 cm, tracking 0.45 cm).

**Answer to Q12 (updated): the pipeline is a seed provider, not a second
solver.** The measurement above stands — a seed provider that reproduces the
trivial answer at 3-15x the wall time is exactly what must not happen — but
"delete it or stop funding it" is superseded. What the pipeline is for now is
the topology sampling (forward / reverse / point-turn) that nothing else in
the tree covers. On the complex mission the refined seed gives **4.936 s**
where the default two-candidate set gives 6.068 s — but the full ten-candidate
stock set gives 4.877 s there, so the refined seed is a **cheaper way to reach
the stock set's speed, not a better one**, and over 30 missions it rescues
nothing. It stays opt-in (`--refine`). The open part is the objective: the
pipeline's cost is duration plus a jerk term, so it buys speed with tracking
accuracy, and the 5 mm gate rejects the trade at the higher weights.

---

## 6. How much of a PASS depends on the declared uncertainty (KI-020)

Every declared uncertainty range was scaled around its nominal value and the
same candidates were re-validated at each width (9 missions x 10 candidates
x 5 widths = 450 validations of the same trajectories):

| spread of the declared ranges | pass rate | median envelope | median tracking |
|---|---:|---:|---:|
| 0.0 — collapsed to the nominal value | 32 % | 0.43 cm | 0.59 cm |
| 0.5 — half the declared width | 32 % | 0.60 cm | 0.60 cm |
| 1.0 — **as declared** | 31 % | 0.79 cm | 0.60 cm |
| 1.5 — 1.5x the declared width | **2 %** | 0.99 cm | 0.60 cm |
| 2.0 — double the declared width | **0 %** | 1.18 cm | 0.60 cm |

Three things fall out of this table.

1. **The tracking error is completely insensitive to the declared
   uncertainty** (0.59-0.60 cm at every width). RQ-A found this on five
   missions; here it is confirmed on 450 validations. The tracking gate is a
   deterministic check, not a robustness check.
2. **The envelope is the uncertainty-driven quantity**: it grows linearly with
   the spread (0.43 -> 1.18 cm), and it is what the spread breaks.
3. **The pass rate is a cliff, and the declared ranges sit on its edge.**
   Collapsing the ranges to their nominal values or halving them changes
   nothing (32 % -> 31 %), but widening them by half takes the pass rate from
   31 % to **2 %**, and doubling them takes it to **0 %**.

**This is the single most important caveat in this document.** The declared
ranges in `examples/fll_choreo.json` are test assumptions (KI-020), and the
system's verdicts are extremely sensitive to them: a 50 % wider range than
declared would make almost every trajectory fail. A PASS today is partly a
statement about the assumptions, not only about the trajectory.

---

## 7. RQ-A's OQ6: the seed conclusions under the discretisation fix

KI-026 added `--self-consistent`: the samples are placed on the path the
plan's own `(v, omega)` profile traces. The whole seed matrix was re-run on
**the same pinned tree**, with and without the flag, on 16 missions x 10
candidates (320 solves).

| | trapezoidal | self-consistent |
|---|---:|---:|
| passing candidates | 33/160 | **50/160** |
| pass flags that flip | 5 lose, 22 gain, 28 unchanged | |
| duration, median change | — | **+0.2 %** (range -27.5 % to +40.9 %) |

The discretisation fix buys **52 % more gate coverage for no time at all**.
And it changes the seed conclusions:

| candidates added | trapezoidal | self-consistent |
|---|---:|---:|
| 1 — baseline / equal-duration | 3/16 | 4/16 |
| 2 — + baseline / per-segment | 4/16 | **6/16** |
| 3 — + rolling / equal-duration | **6/16** | 6/16 |
| 4-10 — the other four seeds | 6/16 | **6/16** |
| missions a non-baseline seed rescued | 2/16 | **0/16** |

**Answer to RQ-A's OQ6: yes, it changes them, and the change is decisive.**
Under the self-consistent collocation the two default candidates cover
everything the seeds used to cover, and the seed set buys **nothing**: no
extra mission, and the median duration improves by only 0.2 % across eight
extra solves.

**The seeds' whole value was escaping a gate that was measuring a
discretisation artefact.** Fix the discretisation and the insurance
disappears. That is the strongest argument in this document for finishing
KI-026 before touching the candidate set.

---

## 8. The answers, question by question

| # | question | answer | evidence |
|---|---|---|---|
| Q1 | curvature-continuous path family? | **No as a speed lever** (mean +3.0 % slower), yes as a cheaper solve (33 vs 48 iterations) and slightly better gate coverage (15 vs 11 passing) | F8 |
| Q2 | time-allocation warm start? | **Refuted as stated** (slower convergence: 40 vs 25 iterations), but the proportional allocation finds a faster trajectory on 18 of 32 missions (mean -1.8 %) | F9 |
| Q3 | is multi-start worth anything? | **It is insurance, and now priced**: one extra seed is worth 3 missions; candidates 4-16 buy 1 mission; the baseline alone passes on 8/32, a non-baseline rescues 5 more | F5 |
| Q4 | reachability as an objective term? | **The cheap proxy exists** (duration explains the envelope with r = 0.90) but it is not the binding constraint; the envelope does bind on 46 % of candidates, yet the failures that block missions are tracking failures | F2, F11 |
| Q5 | feed the critic back into the solver? | **No**: the critic flags 100 % of passing and 100 % of failing candidates, so it cannot select anything | F7 |
| Q6 | does the accuracy weight buy accuracy? | **Yes, within a mission** (median per-mission Spearman -0.68: more weight, less tracking error) **but it is a trade, not a dial** (21 of 192 steps are non-monotone, and the best weight differs per mission). The best trade found: per-segment at `a = 0.005` gives 42 % pass against 17 % at weight 0, for 5 % more time | section 4 |
| Q7 | sample density against timing flexibility? | **Per-segment timing at low density wins.** n = 10 is the sweet spot: more samples cost 5x the solve time and do not improve the tracking error (0.60 cm at n = 10, 0.72 cm at n = 40) and can lose gate coverage (n = 20 passes on 0 of 6 missions) | `MEASUREMENTS_density.md` |
| Q8 | is the baseline's dominance an artefact? | **Yes.** The baseline is the speed winner (11 of 32 missions) and the gate loser (lowest pass rate of the five seeds) — on the six example paths those two roles did not conflict. And under the self-consistent collocation the seed set buys nothing at all | F1, section 7 |
| Q9 | what would make the validator a digital twin? | **Not answerable without hardware.** What *is* answerable: the tracking error is a discretisation artefact (RQ-A), the declared uncertainty does not move it at all (section 6), and the plant still assumes instant tracking, so 8 % of passing plans exceed the hub's acceleration limit | F3, F10, section 6 |
| Q10 | is "smallest envelope" the right selection rule? | **It does not matter**: five rules pick the same trajectory on every mission | F6 |
| Q11 | should the executor be part of the planner? | **Partly answerable now**: 7 % of candidates, and 8 % of the passing ones, demand more acceleration than the DriveBase delivers. The gate cannot see it | F10 |
| Q12 | what should the Multi-Verse path become? | **Repaired and kept, as a seed provider** (KI-004). `main.py --refine` puts the refined trajectory into the candidate set: 4.936 s against the default set's 6.068 s on the complex mission — but the full ten-candidate stock set gives 4.877 s there, and over 30 missions the refined seed rescues nothing. Cheaper, not better; it stays opt-in | section 5 |

---

## 9. What this changes in the agenda

1. **The six-mission base has to go.** Three of the agenda's headline claims
   (the baseline dominates, the envelope never binds, the seeds never win) do
   not survive a 32-mission suite. Every future claim needs a mission suite
   and a stated n.
2. **The seed ordering should be revisited, but only after KI-026.**
   `rolling` has the highest pass rate of the five seeds on the wide suite
   (28 % against the baseline's 16 %), and the baseline is both the speed
   winner and the gate loser. The seeds are not interchangeable lottery
   tickets: they differ. But under the self-consistent collocation the whole
   seed axis is worth nothing (section 7), so this is second priority.
3. **But do not spend effort there yet.** Under the self-consistent
   collocation the seed set buys nothing at all (section 7). The
   discretisation fix is worth more than the whole seed axis.
4. **The selection rule is not worth changing** (F6). Effort belongs elsewhere.
5. **The critic needs thresholds that discriminate** before it can be used as
   a filter (F7).
6. **The discretisation must be fixed before the tracking gate means
   anything** (RQ-A, KI-026). Until then, "the tracking gate binds" is a
   statement about the plan's discretisation, and every measurement above
   inherits it.
7. **The plant's instant-tracking assumption hides a real gap** (F10): 8 % of
   the trajectories the system calls safe cannot be executed by the hub.
8. **The declared uncertainty ranges are load-bearing** (section 6). A 50 %
   wider range than declared takes the pass rate from 31 % to 2 %. Until the
   robot is calibrated, the pass rate is a statement about the assumptions.

## 10. What is still not answerable here

| | |
|---|---|
| Q9 (a real digital twin) | needs hardware. No logs exist in the repository. |
| The hardware truth of any number here | every number is a model prediction; the declared uncertainty ranges are test assumptions, and the pass rate is very sensitive to them (section 6) |
| Whether the discretisation fix changes the seed conclusions | **answered** (section 7): it does, decisively |
| Whether the seeds are worth keeping at all | answered for the self-consistent collocation (they are not, on this sample); not answered for the trapezoidal one after the other KI-02x fixes land |

## 11. The repairs made along the way

| defect | repair |
|---|---|
| KI-015: the random suite generator read a `robot_config.json` that does not exist and treated `solve()`'s `(samples, stats)` tuple as the samples | `examples/generate_random_suite.py` rewritten on top of the new generator; it unpacks the tuple and reports the convergence status |
| KI-015: the random benchmark drew waypoints uniformly, so two waypoints often landed on top of each other and the mission was unsolvable for reasons that had nothing to do with the planner | `benchmark.py`'s random mode now generates a random walk inside the profile's field |
| KI-015: the pipeline benchmark reported failed solves as normal results | it now reads `stats["converged"]` and prints FAILED |
| KI-015: the default `--config` was a file that does not exist | it now defaults to `examples/fll_choreo.json` |

## 12. The honesty clause

Every number in this document is a **model prediction**. The declared
uncertainty ranges in `examples/fll_choreo.json` are test assumptions, not
measurements, and no hardware logs exist in the repository. A `PASS` means
"the trajectory passes the gates of an uncalibrated model", not "the robot
can drive it". The tracking numbers additionally measure a discretisation
artefact (RQ-A), not a physical tracking error.

