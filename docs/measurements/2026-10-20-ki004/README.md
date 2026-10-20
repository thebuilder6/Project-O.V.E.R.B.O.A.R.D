# Measurement evidence — 2026-10-20, the Multi-Verse repair (KI-004)

The Multi-Verse pipeline's refined guess used to lose to the trivial baseline
guess (8.1473 s against 6.0677 s on the complex mission), so the pipeline's
result was discarded. This folder records the repair and the measurement that
decided what the pipeline is worth afterwards.

## What was broken

Four separate defects, each found by measurement and each necessary to fix:

1. **The refinement clobbered the waypoints.** The critic reports two-segment
   windows and the refinement optimised the whole window, so the waypoint in
   its middle was moved. A refined trajectory that does not pass through the
   waypoints is not a starting point for a problem that pins them.
2. **The cost graded a different motion than the solver produces.**
   `local_trajectory_cost` integrated from the left endpoint; IPOPT uses the
   trapezoidal rule over the interval.
3. **The intervals and the velocities were hard-coded** at 0.1 s and 0.1 m/s,
   whatever the segment's length or the robot's speed.
4. **The refined timing was discarded.** The timestamps were not updated, and
   the probe read the guess with the equal-duration model, which keeps only
   the **first** interval and re-reads every sample against it.

Two more defects turned up on the way:

- `path_planning.py` read rsplan's `Path` object as a list; the `TypeError`
  was swallowed and **every** call returned `None`, so the "Reeds-Shepp
  bootstrap" had never run. The shipped config compounded it with
  `turning_radius: 0.0`, which rsplan cannot plan with at all (0 of 12
  example segments). With the API fixed and the radius at the robot's 0.05 m,
  12 of 12 segments plan.
- The STOMP perturbations were seeded from the clock, so the same mission
  came out at 4.62 s on one run and 5.50 s on the next.

## What the repair is worth

| | before | after |
|---|---:|---:|
| the probe on `example_complex_mission.json` | 8.1473 s, loses | **4.617 s, wins** (the primary solve: 6.049 s) |
| the Reeds-Shepp bootstrap | never ran (0 of 12 segments) | 12 of 12 segments plan |
| the same mission twice | 4.62 s and 5.50 s | the same every time |

## The seed provider, measured

The pipeline was re-envisioned as a **seed provider**: phases 1 to 4 hand the
real solver a trajectory that passes through every waypoint and carries a
per-segment interval, and the solver does the magnitudes. `main.py --refine`
adds it to the candidate set.

Over 30 missions on one tree, the refined family against the stock seeds:

| | stock seeds | refined family |
|---|---:|---:|
| candidates | 300 | 150 |
| passing | 69 (23 %) | 29 (19 %) |
| missions with a passing candidate | **11 of 30** | 9 of 30 |
| missions only this family passes | - | **0** |
| missions only the stock seeds pass | **2** | - |

On the nine missions where both have a passing candidate, the refined family
is faster on **two** and slower on six.

**So the refined seed is not a better guess and it rescues nothing.** It is a
*cheaper* way to reach the stock set's speed on some missions: on the complex
mission it gives 4.936 s where the default two-candidate set gives 6.068 s
and the full ten-candidate stock set gives 4.877 s. It stays opt-in.

The Reeds-Shepp bootstrap on its own is not a speed lever either: on the six
example missions it matches the stock seeds (first4 1.680 s, events 4.418 s)
and is worse on the complex mission (7.032 s against 4.877 s).

## The next lever is the objective, not the search

The pipeline's cost is duration plus a jerk term, so it buys speed with
tracking accuracy, and the 5 mm gate rejects the trade. The refined candidate
passes the complex mission only at an accuracy weight of 0.001, and the
S-curve and the stops mission fail for it as they fail for everything else.

## The files

| File | What it is |
|---|---|
| `refined_30_missions.jsonl` | the refined family, 30 missions x 5 accuracy weights |
| `stock_seeds_30_missions.jsonl` | the stock seeds, the same 30 missions, the same tree |
| `reeds_shepp_seed_6_missions.jsonl` | the Reeds-Shepp bootstrap as a seed, on the six example missions |

## How it was measured

```powershell
$env:OVERBOARD_TREE = "<pinned copy of the tree>"
python research/harness.py --missions research/missions examples/example_*.json `
    --out research/results/mv_refined_wide.jsonl --specs refined-weights --samples 10 --workers 6
python research/harness.py --missions research/missions examples/example_*.json `
    --out research/results/mv_seeds_wide.jsonl --specs seeds --samples 10 --workers 6
```

The tree is pinned because it was edited during the first attempt, and every
record carries a `tree` token so a mixed run is detectable rather than
silently wrong. Both runs above used the same pinned copy.

## Caveat

Model predictions, not hardware measurements. No hardware logs exist in this
repository, and the declared uncertainty ranges in `examples/fll_choreo.json`
are test assumptions rather than calibrated values.
