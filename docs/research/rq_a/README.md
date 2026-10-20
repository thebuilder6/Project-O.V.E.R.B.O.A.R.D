# RQ-A experiment scripts

The measurements behind [`../RQ-A_tracking_error.md`](../RQ-A_tracking_error.md).

Each script is standalone and prints its own table. They import the repository's
own `optimizer`, `immrax_validator`, `jax_robot_model` and `robot_model`, so
they measure the shipped code rather than a re-implementation of it.

```bash
# from the repository root
python docs/research/rq_a/exp9_summary.py
python docs/research/rq_a/exp1_uncertainty.py
python docs/research/rq_a/exp2_reference_artefact.py
python docs/research/rq_a/exp3_sampling.py
python docs/research/rq_a/exp4_decompose.py
python docs/research/rq_a/exp5_attribution.py
python docs/research/rq_a/exp6_consistency.py
python docs/research/rq_a/exp7_recipe.py
python docs/research/rq_a/exp8_proposal.py
python docs/research/rq_a/check_hermite.py
python docs/research/rq_a/verify_backlash_fix.py
python docs/research/rq_a/exp10_floor.py
python docs/research/rq_a/exp10_gain_band.py
python docs/research/rq_a/exp11_density.py
python docs/research/rq_a/exp12_reference_consistency.py
python docs/research/rq_a/exp13_package.py
python docs/research/rq_a/exp14_which_inconsistency.py
python docs/research/rq_a/exp15_interval_average.py
python docs/research/rq_a/exp16_firmware_parity.py
python docs/research/rq_a/exp16_combined.py
```

Requirements: `casadi`, `jax`, `immrax`, `numpy`. The scripts write only to a
temporary directory; nothing in the repository is modified.

The output of the 2026-10-20 run against the tree that carries KI-025 to
KI-028 is in
[`../../measurements/2026-10-20-post-ki028/rq_a/`](../../measurements/2026-10-20-post-ki028/rq_a/),
one file per script.

## Which script answers what

| script | question | headline result |
|---|---|---|
| `exp1_uncertainty.py` | which declared uncertainty drives the tracking error? | none of them — collapsing all seven moves it by 0.2–0.7 % |
| `exp2_reference_artefact.py` | how far does a robot that follows the reference exactly end up from the plan? | 78 % of the S-curve's tracking error |
| `exp3_sampling.py` | does more sampling help? | the plan/profile deviation falls as 1/n², and the tracking error follows it down to n=20 |
| `exp4_decompose.py` | quantisation, backlash, spline reference? | quantisation 0 %, backlash 0 % of the tracking error (it only widens the envelope), spline reference catastrophic |
| `exp5_attribution.py` | what does a *self-consistent* trajectory score? | 0.13 cm at every sampling density — the validator is not the problem |
| `exp6_consistency.py` | is the optimizer's plan self-consistent? | no: rebuilding the plan's positions by exact integration removes 40 % of the S-curve's error |
| `exp7_recipe.py` | does exact positions + density fix it? | only at n=10 (40 %); it gets 2–6 % worse at n=20–80 |
| `exp8_proposal.py` | the two halves of the proposed fix | neither works alone; on the stops mission the spline half is the better one |
| `exp9_summary.py` | the summary table in the notes | the headline numbers |
| `check_hermite.py` | is the spline interpolant correct? | yes: 0.0012 mm against the exact circle |
| `verify_backlash_fix.py` | does the KI-025 backlash fix behave as designed? | the backlash no longer touches the tracking error (0.0000 mm) and now widens the envelope |
| `exp10_gain_band.py` | what does the digital twin's actuator-gain band do? | it blows up the **envelope** (54 mm at ±1 % over 1 m), not the tracking error (0.9 mm) - the controller corrects the speed error |
| `exp10_floor.py` | what sets the tracking-error floor at fine sampling? | the 20 ms command hold: at n=40 a 2 ms tick takes the error from 0.271 cm to 0.071 cm; giving the controller the **interval-average velocity** removes it without touching the tick (0.098 cm), which pins the mechanism on the distance covered per interval |
| `exp11_density.py` | how does the error vary with the sampling density? | it is **U-shaped**, with a minimum near 4-5 control ticks per sample interval (n=20 on the S-curve); the default `-n 10` is on the coarse side, where the error is now 6x the minimum |
| `exp12_reference_consistency.py` | is the reference internally inconsistent? | yes: the heading and the turn rate are interpolated independently and disagree by up to 9 degrees mid-interval |
| `exp13_package.py` | the two halves of the fix, together | the reference fix and the self-consistent plan are mirror images; the reference fix wins at fine sampling, the plan fix at coarse |
| `exp14_which_inconsistency.py` | which of the reference's three inconsistencies matters? | the turn rate (2-3.6x); the Ramsete gain is worth nothing |
| `exp15_interval_average.py` | the complete reference fix, measured on top of what ships | the turn-rate half is already in the tree (worth 1.00x everywhere); the **velocity** half is worth up to 7.2x at fine sampling and makes the coarse end 1.45x worse |
| `exp16_combined.py` | the two fixes together, in both timing modes | the reference fix and the self-consistent plan are mirror images across the density |
| `exp16_firmware_parity.py` | does the validator mirror the controller that runs? | **it does now**: the legacy mirror and the shipped model differ by 0.92-1.03x (the rounding and the gain are worth about 1 %), and the third column prices the 20 ms heartbeat: the hold costs the robot 2.6-4.1x the tracking error at the fine densities |
| `sync_tables.py` | regenerate section 9's tables from the measurements | the tables are generated, so they cannot drift from the data they quote (`--check` to verify only) |
| `verify_docs.py` | check the research documentation | encoding, markdown structure, table drift and the line-number citations |

## The tables are generated

Section 9 of [`../../HEURISTICS_AND_VERIFICATION.md`](../../HEURISTICS_AND_VERIFICATION.md) quotes the
measurements, and the numbers in it are rewritten from the JSON rather than
typed in:

```bash
python docs/research/rq_a/sync_tables.py           # rewrite the tables
python docs/research/rq_a/sync_tables.py --check   # verify only, exit 1 on drift
python docs/research/rq_a/verify_docs.py           # the rest of the checks
```

`sync_tables.py` uses the newest folder under `docs/measurements/` unless
`--folder` says otherwise.

## Reproducing against the pinned tree

The notes were measured against a pinned copy of the tree, because the working
tree was being edited during the session. `PINNED_TREE.md5` records the MD5 of
every module in that copy. To reproduce against it:

```bash
export OVERBOARD_TREE=/path/to/pinned/copy
python docs/research/rq_a/exp9_summary.py
```

Without `OVERBOARD_TREE` the scripts use the repository root, which is the
right thing for checking the current behaviour but will not reproduce the
recorded numbers exactly if the planner has changed since.

## Before believing a large effect

`check_hermite.py` exists because the spline reference produced a
convincing-looking 20–47 cm "failure" that was entirely a bug in the
interpolant (the curvature was taken with respect to the wrong variable).
Verify an instrument against ground truth before reading anything into a
large effect. The synthetic arcs in `exp5_attribution.py` are the equivalent
control for the validator.
