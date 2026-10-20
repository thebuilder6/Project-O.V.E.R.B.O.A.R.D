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
```

Requirements: `casadi`, `jax`, `immrax`, `numpy`. The scripts write only to a
temporary directory; nothing in the repository is modified.

## Which script answers what

| script | question | headline result |
|---|---|---|
| `exp1_uncertainty.py` | which declared uncertainty drives the tracking error? | none of them — collapsing all seven moves it by 0.1–0.2 % |
| `exp2_reference_artefact.py` | how far does a robot that follows the reference exactly end up from the plan? | 91 % of the S-curve's tracking error |
| `exp3_sampling.py` | does more sampling help? | the plan/profile deviation falls as 1/n², the tracking error does not |
| `exp4_decompose.py` | quantisation, backlash, spline reference? | quantisation ≤ 0.5 %, backlash 1 % on curves, spline reference catastrophic |
| `exp5_attribution.py` | what does a *self-consistent* trajectory score? | 0.16 cm at every sampling density — the validator is not the problem |
| `exp6_consistency.py` | is the optimizer's plan self-consistent? | no: 61 % of the S-curve error is the plan/profile mismatch |
| `exp7_recipe.py` | does exact positions + density fix it? | only at n=10; it gets worse at n=20–80 |
| `exp8_proposal.py` | the two halves of the proposed fix | neither works alone; both together still fail |
| `exp9_summary.py` | the summary table in the notes | the headline numbers |
| `check_hermite.py` | is the spline interpolant correct? | yes: 0.0012 mm against the exact circle |
| `verify_backlash_fix.py` | does the KI-025 backlash fix behave as designed? | the backlash no longer touches the tracking error (0.0000 mm) and now widens the envelope |
| `exp10_gain_band.py` | what does the digital twin's actuator-gain band do? | it blows up the **envelope** (54 mm at ±1 % over 1 m), not the tracking error (0.9 mm) - the controller corrects the speed error |
| `exp10_floor.py` | what sets the tracking-error floor at fine sampling? | the 20 ms command hold: at n=40 a 2 ms tick takes the error from 0.979 cm to 0.048 cm |
| `exp11_density.py` | how does the error vary with the sampling density? | it is **U-shaped**, with a minimum near 4-6 control ticks per sample interval; the default `-n 10` is on the coarse side |

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
