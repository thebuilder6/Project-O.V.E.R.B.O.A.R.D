# The envelope, 2026-10-20

The measurements behind **KI-049** and RQ-A finding **F12**: the reachability
envelope reported a width of 54 mm against a 20 mm gate, and the tube behind
that number was inflated by a dependency the interval arithmetic could not
see.

## What is here

| file | what it is |
|---|---|
| `exp19_envelope_scatter.txt` | the reachable set sampled over the declared uncertainty, against the box the validator reports, for three plant parameterisations and three inclusion functions |
| `exp17_envelope_headroom.txt` | the same question the other way round: how much of the tube is over-approximation, and whether the integrator or the inclusion function can take any of it back |
| `exp18_real_missions.txt` | the six example missions under the three inclusion functions, with and without a calibrated model |
| `exp20_closed_loop_tick.txt` | the closed-loop tick map against the shipped open-loop propagation |
| `exp21_curve_geometry.txt` | the envelope on a curve: the box against the sampled set, sample by sample |
| `exp22_twin_propagation.txt` | the shipped validator with the two propagations, on the six missions, with and without a calibrated model, and the estimate's error against the estimate |
| `exp23_twin_tick_belief.txt` | is the robot's pose estimate the reason the twin's tick tube is wider? The control: the same map with the controller fed the truth |
| `exp10_gain_band.txt` | the gain band against the tracking and the envelope, re-measured after the fix |
| `envelope_plain_*.png` | the plain plant: the interval box against the sampled reachable set, one figure per inclusion function |
| `envelope_twin_wheels_*.png` | the digital twin carrying the two wheel speeds as states - the layout before the fix |
| `envelope_twin_mean_diff_*.png` | the digital twin carrying the mean wheel speed and the difference - the layout after the fix |

## The result

A 1 m straight at 0.5 m/s with a ±1 % actuator-gain band. The "reachable
set" is the set of poses the shipped plant and the shipped controller
actually reach, sampled over the declared uncertainty with the corners of the
box included.

| plant and inclusion | box | truncated | reachable set | contains it |
|---|---:|---|---:|---|
| plain plant, natural inclusion | 11.77 mm | no | 4.51 mm | yes |
| plain plant, Jacobian inclusion | 11.77 mm | no | 4.51 mm | yes |
| digital twin, wheel speeds as states | 53.61 mm | **yes** | 4.55 mm | yes |
| digital twin, mean and difference | 11.76 mm | no | 4.55 mm | yes |

Before the fix the plain plant read **54.03 mm, truncated**, and the digital
twin read 53.61 mm, truncated. Those numbers were the 50 mm guard's
fingerprint: the same 54.561 mm came out for a 0.5 m path, a 1 m path and a
2 m path, because a truncation depends on the guard and not on the path.

The Jacobian inclusion is sound - the tighter box still contains the sampled
set - and it is not needed: taking the wheel-speed difference from the
commands before the gain multiplies it removes the dependency exactly, and
the digital twin needed the state to carry the mean and the difference
instead of the two wheel speeds. The default stayed on the natural
inclusion because it is now the better one: on the six example missions the
Jacobian is 0.00-0.08 mm **wider** and 5-15 % slower.

## What is left, and where it comes from

On a **curve** the tube still crosses the 50 mm guard. That is not the
reachable set being wide: on the S-curve with a ±1 % band
(`exp21_curve_geometry.txt`) the sampled reachable set is **4.09 mm**
against a 48.44 mm box, and the box with the gain band collapsed is 7.98 mm.

The reason is structural. The validator propagates the **plant** and
evaluates the controller *outside* the propagation, at the tube's midpoint,
so the feedback contraction is never in the tube: what it bounds is the
reachable set of an open-loop plant driven by a pre-recorded command. The
lever is to embed the **closed-loop tick map** as a discrete map, so the
Jacobian carries the feedback term `dF/dx = dPhi/dx + (dPhi/du)(dN/dx)`
(`exp20_closed_loop_tick.txt`):

| propagation | inclusion | tube | reachable set |
|---|---|---:|---:|
| shipped (open loop) | natural | 11.77 mm | 4.51 mm |
| closed-loop tick map | natural | 627.03 mm, diverged | 4.51 mm |
| closed-loop tick map | **Jacobian** | **6.31 mm** | 4.51 mm |
| closed-loop tick map | mixed-Jacobian | 5.98 mm | 4.51 mm |

So the Jacobian is the right tool for the closed loop and the wrong tool
for the open-loop plant: on the tick map the natural extension is
catastrophic and the Jacobian is what makes the enclosure work at all.

## The shipped validator with the two propagations

`exp22_twin_propagation.txt` measures the validator itself, with the
"plant" propagation and the "tick" one, on the six missions. The default
plant, without a calibrated model, with the reachable set sampled over the
declared uncertainty:

| mission | plant | tick (jac) | tick (mjac) | reachable set | contains |
|---|---:|---:|---:|---:|---|
| straight | 1.19 cm | 0.64 cm | 0.61 cm | 0.45 cm | yes |
| S-curve | 5.36 cm, cut off | 2.53 cm | **1.88 cm** | 0.33 cm | yes |
| first four | 1.31 cm | 0.92 cm | 0.79 cm | 0.27 cm | yes |
| events | 5.37 cm, cut off | 5.40 cm | 4.45 cm | 0.37 cm | yes |
| stops | 5.31 cm, cut off | 1.94 cm | **1.30 cm** | 0.39 cm | yes |
| complex | 5.07 cm, cut off | 3.17 cm | **1.83 cm** | 0.26 cm | yes |

Every tube contains the sampled reachable set, with 0.000 mm of overshoot.
The tick propagation with the mixed-Jacobian turns three of the four
cut-off missions into passes of the 20 mm gate, and it costs 7-10x the
validation time.

**With the digital twin it is worse**, and that is the finding that keeps
it off by default: the map has to carry the robot's own pose estimate,
because the controller reacts to the estimate and not to the truth, and an
estimate carried in an interval state grows with the wheel-speed
uncertainty. The belief's spread is then *added* to the plant's, because
the two are perfectly correlated and an interval box cannot say so - the
same lost correlation as the phantom turn rate, one level up. The tick
propagation is the lever for the default plant; the twin needs the belief's
correlation to be made explicit first.

**But the belief is not the reason, and the control says so.** Feeding the
controller the *truth* instead of the estimate (`exp23_twin_tick_belief.txt`)
is **not** smaller (12.48 cm against 12.08 cm on the S-curve), and it is
**unsound** while it is at it: it misses reachable poses by 1 mm, which is
the size of the estimate's initial offset. The reason is the twin's *plant
state*: the two wheel speeds are interval coordinates whose spread feeds the
command, and the interval arithmetic counts that growth twice.

**The obvious repair was tried and is worse.** Carrying the estimate's
*error* instead of the estimate, so the box does not add the estimate's
spread to the truth's, is implemented (`belief_mode`) and measured: the
S-curve goes to 42.7 cm with the Jacobian and 9.5 cm with the
mixed-Jacobian, because the error's update differences two cosines of a wide
heading interval - the difference of two over-approximations is worse than
either. The estimate stays a state of its own.

### And on the curve, where it matters

The table above is a straight. The curve is the case that fails today, and
the fix behaves differently there (`exp20_closed_loop_tick.txt`, the
`s_curve mission` case; the reachable set is **3.27 mm**):

| propagation | inclusion | tube | gate | contains the set |
|---|---|---:|---|---|
| shipped (open loop) | natural | 53.64 mm | diverged | yes |
| shipped (open loop) | Jacobian | 51.18 mm | diverged | yes |
| shipped (open loop) | mixed-Jacobian | 50.77 mm | diverged | yes |
| closed-loop tick map | natural | diverged | diverged | yes |
| closed-loop tick map | Jacobian | 25.28 mm | FAIL | yes |
| closed-loop tick map | **mixed-Jacobian** | **18.80 mm** | **pass** | yes |

So on the curve the tick map with the mixed-Jacobian turns "diverged at
50.8 mm" into "18.80 mm, passing the 20 mm gate" - and the tighter tube still
contains the sampled set. The cost is 7-10x the validation time, and the
tracking error is unchanged to the last digit (it is the tube's midpoint
against the plan, so it is decided by the discretisation and not by the
uncertainty).

### The control that says why

`exp21_curve_geometry.txt` runs the same plant, the same uncertainty and the
same reference with the **feedback switched off** - the controller is fed the
plan's own pose, so its correction term is zero:

| | half-diagonal on the S-curve |
|---|---:|
| the box the gate sees | 48.44 mm |
| the same simulation with the feedback off | 23.91 mm |
| the sampled reachable set (the real, closed-loop robot) | **4.09 mm** |

The tube is twice as wide as an open-loop robot and twelve times as wide as
the real one. The feedback's contraction of the error is not in the tube at
all, and the interval arithmetic adds a factor of about two on top of the
open-loop drift. That is why the tick map is the fix and not a tighter
interval over the same open-loop plant.

## The check is one-sided

Sampled points inside the box do not *prove* the box is sound, they only
fail to disprove it. The soundness argument is the inclusion function's own
(the mean-value form with the interval Jacobian is a valid enclosure); this
is the empirical check on top of it, and the corner cases are the part that
comes closest to a certificate. `tests/test_envelope_soundness.py` runs the
same check in the test suite.

## Reproducing

```bash
python docs/research/rq_a/exp10_gain_band.py
python docs/research/rq_a/exp17_envelope_headroom.py
python docs/research/rq_a/exp18_real_missions.py
python docs/research/rq_a/exp19_envelope_scatter.py --out docs/measurements/2026-10-20-envelope --draws 60
python docs/research/rq_a/exp20_closed_loop_tick.py
python docs/research/rq_a/exp21_curve_geometry.py --mission example_s_curve
```

`exp18`, `exp20` and `exp21` read the solved trajectories from
`measurements/traj/`, so point `OVERBOARD_TRAJ` at it (and solve the six
example missions into it first).

**Save the output as UTF-8.** On Windows, PowerShell's `>` writes UTF-16,
and a UTF-16 evidence file is twice the size, greps as binary and diffs as
binary. Redirect through `cmd` instead:

```powershell
cmd /c "python docs\research\rq_a\exp21_curve_geometry.py > docs\measurements\2026-10-20-envelope\exp21_curve_geometry.txt 2>&1"
```
