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

## The check is one-sided

Sampled points inside the box do not *prove* the box is sound, they only
fail to disprove it. The soundness argument is the inclusion function's own
(the mean-value form with the interval Jacobian is a valid enclosure); this
is the empirical check on top of it, and the corner cases are the part that
comes closest to a certificate. `tests/test_envelope_soundness.py` runs the
same check in the test suite.

## Reproducing

```bash
python docs/research/rq_a/exp19_envelope_scatter.py --out docs/measurements/2026-10-20-envelope --draws 60
python docs/research/rq_a/exp10_gain_band.py
```
