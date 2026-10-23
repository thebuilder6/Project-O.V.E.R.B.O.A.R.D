# Goal: precision-first path generation for a differential-drive robot

## Setting

A differential-drive robot is given a **precomputed path** — poses, and the timing at which
they must be reached — and follows it with a **Ramsete controller**. It must be fast, precise
and accurate.

Its sensing is a gyroscope and wheel odometry, and nothing else: no camera, no rangefinder, no
external tracking. Its pose is dead-reckoned, so the estimate it corrects against is itself
drifting.

Its computing is limited. The control loop runs on a fixed tick and cannot optimise, re-plan, or
run a heavy filter while driving. The split that matters follows from this: **planning is
offline and has all the compute it wants; execution is online and has almost none.** Everything
this project does has to live on the offline side of that line. The plan is fixed before the run;
the robot does no compute between being switched on and its first move.

## Problem

The control loop can only correct the errors it can measure. Failures fall into three classes,
and only the first belongs to the loop:

1. **Recoverable disturbance.** The robot drifts off the path, and Ramsete steers it back — but
   only while the measurement it steers against is honest, and only as fast as the gain allows.
   A higher gain corrects harder, which costs wheel force, which risks slip.
2. **Control failure.** The path asks for more than the wheels can deliver — traction, wheel
   acceleration, turn rate — so no controller can track it. The reference is the problem, not the
   tracking.
3. **Measurement failure.** Wheel slip and drift corrupt odometry, so the loop confidently
   corrects toward a pose that is wrong. It is most wrong exactly where the robot travels
   furthest without a correction: long, fast stretches.

The classes are not independent. The effort a class-1 correction demands is itself a class-3
cause: braking hard to recover a heading error can break traction, and the robot then loses trust
in the odometry it needed for the recovery. A path that provokes corrections therefore damages its
own sensing.

Classes 2 and 3 are not decided online, and no tuning of the controller decides them. They are
decided by the geometry and timing of the path: where it asks the robot to turn hard, accelerate
hard, or dead-reckon over a long stretch. That is the opening this project takes.

## What this project is not

It is not a better controller. Ramsete is given and stays. It is not an online estimator, a sensor
fusion scheme or a new slip model. It does not ask the robot to decide anything while driving. The
claim is narrower and harder: the best place to spend precision is the reference.

## Question

Can spread be reduced at planning time, before the controller runs — by choosing a path that is
easy to track, unlikely to induce slip, unlikely to lose its pose, and still fast? A path whose
outcome is insensitive to the errors that remain.

Stated so it can fail: *a time-optimal path, tracked by the same controller, is no worse in spread
than a precision-aware one.* The project's job is to measure that comparison under identical
uncertainty, on hardware.

## Goal

A **path generator**. A human specifies the poses the robot must reach and when; the generator
returns the path with

1. the **least spread** at those poses,
2. then the **shortest run time**,
3. then the **smallest residual pose error**,

in that order — because a tight execution with a constant offset can be calibrated out, while
scatter cannot.

Inputs are the human's poses and timing, and the specific robot's model with its uncertainties.
The output is an executable trajectory for that robot, together with the evidence that justifies
it: the predicted spread at each required pose. A path is only a result when it comes with its own
uncertainty.

## Approach

1. **System identification** captures the specific robot: motor characteristics, friction and its
   variation, weight transfer, sensor scale, latency — as best guesses with error ranges covering
   mass, moment of inertia, track width, wheel radius, friction and latency.
2. **Model** the robot *including its control loop and its sensing*, so a predicted behaviour is a
   prediction of the closed loop, not of an idealised plant.
3. **Predict** the set of behaviours a candidate path can produce, and read off the spread at each
   required pose.
4. **Optimise** over candidate paths and keep the one whose predicted spread is least, then whose
   time is shortest.
5. **Deploy** code generated from the same numbers used in planning, so the robot does not run a
   re-tuned approximation of the plan we validated.
6. **Feed back** the execution data into system identification, so the model gets less wrong with
   every run.

The honest limit: prediction is only as good as the model, and the current validation model is a
strong approximation of the robot rather than a full digital twin.

## Secondary goal: ease

A method that cannot be run on someone else's robot is not a result, so ease is part of the claim
rather than a nicety. Three touchpoints, each of which should be one command:

- **Calibrate.** A guided loop that collects the robot's data, fits the parameters, and reports the
  drift and confidence of each — rather than a procedure only its author can follow.
- **Generate.** Sensible defaults, honest presets, and no hand-written configuration to get a first
  path out.
- **Execute.** Generated code runs on the robot as-is, with no manual retuning; a push to hardware
  is gated on the same verdict the planner produced.

"Easy" here is testable: a new robot goes from uncalibrated to a deployed, validated path without
anyone editing code or understanding the optimiser, and every failure message names the step to
fix. The tool also must never flatter the user — an unvalidated path has to look unvalidated.

## What success looks like

- On a set of missions, the precision-aware generator produces **less spread** at the required
  poses than a time-optimal one, at an acceptable cost in time, **on hardware** — and the model
  predicts the direction, and roughly the size, of the difference.
- A second person calibrates their own robot and produces their own path unaided.
