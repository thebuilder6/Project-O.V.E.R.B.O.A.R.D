# Going Faster & Staying Accurate

These are the two biggest problems competitive teams face:

1. **The robot is too slow** — it stops completely between every movement.
2. **The robot drifts** — small errors accumulate and it misses targets by match 4 or 5.

This tutorial covers the techniques that solve both.

---

## Smooth Turns: `drivebase.arc()`

### The Problem

The default way to change direction looks like this:

```
drive straight → full stop → spin in place → full stop → drive straight
```

Every full stop costs time and kills momentum. For a robot that needs to arc around an obstacle, this is extremely wasteful.

### The Fix: Sweep Through an Arc

`drivebase.arc()` lets the robot turn *while moving forward* — like a car going around a bend — without ever stopping.

```python
await drivebase.arc(radius=300, angle=90)
```

> [!NOTE]
> You may see older code using `drivebase.curve()`. This still works, but Pybricks v3.6 replaced it with `arc()`. Use `arc()` for all new code — it has slightly different direction conventions so check the docs when converting.

### What the Parameters Mean

**`radius`** — the size of the arc. With a **positive** radius, the robot drives along a circle to its **right**. With a **negative** radius, the robot drives along a circle to its **left**.

**`angle`** — how far to drive along that circle. A **positive** value means driving **forward** along the circle. **Negative** means driving in reverse.

### Before & After

```
Before:  [Robot] → stop → spin 90° → stop → [Robot facing right]

After:   [Robot]
               \
                \  ← smooth arc, never stopped
                 \
                  [Robot] ← now facing right
```

```python
# Sweep forward in a wide arc, turning right
await drivebase.arc(radius=300, angle=90)

# Sweep backward in a tight arc, turning left
await drivebase.arc(radius=-150, angle=-45)
```

> [!TIP]
> Any time your mission code reads `straight → turn → straight`, ask whether an `arc()` could replace the turn. You'll usually save 0.5–1 second per maneuver.

📖 [`arc()` docs](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.arc)

---

## Resetting Alignment: Wall Squaring

### The Problem

Even with the gyro enabled, small errors accumulate over a match. A slight bump, an uneven table, minor wheel slip — any of these can rotate the robot by a degree or two. By mission 4 or 5, small errors stack into big misses.

### The Fix: Physically Drive Into a Wall

The most reliable way to reset alignment mid-match is to drive the robot flat against a wall. The wall forces both drive wheels to a perfectly straight position — no matter what angle the robot was at before.

### Why You Must Turn Off the Gyro First

If you leave the gyro **on**, it fights the wall. One wheel touches the wall a split-second before the other. The gyro detects this as a rotation and tries to correct it — pushing the robot crookedly into the wall instead of squaring up.

**Always disable the gyro before squaring on a wall.**

### The Full Sequence

```python
from pybricks.parameters import Stop

# 1. Turn off the gyro — let the wall do the work
drivebase.use_gyro(False)

# 2. Drive both motors into the wall simultaneously at low power
#    duty_limit=30 prevents motor strain while still pressing firmly
await multitask(
    motorLeft.run_until_stalled(speed=200,  then=Stop.HOLD, duty_limit=30),
    motorRight.run_until_stalled(speed=200, then=Stop.HOLD, duty_limit=30)
)

# 3. Turn the gyro back on
#    Pybricks now treats this perfectly flat position as exactly 0°
drivebase.use_gyro(True)
```

After step 3, every subsequent `straight()` and `turn()` is relative to your new, perfectly aligned heading.

> [!IMPORTANT]
> You **must** use `multitask()` here. Running the motors one after the other means one wheel stalls against the wall while the other keeps spinning — you'll end up just as crooked as before.

📖 [`use_gyro()` docs](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.use_gyro) · [`run_until_stalled()` docs](https://docs.pybricks.com/en/stable/pupdevices/motor.html#pybricks.pupdevices.Motor.run_until_stalled)

---

## Advanced: Tuning Speeds & Tolerances

> [!NOTE]
> Come back to this section once your mission is already working. These settings help you squeeze out extra speed and smoothness, but they're not needed to get started.

### Adjusting Speed and Acceleration

You can control exactly how fast the robot drives and how quickly it reaches that speed:

```python
# straight_speed in mm/s | straight_acceleration in mm/s² | turn_rate in °/s
drivebase.settings(straight_speed=400, straight_acceleration=600, turn_rate=150, turn_acceleration=300)
```

Higher acceleration gets to top speed faster — but on heavy robots it causes wheel slip at the start of each move. If your robot lurches or fishtails, reduce acceleration first.

📖 [`settings()` docs](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.settings)

### Fixing the "Jitter" at the End of Movements

By default, Pybricks holds the robot precisely at its target position after each move. On heavy robots, this causes the motors to micro-jitter back and forth — the robot visibly vibrates for a second before the next command runs.

Fix this by telling Pybricks "close enough is good enough":

```python
# Finish the move when within 5mm and slowing down
drivebase.distance_control.target_tolerances(speed=50, position=5)

# Finish the turn when within 3 degrees and slowing down
drivebase.heading_control.target_tolerances(speed=50, position=3)
```

The robot will feel noticeably snappier — it finishes each movement and flows straight into the next one.

📖 [`distance_control` docs](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.distance_control) · [`heading_control` docs](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.heading_control)

---

**You've finished the tutorials!**

Open `archive/mission_3_advanced.py` to see curves, wall squaring, async multitasking, and speed tuning all combined in a single realistic competitive run.
