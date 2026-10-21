# Making the Robot Move

Start by duplicating `archive/mission_template.py`. All your code goes inside `async def main():`. This tutorial walks you through everything you need to write a real mission, from the first straight line to doing multiple things at once.

---

## Driving

```python
await drivebase.straight(300)    # Drive forward 300mm
await drivebase.straight(-300)   # Drive backward 300mm

await drivebase.turn(90)         # Turn right 90 degrees
await drivebase.turn(-90)        # Turn left 90 degrees
```

- Distance is in **millimeters**. Positive = forward, negative = backward.
- Angle is in **degrees**. Positive = right, negative = left.
- `await` means the program waits for the movement to fully complete before moving to the next line.

📖 [`straight()`](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.straight) · [`turn()`](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.turn)

---

## Controlling Attachments

```python
await move_front_attachment(speed=500, angle=360)   # Spin front motor 360° at speed 500
await move_back_attachment(speed=300, angle=-180)   # Spin back motor -180° at speed 300
```

- `speed` is in degrees per second.
- `angle` is how far to rotate. Positive and negative control direction.

📖 [`Motor.run_angle()`](https://docs.pybricks.com/en/stable/pupdevices/motor.html#pybricks.pupdevices.Motor.run_angle)

---

## Pausing

```python
from pybricks.tools import wait
wait(1000)   # Pause for 1000 milliseconds (1 second)
```

📖 [`wait()`](https://docs.pybricks.com/en/stable/tools/index.html#pybricks.tools.wait)

---

## Debugging: Stepping Through Your Mission

While tuning, you can add a pause that waits for a physical button press before continuing. This lets you walk up to the robot and inspect its position mid-run before it makes its next move.

```python
await drivebase.straight(300)
wait_for_button()            # Robot holds here — press any Hub button to continue
await drivebase.turn(90)
```

**Remove these once you're satisfied with the alignment.** They're a tuning tool, not part of a competition run.

---

## Changing Speed Mid-Mission

You can change the robot's speed at any point — not just at the start.

```python
drivebase.settings(straight_speed=400)    # Fast
await drivebase.straight(800)

drivebase.settings(straight_speed=80)     # Slow for precision approach
await drivebase.straight(50)
```

> [!TIP]
> **Common competition pattern:** set speed high for the long cross-board drive, drop it low for the final approach to the model. You save time on the long stretch without sacrificing accuracy where it counts.

📖 [`drivebase.settings()`](https://docs.pybricks.com/en/stable/robotics.html#pybricks.robotics.DriveBase.settings)

---

## Doing Two Things at Once (Async)

By default, every command waits for the last one to finish. The robot moves, **stops**, turns, **stops**, moves the arm, **stops**. Every stop costs time.

**`multitask()` lets two things happen at the exact same moment.**

```
Before:  Drive → stop → raise arm → stop → turn
After:   Drive AND raise arm at the same time → turn
```

```python
await multitask(
    drivebase.straight(705),
    move_front_attachment(speed=700, angle=2000)
)
# Both are done before this line runs
await drivebase.turn(90)
```

> [!IMPORTANT]
> Both actions inside `multitask()` must be `async def` functions. All the helpers in `robot_config.py` — `move_front_attachment`, `move_back_attachment`, etc. — are already set up correctly for this.
>
> `multitask` itself is also already imported from `robot_config.py`. You don't need to add any extra import lines.

📖 [`multitask()` docs](https://docs.pybricks.com/en/stable/tools/index.html#pybricks.tools.multitask)

---

## Firing an Attachment Without Waiting

Sometimes you want to trigger an arm to move but immediately drive away — you don't care when the arm finishes.

```python
await move_front_attachment_async(speed=800, angle=360)   # Arm starts moving...

# ...but the robot doesn't wait. These run immediately:
drivebase.settings(straight_speed=600)
await drivebase.straight(-800)                            # Robot drives away while arm is still moving
```

This is useful for "flinging" an attachment out of the way while the robot rushes back to base.

---

**Next Up:** [04 - Going Faster & Staying Accurate](./TUTORIAL_04_SPEED_AND_ACCURACY.md)
