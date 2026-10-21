# Running Your Missions: The Hub Menu

In FLL competitions, you run many different missions back-to-back. You can't plug into a computer between each one — you need to launch programs directly from the Hub.

`robot.py` is the file that makes this possible. **This is the only file you run on competition day.**

---

## How the Hub Menu Works

Think of `robot.py` like a TV remote. Each number launches a different "channel" (mission program).

When you run `robot.py`, a number appears on the Hub's screen:

- **Left / Right buttons** — scroll through mission numbers
- **Center button** — select and run the highlighted mission

The robot immediately starts executing whichever mission you selected.

📖 [hub_menu docs](https://docs.pybricks.com/en/stable/tools/index.html#pybricks.tools.hub_menu)

---

## Your Workflow: Build First, Add to Menu Last

While you are building and testing a mission, **you don't need to use the menu at all.**

Open your mission file directly in the Pybricks IDE and click **Run**. It launches immediately. This lets you iterate and tune quickly without having to scroll through a menu every time.

Once your mission is working, **then** add it to `robot.py` for competition day.

---

## How to Add a New Mission

> [!NOTE]
> The hub menu now routes only the two supported programs (the pushed
> trajectory and the SysId collector), and the demo missions it used to list
> live in `archive/`. The workflow below still works — it is how you would
> bring a hand-written mission back — but a planned trajectory does not need
> it: solve it on the host and push it with `--push-hub`.

**Step 1** — Duplicate `archive/mission_template.py` (or any demo in `archive/`) and give it a name:
```
mission_bridge.py
```

**Step 2** — Write your mission code inside `async def main():`

**Step 3** — Open `robot.py` and add a branch for your mission number:

```python
elif selection == 3:
    import mission_bridge
```

Next time you run `robot.py`, pressing **3** on the Hub will launch your new mission.

> [!TIP]
> The line at the very bottom of every mission file (`if __name__ == "__main__"`) is what lets you run the file directly *and* launch it from the menu. Don't delete it — but you don't need to understand how it works to use it.

---

**Next Up:** [03 - Making the Robot Move](./TUTORIAL_03_MISSIONS.md)
