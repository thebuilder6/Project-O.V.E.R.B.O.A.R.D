# TOPIC-05 + TOPIC-06 — Studio UX + classroom onboarding: forum debate (Ref Rina: Mia x Alvarez x Sara)

> Sources: `review_forum/synthesis/CLUSTERING.md:18-21` (TOPIC-05, TOPIC-06 OPEN), `review_forum/round1/student.md:7`, `review_forum/round1/coach.md:5,7`.
> Constraint voices: inventory-test pins markup (`tests/test_studio_inventory.py:3`, `docs/STUDIO_UI_INVENTORY.md:354`), single-connection BLE (`overboard/deploy/hub_ble.py:247-249`).

## Grievance 1 — Plant / tick / mesh jargon (TOPIC-06a)

**Mia (student) opens:** "In Solve > Advanced there's `Auto invariant mesh (50 ms)` checked, and right below `Open-loop plant (fast)` vs `Closed-loop tick map (slower, truer)`. Plant like a leaf? Tick like the bug that bit me at camp? I'm scared to uncheck anything." (`review_forum/round1/student.md:7`, labels at `overboard/viz/studio.html:304-306,323-326`). Mia's fix: rename to kid words — `Auto add points (50 ms spacing)`, `Fast check` vs `Careful check (7–10x slower)` — plant/tick/mesh into a tooltip.

**Alvarez (coach) adds classroom failure mode:** in a 45-min class with 12 kids sharing one laptop (`review_forum/round1/coach.md:3`), nobody reads the tooltip. A kid toggles "Careful check" thinking it is better, the validation takes 7–10x, the lesson ends mid-run with no trajectory. Jargon is not just scary — it is a wrong default under time pressure.

**Sara / Architect rebuts with constraint:** the words are load-bearing. `plant` vs `tick` selects the reachability model the envelope gate uses, and every candidate carries `propagation` + `inclusion` (`AGENTS.md` §Validation). Renaming the `<option>` text without renaming the value is safe; renaming the value breaks `gui_server.py` / validator contract. And every label rename touches the inventory pin: `docs/STUDIO_UI_INVENTORY.md:354` + `tests/test_studio_inventory.py:3` — "a control added to the studio fails the test until this document is updated" (also flagged `review_forum/round1/student.md:11`). So a rename is a two-file diff, not a one-word edit.

**[RESOLUTION]: rename + tooltip, not rename vs tooltip.** Keep `value="plant"/"tick"` and element ids (`opt-propagation`, `opt-auto-mesh`) frozen; change only the visible strings to Mia's words with the engineering term in a `title`/hint (`Fast check (plant, open-loop)`). Update `docs/STUDIO_UI_INVENTORY.md` counts in the same diff or `test_studio_inventory` goes red.

## Grievance 2 — Waypoint drag silently invalidates the M1 budget (TOPIC-06b)

**Mia opens:** "I dragged a dot and my score went red and I don't know why." The M1 rule pins scoring budgets per shipped mission (`examples/example_straight.json:6-9` + `docs/EXAMPLES.md:128-150`), and a fresh kid-drawn mission fails `--validate` with "needs target and maximum time" (`overboard/gui_server.py:294-296`).

**Alvarez adds classroom failure mode:** (`review_forum/round1/coach.md:5`) any kid dragging a waypoint silently invalidates the budget, so the duration score punishes exploration. With 3 teams editing one mission file, every drag is a silent gate failure at the end of class. His fix: on waypoint edit, studio auto-proposes `target = last unvalidated solve duration, max = 2×` into the scoring fields instead of blocking.

**Sara / Architect rebuts:** auto-writing a budget looks like the planner grading its own homework — the duration score only means something against a measured budget, and a silently auto-filled `target_time` turns a real regression into a pass. The gate (`gui_server.py:294-296` raise; server-side approval `gui_server.py:2157-2169`) must stay a block for `--validate`. But blocking with no proposal is what kills exploration.

**[RESOLUTION]: auto-propose budget, keep the block.** On waypoint edit, fill the scoring fields as a *draft proposal* (visibly marked "proposed — re-measure to confirm"), never silently accepted; `--validate` still refuses until the operator confirms. That keeps Alvarez's flow (no dead end) without weakening the M1 rule.

## Grievance 3 — Single-connection hub-busy + wrong-robot risk (TOPIC-06c)

**Mia opens:** "I pressed Push and it said error and I don't know if my robot got it." Push is gated on `hub_approved` + `!dirty` (`overboard/viz/studio.js:1466-1484`), one job at a time.

**Alvarez adds classroom failure mode:** (`review_forum/round1/coach.md:7`) with 3 teams on one laptop the second team gets a refusal/toast while the first robot is driving, and the BLE scan lists every hub in the gym so the wrong robot gets the mission. His fix: sticky `hub_name` per mission plus a confirm line (hub name + trajectory hash) before driving, and a visible "hub busy — wait" state instead of an error toast.

**Solder-Sara rebuts with constraint:** the busyness is physics, not UI stinginess. "Only one Bluetooth central can hold the hub's single connection" (`overboard/deploy/hub_ble.py:247-249`); the server holds that connection for the whole push (`overboard/gui_server.py:2196-2213` `Popen` + log tail), and `hub_name` is already threaded through (`overboard/gui_server.py:2174,2188-2189`). Queuing pushes or allowing a second push would split the single connection and corrupt both runs. The wrong-robot risk is real in a gym full of hubs — the name is currently just a hint (`hub_ble.py:253-254`).

**[RESOLUTION]: sticky hub + busy state, no queue.** Persist `hub_name` per mission (sticky field, not per-push retype), show a pre-drive confirm (`pushing <traj-hash> to <hub_name> over BLE`), and replace the error toast with a latched "hub busy — robot driving" status while the push job owns the connection. Single-connection serialisation stays; the UI stops making it look like a failure.

## Worked example (from brief)

- Student: "UI should be blue."
- Robotics: "Contrast beats brand." The field canvas is read in a bright gym on a dim laptop; brand-blue on dark-grey fails contrast while the current status/result line (`3.70 s | 1.23 m – validated`) stays legible. Brand colour is welcome in the header; legibility wins on the canvas and the gate verdict.

*Skipped: full `studio.js`/`gui_server.py` re-read past cited lines; did not check `STUDIO_UI_INVENTORY.md` counts. Risk: label renames will fail `test_studio_inventory` until the inventory doc is updated in the same diff.*
