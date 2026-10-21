---
description: Start the studio, inspect it in a browser, and stop it
---

Start the browser studio in the background with `python -m overboard.main --gui` (it uses port 8000 or the next free port).

Open the URL in the browser panel, snapshot the page, check the console for errors, and take a screenshot. If $ARGUMENTS names a control or page, exercise it.

If controls changed, check the markup against `docs/STUDIO_UI_INVENTORY.md`. Stop the server when done. Never push to the hub unless a validated trajectory passed the gate.
