# Round 1 v3 — student (Mia, 14)

**1. [GRIEVANCE] `overboard/viz/studio.html::#opt-propagation` — "Open-loop plant (fast)" vs "Closed-loop tick map (slower, truer)"**
Kid-confusion: I thought *plant* was the green FLL mission model and *tick* was a bug. I have no idea which one I'm supposed to pick, and "How the drift budget is measured" doesn't help — drift budget already sounds like money.
Fix: name them what they do for me: `Fast check (good enough)` vs `Careful check (slower, truer)`, one-line sub-text "leave on Fast unless a mentor says otherwise." HOLD [VETO: COGNITIVE_LOAD] — two control-theory words in one dropdown blocks a first-season kid from pressing Solve confidently.

**2. [QUESTION] `overboard/diagnostics.py::run_doctor` — "…so --validate and sysid_fit.py do not work yet" + `port 8000: free` line**
Kid-confusion: I never typed `sysid_fit.py`, I just double-clicked install. Is that a file I broke? And why is the doctor telling me about *ports* — did I unplug something? I just want to know "can I drive yet?"
Fix: say it in kid-words: "The core planner is ready. The safety-check and robot-measuring steps need one more install — run `install.bat` again." Hide the `port 8000` line behind "studio address" or drop it unless it's actually busy.

**3. [SUGGESTION] `overboard/viz/studio.html::#opt-accept-score` — "Deliberately accept a risky plan at score (1-100)"**
Kid-confusion: This sounds like you want me to gamble with our robot. If my plan fails, should I just type 90 here to make the red go away? My coach would kill me.
Fix: reword to what it really is: "Keep a FAILED plan only if it scores at least ___ (leave empty = never keep failures)". Blank must look like the safe normal, not like I'm missing something.

**4. [GRIEVANCE] `install.bat::VENV_PY` / `start_studio.bat::PY` — "Reusing the existing .venv" / "virtual environment .venv could not be created"**
Kid-confusion: What's a `.venv`? Is that a virus folder? Did last year's team leave junk on this laptop? "Reusing the existing" sounds like I'm using someone else's old work.
Fix: never say venv to a kid: "Found your saved setup — reusing it" / "Couldn't make the setup folder. Check the lines above and try again." Same for `start_studio.bat`'s "Checking the environment…" — say "Checking your setup…".

Skipped: full CLI flag help text and `help.html` deep calibration section (too expert to judge on day one). Risk: #1 and #3 are the ones that could make a kid run an unchecked plan — the words make the unsafe option sound normal.
