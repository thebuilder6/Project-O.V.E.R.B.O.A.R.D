# Round 1 — student (Mia, 14)

[GRIEVANCE] requirements.txt:3-6 — I opened this because install.bat said `requirements.txt` and I thought I broke something. It says "upper bound is 3.0, not 2.0, because the optional stack needs numpy 2.2" and "a pin under 2.0 made the two files impossible to install together." I'm 14, first FLL season, I just want to install. What is a pin? What is a stack? What's numpy? Fix: delete the diary entry, replace with one kid line: `# You don't need to open this file. Just double-click install.bat.`

[QUESTION] overboard/diagnostics.py:224-225 — I ran `python -m overboard.main --doctor` like you told me and it said "so --validate and sysid_fit.py do not work yet." Where IS sysid_fit.py? There's no file with that name to click, and the real command is `--sysid-fit` or `fit.py`. Did I delete something? Fix: say what to actually type/click: `--validate and the Calibrate Fit step do not work yet. To fix: pip install -r requirements-optional.txt`.

[GRIEVANCE] overboard/viz/studio.html:304-326 — In Solve > Advanced there's `Auto invariant mesh (50 ms)` checked, and right below `Open-loop plant (fast)` vs `Closed-loop tick map (slower, truer)`. Plant like a leaf? Tick like the bug that bit me at camp? Mesh like a net? I'm scared to uncheck anything. Fix: rename to plain words a kid uses: `Auto add points (50 ms spacing)` and `Fast check` vs `Careful check (7-10x slower)`, put plant/tick/mesh words in a tooltip, not the button.

[SUGGESTION] docs/CALIBRATION_PROCEDURE.md:85-89 — It says "roll the robot one wheel turn and divide the distance by 6.28" to get wheel radius. Where did 6.28 come from? Is that a LEGO part number? I measured 175mm, now what? Fix: say why in one line with an example: `One full spin is 2 x pi = 6.28, so 175 mm / 6.28 = 27.9 mm radius. Type 27.9.`

Skipped: didn't run --gui or the hub push, only read the files. Risk: if you rename studio labels, the studio inventory test pins the markup so it will need updating too.
