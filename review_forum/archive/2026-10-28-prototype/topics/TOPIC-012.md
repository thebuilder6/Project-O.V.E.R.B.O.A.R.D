# TOPIC-012 Web app auto-solve: photo/drag mission → cloud solve → push

Status: OPEN
Owners: [dreamer, skeptic]
Target-Symbols: [overboard/gui_server.py::Handler:do_POST, overboard/deploy/hub_deploy.py::push_trajectory, overboard/deploy/hub_deploy.py::check_trajectory_approval]
Related-Topics: [TOPIC-04, TOPIC-05-06]
Round: R3

## [SUGGESTION by dreamer]

Today: hand-write JSON, run CLI/studio locally, `--push-hub` over BLE from the same laptop (`gui_server.py::Handler:do_POST` shells the CLI; `hub_deploy.py::push_trajectory` re-checks `hub_approved` + `trajectory_sha256`).
Dream: kid photographs the mat / drags waypoints in a browser, cloud solves with `--validate`, phone-positioned hub receives only an approved `.traj` — the existing `check_trajectory_approval` gate travels unchanged, it just runs server-side.
No new planner, no gate relaxation: the export/approval rule is reused verbatim, and unapproved plans never push.

**Invited veto:** `SAFETY_GATE` (untrusted photo → waypoint mapping must not smuggle an unvalidated plan past the gate); `COGNITIVE_LOAD` (45-min classroom budget — auto-solve must not hide what the robot will do). Skeptic: define the photo-to-waypoint trust boundary.
