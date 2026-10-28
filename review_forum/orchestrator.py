"""Continuous review daemon — forum ticket to merged code.

Usage:
  python review_forum/orchestrator.py --plan          # print waves + status
  python review_forum/orchestrator.py --check T004    # quarantine grep on ticket
  python review_forum/orchestrator.py --build T004    # refuse if dirty; print builder brief

Stdlib only. Merge state lives in synthesis/merge_state.json (created on first
merge, never hand-edited). Forum resolutions stay the authority; this file only
schedules and guards them.
"""

import json
import subprocess
import sys
from pathlib import Path

FORUM = Path(__file__).resolve().parent
TICKETS = FORUM / "synthesis" / "tickets.json"
STATE = FORUM / "synthesis" / "merge_state.json"
QUARANTINE = FORUM / "topics" / "quarantine"

WAVES = {
    "Wave 1 — deletions and doc parity": ["T020", "T004", "T014", "T022"],
    "Wave 2 — safety gates": ["T007", "T008", "T001", "T006", "T005", "T009",
                              "T011", "T012", "T013", "T021"],
    "Wave 3 — measurement-blocked": ["T010", "T015", "T023"],
    "Wave 4 — method, pub, dreamer": ["T016", "T017", "T018", "T019",
                                      "T024", "T025", "T002", "T003"],
}

# Terms that must never re-enter the tree (mirrors topics/quarantine/).
RETIRED_TERMS = ["multi-verse", "multiverse", "accuracy_weight+=2.0",
                 "w_anchor=1000", "gradient polish", "--refine"]


def load_tickets():
    data = json.loads(TICKETS.read_text())
    return {t["id"]: t for t in data["tickets"]}, data


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def cmd_plan():
    tickets, data = load_tickets()
    state = load_state()
    print(f"tickets: {len(tickets)}  forum-resolved: {data.get('resolved')}  "
          f"needs-measurement: {data.get('needs_measurement')}")
    for wave, ids in WAVES.items():
        print(f"\n## {wave}")
        for tid in ids:
            t = tickets.get(tid, {})
            st = state.get(tid, t.get("status", "OPEN"))
            print(f"  [{st}] {tid} ({t.get('topic')}) owner={t.get('owner')}")


def cmd_check(tid):
    tickets, _ = load_tickets()
    t = tickets.get(tid)
    if not t:
        print(f"unknown ticket {tid}");
        return 1
    blob = json.dumps(t).lower()
    hits = [term for term in RETIRED_TERMS if term.lower() in blob]
    if hits:
        print(f"QUARANTINE FAIL {tid}: retired terms {hits}");
        return 1
    print(f"QUARANTINE OK {tid}");
    return 0


def cmd_build(tid):
    tickets, _ = load_tickets()
    t = tickets.get(tid)
    if not t:
        print(f"unknown ticket {tid}");
        return 1
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True)
    if dirty.stdout.strip():
        print("refusing: working tree dirty — commit or stash first");
        return 1
    if cmd_check(tid):
        return 1
    print(f"branch: agent/{tid.lower()}-{t.get('owner')}")
    print(f"owner: {t.get('owner')}  topic: {t.get('topic')}")
    print(f"change: {t.get('change')}")
    print("implement minimal diff, then run triad review per PR_PROTOCOL.md");
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3 and sys.argv[1:] != ["--plan"]:
        print(__doc__);
        sys.exit(2)
    if sys.argv[1] == "--plan":
        cmd_plan()
    elif sys.argv[1] == "--check":
        sys.exit(cmd_check(sys.argv[2]))
    elif sys.argv[1] == "--build":
        sys.exit(cmd_build(sys.argv[2]))
    else:
        print(__doc__);
        sys.exit(2)
