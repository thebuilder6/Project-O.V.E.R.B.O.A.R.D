"""Continuous review daemon — forum ticket to merged code, with build-lane management.

Usage:
  python review_forum/orchestrator.py --plan            # waves + ticket status
  python review_forum/orchestrator.py --lanes           # worktree lane table
  python review_forum/orchestrator.py --check T004      # quarantine grep on ticket
  python review_forum/orchestrator.py --claim T005      # open lane (worktree+branch)
  python review_forum/orchestrator.py --release T020    # prune lane after merge
  python review_forum/orchestrator.py --prune           # release all MERGED lanes
  python review_forum/orchestrator.py --build T004      # refuse if dirty; print brief

Stdlib only. Merge state lives in synthesis/merge_state.json (daemon-owned,
never hand-edited). Forum resolutions stay the authority; this file only
schedules, guards, and manages lanes.
"""

import json
import subprocess
import sys
from pathlib import Path

FORUM = Path(__file__).resolve().parent
TICKETS = FORUM / "synthesis" / "tickets.json"
STATE = FORUM / "synthesis" / "merge_state.json"
QUARANTINE = FORUM / "topics" / "quarantine"
LANES_ROOT = Path("C:/Users/jumpi/AppData/Local/Temp/opencode")
MAX_LANES = 4  # bound parallel build chaos; one ticket claimed per run anyway

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


def sh(*args, cwd=None):
    r = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    return r


def load_tickets():
    data = json.loads(TICKETS.read_text())
    return {t["id"]: t for t in data["tickets"]}, data


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2))


def ticket_status(tid, tickets, state):
    s = state.get(tid, tickets.get(tid, {}).get("status", "OPEN"))
    if isinstance(s, dict):
        return s.get("status", "OPEN")
    return s


def lanes():
    """Our lanes: worktrees under LANES_ROOT on agent/* branches."""
    r = sh("git", "worktree", "list", "--porcelain")
    out, cur = [], {}
    for line in (r.stdout or "").splitlines():
        if line.startswith("worktree "):
            if cur:
                out.append(cur)
            cur = {"path": line[9:]}
        elif line.startswith("branch "):
            cur["branch"] = line[7:]  # keep full ref; prefix checked below
    if cur:
        out.append(cur)
    ours = []
    for w in out:
        p = Path(w.get("path", ""))
        br = w.get("branch", "").replace("refs/heads/", "", 1)
        if p.parent == LANES_ROOT and br.startswith("agent/"):
            tid = br.split("/")[1].split("-")[0].upper()
            st = sh("git", "-C", str(p), "status", "--porcelain")
            ahead = sh("git", "-C", str(p), "log", "--oneline", "main..HEAD")
            ours.append({"ticket": tid, "path": str(p), "branch": br,
                         "dirty": [l for l in (st.stdout or "").splitlines()
                                   if l.strip()],
                         "commits": [l for l in (ahead.stdout or "").splitlines()
                                     if l.strip()]})
    return ours


def cmd_lanes():
    tickets, _ = load_tickets()
    state = load_state()
    found = lanes()
    print(f"lanes: {len(found)}/{MAX_LANES}  (root {LANES_ROOT})")
    for lane in sorted(found, key=lambda l: l["ticket"]):
        tid = lane["ticket"]
        t = tickets.get(tid, {})
        st = ticket_status(tid, tickets, state)
        flag = "DIRTY" if lane["dirty"] else "clean"
        print(f"  {tid} [{st}] {flag} {len(lane['commits'])} commits "
              f"owner={t.get('owner')} :: {lane['path']}")
    if len(found) >= MAX_LANES:
        print("at cap: release a MERGED lane before claiming.")


def actionable(tid, tickets, state):
    t = tickets.get(tid)
    if not t:
        return False, "unknown ticket"
    st = ticket_status(tid, tickets, state)
    if st in ("MERGED", "IN_BUILD", "IN_AMEND", "APPROVED"):
        return False, f"status {st}"
    deps = [d for d, s in state.items()
            if isinstance(s, dict) and tid in s.get("after", [])]
    _ = deps  # forward deps live on dependents, not here
    after = []
    for wave in WAVES.values():
        if tid in wave:
            after = wave[:wave.index(tid)]
            break
    blocked = [d for d in after
               if ticket_status(d, tickets, state) != "MERGED"]
    if blocked:
        return False, f"waiting on {blocked}"
    if isinstance(state.get(tid), dict) and state[tid].get("status") == \
            "needs-measurement":
        return False, "needs-measurement: run the measurement first"
    return True, "ok"


def cmd_claim(tid):
    tid = tid.upper()
    tickets, _ = load_tickets()
    state = load_state()
    ok, why = actionable(tid, tickets, state)
    if not ok:
        print(f"cannot claim {tid}: {why}");
        return 1
    if any(l["ticket"] == tid for l in lanes()):
        print(f"lane {tid} already open");
        return 1
    if len(lanes()) >= MAX_LANES:
        print("at cap: --prune MERGED lanes first");
        return 1
    t = tickets[tid]
    branch = f"agent/{tid.lower()}-{t.get('owner')}"
    lane = str(LANES_ROOT / f"wt-{tid.lower()}")
    r = sh("git", "worktree", "add", lane, "-b", branch, "main")
    if r.returncode:
        print(r.stderr.strip() or f"worktree add failed for {tid}");
        return 1
    state[tid] = {"status": "IN_BUILD"}
    save_state(state)
    print(f"claimed {tid} :: {lane} :: {branch}\nchange: {t.get('change')}");
    return 0


def cmd_release(tid, force=False):
    tid = tid.upper()
    for lane in lanes():
        if lane["ticket"] != tid:
            continue
        if lane["dirty"] and not force:
            print(f"refusing: lane {tid} dirty (uncommitted work) — "
                  f"commit on its branch or pass --force");
            return 1
        r = sh("git", "worktree", "remove", lane["path"],
               *(["--force"] if force else []))
        if r.returncode:
            print(r.stderr.strip() or f"worktree remove failed for {tid}");
            return 1
        sh("git", "branch", "-D", lane["branch"])
        state = load_state()
        if tid in state and isinstance(state[tid], dict):
            state[tid]["lane"] = "released"
            save_state(state)
        print(f"released {tid} (worktree + branch removed)");
        return 0
    print(f"no lane for {tid}");
    return 1


def cmd_prune():
    tickets, _ = load_tickets()
    state = load_state()
    released = 0
    for lane in lanes():
        if ticket_status(lane["ticket"], tickets, state) == "MERGED":
            if cmd_release(lane["ticket"]) == 0:
                released += 1
    print(f"released {released} MERGED lane(s)");


def cmd_plan():
    tickets, data = load_tickets()
    state = load_state()
    print(f"tickets: {len(tickets)}  forum-resolved: {data.get('resolved')}  "
          f"needs-measurement: {data.get('needs-measurement')}")
    for wave, ids in WAVES.items():
        print(f"\n## {wave}")
        for tid in ids:
            t = tickets.get(tid, {})
            print(f"  [{ticket_status(tid, tickets, state)}] {tid} "
                  f"({t.get('topic')}) owner={t.get('owner')}")


def cmd_check(tid):
    tid = tid.upper()
    tickets, _ = load_tickets()
    t = tickets.get(tid)
    if not t:
        print(f"unknown ticket {tid}");
        return 1
    blob = json.dumps(t).lower()
    hits = [term for term in RETIRED_TERMS if term in blob]
    if hits:
        print(f"QUARANTINE FAIL {tid}: retired terms {hits}");
        return 1
    print(f"QUARANTINE OK {tid}");
    return 0


def cmd_build(tid):
    tid = tid.upper()
    tickets, _ = load_tickets()
    state = load_state()
    ok, why = actionable(tid, tickets, state)
    if not ok:
        print(f"cannot build {tid}: {why}");
        return 1
    lane = next((l for l in lanes() if l["ticket"] == tid), None)
    if not lane:
        print(f"no lane for {tid} — run --claim first");
        return 1
    if lane["dirty"]:
        print(f"refusing: lane {tid} dirty — agent work in progress");
        return 1
    if cmd_check(tid):
        return 1
    t = tickets[tid]
    print(f"lane: {lane['path']} :: {lane['branch']}\n"
          f"owner: {t.get('owner')}  topic: {t.get('topic')}\n"
          f"change: {t.get('change')}\nimplement minimal diff, then "
          f"triad review per PR_PROTOCOL.md");
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--plan"]:
        cmd_plan()
    elif args == ["--lanes"]:
        cmd_lanes()
    elif args == ["--prune"]:
        cmd_prune()
    elif len(args) == 2 and args[0] == "--check":
        sys.exit(cmd_check(args[1]))
    elif len(args) == 2 and args[0] == "--claim":
        sys.exit(cmd_claim(args[1]))
    elif len(args) in (2, 3) and args[0] == "--release":
        sys.exit(cmd_release(args[1], force=args[2:] == ["--force"]))
    elif len(args) == 2 and args[0] == "--build":
        sys.exit(cmd_build(args[1]))
    else:
        print(__doc__);
        sys.exit(2)
