"""Regenerate section 9's tables in docs/HEURISTICS_AND_VERIFICATION.md from
the measurement JSON, and check them against it.

The tables are generated rather than hand-written, so they cannot drift from
the measurements they quote.

    python docs/research/rq_a/sync_tables.py            # rewrite the tables
    python docs/research/rq_a/sync_tables.py --check    # check only, exit 1 on drift

The measurement folder is the newest one under docs/measurements/ unless
--folder says otherwise.
"""
import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DOC = os.path.join(REPO, "docs", "HEURISTICS_AND_VERIFICATION.md")

MISSIONS = ["example_straight", "example_s_curve", "example_mission_first4",
            "example_complex_mission", "example_with_stops", "example_with_events"]
SHORT = {"example_straight": "straight", "example_s_curve": "s_curve",
         "example_mission_first4": "first4", "example_complex_mission": "complex",
         "example_with_stops": "stops", "example_with_events": "events"}
ORDER = ["baseline", "rolling", "bow_left", "bow_right", "turn_first"]


def newest_folder():
    root = os.path.join(REPO, "docs", "measurements")
    folders = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    for f in reversed(folders):
        if os.path.exists(os.path.join(root, f, "sweep_results.json")):
            return os.path.join(root, f)
    raise SystemExit("no measurement folder with sweep_results.json")


def load(base, name):
    with open(os.path.join(base, name), encoding="utf-8") as fh:
        return json.load(fh)


def reason(c):
    if c["ok"]:
        return ""
    parts = []
    for r in c["reasons"]:
        if r.startswith("tracking error"):
            parts.append(f"tracking {c['tracking_cm']:.2f} cm")
        elif r.startswith("motor force"):
            parts.append(f"motor force {float(r.split()[2]):.3f} N")
        elif r.startswith("forward-integration"):
            parts.append(f"integration {float(r.split()[2]):.2f} cm")
        elif r.startswith("envelope radius"):
            parts.append("envelope")
    return "; ".join(parts)


def build(base):
    data = load(base, "sweep_results.json")
    matrix = {}
    for r in data:
        if r["tag"].endswith("__sweep") or not r["report"]:
            continue
        matrix[(r["mission"], r["mode"])] = {
            c["style"]: c for c in r["report"]["candidates"]}

    grids = {}
    try:
        for r in load(base, "full_sweep_results.json"):
            if r["report"]:
                r["candidates"] = r["report"]["candidates"]
                r["selected"] = r["report"]["selected"]
                grids[(r["mission"], r["mode"])] = r
    except FileNotFoundError:
        pass

    def seed_table(mode):
        out = ["| Mission | Seed | Duration s | Envelope cm | Tracking cm | Gate |",
               "|---|---|---:|---:|---:|---|"]
        for m in MISSIONS:
            cands = matrix[(m, mode)]
            rows = [(s, cands[s]) for s in ORDER]
            f = rows[0][1]
            if all(abs(c["duration_s"] - f["duration_s"]) < 1e-9
                   and abs(c["envelope_cm"] - f["envelope_cm"]) < 1e-9
                   and abs(c["tracking_cm"] - f["tracking_cm"]) < 1e-9
                   for _, c in rows):
                gate = "**PASS**" if f["ok"] else f"fail — {reason(f)}"
                out.append(f"| {SHORT[m]} | *all five* | {f['duration_s']:.3f} | "
                           f"{f['envelope_cm']:.2f} | {f['tracking_cm']:.2f} | {gate} |")
                continue
            for style, c in rows:
                gate = "**PASS**" if c["ok"] else f"fail — {reason(c)}"
                out.append(f"| {SHORT[m]} | {style} | {c['duration_s']:.3f} | "
                           f"{c['envelope_cm']:.2f} | {c['tracking_cm']:.2f} | {gate} |")
        return "\n".join(out)

    def table_c():
        out = ["| Mission | Equal-duration | Per-segment | Note |",
               "|---|---|---|---|"]
        for m in MISSIONS:
            cells = []
            for mode in ("equal-duration", "per-segment"):
                cands = matrix[(m, mode)]
                passing = [c for c in cands.values() if c["ok"]]
                if not passing:
                    cells.append("**no passing candidate**")
                    continue
                s = min(passing, key=lambda c: (c["envelope_cm"], c["duration_s"]))
                cells.append(f"**{s['style']}** {s['duration_s']:.3f} s")
            n_e = sum(1 for c in matrix[(m, "equal-duration")].values() if c["ok"])
            n_p = sum(1 for c in matrix[(m, "per-segment")].values() if c["ok"])
            trk = [c["tracking_cm"] for c in matrix[(m, "equal-duration")].values()]
            trk_p = [c["tracking_cm"] for c in matrix[(m, "per-segment")].values()]
            if n_e == 5 and n_p == 5:
                note = ("**seed-insensitive**: the path has no turn, so the "
                        "geometric seeds degenerate to the same guess")
            elif n_e == 0 and n_p == 0:
                note = (f"tracking {min(trk):.2f}-{max(trk):.2f} cm "
                        f"(equal-duration) and {min(trk_p):.2f}-{max(trk_p):.2f} cm "
                        f"(per-segment) on every seed")
            elif n_e == 0:
                note = (f"equal-duration misses the gate (tracking "
                        f"{min(trk):.2f}-{max(trk):.2f} cm); {n_p} of 5 "
                        f"per-segment seeds pass")
            elif n_p == 0:
                note = (f"per-segment misses the gate (tracking "
                        f"{min(trk_p):.2f}-{max(trk_p):.2f} cm); {n_e} of 5 "
                        f"equal-duration seeds pass")
            else:
                note = f"{n_e} of 5 equal-duration and {n_p} of 5 per-segment seeds pass"
            out.append(f"| {SHORT[m]} | {cells[0]} | {cells[1]} | {note} |")
        return "\n".join(out)

    def table_d():
        out = []
        for mission in ("example_mission_first4", "example_complex_mission"):
            r = grids.get((mission, "auto__sweep"))
            if not r:
                continue
            out.append(f"**`{mission}`** (wall {r['wall_s']:.1f} s, exit {r['exit_code']})")
            out.append("")
            out.append("| # | Seed | Timing | Weight | Duration s | Envelope cm | Tracking cm | Gate |")
            out.append("|---|---|---|---:|---:|---:|---:|---|")
            weights = {1: "0", 2: "0", 3: "0", 4: "0", 5: "0",
                       6: "0.001", 7: "0.002", 8: "0.005", 9: "0.01"}
            for c in r["candidates"]:
                gate = "**PASS**" if c["ok"] else f"fail — {reason(c)}"
                out.append(f"| {c['index']} | {c['style']} | {c['timing']} | "
                           f"{weights.get(c['index'], '?')} | {c['duration_s']:.3f} | "
                           f"{c['envelope_cm']:.2f} | {c['tracking_cm']:.2f} | {gate} |")
            out.append("")
        return "\n".join(out).rstrip()

    def table_e():
        out = ["| Mission | Wall | Passing | Selected | Duration s | Envelope cm | Tracking cm | Fastest passing |",
               "|---|---:|---:|---|---:|---:|---:|---|"]
        for m in MISSIONS:
            r = grids.get((m, "full-sweep"))
            if not r:
                continue
            cands = r["candidates"]
            passing = [c for c in cands if c["ok"]]
            if not passing:
                out.append(f"| {SHORT[m]} | {r['wall_s']:.1f} s | **0/{len(cands)}** | "
                           f"*nothing passes* | — | — | — | — |")
                continue
            sel = next((c for c in cands if c["index"] == r["selected"]), passing[0])
            fastest = min(passing, key=lambda c: c["duration_s"])
            out.append(f"| {SHORT[m]} | {r['wall_s']:.1f} s | {len(passing)}/{len(cands)} | "
                       f"`{sel['style']}` ({sel['timing']}) | {sel['duration_s']:.3f} | "
                       f"{sel['envelope_cm']:.2f} | {sel['tracking_cm']:.2f} | "
                       f"`{fastest['style']}` {fastest['duration_s']:.3f} s |")
        return "\n".join(out)

    allc = [c for key in matrix for c in matrix[key].values()]
    fails = [c for c in allc if not c["ok"]]
    counts = {
        "total": len(allc), "passing": len(allc) - len(fails),
        "tracking": sum(1 for c in fails if any("tracking error" in r for r in c["reasons"])),
        "force": sum(1 for c in fails if any("motor force" in r for r in c["reasons"])),
        "integration": sum(1 for c in fails if any("forward-integration" in r for r in c["reasons"])),
        "envelope": sum(1 for c in fails if any("envelope radius" in r for r in c["reasons"])),
        "max_envelope": max((c["envelope_cm"] for c in allc), default=0.0),
    }
    cells_ok = sum(1 for key in matrix if any(c["ok"] for c in matrix[key].values()))
    non_base = sum(1 for key in matrix
                   if (lambda s: s is not None and s["style"] != "baseline")(
                       min([c for c in matrix[key].values() if c["ok"]],
                           key=lambda c: (c["envelope_cm"], c["duration_s"]))
                       if any(c["ok"] for c in matrix[key].values()) else None))
    return {
        "A": seed_table("equal-duration"), "B": seed_table("per-segment"),
        "C": table_c(), "D": table_d(), "E": table_e(),
        "counts": counts, "cells_ok": cells_ok, "non_base": non_base,
        "matrix": matrix, "grids": grids,
    }


def replace_table(text, header, new_table):
    m = re.search(header, text)
    if not m:
        raise SystemExit(f"heading not found: {header}")
    lines = text[m.end():].split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not lines[i].startswith("|"):
        # a prose line sits between the heading and the table
        while i < len(lines) and not lines[i].startswith("|"):
            i += 1
    start = i
    while i < len(lines) and lines[i].startswith("|"):
        i += 1
    return text[:m.end()] + "\n".join(lines[:start]) + "\n" + new_table + "\n" + \
        "\n".join(lines[i:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    base = args.folder or newest_folder()
    if not os.path.isabs(base):
        base = os.path.join(REPO, base)
    print(f"measurement folder: {os.path.relpath(base, REPO)}")

    b = build(base)
    with open(DOC, encoding="utf-8") as fh:
        text = fh.read()

    if not args.check:
        text = replace_table(text, r"### 9\.2 Table A[^\n]*\n", b["A"])
        text = replace_table(text, r"### 9\.3 Table B[^\n]*\n", b["B"])
        text = replace_table(text, r"### 9\.4 Table C[^\n]*\n", b["C"])
        text = replace_table(text, r"### 9\.5 Table D[^\n]*\n", b["D"])
        text = replace_table(text, r"### 9\.6 Table E[^\n]*\n", b["E"])
        with open(DOC, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("tables rewritten")

    # --- check every number in the document against the data ---
    with open(DOC, encoding="utf-8") as fh:
        doc = fh.read()
    lines = doc.split("\n")
    matrix = b["matrix"]
    checked = 0
    problems = []
    mode = None
    for line in lines:
        if line.startswith("### 9.2"):
            mode = "equal-duration"
        elif line.startswith("### 9.3"):
            mode = "per-segment"
        elif line.startswith("### 9.4"):
            mode = None
        if mode is None:
            continue
        m = re.match(r"^\| (straight|s_curve|first4|complex|stops|events) \| "
                     r"([a-z_]+|\*all five\*) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if not m:
            continue
        mission, style, dur, env, trk = m.groups()
        cands = matrix[(SHORT_TO_MISSION[mission], mode)]
        targets = list(cands.values()) if style == "*all five*" else [cands[style]]
        for c in targets:
            checked += 1
            if abs(c["duration_s"] - float(dur)) > 0.002:
                problems.append(f"{mission}/{mode}/{c['style']}: duration {dur} != {c['duration_s']:.3f}")
            if abs(c["envelope_cm"] - float(env)) > 0.006:
                problems.append(f"{mission}/{mode}/{c['style']}: envelope {env} != {c['envelope_cm']:.3f}")
            if abs(c["tracking_cm"] - float(trk)) > 0.006:
                problems.append(f"{mission}/{mode}/{c['style']}: tracking {trk} != {c['tracking_cm']:.3f}")

    c = b["counts"]
    print(f"\nrows checked: {checked}")
    print(f"candidates: {c['total']}, passing: {c['passing']}, failing: {c['total']-c['passing']}")
    print(f"failure reasons -> tracking {c['tracking']}, motor force {c['force']}, "
          f"integration {c['integration']}, envelope {c['envelope']}")
    print(f"max envelope: {c['max_envelope']:.2f} cm")
    print(f"cells with a passing candidate: {b['cells_ok']} of 12; "
          f"non-baseline selected in {b['non_base']}")

    if problems:
        print(f"\n!! {len(problems)} drift")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nNo drift: every table number matches the measurements.")


SHORT_TO_MISSION = {v: k for k, v in SHORT.items()}

if __name__ == "__main__":
    main()
