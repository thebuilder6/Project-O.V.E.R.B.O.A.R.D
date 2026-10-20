"""
Answer Q3 and Q10 from the candidate matrix.

Q3 asks whether solving more than one trajectory buys anything, and if so
what. Q10 asks whether "smallest envelope" is the right thing to minimise
when choosing which of the passing candidates to export.

Both are answered from the same file: every candidate in the matrix carries
its score vector, so the selection rule can be re-run offline without
solving anything again. Nothing here changes the planner; it reads the
measurements and reports.

    python docs/measurements/2026-10-20/matrix/analyse_matrix.py

    --folder PATH   the measurement folder to read (default: this one)
    --mode MODE     one mode to report on (default: all of them)
"""

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(HERE))))

MISSIONS = [
    "example_straight",
    "example_s_curve",
    "example_mission_first4",
    "example_complex_mission",
    "example_with_stops",
    "example_with_events",
]

# The selection rules Q10 compares. Each takes a candidate and returns the
# key it is ranked by, smallest first.
RULES = {
    "A envelope (the rule today)": lambda c: (c["envelope_cm"], c["duration_s"]),
    "B tracking error": lambda c: (c["tracking_cm"], c["duration_s"]),
    "C excess over the limit": lambda c: (excess_cm(c), c["duration_s"]),
    "D duration": lambda c: (c["duration_s"],),
    "E envelope, then tracking": lambda c: (c["envelope_cm"], c["tracking_cm"],
                                            c["duration_s"]),
}


def excess_cm(candidate):
    """The biggest excess of the envelope radius over the local threshold.

    Negative means the envelope is inside the limit by that much. The report
    carries the number, but it is computed here from the segments as well, so
    the analysis also works on measurements taken before the field existed.
    """
    value = candidate.get("max_excess_cm")
    if value is not None:
        return float(value)
    segments = candidate.get("segments") or []
    return max((float(segment.get("envelope_m", 0.0))
                - float(segment.get("limit_m", 0.0))
                for segment in segments), default=0.0)


def load(folder):
    with open(os.path.join(folder, "matrix_results.json"), encoding="utf-8") as fh:
        return json.load(fh)


def groups(rows):
    """One entry per (mission, mode) with the candidates it produced."""
    out = []
    for row in rows:
        report = row.get("report")
        if not report:
            continue
        candidates = report["candidates"]
        selected = next((c for c in candidates
                         if c["index"] == report.get("selected")), None)
        out.append({
            "mission": row["mission"],
            "mode": row["mode"],
            "wall_s": row.get("wall_s"),
            "candidates": candidates,
            "selected": selected,
            "passing": [c for c in candidates if c["ok"]],
        })
    return out


def verdict(candidates):
    """The fastest passing candidate, or the reason there is none."""
    passing = [c for c in candidates if c["ok"]]
    if not passing:
        return "fail"
    best = min(passing, key=lambda c: c["duration_s"])
    return f"PASS {best['duration_s']:.2f}s"


def q3(groups_):
    """Is the extra candidate worth anything, and what does it buy?"""
    print("=" * 78)
    print("Q3  what does the second (and the fifth) candidate buy?")
    print("=" * 78)
    print()
    print("The default set is two TIMING MODELS of the baseline seed. The other")
    print("seeds only appear from --candidates 3 on, and the sweep adds only")
    print("baseline seeds, so the 'seeds' mode is where the multi-start is")
    print("measured at all.")
    print()

    for mode in ("default", "seeds", "sweep", "sweep-self-consistent"):
        rows = [g for g in groups_ if g["mode"] == mode]
        if not rows:
            continue
        print(f"--- mode: {mode} ---")
        print(f"{'mission':<26} {'baseline/equal':<16} {'+ per-segment':<16} "
              f"{'+ weight grid':<16} {'wall s':<8}")
        layer_counts = {1: 0, 2: 0, 3: 0}
        for g in rows:
            cands = g["candidates"]
            l1 = [c for c in cands
                  if c["style"] == "baseline" and c["timing"] == "equal-duration"
                  and c["accuracy_weight"] == 0.0]
            l2 = [c for c in cands
                  if c["style"] == "baseline" and c["accuracy_weight"] == 0.0]
            l3 = cands
            for level, subset in ((1, l1), (2, l2), (3, l3)):
                if "PASS" in verdict(subset):
                    layer_counts[level] += 1
            print(f"{g['mission']:<26} {verdict(l1):<16} {verdict(l2):<16} "
                  f"{verdict(l3):<16} {str(g['wall_s']):<8}")
        total = len(rows)
        print(f"{'missions with a passer':<26} {layer_counts[1]:>2}/{total:<13} "
              f"{layer_counts[2]:>2}/{total:<13} {layer_counts[3]:>2}/{total}")
        print()

    # The insurance question: would the baseline alone have passed, and if
    # not, what rescued it?
    print("--- the insurance, per mode ---")
    for mode in ("default", "seeds", "sweep", "sweep-self-consistent"):
        rows = [g for g in groups_ if g["mode"] == mode]
        if not rows:
            continue
        rescued = []
        for g in rows:
            # The reference is the baseline seed at the base weight: the
            # candidate the default set would have run on its own.
            base = [c for c in g["candidates"]
                    if c["style"] == "baseline" and c["accuracy_weight"] == 0.0
                    and c["timing"] == "equal-duration"]
            if g["passing"] and not any(c["ok"] for c in base):
                best = min(g["passing"], key=lambda c: c["duration_s"])
                rescued.append((g, best))
        if not rescued:
            print(f"{mode:<24} no mission needs more than the baseline")
            continue
        print(f"{mode:<24} {len(rescued)} of {len(rows)} missions need more:")
        for g, best in rescued:
            what = []
            if best["timing"] != "equal-duration":
                what.append("the other timing model")
            if best["style"] != "baseline":
                what.append(f"the {best['style']} seed")
            if best["accuracy_weight"]:
                what.append(f"accuracy weight {best['accuracy_weight']}")
            print(f"    {g['mission']:<26} rescued by {best['style']:<10} "
                  f"{best['timing']:<14} w={best['accuracy_weight']:<6} "
                  f"{best['duration_s']:6.3f} s  <- {', '.join(what) or 'a weight'}")
    print()

    # The seeds: do they land anywhere the baseline does not?
    print("--- the seeds, where they are in the matrix (mode 'seeds') ---")
    for g in groups_:
        if g["mode"] != "seeds":
            continue
        by_style = defaultdict(list)
        for c in g["candidates"]:
            by_style[(c["style"], c["timing"])].append(c)
        print(f"  {g['mission']}")
        for key in sorted(by_style):
            c = by_style[key][0]
            print(f"      {c['style']:<10} {c['timing']:<14} "
                  f"{c['duration_s']:7.3f} s  trk {c['tracking_cm']:5.2f} cm  "
                  f"env {c['envelope_cm']:5.2f} cm  "
                  f"{'PASS' if c['ok'] else 'fail'}")
    print()


def q10(groups_):
    """Is the envelope the right thing to minimise?"""
    print("=" * 78)
    print("Q10  is 'smallest envelope' the right selection objective?")
    print("=" * 78)
    print()
    everything = [c for g in groups_ for c in g["candidates"]]
    if not everything:
        print("no candidates")
        return
    print(f"candidates: {len(everything)}")

    def corr(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        return num / den if den else 0.0

    durations = [c["duration_s"] for c in everything]
    tubes = [c["envelope_cm"] for c in everything]
    trackings = [c["tracking_cm"] for c in everything]
    print(f"corr(duration, envelope) = {corr(durations, tubes):+.3f}")
    print(f"corr(duration, tracking) = {corr(durations, trackings):+.3f}")
    print(f"corr(envelope, tracking) = {corr(tubes, trackings):+.3f}")
    print()

    # What rejects a candidate?
    from collections import Counter
    reasons = Counter()
    for c in everything:
        if c["ok"]:
            reasons["PASS"] += 1
        for reason in c["reasons"]:
            reasons[reason.split()[0]] += 1
    print("what the candidates fail on:",
          ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    print()

    # The rules, side by side.
    print("--- the selection rules, over the groups that have a passer ---")
    header = f"{'mission':<24} {'mode':<22}"
    for name in RULES:
        header += f" | {name.split()[0]}"
    print(header)
    chosen = {name: [] for name in RULES}
    for g in groups_:
        if not g["passing"]:
            continue
        line = f"{g['mission']:<24} {g['mode']:<22}"
        for name, key in RULES.items():
            pick = min(g["passing"], key=key)
            chosen[name].append(pick["duration_s"])
            line += f" | {pick['duration_s']:6.3f}s"
        print(line)
    print()
    base = chosen["A envelope (the rule today)"]
    for name, values in chosen.items():
        if name.startswith("A"):
            continue
        different = sum(1 for a, b in zip(base, values) if abs(a - b) > 1e-9)
        slower = sum(1 for a, b in zip(base, values) if b > a + 1e-9)
        faster = sum(1 for a, b in zip(base, values) if b < a - 1e-9)
        print(f"{name:<30} differs in {different}/{len(base)} groups, "
              f"slower in {slower}, faster in {faster}")
    print()

    # The per-sample excess against the local threshold.
    print("--- does the excess over the local limit ever differ from the "
          "envelope? ---")
    limits = Counter()
    for c in everything:
        for segment in c.get("segments") or []:
            limits[segment["limit_m"]] += 1
    print("segment thresholds in the matrix:", dict(limits))
    print("(every mission uses the default 2 cm, so the absolute envelope and")
    print(" the per-sample excess are the same ordering; a mission with a")
    print(" tight error_threshold is needed for them to disagree)")
    print()


def markdown_table(groups_):
    """The result table the README carries, in markdown.

    `pass` is the number of candidates that clear every gate; `track/env` is
    the best candidate's tracking error and envelope radius in centimetres.
    The best candidate is the one with the smallest tracking error, which is
    what the README has always meant by it.
    """
    modes = []
    for g in groups_:
        if g["mode"] not in modes:
            modes.append(g["mode"])
    short = {"example_straight": "straight", "example_s_curve": "s_curve",
             "example_mission_first4": "first4",
             "example_complex_mission": "complex",
             "example_with_stops": "with_stops",
             "example_with_events": "with_events"}
    header = "| mission | " + " | ".join(modes) + " |"
    rule = "|---" * (len(modes) + 1) + "|"
    lines = [header, rule]
    for mission in MISSIONS:
        cells = []
        for mode in modes:
            row = next((g for g in groups_
                        if g["mission"] == mission and g["mode"] == mode), None)
            if row is None or not row["candidates"]:
                cells.append("-")
                continue
            candidates = row["candidates"]
            passing = sum(1 for c in candidates if c["ok"])
            best = min(candidates, key=lambda c: c["tracking_cm"])
            cells.append(f"{passing}/{len(candidates)}, "
                         f"{best['tracking_cm']:.2f} / {best['envelope_cm']:.2f} cm")
        lines.append(f"| {short.get(mission, mission)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tree_digests():
    """The fingerprint of the files the numbers depend on.

    The README carries it so a reader can tell which tree the measurements
    describe. It is generated, because a hand-kept list goes stale the
    moment anything is edited.
    """
    import hashlib
    files = [
        "optimizer.py", "immrax_validator.py", "validator.py",
        "robot_model.py", "main.py",
        "pybricks_code/ramsete_fast.py", "pybricks_code/ramsete.py",
        "pybricks_code/ramsete_compact.py",
        "pybricks_code/ramsete_compact_events.py",
    ]
    lines = []
    for name in files:
        path = os.path.join(REPO, name)
        digest = hashlib.md5(open(path, "rb").read()).hexdigest().upper()
        lines.append(f"{name:<42} MD5 {digest}")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    folder = HERE
    mode = None
    table = False
    tree = False
    while argv:
        if argv[0] == "--folder":
            folder = argv[1]
            argv = argv[2:]
        elif argv[0] == "--mode":
            mode = argv[1]
            argv = argv[2:]
        elif argv[0] == "--table":
            table = True
            argv = argv[1:]
        elif argv[0] == "--tree":
            tree = True
            argv = argv[1:]
        else:
            print(__doc__)
            return 1
    if not os.path.isabs(folder):
        folder = os.path.join(REPO, folder)
    if tree:
        print(tree_digests())
        return 0
    rows = load(folder)
    groups_ = groups(rows)
    if mode:
        groups_ = [g for g in groups_ if g["mode"] == mode]
    if table:
        print(markdown_table(groups_))
        return 0
    print(f"measurement folder: {os.path.relpath(folder, REPO)}")
    print()
    q3(groups_)
    q10(groups_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
