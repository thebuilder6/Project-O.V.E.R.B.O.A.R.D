"""The counts the section 9 prose quotes."""
import json
import os

BASE = r"docs\measurements\2026-10-20-post-ki028"
data = json.load(open(os.path.join(BASE, "sweep_results.json"), encoding="utf-8"))
matrix = {}
for r in data:
    if r["tag"].endswith("__sweep") or not r["report"]:
        continue
    matrix[(r["mission"], r["mode"])] = {c["style"]: c for c in r["report"]["candidates"]}

for mode in ("equal-duration", "per-segment"):
    cells = [k for k in matrix if k[1] == mode]
    n = sum(1 for k in cells for c in matrix[k].values() if c["ok"])
    print(f"{mode}: {n} passing of {len(cells)*5}")

cells_ok = [k for k in matrix if any(c["ok"] for c in matrix[k].values())]
base = 0
nonbase = []
for k in cells_ok:
    passing = [c for c in matrix[k].values() if c["ok"]]
    s = min(passing, key=lambda c: (c["envelope_cm"], c["duration_s"]))
    if s["style"] == "baseline":
        base += 1
    else:
        nonbase.append((k, s["style"], s["duration_s"]))
print(f"cells with a passing candidate: {len(cells_ok)} of 12")
print(f"  baseline selected in {base}, non-baseline in {len(nonbase)}: "
      + ", ".join(f"{k[0]}/{k[1]}={s} {d:.3f}s" for k, s, d in nonbase))

allc = [c for k in matrix for c in matrix[k].values()]
fails = [c for c in allc if not c["ok"]]
print(f"\nmatrix: {len(allc)} candidates, {len(allc)-len(fails)} passing, {len(fails)} failing")
print(f"  tracking {sum(1 for c in fails if any('tracking error' in r for r in c['reasons']))}, "
      f"motor force {sum(1 for c in fails if any('motor force' in r for r in c['reasons']))}, "
      f"integration {sum(1 for c in fails if any('forward-integration' in r for r in c['reasons']))}, "
      f"envelope {sum(1 for c in fails if any('envelope radius' in r for r in c['reasons']))}")
print(f"  max envelope {max(c['envelope_cm'] for c in allc):.2f} cm")

full = json.load(open(os.path.join(BASE, "full_sweep_results.json"), encoding="utf-8"))
tot = sum(len(r["report"]["candidates"]) for r in full)
ps = sum(sum(1 for c in r["report"]["candidates"] if c["ok"]) for r in full)
print(f"full sweep: {ps} of {tot} pass ({100*ps//tot} %)")
for r in full:
    cands = r["report"]["candidates"]
    p = [c for c in cands if c["ok"]]
    print(f"  {r['mission']:<26} {r['mode']:<12} {len(p)}/{len(cands)} pass")
