"""Compare two measurement files candidate by candidate.

Used for the self-consistent comparison: the same missions, the same seeds,
the same tree, with and without the flag.

    python research/compare_runs.py <file A> <file B>
"""

import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record") != "meta":
                rows.append(record)
    return rows


a_rows = load(sys.argv[1])
b_rows = load(sys.argv[2])
label_a = sys.argv[3] if len(sys.argv) > 3 else "A"
label_b = sys.argv[4] if len(sys.argv) > 4 else "B"

key = lambda r: (r["mission"], r["label"])
a_map = {key(r): r for r in a_rows}
b_map = {key(r): r for r in b_rows}
common = sorted(set(a_map) & set(b_map))
print(f"matched candidates: {len(common)} "
      f"({label_a}: {len(a_rows)}, {label_b}: {len(b_rows)})")

a_pass = sum(1 for k in common if a_map[k].get("ok"))
b_pass = sum(1 for k in common if b_map[k].get("ok"))
print(f"passing: {label_a} {a_pass}/{len(common)}, "
      f"{label_b} {b_pass}/{len(common)}")

deltas = []
a_only = b_only = both = 0
for k in common:
    a, b = a_map[k], b_map[k]
    if a.get("ok") and not b.get("ok"):
        a_only += 1
    elif b.get("ok") and not a.get("ok"):
        b_only += 1
    elif a.get("ok") and b.get("ok"):
        both += 1
    da = a.get("duration_s") or 0
    db = b.get("duration_s") or 0
    if da and db:
        deltas.append((db - da) / da * 100)

print(f"pass flags: {label_a} only {a_only}, {label_b} only {b_only}, both {both}")
if deltas:
    deltas.sort()
    median = deltas[len(deltas) // 2]
    print(f"duration delta ({label_b} - {label_a}): median {median:+.1f} %, "
          f"min {deltas[0]:+.1f} %, max {deltas[-1]:+.1f} %")

# Per seed and timing model: the pass rate in each mode.
print(f"\n{'seed':<34}{'pass A':>9}{'pass B':>9}{'trk A cm':>11}{'trk B cm':>11}")
buckets = defaultdict(list)
for k in common:
    buckets[k[1]].append((a_map[k], b_map[k]))
for name in sorted(buckets):
    group = buckets[name]
    pa = sum(1 for a, b in group if a.get("ok"))
    pb = sum(1 for a, b in group if b.get("ok"))
    ta = sorted((a.get("tracking_m") or 0) for a, b in group)
    tb = sorted((b.get("tracking_m") or 0) for a, b in group)
    print(f"{name:<34}{pa:>4}/{len(group):<4}{pb:>4}/{len(group):<4}"
          f"{ta[len(ta) // 2] * 100:>11.2f}{tb[len(tb) // 2] * 100:>11.2f}")

# Per mission: the best passing candidate in each mode.
print(f"\n{'mission':<28}{'A best':>10}{'B best':>10}{'A ok':>7}{'B ok':>7}")
missions = defaultdict(list)
for k in common:
    missions[k[0]].append((a_map[k], b_map[k]))
for mission in sorted(missions):
    group = missions[mission]
    a_best = min((a for a, b in group if a.get("ok")),
                 key=lambda r: r.get("duration_s", float("inf")), default=None)
    b_best = min((b for a, b in group if b.get("ok")),
                 key=lambda r: r.get("duration_s", float("inf")), default=None)
    print(f"{mission[:27]:<28}"
          f"{(a_best['duration_s'] if a_best else 0):>10.3f}"
          f"{(b_best['duration_s'] if b_best else 0):>10.3f}"
          f"{str(bool(a_best)):>7}{str(bool(b_best)):>7}")
