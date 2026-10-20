"""Q12: the Multi-Verse pipeline against the simple solver.

Reads the Multi-Verse run and the matrix run and puts the two side by side
per mission: duration, gate outcome, wall time.
"""

import json
import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record") != "meta":
                rows.append(record)
    return rows


mv = load(sys.argv[1])
mx = load(sys.argv[2])

simple = {}
for row in mx:
    if (row.get("style") == "baseline" and row.get("guess", "stock") == "stock"
            and row.get("weight") == 0 and row.get("samples") == 10
            and row.get("timing") == "equal-duration"):
        simple[row["mission"]] = row

print(f"{'mission':<30}{'simple s':>10}{'MV s':>9}{'delta':>9}{'simple ok':>11}"
      f"{'MV ok':>8}{'simple wall':>13}{'MV wall':>10}")
wins = losses = ties = 0
mv_pass = simple_pass = 0
total = 0
for row in mv:
    if row.get("weight") != 0:
        continue
    base = simple.get(row["mission"])
    if base is None:
        continue
    total += 1
    d_mv = row["duration_s"]
    d_simple = base["duration_s"]
    delta = (d_mv - d_simple) / d_simple * 100 if d_simple else 0.0
    if abs(delta) < 1.0:
        ties += 1
    elif delta < 0:
        wins += 1
    else:
        losses += 1
    mv_pass += 1 if row.get("ok") else 0
    simple_pass += 1 if base.get("ok") else 0
    print(f"{row['mission'][:29]:<30}{d_simple:10.3f}{d_mv:9.3f}{delta:+8.1f}%"
          f"{str(base.get('ok')):>11}{str(row.get('ok')):>8}"
          f"{base.get('solve_s') or 0:13.2f}{row.get('solve_s') or 0:10.2f}")

print(f"\nmissions compared: {total}")
print(f"Multi-Verse faster on {wins}, slower on {losses}, within 1 % on {ties}")
print(f"passing: simple {simple_pass}/{total}, multi-verse {mv_pass}/{total}")
