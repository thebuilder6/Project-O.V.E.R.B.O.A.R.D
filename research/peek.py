"""Print a measurement file as a compact table (quick inspection)."""

from __future__ import annotations

import json
import sys


def main() -> None:
    path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record") == "meta":
                continue
            rows.append(record)
    header = (f"{'mission':<22}{'label':<34}{'ok':<6}{'dur':>8}{'env':>8}"
              f"{'trk':>8}{'it':>7}{'solve':>8}  reasons")
    print(header)
    for record in rows[:limit]:
        reasons = "; ".join(record.get("reasons") or [])[:70]
        print(f"{str(record.get('mission_file'))[:21]:<22}"
              f"{str(record.get('label'))[:33]:<34}"
              f"{str(record.get('ok')):<6}"
              f"{record.get('duration_s', 0):8.3f}"
              f"{(record.get('tube_m') or 0) * 100:8.2f}"
              f"{(record.get('tracking_m') or 0) * 100:8.2f}"
              f"{record.get('iter_count', -1):7d}"
              f"{record.get('solve_s') or 0:8.2f}  {reasons}")
    print(f"{len(rows)} records")


if __name__ == "__main__":
    main()
