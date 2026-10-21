"""Offline analysis of a measurement run.

Reads the JSONL the harness writes and answers the questions in
``docs/HEURISTICS_AND_VERIFICATION.md`` Part IV from data.

Usage::

    python research/analyze.py summary research/results/matrix.jsonl
    python research/analyze.py seeds   research/results/matrix.jsonl
    python research/analyze.py gates   research/results/matrix.jsonl
    python research/analyze.py weights research/results/matrix.jsonl
    python research/analyze.py select  research/results/matrix.jsonl
    python research/analyze.py compare research/results/matrix.jsonl --a spline --b baseline
    python research/analyze.py critic  research/results/matrix.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Row = Dict[str, Any]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load(path: str) -> Tuple[Dict[str, Any], List[Row]]:
    meta: Dict[str, Any] = {}
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record") == "meta":
                meta = record
            else:
                rows.append(record)
    return meta, rows


def _finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: Iterable[float]) -> Optional[float]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _stdev(values: Iterable[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, or None when it is undefined."""
    if len(xs) < 3:
        return None
    rx = _ranks(list(xs))
    ry = _ranks(list(ys))
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and not math.isinf(x) and not math.isinf(y)]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in xs))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# --------------------------------------------------------------------------
# Grouping helpers
# --------------------------------------------------------------------------

def group_by_mission(rows: Sequence[Row]) -> Dict[str, List[Row]]:
    out: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        out[row["mission"]].append(row)
    return out


def is_baseline(row: Row) -> bool:
    return row.get("style") == "baseline" and row.get("guess", "stock") == "stock"


def candidate_key(row: Row) -> Tuple:
    return (row.get("style"), row.get("guess", "stock"), row.get("timing"),
            row.get("weight"), row.get("samples"))


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def report_summary(meta: Dict[str, Any], rows: Sequence[Row]) -> None:
    print(f"records: {len(rows)}  missions: {len(group_by_mission(rows))}")
    if meta:
        print(f"config: {meta.get('config')}  samples: {meta.get('samples')}  "
              f"specs: {meta.get('specs')}  started: {meta.get('started')}")
        prov = meta.get("provenance", {})
        for name, digest in prov.items():
            print(f"  {name:<24} {digest}")
    trees: Dict[str, int] = defaultdict(int)
    for row in rows:
        trees[row.get("tree", "unknown")] += 1
    if len(trees) > 1:
        print("WARNING: the run spans several tree states:")
        for token, count in sorted(trees.items()):
            print(f"  {token}: {count} records")
    solved = [r for r in rows if r.get("converged")]
    passing = [r for r in rows if r.get("ok")]
    print(f"converged: {len(solved)}/{len(rows)} ({100 * len(solved) / max(1, len(rows)):.1f}%)")
    print(f"passing:   {len(passing)}/{len(rows)} ({100 * len(passing) / max(1, len(rows)):.1f}%)")
    print(f"duration:  median {_fmt(_median([_finite(r.get('duration_s')) for r in rows]))} s"
          f"  mean {_fmt(_mean([_finite(r.get('duration_s')) for r in rows]))} s")
    print(f"envelope:  median {_fmt(_median([_finite(r.get('tube_m')) for r in rows]) * 100 if _median([_finite(r.get('tube_m')) for r in rows]) else None, 2)} cm")
    print(f"tracking:  median {_fmt(_median([_finite(r.get('tracking_m')) for r in rows]) * 100 if _median([_finite(r.get('tracking_m')) for r in rows]) else None, 2)} cm")
    print(f"solve wall: median {_fmt(_median([_finite(r.get('solve_s')) for r in rows]), 2)} s"
          f"  total {sum(_finite(r.get('solve_s')) or 0 for r in rows) / 60:.1f} min")


def report_seeds(rows: Sequence[Row]) -> None:
    """Q3/Q8: does a seed ever win, and does it ever rescue a gate?"""
    print("\n## Per seed and timing model\n")
    print("(stock seeds only; the extra guess families are compared separately)")
    rows = [r for r in rows if r.get("guess", "stock") == "stock"]
    print("| seed | timing | n | pass | pass rate (95% CI) | median dur | "
          "median env cm | median trk cm |")
    print("|---|---|---:|---:|---|---:|---:|---:|")
    buckets: Dict[Tuple[str, str], List[Row]] = defaultdict(list)
    for row in rows:
        if row.get("weight", 0) != 0 or row.get("samples", 10) != 10:
            continue
        if row.get("guess", "stock") != "stock":
            continue
        buckets[(row.get("style", "?"), row.get("timing", "?"))].append(row)
    for key in sorted(buckets):
        group = buckets[key]
        passing = sum(1 for r in group if r.get("ok"))
        low, high = wilson_interval(passing, len(group))
        print(f"| {key[0]} | {key[1]} | {len(group)} | {passing} | "
              f"{100 * passing / len(group):.0f}% ({100 * low:.0f}-{100 * high:.0f}%) | "
              f"{_fmt(_median([_finite(r.get('duration_s')) for r in group]))} | "
              f"{_fmt((_median([_finite(r.get('tube_m')) for r in group]) or 0) * 100, 2)} | "
              f"{_fmt((_median([_finite(r.get('tracking_m')) for r in group]) or 0) * 100, 2)} |")

    # Fastest per mission, and who produced it.
    fastest: Dict[str, Row] = {}
    for mission, group in group_by_mission(rows).items():
        candidates = [r for r in group
                      if r.get("weight", 0) == 0 and r.get("samples", 10) == 10
                      and _finite(r.get("duration_s")) is not None]
        if candidates:
            fastest[mission] = min(candidates, key=lambda r: r["duration_s"])
    winners: Dict[str, int] = defaultdict(int)
    for row in fastest.values():
        winners[f"{row.get('style')}/{row.get('timing')}"] += 1
    print(f"\nFastest candidate per mission (n={len(fastest)}): " +
          ", ".join(f"{k}: {v}" for k, v in sorted(winners.items())))

    # Gate escape: missions where the baseline fails but something else passes.
    baseline_pass = 0
    rescued = 0
    nothing_passed = 0
    rescued_by: Dict[str, int] = defaultdict(int)
    for mission, group in group_by_mission(rows).items():
        base = [r for r in group if is_baseline(r) and r.get("weight", 0) == 0]
        if not base:
            continue
        base_ok = any(r.get("ok") for r in base)
        others = [r for r in group if not is_baseline(r) or r.get("weight", 0) != 0]
        other_ok = [r for r in others if r.get("ok")]
        if base_ok:
            baseline_pass += 1
            if not other_ok:
                # The baseline is the only candidate that passed: it is
                # counted once, as a baseline success.
                pass
        elif other_ok:
            rescued += 1
            best = min(other_ok, key=lambda r: r.get("duration_s", float("inf")))
            rescued_by[f"{best.get('style')}/{best.get('timing')}"] += 1
        else:
            nothing_passed += 1
    total = baseline_pass + rescued + nothing_passed
    print(f"\nBaseline alone would have produced a passing trajectory on "
          f"{baseline_pass}/{total} missions "
          f"({100 * baseline_pass / max(1, total):.0f}%).")
    print(f"No candidate passed on {nothing_passed}/{total} missions.")
    print(f"A non-baseline candidate rescued {rescued}/{total} missions: " +
          (", ".join(f"{k}: {v}" for k, v in sorted(rescued_by.items())) or "none"))


def report_gates(rows: Sequence[Row]) -> None:
    """Which gate actually rejects candidates?"""
    print("\n## Failure reasons\n")
    counter: Dict[str, int] = defaultdict(int)
    gate_counter: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("ok"):
            continue
        reasons = row.get("reasons") or []
        for reason in reasons:
            key = reason.split(" ")[0]
            if "tracking" in reason:
                key = "tracking error"
            elif "motor force" in reason:
                key = "motor force"
            elif "integration" in reason:
                key = "forward integration"
            elif "envelope" in reason:
                key = "envelope radius"
            elif "slip" in reason:
                key = "wheel slip"
            elif "speed" in reason:
                key = "wheel speed"
            elif "constraint" in reason:
                key = "constraint violations"
            counter[key] += 1
        gate_counter["failing candidates"] += 1
    total = len(rows)
    for key, count in sorted(counter.items(), key=lambda kv: -kv[1]):
        print(f"| {key} | {count} | {100 * count / total:.1f}% |")
    tubes = [_finite(r.get("tube_m")) for r in rows]
    tubes = [t for t in tubes if t is not None]
    print(f"\nEnvelope radius: median {_fmt((_median(tubes) or 0) * 100, 2)} cm, "
          f"max {_fmt((max(tubes) if tubes else 0) * 100, 2)} cm (gate 2.00 cm)")
    track = [_finite(r.get("tracking_m")) for r in rows]
    track = [t for t in track if t is not None]
    print(f"Tracking error:  median {_fmt((_median(track) or 0) * 100, 2)} cm, "
          f"max {_fmt((max(track) if track else 0) * 100, 2)} cm (gate 0.50 cm)")
    print(f"Candidates with no envelope breach: "
          f"{sum(1 for r in rows if (r.get('tube_m') or 0) <= 0.02)}/{total}")


def report_weights(rows: Sequence[Row]) -> None:
    """Q6: is the accuracy weight a dial or a lottery?"""
    print("\n## Accuracy weight vs tracking error (Q6)\n")
    by_mission: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        if row.get("style") != "baseline" or row.get("samples", 10) != 10:
            continue
        by_mission[row["mission"]].append(row)

    pooled_x: List[float] = []
    pooled_y: List[float] = []
    per_mission: List[Tuple[str, Optional[float], Optional[float]]] = []
    flips = 0
    compared = 0
    for mission, group in by_mission.items():
        for timing in ("equal-duration", "per-segment"):
            series = sorted([r for r in group if r.get("timing") == timing],
                            key=lambda r: r.get("weight", 0))
            xs = [r.get("weight", 0) for r in series]
            ys = [(_finite(r.get("tracking_m")) or 0.0) for r in series]
            if len(series) >= 3:
                per_mission.append((f"{mission}/{timing}", spearman(xs, ys), pearson(xs, ys)))
                pooled_x.extend(xs)
                pooled_y.extend(ys)
            for a, b in zip(series, series[1:]):
                compared += 1
                if (b.get("duration_s") or 0) < (a.get("duration_s") or 0) and \
                        (b.get("tracking_m") or 0) > (a.get("tracking_m") or 0):
                    flips += 1
    print(f"Pooled Spearman(weight, tracking) = {_fmt(spearman(pooled_x, pooled_y), 3)}")
    print(f"Pooled Pearson (weight, tracking) = {_fmt(pearson(pooled_x, pooled_y), 3)}")
    rho = [r for _, r, _ in per_mission if r is not None]
    print(f"Per-mission Spearman: median {_fmt(_median(rho), 3)}, "
          f"{sum(1 for r in rho if r < 0)}/{len(rho)} missions negative "
          f"(tracking got worse as the weight grew)")
    print(f"Non-monotone steps (higher weight, slower AND less accurate): "
          f"{flips}/{compared}")

    print("\n### Tracking error by weight (pooled, baseline seed)\n")
    print("| weight | n | median tracking cm | median duration s | pass rate |")
    print("|---:|---:|---:|---:|---:|")
    buckets: Dict[float, List[Row]] = defaultdict(list)
    for row in rows:
        if row.get("style") == "baseline" and row.get("samples", 10) == 10:
            buckets[row.get("weight", 0)].append(row)
    for weight in sorted(buckets):
        group = buckets[weight]
        passing = sum(1 for r in group if r.get("ok"))
        print(f"| {weight:g} | {len(group)} | "
              f"{_fmt((_median([_finite(r.get('tracking_m')) for r in group]) or 0) * 100, 2)} | "
              f"{_fmt(_median([_finite(r.get('duration_s')) for r in group]))} | "
              f"{100 * passing / len(group):.0f}% |")


def _select(attempts: Sequence[Row], rule: str) -> Optional[Row]:
    """Reproduce a selection rule over a candidate list."""
    eligible = [a for a in attempts if a.get("converged")]
    if not eligible:
        return None
    passing = [a for a in eligible if a.get("ok")]
    if rule == "current":
        pool = passing
        key = lambda a: (a.get("tube_m", float("inf")), a.get("duration_s", float("inf")))
    elif rule == "fastest":
        pool = passing
        key = lambda a: (a.get("duration_s", float("inf")), a.get("tube_m", float("inf")))
    elif rule == "tracking_margin":
        pool = passing
        key = lambda a: (max(0.0, (a.get("tracking_m") or 0) - 0.005),
                         a.get("duration_s", float("inf")))
    elif rule == "envelope_margin":
        pool = passing
        key = lambda a: (max(0.0, (a.get("tube_m") or 0) - 0.02),
                         a.get("duration_s", float("inf")))
    elif rule == "fewest_gates":
        pool = passing
        key = lambda a: (len(a.get("reasons") or []), a.get("duration_s", float("inf")))
    else:
        raise ValueError(rule)
    if not pool:
        return None
    return min(pool, key=key)


def report_selection(rows: Sequence[Row]) -> None:
    """Q10: is 'smallest envelope' the right selection rule?"""
    print("\n## Selection rules (Q10)\n")
    rules = ("current", "fastest", "tracking_margin", "envelope_margin", "fewest_gates")
    by_mission = group_by_mission(rows)
    print("| rule | missions with a passing pick | mean duration s | mean envelope cm | "
          "mean tracking cm |")
    print("|---|---:|---:|---:|---:|")
    for rule in rules:
        picks = []
        for mission, group in by_mission.items():
            pick = _select(group, rule)
            if pick is not None:
                picks.append(pick)
        if not picks:
            continue
        print(f"| {rule} | {len(picks)}/{len(by_mission)} | "
              f"{_fmt(_mean([_finite(p.get('duration_s')) for p in picks]))} | "
              f"{_fmt((_mean([_finite(p.get('tube_m')) for p in picks]) or 0) * 100, 2)} | "
              f"{_fmt((_mean([_finite(p.get('tracking_m')) for p in picks]) or 0) * 100, 2)} |")

    # How much does the current rule cost in time?
    current = []
    fastest = []
    for mission, group in by_mission.items():
        a = _select(group, "current")
        b = _select(group, "fastest")
        if a is not None and b is not None:
            current.append(a)
            fastest.append(b)
    if current:
        gaps = [(b.get("duration_s", 0) - a.get("duration_s", 0))
                for a, b in zip(current, fastest)]
        print(f"\nThe current rule (smallest envelope) gives up "
              f"{_fmt(_mean(gaps) * 1000, 1)} ms on average against the fastest "
              f"passing candidate (median {_fmt(_median(gaps) * 1000, 1)} ms, "
              f"max {_fmt(max(gaps) * 1000, 1)} ms).")


def pairs_of(rows: Sequence[Row], match: Any) -> List[Tuple[Row, Row]]:
    """The rows a family predicate selects."""
    return [(r, r) for r in rows if match(r)]


def report_compare(rows: Sequence[Row], label_a: str, label_b: str) -> None:
    """Q1/Q2: compare two guess families on the same missions.

    The two families are matched on (mission, timing model, accuracy weight),
    so the comparison is like for like: the same path, the same timing model
    and the same weight, started from a different initial guess.
    """
    def family(label: str) -> Optional[Any]:
        if label == "spline":
            return lambda r: r.get("guess") == "spline"
        if label == "proportional":
            return lambda r: r.get("guess") == "proportional_dt"
        if label == "baseline":
            return lambda r: (r.get("style") == "baseline"
                              and r.get("guess", "stock") == "stock")
        return lambda r: label in r.get("label", "")

    match_a = family(label_a)
    match_b = family(label_b)

    groups: Dict[Tuple, Dict[str, Row]] = defaultdict(dict)
    for row in rows:
        key = (row["mission"], row.get("timing"), row.get("weight"),
               row.get("samples"))
        if match_a(row):
            groups[key]["a"] = row
        if match_b(row):
            groups[key]["b"] = row

    wins = losses = ties = 0
    a_pass = b_pass = 0
    compared = 0
    deltas: List[float] = []
    detail: List[str] = []
    for key in sorted(groups):
        pair = groups[key]
        if "a" not in pair or "b" not in pair:
            continue
        a, b = pair["a"], pair["b"]
        compared += 1
        a_pass += 1 if a.get("ok") else 0
        b_pass += 1 if b.get("ok") else 0
        da = a.get("duration_s") or float("inf")
        db = b.get("duration_s") or float("inf")
        if abs(da - db) < 1e-6:
            ties += 1
        elif da < db:
            wins += 1
        else:
            losses += 1
        if da and db and db > 0:
            deltas.append((da - db) / db * 100)
        detail.append(f"{key[0]} / {key[1]} / a={key[2]}: "
                      f"{label_a} {da:.3f}s ok={a.get('ok')} | "
                      f"{label_b} {db:.3f}s ok={b.get('ok')}")

    print(f"\n## {label_a} vs {label_b}\n")
    print(f"matched (mission, timing, weight) pairs: {compared}")
    print(f"{label_a} faster on {wins}, slower on {losses}, tied on {ties}")
    print(f"{label_a} passes on {a_pass}, {label_b} passes on {b_pass}")
    iters_a = [r.get("iter_count") for _, r in pairs_of(rows, match_a)]
    iters_b = [r.get("iter_count") for _, r in pairs_of(rows, match_b)]
    if iters_a and iters_b:
        print(f"IPOPT iterations: {label_a} median {_fmt(_median(iters_a), 0)}, "
              f"{label_b} median {_fmt(_median(iters_b), 0)}")
        print(f"solve wall time:  {label_a} median {_fmt(_median([_finite(r.get('solve_s')) for r in pairs_of(rows, match_a)[1]]), 2)} s, "
              f"{label_b} median {_fmt(_median([_finite(r.get('solve_s')) for r in pairs_of(rows, match_b)[1]]), 2)} s")
    if deltas:
        print(f"duration delta: median {_fmt(_median(deltas), 1)} %, "
              f"mean {_fmt(_mean(deltas), 1)} %, stdev {_fmt(_stdev(deltas), 1)} %")
    for line in detail:
        print(f"  {line}")


def report_critic(rows: Sequence[Row]) -> None:
    """Q5: does the critic flag windows that the gates accept?"""
    print("\n## Multi-Verse critic vs the gates (Q5)\n")
    flagged_passing = [r for r in rows if r.get("ok") and (r.get("critic_windows") or 0) > 0]
    clean_passing = [r for r in rows if r.get("ok") and (r.get("critic_windows") or 0) == 0]
    flagged_failing = [r for r in rows if not r.get("ok") and (r.get("critic_windows") or 0) > 0]
    clean_failing = [r for r in rows if not r.get("ok") and (r.get("critic_windows") or 0) == 0]
    print(f"passing candidates flagged by the critic: {len(flagged_passing)}/"
          f"{len(flagged_passing) + len(clean_passing)}")
    print(f"failing candidates flagged by the critic: {len(flagged_failing)}/"
          f"{len(flagged_failing) + len(clean_failing)}")
    precision = (len(flagged_failing) /
                 max(1, len(flagged_failing) + len(flagged_passing)))
    recall = len(flagged_failing) / max(1, len(flagged_failing) + len(clean_failing))
    print(f"precision (of the flagged, how many fail): {100 * precision:.0f}%")
    print(f"recall    (of the failures, how many are flagged): {100 * recall:.0f}%")
    print("\nA critic that flags passing trajectories would make a feedback loop "
          "harmful: it would send good windows back into the solver.")


def report_envelope_proxy(rows: Sequence[Row]) -> None:
    """Q4: is there a cheap proxy for the reachability envelope?"""
    print("\n## Envelope growth vs cheap predictors (Q4)\n")
    passing = [r for r in rows if r.get("ok")]
    xs = [_finite(r.get("duration_s")) for r in passing]
    ys = [_finite(r.get("tube_m")) for r in passing]
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) >= 3:
        px = [p[0] for p in pairs]
        py = [p[1] for p in pairs]
        print(f"envelope vs duration: Pearson {_fmt(pearson(px, py), 3)}, "
              f"Spearman {_fmt(spearman(px, py), 3)} (n={len(pairs)})")
    xs = [_finite(r.get("path_length_m")) for r in passing]
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) >= 3:
        px = [p[0] for p in pairs]
        py = [p[1] for p in pairs]
        print(f"envelope vs path length: Pearson {_fmt(pearson(px, py), 3)}, "
              f"Spearman {_fmt(spearman(px, py), 3)} (n={len(pairs)})")
    xs = [_finite(r.get("samples")) for r in passing]
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) >= 3:
        px = [p[0] for p in pairs]
        py = [p[1] for p in pairs]
        print(f"envelope vs sample count: Pearson {_fmt(pearson(px, py), 3)}, "
              f"Spearman {_fmt(spearman(px, py), 3)} (n={len(pairs)})")


def report_density(rows: Sequence[Row]) -> None:
    """Q7: sample density against timing flexibility."""
    print("\n## Sample density (Q7)\n")
    buckets: Dict[Tuple[int, str], List[Row]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("samples", 10), row.get("timing", "?"))].append(row)
    print("| samples | timing | n | pass | median duration s | median envelope cm | "
          "median tracking cm | median solve s |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(buckets):
        group = buckets[key]
        passing = sum(1 for r in group if r.get("ok"))
        print(f"| {key[0]} | {key[1]} | {len(group)} | {passing} | "
              f"{_fmt(_median([_finite(r.get('duration_s')) for r in group]))} | "
              f"{_fmt((_median([_finite(r.get('tube_m')) for r in group]) or 0) * 100, 2)} | "
              f"{_fmt((_median([_finite(r.get('tracking_m')) for r in group]) or 0) * 100, 2)} | "
              f"{_fmt(_median([_finite(r.get('solve_s')) for r in group]), 2)} |")


def report_spread(rows: Sequence[Row]) -> None:
    """KI-020: how much of a PASS depends on the declared uncertainty?"""
    print("\n## Uncertainty sensitivity (spread of the declared ranges)\n")
    print("| spread | n | pass rate | median envelope cm | median tracking cm |")
    print("|---:|---:|---:|---:|---:|")
    buckets: Dict[float, List[Row]] = defaultdict(list)
    for row in rows:
        buckets[row.get("spread", 1.0)].append(row)
    for spread in sorted(buckets):
        group = buckets[spread]
        passing = sum(1 for r in group if r.get("ok"))
        print(f"| {spread:g} | {len(group)} | {100 * passing / len(group):.0f}% | "
              f"{_fmt((_median([_finite(r.get('tube_m')) for r in group]) or 0) * 100, 2)} | "
              f"{_fmt((_median([_finite(r.get('tracking_m')) for r in group]) or 0) * 100, 2)} |")
    print("\nspread 1.0 is the range as declared in the robot config; 0 collapses "
          "every range to its nominal value; 2 doubles its width.")


def report_timing(rows: Sequence[Row]) -> None:
    """The two timing models, mission by mission."""
    print("\n## Timing model (baseline seed, base weight)\n")
    rows = [r for r in rows
            if r.get("style") == "baseline" and r.get("guess", "stock") == "stock"
            and r.get("weight", 0) == 0 and r.get("samples", 10) == 10]
    by_mission = group_by_mission(rows)
    faster = slower = same = 0
    per_pass = {"equal-duration": 0, "per-segment": 0}
    total = 0
    for mission, group in by_mission.items():
        uniform = [r for r in group if r.get("timing") == "equal-duration"]
        perseg = [r for r in group if r.get("timing") == "per-segment"]
        if not uniform or not perseg:
            continue
        total += 1
        u = min(uniform, key=lambda r: r.get("duration_s", float("inf")))
        p = min(perseg, key=lambda r: r.get("duration_s", float("inf")))
        per_pass["equal-duration"] += 1 if u.get("ok") else 0
        per_pass["per-segment"] += 1 if p.get("ok") else 0
        du = u.get("duration_s") or float("inf")
        dp = p.get("duration_s") or float("inf")
        if abs(du - dp) < 1e-6:
            same += 1
        elif dp < du:
            faster += 1
        else:
            slower += 1
        print(f"| {mission} | {du:.3f} s {'ok' if u.get('ok') else 'FAIL'} | "
              f"{dp:.3f} s {'ok' if p.get('ok') else 'FAIL'} | "
              f"{100 * (du - dp) / du:+.1f} % |")
    print(f"\nper-segment is faster on {faster}, slower on {slower}, tied on {same} "
          f"of {total} missions")
    print(f"passing: equal-duration {per_pass['equal-duration']}/{total}, "
          f"per-segment {per_pass['per-segment']}/{total}")


def report_dtheta(rows: Sequence[Row]) -> None:
    """RQ-A at scale: does the per-sample heading change explain the tracking error?

    RQ-A (docs/research/RQ-A_tracking_error.md) says the tracking gate is a
    discretisation artefact: the plan, the controller's reference and the
    plant's integration are three different discretisations, and they disagree
    by about the heading change per sample. This measures that relation over
    the whole suite instead of five missions.
    """
    print("\n## Tracking error vs the heading change per sample (RQ-A)\n")
    pairs = [(r.get("mean_dtheta_rad"), r.get("tracking_m"), r)
             for r in rows
             if r.get("mean_dtheta_rad") is not None
             and r.get("tracking_m") is not None]
    if len(pairs) < 3:
        print("no data")
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    print(f"candidates: {len(pairs)}")
    print(f"Spearman(mean dtheta, tracking) = {_fmt(spearman(xs, ys), 3)}")
    print(f"Pearson (mean dtheta, tracking) = {_fmt(pearson(xs, ys), 3)}")

    # Per mission, the best candidate of each mission against the mechanism.
    by_mission = group_by_mission([p[2] for p in pairs])
    print("\n| mission | mean dtheta (rad) | tracking (cm) | gate |")
    print("|---|---:|---:|---|")
    for mission, group in sorted(by_mission.items()):
        best = min(group, key=lambda r: r.get("tracking_m", float("inf")))
        print(f"| {mission} | {best.get('mean_dtheta_rad', 0):.4f} | "
              f"{(best.get('tracking_m') or 0) * 100:.2f} | "
              f"{'PASS' if best.get('ok') else 'FAIL'} |")

    # The gate as a function of the mechanism variable.
    print("\n| mean dtheta bucket | n | median tracking cm | pass rate |")
    print("|---|---:|---:|---:|")
    buckets = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 10.0)]
    for low, high in buckets:
        group = [r for r in pairs if low <= r[0] < high]
        if not group:
            continue
        passing = sum(1 for r in group if r[2].get("ok"))
        print(f"| {low:.2f}-{high:.2f} | {len(group)} | "
              f"{_fmt((_median([r[1] for r in group]) or 0) * 100, 2)} | "
              f"{100 * passing / len(group):.0f}% |")


def report_marginal(rows: Sequence[Row]) -> None:
    """Q3: what does each extra candidate buy?

    The candidates are added in the order the CLI adds them: the default two
    (baseline on both timing models), then the remaining seeds, then the
    accuracy-weight grid. The table says how many missions have a passing
    candidate after each addition, and how long the fastest passing candidate
    is.
    """
    print("\n## The marginal value of a candidate (Q3)\n")
    slots: List[Tuple[str, Any]] = [
        ("baseline / equal-duration", lambda r: (r.get("style") == "baseline"
            and r.get("guess", "stock") == "stock"
            and r.get("timing") == "equal-duration" and r.get("weight") == 0)),
        ("baseline / per-segment", lambda r: (r.get("style") == "baseline"
            and r.get("guess", "stock") == "stock"
            and r.get("timing") == "per-segment" and r.get("weight") == 0)),
        ("rolling / equal-duration", lambda r: (r.get("style") == "rolling"
            and r.get("timing") == "equal-duration" and r.get("weight") == 0)),
        ("rolling / per-segment", lambda r: (r.get("style") == "rolling"
            and r.get("timing") == "per-segment" and r.get("weight") == 0)),
        ("bow_left / equal-duration", lambda r: (r.get("style") == "bow_left"
            and r.get("timing") == "equal-duration" and r.get("weight") == 0)),
        ("bow_left / per-segment", lambda r: (r.get("style") == "bow_left"
            and r.get("timing") == "per-segment" and r.get("weight") == 0)),
        ("bow_right / equal-duration", lambda r: (r.get("style") == "bow_right"
            and r.get("timing") == "equal-duration" and r.get("weight") == 0)),
        ("bow_right / per-segment", lambda r: (r.get("style") == "bow_right"
            and r.get("timing") == "per-segment" and r.get("weight") == 0)),
        ("turn_first / equal-duration", lambda r: (r.get("style") == "turn_first"
            and r.get("timing") == "equal-duration" and r.get("weight") == 0)),
        ("turn_first / per-segment", lambda r: (r.get("style") == "turn_first"
            and r.get("timing") == "per-segment" and r.get("weight") == 0)),
    ]
    for weight in (0.001, 0.002, 0.005, 0.01):
        for timing in ("equal-duration", "per-segment"):
            slots.append((f"baseline / {timing} / a={weight:g}",
                          (lambda w, t: (lambda r: (r.get("style") == "baseline"
                              and r.get("guess", "stock") == "stock"
                              and r.get("timing") == t and r.get("weight") == w)))
                          (weight, timing)))

    by_mission = group_by_mission(rows)
    print("| candidates | missions with a passing candidate | mean best duration s | "
          "solves |")
    print("|---|---:|---:|---:|")
    for count in range(1, len(slots) + 1):
        chosen = slots[:count]
        solved = 0
        durations = []
        for mission, group in by_mission.items():
            pool = [r for r in group
                    if any(match(r) for _, match in chosen)]
            if not pool:
                continue
            solved += 1
            passing = [r for r in pool if r.get("ok")]
            if passing:
                durations.append(min(r.get("duration_s", float("inf"))
                                     for r in passing))
        label = chosen[-1][0]
        print(f"| {count} (last: {label}) | {len(durations)}/{len(by_mission)} | "
              f"{_fmt(_mean(durations))} | {count} |")


DRIVEBASE_ACCEL_LIMIT = 7.8  # m/s^2, KI-021 (80 % of the firmware ceiling)


def report_drivebase(rows: Sequence[Row]) -> None:
    """Q11: does the plan ask for more acceleration than the hub can deliver?

    The hub's DriveBase ramps its trajectories at about 7.8 m/s^2 for this
    robot's geometry (KI-021). The optimizer does not model that limit, so
    the plan can demand more than the hardware will give.
    """
    print(f"\n## Peak wheel acceleration vs the DriveBase limit "
          f"({DRIVEBASE_ACCEL_LIMIT} m/s^2, KI-021)\n")
    values = [_finite(r.get("max_accel_m_s2")) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        print("no data")
        return
    over = [v for v in values if v > DRIVEBASE_ACCEL_LIMIT]
    print(f"candidates: {len(values)}")
    print(f"peak acceleration: median {_fmt(_median(values), 2)} m/s^2, "
          f"max {_fmt(max(values), 2)} m/s^2")
    print(f"above the DriveBase limit: {len(over)}/{len(values)} "
          f"({100 * len(over) / len(values):.0f}%)")
    passing = [_finite(r.get("max_accel_m_s2")) for r in rows if r.get("ok")]
    passing = [v for v in passing if v is not None]
    if passing:
        over_passing = [v for v in passing if v > DRIVEBASE_ACCEL_LIMIT]
        print(f"among the passing candidates: {len(over_passing)}/{len(passing)} "
              f"({100 * len(over_passing) / len(passing):.0f}%) ask for more, "
              f"median {_fmt(_median(passing), 2)} m/s^2, "
              f"max {_fmt(max(passing), 2)} m/s^2")


def report_errors(rows: Sequence[Row]) -> None:
    """What broke, and how often."""
    print("\n## Errors and crashes\n")
    counter: Dict[str, int] = defaultdict(int)
    first_seen: Dict[str, Row] = {}
    for row in rows:
        error = row.get("error")
        if not error:
            continue
        key = str(error).strip().splitlines()[0][:160]
        counter[key] += 1
        first_seen.setdefault(key, row)
    if not counter:
        print(f"no solver error in {len(rows)} candidates")
    else:
        print("| count | mission | message |")
        print("|---:|---|---|")
        for key, count in sorted(counter.items(), key=lambda kv: -kv[1]):
            print(f"| {count} | {first_seen[key].get('mission')} | {key} |")
    empty = sum(1 for row in rows if "no samples" in (row.get("reasons") or []))
    print(f"\ncandidates that produced no samples at all: {empty}/{len(rows)}")


def report_robots(rows: Sequence[Row]) -> None:
    """Does the robot's own size decide whether a path is planned?"""
    print("\n## Pass rate by robot constant\n")
    names = set()
    for row in rows:
        names.update((row.get("robot_factors") or {}).keys())
    if not names:
        print("this run used one robot for every mission")
        return
    for name in sorted(names):
        buckets: Dict[float, List[Row]] = defaultdict(list)
        for row in rows:
            factor = (row.get("robot_factors") or {}).get(name)
            if factor is None:
                continue
            buckets[round(float(factor), 1)].append(row)
        print(f"\n### {name}\n")
        print("| factor | n | pass | pass rate | median envelope cm | "
              "median duration s |")
        print("|---:|---:|---:|---:|---:|---:|")
        for factor in sorted(buckets):
            group = buckets[factor]
            passing = sum(1 for r in group if r.get("ok"))
            tubes = [_finite(r.get("tube_m")) for r in group]
            print(f"| {factor:.1f} | {len(group)} | {passing} | "
                  f"{100 * passing / len(group):.0f}% | "
                  f"{_fmt((_median(tubes) or 0) * 100, 2)} | "
                  f"{_fmt(_median([_finite(r.get('duration_s')) for r in group]))} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=("summary", "seeds", "gates", "weights", "select",
                                 "compare", "critic", "density", "envelope",
                                 "spread", "timing", "dtheta", "marginal",
                                 "drivebase", "errors", "robots", "all"))
    parser.add_argument("path")
    parser.add_argument("--a", default="spline", help="First label for compare.")
    parser.add_argument("--b", default="baseline", help="Second label for compare.")
    parser.add_argument("--out", default=None,
                        help="Write the report to this file instead of stdout.")
    args = parser.parse_args()

    meta, rows = load(args.path)
    if not rows:
        raise SystemExit("no records")

    import io
    import contextlib
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if args.command in ("summary", "all"):
            report_summary(meta, rows)
        if args.command in ("seeds", "all"):
            report_seeds(rows)
        if args.command in ("gates", "all"):
            report_gates(rows)
        if args.command in ("weights", "all"):
            report_weights(rows)
        if args.command in ("select", "all"):
            report_selection(rows)
        if args.command in ("critic", "all"):
            report_critic(rows)
        if args.command in ("envelope", "all"):
            report_envelope_proxy(rows)
        if args.command in ("spread", "all"):
            report_spread(rows)
        if args.command in ("timing", "all"):
            report_timing(rows)
        if args.command in ("dtheta", "all"):
            report_dtheta(rows)
        if args.command in ("marginal", "all"):
            report_marginal(rows)
        if args.command in ("drivebase", "all"):
            report_drivebase(rows)
        if args.command in ("errors", "all"):
            report_errors(rows)
        if args.command in ("robots", "all"):
            report_robots(rows)
        if args.command == "density":
            report_density(rows)
        if args.command == "compare":
            report_compare(rows, args.a, args.b)

    text = buffer.getvalue()
    if args.out:
        header = [
            f"# Measurements: {Path(args.path).name}",
            "",
            f"Source: `{args.path}`",
            "",
        ]
        if meta:
            header += [
                f"- config: `{meta.get('config')}`",
                f"- spec set: `{meta.get('specs')}`, samples: {meta.get('samples')}, "
                f"base weight: {meta.get('base_weight')}",
                f"- started: {meta.get('started')}",
                "- tree hashes:",
            ]
            for name, digest in (meta.get("provenance") or {}).items():
                header.append(f"  - `{name}`: `{digest}`")
        header += [
            "",
            "Every number below is a model prediction. The declared uncertainty "
            "ranges are test assumptions, not measurements, and no hardware logs "
            "exist in the repository.",
            "",
        ]
        Path(args.out).write_text("\n".join(header) + text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
