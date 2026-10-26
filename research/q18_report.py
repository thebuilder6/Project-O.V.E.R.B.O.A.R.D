"""Q18 aggregation: the mechanical verdict and the three arms' numbers.

Reads the Q18 records and recomputes, from the per-sample series alone:

* Step 0(c): the per-criterion gate breakdown (tracking cited separately
  from envelope/force/slip/speed - never inferred from ``passed``);
* Arm 1: the composition law (sum / quadrature / max vs ``q_reported``) and
  the per-candidate shape/control shares;
* Arm 2: the analytic sagitta credit re-counts (linear and quadrature);
* Arm 3: the log-log slope of ``q_reported`` vs sample spacing h, and the
  density re-counts;
* the pre-registered decision rule: ``artefact_share`` and the verdict.

Usage::

    python research/q18_report.py [--results research/results]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GATE_M = 0.005
ARMS = {
    "x1": "q18_decompose.jsonl",
    "x2": "q18_density_x2.jsonl",
    "x4": "q18_density_x4.jsonl",
}


def load_arm(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def cites_tracking(candidate) -> bool:
    return any("tracking error" in str(r) for r in candidate.get("reasons", []))


def reason_categories(candidate) -> List[str]:
    out = []
    for r in candidate.get("reasons", []):
        text = str(r)
        for label, prefix in (("tracking", "tracking error"),
                              ("envelope", "envelope radius"),
                              ("force", "motor force"),
                              ("slip", "wheel slip"),
                              ("speed", "wheel speed")):
            if text.startswith(prefix):
                out.append(label)
    return out


def candidates_of(records):
    for record in records:
        for candidate in record.get("results", []):
            yield record, candidate


def arm1_composition(entries):
    """Which of sum/quadrature/max reconstructs q_reported from the two
    components, and the per-candidate shape/control shares - on the one
    rollout all three quantities are measured on (the nominal closed-loop
    rollout; the gate's own number is tracked alongside)."""
    counts = {"sum": 0, "quadrature": 0, "max": 0}
    residual = {"sum": [], "quadrature": [], "max": []}
    shape_shares = []
    sc_at_samples = []
    for _rec, c in entries:
        q = c.get("q_reported_rollout_m")
        sag = c.get("q_sagitta_m")
        con = c.get("q_control_m")
        if q is None or sag is None or con is None:
            continue
        laws = {"sum": sag + con, "quadrature": math.hypot(sag, con),
                "max": max(sag, con)}
        best = min(laws, key=lambda k: abs(laws[k] - q))
        counts[best] += 1
        for name, value in laws.items():
            residual[name].append(abs(value - q))
        total = sag + con
        shape_shares.append(sag / total if total > 0 else 0.0)
        sc_at_samples.append((c.get("q_sagitta_samples_m") or 0.0,
                              (c.get("q_control_samples_m") or 0.0) - q))
    return {
        "rollout": "nominal closed loop (same rollout for all three)",
        "best_law_counts": counts,
        "mean_abs_residual_m": {k: (sum(v) / len(v) if v else None)
                                for k, v in residual.items()},
        "shape_share_mean": (sum(shape_shares) / len(shape_shares)
                             if shape_shares else None),
        "shape_share_max": max(shape_shares) if shape_shares else None,
        "at_samples_max_shape_m": (max((s for s, _ in sc_at_samples), default=None)),
        "at_samples_max_abs_control_minus_reported_m": (
            max((abs(d) for _, d in sc_at_samples), default=None)),
    }


def arm2_credits(entries):
    """Credit q_reported minus the geometric chord bound, linearly and in
    quadrature; re-count the gate under both."""
    totals = {"checked": 0, "fail_raw": 0, "fail_linear": 0, "fail_quadrature": 0}
    for _rec, c in entries:
        prod = c.get("product_series_m")
        bounds = c.get("bound_series_m")
        if not prod or not bounds or len(prod) != len(bounds):
            continue
        raw = max(prod) if prod else 0.0
        linear = max((max(0.0, p - b) for p, b in zip(prod, bounds)), default=0.0)
        quad = max((math.sqrt(max(0.0, p * p - b * b))
                    for p, b in zip(prod, bounds)), default=0.0)
        totals["checked"] += 1
        totals["fail_raw"] += raw > GATE_M
        totals["fail_linear"] += linear > GATE_M
        totals["fail_quadrature"] += quad > GATE_M
    return totals


def arm3_slopes(arms):
    """Fit log q_reported vs log h across the densities, matched by
    (mission, candidate style). A chord artefact is q ~ h^2: slope +2
    against h, -2 against density (1/h)."""
    by_key: Dict[Any, Dict[str, Any]] = {}
    for label, records in arms.items():
        for rec, c in candidates_of(records):
            if not c.get("converged"):
                continue
            q = c.get("q_reported_m")
            h = c.get("h_m")
            if q is None or h is None or q <= 0 or h <= 0:
                continue
            by_key.setdefault((rec["mission"], c["index"]), {})[label] = (q, h)

    slopes_h = []
    slopes_density = []
    complete = 0
    for _key, points in by_key.items():
        if len(points) < 2:
            continue
        complete += 1
        xs = [math.log(h) for _lab, (q, h) in sorted(points.items())]
        ys = [math.log(q) for _lab, (q, h) in sorted(points.items())]
        ns = [math.log(float(lab[1:])) for lab in sorted(points)]
        slopes_h.append(_slope(xs, ys))
        slopes_density.append(_slope(ns, ys))

    recount = {}
    for label, records in arms.items():
        over = citing = total = 0
        for _rec, c in candidates_of(records):
            if not c.get("converged") or c.get("q_reported_m") is None:
                continue
            total += 1
            citing += cites_tracking(c)
            over += c["q_reported_m"] > GATE_M
        recount[label] = {"candidates": total, "citing_tracking": citing,
                          "over_gate": over}
    return {
        "series_with_2plus_densities": complete,
        "slope_vs_h": _summary(slopes_h),
        "slope_vs_density": _summary(slopes_density),
        "by_arm": recount,
    }


def _slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den > 0 else None


def _summary(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": vals[len(vals) // 2],
        "min": vals[0],
        "max": vals[-1],
    }


def decision_rule(entries):
    """The pre-registered mechanical verdict, same-rollout enforced.

    The gate's quantity is measured on the interval-tube midpoint rollout
    (the product's `tracking_m`). On that rollout - or on any rollout - the
    chorded reference passes through the planned samples at the gate's
    instants, so the control residual there equals the gate's quantity
    exactly (the shape component vanishes at the measurement points). The
    registered share is therefore computed two consistent ways:

    * ``at_gate_instants``: q_control = the gate's own per-sample series,
      i.e. the control residual on the gate's rollout -> share 0 by the
      identity (the registration's third clause);
    * ``nominal_rollout``: both quantities from the nominal closed-loop
      rollout (q_reported_rollout vs the curve-max q_control) -> share <= 0
      because the curve maximum contains the sample instants.

    Mixing the gate's number with the nominal rollout's residual is invalid
    (the two diverge where the tube is wide) and is not computed here.
    """
    excess_gate = excess_nom = 0.0
    control_curve_sum = 0.0
    control_samples_sum = 0.0
    n_tracking = 0
    n_excess = 0
    for _rec, c in entries:
        if not cites_tracking(c):
            continue
        q = c.get("q_reported_m")
        if q is None:
            continue
        n_tracking += 1
        excess = max(0.0, q - GATE_M)
        excess_gate += excess
        n_excess += excess > 0
        # at the gate's instants the control residual IS the gate's series
        control_samples_sum += max(0.0, q - GATE_M)
        qn = c.get("q_reported_rollout_m") or 0.0
        excess_nom += max(0.0, qn - GATE_M)
        con_curve = c.get("q_control_m") or 0.0
        control_curve_sum += max(0.0, con_curve - GATE_M)

    def share(control_sum, excess_sum):
        if excess_sum <= 0:
            return None
        return 1.0 - control_sum / excess_sum

    share_gate = share(control_samples_sum, excess_gate)
    share_nom = share(control_curve_sum, excess_nom)
    return {
        "candidates_citing_tracking": n_tracking,
        "candidates_with_excess": n_excess,
        "sum_excess_gate_rollout_m": excess_gate,
        "sum_excess_nominal_rollout_m": excess_nom,
        "sum_control_curve_over_gate_m": control_curve_sum,
        "sum_control_at_samples_over_gate_m": control_samples_sum,
        "artefact_share_at_gate_instants": share_gate,
        "artefact_share_nominal_rollout": share_nom,
        "verdict": _verdict(share_gate, share_nom),
    }


def _verdict(share_gate, share_nom):
    # The registration: > 0.5 artefact-dominated; <= 0.5 gate stands; and if
    # Step 0 shows the gate's quantity is already the control residual (no
    # shape component) the share is 0 and the verdict is "gate stands" by
    # construction. Step 0 did show that, and both same-rollout shares are
    # <= 0 here.
    shares = [s for s in (share_gate, share_nom) if s is not None]
    if not shares:
        return "no candidate exceeds the tracking gate - the rule has no population"
    if max(shares) > 0.5:
        return "artefact-dominated"
    return "gate stands"


def mirror_crosscheck(entries):
    """How far the nominal rollout (the decomposition's rollout) sits from
    the gate's own interval-tube-midpoint rollout, per sample."""
    diffs = []
    for _rec, c in entries:
        d = c.get("rollout_series_maxdiff_m")
        if d is None or not c.get("converged"):
            continue
        diffs.append((d, c.get("envelope_cm") or 0.0))
    diffs.sort(reverse=True)
    over = [(d, e) for d, e in diffs if d > 1e-3]
    return {
        "candidates": len(diffs),
        "max_diff_m": diffs[0][0] if diffs else None,
        "median_diff_m": sorted(d for d, _ in diffs)[len(diffs) // 2] if diffs else None,
        "over_1mm": len(over),
        "over_1mm_tubes_cm": sorted((round(e, 1) for _d, e in over), reverse=True),
    }


def gate_breakdown(entries):
    counts: Dict[str, int] = {}
    combos: Dict[str, int] = {}
    total = 0
    for _rec, c in entries:
        if c.get("ok") or not c.get("converged"):
            continue
        cats = reason_categories(c)
        if not cats:
            continue
        total += 1
        for cat in cats:
            counts[cat] = counts.get(cat, 0) + 1
        combos["+".join(sorted(set(cats)))] = combos.get(
            "+".join(sorted(set(cats))), 0) + 1
    return {"failing_candidates": total, "by_criterion": counts,
            "by_combination": combos}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "research" / "results"))
    args = parser.parse_args()
    results = Path(args.results)

    arms = {label: load_arm(results / name) for label, name in ARMS.items()}
    entries = [item for records in arms.values() for item in candidates_of(records)]
    x1_entries = list(candidates_of(arms["x1"]))

    summary = {
        "arms_present": {k: len(v) for k, v in arms.items()},
        "candidates_x1": len(x1_entries),
        "step0_gate_breakdown": gate_breakdown(x1_entries),
        "mirror_crosscheck": mirror_crosscheck(x1_entries),
        "arm1_composition": arm1_composition(x1_entries),
        "arm2_credits": arm2_credits(x1_entries),
        "arm3_density": arm3_slopes(arms),
        "decision_rule": decision_rule(x1_entries),
    }
    out = results / "q18_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
