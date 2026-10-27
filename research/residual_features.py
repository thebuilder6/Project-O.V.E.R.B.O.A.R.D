"""Phase 1 (F2): can plan features predict the tracking residual within a mission?

E2 (research/separability.py) showed the gate quantity - the closed-loop tracking
residual, the thing the 5 mm gate (Q18) measures - is separable from the clock
(within-mission r = 0.107 pooled) and has real spread where plans differ
(2.8-4.6 mm on the aggressiveness corpora). A residual term in the objective is
therefore not automatically a re-priced clock. But before building one, the
cheapest question is whether the residual is predictable from numbers already on
the plan at all - a static surrogate term - or whether only a closed-loop rollout
can price it.

Pre-registered reading (delivered with the plan, 2026-01):

    FE R2 >= 0.75 on the residual, with feature signs stable across >= 80% of
    mission groups  ->  a static surrogate term is viable for Phase 2
    otherwise        ->  the objective term must be a closed-loop rollout
                         (differentiable-tube route)

Amendment, honest and measured here: the pre-registration targeted q_control,
but q_control (the q18 decomposition) exists only on q18_decompose rows, which
carry no shape features. So 1a applies the gate to the total tracking residual
(the gate quantity itself) on the feature-rich corpora, and reports q_control
predictability on q18 for whatever q18 has (the clock). The q_control-vs-shape
gate is 1b's job.

    python research/residual_features.py
    python research/residual_features.py --out research/results/residual_features.json

Everything here is a re-analysis of existing measurement files; no solve runs.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from separability import (  # noqa: E402
    RESULTS, OLD_CORPORA, NEW_CORPORA, Q18_CORPORA,
    demean, within_r2, pearson,
)

# Feature blocks per corpus family. Outcomes (spread, tracking, tube/envelope,
# violations) and treatment knobs (weight/accuracy_weight, style, mesh) are
# deliberately NOT features - they are what the regression explains or what the
# experimenter turned.
OLD_FEATURES = ["max_accel_m_s2", "max_wheel_speed_m_s", "path_length_m",
                "max_dtheta_rad", "mean_dtheta_rad"]
NEW_FEATURES = ["rms_force_util", "peak_force_util", "peak_traction_util",
                "slip_risk"]


def load_rows(name, family):
    rows = []
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if family == "old":
            if rec.get("record") == "meta" or rec.get("duration_s") is None:
                continue
            if not rec.get("converged"):
                continue
            rows.append({
                "corpus": name.replace(".jsonl", ""),
                "mission": rec.get("mission", "?"),
                "duration_s": rec.get("duration_s"),
                "tracking_m": rec.get("tracking_m"),
                "jerk": rec.get("max_jerk_m_s3"),
                **{k: rec.get(k) for k in OLD_FEATURES},
            })
        elif family == "new":
            for cand in (rec.get("report") or {}).get("candidates") or []:
                if not cand.get("converged"):
                    continue
                rows.append({
                    "corpus": name.replace(".jsonl", ""),
                    "mission": rec.get("mission", "?"),
                    "duration_s": cand.get("duration_s"),
                    "tracking_m": (cand.get("tracking_cm") / 100.0)
                    if cand.get("tracking_cm") is not None else None,
                    "jerk": cand.get("peak_jerk_m_s3"),
                    **{k: cand.get(k) for k in NEW_FEATURES},
                })
        else:  # q18
            for cand in rec.get("results") or []:
                if not cand.get("converged"):
                    continue
                rows.append({
                    "corpus": name.replace(".jsonl", ""),
                    "mission": rec.get("mission", "?"),
                    "duration_s": cand.get("duration_s"),
                    "tracking_m": cand.get("q_reported_m"),
                    "q_control_m": cand.get("q_control_m"),
                    "jerk": None,
                })
    return rows


def slope_sign(rows, y_key, x_key):
    """Fraction of (corpus, mission) groups whose within-group x->y slope is > 0."""
    groups = {}
    for r in rows:
        groups.setdefault((r["corpus"], r["mission"]), []).append(r)
    pos = tot = 0
    for g in groups.values():
        if len(g) < 3:
            continue
        xs = [r[x_key] for r in g]
        ys = [r[y_key] for r in g]
        if max(xs) - min(xs) <= 0:
            continue
        tot += 1
        if pearson(xs, ys) > 0:
            pos += 1
    return pos, tot


def complete(rows, keys):
    return [r for r in rows
            if all(r.get(k) is not None and math.isfinite(r[k]) for k in keys)]


def block_report(rows, y_key, feat_keys, label):
    rows = complete(rows, [y_key] + feat_keys)
    if len(rows) < 5:
        return None
    return {
        "block": label,
        "features": feat_keys,
        "n": len(rows),
        "fe_r2": within_r2(rows, y_key, feat_keys),
        "groups": len({(r["corpus"], r["mission"]) for r in rows}),
    }


def greedy_block(rows, y_key, feat_keys, base_keys, label):
    """Best non-singular subset: add features one at a time while FE R2 improves.

    The feature sets carry near-identical pairs (peak_force_util vs
    peak_traction_util), which makes X'X singular and the block NaN.
    """
    chosen = list(base_keys)
    best = block_report(rows, y_key, chosen, label)
    for f in feat_keys:
        if f in chosen:
            continue
        trial = block_report(rows, y_key, chosen + [f], label)
        if trial and math.isfinite(trial["fe_r2"]) and (
                best is None or trial["fe_r2"] > best["fe_r2"] + 1e-6):
            chosen, best = chosen + [f], trial
    return best


def spread_line(rows, y_key):
    """Median within-(corpus,mission) std of y, in mm - the scale check."""
    groups = {}
    for r in rows:
        if r.get(y_key) is not None:
            groups.setdefault((r["corpus"], r["mission"]), []).append(r[y_key])
    stds = []
    for g in groups.values():
        if len(g) < 3:
            continue
        m = sum(g) / len(g)
        stds.append(math.sqrt(sum((v - m) ** 2 for v in g) / len(g)) * 1000.0)
    if not stds:
        return None
    return sorted(stds)[len(stds) // 2]


def feature_report(rows, y_key, x_key):
    rows = complete(rows, [y_key, x_key])
    if len(rows) < 5:
        return None
    y_dm, _ = demean(rows, y_key)
    x_dm, _ = demean(rows, x_key)
    pos, tot = slope_sign(rows, y_key, x_key)
    return {
        "feature": x_key,
        "n": len(rows),
        "within_r": pearson(x_dm, y_dm),
        "fe_r2": within_r2(rows, y_key, [x_key]),
        "groups_pos_slope": pos,
        "groups": tot,
        "sign_stability": (max(pos, tot - pos) / tot) if tot else None,
    }


def analyse(rows, y_key, features, corpus):
    spread = spread_line(rows, y_key)
    spread_s = f"{spread:.3f}mm" if spread is not None else "-"
    print(f"\n== {corpus}: {y_key}   (median within-mission std {spread_s}) ==")
    base = block_report(rows, y_key, ["duration_s"], "duration")
    base_j = block_report(rows, y_key, ["duration_s", "jerk"], "duration+jerk")
    shape = greedy_block(rows, y_key, features, [], "shape*")
    shape_dj = greedy_block(rows, y_key, features, ["duration_s", "jerk"],
                            "duration+jerk+shape*")
    shape_d = greedy_block(rows, y_key, features, ["duration_s"],
                           "duration+shape*")
    out_blocks = []
    print(f"{'block':<24}{'n':>6}{'FE R2':>8}{'groups':>8}")
    for b in (base, base_j, shape_d, shape_dj, shape):
        if b is None:
            continue
        print(f"{b['block']:<24}{b['n']:>6}{b['fe_r2']:>8.3f}{b['groups']:>8}")
        out_blocks.append(b)
    print(f"{'feature (univariate)':<24}{'n':>6}{'within r':>10}{'FE R2':>8}"
          f"{'sign stab':>10}")
    out_feats = []
    for f in features:
        fr = feature_report(rows, y_key, f)
        if fr is None:
            continue
        if fr["sign_stability"] is not None:
            stab = "%.2f (%d/%d)" % (fr["sign_stability"],
                                     fr["groups_pos_slope"], fr["groups"])
        else:
            stab = "-"
        print(f"{fr['feature']:<24}{fr['n']:>6}{fr['within_r']:>10.3f}"
              f"{fr['fe_r2']:>8.3f}{stab:>10}")
        out_feats.append(fr)
    return {"corpus": corpus, "target": y_key, "spread_mm": spread,
            "blocks": out_blocks, "features": out_feats}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "residual_features.json"))
    args = ap.parse_args(argv)

    old_rows, new_rows, q18_rows = [], [], []
    for name in OLD_CORPORA:
        if (RESULTS / name).exists():
            old_rows += load_rows(name, "old")
    for name in NEW_CORPORA:
        if (RESULTS / name).exists():
            new_rows += load_rows(name, "new")
    for name in Q18_CORPORA:
        if (RESULTS / name).exists():
            q18_rows += load_rows(name, "q18")

    # Self-checks: this script must reproduce separability.py's numbers and the
    # identity fe_r2(y ~ x) == within_r(x, y)^2 on the shared pair.
    w = [r for r in old_rows if r["corpus"] == "weights"]
    w = complete(w, ["tracking_m", "duration_s", "jerk"])
    r2_d = within_r2(w, "tracking_m", ["duration_s"])
    r2_dj = within_r2(w, "tracking_m", ["duration_s", "jerk"])
    y_dm, _ = demean(w, "tracking_m")
    x_dm, _ = demean(w, "duration_s")
    wr = pearson(x_dm, y_dm)
    assert abs(r2_d - 0.405) < 0.01, f"weights FE R2(track~dur) {r2_d:.3f} != 0.405"
    assert abs(r2_dj - 0.506) < 0.01, f"weights FE R2(track~dur+jerk) {r2_dj:.3f} != 0.506"
    assert abs(r2_d - wr * wr) < 1e-9, "fe_r2 != within_r^2 on the univariate pair"
    print(f"self-checks ok (weights FE R2 dur {r2_d:.3f}, dur+jerk {r2_dj:.3f}, "
          f"within_r^2 {wr * wr:.3f})")

    out = {"rows": {"old": len(old_rows), "new": len(new_rows), "q18": len(q18_rows)},
           "analyses": []}

    # Per corpus family and per corpus, on the tracking residual.
    out["analyses"].append(analyse(old_rows, "tracking_m", OLD_FEATURES, "old pooled"))
    out["analyses"].append(analyse(new_rows, "tracking_m", NEW_FEATURES, "new pooled"))
    for corpus in sorted({r["corpus"] for r in old_rows}):
        out["analyses"].append(analyse([r for r in old_rows if r["corpus"] == corpus],
                                       "tracking_m", OLD_FEATURES, corpus))
    for corpus in sorted({r["corpus"] for r in new_rows}):
        out["analyses"].append(analyse([r for r in new_rows if r["corpus"] == corpus],
                                       "tracking_m", NEW_FEATURES, corpus))
    # q18: the control share, with the clock as the only numeric feature.
    out["analyses"].append(analyse(q18_rows, "q_control_m", ["duration_s"],
                                   "q18_decompose"))

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
