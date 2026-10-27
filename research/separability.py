"""E2: is the gate quantity separable from the clock?

The plan-conditioned tube idea (KI-020's missing half) only has something to
buy if the gate quantities - the tracking residual and the reachability tube -
are NOT just functions of duration. Q4 established the envelope is (r = +0.90
with duration, +0.97 on the 204-candidate matrix), which is why an
envelope-shaped objective term re-prices the clock. This script asks the same
question of the tracking residual (the quantity the 5 mm gate measures, Q18),
against duration and against the jerk the objective prices, on every saved
corpus.

The headline number is the **within-mission** correlation (candidates of one
mission demeaned): across missions, hard geometry makes plans both slow and
hard to track, which flatters any correlation. Within a mission the geometry
is fixed and the plans differ in aggressiveness - that is the plan-conditioned
tube question.

Pre-registered reading (see docs/RESEARCH_AGENDA.md Q4/Q6 for the style):

    |within r| >= 0.90   gate quantity is a function of the clock - nothing
                         left for a tube term to buy
    0.50 - 0.90          partial headroom; a term would price part clock,
                         part plan
    |within r| <= 0.50   separable - real headroom for a plan-conditioned term

    python research/separability.py
    python research/separability.py --out research/results/separability.json

Everything here is a re-analysis of existing measurement files; no solve runs.
"""

import argparse
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

OLD_CORPORA = ["matrix.jsonl", "weights.jsonl", "density.jsonl",
               "sc_on.jsonl", "sc_off.jsonl",
               "stress_cli.jsonl", "stress_cli_p3.jsonl"]
NEW_CORPORA = ["q13_plant.jsonl", "q13_tick.jsonl", "q14_plant.jsonl", "q14_tick.jsonl"]
Q18_CORPORA = ["q18_decompose.jsonl"]


def load_old(name):
    rows = []
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("record") == "meta" or rec.get("duration_s") is None:
            continue
        if not rec.get("converged"):
            continue
        rows.append({
            "corpus": name.replace(".jsonl", ""),
            "mission": rec.get("mission", "?"),
            "duration_s": rec["duration_s"],
            "tracking_m": rec.get("tracking_m"),
            "tube_m": rec.get("tube_m"),
            "jerk": rec.get("max_jerk_m_s3"),
        })
    return rows


def load_new(name):
    rows = []
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        report = rec.get("report") or {}
        for cand in report.get("candidates") or []:
            if not cand.get("converged"):
                continue
            rows.append({
                "corpus": name.replace(".jsonl", ""),
                "mission": rec.get("mission", "?"),
                "duration_s": cand.get("duration_s"),
                "tracking_m": (cand.get("tracking_cm") or 0.0) / 100.0
                if cand.get("tracking_cm") is not None else None,
                "tube_m": (cand.get("envelope_cm") or 0.0) / 100.0
                if cand.get("envelope_cm") is not None else None,
                "jerk": cand.get("peak_jerk_m_s3"),
            })
    return rows


def load_q18(name):
    rows = []
    for line in (RESULTS / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for cand in rec.get("results") or []:
            if not cand.get("converged"):
                continue
            rows.append({
                "corpus": name.replace(".jsonl", ""),
                "mission": rec.get("mission", "?"),
                "duration_s": cand.get("duration_s"),
                "tracking_m": cand.get("q_reported_m"),
                "tube_m": None,
                "jerk": None,
                "q_control_share": cand.get("share_control"),
            })
    return rows


def finite(rows, key):
    return [r for r in rows
            if r.get(key) is not None and r.get("duration_s") is not None
            and math.isfinite(r[key]) and math.isfinite(r["duration_s"])]


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def demean(rows, key):
    """Center key within each (corpus, mission) group. Returns list of values."""
    groups = {}
    for r in rows:
        groups.setdefault((r["corpus"], r["mission"]), []).append(r)
    out = []
    for r in rows:
        g = groups[(r["corpus"], r["mission"])]
        vals = [x[key] for x in g]
        out.append(r[key] - sum(vals) / len(vals))
    return out, groups


def within_slope_sign(rows, key):
    """Fraction of (corpus, mission) groups whose within-group slope is > 0."""
    groups = {}
    for r in rows:
        groups.setdefault((r["corpus"], r["mission"]), []).append(r)
    pos = tot = 0
    for g in groups.values():
        if len(g) < 3:
            continue
        xs = [r["duration_s"] for r in g]
        ys = [r[key] for r in g]
        if max(xs) - min(xs) <= 0:
            continue
        tot += 1
        if pearson(xs, ys) > 0:
            pos += 1
    return pos, tot


def within_r2(rows, y_key, x_keys):
    """Within (fixed-effects) R^2 of y on x, by group demeaning."""
    y_dm, groups = demean(rows, y_key)
    x_dm = [demean(rows, k)[0] for k in x_keys]
    n = len(rows)
    p = len(x_keys)
    # Least squares on demeaned data with no intercept.
    cols = [[x_dm[j][i] for j in range(p)] for i in range(n)]
    xtx = [[sum(cols[i][a] * cols[i][b] for i in range(n)) for b in range(p)]
           for a in range(p)]
    xty = [sum(cols[i][a] * y_dm[i] for i in range(n)) for a in range(p)]
    # Gaussian elimination on the p x p system (p is 1 or 2).
    aug = [xtx[a] + [xty[a]] for a in range(p)]
    for a in range(p):
        piv = max(range(a, p), key=lambda r: abs(aug[r][a]))
        aug[a], aug[piv] = aug[piv], aug[a]
        if abs(aug[a][a]) < 1e-30:
            return float("nan")
        for r in range(p):
            if r == a:
                continue
            f = aug[r][a] / aug[a][a]
            for c in range(a, p + 1):
                aug[r][c] -= f * aug[a][c]
    beta = [aug[a][p] / aug[a][a] for a in range(p)]
    ss_res = sum((y_dm[i] - sum(beta[j] * cols[i][j] for j in range(p))) ** 2
                 for i in range(n))
    ss_tot = sum(y * y for y in y_dm)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def report_pair(rows, y_key, x_key, label):
    pair = [r for r in finite(rows, y_key) if r.get(x_key) is not None
            and math.isfinite(r[x_key])]
    if len(pair) < 5:
        return None
    xs = [r[x_key] for r in pair]
    ys = [r[y_key] for r in pair]
    y_dm, _ = demean(pair, y_key)
    x_dm, _ = demean(pair, x_key)
    pos, tot = within_slope_sign(pair, y_key)
    return {
        "pair": label,
        "n": len(pair),
        "pooled_r": pearson(xs, ys),
        "pooled_rho": spearman(xs, ys),
        "within_r": pearson(x_dm, y_dm),
        "missions_pos_slope": f"{pos}/{tot}",
        "fe_r2_y_on_x": within_r2(pair, y_key, [x_key]),
        "fe_r2_y_on_x_plus_jerk": within_r2(pair, y_key, [x_key, "jerk"])
        if x_key != "jerk" and all(r.get("jerk") is not None for r in pair) else None,
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "separability.json"))
    args = ap.parse_args(argv)

    rows = []
    for name in OLD_CORPORA:
        if (RESULTS / name).exists():
            rows += load_old(name)
    for name in NEW_CORPORA:
        if (RESULTS / name).exists():
            rows += load_new(name)
    for name in Q18_CORPORA:
        if (RESULTS / name).exists():
            rows += load_q18(name)

    corpora = sorted({r["corpus"] for r in rows})
    missions = {(r["corpus"], r["mission"]) for r in rows}
    print(f"rows: {len(rows)}   corpora: {len(corpora)}   (corpus, mission) groups: {len(missions)}")

    # Scale check: is within-mission tracking variance real signal or noise?
    # A within-mission correlation means nothing if the residual barely moves.
    print("\n== within-mission spread (medians over groups) ==")
    print(f"{'corpus':<16}{'d(track)':>12}{'d(dur) ms':>12}{'d(track)/5mm':>14}{'n grp':>8}")
    for corpus in corpora:
        sub = [r for r in rows if r["corpus"] == corpus]
        groups = {}
        for r in sub:
            if r.get("tracking_m") is not None and r["duration_s"] is not None:
                groups.setdefault(r["mission"], []).append(r)
        t_stds, d_stds, n_grp = [], [], 0
        for g in groups.values():
            if len(g) < 3:
                continue
            n_grp += 1
            ts = [r["tracking_m"] for r in g]
            ds_ = [r["duration_s"] for r in g]
            mt, md = sum(ts) / len(ts), sum(ds_) / len(ds_)
            t_stds.append(math.sqrt(sum((t - mt) ** 2 for t in ts) / len(ts)))
            d_stds.append(math.sqrt(sum((d - md) ** 2 for d in ds_) / len(ds_)))
        if t_stds:
            t_med = sorted(t_stds)[len(t_stds) // 2] * 1000.0
            d_med = sorted(d_stds)[len(d_stds) // 2] * 1000.0
            print(f"{corpus:<16}{t_med:>10.3f}mm{d_med:>10.1f}{t_med / 5.0:>13.1%}{n_grp:>8}")

    q18 = [r for r in rows if r["corpus"] == "q18_decompose" and r.get("q_control_share") is not None]
    if q18:
        shares = sorted(r["q_control_share"] for r in q18)
        print(f"\nq18 residual composition: median control share "
              f"{shares[len(shares) // 2]:.1%} (n={len(shares)})")

    pairs = [("tracking_m", "duration_s", "tracking vs duration"),
             ("tube_m", "duration_s", "tube vs duration"),
             ("tracking_m", "jerk", "tracking vs jerk")]
    out = {"rows": len(rows), "corpora": corpora, "groups": len(missions), "pairs": []}

    def emit(rows_, title):
        print(f"\n== {title} ==")
        print(f"{'pair':<26}{'n':>6}{'pooled r':>10}{'rho':>8}{'within r':>10}"
              f"{'FE R2':>8}{'+jerk':>8}  missions+")
        for y_key, x_key, label in pairs:
            res = report_pair(rows_, y_key, x_key, label)
            if res is None:
                continue
            r2j = res["fe_r2_y_on_x_plus_jerk"]
            print(f"{label:<26}{res['n']:>6}{res['pooled_r']:>10.3f}"
                  f"{res['pooled_rho']:>8.3f}{res['within_r']:>10.3f}"
                  f"{res['fe_r2_y_on_x']:>8.3f}"
                  f"{(f'{r2j:.3f}' if r2j is not None else '-'):>8}"
                  f"  {res['missions_pos_slope']}")
            out["pairs"].append({"scope": title, **res})

    emit(rows, "all corpora pooled")
    for corpus in corpora:
        sub = [r for r in rows if r["corpus"] == corpus]
        emit(sub, corpus)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
