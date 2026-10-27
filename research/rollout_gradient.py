"""Phase 2 probe (F2): is the rollout residual a usable gradient signal?

Phase 1 (research/residual_features.py) killed the static surrogate: the
tracking residual is not predictable from recorded plan features with stable
signs. The remaining route for an in-loop residual term is the differentiated
rollout itself - the same physics the gate checks. Before building anything
into the optimizer, three mechanical questions decide whether that route is
alive:

  A. activity   - when is the polish loss's hinge nonzero? (the loss in
                  reachability_engine._get_grad_fn is a hinge-squared on the
                  per-sample tracking errors and tube radii, with targets
                  0.4 mm / 1.0 mm below the gates)
  B. descent    - is the gradient a descent direction of the loss the gate
                  actually evaluates (30 ticks x 5 plant substeps), not just
                  of the coarse loss it was taken through (1 tick x 1
                  substep, GRAD_TICKS_PER_INTERVAL = 1)?
  C. fidelity   - how much do the two resolutions' losses disagree at the
                  probe points?

Two arms per plan:
  real    (0.005 / 0.020)   - documents A on ordinary saved plans. Expect
                              exactly zero gradient on comfortable plans:
                              that is the mechanical explanation of Q16's
                              byte-identical gradient polish (the hinge is a
                              gate-violation penalty, not a general price).
  relaxed (0.001 / 0.005)   - activates the hinge on ordinary plans and
                              probes B and C on the landscape mechanics.
                              Proxy for the near-limit situation at real
                              gates (Q13's failures ride ~5 mm).

Stage 1 (as originally pre-registered) tested the existing gradient: taken
through the 1-tick x 1-substep rollout (GRAD_TICKS_PER_INTERVAL = 1), it
describes a diverged propagation (phantom tracking errors of 133-351 mm where
the gate reports 0.3-3.0 mm) and is dead as an in-loop signal. A resolution
sweep then locates the smallest faithful rollout (the fidelity cliff sits
between 2x2 and 5x5; 5x5 matches the gate exactly).

Stage 2 asks the refined question: does the gradient of the SMALLEST FAITHFUL
rollout (5x5) descend the gate-resolution (30x5) loss? Pre-registered reading:
the in-loop rollout term (M1) has legs iff that gradient steps monotonically
reduce the full-resolution loss for at least two step sizes and the 5x5 loss
agrees with the 30x5 loss to within 10% at the probe points. Otherwise a term
built on this gradient would descend a function the gate does not measure.

    python research/rollout_gradient.py
    python research/rollout_gradient.py --limit 2

Measurement only; nothing in the planner changes. Needs JAX + Immrax.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS = Path(__file__).resolve().parent / "results"
TRAJ_DIR = RESULTS / "traj"

TRACK_GATE = 0.005
TUBE_GATE = 0.020
TRACK_RELAXED = 0.001
TUBE_RELAXED = 0.005
ETAS = (1e-4, 1e-3)  # column-normalised step sizes (metres-ish per state dim)
CONTROL_DT = 0.020


def loss_from_report(rep, track_gate, tube_gate):
    """The _get_grad_fn loss, recomputed from a rollout report.

    targets are 0.4 mm / 1.0 mm below the gates, exactly as in the engine.
    """
    t_errs = np.asarray(rep["tracking_errors"], dtype=np.float64).ravel()
    r_errs = np.asarray(rep["radii"], dtype=np.float64).ravel()
    t_target = track_gate - 0.0004
    r_target = tube_gate - 0.0010
    th = np.maximum(0.0, t_errs - t_target)
    rh = np.maximum(0.0, r_errs - r_target)
    return 1e6 * float(np.sum(th ** 2)) + 1e5 * float(np.sum(rh ** 2)), \
        int(np.count_nonzero(th)), int(np.count_nonzero(rh))


def norm_cols(g):
    """Per-column absolute scale of a gradient array, for unit-free steps."""
    return np.maximum(np.abs(g).max(axis=0), 1e-30)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS / "rollout_gradient.json"))
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp

    from overboard.models.jax_robot_model import JAXRobotConfig
    from overboard.models.robot_model import RobotConfig
    from overboard.verification.immrax_validator import ImmraxValidator
    from overboard.verification.reachability_engine import (
        CompiledReachabilityEngine,
        _extract_state_and_dt_arrays,
    )

    paths = sorted(TRAJ_DIR.glob("*.traj"))[:args.limit]
    out = {"plans": []}
    verdicts = []
    fixed_verdicts = []

    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg = RobotConfig(raw["trajectory"]["config"])
        samples = raw["trajectory"]["samples"]
        n = len(samples)
        if n < 3:
            continue
        # Every interval is its own segment, so the engine's per-segment dt
        # model reproduces the plan's true interval times exactly.
        seg_samples = [1] * (n - 1)
        states, dts = _extract_state_and_dt_arrays(samples, n - 1, seg_samples)
        s2s = list(range(n - 1))
        states = np.asarray(states, dtype=np.float64)
        dts = np.asarray(dts, dtype=np.float64)

        validator = ImmraxValidator(JAXRobotConfig(cfg), control_dt=CONTROL_DT,
                                    propagation="plant")
        engine = CompiledReachabilityEngine(validator)

        plan_out = {"plan": path.stem, "samples": n}
        for arm, (g_track, g_tube) in (("real", (TRACK_GATE, TUBE_GATE)),
                                       ("relaxed", (TRACK_RELAXED, TUBE_RELAXED))):
            rep_full = engine.evaluate_states(states, dts, s2s)
            rep_g, g_states, g_dts = engine.sample_sensitivities(
                states, dts, s2s, g_track, g_tube)
            loss_full, th_f, rh_f = loss_from_report(rep_full, g_track, g_tube)
            loss_grad, th_g, rh_g = loss_from_report(rep_g, g_track, g_tube)

            # Self-check: the rollout is deterministic.
            rep_full2 = engine.evaluate_states(states, dts, s2s)
            assert float(rep_full2["max_tracking_error_m"]) == \
                float(rep_full["max_tracking_error_m"]), "rollout not deterministic"

            g = np.asarray(g_states, dtype=np.float64)
            gnorm = float(np.abs(g).max()) if g.size else 0.0
            # Self-check: no active hinge -> exactly zero gradient.
            if th_g == 0 and rh_g == 0:
                assert gnorm == 0.0 and float(np.abs(np.asarray(g_dts)).max()) == 0.0, \
                    "gradient nonzero with no active hinge"

            steps = []
            g_dts_arr = np.asarray(g_dts, dtype=np.float64).reshape(-1)
            scale_s = norm_cols(g) if g.size else np.ones(5)
            scale_d = float(np.max(np.abs(g_dts_arr))) if g_dts_arr.size else 1.0
            scale_d = max(scale_d, 1e-30)
            for eta in ETAS:
                # Each state column and dt step at most eta in its own unit -
                # a unit-free read of the descent direction.
                s2 = states - eta * g / scale_s
                d2 = np.maximum(dts - eta * g_dts_arr / scale_d, 1e-4)
                rep2_full = engine.evaluate_states(s2, d2, s2s)
                rep2_g, _, _ = engine.sample_sensitivities(s2, d2, s2s, g_track, g_tube)
                l2_full, _, _ = loss_from_report(rep2_full, g_track, g_tube)
                l2_grad, _, _ = loss_from_report(rep2_g, g_track, g_tube)
                steps.append({
                    "eta": eta,
                    "loss_full": l2_full,
                    "loss_grad": l2_grad,
                    "max_track_m": float(rep2_full["max_tracking_error_m"]),
                    "max_tube_m": float(rep2_full["max_envelope_radius_m"]),
                })
            arm_out = {
                "gates": [g_track, g_tube],
                "hinges_track": th_g,
                "hinges_tube": rh_g,
                "grad_max_abs": gnorm,
                "loss_full": loss_full,
                "loss_grad": loss_grad,
                "fidelity_rel": abs(loss_grad - loss_full) / max(loss_full, 1.0),
                "max_track_m": float(rep_full["max_tracking_error_m"]),
                "max_track_gradres_m": float(rep_g["max_tracking_error_m"]),
                "max_tube_m": float(rep_full["max_envelope_radius_m"]),
                "steps": steps,
            }
            plan_out[arm] = arm_out

            if arm == "relaxed":
                desc_full = [s["loss_full"] < loss_full for s in steps]
                desc_grad = [s["loss_grad"] < loss_grad for s in steps]
                verdicts.append({
                    "plan": path.stem,
                    "descent_full": desc_full,
                    "descent_grad": desc_grad,
                    "fidelity_rel": arm_out["fidelity_rel"],
                })
            print(f"{path.stem:8} {arm:8} hinges t/r {th_g:4d}/{rh_g:3d}  |g|max {gnorm:.3e}  "
                  f"loss full {loss_full:.3e} grad {loss_grad:.3e} "
                  f"fidelity {arm_out['fidelity_rel']:.2%}  "
                  f"track {arm_out['max_track_m'] * 1000:.2f}mm "
                  f"(grad-res {arm_out['max_track_gradres_m'] * 1000:.2f}mm) "
                  f"tube {arm_out['max_tube_m'] * 1000:.1f}mm")
            # Resolution sweep: how fine must the grad rollout be before it
            # measures what the gate measures? One plan, once.
            if arm == "real" and out.get("sweep") is None:
                seg_idx_arr = np.asarray(s2s, dtype=np.int32)
                active_mask = np.ones(n - 1, dtype=bool)
                w_lo, w_hi = engine._uncertainty_bounds()
                ref = float(rep_full["max_tracking_error_m"])
                sweep = []
                for t_, s_ in ((1, 1), (2, 2), (5, 5), (10, 5), (30, 5)):
                    fn = engine._get_scan_fn(t_, s_)
                    rp = fn(states, dts, seg_idx_arr, active_mask, w_lo, w_hi)
                    mt = float(rp["max_tracking_error_m"])
                    sweep.append({"ticks": t_, "substeps": s_, "max_track_m": mt})
                    ok = "match" if abs(mt - ref) <= 0.10 * max(ref, 1e-9) else "off"
                    print(f"{'':17}sweep {t_:2d} ticks x {s_} substeps  "
                          f"max_track {mt * 1000:9.2f}mm  ({ok})")
                out["sweep"] = {"plan": path.stem, "ref_max_track_m": ref,
                                "configs": sweep}
            for s in steps:
                print(f"{'':17}eta {s['eta']:.0e}  loss full {s['loss_full']:.3e} "
                      f"grad {s['loss_grad']:.3e}  track {s['max_track_m'] * 1000:.2f}mm")

        # Stage 2: differentiate the smallest faithful rollout (5x5, per the
        # sweep) and ask whether its gradient descends the gate-resolution
        # loss. Relaxed gates only - the near-limit proxy, hinge active.
        g_track, g_tube = TRACK_RELAXED, TUBE_RELAXED
        seg_idx_arr = np.asarray(s2s, dtype=np.int32)
        active_mask = np.ones(n - 1, dtype=bool)
        w_lo, w_hi = engine._uncertainty_bounds()
        fn55 = engine._get_scan_fn(5, 5)

        def loss55(s_arr, d_arr):
            rep = fn55(s_arr, d_arr, seg_idx_arr, active_mask, w_lo, w_hi)
            th = jnp.maximum(0.0, rep["tracking_errors"] - (g_track - 0.0004))
            rh = jnp.maximum(0.0, rep["radii"] - (g_tube - 0.0010))
            return 1e6 * jnp.sum(th ** 2) + 1e5 * jnp.sum(rh ** 2)

        gfn = jax.jit(jax.value_and_grad(loss55, argnums=(0, 1)))
        val55, (gs55, gd55) = gfn(jnp.asarray(states), jnp.asarray(dts))
        val55 = float(val55)
        gs55 = np.asarray(gs55, dtype=np.float64)
        gd55 = np.asarray(gd55, dtype=np.float64).reshape(-1)
        rep_full = engine.evaluate_states(states, dts, s2s)
        loss_full, _, _ = loss_from_report(rep_full, g_track, g_tube)
        fid55 = abs(val55 - loss_full) / max(loss_full, 1.0)

        gnorm55 = float(np.abs(gs55).max()) if gs55.size else 0.0
        steps55 = []
        scale_s = norm_cols(gs55) if gs55.size else np.ones(5)
        scale_d = max(float(np.max(np.abs(gd55))) if gd55.size else 1.0, 1e-30)
        for eta in ETAS:
            s2 = states - eta * gs55 / scale_s
            d2 = np.maximum(dts - eta * gd55 / scale_d, 1e-4)
            rep2 = engine.evaluate_states(s2, d2, s2s)
            l2, _, _ = loss_from_report(rep2, g_track, g_tube)
            l2_55 = float(loss55(jnp.asarray(s2), jnp.asarray(d2)))
            steps55.append({"eta": eta, "loss_full": l2, "loss_grad": l2_55,
                            "max_track_m": float(rep2["max_tracking_error_m"]),
                            "max_tube_m": float(rep2["max_envelope_radius_m"])})
        desc_full = [s["loss_full"] < loss_full for s in steps55]
        plan_out["fixed_5x5"] = {
            "gates": [g_track, g_tube],
            "loss_5x5": val55,
            "loss_full": loss_full,
            "fidelity_rel": fid55,
            "grad_max_abs": gnorm55,
            "descent_full": desc_full,
            "steps": steps55,
        }
        fixed_verdicts.append({"plan": path.stem, "descent_full": desc_full,
                               "fidelity_rel": fid55})
        print(f"{path.stem:8} fixed5x5  |g|max {gnorm55:.3e}  loss 5x5 {val55:.3e} "
              f"full {loss_full:.3e} fidelity {fid55:.2%}")
        for s in steps55:
            print(f"{'':17}eta {s['eta']:.0e}  loss full {s['loss_full']:.3e} "
                  f"5x5 {s['loss_grad']:.3e}  track {s['max_track_m'] * 1000:.2f}mm")
        out["plans"].append(plan_out)

    if verdicts:
        ok_desc = sum(1 for v in verdicts if any(v["descent_full"]))
        mono = sum(1 for v in verdicts if all(v["descent_full"]))
        fid = max(v["fidelity_rel"] for v in verdicts)
        out.setdefault("verdict", {})["production_gradient"] = {
            "plans_descenting_full_loss": ok_desc,
            "plans_monotone": mono,
            "plans": len(verdicts),
            "worst_fidelity_rel": fid,
        }
        print(f"\nverdict production gradient: full-loss descent on "
              f"{ok_desc}/{len(verdicts)} plans (monotone {mono}/{len(verdicts)}), "
              f"worst grad-vs-full loss disagreement {fid:.2%}")
    if fixed_verdicts:
        ok_f = sum(1 for v in fixed_verdicts if any(v["descent_full"]))
        mono_f = sum(1 for v in fixed_verdicts if all(v["descent_full"]))
        fid_f = max(v["fidelity_rel"] for v in fixed_verdicts)
        out.setdefault("verdict", {})["fixed_5x5"] = {
            "plans_descenting_full_loss": ok_f,
            "plans_monotone": mono_f,
            "plans": len(fixed_verdicts),
            "worst_fidelity_rel": fid_f,
        }
        print(f"verdict 5x5 gradient: full-loss descent on {ok_f}/{len(fixed_verdicts)} "
              f"plans (monotone {mono_f}/{len(fixed_verdicts)}), worst 5x5-vs-full "
              f"loss disagreement {fid_f:.2%}")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
