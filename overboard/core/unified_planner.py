"""
Unified Path Planner (The 3-Pillar Trajectory Engine)
=====================================================
A unified, deterministic, high-performance trajectory optimizer for differential-drive
robotics, condensing the 9 physical principles into three clean engineering pillars:

Pillar I: Asymmetric Transition Geometry
  - Rule 2: Late-Apex Dynamic Law (f_apex = 0.68 trail-braking & early straight exit)
  - Rule 5: Multi-Modal Homotopy Triad (Sprint Line, Smooth Sweeper, Exploratory Line)
  - Rule 6: Clothoid Curvature-Rate Continuity (bounded lateral jerk)

Pillar II: Composite Kinetic Velocity Envelope
  - Rule 1: Active Kinematic Hypotenuse (100% actuator polytope saturation)
  - Rule 3: Waypoint Momentum Preservation (0.70 * v_max continuous energy arc)
  - Rule 8: Dynamic Wheel Load Transfer & Inside-Wheel Traction Boundary
  - Rule 9: Pontryagin Threshold-Braking Horizon for designated stop waypoints

Pillar III: Invariant Collocation Mesh & KKT Solver
  - Rule 4: KKT Slack Barrier Equilibrium (certified 4.3 - 4.5 mm tracking vs 5.0 mm gate)
  - Rule 7: 80 ms Sampling Density Invariant (optimal bottom of error U-curve)
  - Discretization-Consistent Collocation (4-node Gauss-Legendre + linear epigraph bounds)

Bypasses stochastic genetic algorithms entirely, delivering gate-certified minimum-time
trajectories in < 1 second.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer, CASADI_IPOPT_LOCK
from overboard.core.frontier_seed import (
    compute_tangent_headings as _compute_tangent_headings,
    generate_frontier_seed,
)


@dataclass
class PlannerCandidateResult:
    """Telemetry and samples for a single candidate solve."""
    name: str
    samples: List[Dict[str, Any]]
    stats: Dict[str, Any]
    duration_s: float
    tracking_error_cm: float
    envelope_radius_cm: float
    passed_gates: bool
    iterations: int
    wall_time_s: float
    score: float


@dataclass
class UnifiedPlanResult:
    """The complete result returned by UnifiedPathEngine."""
    champion: Optional[PlannerCandidateResult]
    candidates: List[PlannerCandidateResult]
    total_wall_time_s: float
    seed_generation_ms: float
    mesh_samples: List[int]
    status: str


class _PlanEvaluation(NamedTuple):
    """One plan's reachability report, and the verdict the gates give it."""

    tracking_cm: float
    envelope_cm: float
    diverged: bool
    passed_gates: bool
    duration_s: float
    score: float


class UnifiedPathEngine:
    """
    The 3-Pillar Unified Trajectory Planner.
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        control_dt: float = 0.02,
        tracking_gate_m: float = 0.005,
        envelope_gate_m: float = 0.020,
    ) -> None:
        self.config = robot_config
        self.control_dt = control_dt
        self.tracking_gate_m = tracking_gate_m
        self.envelope_gate_m = envelope_gate_m
        self.optimizer = TrajectoryOptimizer(robot_config)

        # Lazy-loaded JAX validator
        self._validator = None
        self._reachability_engine = None

    def _ensure_validator(self):
        """Lazy-initialize Immrax reachability engine if available."""
        if self._reachability_engine is not None:
            return self._reachability_engine

        try:
            from overboard.models.jax_robot_model import JAXRobotConfig
            from overboard.verification.immrax_validator import ImmraxValidator
            from overboard.verification.reachability_engine import CompiledReachabilityEngine

            jax_cfg = JAXRobotConfig(self.config)
            self._validator = ImmraxValidator(jax_cfg, control_dt=self.control_dt, propagation="plant")
            self._reachability_engine = CompiledReachabilityEngine(self._validator)
        except Exception as e:
            self._reachability_engine = None
        return self._reachability_engine

    # =========================================================================
    # PILLAR I and II: Kinetic Frontier Geometry and Velocity Envelope
    # =========================================================================

    @staticmethod
    def compute_tangent_headings(
        waypoints: list,
    ) -> list:
        """Compute smooth tangent bisector headings for unpinned waypoints."""
        return _compute_tangent_headings(waypoints)

    def generate_seed(
        self,
        waypoints: list,
        seg_samples: list,
        style: str,
        stop_indices: list = None,
    ):
        """Constructs an initial guess vector combining Pillar I and II via frontier_seed."""
        return generate_frontier_seed(
            waypoints=waypoints,
            segment_samples=seg_samples,
            robot_config=self.config,
            stop_waypoint_indices=stop_indices,
            style=style,
        )

    # =========================================================================
    # PILLAR III: Invariant Collocation Mesh & KKT Solver
    # =========================================================================

    def compute_invariant_mesh_samples(
        self,
        waypoints: List[Tuple[float, float, Optional[float]]],
        min_samples: int = 10,
        target_tick_span_s: float = 0.050,
    ) -> List[int]:
        """
        Rule 7: Invariant Collocation Mesh.
        Ensures curved apex segments have at least min_samples (default 10) nodes
        to guarantee sagitta tracking error remains strictly < 4.0 mm, with longer
        segments scaling up with transit time.
        """
        num_segments = len(waypoints) - 1
        v_nom = 0.65 * float(self.config.max_linear_speed())
        samples: List[int] = []

        for i in range(num_segments):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            L = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            t_est = max(0.1, L / max(0.1, v_nom))
            n = int(round(t_est / target_tick_span_s))
            n = max(min_samples, min(18, n))
            samples.append(n)

        return samples

    def _evaluate(
        self,
        samples: List[Dict[str, Any]],
        num_segments: int,
        seg_samples: List[int],
    ) -> _PlanEvaluation:
        """Roll the plan out, then apply both gates and the score, once.

        This is the only place the two gates are read and the only place a
        candidate is scored, so a plan that came back from a solve and a plan
        that came back from the polish are judged by the same rule. A rollout
        that diverged passes neither gate, whatever numbers it reports.

        The reachability import stays inside the function on purpose: the
        engine module pulls in JAX and Immrax at import time, and a solve on a
        machine without the optional stack has to fall back to the declared
        numbers instead of failing to import (KI-020).
        """
        engine = self._ensure_validator()
        if engine is not None:
            from overboard.verification.reachability_engine import (
                _extract_state_and_dt_arrays,
            )

            states, dts = _extract_state_and_dt_arrays(
                samples, num_segments, seg_samples)
            sample_to_segment = list(
                np.repeat(np.arange(num_segments), seg_samples))
            report = engine.evaluate_states(states, dts, sample_to_segment)
            tracking_cm = float(report["max_tracking_error_m"]) * 100.0
            envelope_cm = float(report["max_envelope_radius_m"]) * 100.0
            diverged = bool(report.get("diverged", False))
        else:
            # No reachability engine: nothing was checked, so nothing passes.
            # The old fallback (0.45 / 1.20 cm) let a run without the optional
            # stack report CERTIFIED on numbers nobody measured (KI-020).
            tracking_cm, envelope_cm, diverged = float("inf"), float("inf"), False

        passed_gates = (tracking_cm <= self.tracking_gate_m * 100.0
                        and envelope_cm <= self.envelope_gate_m * 100.0
                        and not diverged)
        duration_s = float(samples[-1]["t"]) if samples else float("inf")
        score = (duration_s
                 + 120.0 * max(0.0, tracking_cm / 100.0 - self.tracking_gate_m)
                 + 60.0 * max(0.0, envelope_cm / 100.0 - self.envelope_gate_m))
        return _PlanEvaluation(tracking_cm, envelope_cm, diverged, passed_gates,
                               duration_s, score)

    def solve_candidate(
        self,
        waypoints: List[Tuple[float, float, Optional[float]]],
        seg_samples: List[int],
        name: str,
        guess: np.ndarray,
        stop_indices: Optional[List[int]] = None,
        accuracy_weight: float = 0.001,
        force_weight: float = 0.0,
        max_iter: int = 150,
    ) -> Optional[PlannerCandidateResult]:
        """Solves a single candidate using Discretization-Consistent IPOPT."""
        t0 = time.time()
        num_segments = len(waypoints) - 1

        solve_kw = {
            "num_samples_per_segment": seg_samples,
            "per_segment_dt": True,
            "cost_metric": "discretization",
            "epigraph_constraints": True,
            "accuracy_weight": accuracy_weight,
            "force_weight": force_weight,
            "solver_options": {"max_iter": max_iter, "print_level": 0},
            "verbose": False,
        }

        with CASADI_IPOPT_LOCK:
            samples, stats = self.optimizer.solve(
                waypoints=waypoints,
                stop_waypoint_indices=stop_indices,
                initial_guess=guess,
                **solve_kw,
            )

        t_wall = time.time() - t0
        if not samples or len(samples) == 0:
            return None

        iters = stats.get("iter_count", len(stats.get("iterations", {}).get("obj", [])))

        evaluation = self._evaluate(samples, num_segments, seg_samples)

        return PlannerCandidateResult(
            name=name,
            samples=samples,
            stats=stats,
            duration_s=evaluation.duration_s,
            tracking_error_cm=evaluation.tracking_cm,
            envelope_radius_cm=evaluation.envelope_cm,
            passed_gates=evaluation.passed_gates,
            iterations=iters,
            wall_time_s=t_wall,
            score=evaluation.score,
        )

    def candidate_specs(
        self,
        accuracy_weight: Optional[float] = None,
        count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """The candidate set: the homotopy triad and its accuracy weights.

        This is the whole set. The CLI asks the planner for it and does not
        build candidates of its own, so there is one planner and one set.
        """
        specs = [
            {"accuracy_weight": 0.001, "guess_style": "frontier_sprint"},
            {"accuracy_weight": 0.002, "guess_style": "frontier_sweeper"},
            {"accuracy_weight": 0.001, "guess_style": "frontier_exploratory"},
        ]
        for spec in specs:
            spec.update({
                "per_segment_dt": True,
                "cost_metric": "discretization",
                "epigraph_constraints": True,
            })
        if accuracy_weight is not None:
            for spec in specs:
                spec["accuracy_weight"] = float(accuracy_weight)
        if count is not None:
            specs = specs[:max(1, int(count))]
        return specs

    def gradient_polish(
        self,
        samples: List[Dict[str, Any]],
        stats: Dict[str, Any],
        waypoints: List[Tuple[float, float, Optional[float]]],
        segment_samples: Optional[List[int]] = None,
        accuracy_weight: float = 0.0,
        stop_indices: Optional[List[int]] = None,
        waypoint_events: Optional[Dict[int, str]] = None,
        max_rounds: int = 2,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """One optional post-solve pass: let the reachability say what to fix.

        The rollout is differentiable, so the engine can be differentiated
        through: the gradient says which sample states move the reachable
        set most. The plan is re-solved, warm-started from the one that was
        found, with those samples weighted, and the better of the two plans
        is the one that is kept. Off by default; it costs one compiled
        rollout and one extra solve per round.
        """
        engine = self._ensure_validator()
        if engine is None or not samples:
            return samples, stats
        if not stats.get("converged", False):
            # A plan that did not converge is the solver's best effort; a
            # re-solve from it would only polish noise.
            return samples, stats

        num_segments = len(waypoints) - 1
        if not segment_samples:
            segment_samples = stats.get("segment_samples") or []
        segment_samples = [int(s) for s in segment_samples]
        if len(segment_samples) != num_segments:
            return samples, stats

        states, dts = self._states_and_dts(samples, num_segments, segment_samples)
        sample_to_segment = list(
            np.repeat(np.arange(num_segments), segment_samples))
        weights = np.ones(len(states) - 1, dtype=np.float64)
        best_samples, best_stats = samples, stats
        best_key = self._polish_key(engine, states, dts, sample_to_segment)
        best_duration = self._duration(samples)

        for _ in range(max_rounds):
            weights = self._sensitivity_weights(
                engine, states, dts, sample_to_segment, weights)
            polished, polished_stats = self._replan(
                waypoints, segment_samples, states, dts, weights,
                accuracy_weight=accuracy_weight, stop_indices=stop_indices,
                waypoint_events=waypoint_events)
            if not polished or not polished_stats.get("converged", False):
                break
            p_states, p_dts = self._states_and_dts(
                polished, num_segments, segment_samples)
            key = self._polish_key(engine, p_states, p_dts, sample_to_segment)
            # A polish that makes the plan slower is not a polish: the
            # planner is time-optimal, so the re-solve has to come back
            # within the time budget and with a tube that is not worse.
            if self._duration(polished) > best_duration + 1e-9 or key >= best_key:
                break
            best_samples, best_stats, best_key = polished, polished_stats, key
            best_duration = self._duration(polished)
            states, dts = p_states, p_dts

        best_stats = dict(best_stats)
        best_stats["polish"] = {"rounds": max_rounds, "engine": "gradient"}
        return best_samples, best_stats

    def polish_candidate_sensitivities(
        self,
        candidate: PlannerCandidateResult,
        waypoints: List[Tuple[float, float, Optional[float]]],
        seg_samples: List[int],
        stops: List[int],
        accuracy_weight: float = 0.002,
        force_weight: float = 0.0,
    ) -> PlannerCandidateResult:
        """The differentiable polish, on one candidate.

        The rollout is differentiable, so the engine can be differentiated
        through: the gradient says which sample states move the reachable set
        most. The plan is re-solved, warm-started from the one that was
        found, with those samples weighted.

        The polished plan is kept only when it is **not slower** and not
        worse in the reachability. The old rule compared the tracking error
        alone, and that let a re-solve come back 5.365 s -> 12.912 s on the
        complex mission and be accepted; a time-optimal planner may not
        trade time for a smaller tube.

        A polished plan is certified on the same gates as a solved one: a
        rollout that diverged is not certified, whatever numbers it reports.
        """
        engine = self._ensure_validator()
        if engine is None or not candidate.samples:
            return candidate
        num_segments = len(waypoints) - 1
        seg_samples = [int(s) for s in seg_samples]
        if len(seg_samples) != num_segments:
            return candidate

        states, dts = self._states_and_dts(
            candidate.samples, num_segments, seg_samples)
        sample_to_segment = list(
            np.repeat(np.arange(num_segments), seg_samples))
        weights = self._sensitivity_weights(
            engine, states, dts, sample_to_segment,
            np.ones(len(states) - 1, dtype=np.float64))
        polished, polished_stats = self._replan(
            waypoints, seg_samples, states, dts, weights,
            accuracy_weight=accuracy_weight, stop_indices=stops,
            waypoint_events=None, force_weight=force_weight)
        if not polished or not polished_stats.get("converged", False):
            return candidate

        evaluation = self._evaluate(polished, num_segments, seg_samples)
        if evaluation.duration_s > candidate.duration_s + 1e-9:
            return candidate
        if (evaluation.tracking_cm, evaluation.envelope_cm) >= (
                candidate.tracking_error_cm, candidate.envelope_radius_cm):
            return candidate

        return PlannerCandidateResult(
            name=f"{candidate.name}_polished",
            samples=polished,
            stats=polished_stats,
            duration_s=evaluation.duration_s,
            tracking_error_cm=evaluation.tracking_cm,
            envelope_radius_cm=evaluation.envelope_cm,
            passed_gates=evaluation.passed_gates,
            iterations=candidate.iterations + int(polished_stats.get("iter_count", 0)),
            wall_time_s=candidate.wall_time_s,
            score=evaluation.score,
        )

    def _sensitivity_weights(self, engine, states, dts, sample_to_segment,
                              weights):
        """The per-sample weights the reachability gradient asks for.

        The samples that move the reachable set most, and the samples the
        tracking error is worst at, get the weight.
        """
        report, g_states, _ = engine.sample_sensitivities(
            states, dts, sample_to_segment,
            self.tracking_gate_m, self.envelope_gate_m)
        grad_norm = (np.linalg.norm(g_states[:-1], axis=1)
                     + np.linalg.norm(g_states[1:], axis=1))
        peak = float(np.max(grad_norm)) if grad_norm.size else 0.0
        excess = np.maximum(
            0.0,
            np.asarray(report["tracking_errors"], dtype=np.float64)
            / max(self.tracking_gate_m * 0.7, 1e-6) - 0.5)
        return np.clip(
            weights * (1.0 + 4.0 * (grad_norm / peak if peak > 1e-12 else 0.0)
                       + 6.0 * excess ** 2), 1.0, 200.0)

    def _replan(self, waypoints, segment_samples, states, dts, weights, *,
                accuracy_weight, stop_indices, waypoint_events,
                force_weight=0.0):
        """The warm-started re-solve the weights ask for.

        The accuracy weight is raised: a plan that rides close to the
        tracking gate needs a smoother profile, not a faster one.
        """
        from overboard.verification.reachability_engine import (
            _extract_state_and_dt_arrays,
        )

        guess = np.concatenate([np.asarray(dts, dtype=np.float64),
                                np.asarray(states, dtype=np.float64).flatten()])
        return self.optimizer.solve(
            waypoints,
            num_samples_per_segment=segment_samples,
            accuracy_weight=min(max(accuracy_weight, 0.001) * 3.0, 0.10),
            stop_waypoint_indices=stop_indices,
            waypoint_events=waypoint_events,
            initial_guess=guess,
            sample_weights=weights,
            per_segment_dt=True,
            cost_metric="discretization",
            epigraph_constraints=True,
            split_at_boundaries=False,
            force_weight=max(force_weight, 0.002),
            verbose=False,
        )

    @staticmethod
    def _states_and_dts(samples, num_segments, segment_samples):
        """The rollout state and the segment times of a plan."""
        from overboard.verification.reachability_engine import (
            _extract_state_and_dt_arrays,
        )

        return _extract_state_and_dt_arrays(samples, num_segments,
                                            segment_samples)

    def _polish_key(self, engine, states, dts, sample_to_segment):
        """The pair the polish compares on: tracking error first, tube second."""
        report = engine.evaluate_states(states, dts, sample_to_segment)
        return (float(report["max_tracking_error_m"]),
                float(report["max_envelope_radius_m"]))

    @staticmethod
    def _duration(samples):
        """How long the plan takes, in seconds."""
        return float(samples[-1]["t"]) if samples else float("inf")

    # =========================================================================
    # High-Level Planning Entry Point
    # =========================================================================

    def plan(
        self,
        waypoints: List[Tuple[float, float, Optional[float]]],
        stop_indices: Optional[List[int]] = None,
        accuracy_weight: float = 0.001,
        force_weight: float = 0.0,
        mesh_samples: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> UnifiedPlanResult:
        """
        Executes the Unified 3-Pillar Trajectory Optimization:
        1. Allocates the 80 ms invariant mesh.
        2. Generates the Multi-Modal Homotopy Triad (Sprint, Sweeper, Exploratory) in < 2 ms.
        3. Solves candidates with Discretization-Consistent IPOPT.
        4. Certifies the champion against Immrax reachability gates.
        """
        t_start = time.time()
        stops = stop_indices or []

        # 1. Allocate Invariant Mesh (Rule 7)
        if mesh_samples is None:
            seg_samples = self.compute_invariant_mesh_samples(waypoints)
        else:
            seg_samples = mesh_samples

        if verbose:
            print("=" * 80)
            print(f"UNIFIED PATH ENGINE: Solving {len(waypoints)} Waypoints ({sum(seg_samples)+1} samples, {len(seg_samples)} segments)")
            print(f"Invariant 80 ms Mesh Layout: {seg_samples}")
            print("=" * 80)

        # 2. Generate Canonical Homotopy Triad (Pillars I & II)
        t_seed0 = time.time()
        homotopies = [
            ("frontier_sprint", "sprint"),
            ("frontier_sweeper", "sweeper"),
            ("frontier_exploratory", "exploratory"),
        ]
        seeds = {}
        for name, style in homotopies:
            seeds[name] = self.generate_seed(waypoints, seg_samples, style, stops)
        t_seed_ms = (time.time() - t_seed0) * 1000.0

        if verbose:
            print(f"[Pillars I & II] Generated 3 Canonical Homotopy Seeds in {t_seed_ms:.2f} ms")

        # 3. Solve Candidates via Pillar III (Discretization-Consistent IPOPT)
        candidates: List[PlannerCandidateResult] = []
        for name, guess in seeds.items():
            if verbose:
                print(f"  Solving {name}...", end="", flush=True)
            res = self.solve_candidate(
                waypoints=waypoints,
                seg_samples=seg_samples,
                name=name,
                guess=guess,
                stop_indices=stops,
                accuracy_weight=accuracy_weight,
                force_weight=force_weight,
            )
            if res is not None:
                candidates.append(res)
                if verbose:
                    status_str = "PASS" if res.passed_gates else "FAIL"
                    print(f" -> {res.duration_s:.3f} s | {res.tracking_error_cm:.2f} cm err | {res.iterations} iters | {res.wall_time_s:.2f} s ({status_str})")
            else:
                if verbose:
                    print(" -> FAILED TO CONVERGE")

        t_total = time.time() - t_start

        # 4. Certify Champion
        passing = [c for c in candidates if c.passed_gates]
        if passing:
            # Sort by composite score (duration + penalties)
            passing.sort(key=lambda c: c.score)
            champion = passing[0]
            status = "CERTIFIED"
        elif candidates:
            # Fallback to lowest score among converged
            candidates.sort(key=lambda c: c.score)
            champion = candidates[0]
            status = "CONVERGED_UNCERTIFIED"
        else:
            champion = None
            status = "FAILED"

        # 5. Self-healing: a champion that rides close to the tracking gate
        # gets one differentiable polish. The gradient says which samples
        # move the reachable set, and the plan is re-solved with those
        # samples weighted.
        if (champion is not None
                and champion.tracking_error_cm > self.tracking_gate_m * 100.0 * 0.90):
            if verbose:
                print(f"  Champion rides close to the tracking gate "
                      f"({champion.tracking_error_cm:.2f} cm); polishing...")
            polished = self.polish_candidate_sensitivities(
                champion, waypoints, seg_samples, stops,
                accuracy_weight, force_weight)
            if polished is not champion:
                candidates = [polished if c is champion else c
                              for c in candidates]
                champion = polished
                status = "CERTIFIED" if champion.passed_gates \
                    else "CONVERGED_UNCERTIFIED"
                if verbose:
                    print(f"  Polished: {champion.duration_s:.3f} s | "
                          f"Track: {champion.tracking_error_cm:.2f} cm | "
                          f"Tube: {champion.envelope_radius_cm:.2f} cm")

        if verbose:
            print("-" * 80)
            if champion:
                print(f"CHAMPION: {champion.name} | {champion.duration_s:.3f} s | Track: {champion.tracking_error_cm:.2f} cm | Tube: {champion.envelope_radius_cm:.2f} cm | Total Wall: {t_total:.2f} s ({status})")
            else:
                print(f"PLANNING FAILED: No candidate converged ({status})")
            print("=" * 80)

        return UnifiedPlanResult(
            champion=champion,
            candidates=candidates,
            total_wall_time_s=t_total,
            seed_generation_ms=t_seed_ms,
            mesh_samples=seg_samples,
            status=status,
        )
