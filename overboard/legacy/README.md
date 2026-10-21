# Legacy & Obsolete Pathing Archive

This directory contains experimental, prototype, and research implementations that have been superseded by the **3-Pillar Unified Trajectory Engine** ([`unified_planner.py`](../core/unified_planner.py)). The Multi-Verse pipeline is archived here and has no CLI entry point.

---

## 1. Pipelines & Algorithms
* **`multiverse_optimizer.py`**:
  The experimental 5-phase Multi-Verse pipeline (Reeds-Shepp bootstrap, critic, JAX STOMP/TEB refinement). Superseded by the closed-form Homotopy Triad in `unified_planner.py`, which is $100\times$ faster ($< 1.5\text{ ms}$ vs. $2\text{ s}$) and achieves 100% gate certification.
* **`path_planning.py`**:
  Reeds-Shepp wrapper originally used by the Multi-Verse Phase 1 bootstrap.
* **`jax_optimizer.py`**:
  JAX STOMP/TEB gradient projection refiner used in Multi-Verse Phase 4. Running this archive requires `jaxopt`, which is no longer installed by `requirements-optional.txt`.
* **`benchmark_memetic_evolution.py`**:
  Stochastic memetic evolutionary search benchmark. Superseded by deterministic collocation with analytical seeds.
* **`benchmark_frontier_seed.py`**:
  Initial verification benchmark for analytical kinetic frontier seeds.

---

## 2. Research & Exploration (`legacy/evolutionary_research/`)
Contains the full suite of exploratory research scripts that mapped the non-convex solution landscape and discovered the 5 invariant differential-drive optimal control laws:
* 5-Pillar Multi-Island GA engines (`run_full_5pillar_evolution.py`, `run_extended_5pillar_marathon.py`, `run_centurion_evolution.py`)
* Geometric manifold sensitivity experiments (`compare_geometric_shapes.py`, `plot_individual_shapes_and_run_s_curve.py`)
* Convergence and phase portrait diagnostics (`plot_centurion_full_gallery.py`, `plot_seed_evolution.py`)

---

## 3. Sample & Debug Trajectories (`legacy/sample_trajectories/`)
Contains temporary output files and legacy benchmarks generated during development (`test_chimera.traj`, `test_sp.traj`, `complex_mission_optimized.json`, etc.).
