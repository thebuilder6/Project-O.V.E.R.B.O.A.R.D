"""Q20: the solver-failure taxonomy.

Labels describe *why* a solve failed (max_iter / infeasible / numeric / unknown);
they never change eligibility - a failed solve is a failed solve either way.
"""

import unittest

from overboard.models.robot_model import RobotConfig
from overboard.core.optimizer import TrajectoryOptimizer, classify_solver_error


CONFIG = {
    "robot": {
        "mass": 0.8,
        "inertia": 0.002,
        "track_width": 0.1,
        "wheel_radius": 0.03,
        "v_max_rad_s": 15.0,
        "t_max_nm": 0.05,
        "gearing": 1.0,
        "cof": 0.5,
    }
}

# The real CasADi assertion text, as recorded in the Q19 arm records.
MAX_ITER_MSG = (
    "Error in Opti::solve [OptiNode] at .../casadi/core/optistack.cpp:217:\n"
    ".../casadi/core/optistack_internal.cpp:1338: Assertion "
    "\"return_success(accept_limit)\" failed:\n"
    "Solver failed. You may use opti.debug.value to investigate the latest "
    "values of variables. return_status is 'Maximum_Iterations_Exceeded'"
)


class TestClassifySolverError(unittest.TestCase):
    def test_max_iter_from_the_recorded_assertion(self):
        self.assertEqual(classify_solver_error(MAX_ITER_MSG), "max_iter")

    def test_infeasible_and_restoration(self):
        self.assertEqual(
            classify_solver_error("return_status is 'Infeasible_Problem_Detected'"),
            "infeasible")
        self.assertEqual(
            classify_solver_error("return_status is 'Restoration_Failed'"),
            "infeasible")

    def test_numeric(self):
        self.assertEqual(
            classify_solver_error("return_status is 'Invalid_Number_Detected'"),
            "numeric")

    def test_unknown_stays_honest(self):
        self.assertEqual(classify_solver_error("something else entirely"), "unknown")
        self.assertEqual(classify_solver_error(""), "unknown")


class TestFailureStatsCarryKind(unittest.TestCase):
    """A budget-capped solve must fail exactly as before, now labelled."""

    @staticmethod
    def _capped_solve(**kwargs):
        optimizer = TrajectoryOptimizer(RobotConfig(CONFIG))
        return optimizer.solve(
            [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
            num_samples_per_segment=5,
            accuracy_weight=0.0,
            verbose=False,
            solver_options={"max_iter": 1},
            **kwargs,
        )

    def test_unsplit_path_labels_the_failure(self):
        _, stats = self._capped_solve(split_at_boundaries=False)
        self.assertFalse(stats["converged"])
        self.assertEqual(stats["error_kind"], "max_iter")
        self.assertIn("Maximum_Iterations_Exceeded", stats["error"])

    def test_split_path_propagates_the_label(self):
        # A boundary stop (pinned heading + stop index) splits the path; the
        # stitched stats must still name the failure (Q20 merge propagation).
        _, stats = self._capped_solve(stop_waypoint_indices=[1])
        self.assertEqual(stats.get("sub_mission_count"), 2)
        self.assertFalse(stats["converged"])
        self.assertEqual(stats["error_kind"], "max_iter")
        self.assertIn("Maximum_Iterations_Exceeded", stats["error"])


if __name__ == "__main__":
    unittest.main()
