"""The Q18 decomposition instrument must read zero where there is nothing to
measure, and reproduce the analytic chord bound where there is.

The synthetic self-consistent arc (analytic circle positions, heading =
tangent) is the fixture: at the gate's measurement instants the smooth
planned path and the chorded reference share the planned sample point, so
every decomposition component must read ~0 there, and the instrument's chord
bound must match the closed-form sagitta of a circular arc. This is the
identity the Q18 decision rule turns on: the gate's tracking quantity is the
control residual at the gate's instants (docs/RESEARCH_AGENDA.md Q18).
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research import q18_decompose  # noqa: E402  (needs the path above)


class TestQ18Decomposition(unittest.TestCase):

    def test_arc_control_instrument_zero(self):
        result = q18_decompose.arc_control()
        q18_decompose.arc_control_self_check(result)
        zero = result["self_consistent"]
        self.assertLess(zero["max_plan_hits_samples_m"], 1e-9)
        self.assertLess(zero["at_samples_rs_m"], 1e-9)
        self.assertLess(zero["at_samples_sc_m"], 1e-9)
        self.assertLess(zero["at_samples_rc_m"], 1e-9)

    def test_chord_bound_matches_the_analytic_sagitta(self):
        result = q18_decompose.arc_control()
        self.assertLess(abs(result["analytic_midpoint_chord_m"]
                            - result["instrument_midpoint_chord_m"]), 1e-12)
        self.assertLess(abs(result["self_consistent"]["interval_chord_bound_m"]
                            - result["analytic_midpoint_chord_m"]), 1e-6)

    def test_shape_vanishes_at_the_gate_instants(self):
        # The structural identity on a closed-loop rollout: at the sample
        # instants |S - C| is zero and |R - C| equals |R - S|, so the gate's
        # quantity is already the control residual there.
        result = q18_decompose.arc_control()
        self.assertLess(result["closed_loop_q_sagitta_samples_m"], 1e-9)
        self.assertAlmostEqual(result["closed_loop_q_control_samples_m"],
                               result["closed_loop_q_reported_m"], places=9)


if __name__ == "__main__":
    unittest.main()
