"""The margin measurement behind Q14 must keep its sign convention.

A margin is positive inside a limit and negative past it, each as a fraction
of its own limit, so the envelope, the tracking error and the wheel force can
be compared and the smallest one is the one that binds. A sign flip would turn
"0.9 of the way to the edge" into "0.9 of the margin left" and the selection
comparison in ``research/risk_margins.py`` would pick the riskiest plan.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research import risk_margins  # noqa: E402  (needs the path above)


def _candidate(envelope_m, limit_m, tracking_cm=0.0, peak=0.5):
    return {
        "duration_s": 1.0,
        "tracking_cm": tracking_cm,
        "peak_force_util": peak,
        "peak_traction_util": peak,
        "segments": [{"envelope_m": envelope_m, "limit_m": limit_m}],
    }


class TestRiskMargins(unittest.TestCase):

    def test_a_plan_inside_every_limit_has_a_positive_margin(self):
        found = risk_margins.margins(_candidate(0.015, 0.020, tracking_cm=0.40))
        self.assertAlmostEqual(found["envelope"], 0.25)
        self.assertAlmostEqual(found["tracking"], 0.20)
        self.assertAlmostEqual(found["force"], 0.50)
        self.assertAlmostEqual(found["accuracy"], 0.20)

    def test_a_plan_past_a_limit_has_a_negative_margin(self):
        found = risk_margins.margins(_candidate(0.030, 0.020, tracking_cm=0.60))
        self.assertAlmostEqual(found["envelope"], -0.5)
        self.assertAlmostEqual(found["tracking"], -0.2)
        self.assertAlmostEqual(found["accuracy"], -0.5)

    def test_the_worst_segment_binds(self):
        candidate = _candidate(0.010, 0.020)
        candidate["segments"].append({"envelope_m": 0.005, "limit_m": 0.005})
        self.assertAlmostEqual(risk_margins.margins(candidate)["envelope"], 0.0)

    def test_the_force_margin_does_not_mask_the_accuracy_margin(self):
        """A saturated motor is not a risk the operator can trade."""
        found = risk_margins.margins(_candidate(0.015, 0.020, peak=0.99))
        self.assertAlmostEqual(found["force"], 0.01)
        self.assertAlmostEqual(found["accuracy"], 0.25)
        self.assertAlmostEqual(found["min"], 0.01)

    def test_the_floor_rule_never_picks_a_thinner_margin(self):
        pool = [
            {"duration_s": 1.0, "margin": {"accuracy": 0.05}},
            {"duration_s": 1.3, "margin": {"accuracy": 0.40}},
        ]
        self.assertEqual(risk_margins._rule_fastest(pool)["duration_s"], 1.0)
        self.assertEqual(risk_margins._rule_fastest(pool, floor=0.10)["duration_s"], 1.3)
        self.assertIsNone(risk_margins._rule_fastest(pool, floor=0.60))


if __name__ == "__main__":
    unittest.main()
