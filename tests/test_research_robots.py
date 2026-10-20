"""The random robot configs must stay inside the robot's own ranges.

A robot is a set of constants, and the planner reads every one of them. The
randomizer moves the constants and the declared ranges with them, so a robot
with 30 mm wheels has a wheel-radius range around 30 mm. If the nominal
value could fall outside its own range, the validator would be planning
against a robot that does not exist.

The test module lives with the other research tests: the randomizer is
measurement code, and nothing in the production tree imports it.
"""

import copy
import json
import os
import random
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research import robots  # noqa: E402  (needs the path above)
from robot_model import RobotConfig  # noqa: E402

CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "fll_choreo.json")


def _config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


class TestRandomRobots(unittest.TestCase):

    def test_no_jitter_is_the_declared_robot(self):
        """A spread of 0 must not move anything."""
        data = _config()
        jittered, factors = robots.jitter(data, random.Random(1), 0.0)
        self.assertEqual(jittered, data)
        for name, factor in factors.items():
            self.assertEqual(factor, 1.0, name)

    def test_the_config_of_the_caller_is_not_changed(self):
        data = _config()
        before = copy.deepcopy(data)
        robots.jitter(data, random.Random(2), 1.0)
        self.assertEqual(data, before)

    def test_the_nominal_stays_inside_its_own_range(self):
        """Every drawn robot is a robot the declared ranges describe."""
        data = _config()
        for index in range(50):
            jittered, factors = robots.robot_for(data, 7, f"m{index}", 1.0)
            config = RobotConfig(jittered)
            ranges = config.uncertainties
            self.assertLessEqual(min(ranges["wheel_radius_range"]),
                                 config.wheel_radius,
                                 factors)
            self.assertLessEqual(config.wheel_radius,
                                 max(ranges["wheel_radius_range"]), factors)
            self.assertLessEqual(min(ranges["track_width_range"]),
                                 config.track_width, factors)
            self.assertLessEqual(config.track_width,
                                 max(ranges["track_width_range"]), factors)
            self.assertLessEqual(min(ranges["cof"]), config.cof, factors)
            self.assertLessEqual(config.cof, max(ranges["cof"]), factors)
            self.assertLessEqual(min(ranges["inertia_kg_m2"]),
                                 config.inertia, factors)
            self.assertLessEqual(config.inertia,
                                 max(ranges["inertia_kg_m2"]), factors)

    def test_the_constants_actually_move(self):
        """A spread of 1 has to move something, or the run tests one robot."""
        data = _config()
        moved = 0
        for index in range(20):
            _, factors = robots.robot_for(data, 7, f"m{index}", 1.0)
            moved += sum(1 for factor in factors.values() if factor != 1.0)
        self.assertGreater(moved, 20)

    def test_the_same_key_gives_the_same_robot(self):
        """A run has to be reproducible."""
        data = _config()
        first, first_factors = robots.robot_for(data, 7, "m000#0", 1.0)
        second, second_factors = robots.robot_for(data, 7, "m000#0", 1.0)
        self.assertEqual(first, second)
        self.assertEqual(first_factors, second_factors)

    def test_the_legacy_config_shape_is_read(self):
        """The Choreo shape nests the constants under {"val": ...}."""
        legacy = {"config": {"mass": {"val": 0.7}, "wheel_radius": {"val": 0.03}}}
        jittered, factors = robots.jitter(
            legacy, random.Random(3), 1.0)
        self.assertIn("mass", factors)
        self.assertNotEqual(jittered["config"]["mass"]["val"], 0.7)
        self.assertLessEqual(0.0, jittered["config"]["mass"]["val"])


if __name__ == "__main__":
    unittest.main()
