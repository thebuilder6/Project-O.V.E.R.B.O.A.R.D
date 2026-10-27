"""The Q19 arm configs must differ from the declared robot in exactly one
constant at exactly the registered factor (docs/RESEARCH_AGENDA.md Q19).

The registration is binding: eight knobs at two levels each plus one nominal
arm (17 arms), each level's factor the range's low/high multiplier (the
``_factor`` semantics at spread 1 applied to the declared value), every other
constant and every declared uncertainty range as ``examples/fll_choreo.json``
declares it. ``gear_backlash_m`` is the knob whose declared range itself moves.
This fixture pins that identity: an arm that moved a second leaf, or moved its
knob at the wrong factor, would not be the registered experiment.
"""

import copy
import json
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from research import q19_oneknob  # noqa: E402  (needs the path above)

CONFIG_PATH = os.path.join(REPO_ROOT, "examples", "fll_choreo.json")

# The registration's own table, pinned here so a drift in research/robots.py
# cannot silently re-level the experiment.
REGISTERED = {
    "mass": (0.60, 1.50),
    "inertia": (0.60, 1.60),
    "track_width": (0.85, 1.15),
    "wheel_radius": (0.85, 1.15),
    "v_max_rad_s": (0.85, 1.15),
    "t_max_nm": (0.70, 1.30),
    "cof": (0.75, 1.25),
    "gear_backlash_m": (0.50, 2.00),
}

# Where each knob's one leaf lives in the canonical config.
LEAF_PATHS = {name: ("robot", name) for name in REGISTERED
              if name != "gear_backlash_m"}
LEAF_PATHS["gear_backlash_m"] = ("robot", "uncertainties", "gear_backlash_m")


def _config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _leaves(node, prefix=()):
    """Every scalar leaf of the config as (path, value); a list is a leaf."""
    if isinstance(node, dict):
        for key in sorted(node):
            yield from _leaves(node[key], prefix + (key,))
    else:
        yield prefix, node


class TestQ19OneKnob(unittest.TestCase):

    def test_the_levels_are_the_registered_bounds(self):
        self.assertEqual(dict(q19_oneknob.KNOBS), REGISTERED)
        self.assertEqual(len(q19_oneknob.ARMS), 17)
        self.assertEqual(q19_oneknob.ARMS[0], ("nominal", None, 1.0))

    def test_nominal_is_the_declared_robot(self):
        self.assertEqual(q19_oneknob.arm_config(_config(), "nominal"), _config())

    def test_each_arm_moves_exactly_one_constant_at_the_registered_factor(self):
        declared = _config()
        for arm, knob, factor in q19_oneknob.ARMS:
            if knob is None:
                continue
            with self.subTest(arm=arm):
                moved = q19_oneknob.arm_config(declared, arm)
                old_leaves = dict(_leaves(declared))
                new_leaves = dict(_leaves(moved))
                self.assertEqual(set(old_leaves), set(new_leaves))
                differences = [(path, old_leaves[path], new_leaves[path])
                               for path in old_leaves
                               if old_leaves[path] != new_leaves[path]]
                self.assertEqual(
                    len(differences), 1,
                    f"{arm} differs from the declared config in "
                    f"{len(differences)} leaves, not exactly one")
                path, old, new = differences[0]
                self.assertEqual(path, LEAF_PATHS[knob],
                                 f"{arm} moved {path}, not the knob's leaf")
                if isinstance(old, list):
                    self.assertEqual(len(old), len(new))
                    for before, after in zip(old, new):
                        self.assertAlmostEqual(after, before * factor, places=12)
                else:
                    self.assertAlmostEqual(new, old * factor, places=12)

    def test_building_an_arm_does_not_touch_the_declaration(self):
        declared = _config()
        before = copy.deepcopy(declared)
        for arm, knob, factor in q19_oneknob.ARMS:
            q19_oneknob.arm_config(declared, arm)
        self.assertEqual(declared, before)


if __name__ == "__main__":
    unittest.main()
