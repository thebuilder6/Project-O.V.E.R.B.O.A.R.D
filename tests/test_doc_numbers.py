"""The numbers a document quotes must agree with the code that owns them.

Every documented constant drifts the moment the code moves, and a doc that
quotes a stale one is a claim the tree cannot back. These tests read the
number back out of the code and refuse a doc that disagrees. The test count
is deliberately not one of them: `run_tests.py` prints that, so no document
states it at all.
"""

import inspect
import os
import re
import unittest

from overboard.core.unified_planner import UnifiedPathEngine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the invariant mesh is named, in the code and in the docs.
MESH_SPAN_FILES = (
    "overboard/core/unified_planner.py",
    "overboard/gui_server.py",
    "overboard/viz/studio.html",
    "overboard/viz/studio.js",
    "docs/AGENT_OPS.md",
    "docs/STUDIO_UI_INVENTORY.md",
)

# The widest accuracy-weight ladder anyone measured is 0.02 (pq_b, section 4).
MEASURED_MAX_ACCURACY_WEIGHT = 0.02


def _read(relative_path):
    with open(os.path.join(REPO_ROOT, relative_path), encoding="utf-8") as handle:
        return handle.read()


# The docs that describe the tree as it is now. The research records
# (HEURISTICS, STRATEGY_LEDGER, RESEARCH_AGENDA, research/*) quote the counts of
# the tree they were measured on, which is provenance, not a current claim.
CURRENT_STATE_DOCS = (
    "README.md",
    "AGENTS.md",
    "KNOWN_ISSUES.md",
    "docs/AGENT_OPS.md",
    "docs/WHITEPAPER.md",
    "docs/CONFIGURATION.md",
    "docs/EXAMPLES.md",
)


class TestDocNumbers(unittest.TestCase):
    def test_the_documented_mesh_span_is_the_one_the_allocator_uses(self):
        default = inspect.signature(
            UnifiedPathEngine.compute_invariant_mesh_samples
        ).parameters["target_tick_span_s"].default
        span_ms = int(round(default * 1000.0))
        patterns = (
            r"invariant (\d+) ms",
            r"(\d+) ms invariant",
            r"mesh \((\d+) ms\)",
            r"(\d+) ms Sampling",
        )
        for path in MESH_SPAN_FILES:
            text = _read(path)
            for pattern in patterns:
                for found in re.findall(pattern, text, re.IGNORECASE):
                    self.assertEqual(
                        int(found), span_ms,
                        f"{path} says the mesh spans {found} ms; the code uses {span_ms} ms",
                    )

    def test_the_documented_accuracy_weights_are_on_the_measured_scale(self):
        readme = _read("README.md")
        start = readme.index("## Accuracy vs. Speed Tradeoff")
        end = readme.index("\n## ", start + 1)
        quoted = []
        for line in readme[start:end].splitlines():
            label = re.match(r"- `([^`]+)`", line)
            if label:
                quoted.extend(float(n) for n in re.findall(r"\d+\.\d+", label.group(1)))
        self.assertTrue(quoted, "the README accuracy ladder is gone")
        self.assertLessEqual(
            max(quoted), MEASURED_MAX_ACCURACY_WEIGHT,
            "the README recommends a weight outside the measured ladder",
        )

    def test_no_document_quotes_a_test_count(self):
        pattern = re.compile(r"\b\d{3,4}\s+(?:automated\s+)?tests?\b", re.IGNORECASE)
        for path in CURRENT_STATE_DOCS:
            found = pattern.search(_read(path))
            self.assertIsNone(
                found,
                f"{path} quotes {found.group(0) if found else ''}; "
                "`python run_tests.py` prints the count, so no doc should",
            )


if __name__ == "__main__":
    unittest.main()
