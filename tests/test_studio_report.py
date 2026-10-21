"""The studio's report and its project file.

The report is built in the browser and posted to the server, which writes it
next to the trajectory and serves it back as a page. The project file is
one JSON with the mission, the robot, the options and the field: a setup
that can be handed to somebody else.
"""

import io
import json
import re
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ = REPO_ROOT / "overboard" / "viz"


def _read(path):
    with io.open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class TestReportBuilder(unittest.TestCase):
    """The report is a plain script that builds a document."""

    def setUp(self):
        self.source = _read(VIZ / "report.js")

    def test_the_builder_is_defined_and_exported(self):
        self.assertIn("function buildMissionReport(", self.source)
        self.assertIn("global.buildMissionReport = buildMissionReport", self.source)

    def test_the_report_escapes_what_the_operator_typed(self):
        """A mission name is the operator's text, so it cannot become markup."""
        self.assertIn("function esc(", self.source)
        self.assertIn("esc(mission.name", self.source)

    def test_the_report_is_printable_and_self_contained(self):
        self.assertIn("@media print", self.source)
        self.assertIn("@page", self.source)
        # The style is inside the document, so a saved report still looks right.
        self.assertIn("<style>${CSS}</style>", self.source)

    def test_the_report_carries_the_mission_the_robot_and_the_result(self):
        for heading in ("The mission", "The robot", "The result"):
            self.assertIn(heading, self.source)


class TestStudioReportRoute(unittest.TestCase):
    """The server writes the report and serves it back."""

    def test_the_report_is_a_static_asset(self):
        import overboard.gui_server as gui_server
        self.assertIn("/report.js", gui_server.STATIC_FILES)

    def test_the_report_is_written_next_to_the_trajectory(self):
        """The report is an artifact, so it lives with the solve."""
        from overboard.gui_server import artifact_path

        class FakeJob:
            name = "unit-test"
            output_dir = Path("C:/nowhere")

        path = artifact_path(FakeJob(), "report")
        self.assertIsNotNone(path)
        self.assertTrue(path.name.endswith("_report.html"))
        self.assertEqual(path.parent, Path("C:/nowhere"))


class TestProjectFile(unittest.TestCase):
    """The project file is one JSON with everything a setup needs."""

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def test_the_project_has_a_format_and_a_version(self):
        self.assertIn("const PROJECT_FORMAT = 'overboard-project'", self.source)
        self.assertIn("const PROJECT_VERSION = 1", self.source)

    def test_the_project_carries_the_mission_the_robot_and_the_options(self):
        for key in ("mission:", "robot:", "options:", "model:", "field:"):
            self.assertIn(key, self.source)

    def test_a_file_that_is_not_a_project_is_refused(self):
        """Opening a plain mission file as a project must say so, not guess."""
        self.assertIn("data.format !== PROJECT_FORMAT", self.source)
        self.assertIn("not an OVERBOARD project", self.source)


class TestStudioInventoryNewControls(unittest.TestCase):
    """The new controls are in the page and in the inventory."""

    def test_the_report_and_project_buttons_exist(self):
        html = _read(VIZ / "studio.html")
        for button in ("btn-report", "btn-download-project", "btn-upload-project",
                       "file-project"):
            self.assertIn(f'id="{button}"', html)

    def test_the_studio_loads_the_report_builder(self):
        html = _read(VIZ / "studio.html")
        self.assertIn('src="report.js"', html)
        self.assertLess(html.index('src="report.js"'), html.index('src="studio.js"'))


if __name__ == "__main__":
    unittest.main()
