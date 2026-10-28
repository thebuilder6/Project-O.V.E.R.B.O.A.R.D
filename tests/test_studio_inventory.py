"""The studio's UI inventory must match the page it describes.

`docs/STUDIO_UI_INVENTORY.md` is the reference for the studio's structure.
A list that drifts from the page is worse than no list, so the counts and
the structure in it are checked here: a control added to the studio fails
the test until the document is updated.
"""

import io
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "overboard" / "viz" / "studio.html"
JS_PATH = ROOT / "overboard" / "viz" / "studio.js"
DOC_PATH = ROOT / "docs" / "STUDIO_UI_INVENTORY.md"
SYSID_HTML_PATH = ROOT / "overboard" / "viz" / "sysid.html"
SYSID_JS_PATH = ROOT / "overboard" / "viz" / "sysid.js"
SYSID_CSS_PATH = ROOT / "overboard" / "viz" / "sysid.css"


def _read(path):
    with io.open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class TestStudioInventory(unittest.TestCase):
    """The document's counts and structure must be the page's."""

    def setUp(self):
        self.html = _read(HTML_PATH)
        self.js = _read(JS_PATH)
        self.doc = _read(DOC_PATH)

    def test_every_id_the_script_looks_up_exists_in_the_page(self):
        """A dangling id is a silent failure in the browser."""
        in_html = set(re.findall(r'id="([a-z0-9-]+)"', self.html))
        used = set(re.findall(r"el\('([a-z0-9-]+)'\)", self.js))
        missing = sorted(used - in_html)
        self.assertEqual(missing, [], f"The script looks up ids the page lacks: {missing}")

    def test_the_control_counts_in_the_document_are_right(self):
        """The document quotes the counts, so they have to be the real ones."""
        ids = re.findall(r'id="([a-z0-9-]+)"', self.html)
        buttons = len(re.findall(r'<button[^>]*id="', self.html))
        inputs = len(re.findall(r'<input[^>]*id="', self.html))
        selects = len(re.findall(r'<select[^>]*id="', self.html))
        self.assertIn(f"{len(ids)} element ids", self.doc)
        self.assertIn(f"{buttons} buttons, {inputs} inputs, {selects} selects", self.doc)
        self.assertEqual(len(ids), len(set(ids)), "An id appears twice in the page.")

    def test_the_document_lists_every_button_in_the_page(self):
        """A button with no row in the inventory is a gap in the inventory."""
        missing = []
        for match in re.finditer(r"<button[^>]*id=\"([a-z0-9-]+)\"[^>]*>(.*?)</button>",
                                 self.html, re.S):
            button_id, body = match.group(1), match.group(2)
            label = " ".join(re.sub(r"<[^>]+>", " ", body).split())
            # The row may name the id or the label; either proves it is listed.
            if button_id not in self.doc and label and label not in self.doc:
                missing.append(f"{button_id} ({label})")
        self.assertEqual(missing, [], f"Buttons missing from the inventory: {missing}")

    def test_the_document_lists_every_generated_region(self):
        """The regions the script fills in must be named in the document."""
        for element_id in ["robot-fields", "robot-range-fields", "sheet-table",
                           "weigh-table", "plot-tabs", "legend-items",
                           "validation-candidates", "validation-table",
                           "fix-row", "downloads", "history-list",
                           "wp-list", "score-categories", "lever-table"]:
            self.assertIn(element_id, self.html, f"{element_id} is gone from the page")
            self.assertIn(element_id, self.doc,
                          f"{element_id} is not in the inventory document")

    def test_every_tab_has_a_panel_and_vice_versa(self):
        """A tab that points nowhere, or a panel with no tab, is a dead end."""
        tabs = re.findall(r'<button[^>]*role="tab"[^>]*aria-controls="([a-z0-9-]+)"', self.html)
        panels = re.findall(r'<section[^>]*class="tabpanel"[^>]*id="([a-z0-9-]+)"', self.html)
        self.assertTrue(tabs, "The page has no tabs.")
        self.assertEqual(sorted(tabs), sorted(panels),
                         "A tab has no panel, or a panel has no tab.")
        self.assertEqual(len(tabs), len(set(tabs)), "A panel is claimed by two tabs.")
        for tab in tabs:
            self.assertIn(tab, self.doc, f"The tab {tab} is not in the inventory.")

    def test_exactly_one_tab_per_strip_starts_open(self):
        """A strip with two open tabs, or none, shows the wrong panel."""
        for strip in re.findall(r'<div[^>]*role="tablist"[^>]*>(.*?)</div>', self.html, re.S):
            selected = re.findall(r'aria-selected="true"', strip)
            self.assertEqual(len(selected), 1,
                             "A tab strip must start with exactly one open tab.")


class TestSysIdInventory(unittest.TestCase):
    """The guided SysId page (`/sysid`) has its own inventory section.

    It is a separate page, not a tab, so the studio counts above do not
    cover it. A control added to it fails here until the document names it.
    """

    def setUp(self):
        self.html = _read(SYSID_HTML_PATH)
        self.js = _read(SYSID_JS_PATH)
        self.css = _read(SYSID_CSS_PATH)
        self.doc = _read(DOC_PATH)

    def test_the_sysid_files_exist(self):
        for path in (SYSID_HTML_PATH, SYSID_JS_PATH, SYSID_CSS_PATH):
            self.assertTrue(path.is_file(), f"{path.name} is missing.")

    def test_every_id_the_sysid_script_looks_up_exists_on_its_page(self):
        in_html = set(re.findall(r'id="([a-z0-9-]+)"', self.html))
        used = set(re.findall(r"getElementById\('([^']+)'\)", self.js))
        missing = sorted(used - in_html)
        self.assertEqual(missing, [], f"The SysId script looks up ids the page lacks: {missing}")

    def test_the_sysid_hides_behind_hidden_like_the_studio(self):
        rules = re.findall(r"([^{}]+)\{([^{}]*)\}", self.css)
        hidden = [body for selector, body in rules
                  if "[hidden]" in selector and "display" in body
                  and "none" in body]
        self.assertTrue(hidden, "sysid.css has no rule that hides [hidden].")

    def test_the_sysid_never_solves(self):
        """Finish hands the fit to Solve; the page itself solves nothing."""
        self.assertNotIn("solve(", self.js)
        self.assertIn("overboard.sysid.handoff.v1", self.js)

    def test_the_pages_link_both_ways(self):
        studio = _read(HTML_PATH)
        self.assertIn('id="link-sysid"', studio)
        self.assertIn('href="/sysid"', studio)
        self.assertIn('link-sysid', self.doc)
        self.assertIn('href="/"', self.html)

    def test_the_step_mode_chains_single_pushes(self):
        self.assertIn("sys-step-mode", self.html)
        self.assertIn("runNextStep", self.js)
        self.assertIn("state.step", self.js)

    def test_the_prompts_feed_mirrors_the_hub(self):
        self.assertIn("sys-prompts", self.html)
        self.assertIn("renderPrompts", self.js)
        self.assertIn("bindSheetInput", self.js)

    def test_every_test_names_its_setup(self):
        self.assertIn("Setup first", self.js)

    def test_the_document_lists_the_sysid_controls(self):
        for element_id in ["sys-profile", "sys-collect", "sys-save", "sys-fit", "sys-finish",
                           "sys-copy-log", "sys-log-live", "sys-queue", "sys-select-all", "sys-select-none",
                           "sys-express", "sys-anchor-pick", "sys-lab",
                           "sys-tier", "sys-anchor",
                           "sys-anchor-note", "sys-step-mode", "sys-bench",
                           "sys-prompts", "sys-sheet", "sys-weigh",
                           "sys-summary", "sys-compare", "sys-meta",
                           "sys-tests"]:
            self.assertIn(element_id, self.html, f"{element_id} is gone from the page")
            self.assertIn(element_id, self.doc,
                           f"{element_id} is not in the inventory document")


if __name__ == "__main__":
    unittest.main()
