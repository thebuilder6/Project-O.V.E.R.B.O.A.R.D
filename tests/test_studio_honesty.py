"""The studio must not tell the operator something that is not true.

These are the rules a beginner leans on: the button that says a trajectory
is approved is only enabled when it was validated, a failed solve does not
leave the last solve's path on the field, a solve that never starts does not
leave the last verdict on the screen, and the error the operator reads does
not carry the internal path of a temporary folder.
"""

import io
import json
import re
import threading
import unittest
import urllib.request
from pathlib import Path

import gui_server

REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ = REPO_ROOT / "viz"


def _read(path):
    with io.open(path, "r", encoding="utf-8") as handle:
        return handle.read()


class TestTheHiddenAttributeWins(unittest.TestCase):
    """A hidden element must be hidden.

    A class that sets `display` - a `.row` is a flex row - is an author
    rule, so it beats the browser's own `[hidden]` rule. Every element the
    studio hides with the attribute was therefore on the screen anyway.
    """

    def setUp(self):
        self.css = _read(VIZ / "studio.css")

    def test_there_is_a_rule_that_hides_the_hidden_attribute(self):
        rules = re.findall(r"([^{}]+)\{([^{}]*)\}", self.css)
        hidden = [body for selector, body in rules
                  if "[hidden]" in selector and "display" in body
                  and "none" in body]
        self.assertTrue(hidden, "The CSS has no rule that hides [hidden].")

    def test_the_rule_beats_a_class_that_sets_display(self):
        """`.row { display: flex }` used to put hidden rows back on screen."""
        rules = re.findall(r"([^{}]+)\{([^{}]*)\}", self.css)
        row_at = None
        hidden_at = None
        hidden_important = False
        for index, (selector, body) in enumerate(rules):
            selector = " ".join(selector.split())
            if ".row" in selector and "display" in body and row_at is None:
                row_at = index
            if "[hidden]" in selector and "display" in body and "none" in body:
                hidden_at = index
                hidden_important = "important" in body
        self.assertIsNotNone(row_at, "The CSS has no .row rule to check against.")
        self.assertIsNotNone(hidden_at, "The CSS has no [hidden] rule.")
        self.assertTrue(hidden_important or hidden_at > row_at,
                        "The [hidden] rule has to come after the .row rule, "
                        "or it has to be important.")


class TestThePushNeedsAValidatedTrajectory(unittest.TestCase):
    """Only a checked trajectory goes to the robot."""

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_button_is_off_without_a_validated_solve(self):
        body = self._function("updatePushButton")
        self.assertIn("state.solveValidated", body)
        self.assertIn("el('btn-push').disabled = !ready || !checked", body)

    def test_the_status_does_not_call_an_unchecked_trajectory_approved(self):
        body = self._function("updatePushButton")
        said = " ".join(re.findall(r"status\.textContent = '([^']*)'", body))
        self.assertNotIn("approved", said)
        self.assertIn("not been validated", said)

    def test_the_push_itself_refuses_an_unchecked_trajectory(self):
        body = self._function("pushToHub")
        self.assertIn("state.solveValidated", body)

    def test_the_verdict_comes_from_the_validator_not_the_switch(self):
        """A solve with a measured model is validated even when the box was
        not ticked, so the switch cannot be the answer."""
        body = self._function("pollJob")
        self.assertIn("state.solveValidated = Boolean(selected.ok)", body)


class TestAFailedSolveLeavesNoPath(unittest.TestCase):
    """The field must not keep the last solve's path as the answer."""

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_there_is_a_way_to_clear_the_path(self):
        body = self._function("clearTrajectory")
        self.assertIn("state.trajectory = null", body)
        self.assertIn("playback.duration = 0", body)

    def test_the_failure_branch_clears_it(self):
        body = self._function("pollJob")
        failure = body.index("job.status === 'failed'") if "job.status === 'failed'" in body \
            else body.index("} else {")
        self.assertIn("clearTrajectory()", body[failure:])

    def test_the_solve_clears_it_before_it_starts(self):
        body = self._function("solve")
        self.assertIn("clearTrajectory()", body)


class TestASolveThatNeverStartsClearsTheVerdict(unittest.TestCase):
    """The header must not keep the last solve's answer."""

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def test_the_guards_refuse_the_solve(self):
        body = self.source[self.source.index("async function solve()"):]
        body = body[:body.index("\nfunction ")]
        self.assertIn("refuseSolve(", body)

    def test_refusing_sets_the_status_and_the_result(self):
        start = self.source.index("function refuseSolve(")
        body = self.source[start:start + 600]
        self.assertIn("setStatus(", body)
        self.assertIn("setResultStatus(", body)


class TestTheMessageTheOperatorReads(unittest.TestCase):
    """The error on the screen is for the operator, not for the machine."""

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_temporary_path_is_left_out(self):
        body = self._function("cleanError")
        self.assertIn("No trajectory written to", body)
        self.assertIn("No trajectory was written", body)

    def test_the_error_prefix_is_not_repeated(self):
        body = self._function("cleanError")
        self.assertIn("Error:", body)

    def test_the_result_names_the_next_step(self):
        """The sentence after the badge has to be a sentence."""
        source = self.source
        self.assertIn("Not validated</span> This trajectory has not been", source)
        self.assertNotIn("Not validated</span> enable validation", source)

    def test_the_lever_the_message_names_is_one_click_away(self):
        html = _read(VIZ / "studio.html")
        self.assertIn('id="btn-validate"', html)
        self.assertIn('id="validate-row"', html)
        self.assertIn("el('btn-validate').addEventListener", self.source)


class TestTheResultExplainsItself(unittest.TestCase):
    """The rule behind the winner is said out loud."""

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_selection_rule_is_stated(self):
        body = self._function("describeCandidates")
        self.assertIn("tightest envelope", body)
        self.assertIn("breaks a tie", body)

    def test_a_candidate_that_passed_and_lost_says_why(self):
        body = self._function("describeCandidates")
        self.assertIn("did not win", body)

    def test_the_chip_names_the_seed(self):
        """Two candidates with the same timing model are the same chip
        without it."""
        body = self._function("renderValidationReport")
        self.assertIn("entry.style || 'baseline'", body)

    def test_the_key_hides_what_is_not_on_the_field(self):
        body = self._function("updateLegend")
        self.assertIn("row.hidden = !visible[key]", body)


class TestTheSheetWithoutASession(unittest.TestCase):
    """No session yet is not an error: the studio asks on every load."""

    def setUp(self):
        self.server = gui_server.create_server(
            0, {"fast": {"samples": 4}}, {"config": None, "waypoints": None})
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_the_sheet_is_an_empty_one_not_an_error(self):
        request = urllib.request.Request(
            f"{self.base}/api/calibration/sheet?name=unit-test-no-such-robot")
        with urllib.request.urlopen(request, timeout=30) as response:
            self.assertEqual(response.status, 200)
            payload = json.loads(response.read().decode("utf-8"))
        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["total"], 0)
        self.assertFalse(payload["session"])


if __name__ == "__main__":
    unittest.main()
