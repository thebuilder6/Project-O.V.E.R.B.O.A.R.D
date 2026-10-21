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

import overboard.gui_server as gui_server

REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ = REPO_ROOT / "overboard" / "viz"


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


class TestANewSolveClearsTheLastVerdict(unittest.TestCase):
    """A new run must not be read through the last run's verdict.

    A solve without validation prints no `##OVERBOARD-VALIDATION##` line,
    so nothing took the old report off the screen: the old candidate
    chips, the old segment table, the old "the robot tracked the path to
    within X cm" note and the old red segment on the field all stayed,
    while the header said "not validated".
    """

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_solve_clears_the_verdict_before_it_starts(self):
        body = self._function("solve")
        self.assertIn("renderValidationReport(null)", body)

    def test_the_report_owns_the_verdict(self):
        """One place owns it: a report draws it, no report takes it away."""
        body = self._function("renderValidationReport")
        self.assertIn("state.validationReport = report || null", body)
        self.assertIn("validationReport: null", self.source)

    def test_a_report_without_candidates_clears_the_table(self):
        body = self._function("renderValidationReport")
        empty = body.index("if (!report")
        for element in ("el('validation-table')", "el('validation-candidates')",
                        "el('validation-note')", "el('validation-explainer')"):
            self.assertIn(element, body[empty:])

    def test_the_report_without_candidates_hides_the_box(self):
        body = self._function("renderValidationReport")
        empty = body.index("if (!report")
        self.assertIn("box.hidden = true", body[empty:])

    def test_the_failing_segments_are_a_set_and_the_worst_is_a_number(self):
        self.assertIn("failingSegments: new Set()", self.source)
        self.assertIn("worstSegment: -1", self.source)

    def test_the_field_reads_the_failing_segments_fresh(self):
        body = self._function("drawNow")
        self.assertIn("state.failingSegments.has(i)", body)
        self.assertIn("state.worstSegment === i", body)

    def test_a_cleared_verdict_takes_the_red_segments_with_it(self):
        body = self._function("renderValidationReport")
        empty = body.index("if (!report")
        self.assertIn("state.failingSegments = new Set()", body[empty:])
        self.assertIn("state.worstSegment = -1", body[empty:])
        self.assertIn("draw()", body[empty:])


class TestThePushNeedsAMissionThatMatches(unittest.TestCase):
    """A trajectory that does not match the mission on screen is not pushed.

    The robot would drive the old path, and the studio would be telling
    the operator that the path on the field is the answer to the mission
    in front of them.
    """

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_readiness_looks_at_the_mission(self):
        body = self._function("updatePushButton")
        self.assertIn("state.dirty", body)
        self.assertIn("!state.dirty", body)

    def test_the_status_says_the_mission_changed(self):
        body = self._function("updatePushButton")
        said = " ".join(re.findall(r"status\.textContent = '([^']*)'", body))
        self.assertIn("changed since this solve", said)

    def test_the_other_wording_is_kept(self):
        body = self._function("updatePushButton")
        said = " ".join(re.findall(r"status\.textContent = '([^']*)'", body))
        self.assertIn("not been validated", said)
        self.assertIn("Solve a mission first.", said)

    def test_the_push_itself_refuses_a_changed_mission(self):
        body = self._function("pushToHub")
        self.assertIn("state.dirty", body)

    def test_a_completed_solve_clears_the_change(self):
        """The trajectory is the answer to the mission on the screen, so the
        mission has not moved on since the solve."""
        body = self._function("pollJob")
        self.assertIn("state.dirty = false", body)
        self.assertIn("missionFingerprint()", body)

    def test_the_solve_remembers_the_mission_it_solves(self):
        body = self._function("solve")
        self.assertIn("state.solvingFingerprint = missionFingerprint()", body)

    def test_every_edit_marks_the_mission_as_changed(self):
        """The button has to follow the edit, not the next solve."""
        self.assertIn("function markMissionChanged()", self.source)
        body = self._function("markMissionChanged")
        self.assertIn("state.dirty = true", body)
        self.assertIn("updatePushButton()", body)


class TestAReloadAttachesToTheRunningSolve(unittest.TestCase):
    """A reload in the middle of a solve must attach to that solve.

    The server refuses a second solve, so a studio that did not ask came
    back with "a solve is already running" on every Solve press, and the
    Cancel button had no job to cancel.
    """

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_running_job_is_asked_for_outside_the_auto_solve_branch(self):
        body = self._function("init")
        ask = body.index("api('/api/jobs/current')")
        branch = body.rindex("initial.auto_solve")
        self.assertLess(ask, branch,
                        "The studio only asks for the running job under --live.")

    def test_the_ask_comes_before_the_studio_starts_a_solve(self):
        body = self._function("init")
        self.assertLess(body.index("api('/api/jobs/current')"), body.rindex("solve()"))

    def test_a_running_job_is_adopted(self):
        body = self._function("init")
        ask = body.index("api('/api/jobs/current')")
        branch = body.rindex("initial.auto_solve")
        adopted = body[ask:branch]
        self.assertIn("state.jobId = running.id", adopted)
        self.assertIn("setBusy(true)", adopted)
        self.assertIn("pollJob()", adopted)

    def test_the_trajectory_is_only_remembered_with_a_trajectory(self):
        body = self._function("pollJob")
        ok = body.index("job.status === 'ok'")
        self.assertIn("state.solveJobId = state.jobId", body[ok:])
        self.assertIn("state.solveHasTrajectory", body[ok:])


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


class TestTheDownloadsBelongToTheirJob(unittest.TestCase):
    """A download must ask for the job that owns the file.

    After a push the downloads box holds the *push* job's artifacts (the
    session the hub printed), but the download asked for the solve job, so
    the button 404'd (KI-044).
    """

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_download_asks_for_the_job_it_was_given(self):
        body = self._function("renderDownloads")
        self.assertIn("jobId", body)
        self.assertIn("${owner}", body)

    def test_the_caller_passes_the_job_id(self):
        self.assertIn("renderDownloads(job.artifacts, job.id);", self.source)


class TestTheEnvelopeNamesItsModel(unittest.TestCase):
    """An envelope is the width of a reachability tube.

    The tube is a property of the propagation, and the two propagations
    answer different questions: the plant propagation bounds an open-loop
    plant driven by a pre-recorded command, and the tick map bounds the
    closed loop the robot actually runs. An envelope shown without its
    model cannot be judged, so the studio names it (KI-049).
    """

    def setUp(self):
        self.source = _read(VIZ / "studio.js")

    def _function(self, name):
        start = self.source.index(f"function {name}(")
        end = self.source.find("\nfunction ", start + 1)
        return self.source[start:end if end > 0 else len(self.source)]

    def test_the_note_names_the_model(self):
        body = self._function("renderValidationReport")
        self.assertIn("candidate.propagation", body)
        self.assertIn("report.propagation", body)

    def test_both_models_are_named(self):
        body = self._function("renderValidationReport")
        self.assertIn("closed-loop tick map", body)
        self.assertIn("open-loop plant", body)

    def test_the_chip_tooltip_carries_it(self):
        body = self._function("renderValidationReport")
        self.assertIn("entry.propagation", body)


if __name__ == "__main__":
    unittest.main()
