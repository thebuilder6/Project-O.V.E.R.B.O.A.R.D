"""The drivebase profile wizard and the example profile.

The wizard asks a few questions with measurement tips and writes the JSON
every calibration tool takes as ``--profile``. These tests pin that the file
it writes is one ``read_drivebase_profile`` reads back whole, that an
implausible answer is refused with a message that says why, and that the
shipped example carries every field the reader reads.
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from overboard.sysid import profile as profile_wizard
from overboard.sysid.fit import read_drivebase_profile

EXAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "drivebase_profile.example.json")


class TestProfileWizard(unittest.TestCase):
    """Questions in, a profile the tools accept out."""

    def _main(self, answers, extra=None):
        given = iter(answers)

        def input_fn(_prompt):
            return next(given)

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            status = profile_wizard.main(
                ["--out", self.path] + (extra or []), input_fn=input_fn)
        return status, out.getvalue(), err.getvalue()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "my_drivebase.json")

    def test_it_writes_a_profile_the_reader_roundtrips(self):
        status, out, err = self._main(
            ["my robot", "28", "96.5", "1.67", "1.05"])
        self.assertEqual(status, 0, err)
        self.assertTrue(os.path.exists(self.path))

        profile = read_drivebase_profile(self.path)
        self.assertEqual(profile["name"], "my robot")
        self.assertAlmostEqual(profile["wheel_radius_m"], 0.028)
        self.assertAlmostEqual(profile["track_width_m"], 0.0965)
        self.assertAlmostEqual(profile["gearing"], 1.67)
        self.assertAlmostEqual(profile["wall_distance_m"], 1.05)
        self.assertIn("python -m overboard.sysid.collect", out)

    def test_a_blank_wall_distance_leaves_the_wall_out(self):
        status, _out, err = self._main(["robot", "28", "96.5", "1", ""])
        self.assertEqual(status, 0, err)
        with open(self.path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertNotIn("wall", raw)
        self.assertIsNone(read_drivebase_profile(self.path)["wall_distance_m"])

    def test_an_implausible_answer_is_refused_with_the_range(self):
        # A wheel radius typed in metres (0.028) is the classic mistake; the
        # wizard must refuse it and say the range, then take the right one.
        status, out, err = self._main(
            ["robot", "0.028", "28", "96.5", "1", ""])
        self.assertEqual(status, 0, err)
        self.assertIn("between 5 and 100 mm", out)
        self.assertAlmostEqual(
            read_drivebase_profile(self.path)["wheel_radius_m"], 0.028)

    def test_numbers_that_are_not_numbers_are_refused(self):
        for text in ("", "wide", "nan", "inf", "-5", "0"):
            self.assertRaises(
                profile_wizard.AnswerError,
                profile_wizard.parse_wheel_radius_mm, text)
        self.assertRaises(
            profile_wizard.AnswerError, profile_wizard.parse_track_width_mm, "5")
        self.assertRaises(
            profile_wizard.AnswerError, profile_wizard.parse_gearing, "0")

    def test_it_refuses_to_overwrite_an_existing_profile(self):
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write("{}")
        status, _out, err = self._main(["robot", "28", "96.5", "1", ""])
        self.assertEqual(status, 1)
        self.assertIn("already exists", err)
        # --force is the way to overwrite.
        status, _out, err = self._main(
            ["robot", "28", "96.5", "1", ""], extra=["--force"])
        self.assertEqual(status, 0, err)
        self.assertAlmostEqual(
            read_drivebase_profile(self.path)["wheel_radius_m"], 0.028)


class TestExampleProfile(unittest.TestCase):
    """The shipped example must be a file the reader reads whole."""

    def test_the_example_parses_and_carries_every_field_the_reader_reads(self):
        profile = read_drivebase_profile(EXAMPLE_PATH)
        self.assertIsNotNone(profile["name"])
        self.assertIsNotNone(profile["wheel_radius_m"])
        self.assertIsNotNone(profile["track_width_m"])
        self.assertIsNotNone(profile["gearing"])
        self.assertIsNotNone(profile["wall_distance_m"])


class TestProfileReadErrors(unittest.TestCase):
    """A broken profile says what is wrong instead of tracebacking."""

    def _write(self, text):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "profile.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_broken_json_says_it_is_not_valid_json(self):
        path = self._write("{not json")
        with self.assertRaises(ValueError) as ctx:
            read_drivebase_profile(path)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_a_missing_field_names_the_field(self):
        path = self._write(json.dumps({"name": "x"}))
        with self.assertRaises(ValueError) as ctx:
            read_drivebase_profile(path)
        message = str(ctx.exception)
        self.assertIn("wheel_radius_m", message)
        self.assertIn("python -m overboard.sysid.profile", message)

    def test_a_field_that_is_not_a_number_names_the_field(self):
        path = self._write(json.dumps({"wheel_radius_m": "wide"}))
        with self.assertRaises(ValueError) as ctx:
            read_drivebase_profile(path)
        self.assertIn("wheel radius must be a number", str(ctx.exception))

    def test_a_field_that_is_not_positive_names_the_field(self):
        path = self._write(json.dumps({"track_width_m": -0.1}))
        with self.assertRaises(ValueError) as ctx:
            read_drivebase_profile(path)
        self.assertIn("track width must be positive", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
