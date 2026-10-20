"""Tests for the field model and its collision check."""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fields import (
    Collision,
    FieldModel,
    Obstacle,
    field_from_dict,
    list_fields,
    load_all_fields,
    load_field,
)


def _pdf_bytes(mat_shaped=True):
    """A one-page PDF with a drawing on it, made with Pillow."""
    import io as _io

    from PIL import Image, ImageDraw

    page = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(page)
    draw.rectangle((80, 90, 520, 320), outline="black", width=4)
    # The field's document is read from its second page. Its drawing has
    # the mat's shape (2004 x 1143), because a crop that does not would be
    # stretched when it is fitted to the mat.
    second = Image.new("RGB", (600, 400), "white")
    bottom = 348 if mat_shaped else 300
    ImageDraw.Draw(second).rectangle((100, 120, 500, bottom), outline="black",
                                     width=4)
    buffer = _io.BytesIO()
    page.save(buffer, "PDF", save_all=True, append_images=[second])
    return buffer.getvalue()



def _rect(x=1.0, y=1.0, w=0.2, h=0.1, rot=0.0):
    return Obstacle(id="r", name="Rect", kind="rect", x_m=x, y_m=y,
                    w_m=w, h_m=h, rot_deg=rot)


def _circle(x=1.0, y=1.0, r=0.05):
    return Obstacle(id="c", name="Circle", kind="circle", x_m=x, y_m=y,
                    w_m=2 * r, h_m=2 * r)


class TestObstacleOverlap(unittest.TestCase):
    def test_a_point_inside_a_rectangle_is_inside_it(self):
        obstacle = _rect()
        self.assertTrue(obstacle.contains_point(1.0, 1.0))
        self.assertFalse(obstacle.contains_point(2.0, 1.0))

    def test_a_point_outside_a_rectangle_does_not_overlap(self):
        obstacle = _rect()
        self.assertEqual(obstacle.overlap(2.0, 1.0, 0.05), 0.0)

    def test_the_overlap_is_the_penetration_depth(self):
        """How far the robot's body would have to back off to just touch."""
        obstacle = _rect(w=0.2, h=0.2)
        # 10 mm from the centre of a 200 mm square: the edge is 90 mm away,
        # so a 20 mm disc reaches 110 mm past it.
        self.assertAlmostEqual(obstacle.overlap(1.01, 1.0, 0.02), 0.11)
        # Right on the edge: the disc is tangent, so it overlaps by its radius.
        self.assertAlmostEqual(obstacle.overlap(1.1, 1.0, 0.02), 0.02)

    def test_a_rotated_rectangle_is_rotated(self):
        square = _rect(w=0.2, h=0.2, rot=0.0)
        turned = _rect(w=0.2, h=0.2, rot=45.0)
        # The corner of the square is the corner of the turned one.
        corner = 0.1 * (2 ** 0.5)
        self.assertEqual(square.overlap(1.0 + corner, 1.0, 0.01), 0.0)
        self.assertAlmostEqual(turned.overlap(1.0 + corner, 1.0, 0.01), 0.01)

    def test_a_circle_overlaps_by_its_distance(self):
        obstacle = _circle(r=0.05)
        self.assertAlmostEqual(obstacle.overlap(1.05, 1.0, 0.02), 0.02)
        self.assertEqual(obstacle.overlap(1.2, 1.0, 0.02), 0.0)

    def test_a_bad_obstacle_is_rejected(self):
        with self.assertRaises(ValueError):
            Obstacle(id="x", name="X", kind="triangle", x_m=0.0, y_m=0.0,
                     w_m=0.1, h_m=0.1)
        with self.assertRaises(ValueError):
            Obstacle(id="x", name="X", kind="rect", x_m=0.0, y_m=0.0,
                     w_m=0.0, h_m=0.1)


class TestFieldCollisions(unittest.TestCase):
    def _field(self, uncertainty=0.05):
        return FieldModel(
            season="test",
            label="Test",
            mat_width_m=2.0,
            mat_height_m=1.14,
            placement_uncertainty_m=uncertainty,
            obstacles=(_rect(x=1.0, y=1.0, w=0.2, h=0.2),),
        )

    def test_a_path_that_misses_is_clean(self):
        field = self._field()
        samples = [{"x": 0.0, "y": 0.0, "t": 0.0}, {"x": 0.5, "y": 0.0, "t": 1.0}]
        self.assertEqual(field.collisions(samples, 0.05), [])

    def test_a_path_that_hits_is_reported(self):
        field = self._field()
        samples = [{"x": 1.0, "y": 1.0, "t": 0.5}]
        hits = field.collisions(samples, 0.05)
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].depth_m, 0.15)
        self.assertAlmostEqual(hits[0].t, 0.5)

    def test_a_shallow_hit_is_uncertain(self):
        """The footprint's own position is not known to better than 50 mm."""
        field = self._field(uncertainty=0.05)
        # The robot's body reaches 40 mm into the footprint, which is less
        # than the placement uncertainty, so it might not be a hit at all.
        samples = [{"x": 1.11, "y": 1.0, "t": 0.5}]
        hits = field.collisions(samples, 0.05)
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].uncertain)
        self.assertAlmostEqual(hits[0].depth_m, 0.04)

    def test_a_deep_hit_is_not_uncertain(self):
        field = self._field(uncertainty=0.05)
        samples = [{"x": 1.0, "y": 1.0, "t": 0.5}]
        hits = field.collisions(samples, 0.05)
        self.assertFalse(hits[0].uncertain)
        self.assertAlmostEqual(hits[0].depth_m, 0.15)

    def test_one_entry_per_obstacle(self):
        field = self._field()
        samples = [{"x": 1.0, "y": 1.0, "t": 0.1}, {"x": 1.0, "y": 1.0, "t": 0.2}]
        self.assertEqual(len(field.collisions(samples, 0.05)), 1)

    def test_a_sample_without_a_position_is_skipped(self):
        field = self._field()
        self.assertEqual(field.collisions([{"t": 0.0}], 0.05), [])
        self.assertEqual(field.collisions([{"x": "a", "y": 1.0}], 0.05), [])

    def test_a_model_between_two_samples_is_still_found(self):
        """The path between the samples is part of the path.

        The optimizer puts ten samples on a waypoint segment, so their
        spacing is set by the length of the segment: a two-metre straight
        is sampled every 200 mm. A model narrower than that can be driven
        straight through without any sample landing on it.
        """
        field = FieldModel(
            season="test",
            label="Test",
            mat_width_m=2.0,
            mat_height_m=1.14,
            placement_uncertainty_m=0.05,
            obstacles=(_rect(x=1.0, y=1.0, w=0.036, h=0.036),),
        )
        # The samples are 400 mm apart and neither is on the 36 mm model; the
        # path runs straight through it.
        samples = [{"x": 0.8, "y": 1.0, "t": 0.0}, {"x": 1.2, "y": 1.0, "t": 1.0}]
        hits = field.collisions(samples, 0.04825)
        self.assertEqual(len(hits), 1,
                         "the path between the samples was not checked")
        # The hit is between the two samples: neither of them is on the model.
        self.assertNotIn(hits[0].x_m, (0.8, 1.2))
        self.assertGreater(hits[0].depth_m, 0.0)

    def test_a_path_that_misses_by_more_than_the_robot_is_clean(self):
        """Walking between the samples must not invent a collision."""
        field = self._field()
        samples = [{"x": 0.8, "y": 0.0, "t": 0.0}, {"x": 1.2, "y": 0.0, "t": 1.0}]
        self.assertEqual(field.collisions(samples, 0.05), [])


class TestFieldFiles(unittest.TestCase):
    def test_the_shipped_field_loads(self):
        field = load_field("2026-27")
        self.assertIsNotNone(field)
        self.assertGreater(len(field.obstacles), 5)
        self.assertAlmostEqual(field.mat_width_m, 2.004)
        self.assertAlmostEqual(field.mat_height_m, 1.143)
        self.assertGreater(field.placement_uncertainty_m, 0.0)

    def test_every_obstacle_is_on_the_mat(self):
        field = load_field("2026-27")
        for obstacle in field.obstacles:
            self.assertGreaterEqual(obstacle.x_m, -0.1)
            self.assertLessEqual(obstacle.x_m, field.mat_width_m + 0.1)
            self.assertGreaterEqual(obstacle.y_m, -0.1)
            self.assertLessEqual(obstacle.y_m, field.mat_height_m + 0.1)

    def test_the_obstacle_ids_are_unique(self):
        field = load_field("2026-27")
        ids = [item.id for item in field.obstacles]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_two_models_share_a_footprint(self):
        """One entry per thing on the mat.

        Two missions can share one physical model - the "m06-07" build
        serves both - but the model is one thing on the mat, and a path
        that runs into it must be reported once, not twice.
        """
        for season in ("2026-27", "fll2026founders-robovibe"):
            field = load_field(season)
            if field is None:
                continue
            seen = {}
            for obstacle in field.obstacles:
                key = (round(obstacle.x_m, 3), round(obstacle.y_m, 3),
                       round(obstacle.w_m, 3), round(obstacle.h_m, 3),
                       round(obstacle.rot_deg, 1))
                self.assertNotIn(key, seen,
                                 f"{season}: {obstacle.id} and {seen.get(key)} "
                                 "are the same footprint")
                seen[key] = obstacle.id

    def test_the_seasons_are_listed(self):
        rows = list_fields()
        self.assertTrue(rows)
        self.assertIn("2026-27", [row["id"] for row in rows])

    def test_all_the_fields_load(self):
        models = load_all_fields()
        self.assertTrue(models)
        for model in models.values():
            self.assertTrue(model.obstacles)

    def test_an_unknown_season_is_nothing(self):
        self.assertIsNone(load_field("no-such-season"))
        self.assertIsNone(load_field(""))
        self.assertIsNone(load_field("../../etc/passwd"))

    def test_the_field_says_where_it_came_from(self):
        field = load_field("2026-27")
        self.assertIn("provenance", field.source)
        self.assertIn("firstinspires", field.source["provenance"])

    def test_a_field_round_trips_through_its_dict(self):
        field = load_field("2026-27")
        again = field_from_dict(field.as_dict())
        self.assertEqual(len(again.obstacles), len(field.obstacles))
        self.assertAlmostEqual(again.mat_width_m, field.mat_width_m)


class TestServerPayload(unittest.TestCase):
    def test_the_studio_gets_the_fields(self):
        from gui_server import field_payload
        payload = field_payload()
        self.assertTrue(payload["seasons"])
        self.assertIn("2026-27", payload["fields"])
        model = payload["fields"]["2026-27"]
        self.assertTrue(model["obstacles"])
        self.assertIn("mat", model)

    def test_the_collisions_are_computed_for_a_mission(self):
        from gui_server import field_collisions
        config = {"robot": {"track_width": 0.0965, "wheel_radius": 0.028}}
        # Across the middle of the mat, well clear of every model.
        clean = [{"x": 0.3, "y": 0.2, "t": 0.0}, {"x": 0.4, "y": 0.2, "t": 0.1}]
        self.assertEqual(field_collisions(clean, config, "2026-27"), [])
        # Through the "Flip the Rock" model at (0.084, 0.657).
        hit = [{"x": 0.084, "y": 0.657, "t": 0.0}]
        hits = field_collisions(hit, config, "2026-27")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "m03")
        self.assertFalse(hits[0]["uncertain"])

    def test_a_mission_without_a_field_has_no_collisions(self):
        from gui_server import field_collisions
        samples = [{"x": 0.084, "y": 0.657, "t": 0.0}]
        self.assertEqual(field_collisions(samples, {}, ""), [])
        self.assertEqual(field_collisions(samples, {}, "no-such-season"), [])


if __name__ == "__main__":
    unittest.main()


class TestLaunchAreas(unittest.TestCase):
    """The field's launch areas: the wall arcs in the corners."""

    def test_the_launch_areas_are_read(self):
        field = load_field("fll2026founders-robovibe")
        self.assertIsNotNone(field)
        self.assertEqual(len(field.launch_areas), 2)
        left, right = field.launch_areas
        self.assertAlmostEqual(left["x_m"], 0.002)
        self.assertAlmostEqual(left["y_m"], 0.0)
        self.assertAlmostEqual(left["radius_m"], 0.485)
        self.assertEqual(left["quadrant"], 1)
        self.assertAlmostEqual(right["x_m"], 2.002)
        self.assertEqual(right["quadrant"], 2)

    def test_the_launch_areas_reach_the_studio(self):
        from gui_server import field_payload
        payload = field_payload()
        model = payload["fields"]["fll2026founders-robovibe"]
        self.assertEqual(len(model["launch_areas"]), 2)

    def test_a_field_without_launch_areas_is_fine(self):
        field = load_field("2026-27")
        self.assertEqual(field.launch_areas, ())


class TestRoboVibeField(unittest.TestCase):
    """The converted RoboVibe field."""

    def test_the_models_land_where_the_wireframe_is(self):
        """RoboVibe's areas are the top-left corner of a box.

        With that reading and the mat's offset applied, the models that have
        a single area agree with the other field file to a few millimetres.
        """
        robovibe = load_field("fll2026founders-robovibe")
        ours = load_field("2026-27")
        pairs = (("m011", "m01"), ("m121", "m12"))
        for their_id, our_id in pairs:
            theirs = robovibe.obstacle(their_id)
            mine = ours.obstacle(our_id)
            self.assertIsNotNone(theirs)
            self.assertIsNotNone(mine)
            apart = math.hypot(theirs.x_m - mine.x_m, theirs.y_m - mine.y_m)
            self.assertLess(apart, 0.02)

    def test_the_turned_models_land_on_the_wireframe(self):
        """A turned shape rotates clockwise about its anchor corner.

        Reading the anchor as the shape's centre, or turning it the other
        way, puts the turned models tens of millimetres out.
        """
        robovibe = load_field("fll2026founders-robovibe")
        ours = load_field("2026-27")
        for their_id, our_id in (("m02", "m02"), ("m11", "m11")):
            theirs = robovibe.obstacle(their_id)
            mine = ours.obstacle(our_id)
            self.assertIsNotNone(theirs)
            self.assertIsNotNone(mine)
            apart = math.hypot(theirs.x_m - mine.x_m, theirs.y_m - mine.y_m)
            self.assertLess(apart, 0.02, f"{their_id} is {apart * 1000:.0f} mm out")

    def test_the_equipment_inspection_is_off_the_mat(self):
        """It is at the table's corners, not the mat's."""
        field = load_field("fll2026founders-robovibe")
        inspection = field.obstacle("ei1")
        self.assertIsNotNone(inspection)
        self.assertLess(inspection.x_m, 0.0)

    def test_the_collisions_use_this_field(self):
        from gui_server import field_collisions
        config = {"robot": {"track_width": 0.0965, "wheel_radius": 0.028}}
        # Through the "Forest Elder" tree at (1.940, 0.758).
        hits = field_collisions([{"x": 1.940, "y": 0.758, "t": 0.0}],
                                config, "fll2026founders-robovibe")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "m121")


class TestRoboVibeConverter(unittest.TestCase):
    """The YAML to field-file conversion."""

    def _yaml(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is not installed")
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "research", "field_maps", "robovibe_yaml", "fll2026founders.yaml")
        if not os.path.isfile(path):
            self.skipTest("the extracted YAML is not here")
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def test_the_conversion_moves_the_frame_and_finds_the_centre(self):
        from research.field_maps.convert_robovibe import convert
        result = convert(self._yaml(), mat_mm="2004x1143")

        # The launch areas are at the table's corners. The mat is 2004 mm
        # wide in a 2362 mm table, so it is inset 179 mm and the arcs land
        # 2 mm inside the mat's edges.
        self.assertAlmostEqual(result["launch_areas"][0]["x_m"], 0.002)
        self.assertAlmostEqual(result["launch_areas"][1]["x_m"], 2.002)
        # The drone survey's circle: RoboVibe's box starts at the table's
        # 1301 mm and is 64 mm wide, so its centre is at the mat's 1154 mm.
        drone = [o for o in result["obstacles"] if o["id"] == "m011"][0]
        self.assertAlmostEqual(drone["x_m"], 1.154, places=3)
        self.assertAlmostEqual(drone["y_m"], 0.057, places=3)
        self.assertEqual(drone["kind"], "circle")

    def test_the_models_with_a_single_shape_are_converted(self):
        """A model may carry one shape instead of a list of areas.

        Eleven of the seventeen models are shaped that way, and reading only
        the "areas" list drops all of them.
        """
        try:
            from research.field_maps.convert_robovibe import convert
        except ImportError:
            self.skipTest("the converter is not here")
        result = convert(self._yaml(), mat_mm="2004x1143")
        ids = {obstacle["id"] for obstacle in result["obstacles"]}
        for model in ("m02", "m03", "m05", "m06", "m07", "m08", "m09",
                      "m11", "m13", "m14", "m15"):
            self.assertIn(model, ids)
        # Every model in the field is there, and each of the ones with a
        # list of areas contributed all of them.
        self.assertEqual(len(result["obstacles"]), 24)

    def test_the_conversion_keeps_the_provenance(self):
        from research.field_maps.convert_robovibe import convert
        result = convert(self._yaml(), mat_mm="2004x1143")
        self.assertIn("RoboVibe", result["source"]["provenance"])
        self.assertIn("firstinspires", result["source"]["rules"])
        self.assertIn("top-left corner", result["source"]["frame"])

class TestMatDownload(unittest.TestCase):
    """The mat picture is composed from FIRST's own document."""

    def test_the_field_says_where_its_mat_is(self):
        field = load_field("2026-27")
        self.assertIn("url", field.mat_source)
        self.assertIn("firstinspires", field.mat_source["url"])
        self.assertIn("credit", field.mat_source)

    def test_the_picture_is_composed_from_firsts_document(self):
        """FIRST's field guide is fetched and the mat is composed from it."""
        import gui_server

        calls = []

        class _Response:
            def __init__(self, data):
                self._data = data

            def read(self, size=-1):
                # The real response reads a chunk at a time.
                if size is None or size < 0:
                    size = len(self._data)
                chunk, self._data = self._data[:size], self._data[size:]
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            calls.append(request.full_url)
            return _Response(_pdf_bytes())

        with tempfile.TemporaryDirectory() as tmp:
            mats = Path(tmp) / "mats"
            original = gui_server.MATS_DIR
            gui_server.MATS_DIR = mats
            try:
                with patch("urllib.request.urlopen", fake_urlopen):
                    result = gui_server.download_field_mat("2026-27")
            finally:
                gui_server.MATS_DIR = original

            self.assertTrue(calls, "nothing was fetched")
            self.assertIn("firstinspires", calls[0])
            self.assertTrue(result["url"].startswith("/mats/"))
            self.assertAlmostEqual(result["width_m"], 2.004)
            self.assertAlmostEqual(result["height_m"], 1.143)
            self.assertTrue(result["credit"])
            # The document and the composed picture are both kept out of
            # the repository.
            made = mats / Path(result["url"]).name
            self.assertTrue(made.is_file())
            with open(made, "rb") as handle:
                self.assertTrue(handle.read(4) == b"\x89PNG")

    def test_a_field_without_a_document_says_so(self):
        import gui_server
        with self.assertRaises(ValueError):
            gui_server.download_field_mat("no-such-season")

    def test_a_picture_that_is_not_the_mat_s_shape_is_refused(self):
        """A stretched picture puts the models away from where they are.

        The picture is fitted to the mat's size, so a crop that is not the
        mat's shape moves its features by a tenth of the mat. The studio
        says so instead of serving it.
        """
        import gui_server

        class _Response:
            def __init__(self, data):
                self._data = data

            def read(self, size=-1):
                if size is None or size < 0:
                    size = len(self._data)
                chunk, self._data = self._data[:size], self._data[size:]
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            return _Response(_pdf_bytes(mat_shaped=False))

        with tempfile.TemporaryDirectory() as tmp:
            mats = Path(tmp) / "mats"
            original = gui_server.MATS_DIR
            gui_server.MATS_DIR = mats
            try:
                with patch("urllib.request.urlopen", fake_urlopen):
                    with self.assertRaises(ValueError) as ctx:
                        gui_server.download_field_mat("2026-27")
            finally:
                gui_server.MATS_DIR = original
        self.assertIn("shape", str(ctx.exception))

    def test_something_that_is_not_a_pdf_is_refused(self):
        import gui_server

        class _Response:
            def __init__(self):
                self._data = b"<html>not a pdf</html>"

            def read(self, size=-1):
                if size is None or size < 0:
                    size = len(self._data)
                chunk, self._data = self._data[:size], self._data[size:]
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            mats = Path(tmp) / "mats"
            original = gui_server.MATS_DIR
            gui_server.MATS_DIR = mats
            try:
                with patch("urllib.request.urlopen", lambda *a, **k: _Response()):
                    with self.assertRaises(ValueError):
                        gui_server.download_field_mat("2026-27")
            finally:
                gui_server.MATS_DIR = original
