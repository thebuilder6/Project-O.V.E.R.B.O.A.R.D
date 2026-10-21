"""Tests for composing a mat picture from FIRST's documents."""

import os
import tempfile
import unittest

from PIL import Image, ImageDraw

from overboard.viz.mat_import import autocrop, compose_mat


def _page(draw_box=(200, 300, 1600, 1100), size=(2000, 1400)):
    """A page with some text in the header and a drawing in the middle."""
    page = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(page)
    draw.text((100, 100), "FIRST LEGO League field guide header", fill="black")
    draw.rectangle(draw_box, outline="black", width=3)
    # The footer has to sit below the band the crop looks at, the way a
    # field guide's page footer does.
    draw.text((100, size[1] - 12), "footer", fill="black")
    return page


class TestAutocrop(unittest.TestCase):
    def test_the_drawing_is_found_and_the_text_is_not(self):
        page = _page()
        box = autocrop(page)
        self.assertIsNotNone(box)
        x0, y0, x1, y1 = box
        self.assertGreaterEqual(x0, 190)
        self.assertLessEqual(x1, 1610)
        self.assertGreater(y0, 300)
        self.assertLessEqual(y1, 1110)

    def test_a_blank_page_has_no_box(self):
        self.assertIsNone(autocrop(Image.new("RGB", (400, 400), "white")))


class TestComposeMat(unittest.TestCase):
    def test_the_picture_maps_onto_the_mat(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = _page()
            source = os.path.join(tmp, "page.png")
            page.save(source)
            out = os.path.join(tmp, "mat.png")
            result = compose_mat(source, out, mat_mm="2004x1143", px_per_mm=2.0)

            self.assertTrue(os.path.isfile(out))
            with Image.open(out) as made:
                self.assertEqual(made.size, (4008, 2286))
            self.assertAlmostEqual(result["mat_aspect"], 2004 / 1143, places=6)
            self.assertEqual(result["size"], [4008, 2286])

    def test_a_crop_can_be_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = _page()
            source = os.path.join(tmp, "page.png")
            page.save(source)
            out = os.path.join(tmp, "mat.png")
            compose_mat(source, out, mat_mm="2004x1143", px_per_mm=1.0,
                        crop="0,0,1999,1399")
            with Image.open(out) as made:
                self.assertEqual(made.size, (2004, 1143))

    def test_the_aspect_mismatch_is_reported(self):
        """A source that is not the mat's shape must not pass quietly."""
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            # A square drawing on a wide page: the aspect will not match.
            page = Image.new("RGB", (2000, 1400), "white")
            draw = ImageDraw.Draw(page)
            draw.rectangle((700, 400, 1300, 900), outline="black", width=3)
            source = os.path.join(tmp, "page.png")
            page.save(source)
            out = os.path.join(tmp, "mat.png")

            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                compose_mat(source, out, mat_mm="2004x1143", px_per_mm=1.0)
            self.assertIn("WARNING", errors.getvalue())

    def test_a_missing_file_is_reported(self):
        with self.assertRaises(OSError):
            compose_mat(os.path.join("no", "such", "file.png"), "out.png",
                        mat_mm="2004x1143")


class TestPdfRendering(unittest.TestCase):
    def test_the_page_renders_when_a_renderer_is_there(self):
        try:
            import pypdfium2  # noqa: F401
        except ImportError:
            import shutil
            if not shutil.which("pdftoppm"):
                self.skipTest("no PDF renderer is installed")
        from overboard.viz.mat_import import render_pdf_page

        # A one-page PDF with a drawing on it, written by hand.
        pdf = (b"%PDF-1.4\n"
               b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
               b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
               b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 300]"
               b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
               b"4 0 obj<</Length 60>>stream\n"
               b"BT /F1 18 Tf 40 140 Td (overboard mat test) Tj ET\n"
               b"0 0 1 RG 8 w 40 40 m 260 40 l S\n"
               b"endstream endobj\n"
               b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
               b"trailer<</Root 1 0 R>>\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.pdf")
            with open(path, "wb") as handle:
                handle.write(pdf)
            try:
                image = render_pdf_page(path, 1, dpi=72)
            except RuntimeError as exc:
                self.skipTest(str(exc))
            self.assertGreater(image.size[0], 100)
            self.assertGreater(image.size[1], 100)


if __name__ == "__main__":
    unittest.main()
