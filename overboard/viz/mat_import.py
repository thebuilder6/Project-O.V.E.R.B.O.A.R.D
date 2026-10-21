"""
Compose a mat picture from FIRST's own documents.

FIRST's mat artwork is copyrighted, so it is not in this repository and this
never puts it in one. The picture is *composed* on this machine from the
documents FIRST publishes: the field guide's wireframe page is rendered,
cropped to the mat and scaled to the mat's physical size, so the picture
maps exactly onto the mat the robot drives on.

This is the same idea as the reference simulator's ``tools/mat-import``: the
artwork is never redistributed, it is made from the official document by the
person who owns the mat.

Two renderers are supported:

* ``pypdfium2`` (a pip wheel, no system packages), and
* ``pdftoppm`` from poppler, which is what the reference simulator uses.

Usage:
    python mat_import.py wireframe.pdf --page 2 --mat-mm 2004x1143 --out mat.png
    python mat_import.py scan.jpg --mat-mm 2004x1143 --out mat.png
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

from PIL import Image

# The reference simulator's crop: the content below the page header and
# above the footer. It matches FIRST's "Wireframe & Path Diagram" layout.
HEADER_FRAC = 0.22
FOOTER_FRAC = 0.95
# A pixel is "content" when its brightness is below this.
DARK = 200


def autocrop(image: Image.Image, header_frac: float = HEADER_FRAC,
             footer_frac: float = FOOTER_FRAC) -> Optional[Tuple[int, int, int, int]]:
    """The bounding box of the drawing on the page, in pixels.

    The header and the footer are skipped, because a field guide page has
    text in both and the drawing is in between.

    Returns:
        The box as ``(x0, y0, x1, y1)``, or None when the page has no
        drawing in it.
    """
    width, height = image.size
    band = image.convert("L").crop(
        (0, int(height * header_frac), width, int(height * footer_frac)))
    # Anything dark enough is content; the mask's bounding box is the crop.
    mask = band.point(lambda value: 255 if value < DARK else 0)
    box = mask.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return (x0, y0 + int(height * header_frac), x1, y1 + int(height * header_frac))


def render_pdf_page(path: str, page: int, dpi: int = 300) -> Image.Image:
    """Render one page of a PDF as an image.

    Raises:
        RuntimeError: when no renderer is available.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        pdfium = None

    if pdfium is not None:
        document = pdfium.PdfDocument(path)
        try:
            bitmap = document[page - 1].render(scale=dpi / 72.0)
            return bitmap.to_pil().convert("RGB")
        finally:
            document.close()

    if shutil.which("pdftoppm"):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "page")
            subprocess.run(
                ["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page),
                 "-png", path, base],
                check=True, stderr=subprocess.DEVNULL)
            files = sorted(os.listdir(tmp))
            if not files:
                raise RuntimeError("the page did not render")
            with Image.open(os.path.join(tmp, files[0])) as handle:
                return handle.convert("RGB")

    raise RuntimeError(
        "no PDF renderer: pip install pypdfium2, or install poppler's "
        "pdftoppm")


def compose_mat(source: str, out_path: str, mat_mm: str = "2004x1143",
                page: int = 2, px_per_mm: float = 2.0,
                crop: Optional[str] = None) -> dict:
    """Make a mat picture that maps exactly onto the physical mat.

    Args:
        source: a PDF (a page of it is rendered) or an image.
        out_path: where to write the PNG.
        mat_mm: the mat's physical size, ``WIDTHxHEIGHT`` in millimetres.
        page: the PDF page to render.
        px_per_mm: the output's resolution.
        crop: an explicit crop box, ``x0,y0,x1,y1``, in the rendered
            pixels. None crops to the drawing.

    Returns:
        A dict with the crop, the source aspect, the mat aspect and the
        output's size.
    """
    width_mm, height_mm = (float(value) for value in mat_mm.lower().split("x"))
    if source.lower().endswith(".pdf"):
        image = render_pdf_page(source, page)
    else:
        with Image.open(source) as handle:
            image = handle.convert("RGB")

    if crop:
        box = tuple(int(value) for value in crop.split(","))
    else:
        found = autocrop(image)
        if found is None:
            raise RuntimeError("the page has no drawing on it")
        box = found

    x0, y0, x1, y1 = box
    cropped = image.crop((x0, y0, x1 + 1, y1 + 1))
    size = (round(width_mm * px_per_mm), round(height_mm * px_per_mm))
    cropped.resize(size, Image.LANCZOS).save(out_path)

    source_aspect = (x1 - x0 + 1) / (y1 - y0 + 1)
    mat_aspect = width_mm / height_mm
    aspect_error = abs(source_aspect - mat_aspect) / mat_aspect
    result = {
        "crop": list(box),
        "source_aspect": source_aspect,
        "mat_aspect": mat_aspect,
        "aspect_error": aspect_error,
        "size": list(size),
        "out": out_path,
    }
    if aspect_error > 0.01:
        # The picture is stretched to the mat's size, so a crop that is not
        # the mat's shape puts its features away from where they belong.
        # The caller decides what to do about it; the number is in the
        # result either way.
        print(
            "WARNING: the source aspect differs from the mat by more than 1% "
            f"({source_aspect:.4f} against {mat_aspect:.4f}); check the mat "
            "size or the crop.",
            file=sys.stderr)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compose a mat picture from FIRST's own documents.")
    parser.add_argument("source", help="a PDF page or an image")
    parser.add_argument("--page", type=int, default=2)
    parser.add_argument("--mat-mm", default="2004x1143")
    parser.add_argument("--px-per-mm", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop", help="x0,y0,x1,y1 in the rendered pixels")
    args = parser.parse_args(argv)

    try:
        result = compose_mat(
            args.source, args.out, mat_mm=args.mat_mm, page=args.page,
            px_per_mm=args.px_per_mm, crop=args.crop)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not make the mat picture: {exc}", file=sys.stderr)
        return 1
    print(
        "crop={crop} source aspect={source_aspect:.4f} mat aspect="
        "{mat_aspect:.4f} -> {out} {size}".format(**result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
