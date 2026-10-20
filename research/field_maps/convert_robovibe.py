"""
Turn RoboVibe's field YAML into OVERBOARD's field files.

RoboVibe (<https://robovibe.afarago.hu/>) publishes one YAML per season with
the field's geometry: the mission models' areas, the launch areas, and the
scoring rules. The YAML is embedded in the app's public JavaScript bundle
under the virtual path ``assets/config/<id>.yaml``; the copies in this
directory were read out of that bundle.

This script converts the *geometry* into OVERBOARD's own field format. The
scoring rules are not converted: they are RoboVibe's text, and OVERBOARD
does not score.

Two things have to be right for the geometry to land on the mat:

* **The frame.** RoboVibe's coordinates are in the *table's* frame (the
  2362 x 1143 mm table interior), with the mat centred in it. OVERBOARD's
  frame is the *mat's* (2004 x 1143 mm, the south-west corner at the
  origin), so the table's x is shifted by the mat's offset.
* **The anchor.** RoboVibe's ``x`` and ``y`` are the *top-left corner* of a
  shape's bounding box, not its centre. With that reading, RoboVibe's and
  martijas' independent measurements of the same printed wireframe agree to
  a few millimetres; with the other reading they disagree by tens.

Usage:
    python research/field_maps/convert_robovibe.py \
        research/field_maps/robovibe_yaml/fll2026founders.yaml \
        --out fields/2026-27-robovibe.json --mat-mm 2004x1143
"""

import argparse
import json
import math
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None

# The FLL table's interior, from the table guide: 93 in x 45 in.
FLL_TABLE_WIDTH_MM = 2362.0


def _to_metres(value):
    return float(value) / 1000.0


def convert(data, mat_mm="2004x1143", table_width_mm=FLL_TABLE_WIDTH_MM,
            mat_px_per_mm=2.0):
    """Convert RoboVibe's YAML into OVERBOARD's field description.

    Args:
        data: the parsed YAML.
        mat_mm: the mat's physical size, ``WIDTHxHEIGHT`` in millimetres.
        table_width_mm: the table's interior width, which is the frame
            RoboVibe's coordinates are in.
        mat_px_per_mm: the resolution of the mat picture.

    Returns:
        The field description, ready to be written as JSON.
    """
    mat_w_mm, mat_h_mm = (float(value) for value in mat_mm.lower().split("x"))
    # The mat is centred in the table, so the table's frame is the mat's
    # frame plus this much in x.
    offset_mm = (float(table_width_mm) - mat_w_mm) / 2.0

    season = str(data.get("id", "")).replace(".yaml", "")
    challenge = data.get("challenge", {}) or {}
    field_map = data.get("map", {}) or {}

    launch_areas = []
    for area in field_map.get("startareas", []) or []:
        launch_areas.append({
            "id": str(area.get("id", "launch area")),
            "x_m": _to_metres(float(area["x"]) - offset_mm),
            "y_m": _to_metres(float(area["y"])),
            "radius_m": _to_metres(area.get("radius", 0)),
            "quadrant": int(area.get("quadrant", 1) or 1),
        })

    obstacles = []
    for model_id, model in (field_map.get("models", {}) or {}).items():
        name = str(model.get("name", model_id))
        # A model either lists its shapes under "areas" or carries a single
        # shape on the model itself. Both are shapes on the mat.
        shapes = []
        if model.get("areas"):
            shapes = [(index, area)
                      for index, area in enumerate(model["areas"])]
        elif model.get("width") and model.get("length") and "x" in model:
            shapes = [(None, model)]
        for index, shape in shapes:
            area = shape
            width = float(area.get("width", area.get("length", 0)) or 0)
            length = float(area.get("length", area.get("width", 0)) or 0)
            if width <= 0.0 or length <= 0.0:
                continue
            # The anchor is the corner of the shape's box, and the shape
            # rotates clockwise about that corner. So the centre is the
            # corner plus the half box turned clockwise, and the shape's own
            # rotation is the other way round: OVERBOARD's format counts
            # counter-clockwise as positive.
            rotation = float(area.get("rotation", 0) or 0)
            angle = math.radians(rotation)
            centre_x = (float(area["x"]) - offset_mm
                        + (width / 2.0) * math.cos(angle)
                        + (length / 2.0) * math.sin(angle))
            centre_y = (float(area["y"])
                        - (width / 2.0) * math.sin(angle)
                        + (length / 2.0) * math.cos(angle))
            total = len(shapes)
            suffix = f" ({index + 1} of {total})" if total > 1 else ""
            obstacle_id = (f"{model_id.lower()}{index + 1}"
                           if index is not None else model_id.lower())
            obstacles.append({
                "id": obstacle_id,
                "name": name + suffix,
                "kind": "circle" if area.get("type") == "circle" else "rect",
                "x_m": _to_metres(centre_x),
                "y_m": _to_metres(centre_y),
                "w_m": _to_metres(width),
                "h_m": _to_metres(length),
                "rot_deg": -rotation,
            })

    rules = challenge.get("rules", "")
    return {
        "season": f"{season}-robovibe",
        "label": f"{challenge.get('season', season)} (RoboVibe's field)",
        "mat": {
            "width_m": mat_w_mm / 1000.0,
            "height_m": mat_h_mm / 1000.0,
        },
        "launch_areas": launch_areas,
        "placement_uncertainty_m": 0.02,
        "obstacles": obstacles,
        "source": {
            "provenance": (
                "Read from RoboVibe's field YAML for this season, which is "
                "embedded in the app's public bundle. RoboVibe measured it "
                "from FIRST's own field wireframe, the same document the "
                "footprints in the other field file came from."),
            "frame": (
                "RoboVibe's coordinates are in the table's frame with the mat "
                f"centred in it, so x is shifted by {offset_mm:.0f} mm. A "
                "shape's x and y are the top-left corner of its box, not its "
                "centre."),
            "checked_against": (
                "With that reading, the models that have a single area agree "
                "with the other field file to a few millimetres."),
            "rules": rules,
            "adjust_it": (
                "The operator can move a model on their own mat. A team that "
                "has measured their own field should trust their measurement "
                "over this file."),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a RoboVibe field YAML into OVERBOARD's format.")
    parser.add_argument("source", help="RoboVibe's YAML for a season")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mat-mm", default="2004x1143")
    parser.add_argument("--table-width-mm", type=float,
                        default=FLL_TABLE_WIDTH_MM)
    parser.add_argument("--mat-from",
                        help="a field file to take the mat's document from, "
                             "for the seasons that share a mat")
    args = parser.parse_args(argv)

    if yaml is None:
        print("PyYAML is needed: pip install pyyaml", file=sys.stderr)
        return 1
    with open(args.source, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    result = convert(data, mat_mm=args.mat_mm,
                     table_width_mm=args.table_width_mm)
    if args.mat_from:
        with open(args.mat_from, "r", encoding="utf-8") as handle:
            other = json.load(handle)
        for key in ("mat_source", "mat_image"):
            if other.get(key):
                result[key] = other[key]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(
        f"{args.out}: {len(result['obstacles'])} obstacles, "
        f"{len(result['launch_areas'])} launch areas, "
        f"season {result['season']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
