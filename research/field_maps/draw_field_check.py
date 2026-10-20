import io, json, math
from PIL import Image, ImageDraw

ROOT = r"C:\Users\jumpi\Documents\Github\Project-O.V.E.R.B.O.A.R.D"
mat = Image.open(ROOT + r"\mats\mat-8e87184106ad.png").convert("RGB")
d = json.load(io.open(ROOT + r"\fields\2026-27-robovibe.json", encoding="utf-8"))
MMX, MMY = 2000.0, 1140.0
px = mat.size[0] / MMX

def to_px(x_mm, y_mm):
    return (x_mm * px, (MMY - y_mm) * px)

def corners(o):
    w, h = o["w_m"] * 1000, o["h_m"] * 1000
    a = math.radians(o["rot_deg"])
    pts = []
    for dx, dy in ((-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)):
        wx = o["x_m"]*1000 + dx*math.cos(a) - dy*math.sin(a)
        wy = o["y_m"]*1000 + dx*math.sin(a) + dy*math.cos(a)
        pts.append(to_px(wx, wy))
    return pts

draw = ImageDraw.Draw(mat, "RGBA")
for o in d["obstacles"]:
    pts = corners(o)
    if o["kind"] == "circle":
        r = o["w_m"] * 1000 / 2 * px
        c = to_px(o["x_m"]*1000, o["y_m"]*1000)
        draw.ellipse([c[0]-r, c[1]-r, c[0]+r, c[1]+r], fill=(56,189,248,50), outline=(56,189,248,230), width=5)
    else:
        draw.polygon(pts, fill=(56,189,248,50), outline=(56,189,248,230))
        draw.line(pts + [pts[0]], fill=(56,189,248,230), width=5)
    c = to_px(o["x_m"]*1000, o["y_m"]*1000)
    draw.text((c[0]+8, c[1]-8), o["name"], fill=(14, 165, 233, 255))
mat.thumbnail((1600, 1600))
out = r"C:\Users\jumpi\AppData\Local\Temp\opencode\field_check.png"
mat.save(out)
print("wrote", out, mat.size)
