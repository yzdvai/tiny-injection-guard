"""Styled QR code for the showcase: round dots, rounded finder eyes, blue-violet gradient, shield in the middle.
Writes an SVG (site + slides) and a 2048 px PNG (slides). Usage: make_qr.py URL OUT_DIR"""
import sys, os
import segno
from PIL import Image, ImageDraw

URL, OUT = sys.argv[1], sys.argv[2]
qr = segno.make(URL, error="h", boost_error=False)
M = [list(row) for row in qr.matrix]
n = len(M)
QUIET = 3                      # modules of white margin (inside the rounded white card)
LOGO = 0.24                    # logo box as a share of the code width (error level H tolerates ~30% damage)
C1, C2 = "#1d4ed8", "#6d28d9"  # gradient ends: both dark enough for strong contrast on white

def is_finder(r, c):
    return (r < 7 and c < 7) or (r < 7 and c >= n - 7) or (r >= n - 7 and c < 7)

lo = (n - n * LOGO) / 2
hi = n - lo
def under_logo(r, c):
    return lo - 0.5 <= r <= hi - 0.5 and lo - 0.5 <= c <= hi - 0.5

finders = [(0, 0), (0, n - 7), (n - 7, 0)]
size = n + 2 * QUIET

# ---------- SVG ----------
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" role="img" aria-label="QR code linking to {URL}">',
         '<defs><linearGradient id="qrg" x1="0" y1="0" x2="1" y2="1">'
         f'<stop offset="0" stop-color="{C1}"/><stop offset="1" stop-color="{C2}"/></linearGradient></defs>',
         f'<rect width="{size}" height="{size}" rx="{size * 0.08:.2f}" fill="#ffffff"/>',
         '<g fill="url(#qrg)">']
dots = []
for r in range(n):
    for c in range(n):
        if M[r][c] and not is_finder(r, c) and not under_logo(r, c):
            dots.append(f'<circle cx="{c + QUIET + 0.5}" cy="{r + QUIET + 0.5}" r="0.46"/>')
parts += dots
for fr, fc in finders:
    x, y = fc + QUIET, fr + QUIET
    # outer ring as an even-odd path, then the rounded centre square
    parts.append(f'<path fill-rule="evenodd" d="M{x + 2},{y} h3 a2,2 0 0 1 2,2 v3 a2,2 0 0 1 -2,2 h-3 a2,2 0 0 1 -2,-2 v-3 a2,2 0 0 1 2,-2 z '
                 f'M{x + 2},{y + 1} a1,1 0 0 0 -1,1 v3 a1,1 0 0 0 1,1 h3 a1,1 0 0 0 1,-1 v-3 a1,1 0 0 0 -1,-1 z"/>')
    parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="3" height="3" rx="0.9"/>')
parts.append("</g>")
# shield logo on a white rounded square
cx = size / 2
box = n * LOGO * 0.86
parts.append(f'<rect x="{cx - box / 2:.2f}" y="{cx - box / 2:.2f}" width="{box:.2f}" height="{box:.2f}" rx="{box * 0.28:.2f}" fill="#ffffff"/>')
s = box * 0.72 / 32
ox, oy = cx - 16 * s, cx - 16 * s
parts.append(f'<g transform="translate({ox:.3f},{oy:.3f}) scale({s:.4f})">'
             '<path d="M16 2l11 4v9c0 7-4.7 12.4-11 15C9.7 27.4 5 22 5 15V6z" fill="url(#qrg)"/>'
             '<path d="M11 16l3.5 3.5L21 13" stroke="#fff" stroke-width="2.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></g>')
parts.append("</svg>")
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "qr.svg"), "w").write("".join(parts))

# ---------- PNG (drawn at 2x, then downscaled for smooth edges) ----------
PX = 2048
big = PX * 2
u = big / size
mask = Image.new("L", (big, big), 0)
d = ImageDraw.Draw(mask)
for r in range(n):
    for c in range(n):
        if M[r][c] and not is_finder(r, c) and not under_logo(r, c):
            x0, y0 = (c + QUIET + 0.04) * u, (r + QUIET + 0.04) * u
            d.ellipse([x0, y0, x0 + 0.92 * u, y0 + 0.92 * u], fill=255)
for fr, fc in finders:
    x, y = (fc + QUIET) * u, (fr + QUIET) * u
    d.rounded_rectangle([x, y, x + 7 * u, y + 7 * u], radius=2 * u, fill=255)
    d.rounded_rectangle([x + u, y + u, x + 6 * u, y + 6 * u], radius=1 * u, fill=0)
    d.rounded_rectangle([x + 2 * u, y + 2 * u, x + 5 * u, y + 5 * u], radius=0.9 * u, fill=255)
# shield shape in the mask too (same gradient)
sc = box * 0.72 / 32 * u
sx, sy = cx * u - 16 * sc, cx * u - 16 * sc
shield = [(16, 2), (27, 6), (27, 15), (25.5, 21), (21.5, 26.5), (16, 30), (10.5, 26.5), (6.5, 21), (5, 15), (5, 6)]
d.polygon([(sx + px * sc, sy + py * sc) for px, py in shield], fill=255)

def hexrgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
a, b = hexrgb(C1), hexrgb(C2)
grad = Image.new("RGB", (big, big))
gp = grad.load()
row = [tuple(int(a[k] + (b[k] - a[k]) * t / (2 * big - 2)) for k in range(3)) for t in range(2 * big - 1)]
for yy in range(big):
    for xx in range(big):
        gp[xx, yy] = row[xx + yy]
card = Image.new("RGBA", (big, big), (0, 0, 0, 0))
ImageDraw.Draw(card).rounded_rectangle([0, 0, big - 1, big - 1], radius=int(size * 0.08 * u), fill=(255, 255, 255, 255))
card.paste(grad, (0, 0), mask)
# white logo box drawn under the shield: re-draw box then shield
ImageDraw.Draw(card).rounded_rectangle([(cx - box / 2) * u, (cx - box / 2) * u, (cx + box / 2) * u, (cx + box / 2) * u], radius=box * 0.28 * u, fill=(255, 255, 255, 255))
smask = Image.new("L", (big, big), 0)
ImageDraw.Draw(smask).polygon([(sx + px * sc, sy + py * sc) for px, py in shield], fill=255)
card.paste(grad, (0, 0), smask)
check = [(11, 16), (14.5, 19.5), (21, 13)]
ImageDraw.Draw(card).line([(sx + px * sc, sy + py * sc) for px, py in check], fill=(255, 255, 255, 255), width=int(2.8 * sc), joint="curve")
card.resize((PX, PX), Image.LANCZOS).save(os.path.join(OUT, "qr.png"))
print(f"version {qr.version}, {n}x{n} modules, error level {qr.error}, {len(dots)} dots")
