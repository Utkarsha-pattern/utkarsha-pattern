"""
Convert a portrait photo into a duotone ASCII-art SVG that "types" itself in
like a terminal, then holds.

Glyph density comes from the prepped luminance grid (prep_photo.py). Fill color
does not come from the photo — a low-chroma headshot only produces a gray wash.
Brightness is mapped onto a designed palette instead: warm cream and amber for
the face, teal for hair and shadows, deep slate for the sweater.

GitHub renders SVGs embedded via <img> and runs their SMIL animations there (JS
does not run). Each row is revealed with a left-to-right clip wipe plus a small
block cursor riding the wipe edge, staggered top -> bottom, so the whole
portrait prints once and freezes.
"""
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import html
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
# luminance only. portrait fills are a designed duotone, not sampled RGB.
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "source-prepped.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "utkarsha-ascii.svg")
BIO = "Utkarsha Gupta · AI & Product @Pattern"

# more columns = more detail (eyes need ~6+ chars across to read). the art
# stays ART_W px wide either way; cells shrink, keeping a ~1:1.875 char aspect.
COLS = int(os.environ.get("COLS", 180))
ART_W_TARGET = 800
CELL_W = ART_W_TARGET / COLS
CELL_H = CELL_W * 15 / 8
ROWS = round(COLS * 8 / 15)
RAMP = " .`:-=+*cs#%@"  # bright(sparse) -> dark(dense); leading space clears bg

# the prepped image already has bg removed + CLAHE local contrast, so only
# light global tuning is needed here.
CONTRAST = 1.15
BRIGHTNESS = 1.0
GAMMA = 1.0           # 1 keeps facial midtones; >1 washes a photo toward blank
SHARPEN = False
WHITE_FLOOR = 0.93    # luminance above this is forced to blank (space)

PAD = 20
TITLEBAR_H = 30
STATUS_H = 30
ART_W = COLS * CELL_W
ART_H = ROWS * CELL_H
CANVAS_W = ART_W + PAD * 2
CANVAS_H = TITLEBAR_H + ART_H + STATUS_H + PAD

BG = "#0d1117"
BG2 = "#111722"
FRAME = "#30363d"
TITLE_TEXT = "#7d8590"
INK = "#c9d1d9"      # chrome / status text, not the portrait fill
CURSOR = "#c9d1d9"
# Absolute luminance stops (0 = black, 1 = white), not a rank across pixels.
# The sweater is a large field of near-black; ranking it would smear amber
# through the clothing. These cuts keep that field slate/teal and the brighter
# face cream and peach. No gray stop sits between the two hues.
DUOTONE = (
    (0.04, "#2a3540"),  # sweater body — visible slate on #0d1117
    (0.08, "#3d6b6b"),  # deep teal, hair mass
    (0.13, "#5b8a8a"),  # hair edges
    (0.24, "#c9895a"),  # face — hue break, peach-amber
    (0.38, "#d4a574"),
    (0.52, "#e8b86d"),  # soft amber
    (1.01, "#f0e6d2"),  # warm cream highlights
)
# Color follows a small blur of luminance so eyes and beard stay inside the
# warm face instead of flipping the whole head to teal. Glyph shape is still
# the sharp sample above.
COLOR_BLUR = 3

# ---- reveal timing (one-shot; a cursor rasters top -> bottom) -------------
ROW_DUR = 5.8 / ROWS  # whole portrait prints in ~6s at any resolution
STAGGER = ROW_DUR       # == ROW_DUR -> a single cursor sweeping down

# ---- 1. sample the image into a COLS x ROWS grayscale grid ----------------
im = Image.open(SRC).convert("L")               # grayscale
if SHARPEN:
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))
im = ImageEnhance.Brightness(im).enhance(BRIGHTNESS)
im = ImageEnhance.Contrast(im).enhance(CONTRAST)
# Block-sample instead of a blurring resize. A plain average washes eyes and
# the mouth into the skin; biasing each cell toward its darkest pixel keeps
# those features at ASCII resolution.
src_arr = np.asarray(im, dtype=np.float32)
sh, sw = src_arr.shape
sampled = np.empty((ROWS, COLS), dtype=np.float32)
for y in range(ROWS):
    y0 = int(y * sh / ROWS)
    y1 = max(y0 + 1, int((y + 1) * sh / ROWS))
    for x in range(COLS):
        x0 = int(x * sw / COLS)
        x1 = max(x0 + 1, int((x + 1) * sw / COLS))
        cell = src_arr[y0:y1, x0:x1]
        sampled[y, x] = 0.55 * cell.mean() + 0.45 * cell.min()

def duotone_for(lum):
    """Designed palette from glyph luminance. lum is 0 (black) .. 1 (white)."""
    for limit, color in DUOTONE:
        if lum <= limit:
            return color
    return DUOTONE[-1][1]

def blur_ink(lum, radius):
    """Mean of nearby ink. Background (spaces) contributes 0 so it does not tint the subject."""
    base = np.where(lum < WHITE_FLOOR, lum, 0.0).astype(np.float32)
    r = radius
    padded = np.pad(base, r, mode="edge")
    acc = np.zeros_like(base)
    span = 2 * r + 1
    for dy in range(span):
        for dx in range(span):
            acc += padded[dy:dy + ROWS, dx:dx + COLS]
    return acc / (span * span)

color_lum = blur_ink(np.power(sampled / 255.0, GAMMA), COLOR_BLUR)

STATIC = bool(os.environ.get("STATIC"))  # emit frozen state for previews

rows_txt = []
rows_hex = []
for y in range(ROWS):
    chars = []
    hexes = []
    for x in range(COLS):
        lum = sampled[y, x] / 255.0
        lum = pow(float(lum), GAMMA)
        if lum >= WHITE_FLOOR:
            chars.append(" ")
            hexes.append(None)
            continue
        idx = int((1.0 - lum) * (len(RAMP) - 1) + 0.5)
        idx = max(0, min(len(RAMP) - 1, idx))
        chars.append(RAMP[idx])
        hexes.append(duotone_for(float(color_lum[y, x])))
    rows_txt.append("".join(chars))
    rows_hex.append(hexes)

def colored_runs(line, hexes):
    """Group a row into tspans. Spaces stay in the string so textLength holds."""
    parts = []
    i = 0
    n = len(line)
    while i < n:
        fill = hexes[i] or INK
        j = i + 1
        while j < n and (hexes[j] or INK) == fill:
            j += 1
        parts.append(f'<tspan fill="{fill}">{html.escape(line[i:j])}</tspan>')
        i = j
    return "".join(parts)

art_top = TITLEBAR_H + PAD * 0.35

# ---- 2. assemble SVG ------------------------------------------------------
parts = []
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" font-family="ui-monospace, SFMono-Regular, '
    f'Menlo, Consolas, monospace">'
)
parts.append('<defs>'
             f'<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="{BG2}"/><stop offset="1" stop-color="{BG}"/>'
             f'</linearGradient></defs>')

parts.append(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" rx="12" fill="url(#bg)"/>')
parts.append(f'<rect x="0.5" y="0.5" width="{CANVAS_W-1}" height="{CANVAS_H-1}" rx="12" '
             f'fill="none" stroke="{FRAME}" stroke-width="1"/>')

parts.append(f'<line x1="0" y1="{TITLEBAR_H}" x2="{CANVAS_W}" y2="{TITLEBAR_H}" stroke="{FRAME}"/>')
for i, dotcol in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
    parts.append(f'<circle cx="{PAD + i*16}" cy="{TITLEBAR_H/2}" r="5" fill="{dotcol}"/>')
parts.append(f'<text x="{CANVAS_W/2}" y="{TITLEBAR_H/2 + 4}" fill="{TITLE_TEXT}" font-size="12" '
             f'text-anchor="middle">utkarsha@github: ~$ ./portrait.sh</text>')

# one <text> per row; glyphs share a fill across short runs of the same muted color
font_size = CELL_H * 0.86
for ry, line in enumerate(rows_txt):
    y = art_top + ry * CELL_H + CELL_H * 0.74
    row_y = art_top + ry * CELL_H
    delay = ry * STAGGER
    body = colored_runs(line, rows_hex[ry])
    text = (f'<text xml:space="preserve" x="{PAD}" y="{y:.1f}" fill="{INK}" '
            f'font-size="{font_size:.1f}" textLength="{ART_W}" lengthAdjust="spacing">{body}</text>')

    if STATIC:
        parts.append(text)
        continue

    parts.append(
        f'<clipPath id="r{ry}"><rect x="{PAD}" y="{row_y:.1f}" height="{CELL_H}" width="0">'
        f'<animate attributeName="width" from="0" to="{ART_W}" begin="{delay:.3f}s" '
        f'dur="{ROW_DUR:.2f}s" fill="freeze"/></rect></clipPath>'
    )
    parts.append(f'<g clip-path="url(#r{ry})">{text}</g>')
    parts.append(
        f'<rect y="{row_y+1:.1f}" width="{CELL_W}" height="{CELL_H-2}" fill="{CURSOR}" opacity="0">'
        f'<animate attributeName="x" from="{PAD}" to="{PAD+ART_W}" begin="{delay:.3f}s" '
        f'dur="{ROW_DUR:.2f}s" fill="freeze"/>'
        f'<set attributeName="opacity" to="0.85" begin="{delay:.3f}s"/>'
        f'<set attributeName="opacity" to="0" begin="{delay+ROW_DUR:.3f}s"/></rect>'
    )

# status bar with a steady blinking cursor
status_line_y = TITLEBAR_H + ART_H + PAD * 0.35
status_y = status_line_y + 19
parts.append(f'<line x1="0" y1="{status_line_y:.1f}" x2="{CANVAS_W}" y2="{status_line_y:.1f}" stroke="{FRAME}"/>')
parts.append(f'<text x="{PAD}" y="{status_y:.1f}" fill="{TITLE_TEXT}" font-size="13">'
             f'utkarsha@github:~$ whoami <tspan fill="{INK}">{html.escape(BIO)}</tspan></text>')
status_chars = len(f"utkarsha@github:~$ whoami {BIO} ")  # cursor sits after the bio
parts.append(f'<rect x="{PAD + status_chars * 13 * 0.6:.1f}" y="{status_y-12:.1f}" width="8" height="14" fill="{INK}">'
             f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.5;0.51;1" '
             f'dur="1s" repeatCount="indefinite"/></rect>')

parts.append("</svg>")
svg = "".join(parts)
with open(OUT, "w") as f:
    f.write(svg)
print("wrote", OUT, len(svg), "bytes;", CANVAS_W, "x", CANVAS_H)
