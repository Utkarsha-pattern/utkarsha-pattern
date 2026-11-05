"""
Convert a portrait photo into a muted true-color ASCII-art SVG that "types"
itself in like a terminal, then holds.

Density still comes from the prepped luminance grid (prep_photo.py). Color is
sampled from the original photo, then desaturated and pinned into a soft
lightness band so the portrait stays readable on GitHub's dark theme — a
quiet palette, not a neon rainbow.

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
# luminance (density). color is read separately from the original photo.
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "source-prepped.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "utkarsha-ascii.svg")
COLOR_SRC = os.environ.get("COLOR_SRC", os.path.join(ROOT, "source-photo.png"))
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
# muted portrait ink: keep chroma low and lightness in a band that reads on #0d1117
CHROMA_KEEP = 0.42
SAT_CAP = 0.30
LIGHT_MIN = 0.56
LIGHT_MAX = 0.82
QUANT = 8            # channel step; merges neighboring glyphs into color runs

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

# ---- 1b. muted color from the original photo, aligned to the prepped crop --
def align_photo(photo_rgb, prep_h, prep_w):
    """Paste source-photo onto the prepped canvas using prep_photo's crop."""
    ph, pw = photo_rgb.shape[:2]
    try:
        from rembg import remove
        photo_u8 = np.clip(photo_rgb, 0, 255).astype(np.uint8)
        cut = remove(Image.fromarray(photo_u8).convert("RGBA"))
        alpha = np.array(cut.split()[-1])
        ys, xs = np.where(alpha > 20)
        if len(xs) == 0:
            raise RuntimeError("empty subject mask")
        side = max(int(xs.max() - xs.min()), int(ys.max() - ys.min())) + 40
        cx = int((xs.min() + xs.max()) // 2)
        cy = int((ys.min() + ys.max()) // 2)
        x0 = cx - side // 2
        y0 = cy - side // 2
        if side != prep_w or side != prep_h:
            # prepped file was cropped differently; still paste into its frame
            pass
    except Exception as exc:
        print(f"color align: centering photo ({exc})")
        x0 = (pw - prep_w) // 2
        y0 = (ph - prep_h) // 2
    canvas = np.full((prep_h, prep_w, 3), 18, np.float32)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + prep_w, pw), min(y0 + prep_h, ph)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = photo_rgb[sy0:sy1, sx0:sx1]
    return canvas

def cell_colors(rgb_canvas):
    h, w = rgb_canvas.shape[:2]
    grid = np.zeros((ROWS, COLS, 3), np.float32)
    for y in range(ROWS):
        y0 = int(y * h / ROWS)
        y1 = max(y0 + 1, int((y + 1) * h / ROWS))
        for x in range(COLS):
            x0 = int(x * w / COLS)
            x1 = max(x0 + 1, int((x + 1) * w / COLS))
            grid[y, x] = rgb_canvas[y0:y1, x0:x1].mean(axis=(0, 1))
    # soften speckle so neighboring glyphs share a color
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
    acc = np.zeros_like(grid)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy:dy + ROWS, dx:dx + COLS]
    return acc / 9.0

def hsv_of(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    df = mx - mn
    h = np.zeros_like(mx)
    mask = df > 1e-6
    rmask = mask & (mx == r)
    gmask = mask & (mx == g) & ~rmask
    bmask = mask & ~rmask & ~gmask
    h[rmask] = ((g[rmask] - b[rmask]) / df[rmask]) % 6
    h[gmask] = ((b[gmask] - r[gmask]) / df[gmask]) + 2
    h[bmask] = ((r[bmask] - g[bmask]) / df[bmask]) + 4
    h = (h / 6.0) % 1.0
    s = np.zeros_like(mx)
    s[mask] = df[mask] / np.maximum(mx[mask], 1e-6)
    return h, s, mx

def hsv_to_rgb(h, s, v):
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    out = np.zeros(h.shape + (3,), np.float32)
    for idx, (a, b, c) in enumerate(((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))):
        m = i == idx
        out[m, 0], out[m, 1], out[m, 2] = a[m], b[m], c[m]
    return out

def mute_palette(rgb_grid, ink_mask):
    """Desaturate photo colors and pin lightness so glyphs read on a dark README."""
    rgb = np.clip(rgb_grid / 255.0, 0, 1)
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # pull toward gray, then hard-cap saturation
    rgb = y[..., None] + (rgb - y[..., None]) * CHROMA_KEEP
    h, s, _v = hsv_of(np.clip(rgb, 0, 1))
    s = np.minimum(s, SAT_CAP)
    subject = y[ink_mask]
    if subject.size:
        lo, hi = np.percentile(subject, [8, 92])
    else:
        lo, hi = 0.0, 1.0
    span = max(float(hi - lo), 1e-3)
    t = np.clip((y - lo) / span, 0, 1)
    v = LIGHT_MIN + (LIGHT_MAX - LIGHT_MIN) * np.power(t, 0.9)
    muted = np.clip(hsv_to_rgb(h, s, v), 0, 1)
    q = np.clip(np.round(muted * 255 / QUANT) * QUANT, 0, 255).astype(np.uint8)
    return q

photo = np.asarray(Image.open(COLOR_SRC).convert("RGB"), dtype=np.float32)
aligned = align_photo(photo, sh, sw)
color_grid = mute_palette(cell_colors(aligned), sampled / 255.0 < WHITE_FLOOR)

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
        r, g, b = (int(c) for c in color_grid[y, x])
        hexes.append(f"#{r:02x}{g:02x}{b:02x}")
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
