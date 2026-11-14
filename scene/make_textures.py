"""Generate simple procedural textures for the library scene.

Usage (project root, venv active):
    python scene/make_textures.py

Writes PNG files into scene/textures/. Run once; generate_scene.py uses them.
Needs numpy, scipy and imageio (already in requirements).
"""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parent / "textures"
rng = np.random.default_rng(7)


def norm(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


def save(name, img):
    OUT.mkdir(exist_ok=True)
    imageio.imwrite(OUT / name, (np.clip(img, 0, 1) * 255).astype(np.uint8))
    print("wrote", OUT / name)


def grain(h, w, stretch=30):
    """Wood-like grain: noise smeared along the x direction (wraps, so it tiles)."""
    coarse = norm(gaussian_filter(rng.random((h, w)), (2.0, 2.0 * stretch), mode="wrap"))
    fine = norm(gaussian_filter(rng.random((h, w)), (0.6, 0.6 * stretch), mode="wrap"))
    return 0.65 * coarse + 0.35 * fine


def tint(g, light, dark):
    light, dark = np.array(light), np.array(dark)
    return light[None, None, :] * (1 - g[..., None]) + dark[None, None, :] * g[..., None]


def make_wood(name, light=(0.62, 0.43, 0.26), dark=(0.40, 0.25, 0.13), size=256):
    save(name, tint(grain(size, size), light, dark))


def make_parquet(name, size=512, plank_h=64, plank_len=256):
    """Wooden floor made of staggered planks with dark seams."""
    img = np.zeros((size, size, 3))
    rows = size // plank_h
    for r in range(rows):
        offset = 0 if r % 2 == 0 else plank_len // 2
        for k in range(size // plank_len + 1):
            x0 = k * plank_len - offset
            tile = tint(grain(plank_h, plank_len), (0.72, 0.55, 0.36), (0.50, 0.34, 0.20))
            tile *= rng.uniform(0.88, 1.08)           # each plank slightly different
            tile[:, :2] *= 0.55                        # seam at the plank ends
            tile[:2, :] *= 0.55                        # seam along the plank
            for dx in range(plank_len):                # place with wrap-around
                img[r * plank_h:(r + 1) * plank_h, (x0 + dx) % size] = tile[:, dx]
    save(name, img)


PLASTER_BASE = (0.90, 0.88, 0.83)


def make_plaster(name, base=PLASTER_BASE, size=256):
    n = norm(gaussian_filter(rng.random((size, size)), 1.5, mode="wrap")) - 0.5
    save(name, np.array(base)[None, None, :] + 0.04 * n[..., None])


FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "SedgwickAve-Regular.ttf"
SIGN_SIZE = (2400, 800)   # keep the same aspect as SIGN_HALF in generate_scene.py (3:1)


def make_sign(name, text="ShelfSage", ink=(24, 24, 28), tilt=-6, seed=11):
    """Graffiti-style wall tag with paint drips, painted onto plaster.

    MuJoCo ignores PNG transparency here, so the background is opaque plaster that matches
    plaster.png (same base colour and grain) and the sign blends into the wall.

    Font: Sedgwick Ave (SIL Open Font License, see assets/fonts/OFL.txt).
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    r = np.random.default_rng(seed)
    W, H = SIGN_SIZE
    font = ImageFont.truetype(str(FONT_PATH), 400)

    # 1. text on a big canvas, rotated a little like a hand-sprayed tag
    big = Image.new("L", (W * 2, H * 3), 0)
    d = ImageDraw.Draw(big)
    d.text((W, H * 1.5), text, font=font, fill=255, anchor="mm", stroke_width=2, stroke_fill=255)
    big = big.rotate(-tilt, resample=Image.BICUBIC, center=(W, H * 1.5))
    x0, y0, x1, y1 = big.getbbox()
    mask = big.crop((x0, y0, x1, y1))

    # 2. scale to fit the sign, leaving room underneath for drips
    scale = min((W * 0.94) / mask.width, (H * 0.70) / mask.height)
    mask = mask.resize((int(mask.width * scale), int(mask.height * scale)), Image.LANCZOS)
    canvas = Image.new("L", (W, H), 0)
    ox, oy = (W - mask.width) // 2, int(H * 0.06)
    canvas.paste(mask, (ox, oy))

    # 3. drips: start at bottom edges of strokes, taper as they run down, end in a small bead
    m = np.array(canvas) > 128
    d = ImageDraw.Draw(canvas)
    bottoms = [(x, int(np.nonzero(m[:, x])[0].max())) for x in range(W) if m[:, x].any()]
    picked = []
    while len(picked) < 9:   # spread drips out: at least 6 % of the width apart
        x, y = bottoms[r.integers(len(bottoms))]
        if all(abs(x - px) > 0.06 * W for px in picked):
            picked.append(x)
        else:
            continue
        length = int(r.uniform(0.10, 0.28) * H)
        w0 = r.uniform(10, 16)
        y_end = min(y + length, H - 14)
        d.polygon([(x - w0 / 2, y - 6), (x + w0 / 2, y - 6),
                   (x + w0 * 0.22, y_end), (x - w0 * 0.22, y_end)], fill=255)
        rb = w0 * 0.35
        d.ellipse([x - rb, y_end - rb, x + rb, y_end + rb], fill=255)

    # 4. soft spray edge, then paint over a plaster background
    #    (plaster.png grain is ~1.2 cm on the wall; the sign is ~1.75 mm per pixel -> sigma ~8 px)
    a = np.array(canvas.filter(ImageFilter.GaussianBlur(1.6)), dtype=float)[..., None] / 255
    n = norm(gaussian_filter(r.random((H, W)), 8.0)) - 0.5
    plaster = np.array(PLASTER_BASE)[None, None, :] + 0.04 * n[..., None]
    save(name, plaster * (1 - a) + (np.array(ink) / 255)[None, None, :] * a)


def main():
    make_parquet("floor_wood.png")
    make_wood("wood.png")
    make_wood("shelf_wood.png", light=(0.50, 0.34, 0.20), dark=(0.30, 0.19, 0.10))
    make_plaster("plaster.png")
    make_sign("sign_shelfsage.png")


if __name__ == "__main__":
    main()
