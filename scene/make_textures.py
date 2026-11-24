"""Generate simple procedural textures for the library scene.

Usage (project root, venv active):
    python scene/make_textures.py

Writes PNG files into scene/textures/: the wall plaster (run once), plus the book spines, covers
and the robot zone label, which generate_scene.py makes when it runs. The floor and wood
textures are photo textures (floor_laminate.png, oak_veneer.png, Poly Haven, CC0).
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


PLASTER_BASE = (0.90, 0.88, 0.83)


def make_plaster(name, base=PLASTER_BASE, size=256):
    n = norm(gaussian_filter(rng.random((size, size)), 1.5, mode="wrap")) - 0.5
    save(name, np.array(base)[None, None, :] + 0.04 * n[..., None])


SPINE_FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "BebasNeue-Regular.ttf"
SPINE_PX_PER_M = 5000      # 5 px per mm


def make_spine(name, title, color, length=0.22, thickness=0.04):
    """Spine label: title in Bebas Neue (SIL OFL, assets/fonts/OFL-BebasNeue.txt) on the cover
    colour, with two thin bands near head and tail. Image x runs along the spine (book length),
    image y across it (book thickness), so the text reads along the spine."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = int(length * SPINE_PX_PER_M), int(thickness * SPINE_PX_PER_M)
    base = np.array(color) * 0.85
    lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    ink = (20, 20, 22) if lum > 0.55 else (245, 242, 232)
    img = Image.new("RGB", (W, H), tuple(int(c * 255) for c in base))
    d = ImageDraw.Draw(img)
    for x in (int(0.07 * W), int(0.93 * W)):             # decorative bands
        d.rectangle([x - 6, 0, x + 6, H], fill=ink)
    size = int(H * 0.62)
    font = ImageFont.truetype(str(SPINE_FONT), size)
    while d.textlength(title.upper(), font=font) > 0.74 * W and size > 20:
        size -= 4
        font = ImageFont.truetype(str(SPINE_FONT), size)
    d.text((W / 2, H / 2), title.upper(), font=font, fill=ink, anchor="mm")
    OUT.mkdir(exist_ok=True)
    img.save(OUT / name)


def make_cover(src, stem, length=0.22, width=0.15, rib_w=0.02, px_per_m=2500, orient=None):
    """Front cover textures from a real cover image (scene/textures/covers/<id>.jpg).

    The picture is scaled to the book's front (length x width, a mild stretch: real covers are
    ~4:3, the book 22:15) and cut into three parts, because the 2 cm ribs at head and tail sit on
    top of the cover: <stem>.png (whole cover, for the cover box between the ribs) and
    <stem>_head.png / <stem>_tail.png (the 2 cm strips, for the tops of the ribs). Seen from
    above, the parts make up the full cover. `orient` (name of a PIL transpose) is applied last so the
    picture lands on the box face the right way round (see generate_scene.COVER_ORIENT)."""
    from PIL import Image

    W, H = int(width * px_per_m), int(length * px_per_m)      # upright cover, head at the top
    img = Image.open(src).convert("RGB").resize((W, H), Image.LANCZOS)
    r = int(rib_w * px_per_m)
    parts = {"": img, "_head": img.crop((0, 0, W, r)), "_tail": img.crop((0, H - r, W, H))}
    OUT.mkdir(exist_ok=True)
    for suffix, im in parts.items():
        (im.transpose(getattr(Image.Transpose, orient)) if orient else im).save(OUT / f"{stem}{suffix}.png")


ZONE_RED = (0.78, 0.08, 0.08)


def make_floor_label(name, text, subtext, size=(0.90, 0.22)):
    """Floor marking plate: white text on safety red (Bebas Neue)."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = int(size[0] * 2000), int(size[1] * 2000)
    img = Image.new("RGB", (W, H), tuple(int(c * 255) for c in ZONE_RED))
    d = ImageDraw.Draw(img)
    white = (250, 248, 240)
    d.rectangle([12, 12, W - 13, H - 13], outline=white, width=10)
    big = ImageFont.truetype(str(SPINE_FONT), int(H * 0.50))
    small = ImageFont.truetype(str(SPINE_FONT), int(H * 0.30))
    d.text((W / 2, H * 0.36), text, font=big, fill=white, anchor="mm")
    d.text((W / 2, H * 0.76), subtext, font=small, fill=white, anchor="mm")
    OUT.mkdir(exist_ok=True)
    img.save(OUT / name)


SIGN_BG, SIGN_INK = (0.11, 0.15, 0.23), (0.97, 0.94, 0.86)     # navy, warm white


SIGN_BRASS, SIGN_ENGRAVED = (0.80, 0.66, 0.38), (0.13, 0.10, 0.07)   # compartment plates


def make_sign(name, text, size, subtext=None, px_per_m=1500, border=True, bg=SIGN_BG, ink=SIGN_INK):
    """Sign plate (table / shelf / compartment names): warm white Bebas Neue on navy, the text
    filling most of the plate (or other colours: bg, ink). size = (width, height) in metres."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = max(64, int(size[0] * px_per_m)), max(32, int(size[1] * px_per_m))
    img = Image.new("RGB", (W, H), tuple(int(c * 255) for c in bg))
    d = ImageDraw.Draw(img)
    ink = tuple(int(c * 255) for c in ink)
    if border:
        w = max(2, H // 30)
        d.rectangle([2 * w, 2 * w, W - 1 - 2 * w, H - 1 - 2 * w], outline=ink, width=w)
    main_h = 0.62 if subtext else 0.80
    size_px = int(H * main_h)
    font = ImageFont.truetype(str(SPINE_FONT), size_px)
    while d.textlength(text, font=font) > 0.86 * W and size_px > 8:
        size_px -= 2
        font = ImageFont.truetype(str(SPINE_FONT), size_px)
    d.text((W / 2, H * (0.42 if subtext else 0.53)), text, font=font, fill=ink, anchor="mm")
    if subtext:
        small = ImageFont.truetype(str(SPINE_FONT), int(H * 0.20))
        d.text((W / 2, H * 0.82), subtext, font=small, fill=ink, anchor="mm")
    OUT.mkdir(exist_ok=True)
    img.save(OUT / name)


def main():
    make_plaster("plaster.png")


if __name__ == "__main__":
    main()
