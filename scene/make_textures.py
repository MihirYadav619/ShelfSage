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


def make_plaster(name, base=(0.90, 0.88, 0.83), size=256):
    n = norm(gaussian_filter(rng.random((size, size)), 1.5, mode="wrap")) - 0.5
    save(name, np.array(base)[None, None, :] + 0.04 * n[..., None])


def main():
    make_parquet("floor_wood.png")
    make_wood("wood.png")
    make_wood("shelf_wood.png", light=(0.50, 0.34, 0.20), dark=(0.30, 0.19, 0.10))
    make_plaster("plaster.png")


if __name__ == "__main__":
    main()
