"""Make the pictures and GIFs shown in README.md (docs/images/).

    python docs\\make_readme_media.py

Needs the raw demo recordings in presentation/videos/ (not in git).
"""
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "presentation"))
import make_showcase as ms  # noqa: E402

OUT = os.path.join(ROOT, "docs", "images")

# name -> (recording, [(start, end)], speed)
GIFS = {
    "full_task": ("1_robotics_to_shelf", [(2.2, None)], 3.5),
    "camera_search": ("5_camera_finds_book", [(2.6, 8.5), (24.5, 29.0), (33.0, None)], 1.4),
    "navigation": ("4_navigation_obstacles", [(2.2, None)], 3.0),
    "drag_grasp": ("2_drag_then_grasp", [(7.0, None)], 3.5),
    "clear_lane": ("3_clear_lane_then_drag", [(7.0, None)], 4.5),
}
GIF_W, GIF_FPS = 560, 10


def gif(name, rec, parts, speed):
    path = os.path.join(OUT, name + ".gif")
    vf = (f"fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=96:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ms.W}x{ms.H}", "-r", str(ms.FPS), "-i", "-",
           "-filter_complex", vf, "-loop", "0", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for f in ms.clip(rec, parts, speed):
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    print(f"{name}.gif  {n / ms.FPS:.1f} s  {os.path.getsize(path) / 1e6:.1f} MB")


def save_jpg(img, name, width=1280):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if img.width != width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    path = os.path.join(OUT, name)
    img.convert("RGB").save(path, quality=88, optimize=True, progressive=True)
    print(f"{name}  {os.path.getsize(path) / 1e3:.0f} kB")


def main():
    os.makedirs(OUT, exist_ok=True)
    hero = Image.open(os.path.join(ROOT, "presentation", "title_image_wide.png"))
    save_jpg(hero.crop((0, 110, hero.width, hero.height)), "hero.jpg", 1280)
    save_jpg(ms.arch_frame_fn()(60.0), "architecture.jpg")
    save_jpg(ms.results_card(), "results.jpg")
    for name, (rec, parts, speed) in GIFS.items():
        gif(name, rec, parts, speed)


if __name__ == "__main__":
    main()
