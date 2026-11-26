"""Build the ~2.5 min ShelfSage showcase video (silent, captions only).

    python presentation\\make_showcase.py      ->  presentation\\ShelfSage_Showcase.mp4

Title -> architecture -> 5 demo clips (important parts only, slightly sped up)
-> benchmark results -> closing. Needs opencv, pillow, imageio-ffmpeg.
"""
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = os.path.join(ROOT, "presentation", "videos")
PLOTS = os.path.join(ROOT, "benchmark", "results", "plots")
OUT = os.path.join(ROOT, "presentation", "ShelfSage_Showcase.mp4")
W, H, FPS = 1280, 720, 30
FADE = 0.35  # s, fade from / to black at every segment edge

BG = (11, 13, 23)
FG = (240, 242, 248)
DIM = (150, 156, 175)
ACCENT = (236, 104, 54)   # LLM orange, as in the plots
BLUE = (43, 124, 214)     # baseline blue
PANEL = (24, 28, 44)


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", name), size)


def text_c(d, y, s, f, fill=FG):
    w = d.textlength(s, font=f)
    d.text(((W - w) / 2, y), s, font=f, fill=fill)


def to_bgr(img):
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def blank():
    return Image.new("RGB", (W, H), BG)


def backdrop(dark=0.35, blur=6):
    img = Image.open(os.path.join(ROOT, "presentation", "title_image_wide.png")).convert("RGB")
    img = img.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(Image.new("RGB", (W, H), BG), img, dark)


# ---------------------------------------------------------------- segments
def still(frame_fn, dur):
    """frame_fn(t) -> BGR frame; yields dur seconds of frames."""
    n = int(round(dur * FPS))
    for k in range(n):
        yield frame_fn(k / FPS)


def static(img, dur):
    f = to_bgr(img)
    return still(lambda t: f, dur)


def clip(name, parts, speed):
    """parts: [(start_s, end_s or None)], played at `speed`."""
    cap = cv2.VideoCapture(os.path.join(VID, name + ".mp4"))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for start, end in parts:
        a = int(start * src_fps)
        b = total - 1 if end is None else min(total - 1, int(end * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        pos, frame = a, None
        x = float(a)
        while x <= b:
            want = int(round(x))
            while pos <= want:
                ok, f = cap.read()
                if not ok:
                    break
                frame, pos = f, pos + 1
            if frame is None:
                break
            if frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            yield frame
            x += speed * src_fps / FPS
    cap.release()


def intro_end(name):
    """Seconds where the clip's own dark title card ends."""
    cap = cv2.VideoCapture(os.path.join(VID, name + ".mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or f.mean() > 60:
            break
        i += 1
    cap.release()
    return i / fps + 0.1


def faded(gen):
    frames = list(gen)
    n, k = len(frames), int(FADE * FPS)
    for i, f in enumerate(frames):
        a = min(1.0, (i + 1) / k, (n - i) / k)
        yield f if a >= 1 else (f * a).astype(np.uint8)


# ---------------------------------------------------------------- cards
def title_card():
    img = backdrop(0.55, 3)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 230, W, 500], fill=(11, 13, 23, 200))
    text_c(d, 250, "ShelfSage", font(84, True))
    text_c(d, 360, "LLM-Guided Autonomous Mobile Librarian Robot", font(30), FG)
    text_c(d, 410, "MuJoCo  ·  Franka Panda on a mobile base  ·  Gemini task planner", font(22), DIM)
    text_c(d, 450, "Robot Manipulators course project, IIT Mandi  ·  Chirag & Mihir", font(20), DIM)
    return img


def problem_card():
    img = blank()
    d = ImageDraw.Draw(img)
    text_c(d, 120, "The problem", font(26, True), ACCENT)
    text_c(d, 175, "\u201cFind the Algorithms book and place it in the shelf.\u201d", font(40, True))
    lines = [
        "A robot in a library gets a plain-English instruction and must",
        "understand it, find the book, drive there, grasp it, place it, and check the result.",
        "",
        "Research question: does an LLM planner make the robot more flexible",
        "than a hand-written rule-based planner, on the very same robot stack?",
    ]
    for i, s in enumerate(lines):
        text_c(d, 290 + i * 42, s, font(26), FG if i < 2 else DIM)
    return img


def arch_frame_fn():
    boxes = [
        ("Instruction", "natural language", (60, 300)),
        ("LLM planner", "Gemini + world state JSON", (300, 300)),
        ("Validator", "rejects invalid plans", (540, 300)),
        ("Skills", "FIND · NAVIGATE · GRASP\nPLACE · VERIFY", (780, 300)),
        ("MuJoCo robot", "LiDAR · MCL · IK · YOLOv8", (1020, 300)),
    ]
    bw, bh = 200, 120
    f_t, f_s, f_h = font(24, True), font(16), font(22)
    cache = {}

    def draw(t):
        step = min(len(boxes) + 1, int(t / 1.1) + 1)
        if step in cache:
            return cache[step]
        img = blank()
        d = ImageDraw.Draw(img)
        text_c(d, 90, "How it works", font(26, True), ACCENT)
        text_c(d, 140, "The LLM plans.  Proven controllers execute.", font(40, True))
        for i, (title, sub, (x, y)) in enumerate(boxes[:step]):
            col = ACCENT if i == 1 else (90, 100, 130)
            d.rounded_rectangle([x, y, x + bw, y + bh], 14, fill=PANEL, outline=col, width=3)
            tw = d.textlength(title, font=f_t)
            d.text((x + (bw - tw) / 2, y + 18), title, font=f_t, fill=FG)
            for j, line in enumerate(sub.split("\n")):
                lw = d.textlength(line, font=f_s)
                d.text((x + (bw - lw) / 2, y + 60 + j * 22), line, font=f_s, fill=DIM)
            if i > 0:
                ax = x - 40
                d.line([ax + 2, y + bh / 2, x - 4, y + bh / 2], fill=DIM, width=3)
                d.polygon([(x - 4, y + bh / 2 - 7), (x + 4, y + bh / 2), (x - 4, y + bh / 2 + 7)], fill=DIM)
        if step > len(boxes):
            # replanning loop: robot -> LLM
            y0 = 300 + bh
            d.line([1120, y0 + 4, 1120, y0 + 60, 400, y0 + 60, 400, y0 + 12], fill=ACCENT, width=3, joint="curve")
            d.polygon([(393, y0 + 14), (400, y0 + 2), (407, y0 + 14)], fill=ACCENT)
            text_c(d, y0 + 70, "on failure: error + new state go back to the LLM, which replans", f_h, ACCENT)
            text_c(d, 600, "The LLM never touches the joints; the rule-based baseline uses the same skills.", font(20), DIM)
        cache[step] = to_bgr(img)
        return cache[step]

    return draw


def section_card(num, title, sub):
    img = blank()
    d = ImageDraw.Draw(img)
    text_c(d, 250, f"DEMO {num} / 5", font(22, True), ACCENT)
    text_c(d, 295, title, font(52, True))
    text_c(d, 385, sub, font(24), DIM)
    return img


def results_card():
    img = blank()
    d = ImageDraw.Draw(img)
    text_c(d, 70, "Results", font(26, True), ACCENT)
    text_c(d, 112, "42-task benchmark, fixed before the LLM was written, executed in the simulator", font(22), DIM)
    cards = [
        ("92 %", "LLM planner success", "126 runs (3 repeats)", ACCENT),
        ("40 %", "rule-based baseline", "same skills, same tasks", BLUE),
        ("4 / 5", "failure types recovered", "drop, slip, blocked path ...", ACCENT),
    ]
    cw, gap = 340, 40
    x0 = (W - (3 * cw + 2 * gap)) // 2
    for i, (big, l1, l2, col) in enumerate(cards):
        x = x0 + i * (cw + gap)
        d.rounded_rectangle([x, 200, x + cw, 470], 18, fill=PANEL, outline=col, width=3)
        f = font(96, True)
        bw = d.textlength(big, font=f)
        d.text((x + (cw - bw) / 2, 225), big, font=f, fill=col)
        for j, (s, ff, c) in enumerate([(l1, font(26, True), FG), (l2, font(19), DIM)]):
            sw = d.textlength(s, font=ff)
            d.text((x + (cw - sw) / 2, 375 + j * 40), s, font=ff, fill=c)
    text_c(d, 525, "McNemar test p = 5e-7   ·   book detector mask mAP50 = 0.98", font(24), FG)
    text_c(d, 570, "The baseline handles fixed templates; new phrasings, multi-step and spatial tasks need the LLM.", font(20), DIM)
    return img


def plot_card(png, caption):
    img = blank()
    d = ImageDraw.Draw(img)
    p = Image.open(os.path.join(PLOTS, png)).convert("RGB")
    s = min(1180 / p.width, 600 / p.height)
    p = p.resize((int(p.width * s), int(p.height * s)), Image.LANCZOS)
    x, y = (W - p.width) // 2, 30 + (600 - p.height) // 2
    d.rounded_rectangle([x - 12, y - 12, x + p.width + 12, y + p.height + 12], 14, fill=(252, 252, 250))
    img.paste(p, (x, y))
    text_c(d, 660, caption, font(22), FG)
    return img


def closing_card():
    img = backdrop(0.45, 5)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 200, W, 530], fill=(11, 13, 23, 210))
    text_c(d, 225, "ShelfSage", font(64, True))
    text_c(d, 315, "From one sentence to a book on the shelf:", font(28), FG)
    text_c(d, 355, "language planning on top of verifiable navigation, perception and manipulation.", font(24), DIM)
    text_c(d, 430, "Chirag & Mihir  ·  Robot Manipulators, IIT Mandi", font(24), FG)
    text_c(d, 470, "Simulated in MuJoCo  ·  python run_demo.py --task X01", font(18), DIM)
    return img


# ---------------------------------------------------------------- timeline
def timeline():
    yield static(title_card(), 6.0)
    yield static(problem_card(), 8.0)
    yield still(arch_frame_fn(), 9.5)

    n = "1_robotics_to_shelf"
    yield static(section_card(1, "The complete task", "\u201cPick up the Robotics book and place it in the shelf\u201d  ->  LLM plan  ->  executed"), 2.6)
    yield clip(n, [(intro_end(n), None)], 1.15)

    n = "5_camera_finds_book"
    yield static(section_card(2, "Searching with the wrist camera", "the robot is not told where the book is: YOLOv8-seg names every book it sees"), 2.6)
    yield clip(n, [(intro_end(n), 8.5), (14.0, 18.5), (24.5, 29.0), (33.0, None)], 1.1)

    n = "4_navigation_obstacles"
    yield static(section_card(3, "Navigation around obstacles", "A* on the known map  ·  MCL localization from LiDAR  ·  unseen obstacles avoided"), 2.6)
    yield clip(n, [(intro_end(n), None)], 1.25)

    n = "2_drag_then_grasp"
    yield static(section_card(4, "Drag to the edge, then grasp", "a book 40 cm inside the table: the fingertips press, the base pulls it to the edge"), 2.6)
    yield clip(n, [(6.5, None)], 1.4)

    n = "3_clear_lane_then_drag"
    yield static(section_card(5, "Clear the lane, drag, pick up", "another book blocks the way: push it aside first, then drag and grasp the target"), 2.6)
    yield clip(n, [(6.5, None)], 1.7)

    yield static(results_card(), 8.0)
    yield static(plot_card("success_by_category.png",
                           "LLM wins on multi-step, new phrasings and spatial relations; both solve the simple templates"), 7.0)
    yield static(plot_card("failure_recovery.png",
                           "Injected failures: the LLM replans from the error message and recovers"), 6.0)
    yield static(closing_card(), 6.0)


def main():
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    count = 0
    for seg in timeline():
        for f in faded(seg):
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
            count += 1
        print(f"  {count / FPS:6.1f} s")
    proc.stdin.close()
    proc.wait()
    print(f"wrote {OUT}: {count / FPS:.1f} s")


if __name__ == "__main__":
    main()
