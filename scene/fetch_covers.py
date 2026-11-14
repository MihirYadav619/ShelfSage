"""Download a real front cover for every book in the scene (Open Library covers).

Each generic title in layout.json is stood in for by a well-known textbook (BOOKS below). The
script searches Open Library by title + author, takes the first result with a cover and saves
the large cover image as scene/textures/covers/<book id>.jpg. Existing files are kept, so a
cover can be replaced by hand (any .jpg / .png with the book id as its name) and survives
re-runs. generate_scene.py turns them into the cover textures.

Usage (project root, venv active):
    python scene/fetch_covers.py            # download the missing covers
    python scene/fetch_covers.py --force    # download all again
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent / "textures" / "covers"
UA = {"User-Agent": "ShelfSage-covers/1.0 (course project scene builder)"}

# book id -> (real book title, author) used as its cover
BOOKS = {
    "linalg":       ("Introduction to Linear Algebra", "Gilbert Strang"),
    "calculus":     ("Calculus: Early Transcendentals", "James Stewart"),
    "ml":           ("Pattern Recognition and Machine Learning", "Christopher Bishop"),
    "dl":           ("Deep Learning", "Ian Goodfellow"),
    "robotics":     ("Introduction to Robotics: Mechanics and Control", "John J. Craig"),
    "control":      ("Modern Control Engineering", "Katsuhiko Ogata"),
    "signals":      ("Signals and Systems", "Alan V. Oppenheim"),
    "python":       ("Python Crash Course", "Eric Matthes"),
    "algo":         ("Introduction to Algorithms", "Thomas H. Cormen"),
    "db":           ("Database System Concepts", "Abraham Silberschatz"),
    "thermo":       ("Thermodynamics: An Engineering Approach", "Yunus A. Cengel"),
    "physics":      ("Fundamentals of Physics", "David Halliday"),
    "stats":        ("All of Statistics", "Larry Wasserman"),
    "os":           ("Operating System Concepts", "Abraham Silberschatz"),
    "networks":     ("Computer Networks", "Andrew S. Tanenbaum"),
    "compilers":    ("Compilers: Principles, Techniques, and Tools", "Alfred V. Aho"),
    "arch":         ("Computer Architecture: A Quantitative Approach", "John L. Hennessy"),
    "discrete":     ("Discrete Mathematics and Its Applications", "Kenneth H. Rosen"),
    "graphs":       ("Introduction to Graph Theory", "Douglas B. West"),
    "prob":         ("A First Course in Probability", "Sheldon Ross"),
    "optim":        ("Convex Optimization", "Stephen Boyd"),
    "numerics":     ("Numerical Recipes", "William H. Press"),
    "dsa":          ("Data Structures and Algorithm Analysis in C++", "Mark Allen Weiss"),
    "kinematics":   ("Modern Robotics: Mechanics, Planning, and Control", "Kevin M. Lynch"),
    "dynamics":     ("Engineering Mechanics: Dynamics", "J. L. Meriam"),
    "mechatronics": ("Mechatronics", "W. Bolton"),
    "embedded":     ("Introduction to Embedded Systems", "Edward A. Lee"),
    "digital":      ("Digital Design", "M. Morris Mano"),
    "power":        ("Power Electronics: Converters, Applications, and Design", "Ned Mohan"),
    "cv":           ("Computer Vision: Algorithms and Applications", "Richard Szeliski"),
    "nlp":          ("Speech and Language Processing", "Daniel Jurafsky"),
    "rl":           ("Reinforcement Learning: An Introduction", "Richard S. Sutton"),
    "ai":           ("Artificial Intelligence: A Modern Approach", "Stuart Russell"),
    "infotheory":   ("Elements of Information Theory", "Thomas M. Cover"),
    "crypto":       ("Cryptography and Network Security", "William Stallings"),
    "quantum":      ("Quantum Computation and Quantum Information", "Michael A. Nielsen"),
    "fluids":       ("Fluid Mechanics", "Frank M. White"),
    "heat":         ("Fundamentals of Heat and Mass Transfer", "Frank P. Incropera"),
    "materials":    ("Materials Science and Engineering: An Introduction", "William D. Callister"),
}

# Open Library cover ids picked by hand where the search's first hit is not the real cover
COVER_IDS = {"rl": 150672}      # the search found a plain title page of an edited volume


def get(url, timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def cover_ids(title, author):
    """Open Library cover ids of the best matches, most-edited work first."""
    q = urllib.parse.urlencode({"title": title.split(":")[0], "author": author.split()[-1],
                                "fields": "title,author_name,cover_i,edition_count", "limit": 10})
    docs = json.loads(get(f"https://openlibrary.org/search.json?{q}"))["docs"]
    docs = [d for d in docs if d.get("cover_i")]
    docs.sort(key=lambda d: -d.get("edition_count", 0))
    return [(d["cover_i"], d.get("title"), d.get("author_name")) for d in docs]


def main():
    sys.stdout.reconfigure(encoding="utf-8")         # titles with accents on a Windows console
    force = "--force" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    for bid, (title, author) in BOOKS.items():
        have = [p for p in OUT.glob(f"{bid}.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if have and not force:
            print(f"  keep  {bid:13s} {have[0].name}")
            continue
        try:
            ids = ([(COVER_IDS[bid], title, [author])] if bid in COVER_IDS else
                   cover_ids(title, author))
            if not ids:
                raise LookupError("no result with a cover")
            for cid, found, who in ids[:5]:           # first real image (some are 1x1 placeholders)
                img = get(f"https://covers.openlibrary.org/b/id/{cid}-L.jpg")
                w, h = Image.open(BytesIO(img)).size
                if h >= 300 and 0.5 < w / h < 0.9:    # a portrait front cover
                    break
            else:
                raise LookupError("no usable cover image")
            (OUT / f"{bid}.jpg").write_bytes(img)
            print(f"  ok    {bid:13s} {found} ({', '.join(who or [])[:40]}), {w}x{h}")
        except Exception as e:                        # network or lookup problem: report, go on
            missing.append(bid)
            print(f"  MISS  {bid:13s} {title}: {e}")
        time.sleep(0.5)                               # be polite to the API
    print(f"\n{len(BOOKS) - len(missing)}/{len(BOOKS)} covers in {OUT}")
    if missing:
        print("missing (put <id>.jpg in that folder by hand):", ", ".join(missing))


if __name__ == "__main__":
    main()
