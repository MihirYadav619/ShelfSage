"""Library catalog as a PDF: every book with the shelf compartment it belongs to (its "home",
scene/layout.json). The robot knows this catalog; it does not know where loose books lie.

Usage (project root, venv active):
    python scene/make_catalog.py            # writes Book_Catalog.pdf in the project root

Page 1: books A-Z with their shelf ID. Page 2: the three shelves, each compartment with its books.
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "scene/library_meta.json").read_text())
OUT = ROOT / "Book_Catalog.pdf"
NAVY, INK, LIGHT = "#1c2638", "#222222", "#eef1f6"
ROWS = {"A": "bottom", "B": "top"}


def books():
    out = []
    for bid, b in META["books"].items():
        shelf, rc = b["home"].split("-")
        out.append({"id": bid, "title": b["title"], "home": b["home"], "shelf": shelf,
                    "row": rc[0], "col": int(rc[1:])})
    return sorted(out, key=lambda b: b["title"].lower())


def page_list(pdf, bs):
    fig = plt.figure(figsize=(8.27, 11.69))                       # A4 portrait
    fig.text(0.08, 0.95, "ShelfSage Library", fontsize=22, weight="bold", color=NAVY)
    fig.text(0.08, 0.925, "Book catalog: the shelf each book belongs to", fontsize=12, color=INK)
    fig.text(0.08, 0.895, "Shelf ID S2-A3 = shelf S2, row A (bottom; B is the top row), column 3 "
             "(counted from the left, facing the shelf).", fontsize=9, color="#555555")
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.82])
    ax.axis("off")
    rows = [[str(i + 1), b["title"], b["home"], b["shelf"], f"{b['row']} ({ROWS[b['row']]})", str(b["col"])]
            for i, b in enumerate(bs)]
    t = ax.table(cellText=rows, colLabels=["#", "Book", "Shelf ID", "Shelf", "Row", "Column"],
                 colWidths=[0.06, 0.44, 0.14, 0.1, 0.16, 0.1], loc="upper center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.32)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#c8ceda")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT)
        if c == 2 and r > 0:
            cell.get_text().set_weight("bold")
    fig.text(0.5, 0.025, f"{len(bs)} books - generated from scene/layout.json by scene/make_catalog.py",
             ha="center", fontsize=7, color="#888888")
    pdf.savefig(fig)
    plt.close(fig)


def page_map(pdf, bs):
    fig = plt.figure(figsize=(11.69, 8.27))                       # A4 landscape
    fig.text(0.05, 0.94, "Shelf map", fontsize=20, weight="bold", color=NAVY)
    fig.text(0.05, 0.905, "Seen from the room, facing the north wall. Each compartment lists the "
             "books that belong there.", fontsize=10, color=INK)
    by_comp = {}
    for b in bs:
        by_comp.setdefault(b["home"], []).append(b["title"])
    shelves = sorted(META["shelves"])
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.82])
    ax.set_xlim(0, 3 * 4.4)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for k, s in enumerate(shelves):
        x0 = k * 4.4 + 0.1
        ax.add_patch(Rectangle((x0 + 1.2, 2.75), 1.6, 0.32, color=NAVY))
        ax.text(x0 + 2.0, 2.91, f"SHELF {s}", color="white", ha="center", va="center", fontsize=13, weight="bold")
        for r, row in enumerate(["B", "A"]):                      # top row drawn first
            y = 1.4 - r * 1.3
            for c in range(1, 5):
                x = x0 + (c - 1) * 1.05
                comp = f"{s}-{row}{c}"
                ax.add_patch(Rectangle((x, y), 1.0, 1.25, facecolor="#f6efe3", edgecolor="#9c7a52", lw=1.2))
                ax.add_patch(Rectangle((x, y), 1.0, 0.2, color=NAVY))
                ax.text(x + 0.5, y + 0.1, comp, color="white", ha="center", va="center", fontsize=8, weight="bold")
                ty = y + 1.15
                for title in by_comp.get(comp, []):
                    lines = textwrap.wrap(title, 13, break_long_words=False)
                    ax.text(x + 0.06, ty, "\n".join(lines), fontsize=6.8, color=INK, va="top", linespacing=1.15)
                    ty -= 0.105 * len(lines) + 0.09
                if comp not in by_comp:
                    ax.text(x + 0.5, y + 0.72, "(empty)", fontsize=6.5, color="#aaaaaa", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    bs = books()
    with PdfPages(OUT) as pdf:
        page_list(pdf, bs)
        page_map(pdf, bs)
        d = pdf.infodict()
        d["Title"], d["Subject"] = "ShelfSage Library - Book catalog", "Which shelf each book belongs to"
    print(f"wrote {OUT.name}: {len(bs)} books")


if __name__ == "__main__":
    main()
