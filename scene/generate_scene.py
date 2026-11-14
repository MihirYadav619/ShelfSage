"""Generate the MuJoCo library scene from scene/layout.json.

Usage (from the project root, venv active):
    python scene/generate_scene.py             # writes scene/library.xml + library_meta.json
    python scene/generate_scene.py --check     # also simulates 10 s and checks books stay put
    python scene/generate_scene.py --preview   # also renders scene/preview_*.png

Geom groups (viewer: Rendering tab -> Group enable -> Geom 0..5):
    0  everything you normally see
    3  invisible collision proxies of the books (hidden by default)
    4  hidden walls (listed in layout.json "hidden_walls"); they still collide, only the
       rendering is skipped. Tick Geom group 4 in the viewer to see them again.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robot_model  # noqa: E402

HERE = Path(__file__).resolve().parent
LAYOUT_PATH = HERE / "layout.json"
XML_PATH = HERE / "library.xml"
META_PATH = HERE / "library_meta.json"
TEXTURE_DIR = HERE / "textures"

# Half-extents (metres)
# One book size for every book, lying or standing ("customised" library books).
# Book frame: x = length (head -> tail), y = width (spine at -y), z = thickness.
BOOK_L, BOOK_W, BOOK_T = 0.22, 0.15, 0.04
RIB_H, RIB_W = 0.01, 0.02          # bumper strips at head and tail on both covers: lying flat,
                                   # the book rests on them and leaves a 1 cm gap for a fingertip
BOOK_MASS, RIB_MASS = 0.3, 0.01
# Friction (Phase 5): MuJoCo takes the max of two geoms' friction unless one has a higher priority,
# then that geom's friction is used. Books: paper / cardboard on wood, 0.4 (priority 1, so it also
# holds against the 1.0 table and shelf). Finger pads (robot_model.py): rubber, 1.5, priority 2.
# Pressing on a book and dragging it then works: 1.5 * F > 0.4 * (F + m g) for any press force F.
BOOK_FRICTION, BOOK_PRIORITY = 0.4, 1
TABLE_OVERHANG = 0.03             # flat at a table edge: spine 3 cm past the edge. The gripper holds
                                   # the overhanging part; with the spine inside the edge the lower
                                   # finger (pad ~1 cm, convex hull 24 mm thick near the hand) hits the
                                   # table even in the 1 cm rib gap (Phase 4 test)
GRIP_IN = {"table": 0.015, "shelf": 0.02}   # grasp point: this far in from the spine
SHELF_INSET = 0.005                # standing in the shelf: spine 5 mm behind the shelf front (flush);
                                   # deeper and the 20 cm wide hand hits the dividers when grasping
QUAT_FLAT = (1, 0, 0, 0)
QUAT_STAND = (0.7071068, 0, 0.7071068, 0)   # length -> down (head on top), thickness -> +x

TABLE_TOP_Z = 0.75
TABLE_TOP_THICK = 0.03
BOARD_T = 0.02
EPS = 0.001
EDGE_X_OFFSETS = [-0.35, 0.0, 0.35]  # 3 edge spots per table (robot side), the default

PAGES = (0.95, 0.93, 0.85, 1)
WHITE = (0.95, 0.95, 0.93, 1)
BASEBOARD = (0.35, 0.24, 0.15, 1)
COLLISION_GROUP = 3
HIDDEN_WALL_GROUP = 4


def vec(v):
    return " ".join(f"{x:.4f}" for x in v)


def box(name, pos, half, rgba=None, material=None, collide=True, group=None, massless=False):
    a = []
    if name:
        a.append(f'name="{name}"')
    a += ['type="box"', f'pos="{vec(pos)}"', f'size="{vec(half)}"']
    if material:
        a.append(f'material="{material}"')
    if rgba is not None:
        a.append(f'rgba="{vec(rgba)}"')
    if not collide:
        a.append('contype="0" conaffinity="0"')
    if group is not None:
        a.append(f'group="{group}"')
    if massless:
        a.append('mass="0"')
    return "<geom " + " ".join(a) + "/>"


# ----------------------------------------------------------------------------- room
def room_frame(room):
    """Centre and half-size of the room floor (the room need not be centred on the origin)."""
    cx, cy = room.get("center", [0.0, 0.0])
    return cx, cy, room["size"][0] / 2, room["size"][1] / 2


def build_walls(room, hidden):
    cx, cy, hx, hy = room_frame(room)
    h, t = room["wall_height"], 0.05
    bh = 0.06  # baseboard half-height
    specs = {
        "north": ("wall_n", (0, hy + t, h / 2), (hx + t, t, h / 2),
                  (0, hy - 0.01, bh), (hx, 0.01, bh)),
        "south": ("wall_s", (0, -hy - t, h / 2), (hx + t, t, h / 2),
                  (0, -hy + 0.01, bh), (hx, 0.01, bh)),
        "east": ("wall_e", (hx + t, 0, h / 2), (t, hy + t, h / 2),
                 (hx - 0.01, 0, bh), (0.01, hy, bh)),
        "west": ("wall_w", (-hx - t, 0, h / 2), (t, hy + t, h / 2),
                 (-hx + 0.01, 0, bh), (0.01, hy, bh)),
    }
    shift = lambda p: (p[0] + cx, p[1] + cy, p[2])
    out = []
    for side, (name, pos, half, bpos, bhalf) in specs.items():
        g = HIDDEN_WALL_GROUP if side in hidden else None
        out.append(box(name, shift(pos), half, material="plaster_mat", group=g))
        out.append(box(f"baseboard_{side}", shift(bpos), bhalf, BASEBOARD, collide=False, group=g))
    return out


def build_sign(room, sign):
    """Graffiti 'ShelfSage' tag painted on the north wall: a textured, non-colliding plane
    1 mm in front of the wall, facing the room. Texture aspect is 3:1 (see make_textures.py)."""
    cx, cy, hx, hy = room_frame(room)
    w = sign["width"]
    pos = (sign["center_x"], cy + hy - 0.001, sign["center_z"])
    # rotate 90 deg about x: plane normal (+z) -> -y (into the room), plane +y -> world +z (up)
    return [f'<geom name="wall_sign" type="plane" pos="{vec(pos)}" quat="0.7071068 0.7071068 0 0" '
            f'size="{w / 2:.4f} {w / 6:.4f} 0.01" material="sign_mat" contype="0" conaffinity="0"/>']


def build_window(win):
    """Window on the east or west wall (thin along x): glass, frame, sill."""
    x, y, z = win["pos"]
    dx, dy, dz = win["half_size"]
    b = 0.05
    out = [box("window_glass", (x, y, z), (dx, dy, dz), (0.70, 0.85, 1.0, 0.25), collide=False)]
    for s in (-1, 1):
        out.append(box(None, (x, y, z + s * dz), (dx + 0.015, dy + b, b / 2), WHITE, collide=False))
        out.append(box(None, (x, y + s * dy, z), (dx + 0.015, b / 2, dz + b), WHITE, collide=False))
    out.append(box(None, (x, y, z), (dx + 0.01, 0.015, dz), WHITE, collide=False))
    out.append(box(None, (x, y, z), (dx + 0.01, dy, 0.015), WHITE, collide=False))
    out.append(box(None, (x - 0.05, y, z - dz - b), (0.08, dy + b + 0.03, 0.015), WHITE, collide=False))
    return out


# --------------------------------------------------------------------------- tables
def build_tables(layout):
    hx, hy = layout["table_half_size"]
    leg_h = TABLE_TOP_Z - TABLE_TOP_THICK
    out, meta = [], {}
    for name, t in layout["tables"].items():
        x, y = t["pos"]
        lines = [f'<body name="table_{name}" pos="{x:.4f} {y:.4f} 0">']
        lines.append("  " + box(None, (0, 0, TABLE_TOP_Z - TABLE_TOP_THICK / 2),
                                (hx, hy, TABLE_TOP_THICK / 2), material="wood_mat"))
        for sx in (-1, 1):
            for sy in (-1, 1):
                lines.append("  " + box(None, (sx * (hx - 0.06), sy * (hy - 0.06), leg_h / 2),
                                        (0.035, 0.035, leg_h / 2), material="wood_mat"))
        # apron under the top, makes the table look sturdier
        lines.append("  " + box(None, (0, 0, leg_h - 0.04), (hx - 0.06, hy - 0.06, 0.04),
                                material="wood_mat", collide=False))
        lines.append("</body>")
        out.append("\n".join(lines))
        flat_z = TABLE_TOP_Z + RIB_H + BOOK_T / 2 + EPS
        edge_y = y - hy + BOOK_W / 2 - TABLE_OVERHANG
        meta[name] = {"pos": [x, y], "top_z": TABLE_TOP_Z, "half_size": [hx, hy],
                      "edge_spots": [[x + dx, edge_y, flat_z] for dx in EDGE_X_OFFSETS]}
        # chairs on the reader side (north), pushed in a little; the robot side (south) stays free
        ch = layout.get("chairs")
        if ch:
            cy = y + hy + CHAIR_SEAT / 2 - ch["tuck"]
            chairs = [[x + dx, cy] for dx in ch["x_offsets"]]
            meta[name]["chairs"] = chairs
            out += [build_chair(f"chair_{name}{i + 1}", cx_, cy_) for i, (cx_, cy_) in enumerate(chairs)]
    return out, meta


CHAIR_SEAT, CHAIR_SEAT_Z, CHAIR_BACK_TOP = 0.42, 0.46, 0.90


def build_chair(name, x, y):
    """Static wooden chair facing -y (towards its table); backrest on the +y side."""
    s, h, t, leg = CHAIR_SEAT / 2, CHAIR_SEAT_Z, 0.03, 0.015
    m = "shelf_mat"
    lines = [f'<body name="{name}" pos="{x:.4f} {y:.4f} 0">',
             "  " + box(None, (0, 0, h - t / 2), (s, s, t / 2), material=m)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            top = CHAIR_BACK_TOP if sy > 0 else h - t       # rear legs carry the backrest
            lines.append("  " + box(None, (sx * (s - 0.03), sy * (s - 0.03), top / 2), (leg, leg, top / 2), material=m))
    lines.append("  " + box(None, (0, s - 0.03, CHAIR_BACK_TOP - 0.12), (s - 0.03, 0.012, 0.11), material=m))
    lines.append("  " + box(None, (0, s - 0.03, h + 0.10), (s - 0.03, 0.010, 0.025), material=m))
    lines.append("</body>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- shelf
def build_shelf(name, pos, s):
    """One bookcase. Compartments are named <shelf>-<row><col>, e.g. "S2-A3" (row A = bottom).

    Each compartment holds up to `books_per_compartment` standing books, `book_pitch` apart
    (ribs touch, leaving finger gaps between neighbours). Returns geoms and
    {compartment: {"surface": [x, y, z], "positions": [[x, y, z_centre], ...]}}.
    """
    cx, cy = pos
    W, D, cols = s["width"], s["depth"], s["cols"]
    surfaces, top_z = s["row_surface_z"], s["top_z"]
    n, pitch = s["books_per_compartment"], s["book_pitch"]
    t = BOARD_T
    total_h = top_z + t
    m = "shelf_mat"
    geoms = []
    for sx in (-1, 1):
        geoms.append(box(None, (cx + sx * (W / 2 + t / 2), cy, total_h / 2),
                         (t / 2, D / 2, total_h / 2), material=m))
    geoms.append(box(f"{name}_back", (cx, cy + D / 2 + 0.005, total_h / 2),
                     (W / 2 + t, 0.005, total_h / 2), material=m))
    geoms.append(box(None, (cx, cy, top_z + t / 2), (W / 2 + t, D / 2 + 0.01, t / 2), material=m))
    geoms.append(box(None, (cx, cy - D / 2 + 0.01, 0.06), (W / 2, 0.01, 0.06), material=m))   # plinth

    front = cy - D / 2
    book_y = front + SHELF_INSET + BOOK_W / 2
    comps = {}
    for i, z_s in enumerate(surfaces):
        row = "ABCDEFGH"[i]
        geoms.append(box(f"{name}_board_{row}", (cx, cy, z_s - t / 2), (W / 2, D / 2, t / 2), material=m))
        ceiling = (surfaces[i + 1] - t) if i + 1 < len(surfaces) else top_z
        row_h = ceiling - z_s
        for c in range(1, cols):
            x_div = cx - W / 2 + c * W / cols
            geoms.append(box(None, (x_div, cy, z_s + row_h / 2), (0.006, D / 2, row_h / 2), material=m))
        for c in range(1, cols + 1):
            x_c = cx - W / 2 + (c - 0.5) * W / cols
            offsets = [(k - (n - 1) / 2) * pitch for k in range(n)]
            comps[f"{name}-{row}{c}"] = {
                "surface": [x_c, cy, z_s],
                "positions": [[x_c + o, book_y, z_s + BOOK_L / 2 + EPS] for o in offsets]}
    return geoms, comps


# ---------------------------------------------------------------------------- books
def book_geoms(name, color):
    """Geoms of one book in the book frame (x length, y width with the spine at -y, z thickness).

    Collision: one box for the body + 4 rib boxes (the ribs touch the table / neighbours).
    Visual: covers, spine, page block, ribs, and a textured spine label (material spine_<id>_mat).
    """
    hl, hw, ht = BOOK_L / 2, BOOK_W / 2, BOOK_T / 2
    cover = tuple(color) + (1,)
    rib = tuple(c * 0.85 for c in color) + (1,)   # a shade darker than the cover
    book_contact = f'friction="{BOOK_FRICTION} 0.005 0.0001" priority="{BOOK_PRIORITY}"'
    g = [f'<geom name="{name}_geom" type="box" size="{vec((hl, hw, ht))}" mass="{BOOK_MASS}" '
         f'rgba="{vec(cover)}" group="{COLLISION_GROUP}" {book_contact}/>']
    for sx, tx in ((-1, "head"), (1, "tail")):
        for sz, tz in ((1, "top"), (-1, "bottom")):
            g.append(f'<geom name="{name}_rib_{tx}_{tz}" type="box" '
                     f'pos="{vec((sx * (hl - RIB_W / 2), 0, sz * (ht + RIB_H / 2)))}" '
                     f'size="{vec((RIB_W / 2, hw, RIB_H / 2))}" mass="{RIB_MASS}" rgba="{vec(rib)}" '
                     f'{book_contact}/>')
    c, st = 0.0015, 0.0025
    g += [box(f"{name}_cover_{t}", (0, 0, sz * (ht - c)), (hl, hw, c), cover, collide=False, massless=True)
          for sz, t in ((1, "top"), (-1, "bottom"))]
    g.append(box(f"{name}_spine", (0, -(hw - st), 0), (hl, st, ht), cover, collide=False, massless=True))
    g.append(box(f"{name}_pages", (0, 0.0015, 0), (hl - 0.003, hw - 0.0035, ht - 0.003), PAGES,
                 collide=False, massless=True))
    # spine label: 1 mm box just outside the spine (planes are only allowed on static bodies);
    # MuJoCo maps the 2D texture onto its -y face with texture x along the book length
    g.append(f'<geom name="{name}_label" type="box" pos="{vec((0, -hw - 0.0006, 0))}" '
             f'size="{vec((hl - 0.004, 0.0005, ht - 0.002))}" material="spine_{name[5:]}_mat" '
             f'contype="0" conaffinity="0" mass="0"/>')
    return g


def build_books(layout, tables_meta, comps):
    """Place the books. Table books lie flat with the spine towards the robot side (-y):
    at a free edge spot (default) or anywhere on the table with "pos": [dx, dy] (book centre
    relative to the table centre). Shelf books stand in a compartment ("shelf:S2-A1": next free
    position from the left), spine facing the room."""
    used_spots, used_pos, footprints = {}, {}, {}
    hl, hw = BOOK_L / 2, BOOK_W / 2
    xml, meta = [], {}
    for b in layout["books"]:
        kind, where = b["location"].split(":")
        if kind == "table":
            if where not in tables_meta:
                raise ValueError(f"Book {b['id']}: unknown table {where}")
            t = tables_meta[where]
            (tx, ty), (thx, thy) = t["pos"], t["half_size"]
            if "pos" in b:
                dx, dy = b["pos"]
                if abs(dx) > thx - hl or abs(dy) > thy - hw:
                    raise ValueError(f"Book {b['id']}: pos {b['pos']} is off table {where}")
                pos = (tx + dx, ty + dy, t["edge_spots"][0][2])
            else:
                taken = used_spots.setdefault(where, set())
                free = [i for i in range(len(EDGE_X_OFFSETS)) if i not in taken]
                if not free:
                    raise ValueError(f"Table {where}: all edge spots are taken")
                taken.add(free[0])
                pos = tuple(t["edge_spots"][free[0]])
            for ox, oy in footprints.get(where, []):
                if abs(pos[0] - ox) < 2 * hl + 0.02 and abs(pos[1] - oy) < 2 * hw + 0.02:
                    raise ValueError(f"Book {b['id']} overlaps another book on table {where}")
            footprints.setdefault(where, []).append((pos[0], pos[1]))
            quat = QUAT_FLAT
        elif kind == "shelf":
            if where not in comps:
                raise ValueError(f"Book {b['id']}: unknown shelf compartment {where}")
            k = used_pos.get(where, 0)
            if k >= len(comps[where]["positions"]):
                raise ValueError(f"Compartment {where} is full")
            used_pos[where] = k + 1
            pos = tuple(comps[where]["positions"][k])
            quat = QUAT_STAND
        else:
            raise ValueError(f"Book {b['id']}: bad location {b['location']}")
        name = f"book_{b['id']}"
        parts = [f'<body name="{name}" pos="{vec(pos)}" quat="{vec(quat)}">',
                 f'  <freejoint name="{name}_joint"/>']
        parts += ["  " + g for g in book_geoms(name, b["color"])]
        parts.append("</body>")
        xml.append("\n".join(parts))
        meta[b["id"]] = {"title": b["title"], "location": b["location"], "body": name,
                         "color": b["color"]}
    return xml, meta


def book_assets(layout):
    """Spine label textures + materials (regenerated every run, they are cheap)."""
    import make_textures
    out = []
    for b in layout["books"]:
        f = f"spine_{b['id']}.png"
        make_textures.make_spine(f, b["title"], b["color"], BOOK_L, BOOK_T)
        out.append(f'<texture name="spine_{b["id"]}_tex" type="2d" file="{f}"/>')
        out.append(f'<material name="spine_{b["id"]}_mat" texture="spine_{b["id"]}_tex" '
                   f'specular="0.2" shininess="0.3"/>')
    return out


# --------------------------------------------------------------------- robot zones
def parking_and_zones(layout, tables_meta, shelves_meta):
    """Fixed robot parking poses (where it stands to pick / place) and the floor zones around them.

    Tables: one pose per edge spot, facing the table (+y) from the robot side.
    Shelves: one pose per column of each shelf, facing the shelf.
    The robot stands `park_clearance` (0.50 m for tables and shelves) away from the edge. Closer
    and the pre-grasp / pre-place poses (10-15 cm out) fold the arm into the chassis or put a joint
    at its limit, so straight-line moves from there fail (Phase 4: works from 0.45 to 0.60 m). Each zone covers the chassis at every pose of that
    table / the shelf plus the gap to the furniture, where the arm works.
    """
    rc, zc = layout["robot"], layout["robot_zones"]
    hx, hy = rc["chassis_half"]
    clr, m = rc["park_clearance"]["table"], zc["margin"]
    yaw = 1.5707963
    zones = []
    for name, t in tables_meta.items():
        (tx, ty), (thx, thy) = t["pos"], t["half_size"]
        edge = ty - thy
        park_y = edge - clr - hx
        t["parking"] = [[sp[0], park_y, yaw] for sp in t["edge_spots"]]
        xs = [sp[0] for sp in t["edge_spots"]]
        zones.append({"name": f"table_{name}", "x": [min(min(xs) - hy, tx - thx) - m, max(max(xs) + hy, tx + thx) + m],
                      "y": [park_y - hx - m, edge]})
    clr = rc["park_clearance"]["shelf"]
    design = layout["shelf_design"]
    for sname, sh in shelves_meta.items():
        front = sh["pos"][1] - design["depth"] / 2
        park_y = front - clr - hx
        cols = sorted({k.split("-")[1][1:] for k in sh["compartments"]})
        sh["parking"] = {c: [sh["compartments"][f"{sname}-A{c}"]["surface"][0], park_y, yaw] for c in cols}
        xs = [p[0] for p in sh["parking"].values()]
        zones.append({"name": f"shelf_{sname}", "x": [min(xs) - hy - m, max(xs) + hy + m],
                      "y": [park_y - hx - m, front]})
    return zones


def build_zones(zones, zc):
    """Floor markings: translucent red fill, red tape border, KEEP CLEAR plate near the outer edge.
    Purely visual (no collisions, not in the LiDAR map)."""
    nc = 'contype="0" conaffinity="0"'
    tape = zc["tape"]
    lw, lh = zc["label_size"]
    red = "0.78 0.08 0.08"
    out = []
    for z in zones:
        (x0, x1), (y0, y1) = z["x"], z["y"]
        cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        n = z["name"]
        out.append(f'<geom name="zone_{n}_fill" type="box" pos="{vec((cx, cy, 0.0006))}" '
                   f'size="{vec((hx, hy, 0.0005))}" rgba="{red} 0.13" {nc}/>')
        for side, (px, py, sx, sy) in {"s": (cx, y0 + tape / 2, hx, tape / 2), "n": (cx, y1 - tape / 2, hx, tape / 2),
                                       "w": (x0 + tape / 2, cy, tape / 2, hy), "e": (x1 - tape / 2, cy, tape / 2, hy)}.items():
            out.append(f'<geom name="zone_{n}_tape_{side}" type="box" pos="{vec((px, py, 0.0012))}" '
                       f'size="{vec((sx, sy, 0.0005))}" rgba="{red} 1" {nc}/>')
        out.append(f'<geom name="zone_{n}_label" type="box" pos="{vec((cx, y0 + tape + 0.04 + lh / 2, 0.0014))}" '
                   f'size="{vec((lw / 2, lh / 2, 0.0005))}" material="zone_label_mat" {nc}/>')
    return out


# -------------------------------------------------------------------------- generate
def ensure_textures():
    needed = ["floor_wood.png", "wood.png", "shelf_wood.png", "plaster.png", "sign_shelfsage.png"]
    if all((TEXTURE_DIR / n).exists() for n in needed):
        return
    print("Textures missing, generating them first ...")
    import make_textures
    make_textures.main()


def generate():
    ensure_textures()
    layout = json.loads(LAYOUT_PATH.read_text())
    room = layout["room"]
    cx, cy, hx, hy = room_frame(room)
    hidden = set(layout.get("hidden_walls", []))

    table_xml, tables_meta = build_tables(layout)
    shelf_geoms, comps, shelves_meta = [], {}, {}
    for sname, sh in layout["shelves"].items():
        geoms, c = build_shelf(sname, sh["pos"], layout["shelf_design"])
        shelf_geoms += geoms
        comps.update(c)
        shelves_meta[sname] = {"pos": sh["pos"], "compartments": c}
    book_xml, books_meta = build_books(layout, tables_meta, comps)
    zones = parking_and_zones(layout, tables_meta, shelves_meta)
    zc = layout["robot_zones"]
    import make_textures
    make_textures.make_floor_label("zone_label.png", zc["text"], zc["subtext"], zc["label_size"])
    spine_assets = "\n    ".join(book_assets(layout))
    win = layout["window"]

    body = []
    body += build_walls(room, hidden)
    body += build_zones(zones, zc)
    body += build_window(win)
    if "sign" in layout:
        body += build_sign(room, layout["sign"])
    body += table_xml
    body += shelf_geoms
    body += book_xml
    robot_cfg, start = layout["robot"], layout["robot_start"]
    base_body, base_actuators = robot_model.base_xml(robot_cfg, start)
    body.append(base_body)
    body_text = "\n".join("    " + line.replace("\n", "\n    ") for line in body)

    lamps = "\n".join(
        f'    <light name="lamp{i}" pos="{x} {y} 2.4" dir="0 0 -1" diffuse="0.38 0.36 0.33" '
        f'specular="0.05 0.05 0.05" cutoff="90" exponent="1" castshadow="true"/>'
        for i, (x, y) in enumerate([(-2.2, -0.8), (2.2, -0.8), (0.0, 0.8), (0.0, 2.0),
                                    (-2.2, cy - hy + 1.5), (2.2, cy - hy + 1.5)]))
    # cutaway camera sits outside the (hidden) south wall, looking down into the room
    cut_y, cut_z = cy - hy - 4.5, 4.8 + 0.25 * (2 * hy - 6.0)

    xml = f"""<mujoco model="library">
  <compiler texturedir="textures" angle="radian"/>
  <!-- elliptic friction cone + impratio 10: far less creep in sticking contacts (a fingertip
       dragging a book, later the friction grasp) than the default pyramidal cone -->
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <!-- znear is relative to the scene extent (~16.6 m): 0.001 -> 1.7 cm, so the wrist camera
         sees books close up (the default 0.01 clipped everything nearer than 16.6 cm) -->
    <map znear="0.001"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight diffuse="0.30 0.30 0.30" ambient="0.30 0.30 0.30" specular="0 0 0"/>
    <rgba haze="0.75 0.82 0.9 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.62 0.78 0.95" rgb2="0.20 0.25 0.35" width="512" height="512"/>
    <texture name="floor_tex" type="2d" file="floor_wood.png"/>
    <texture name="wood_tex" type="2d" file="wood.png"/>
    <texture name="shelf_tex" type="2d" file="shelf_wood.png"/>
    <texture name="plaster_tex" type="2d" file="plaster.png"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="0.5 0.5" specular="0.35" shininess="0.35" reflectance="0.10"/>
    <material name="wood_mat" texture="wood_tex" texuniform="true" texrepeat="1 1" specular="0.25" shininess="0.30"/>
    <material name="shelf_mat" texture="shelf_tex" texuniform="true" texrepeat="1 1" specular="0.25" shininess="0.30"/>
    <material name="plaster_mat" texture="plaster_tex" texuniform="true" texrepeat="0.5 0.5" specular="0.05" shininess="0.05"/>
    <texture name="sign_tex" type="2d" file="sign_shelfsage.png"/>
    <material name="sign_mat" texture="sign_tex" specular="0.05" shininess="0.05"/>
    {spine_assets}
    <texture name="zone_label_tex" type="2d" file="zone_label.png"/>
    <material name="zone_label_mat" texture="zone_label_tex" specular="0.1" shininess="0.1"/>
  </asset>
  <worldbody>
{lamps}
    <camera name="overview" pos="0 -2.8 2.3" xyaxes="1 0 0 0 0.425 0.905" fovy="80"/>
    <camera name="cutaway" pos="{cx} {cut_y:.2f} {cut_z:.2f}" xyaxes="1 0 0 0 0.471 0.883" fovy="45"/>
    <geom name="floor" type="plane" pos="{cx} {cy} 0" size="{hx} {hy} 0.05" material="floor_mat"/>
{body_text}
  </worldbody>
  <actuator>
    {base_actuators.replace(chr(10), chr(10) + "    ")}
  </actuator>
</mujoco>
"""
    # attach the Panda (MjSpec), add the 'home' keyframe, then save as plain MJCF
    import mujoco
    spec = mujoco.MjSpec.from_string(xml)
    spec.modelfiledir = str(HERE) + "/"
    robot_model.add_robot(spec, robot_cfg, start)
    # "attach" grasp (README fallback): one inactive weld hand<->book per book; the robot sets the
    # relative pose and switches it on when the fingers have closed around the book
    for b in layout["books"]:
        e = spec.add_equality()
        e.name, e.type, e.objtype = f"grasp_{b['id']}", mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
        e.name1, e.name2, e.active = f"{robot_model.PREFIX}hand", f"book_{b['id']}", False
    robot_model.save_xml(spec, XML_PATH)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))   # make sure the saved file loads

    meta = {
        "room_size": room["size"],
        "room_center": [cx, cy],
        "robot_start": layout["robot_start"],
        "window_pos": win["pos"],
        "hidden_walls": sorted(hidden),
        "tables": tables_meta,
        "shelf_design": layout["shelf_design"],
        "shelves": shelves_meta,
        "robot_zones": zones,
        "books": books_meta,
        "book_dims": {"length": BOOK_L, "width": BOOK_W, "thickness": BOOK_T,
                      "rib_height": RIB_H, "rib_width": RIB_W, "mass": BOOK_MASS + 4 * RIB_MASS,
                      "table_overhang": TABLE_OVERHANG, "grip_in": GRIP_IN,
                      "frame": "x length, y width (spine at -y), z thickness"},
        "robot": robot_model.robot_meta(robot_cfg, start),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {XML_PATH.name} and {META_PATH.name}: "
          f"{len(tables_meta)} tables, {len(shelves_meta)} shelves / {len(comps)} compartments, {len(books_meta)} books, "
          f"hidden walls: {sorted(hidden) or 'none'}; robot: nq={model.nq}, nu={model.nu}")


def check(seconds=10.0, tol=0.01):
    """Simulate the scene and report how far each book moved."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    names = [model.body(i).name for i in range(model.nbody) if model.body(i).name.startswith("book_")]
    start = {n: data.body(n).xpos.copy() for n in names}
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    ok = True
    for n in names:
        d = float(np.linalg.norm(data.body(n).xpos - start[n]))
        ok &= d < tol
        print(f"  {'OK   ' if d < tol else 'MOVED'} {n:15s} moved {d * 100:.2f} cm")
    print("RESULT:", "all books stable" if ok else "some books moved more than 1 cm")
    return ok


def preview(width=1280, height=720):
    """Render preview_overview.png and preview_cutaway.png (hidden walls stay hidden)."""
    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height, width)
    for cam in ("overview", "cutaway"):
        renderer.update_scene(data, camera=cam)
        out = HERE / f"preview_{cam}.png"
        imageio.imwrite(out, renderer.render())
        print("wrote", out.name)


if __name__ == "__main__":
    generate()
    if "--check" in sys.argv:
        check()
    if "--preview" in sys.argv:
        preview()
