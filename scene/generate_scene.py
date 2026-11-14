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
BOOK_FLAT = (0.10, 0.07, 0.015)    # lying flat on a table (spine at the -x end)
BOOK_STAND = (0.025, 0.09, 0.12)   # standing in a shelf slot (spine faces -y, the room)
BOOK_MASS = 0.3

TABLE_TOP_Z = 0.75
TABLE_TOP_THICK = 0.03
BOARD_T = 0.02
EPS = 0.001
FLAT_X_OFFSETS = [-0.35, 0.0, 0.35]  # up to 3 books per table

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
        meta[name] = {"pos": [x, y], "top_z": TABLE_TOP_Z, "half_size": [hx, hy]}
    return out, meta


# ---------------------------------------------------------------------------- shelf
def build_shelf(s):
    cx, cy = s["pos"]
    W, D, cols = s["width"], s["depth"], s["cols"]
    surfaces, top_z = s["row_surface_z"], s["top_z"]
    t = BOARD_T
    total_h = top_z + t
    m = "shelf_mat"
    geoms = []

    for sx in (-1, 1):
        geoms.append(box(None, (cx + sx * (W / 2 + t / 2), cy, total_h / 2),
                         (t / 2, D / 2, total_h / 2), material=m))
    geoms.append(box("shelf_back", (cx, cy + D / 2 + 0.005, total_h / 2),
                     (W / 2 + t, 0.005, total_h / 2), material=m))
    geoms.append(box(None, (cx, cy, top_z + t / 2), (W / 2 + t, D / 2 + 0.01, t / 2), material=m))
    # plinth at the bottom
    geoms.append(box(None, (cx, cy - D / 2 + 0.01, 0.06), (W / 2, 0.01, 0.06), material=m))

    slots = {}
    for i, z_s in enumerate(surfaces):
        row = "ABCDEFGH"[i]
        geoms.append(box(f"shelf_board_{row}", (cx, cy, z_s - t / 2), (W / 2, D / 2, t / 2), material=m))
        ceiling = (surfaces[i + 1] - t) if i + 1 < len(surfaces) else top_z
        row_h = ceiling - z_s
        for c in range(1, cols):
            x_div = cx - W / 2 + c * W / cols
            geoms.append(box(None, (x_div, cy, z_s + row_h / 2), (0.006, D / 2, row_h / 2), material=m))
        for c in range(1, cols + 1):
            x_slot = cx - W / 2 + (c - 0.5) * W / cols
            slots[f"{row}{c}"] = [x_slot, cy, z_s]
    return geoms, slots


# ---------------------------------------------------------------------------- books
def book_visuals(name, half, color, a_t, a_s, a_p, sign):
    """Non-colliding visual parts: two covers, a spine, and a block of pages.

    a_t = thickness axis, a_s = spine axis, a_p = the remaining axis, sign = which end is the spine.
    """
    h = list(half)
    c, st = 0.003, 0.005
    cover = tuple(color) + (1,)

    def part(tag, center, hh, rgba):
        pos, size = [0.0] * 3, [0.0] * 3
        for ax, v in center.items():
            pos[ax] = v
        for ax, v in hh.items():
            size[ax] = v
        return "  " + box(f"{name}_{tag}", pos, size, rgba, collide=False, massless=True)

    out = []
    for s, tag in ((-1, "cover_n"), (1, "cover_p")):
        out.append(part(tag, {a_t: s * (h[a_t] - c / 2)}, {a_t: c / 2, a_s: h[a_s], a_p: h[a_p]}, cover))
    out.append(part("spine", {a_s: sign * (h[a_s] - st / 2)},
                    {a_s: st / 2, a_t: h[a_t], a_p: h[a_p]}, cover))
    out.append(part("pages", {a_s: sign * (0.003 - st) / 2},
                    {a_s: (2 * h[a_s] - st - 0.003) / 2, a_t: h[a_t] - c, a_p: h[a_p] - 0.002}, PAGES))
    return out


def build_books(layout, tables_meta, slots):
    used_on_table, used_slots = {}, set()
    xml, meta = [], {}
    for b in layout["books"]:
        kind, where = b["location"].split(":")
        if kind == "table":
            if where not in tables_meta:
                raise ValueError(f"Book {b['id']}: unknown table {where}")
            idx = used_on_table.get(where, 0)
            if idx >= len(FLAT_X_OFFSETS):
                raise ValueError(f"Table {where} holds at most {len(FLAT_X_OFFSETS)} books")
            used_on_table[where] = idx + 1
            tx, ty = tables_meta[where]["pos"]
            half = BOOK_FLAT
            pos = (tx + FLAT_X_OFFSETS[idx], ty, TABLE_TOP_Z + half[2] + EPS)
            axes = (2, 0, 1, -1)       # thickness z, spine along x at the -x end
        elif kind == "shelf":
            if where not in slots:
                raise ValueError(f"Book {b['id']}: unknown shelf slot {where}")
            if where in used_slots:
                raise ValueError(f"Shelf slot {where} is used twice")
            used_slots.add(where)
            sx, sy, sz = slots[where]
            half = BOOK_STAND
            pos = (sx, sy, sz + half[2] + EPS)
            axes = (0, 1, 2, -1)       # thickness x, spine at the -y end (faces the room)
        else:
            raise ValueError(f"Book {b['id']}: bad location {b['location']}")
        name = f"book_{b['id']}"
        parts = [
            f'<body name="{name}" pos="{vec(pos)}">',
            f'  <freejoint name="{name}_joint"/>',
            # invisible collision proxy: one simple box keeps contacts stable and fast
            f'  <geom name="{name}_geom" type="box" size="{vec(half)}" mass="{BOOK_MASS}" '
            f'rgba="{vec(tuple(b["color"]) + (1,))}" group="{COLLISION_GROUP}"/>',
        ]
        parts += book_visuals(name, half, b["color"], *axes)
        parts.append("</body>")
        xml.append("\n".join(parts))
        meta[b["id"]] = {"title": b["title"], "location": b["location"], "body": name}
    return xml, meta


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
    shelf_geoms, slots = build_shelf(layout["shelf"])
    book_xml, books_meta = build_books(layout, tables_meta, slots)
    win = layout["window"]

    body = []
    body += build_walls(room, hidden)
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
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
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
    robot_model.save_xml(spec, XML_PATH)
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))   # make sure the saved file loads

    meta = {
        "room_size": room["size"],
        "room_center": [cx, cy],
        "robot_start": layout["robot_start"],
        "window_pos": win["pos"],
        "hidden_walls": sorted(hidden),
        "tables": tables_meta,
        "shelf": {"pos": layout["shelf"]["pos"], "slots": slots},
        "books": books_meta,
        "robot": robot_model.robot_meta(robot_cfg, start),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {XML_PATH.name} and {META_PATH.name}: "
          f"{len(tables_meta)} tables, {len(slots)} shelf slots, {len(books_meta)} books, "
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
