"""Room decor for the library scene: a modern look around the robot's work area.

Called by generate_scene.py on the MjSpec before it is saved. Everything here is decoration:

  * visual geoms (group 2, contype/conaffinity 0, massless): furniture and vase meshes, flowers,
    rugs, art prints, LED pendants, wall slats, the ground outside, the east wall redrawn around
    the window opening. The LiDAR skips group 2 and the occupancy maps skip non-colliding geoms.
  * one invisible collision box (group 3, like the books' collision proxies) per piece that
    stands on the floor at LiDAR / chassis height: sofa, chairs, coffee table, planters, plinths.
    The LiDAR sees these boxes, and the navigation and MCL maps contain them (static boxes), so
    the decor is part of the known map like the tables and shelves.

Nothing may stand in a robot work zone (the taped floor areas at the tables and shelves):
generation fails if a collision box overlaps one.

Models and textures (scene/assets/decor, scene/textures/decor, floor_laminate.png,
oak_veneer.png): Poly Haven (https://polyhaven.com), CC0. The models were converted from glTF to
one OBJ + PNG per material, Z up, metres. Art prints are generated here.
"""
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "decor"
TEX = HERE / "textures"
INFO = json.loads((ASSETS / "decor.json").read_text())

VISUAL_GROUP, PROXY_GROUP = 2, 3
ZONE_GAP = 0.05            # m: collision boxes keep at least this far from a robot work zone

# (model, part, scale, dx, dy): tall periwinkle in the middle, a ring of daisies round the rim
BOUQUET_A = [("periwinkle_plant", 0, 1.8, 0.0, 0.03), ("periwinkle_plant", 2, 1.7, -0.03, -0.04),
             ("flower_gazania", 3, 1.5, 0.12, 0.0), ("flower_gazania", 1, 1.4, -0.12, 0.04),
             ("flower_gazania", 2, 1.4, 0.0, 0.13), ("flower_gazania", 4, 1.4, 0.02, -0.13),
             ("flower_gazania", 0, 1.3, -0.1, -0.1)]
BOUQUET_B = [("periwinkle_plant", 2, 1.8, 0.02, 0.0), ("periwinkle_plant", 5, 1.8, -0.04, 0.05),
             ("flower_ursinia", 3, 1.5, 0.12, 0.03), ("flower_ursinia", 2, 1.5, -0.12, -0.02),
             ("flower_gazania", 3, 1.4, 0.0, 0.13), ("flower_gazania", 1, 1.4, 0.03, -0.13),
             ("flower_ursinia", 4, 1.0, -0.08, 0.1)]
BOUQUET_C = [("periwinkle_plant", 3, 1.9, 0.0, 0.0), ("flower_gazania", 0, 1.4, 0.12, 0.05),
             ("flower_gazania", 1, 1.4, -0.12, 0.04), ("flower_gazania", 2, 1.4, 0.02, 0.13),
             ("flower_gazania", 3, 1.4, 0.0, -0.13), ("flower_gazania", 4, 1.3, -0.1, -0.09),
             ("flower_gazania", 5, 1.3, 0.1, -0.08)]
BOUQUET_D = [("periwinkle_plant", 0, 1.8, -0.02, 0.02), ("periwinkle_plant", 5, 1.7, 0.04, -0.04),
             ("flower_ursinia", 2, 1.5, 0.12, 0.04), ("flower_ursinia", 3, 1.4, -0.12, 0.02),
             ("flower_ursinia", 1, 1.5, 0.0, 0.13), ("flower_gazania", 3, 1.3, 0.0, -0.13)]


def _quat(yaw):
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def make_art(path, seed):
    """Abstract print: colour blocks and circles on off-white."""
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    pal = [(214, 118, 82), (232, 184, 72), (38, 61, 92), (128, 160, 140), (226, 214, 196), (30, 30, 34)]
    W, H = 900, 600
    im = Image.new("RGB", (W, H), (243, 239, 232))
    d = ImageDraw.Draw(im)
    for _ in range(4):
        x0, y0 = rng.integers(0, W - 200), rng.integers(0, H - 150)
        d.rectangle([x0, y0, x0 + rng.integers(150, 420), y0 + rng.integers(120, 330)], fill=pal[rng.integers(len(pal))])
    for _ in range(2):
        cx, cy, r = rng.integers(150, W - 150), rng.integers(120, H - 120), rng.integers(70, 170)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=pal[rng.integers(len(pal))])
    d.line([(rng.integers(0, W), 0), (rng.integers(0, W), H)], fill=(30, 30, 34), width=10)
    im.save(path)


class Decor:
    def __init__(self, spec, zones):
        self.spec, self.zones = spec, zones
        self.body = spec.worldbody.add_body()
        self.body.name = "decor"
        self.made = set()
        self.bounds = {}
        self.n_proxy = 0

    # ------------------------------------------------------------------ assets
    def texture(self, name, file):
        t = self.spec.add_texture()
        t.name, t.type, t.file = name, mujoco.mjtTexture.mjTEXTURE_2D, os.path.relpath(file, TEX).replace("\\", "/")
        return t

    def material(self, name, tex=None, rgba=None, repeat=None, uniform=False, specular=0.15, shininess=0.2,
                 emission=0.0):
        m = self.spec.add_material()
        m.name = name
        if tex:
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex
        if rgba is not None:
            m.rgba = rgba
        if repeat is not None:
            m.texrepeat = repeat
        m.texuniform, m.specular, m.shininess, m.emission = uniform, specular, shininess, emission
        return m

    def part_mesh(self, model, k, scale):
        part = INFO[model]["parts"][k]
        mname = f"decor_{model}_{k}_s{int(round(scale * 100))}"
        if mname not in self.made:
            me = self.spec.add_mesh()
            me.name = mname
            me.file = (ASSETS / part["obj"]).as_posix()     # absolute: see relative_mesh_paths
            me.scale = [scale] * 3
            me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            if part["png"]:
                tname = f"decor_{part['png'][:-4]}"
                if tname not in self.made:
                    self.texture(tname, TEX / "decor" / part["png"])
                    self.made.add(tname)
                self.material(mname + "_mat", tex=tname)
            else:
                self.material(mname + "_mat", rgba=part["rgba"])
            self.made.add(mname)
        return mname

    def part_bounds(self, model, k):
        if (model, k) not in self.bounds:
            v = []
            for line in (ASSETS / INFO[model]["parts"][k]["obj"]).read_text().splitlines():
                if line.startswith("v "):
                    v.append([float(x) for x in line.split()[1:4]])
            v = np.array(v)
            self.bounds[(model, k)] = (v.min(0), v.max(0))
        return self.bounds[(model, k)]

    # ------------------------------------------------------------------ geoms
    def geom(self, gtype, pos, size=None, yaw=0.0, material=None, rgba=None):
        g = self.body.add_geom()
        g.type, g.pos, g.quat = gtype, pos, _quat(yaw)
        if size is not None:
            g.size = size
        if material:
            g.material = material
        if rgba is not None:
            g.rgba = rgba
        g.contype = g.conaffinity = 0
        g.group, g.mass = VISUAL_GROUP, 0
        return g

    def box(self, pos, half, material=None, rgba=None, yaw=0.0):
        return self.geom(mujoco.mjtGeom.mjGEOM_BOX, pos, half, yaw, material, rgba)

    def proxy(self, x, y, hx, hy, h, yaw=0.0):
        """Invisible collision box standing on the floor; must keep clear of every robot zone."""
        c, s = math.cos(yaw), math.sin(yaw)
        ax = np.array([[c, s], [-s, c]])                                   # the box's axes
        corners = np.array([x, y]) + np.array([[i * hx, j * hy] for i in (-1, 1) for j in (-1, 1)]) @ ax
        for z in self.zones:
            zx = (z["x"][0] - ZONE_GAP, z["x"][1] + ZONE_GAP)
            zy = (z["y"][0] - ZONE_GAP, z["y"][1] + ZONE_GAP)
            zc = np.array([[a, b] for a in zx for b in zy])
            apart = (corners[:, 0].max() < zx[0] or corners[:, 0].min() > zx[1]
                     or corners[:, 1].max() < zy[0] or corners[:, 1].min() > zy[1])
            for u in ax:                                                  # separating axis test
                pb, pz = corners @ u, zc @ u
                apart = apart or pb.max() < pz.min() or pb.min() > pz.max()
            if not apart:
                raise ValueError(f"decor at ({x:.2f}, {y:.2f}) is in robot zone {z['name']}")
        g = self.body.add_geom()
        g.name = f"decor_proxy_{self.n_proxy}"
        self.n_proxy += 1
        g.type, g.pos, g.quat, g.size = mujoco.mjtGeom.mjGEOM_BOX, [x, y, h / 2], _quat(yaw), [hx, hy, h / 2]
        g.group, g.rgba = PROXY_GROUP, [0.5, 0.5, 0.5, 0.3]
        return g

    def model(self, model, x, y, z=None, yaw=0.0, scale=1.0, collide=True):
        """Every part of a model, its local origin at (x, y) and its base on z (default the floor);
        plus a collision box around its footprint if it stands on the floor."""
        lo, hi = np.array(INFO[model]["lo"]) * scale, np.array(INFO[model]["hi"]) * scale
        zb = -lo[2] if z is None else z - lo[2]
        for k in range(len(INFO[model]["parts"])):
            mname = self.part_mesh(model, k, scale)
            g = self.geom(mujoco.mjtGeom.mjGEOM_MESH, [x, y, zb], yaw=yaw, material=mname + "_mat")
            g.meshname = mname
        if collide and z is None:
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            c, s = math.cos(yaw), math.sin(yaw)
            self.proxy(x + c * cx - s * cy, y + s * cx + c * cy, (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2,
                       hi[2] - lo[2], yaw)

    def flower(self, model, k, x, y, z, scale, yaw):
        """One plant of a model that holds several, centred at (x, y), base at z."""
        lo, hi = self.part_bounds(model, k)
        mname = self.part_mesh(model, k, scale)
        cen = (lo + hi) / 2 * scale
        c, s = math.cos(yaw), math.sin(yaw)
        g = self.geom(mujoco.mjtGeom.mjGEOM_MESH, [x - (c * cen[0] - s * cen[1]), y - (s * cen[0] + c * cen[1]),
                                                   z - lo[2] * scale], yaw=yaw, material=mname + "_mat")
        g.meshname = mname

    # ------------------------------------------------------------------ pieces
    def planter(self, x, y, bouquet, r=0.2, h=0.42):
        """Modern white cylinder planter with soil and flowers."""
        self.geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [x, y, h / 2], [r, h / 2, 0], rgba=[0.96, 0.96, 0.95, 1])
        self.geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [x, y, h - 0.02], [r - 0.015, 0.005, 0], rgba=[0.22, 0.15, 0.10, 1])
        for i, (model, k, sc, dx, dy) in enumerate(bouquet):
            self.flower(model, k, x + dx, y + dy, h - 0.03, sc, 1.7 * i)
        self.proxy(x, y, r, r, h)

    def plinth(self, x, y, h, w=0.32):
        self.box([x, y, h / 2], [w / 2, w / 2, h / 2], rgba=[0.95, 0.95, 0.94, 1])
        self.proxy(x, y, w / 2, w / 2, h)

    def rug(self, x, y, hx, hy, rgba, yaw=0.0):
        self.box([x, y, 0.004], [hx, hy, 0.004], rgba=rgba, yaw=yaw)

    def art(self, k, x, y, z, side, w=1.2, h=0.8):
        """Framed abstract print on the west (-1) / east (+1) wall, x = the wall's inner face."""
        self.box([x - side * 0.012, y, z], [0.012, w / 2 + 0.03, h / 2 + 0.03], "decor_black_mat")
        self.box([x - side * 0.026, y, z], [0.003, w / 2, h / 2], f"decor_art_{k}_mat")

    def sofa(self, x, y, yaw, length=1.9, depth=0.85):
        """Low modern sofa, its back towards -y before turning by yaw: grey fabric, two seat
        cushions, slim black legs."""
        fab, cush, leg = [0.45, 0.47, 0.50, 1], [0.55, 0.57, 0.60, 1], [0.08, 0.08, 0.09, 1]
        L, D = length / 2, depth / 2
        c, s = math.cos(yaw), math.sin(yaw)

        def b(dx, dy, z, half, rgba):
            self.box([x + c * dx - s * dy, y + s * dx + c * dy, z], half, rgba=rgba, yaw=yaw)
        b(0, 0, 0.25, [L, D, 0.13], fab)                                               # base
        b(0, -D + 0.09, 0.55, [L, 0.09, 0.2], fab)                                     # back
        for sx in (-1, 1):
            b(sx * (L - 0.08), 0, 0.42, [0.08, D, 0.08], fab)                          # arms
            b(sx * (L / 2 - 0.08), 0.06, 0.42, [L / 2 - 0.1, D - 0.17, 0.05], cush)    # cushions
            for sy in (-1, 1):
                b(sx * (L - 0.1), sy * (D - 0.1), 0.06, [0.015, 0.015, 0.06], leg)     # legs
        self.proxy(x, y, L, D, 0.75, yaw)


def add_decor(spec, layout, zones):
    """Add the decor to `spec` (the library scene before saving); zones: the robot work zones.
    The decor meshes get absolute file names (the spec compiles them while saving); call
    relative_mesh_paths on the saved file."""
    room = layout["room"]
    cx, cy = room.get("center", [0.0, 0.0])
    hx, hy = room["size"][0] / 2, room["size"][1] / 2
    wx_w, wx_e, wy_n = cx - hx, cx + hx, cy + hy                  # inner faces of the walls
    d = Decor(spec, zones)

    # materials: light oak, black metal, LED strip, art prints
    d.texture("decor_oak", TEX / "oak_veneer.png")
    d.material("decor_oak_mat", tex="decor_oak", repeat=[2.0, 2.0], uniform=True, specular=0.2, shininess=0.3)
    d.material("decor_black_mat", rgba=[0.08, 0.08, 0.09, 1], specular=0.4)
    d.material("decor_led_mat", rgba=[1.0, 0.98, 0.94, 1], emission=1.0)
    for k in range(3):
        f = TEX / "decor" / f"art_{k}.png"
        if not f.exists():
            make_art(f, 11 + 7 * k)
        d.texture(f"decor_art_{k}", f)
        d.material(f"decor_art_{k}_mat", tex=f"decor_art_{k}")

    # light oak slats over the whole north wall, behind the shelves
    for x in np.arange(wx_w + 0.05, wx_e - 0.04, 0.1):
        d.box([x, wy_n - 0.025, 1.25], [0.025, 0.012, 1.25], "decor_oak_mat")

    # linear LED pendants over the tables
    for lx, ly in [(-2.5, -2.0), (2.5, -2.0), (-2.5, 0.5), (2.5, 0.5)]:
        d.box([lx, ly, 2.36], [0.6, 0.03, 0.02], "decor_black_mat")
        d.box([lx, ly, 2.338], [0.58, 0.022, 0.004], "decor_led_mat")
        for dx in (-0.5, 0.5):
            g = d.geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], [0.003, 0, 0], rgba=[0.1, 0.1, 0.1, 1])
            g.fromto = [lx + dx, ly, 2.38, lx + dx, ly, 2.5]

    # lounge in the south-west corner: sofa against the west wall facing east, round coffee
    # table, rug, planter; art above the sofa (all south of Table A's work zone)
    d.rug(-3.05, -4.35, 0.85, 0.62, [0.78, 0.78, 0.76, 1])
    d.sofa(wx_w + 0.43, -4.3, -math.pi / 2, length=1.3, depth=0.8)
    d.model("coffee_table_round_01", -2.75, -4.35, scale=0.5)
    d.model("ceramic_vase_03", -2.65, -4.35, z=INFO["coffee_table_round_01"]["hi"][2] * 0.5)
    d.planter(-1.75, -4.7, BOUQUET_C)
    d.art(0, wx_w + 0.005, -4.3, 1.55, -1, w=1.1, h=0.75)

    # reading corner south-east: lounge chair, armchair, plinth with a vase, rug, planter
    d.rug(3.05, -4.35, 0.9, 0.6, [0.32, 0.36, 0.40, 1])
    d.model("mid_century_lounge_chair", 3.3, -4.3, yaw=math.pi)
    d.model("modern_arm_chair_01", 2.1, -4.42, yaw=math.pi * 1.1)
    d.plinth(2.68, -4.78, 0.45, w=0.24)
    d.model("ceramic_vase_01", 2.68, -4.78, z=0.45)
    d.planter(3.7, -3.42, BOUQUET_B)

    # side walls: art and planters (the window is on the east wall)
    d.art(1, wx_w + 0.005, -0.4, 1.55, -1)
    d.art(2, wx_e - 0.005, -2.6, 1.55, 1)
    d.planter(3.62, 0.9, BOUQUET_D)
    d.planter(3.62, -1.35, BOUQUET_A)
    d.planter(-3.62, 0.9, BOUQUET_A)

    # north wall: white vases on plinths between the shelves, planters in the corners
    for sx, vase in ((-1.3, "ceramic_vase_04"), (1.3, "ceramic_vase_01")):
        d.plinth(sx, 2.72, 0.9)
        d.model(vase, sx, 2.72, z=0.9)
    d.planter(-3.77, 2.6, BOUQUET_D, r=0.2, h=0.55)
    d.planter(3.77, 2.6, BOUQUET_C, r=0.2, h=0.55)

    # outside: a large ground of dark blue tiles with light grid lines, seen around the room and
    # through the window
    gt = spec.add_texture()
    gt.name, gt.type, gt.builtin = "decor_ground_tex", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    gt.rgb1, gt.rgb2 = [0.16, 0.24, 0.34], [0.11, 0.17, 0.26]
    gt.mark, gt.markrgb = mujoco.mjtMark.mjMARK_EDGE, [0.42, 0.52, 0.64]
    gt.width = gt.height = 512
    gm = d.material("decor_ground_mat", tex="decor_ground_tex", repeat=[0.5, 0.5], uniform=True,
                    specular=0.3, shininess=0.4)
    gm.reflectance = 0.15
    d.geom(mujoco.mjtGeom.mjGEOM_PLANE, [cx, cy, -0.01], [40, 40, 0.1], material="decor_ground_mat")

    # east wall: hide the solid wall (it still collides and the LiDAR still sees it: group 3) and
    # draw it again in four pieces around the window, so the window looks out
    win = layout["window"]
    wy, wz = win["pos"][1], win["pos"][2]
    wdy, wdz = win["half_size"][1], win["half_size"][2]
    wall = spec.geom("wall_e")
    wall.group = PROXY_GROUP
    wpos, whalf = list(wall.pos), list(wall.size)
    y0, y1, z1 = wpos[1] - whalf[1], wpos[1] + whalf[1], wpos[2] + whalf[2]
    for (ya, yb), (za, zb) in [((y0, y1), (0.0, wz - wdz)), ((y0, y1), (wz + wdz, z1)),
                               ((y0, wy - wdy), (wz - wdz, wz + wdz)), ((wy + wdy, y1), (wz - wdz, wz + wdz))]:
        d.box([wpos[0], (ya + yb) / 2, (za + zb) / 2], [whalf[0], (yb - ya) / 2, (zb - za) / 2], "plaster_mat")
    return d


def relative_mesh_paths(xml_path, meshdir):
    """Make the decor mesh files in the saved scene relative to <compiler meshdir> (the Panda's
    asset folder), so the scene does not depend on where the project lives."""
    rel = os.path.relpath(ASSETS, meshdir).replace("\\", "/")
    xml_path.write_text(xml_path.read_text().replace(ASSETS.as_posix(), rel))
