# ShelfSage

**LLM-Guided Autonomous Mobile Librarian Robot**
*Language-guided mobile manipulation in a simulated library (MuJoCo + Franka Panda)*

- **Course:** Robot Manipulators (IIT Mandi), course project
- **Team:** Chirag, Mihir
- **Status:** Phases 1-4 of 10 complete; Phase 5 next (see [Progress](#8-progress-tracker))
- **Note:** This is the *initial* idea. It can be simplified later if parts become too hard (see [Fallbacks](#9-risks-and-fallbacks)).

> This README is written so that the whole project can be resumed from scratch, for example in a new chat with an AI assistant. See [How to resume](#12-how-to-resume-in-a-new-chat).

---

## 1. Project in one paragraph

We are building a **simulated mobile manipulator** (a cart with a Franka Panda arm) that receives a natural-language instruction such as *"Find the Machine Learning book on Table B and place it in shelf S2-A3"* and carries it out on its own inside a simulated library. A **pretrained LLM acts as the high-level task planner**: it turns the instruction into a structured action sequence. **Low-level control stays conventional and verifiable** (A* navigation, inverse kinematics, Jacobian control, RRT, grasp and place routines). The LLM never touches the joints.

The core research question: **does an LLM planner make the robot more flexible than a hand-written rule-based planner, when the navigation and manipulation stack underneath is identical?**

---

## 2. Problem statement and motivation

Develop an autonomous mobile manipulator that understands natural-language instructions and executes them in a simulated library.

The robot must:

1. **Understand** the user's intent and identify the relevant objects
2. **Locate** the target book and the destination
3. **Navigate** autonomously to the required location
4. **Grasp** the correct book with its robotic arm
5. **Deliver**: navigate to the destination and place the book correctly
6. **Verify** the outcome and recover or replan if an action fails

**Why it is interesting:** traditional robots need predefined task sequences, coordinates and explicit rules. Natural language lets one robot handle varied phrasings, multi-step tasks, and spatial relations such as *closest*, *left of*, *empty shelf*. Broader relevance: warehouse automation, hospital logistics, household and service robotics.

---

## 3. Approach

### 3.1 Two planners, same stack

| | Baseline (no LLM) | LLM planner |
|---|---|---|
| Idea | Fixed rules map known command templates to a fixed task sequence | LLM reads instruction + world state and produces a dynamic plan |
| Strength | Simple, predictable | Handles new phrasings, multi-step tasks, spatial relations |
| Weakness | Every new instruction needs new hand-written logic (brittle, hard to scale) | Can produce invalid or wrong plans, adds latency |

Both planners call the **same skills** (navigate, grasp, place, verify), so the comparison is fair: only the planner changes.

Example of a command the baseline struggles with: *"Take the book closest to Table D and place it in the empty shelf nearest to the window."*

### 3.2 Architecture

```
Natural-language command
        |
        v
   LLM Planner  <---- World State JSON (objects, shelves, robot pose)
        |               + skill list + constraints
        v
  JSON action plan --> Validator (reject unknown / invalid actions)
        |
        v
  Skill Executor:  NAVIGATE -> GRASP -> PLACE -> VERIFY
        |               |         |
     A* + cmd_vel    IK/Jacobian  attach grasp (contact grasp = stretch)
     on MCL pose     + RRT-Connect
     LiDAR obstacle  wrist RGB-D -> YOLOv8-seg -> book pose
     detection
        |
        v
   MuJoCo simulation (library scene, Panda on a cart; LiDAR, odometry, wrist RGB-D)
        |
        v
   VERIFY (simulator ground truth) --> on failure: send new state to LLM and replan
```

Example LLM output:

```
FIND_BOOK -> NAVIGATE(Table B) -> GRASP(Book) -> NAVIGATE(Shelf S2-A3) -> PLACE(Book, S2-A3) -> VERIFY()
```

### 3.3 Design principles

- **The LLM plans, proven controllers execute.** Keeps the system verifiable and safe (same philosophy as SayCan).
- **Skills are simulator-independent.** Planner code only knows skill names, never MuJoCo. Switching simulator would only change skill backends.
- **The LLM is not in the control loop.** The plan is generated once before execution, so API latency does not block the simulation.
- **Spatial facts are computed in code**, not left to the LLM (for example, distance from each book to a table is pre-computed and put in the world state).
- **Everything is validated.** Unknown actions, wrong objects or wrong arguments are rejected and the LLM is asked again.
- **Layout file is the single source of truth.** The same file generates the MuJoCo XML and the world-state JSON.

---

## 4. Technical decisions (and why)

### 4.1 Simulator: MuJoCo (decided)

The deck originally said NVIDIA Isaac Sim. We switched to **MuJoCo** because:

- The development machine has an **RTX 4060 (8 GB VRAM)**. Current Isaac Sim documentation lists a minimum of an RTX 4080 with 16 GB VRAM; older versions listed around 8 GB. The 4060 is below the current minimum, and sensor-heavy workflows (LiDAR, cameras) are affected most.
- MuJoCo runs on the CPU, installs with `pip`, has good contact physics (useful for grasping), supports RGB-D cameras, and Python access to all state is simple.
- Isaac Sim's base + arm assembly is known to be painful (articulation and prim-path issues).
- LLM integration is easier: the LLM client and MuJoCo live in the same plain Python environment.

Trade-offs accepted:
- No drag-and-drop scene editor. The scene is written as MJCF XML, generated by a Python script from a layout file.
- No built-in ROS/Nav2. Navigation is our own A* on an occupancy grid.
- Visuals are simpler than Isaac Sim (can be improved with textures, skybox and lighting).

Simulators considered and rejected: PyBullet (less stable contact, basic rendering; kept as backup), Gazebo + ROS2 (heavy setup, no Panda by default), AI2-THOR (removes the arm-control part, which is the point of this course), Hyfydy (biomechanics engine, not relevant).

### 4.2 Robot

- **Arm:** Franka Emika Panda, 7-DoF serial arm, revolute joints, redundant DoF useful for obstacle avoidance and shelf poses.
- **Model source:** MuJoCo Menagerie (`franka_emika_panda`). It is a *simplified* MJCF derived from the public URDF, with position-controlled arm actuators and custom fingertip collision geoms. It is not an exact digital twin: do not quote simulated forces as real Panda values. (If the folder is missing in a newer Menagerie, the FR3 model is the likely replacement; also 7-DoF.)
- **Gripper:** standard Panda **2-finger parallel gripper** (changed from the "3-point gripper" in the original deck; Panda does not ship with a 3-finger gripper, and a parallel gripper suits flat books better).
- **Base ("cart"):** a **virtual holonomic base** built from x, y and yaw joints under the arm, driven by **velocity commands** (`cmd_vel`: vx, vy, wz) like a real mobile base. The controller closes the loop on the **estimated** pose, never on simulator ground truth. This replaces "differential-drive base" from the deck. Real wheel physics is not simulated. The arm is mounted at the **front edge** of the chassis (reach is tight otherwise).
- **Optional:** a vertical lift (extra prismatic z joint) between base and arm, to reach upper shelf rows. Panda reach is only about 85 cm.
- **Sensors (core, decided):**
  - **2D LiDAR**: 360 rays via `mujoco.mj_multiRay` from a site on the front of the chassis, about 0.25-0.3 m high (below the arm), 8 m range, with noise. At this height it sees walls, shelf and table *legs*, as a real robot would.
  - **Odometry**: base velocities integrated with noise, so it drifts like wheel odometry.
  - **Wrist RGB-D camera** (eye-in-hand, on the `hand` body), rendered on demand with the MuJoCo offscreen renderer.

### 4.2a Localization, obstacles and perception (decided: realistic route)

- **Map:** static prior map from `layout.json` (in practice a map is built once with SLAM, then reused). Full SLAM is not implemented.
- **Localization:** Monte Carlo Localization (particle filter, as in AMCL): predict with odometry, weight particles by matching the LiDAR scan against the scan expected from the map, resample. `get_pose()` returns odometry in Phase 3 and the MCL estimate from Phase 6; `true_pose()` is used only to measure error.
- **Obstacles:** LiDAR returns much shorter than the map predicts are marked in the occupancy grid; the robot stops if one is too close and replans with A*. If no path exists, `NAVIGATE` fails with a reason and the LLM replans (Phase 9). Test obstacles are pre-placed out of the room and moved into the path at runtime.
- **Perception:** a learned detector, **YOLOv8n-seg** with 13 classes (one per book), trained on images generated from the simulator with domain randomization (book poses, lighting, camera pose, textures). Labels come for free from MuJoCo segmentation renders. Mask + depth -> 3D points -> centre and yaw (PCA) -> grasp target in the robot frame, so localization error does not affect the grasp. Colour + depth thresholding is the classical baseline; ground truth is always the fallback.
- **Assumption to state in the report:** after parking, the book pose comes from the wrist camera relative to the robot, not from the global map.

### 4.3 Libraries and tools

- Python virtual environment (`venv`), packages: `mujoco`, `numpy`, `scipy`, `imageio` (pinned in `requirements.txt`)
- Perception (Phase 5): PyTorch with CUDA + `ultralytics` (YOLOv8), trained on the RTX 4060
- LLM: API-based model returning JSON (final choice still open)
- Git for version control

### 4.4 Simplifications chosen (hardest parts made easier)

| Hard part | Simplification |
|---|---|
| Mobile manipulation (base + arm coordination) | Fixed **parking pose** in front of each table and shelf; the base stops, then the arm works. Base and arm never move together. |
| Grasping | Start with ground-truth object poses and an **"attach" (weld) grasp**. Contact-based friction grasp is a stretch goal. |
| SLAM / Nav2 | **Known map** from the layout file + A* waypoint navigation; the robot still localizes itself with LiDAR + odometry + MCL. Full SLAM is not implemented. |
| RRT | Use RRT-Connect only where needed (entering the shelf). Fallback: hand-designed approach trajectory. |
| Perception | Ground truth first, then a YOLOv8-seg detector on the wrist RGB-D camera (trained on auto-labelled sim images). Ground truth stays as fallback. |
| Shelf extraction | Leave gaps between books, spines facing out, pull straight out. |

Easy and not needing simplification: IK, Jacobian control (numpy), PID, collision checks, LLM API call and JSON schema, the action library, world-state JSON, the rule-based baseline, VERIFY from ground truth.

---

## 5. Simulated library environment (built in Phase 2)

Everything is defined in [`scene/layout.json`](scene/layout.json) and generated by `scene/generate_scene.py` into `scene/library.xml` (MJCF) and `scene/library_meta.json` (positions of tables, shelf slots and books; the starting point for the world-state JSON).

- Room 8 m x 8 m (x -4..4, y -5..3, `room.center` = (0, -1)), wall height 2.5 m; the south side was extended 2 m to leave free floor for the robot. South wall collides but is hidden (geom group 4) so the cameras can see in
- "ShelfSage" graffiti tag on the north wall above the shelf (`sign` in the layout; texture from `make_textures.py` using the Sedgwick Ave font, SIL OFL, in `assets/fonts/`)
- Window on the east wall at x = 3.97 (used for commands like "nearest to the window")
- 6 tables (A to F) in two rows (front row A-C at y = -2.0, back row D-F at y = 0.5), top at z = 0.75 m; each has 3 **edge spots** on the robot side (south), where the robot puts books it brings from the shelf
- **3 bookshelves** on the north wall: S1 (west, x = -2.6), S2 (centre, x = 0), S3 (east, x = 2.6). Each has 2 rows (A bottom, B top) x 4 columns; row A surface at z = 0.5 m, row B at 1.0 m, top at 1.4 m (row B moved up from 0.9 m to stay out of the arm's dead band, see Phase 3). Compartments are named `<shelf>-<row><col>`, e.g. **S2-A3**, so they cannot be confused with tables A-F. Each compartment holds up to 4 standing books, 7 cm apart
- **39 books** (id, title, colour, location): 11 lying on tables, 28 standing in the shelves (14 of the 24 compartments used, some full, some empty for placing). Shelf locations are written `shelf:S2-A1` (the next free position from the left)
- **Book design ("customised library books")**: every book is 22 x 15 x 4 cm, with 1 cm high, 2 cm wide bumper strips (ribs) across both covers at head and tail. Lying flat, a book rests on its ribs, leaving a 1 cm gap underneath for the lower fingertip (the Panda fingertip is 7.6 mm thick); standing in the shelf, neighbours touch only at the ribs, leaving gaps for the fingers. The title is printed on the spine (texture, Bebas Neue font, SIL OFL)
- On a table a book lies flat with its **spine towards the robot side**, either at an edge spot (default) or anywhere on the table (`"pos": [dx, dy]` in the layout, e.g. Calculus, Control Systems, Databases). At an edge spot the spine **overhangs the table edge by 3 cm** (the gripper holds that overhanging part, see Phase 4); the robot drags other books to the edge and puts books it brings there. In a shelf it stands with the spine facing the room, **3 cm past the shelf front** (like the table edge), packed from the left divider, up to 5 per compartment, 7 cm apart
- 12 static wooden chairs, 2 per table on the readers' side (north), pushed in 8 cm; the robot side (south) stays free. The front row was moved 0.5 m south so its chairs stay out of the back-row robot zones. Chair legs show up in the LiDAR map like everything static
- Robot start pose (0, -3.9, yaw 90 deg = facing the tables), on open floor outside every zone; the robot itself is described in Phase 3 below
- **Fixed robot parking poses and floor zones**: the robot always picks / places from fixed poses, 0.50 m from the furniture: one per table edge spot (`tables.<T>.parking` in `library_meta.json`) and one per column of each shelf (`shelves.<S>.parking`). Around them the floor is marked with red tape, a light red fill and a **KEEP CLEAR - ROBOT WORK ZONE** plate (`robot_zones` in the layout and the meta). The markings are visual only (no collisions, not in the LiDAR map); the zone rectangles in the meta are meant for later use, e.g. injecting a "chair left in the robot zone" failure in Phase 9
- The whole scene uses radians (`<compiler angle="radian">`), like the Panda
- Books are free bodies (`freejoint`): collision = body box (geom group 3, hidden) + 4 rib boxes; visual = covers, spine, pages, ribs, spine label. Tables and shelf are fixed
- Procedural textures (`scene/make_textures.py`), cameras `overview` and `cutaway`
- **Real front covers** on every book: each generic title stands in for a well-known textbook (e.g. Deep Learning -> Goodfellow et al., Robotics -> Craig), list in `scene/fetch_covers.py`. The script downloads the covers from Open Library into `scene/textures/covers/<id>.jpg` (a file put there by hand is kept); `generate_scene.py` scales each to the 22 x 15 cm front and splits off the 2 cm strips under the head / tail ribs, so the cover is complete seen from above. The spine keeps the generic title (the planner's book names). A book without a cover file keeps its plain colour.

Commands (project root, venv active):

```powershell
python sceneetch_covers.py               # download the book covers (once; --force to redo)
python scene\generate_scene.py             # regenerate library.xml + library_meta.json
python scene\generate_scene.py --check     # also simulate 10 s and check books stay put
python scene\generate_scene.py --preview   # also render scene\preview_*.png
python sim\view_scene.py                   # open the scene in the viewer
python sim\demo_phase3.py                  # watch the robot drive, reach the table and the shelf
python sim\demo_pick_place.py              # table -> shelf -> table round trip (add --headless for a report)
python sim\demo_drag_grasp.py              # pull a mid-table book to the edge, then grasp it (--book db, --headless)
```

---

## 6. Skill interface

Defined in `skills/interface.py` (bind a robot with `bind(robot)`; books by title or id):

| Skill | Purpose | Status |
|---|---|---|
| `FIND_BOOK(title)` | Look up which table or shelf holds the book (world-state lookup) | ground truth (Phase 5); world model in 7 |
| `NAVIGATE(target)` | Drive the base to the parking pose of a table ("Table B") or shelf ("S2-A3") | fixed lanes + aisle (Phase 5); A* in 6 |
| `GRASP(obj)` | Pick up a book with the 2-finger gripper (drags it to the table edge first if needed) | done (Phase 5) |
| `PLACE(obj, shelf)` | Put the held book in a shelf compartment, leftmost free slot | done (Phase 5) |
| `VERIFY(obj, shelf)` | Check that the book stands inside the compartment | ground truth (Phase 5); camera in 9 |

Every skill returns a `SkillResult(success, message, data)`, so a failure and its reason can be sent to the LLM for replanning. The `SKILLS` dictionary maps action names to functions; the LLM may only call actions in this dictionary.

---

## 7. The 10-phase plan

Each phase has a "Done when" check. Phases 2, 3 and 4 can be split between two people. The detector does not need the robot: once the wrist camera is fixed in Phase 3 (640x480, fovy about 58 deg, similar to a RealSense D435), dataset generation and YOLO training run in parallel with Phase 4, and Phase 5 only plugs the detector into the grasp. Changing the camera later means regenerating the dataset.

### Phase 1: Setup and foundations
- Install MuJoCo, load the Panda from Menagerie, create repo and folders, define the skill interface.
- **Done when:** the Panda appears in the viewer and a joint can be moved from code.

### Phase 2: Library scene
- Write the layout file and `generate_scene.py` that outputs the MJCF (floor, lights, tables, shelf with compartments, books, window).
- **Done when:** the scene loads and no book falls or jitters during a 10-second simulation.

### Phase 3: Robot assembly (cart + arm)
- Virtual holonomic base (x, y, yaw) with velocity actuators; Panda attached at the front edge (MuJoCo `attach` API, loaded from Menagerie, not copy-pasted); `home` keyframe; `tcp` site; inserted by `generate_scene.py` at `robot_start`.
- Reach check that picks the chassis height for table tops and both shelf rows; optional lift only if needed.
- Sensors: `lidar` site + `lidar_scan()`, noisy `odometry()`, wrist RGB-D camera + `get_rgbd()`.
- `sim/robot.py`: `send_cmd_vel`, `get_pose` (odometry for now), `true_pose` (evaluation only), `go_to` P-controller, arm and gripper helpers.
- **Done when:** the robot drives to a table, stops, and the arm reaches over the table surface; a LiDAR scan drawn at the true pose lines up with the map; odometry drift over a drive is logged; a wrist RGB + depth image is saved; books are still stable.

### Phase 4: Arm control (FK, IK, Jacobian)
- Forward kinematics (checked against MuJoCo), Jacobian, differential IK with damped least squares and null-space joint-limit avoidance, Cartesian straight-line moves, gripper open/close.
- **Done when:** the gripper reaches 10 random reachable poses with about 1-2 cm accuracy.

### Phase 5: Grasp and place skills
- `GRASP`: pre-grasp pose, descend, close, lift (attach grasp first). `PLACE`: pre-place in front of compartment, insert, release, retract. RRT-Connect for entering the shelf. Small success checks after each step.
- Perception: dataset generator (domain randomization, auto labels from segmentation renders), YOLOv8n-seg training, mask + depth -> book pose; colour + depth baseline.
- **Done when:** with the robot standing still, a book moves from a table to shelf compartment S2-A3 correctly at least 8 of 10 times, first with ground-truth poses, then with poses from the detector.

### Phase 6: Navigation
- Occupancy grid from the layout, obstacle inflation, A*, smoothed waypoints, parking pose per table and shelf.
- MCL localization (odometry + LiDAR vs map); LiDAR obstacle detection, stop and A* replanning.
- **Done when:** the robot reaches every table and shelf without collisions using only the MCL pose, the arm can reach from each parking pose, localization error is logged against ground truth, and an obstacle placed in the path is detected and avoided.

### Phase 7: Skills integration and baseline planner
- Executor that runs an action list step by step; rule-based parser for fixed templates; logging of every step (time, success).
- **Done when:** 5-10 fixed commands run end to end, from navigation to placement.

### Phase 8: LLM planner
- Skill schema (JSON), world-state JSON with pre-computed spatial facts, prompt with 2-3 examples, low temperature, validator with retry, plan once before execution, latency and plan logging.
- **Done when:** the same commands work via the LLM and newly phrased commands (which the baseline cannot handle) also produce correct plans.

### Phase 9: Verify, failure injection and replanning
- `VERIFY` from ground truth, injected failures (dropped book, full target shelf, blocked path), replanning with the new state and error sent back to the LLM (retry limit 2-3), same failures through the baseline. Optional: VERIFY with the wrist camera, compared to ground truth.
- **Done when:** the LLM planner recovers after at least 2-3 different injected failures.

### Phase 10: Benchmark, evaluation and report
- Benchmark task set (single-step, multi-step, different phrasings, spatial relations, injected failures), both planners on the same tasks, metrics and statistics finalized at this stage, plots, success/failure examples, reproducible `run_demo.py`, final report and updated slide deck.
- **Done when:** results for both planners are clear in a table and at least 2-3 plots.

### If time runs short
Cut first: contact-based grasping, camera-based VERIFY, then the learned detector (fall back to colour + depth or ground truth), then MCL (fall back to ground-truth pose; keep LiDAR obstacle detection). Do **not** cut Phases 1-8 (ground-truth version) or verify and replanning in Phase 9, since that is the core story.

**Evaluation is deliberately deferred.** Metrics and statistical tests will be decided at the end. Ideas from the original deck (task success rate, navigation success, grasp success, placement accuracy, collision rate, replanning success, planning latency, execution time; ablation LLM only / + world model / + verification and replanning; McNemar or Wilcoxon tests) are options, not commitments.

---

## 8. Progress tracker

Update this section as work proceeds.

### Phase 1: Setup and foundations (done)

- [x] Project directory created (`ShelfSage`) and Python virtual environment (`venv`) created
- [x] Git repository initialised
- [x] Folder structure created: `scene/ sim/ skills/ control/ planner/ benchmark/ assets/`
- [x] `.gitignore` created (`venv/`, `.venv/`, `__pycache__/`, `assets/mujoco_menagerie/`)
- [x] MuJoCo Menagerie cloned into `assets/mujoco_menagerie` (Panda model)
- [x] Test script `sim/test_panda.py` ran: Panda showed in the viewer and a joint moved (removed later; the whole robot is tested in Phases 3-4)
- [x] Skill interface written: `skills/__init__.py` and `skills/interface.py` (stubs)
- [x] Joint names and actuator count saved into `NOTES.md` (via `sim/save_joint_names.py`; now regenerated for the full robot)
- [x] `requirements.txt` committed (UTF-8)
- [x] First Git commit made

### Phase 2: Library scene (done)

- [x] `scene/layout.json`: room, window, 6 tables, shelf (2 x 4 compartments), 13 books, robot start
- [x] `scene/make_textures.py`: procedural floor, wood, shelf and plaster textures, plus the ShelfSage wall tag
- [x] Room extended 2 m to the south (8 m x 8 m) for robot clearance; check re-run, all books stable
- [x] `scene/generate_scene.py`: writes `library.xml` + `library_meta.json`, with `--check` and `--preview`
- [x] `sim/view_scene.py`: opens the scene in the viewer
- [x] Done-when check passed: `--check` shows all 13 books stable over 10 s (each settles 0.11 cm, the 1 mm spawn gap, then stays put)

### Phase 3: Robot assembly (done)

- [x] `scene/robot_model.py`: chassis 0.6 x 0.5 m, top (arm mount) at **0.65 m** (was 0.45 m, see reach check), arm 10 cm behind the front edge; base joints `base_x/base_y/base_yaw` (their qpos *is* the world pose, via `ref`) with velocity actuators `base_vx/vy/wz`; Menagerie Panda attached with MjSpec (prefix `panda_`), gravity compensation on, `panda_tcp` site and `panda_wrist_cam` (640x480, fovy 58) on the hand; `home` keyframe; LiDAR site at the front, 0.20 m high. Robot settings live in `layout.json` -> `robot`.
- [x] `sim/reach_check.py`: numerical IK for every Phase 5 pose (table edge grasp + lift, press-and-drag of the farthest book, shelf A/B grasp + carry) over forward distances 0.20-0.90 m from the arm mount. **Finding:** with the gripper pointing straight ahead the Panda cannot reach 0.27-0.38 m above its mount at any distance (joint 4 / joint 6 limits). At 0.45 m that dead band (0.72-0.83 m) swallowed the table grasp, so the mount went to **0.65 m** (band 0.92-1.03 m) and shelf row B to 1.00 m. Every task is now reachable from about 0.25 to 0.85 m ahead of the mount.
- [x] `sim/robot.py`: `send_cmd_vel`, noisy `odometry` (2 % scale error + white noise), `lidar_scan` (270 deg, 541 rays, 8 m, 1 cm noise), `get_rgbd` + `camera_intrinsics`, `get_pose` (odom / true; MCL in Phase 6), `true_pose` (evaluation only), `go_to` (P-controller, speed and acceleration limits), `move_arm`, `home`, gripper, `robot_contacts`.
- [x] `control/occupancy.py`: occupancy grid of static boxes at a height band (the known map for Phase 6).
- [x] `sim/test_robot.py`: **all 10 checks pass.** Drive to Table B: 0.4 cm / 0.0 deg error, no overshoot, no collisions; TCP reaches 10 cm above the middle book (error < 1 mm); LiDAR scan 100 % within 4 cm of the map; wrist depth sees the book cover at 15.5 cm and the top ribs at 14.5 cm (as expected); books stable. A combined image is written to `sim/output/phase3/results.png`. Odometry over an 11 m loop drifts about 9 cm and 5 deg, which is what MCL must fix. Images in `sim/output/phase3/` (git-ignored).

Notes for later phases:
- With the camera 10 cm above a book, the book fills the image; Phase 5 needs a "look" pose about 40-50 cm above the table to see all books.
- Travel pose: `robot.stow()` folds the arm over the chassis before driving; `home` sticks out 0.84 m and hit the shelf when parking there. At a table, unfold to `home` first (elbow up) before reaching down.
- Camera near plane: `<visual><map znear="0.001">` (1.7 cm). The default clipped everything nearer than 16.6 cm, so close-up books were cut open.
- The hand is mounted rotated 45 deg (Menagerie), so the wrist image is rotated; the detector / PCA yaw must handle any rotation.
- **Grasp design (decided, refined in Phase 4):** books are picked by their thickness, not top-down (15 cm is wider than the 8 cm gripper). Table: gripper horizontal, fingers close vertically around the **3 cm overhanging spine**, TCP 1.5 cm in from the spine (inside the 1 cm rib gap the lower finger still hit the table: its pad is ~1 cm thick and MuJoCo collides the finger mesh as a convex hull, 24 mm thick near the hand). A book that is not at the edge is first **dragged to the edge** (closed fingertips press on its top 4 cm in from the spine, gripper tilted 30 deg *back* towards the robot; see Phase 5). Shelf: gripper horizontal, fingers close sideways around the standing book, TCP 1.5 cm in from the spine; books stick out 3 cm past the shelf front, so the fingers stay outside the compartment and a book can stand right against a divider (see Phase 5). Neighbouring shelf books are 7 cm apart (3 cm between covers). The book is rotated 90 deg with the wrist between lying and standing. The weld (attach) grasp stays as a fallback.
- **Parking distances (Phase 4):** 0.50 m between chassis and table / shelf. At 0.30-0.40 m the pre-grasp / pre-place poses (10-15 cm out) folded the arm into the chassis or put joints 4/5 at their limits, so straight-line moves from there failed; 0.45-0.60 m works. Drive closer (3-50 cm, chosen per stroke) only to drag a far book.

### Phase 4: Arm control (done)

- [x] `control/kinematics.py`: FK from Franka's modified DH parameters + Menagerie hand (-45 deg) + TCP offset, in the robot base frame (arm mount at 0.20 m / 0.65 m); geometric Jacobian. Matches MuJoCo to 0.004 mm / 0.0003 deg and `mj_jacSite` to 4e-6 (the saved XML rounds the base quaternion).
- [x] `control/ik.py`: damped least squares IK with partial orientation targets (approach axis only; approach + finger closing axis, sign free; or full rotation). Null space: joint-limit avoidance + preferred posture (elbow up), with an exact projector (the damped one leaked into the task and stalled 3 cm short). Early stop when stuck, multi-start; solutions are ranked by joint motion, distance to the home posture and joint-limit margin (a solution with a joint at its limit can reach a pose but not move on from it). About 30 ms per IK incl. collision checks (scipy needed ~400 ms).
- [x] `sim/robot.py`: `ik`, `move_to` (IK + joint move), `move_linear` (straight Cartesian line, differential IK every control step, aborts and holds when the line leaves the workspace), `set_gripper_width`, `arm_collisions` (MuJoCo contact check of a candidate configuration), world <-> base transforms through the *estimated* pose. Test, demo and Phase 3 test now use it (no more scipy IK there; `reach_check.py` keeps its own analysis IK).
- [x] `sim/test_phase4.py`: **all 9 checks pass.** 10 random reachable poses reached in simulation with 2.8 mm max error (README done-when: 1-2 cm); every Phase 5 pose reachable without collisions at the fixed parking poses (table edge grasp, pre-grasp, lift; shelves S1-S3, rows A and B, grasp and pre-place); straight-line approach / lift / retract within 2.5 mm of the line; an unreachable line is aborted safely.
- [x] Demo (`sim/demo_phase3.py`) shows the Phase 4 motions: pre-grasp, straight line in around the overhanging spine and back out at Table B; pre-place and straight line in/out at empty compartment S2-B3.

Design changes found by the Phase 4 test (scene regenerated, all checks re-run):
- Table books at an edge spot overhang the edge by 3 cm (was 5 mm inside); shelf books flush with the front (was 2 cm behind).
- Parking 0.50 m from tables and shelves (was 0.30 m); the KEEP CLEAR zones grew with it.

### Phase 5: Grasp and place skills (next)

- [x] First step: **attach (weld) grasp** (the README fallback). One inactive weld hand<->book per book in the scene; `robot.attach(book)` switches it on with the book's current pose relative to the hand, `robot.release(book)` switches it off. `sim/demo_pick_place.py` runs the full round trip with it (Deep Learning: Table B -> shelf S2-B3 -> back to its edge spot), using the Phase 3-4 stack: parking poses, IK, straight-line moves, wrist rotation between lying and standing, a carry pose for driving. Headless run: 30/30 steps ok, the book ends 0.3 cm from its spot, other books move at most 0.11 cm.
- [x] **Drag to the edge, then grasp** ("kheencho, phir pakdo") for a book lying inside a table: `skills/manipulation.py` (`drag_to_edge`, `pick_from_table_edge`), demo `sim/demo_drag_grasp.py`. In strokes: park in line with the book so the press point is ~0.58 m ahead of the arm mount (3-50 cm from the edge), closed fingertips above the cover 4 cm in from the spine, guarded move down to 10 N, pull straight towards the robot (max 25 cm), lift, measure the book again, repeat until the spine overhangs 3 cm (+-1 cm). Then the normal edge grasp (attach), lift 10 cm, back off, carry pose. Headless: **Calculus** (40 cm in, Table A) and **Databases** (middle of Table E) both reach 3.0 / 3.2 cm overhang in 2 strokes, book yaw < 0.2 deg, lifted 9.9 cm, other books move at most 0.11 cm. **Control Systems** (Table C) is refused up front: Signals and Systems lies in its lane to the edge (`drag_blockers`), it would be pushed off; the planner has to move that one first.
  - Friction (scene): books 0.4 (paper on wood), finger pads 1.5 (rubber), set with geom **priority** so the book's value also holds against the 1.0 table / shelf and the pad's against the book. Pressing and pulling works when 1.5 F > 0.4 (F + m g), any F. Before, everything was 1.0 and the fingers just slid over the cover.
  - **Elliptic friction cone + impratio 10** (`<option>`): with the default pyramidal cone the fingertips crept 4 cm per 12 cm stroke and slid off the book; now ~5 mm per stroke (the skill pulls 5 mm further to compensate).
  - **Press force by a guarded move** (`robot.guarded_move`, force from `robot.finger_force`, the simulated fingertip force sensing): a fixed 1 cm press gave 120-170 N.
  - **Gripper tilted back** (-30 deg, i.e. fingertips towards the robot): tilted forwards, the arm cannot reach closer than 0.40 m from the mount, so the strokes ended early. The skill checks the IK at both ends of each stroke and falls back to -15 / 0 deg.
  - **Base station keeping** (`robot.hold_base`): the velocity actuators only damp motion, so the ~5 N drag reaction pulled the base 13 cm towards (and under) the table in one stroke. Held with a P-loop on the estimated pose it stays within 5 mm. Off by default: during fast arm swings the yaw loop rings (2 deg), which broke the Phase 4 accuracy checks.
  - Re-run after these changes: books stable (`--check`), Phase 3 test 9/9, Phase 4 test 9/9, pick-and-place demo 30/30.
- [x] **Whole-body drag: the arm presses, the base pulls** (`drag_to_edge(..., method="base")`, now the default; the arm strokes above stay as `method="arm"`). The fingertips press 0.50 m ahead of the arm mount and the arm keeps its joint targets while the base backs away at 4 cm/s, so the book comes the whole way in one stroke, however far in it lies (with the softer finger contacts of that time; since the friction grasp it takes 3-4 strokes, see below). A long pull leaves the book ~4 % short (fingertip creep + the base stopping ~5 mm early under the load), so the base pulls 4 % further. Headless, drag only (simulated time):

  | book (distance to pull) | arm strokes | base pull |
  |---|---|---|
  | Calculus, Table A (41 cm) | 2 strokes, 3.0 cm overhang, 50 s | 1 stroke, 3.4 cm, 33 s |
  | Databases, Table E (36 cm) | 2 strokes, 3.2 cm, 48 s | 1 stroke, 3.3 cm, 32 s |
  | Control Systems, Table C (55 cm; Signals and Systems moved away for this test) | 3 strokes, 3.5 cm, 70 s | 1 stroke, 3.5 cm, 37 s |

  Book yaw stays below 0.5 deg, the grasp afterwards lifts it 9.9 cm, other books move at most 0.11 cm. `python sim\demo_drag_grasp.py --method arm` runs the old way.
- [x] **Shelf books packed against the divider.** Before, the 4 slots sat in the middle of the 40 cm compartment (first book ~9 cm from the divider), because a finger does not fit in the 1 cm rib gap at a divider. Now shelf books stick out **3 cm past the shelf front** (`SHELF_OVERHANG`, was 5 mm inside) and are gripped 1.5 cm in from the spine (was 2 cm), so the fingers stay outside the compartment, where the divider ends. Books are packed from the left divider (ribs 1 mm from it), 5 per compartment (was 4). The pick-and-place demo now puts Deep Learning into the slot against the divider and takes it out again: 30/30 ok.
  - Pitch stays 7 cm (3 cm between covers). With ribs touching (6 cm pitch, 2 cm between covers, 6 per compartment) the Phase 4 test failed at every shelf grasp: the finger's collision hull is 24 mm thick towards the hand, and there it hit the neighbour's spine, which now sticks out too. Even with the gripper opened only 4.4 cm it fits only at 5 mm grip depth, too little for a real grasp.
  - Re-run: books stable, Phase 3 test 9/9, Phase 4 test 9/9 (all shelf grasps collision-free), pick-and-place 30/30, drag-then-grasp ok.
  - Driving to the north-row tables (D-F) goes through the aisle at x = 1.25 m in the demo (straight lines until A* in Phase 6).
- [x] **Friction grasp** (the weld is now only a fallback: `--weld` in the demos, `Robot(grasp_mode="weld")`). `robot.grasp(book)` commands the fingers shut and the book is held by friction alone. Headless pick-and-place (Table B -> S2-B3 against the divider -> out again -> Table B): **32/32 ok, the book ends 0.5 cm from its spot** (weld: 0.2 cm). Drag-then-grasp: Calculus and Databases ok with both drag methods, book lifted 10.0 cm. Books stable, Phase 3 9/9, Phase 4 9/9. What it took:
  - **Gripper force:** Menagerie's gripper servo is 100 N/m on the finger gap, ~2 N on a 4 cm book (less than its 3.3 N weight). Now 3000 N/m: ~60 N squeeze, inside the real Franka Hand's 70 N.
  - **Twist, not slide:** the fingers hold the book only by the 1.5 cm next to its spine, 6 cm from its centre of mass, so it turned between the fingers. Fixes: torsional friction on the finger pads (`condim 4`, 0.02 m), stiffer finger contacts (solref 4 ms) and `noslip_iterations 3`. A book held for 40 s crept 2.9 cm before, 0.02 cm now.
  - **Carry pose** moved to 0.60 m ahead / 1.20 m up (was 0.45 / 1.15, with joint 5 at its limit). While holding, IK drops solutions within 0.2 rad of a joint limit (`reject_margin`) and arm moves are slowed to 0.8 rad/s peak.
  - **Wrist family switch:** the table grasp (horizontal, 13 cm above the mount, at the edge of the dead band) only has twisted IK solutions, in the other wrist family than the carry pose. So the move to the carry pose still swings joint 5 by ~5 rad; slow, but the book shifts up to ~1 cm. Two things absorb that: **in-hand correction** (before placing, the TCP target is shifted by how far the book sits off in the fingers; ground truth now, wrist camera later) and **guarded set-down** (lower until the surface carries 2.5 N of the book, `robot.object_contact_force`, instead of to a fixed height). Shelf placing aims 6 mm off the divider (`PLACE_CLEARANCE`).
  - **Trade-off:** with the stiff finger contacts, pressing on a cover chatters (the contact keeps breaking), so the base-pull drag now takes 3-4 strokes instead of 1 (still 2.9-3.0 cm overhang). At solref 10 ms the drag took 1 stroke but the ~2 mm creep left in the grasp made the shelf insertion fail. An admittance loop holding 10 N during the pull did keep the force steady but the fingertip still slid over the cover; not kept.
- [x] **Skills** (`skills/interface.py`, on top of `skills/manipulation.py` and the new `skills/navigation.py`): `NAVIGATE`, `GRASP`, `PLACE`, plus ground-truth `FIND_BOOK` and `VERIFY`. GRASP handles a book on a table (drag to the edge if needed, edge grasp, carry pose) or in a shelf (pre-grasp 15 cm out, in around the spine, lift 3 cm, pull out, base 10 cm back, carry pose). PLACE takes the leftmost free slot of the compartment, backs the base off 10 cm to swing the book to the pre-place, drives up again, corrects for the book's pose in the fingers, slides in, guarded set-down, withdraws. NAVIGATE drives the three east-west parking lines and the aisle at x = 1.25 m (A* in Phase 6). `SkillResult` moved to `skills/result.py`.
  - **Done-when test passed** (`sim/test_phase5.py`, ground-truth poses, friction grasp): NAVIGATE(table) -> GRASP -> NAVIGATE(S2-A3) -> PLACE -> VERIFY from the reset scene, **10/10** books from all 6 tables (README asks for 8/10; Calculus and Databases are dragged to the edge first, ~3 min simulated each, the rest ~1.5 min). Also two books one after the other into S2-A3 (slots 1 and 2) and Physics out of S2-A1 into S2-A3; no other book moves more than 1 cm.
  - Found on the way: `home()` at the shelf parking reaches into the shelf; it hit the shelf and turned the base by 17 deg (the shelf grasp now goes from the stow pose straight to the pre-grasp). Pulled out 15 cm, a book's back edge is right at the neighbours' protruding spines and the swing to the carry pose hit them; the arm cannot reach 20 cm out at row A, so the base backs off 10 cm for that swing instead.
- [x] **Perception: dataset + YOLOv8n-seg detector, 39 classes** (one per book; `perception/`):
  - `make_dataset.py`: real wrist-camera images (the arm aimed by our IK, robot teleported, no physics), labels from MuJoCo's segmentation render -> YOLO polygons (parts >= 150 px). Table views (60 %): 2-6 random books anywhere on a random table, any yaw, camera 30-50 cm above, tilt <= 25 deg, any roll; base placed so the view is inside the reach. Shelf views (40 %): 1-5 books packed in 1-2 compartments (20 % head down, spine still out), camera 10-30 cm in front. Light intensity and direction randomized. 4000 images (3428 train / 572 val, seeds 0-3, ~10 min in 4 processes), 228-293 labels per book. `perception/data/` is git-ignored (766 MB).
  - `train_yolo.py`: from COCO-pretrained yolov8n-seg, 60 epochs, imgsz 640, batch 16, left-right flips off (mirrored titles never occur). 55 min on the RTX 4060. **Val: mask mAP50 0.981, mask mAP50-95 0.952, box mAP50 0.982, P 0.973 / R 0.94.** Weights `perception/runs/books/weights/best.pt` (git-ignored; re-train to get them).
  - `detect.py`: `BookDetector.detect(robot)` -> per book: mask (eroded 3 px, edge pixels see the floor) + depth -> world points, camera pose from the robot's own estimate (base pose + arm FK + camera mount; matches MuJoCo exactly) -> plane fit, off-plane points dropped -> `cv2.minAreaRect` in the cover / spine plane: centre, yaw (long side), size, `complete` (whole face seen). `aim_camera()` puts the wrist camera at a point (corrects for the camera sitting 6 cm beside the TCP).
  - `check_detect.py` on the real scene (not training data), tables: **11/11 books detected, right class, no false detection.** First look from the parking pose: centre off by 2-4 cm (books cut by the image border or the near part hidden behind the fingers, flagged `complete=False`). **Second look, camera 40 cm straight above the book: centre error <= 0.2 cm, yaw error <= 0.2 deg** for all 8 reachable books. Mid-table books (Calculus, Control, Databases) cannot be looked at from straight above from the normal parking: look again from the closer drag parking.
  - Setup note: torch from `https://download.pytorch.org/whl/cu126` (driver 560.94 supports CUDA <= 12.6; newest there: torch 2.14.1 / torchvision 0.29.1). `pip install ultralytics` replaced it with the CPU build of torch 2.15, so install ultralytics first, then the CUDA wheels with `--force-reinstall --no-deps`. pip 22.3 cannot resolve the deps from the torch index (typing_extensions name), hence `--no-deps`. On a flaky connection pip's 2.6 GB download kept breaking (hash mismatch): download with `curl -C -` (resumes) and check the sha256 against the index.
- [ ] Shelf views in `check_detect.py` (the look pose was out of reach; move the camera further back / tilt it)
- [ ] Skills on detector poses instead of ground truth: GRASP (first look, second look above the book, re-look after each drag stroke), PLACE (free slot from a look at the compartment); in-hand correction stays on ground truth for now. Then `sim/test_phase5.py --perception`, done when >= 8/10.


### Phases 6-10: not started

---

## 9. Risks and fallbacks

| Risk | Plan B |
|---|---|
| Grasp physics unstable | Stay with the attach (weld) grasp and state this clearly in the report |
| RRT takes too long | Use a library planner or a fixed approach trajectory into the shelf |
| Panda cannot reach upper shelf rows | Lower the shelf rows, or add the vertical lift joint |
| LLM gives wrong or invalid plans | Validator, few-shot examples, pre-computed spatial info in the state, retry |
| LLM API latency or cost | Plan once per task, low temperature, log latency; local model only if hardware allows (8 GB VRAM is limiting) |
| Time runs short | Drop optional perception and LiDAR; reduce benchmark to about 30 tasks |
| MuJoCo proves unsuitable | PyBullet as backup; skills are simulator-independent, so only the backends change |

---

## 10. Repository structure

```
ShelfSage/
├── assets/
│   ├── mujoco_menagerie/      # cloned, git-ignored
│   └── fonts/                 # Sedgwick Ave (wall tag) + Bebas Neue (book spines), SIL OFL
├── scene/
│   ├── layout.json            # single source of truth for the library
│   ├── generate_scene.py      # layout -> library.xml + library_meta.json
│   ├── make_textures.py       # procedural textures -> textures/
│   ├── fetch_covers.py        # real book covers -> textures/covers/
│   ├── robot_model.py         # base + Menagerie Panda (MjSpec attach), sensors, home keyframe
│   ├── library.xml            # generated MJCF
│   ├── library_meta.json      # generated positions (tables, shelf slots, books)
│   ├── textures/
│   └── preview_*.png
├── sim/
│   ├── save_joint_names.py    # writes NOTES.md (robot joints, actuators, sites, poses)
│   ├── view_scene.py          # opens the library scene
│   ├── robot.py               # cmd_vel, odometry, LiDAR, RGB-D, go_to, arm/gripper (Phase 3)
│   ├── reach_check.py         # picks the chassis height (Phase 3)
│   ├── test_phase4.py         # Phase 4 done-when test (FK, Jacobian, IK, straight lines)
│   ├── demo_phase3.py         # watch the robot in the viewer (Phases 3-4)
│   ├── demo_pick_place.py     # table -> shelf -> table with the attach grasp (Phase 5 preview)
│   ├── demo_drag_grasp.py     # mid-table book: drag to the edge, grasp, lift (Phase 5)
│   ├── test_phase5.py         # Phase 5 done-when test: table -> S2-A3 with the skills, 10 books
│   └── test_robot.py          # Phase 3 done-when test
├── skills/
│   ├── __init__.py
│   ├── interface.py           # the skills (FIND_BOOK, NAVIGATE, GRASP, PLACE, VERIFY) + SKILLS dictionary
│   ├── manipulation.py        # drag to edge, table / shelf pick, shelf place (Phase 5)
│   ├── navigation.py          # parking poses, lanes + aisle (A* in Phase 6)
│   └── result.py              # SkillResult
├── control/                   # occupancy.py, kinematics.py, ik.py (done); rrt.py, astar.py, mcl.py (Phases 5-6)
├── perception/                # make_dataset.py, train_yolo.py, detect.py, check_detect.py; data/, runs/ git-ignored
├── planner/                   # baseline_planner.py, llm_planner.py, prompts/, schema.json
├── benchmark/                 # tasks.json, run_eval.py, logs/
├── requirements.txt
├── NOTES.md                   # robot joints, actuators, sites, poses (generated)
└── README.md
```

---

## 11. Setup (Windows, PowerShell)

```powershell
# create and activate the virtual environment (if not already done)
python -m venv venv
venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# download the Panda model
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie

# test: run from the project root (the model paths in the scripts are relative)
python sim\view_scene.py
python sim\test_robot.py
python sim\test_phase4.py
```

Expected: `view_scene.py` opens the library with the robot in its home pose; both tests end with `RESULT: all passed`.

Perception (Phase 5, NVIDIA GPU):

```powershell
pip install ultralytics
pip install --force-reinstall --no-deps torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"     # must print True
python perception\make_dataset.py --n 1000 --seed 0            # run seeds 0-3 for 4000 images
python perception	rain_yolo.py                                # ~55 min on an RTX 4060
python perception\check_detect.py                              # detector vs ground truth on the scene
```

Notes:
- Run commands from the project root with the venv active (prompt shows `(venv)`).
- In PowerShell a stuck `>>` prompt means a multi-line command is not finished; press Ctrl+C to cancel.
- When updating dependencies, `pip freeze > requirements.txt` in Windows PowerShell 5.1 writes UTF-16. Use `pip freeze | Out-File -Encoding utf8 requirements.txt` instead.
- **Laptops with two GPUs (Intel + NVIDIA): make Python render on the NVIDIA GPU.** Windows runs Python on the integrated GPU by default; the viewer then runs at about 5 fps instead of about 100 fps (measured: 183 ms vs 10 ms per frame on an RTX 4060). Fix: *Settings -> System -> Display -> Graphics*, add `python.exe` from the base install (e.g. `%LOCALAPPDATA%ProgramsPythonPython311python.exe`) and `venvScriptspython.exe`, set both to *High performance*. Check with: `python -c "import mujoco; from OpenGL import GL; c=mujoco.GLContext(64,64); c.make_current(); print(GL.glGetString(GL.GL_RENDERER))"` (should print the NVIDIA GPU).

---

## 12. How to resume in a new chat

Paste this block, plus this README, into a new chat:

```
I am working on ShelfSage, a course project (Robot Manipulators): an LLM-guided mobile
librarian robot in MuJoCo with a Franka Panda (2-finger gripper) on a virtual holonomic
cart. The LLM plans, conventional controllers execute. Simulator is MuJoCo (not Isaac Sim,
because of an RTX 4060 with 8 GB). Realistic route: known map + LiDAR/odometry + MCL
localization, LiDAR obstacle detection, wrist RGB-D camera with a YOLOv8-seg detector
trained on sim images. Please read the README, treat "Progress tracker" as the
current state, and continue from the first unchecked item. I use Windows + PowerShell.
```

---

## 13. Open questions and to-dos

- **Deadline** for the project (not decided yet)
- **Which LLM / API** to use (affects latency, cost, JSON reliability)
- **Course requirement:** confirm with the professor that switching from Isaac Sim to MuJoCo is acceptable
- **Slide deck updates:**
  - Simulation: Isaac Sim -> MuJoCo
  - Gripper: "3-Point gripper" -> 2-finger parallel gripper
  - Base: "Differential-drive base" -> holonomic / virtual base; navigation: known map + A*
- **Team split** (suggestion): one person on scene, simulator and navigation (Phases 2, 3, 6); the other on arm skills and planners (Phases 4, 5, 7, 8, 9); both on integration and benchmark
- **Later ideas, not part of the 10 phases:** polished demo video, Blender-rendered visuals, 3-finger gripper variant as an extra experiment, full SLAM demo (map building from LiDAR), pose-noise robustness experiment (grasp success vs localization error)

---

## 14. References and useful resources

**Prior work (from the proposal deck):**
- SayCan: *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (LLM high-level planning grounded in executable robot skills)
- PaLM-E: *An Embodied Multimodal Language Model*
- RT-2: *Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*
- SMART-LLM: *Smart Multi-Agent Robot Task Planning using Large Language Models* (https://github.com/SMARTlab-Purdue/SMART-LLM)

**Useful to read (checked briefly, not run):**
- SMART-LLM repo: use the prompt structure and the "skills as code" idea (it uses AI2-THOR and the OpenAI API, so the simulator code is not reusable).
- `carpit680/say-can-open`: open SayCan variant on a tabletop environment; good for understanding the idea, small repo.
- arXiv 2610.14106 (LLM-VLM task planning with Franka in Isaac Sim): separates symbolic planning from metric execution, the same philosophy as this project.
- arXiv 2602.01870 (BTGenBot-2): a 52-task benchmark in Isaac Sim; useful idea for structuring our benchmark later.
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

**Judged not suitable as dependencies:** `Nav2-with-Isaac-ROS-GEMs` (old ROS2 Foxy, different problem), cuRobo wrappers (version mismatch risk and install complexity; the course may expect us to implement IK and Jacobian ourselves).

*Reach, version and availability statements in this README came from general knowledge or quick lookups; verify against current docs before relying on them.*
