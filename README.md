# ShelfSage

**LLM-Guided Autonomous Mobile Librarian Robot**
*Language-guided mobile manipulation in a simulated library (MuJoCo + Franka Panda)*

- **Course:** Robot Manipulators (IIT Mandi), course project
- **Team:** Chirag, Mihir
- **Status:** Phases 1 and 2 of 10 complete; Phase 3 next (see [Progress](#8-progress-tracker))
- **Note:** This is the *initial* idea. It can be simplified later if parts become too hard (see [Fallbacks](#9-risks-and-fallbacks)).

> This README is written so that the whole project can be resumed from scratch, for example in a new chat with an AI assistant. See [How to resume](#12-how-to-resume-in-a-new-chat).

---

## 1. Project in one paragraph

We are building a **simulated mobile manipulator** (a cart with a Franka Panda arm) that receives a natural-language instruction such as *"Find the Machine Learning book on Table B and place it in shelf A3"* and carries it out on its own inside a simulated library. A **pretrained LLM acts as the high-level task planner**: it turns the instruction into a structured action sequence. **Low-level control stays conventional and verifiable** (A* navigation, inverse kinematics, Jacobian control, RRT, grasp and place routines). The LLM never touches the joints.

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
     A* + base       IK/Jacobian  attach grasp (contact grasp = stretch)
     waypoints       + RRT-Connect
        |
        v
   MuJoCo simulation (library scene, Panda on a cart)
        |
        v
   VERIFY (simulator ground truth) --> on failure: send new state to LLM and replan
```

Example LLM output:

```
FIND_BOOK -> NAVIGATE(Table B) -> GRASP(Book) -> NAVIGATE(Shelf A3) -> PLACE(Book, A3) -> VERIFY()
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
- **Base ("cart"):** a **virtual holonomic base** built from x, y and yaw joints under the arm, driven to waypoints by a simple P-controller. This replaces "differential-drive base" from the deck. Real wheel physics is not simulated.
- **Optional:** a vertical lift (extra prismatic z joint) between base and arm, to reach upper shelf rows. Panda reach is only about 85 cm.
- **Sensors:** RGB-D camera (MuJoCo camera) and optionally LiDAR via ray casting. Both are optional for the core pipeline.

### 4.3 Libraries and tools

- Python virtual environment (`venv`), packages: `mujoco`, `numpy`, `scipy`, `imageio` (pinned in `requirements.txt`)
- LLM: API-based model returning JSON (final choice still open)
- Git for version control

### 4.4 Simplifications chosen (hardest parts made easier)

| Hard part | Simplification |
|---|---|
| Mobile manipulation (base + arm coordination) | Fixed **parking pose** in front of each table and shelf; the base stops, then the arm works. Base and arm never move together. |
| Grasping | Start with ground-truth object poses and an **"attach" (weld) grasp**. Contact-based friction grasp is a stretch goal. |
| SLAM / Nav2 | **Known map** from the layout file + A* waypoint navigation. SLAM only as an optional separate demo. |
| RRT | Use RRT-Connect only where needed (entering the shelf). Fallback: hand-designed approach trajectory. |
| Perception | Ground truth first. RGB-D detection is optional future work. |
| Shelf extraction | Leave gaps between books, spines facing out, pull straight out. |

Easy and not needing simplification: IK, Jacobian control (numpy), PID, collision checks, LLM API call and JSON schema, the action library, world-state JSON, the rule-based baseline, VERIFY from ground truth.

---

## 5. Simulated library environment (built in Phase 2)

Everything is defined in [`scene/layout.json`](scene/layout.json) and generated by `scene/generate_scene.py` into `scene/library.xml` (MJCF) and `scene/library_meta.json` (positions of tables, shelf slots and books; the starting point for the world-state JSON).

- Room 8 m x 6 m, wall height 2.5 m; south wall collides but is hidden (geom group 4) so the cameras can see in
- Window on the east wall at x = 3.97 (used for commands like "nearest to the window")
- 6 tables (A to F), top at z = 0.75 m, holding up to 3 flat books each
- 1 bookshelf at (0, 2.7): 2 rows (A, B) x 4 columns, compartments A1-A4 (surface z = 0.5 m) and B1-B4 (z = 0.9 m), top at 1.3 m
- 13 books (id, title, colour, location): 11 on tables, 2 standing in the shelf (Physics in A1, Statistics in B2)
- Robot start pose (0, -2.5, 0)
- Books are free bodies (`freejoint`) with one invisible box collision proxy (geom group 3) plus non-colliding visual covers, spine and pages; tables and shelf are fixed
- Procedural textures (`scene/make_textures.py`), cameras `overview` and `cutaway`

Commands (project root, venv active):

```powershell
python scene\generate_scene.py             # regenerate library.xml + library_meta.json
python scene\generate_scene.py --check     # also simulate 10 s and check books stay put
python scene\generate_scene.py --preview   # also render scene\preview_*.png
python sim\view_scene.py                   # open the scene in the viewer
```

---

## 6. Skill interface

Defined in `skills/interface.py` (stubs written, bodies still `NotImplementedError`):

| Skill | Purpose | Implemented in phase |
|---|---|---|
| `FIND_BOOK(title)` | Look up which table or shelf holds the book (world-state lookup) | 7 |
| `NAVIGATE(target)` | Drive the base to the parking pose of a table or shelf | 6 |
| `GRASP(obj)` | Pick up a book with the 2-finger gripper | 5 |
| `PLACE(obj, shelf)` | Put the held book in a shelf compartment | 5 |
| `VERIFY(obj, shelf)` | Check with ground truth that the book is inside the compartment | 9 |

Every skill returns a `SkillResult(success, message, data)`, so a failure and its reason can be sent to the LLM for replanning. The `SKILLS` dictionary maps action names to functions; the LLM may only call actions in this dictionary.

---

## 7. The 10-phase plan

Each phase has a "Done when" check. Phases 2, 3 and 4 can be split between two people.

### Phase 1: Setup and foundations
- Install MuJoCo, load the Panda from Menagerie, create repo and folders, define the skill interface.
- **Done when:** the Panda appears in the viewer and a joint can be moved from code.

### Phase 2: Library scene
- Write the layout file and `generate_scene.py` that outputs the MJCF (floor, lights, tables, shelf with compartments, books, window).
- **Done when:** the scene loads and no book falls or jitters during a 10-second simulation.

### Phase 3: Robot assembly (cart + arm)
- Virtual holonomic base (x, y, yaw), Panda on top in one XML, base P-controller, arm and gripper control, state-reading helper, optional lift.
- **Done when:** the robot drives to a table, stops, and the arm reaches over the table surface.

### Phase 4: Arm control (FK, IK, Jacobian)
- Forward kinematics (checked against MuJoCo), Jacobian, differential IK with damped least squares and null-space joint-limit avoidance, Cartesian straight-line moves, gripper open/close.
- **Done when:** the gripper reaches 10 random reachable poses with about 1-2 cm accuracy.

### Phase 5: Grasp and place skills
- `GRASP`: pre-grasp pose, descend, close, lift (attach grasp first). `PLACE`: pre-place in front of compartment, insert, release, retract. RRT-Connect for entering the shelf. Small success checks after each step.
- **Done when:** with the robot standing still, a book moves from a table to shelf A3 correctly at least 8 of 10 times.

### Phase 6: Navigation
- Occupancy grid from the layout, obstacle inflation, A*, smoothed waypoints, parking pose per table and shelf.
- **Done when:** the robot reaches every table and shelf without collisions and the arm can reach from each parking pose.

### Phase 7: Skills integration and baseline planner
- Executor that runs an action list step by step; rule-based parser for fixed templates; logging of every step (time, success).
- **Done when:** 5-10 fixed commands run end to end, from navigation to placement.

### Phase 8: LLM planner
- Skill schema (JSON), world-state JSON with pre-computed spatial facts, prompt with 2-3 examples, low temperature, validator with retry, plan once before execution, latency and plan logging.
- **Done when:** the same commands work via the LLM and newly phrased commands (which the baseline cannot handle) also produce correct plans.

### Phase 9: Verify, failure injection and replanning
- `VERIFY` from ground truth, injected failures (dropped book, full target shelf, blocked path), replanning with the new state and error sent back to the LLM (retry limit 2-3), same failures through the baseline. Optional: RGB-D perception instead of ground truth.
- **Done when:** the LLM planner recovers after at least 2-3 different injected failures.

### Phase 10: Benchmark, evaluation and report
- Benchmark task set (single-step, multi-step, different phrasings, spatial relations, injected failures), both planners on the same tasks, metrics and statistics finalized at this stage, plots, success/failure examples, reproducible `run_demo.py`, final report and updated slide deck.
- **Done when:** results for both planners are clear in a table and at least 2-3 plots.

### If time runs short
Cut first: optional perception, LiDAR simulation, contact-based grasping. Do **not** cut Phases 1-8 (ground-truth version) or verify and replanning in Phase 9, since that is the core story.

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
- [x] Test script `sim/test_panda.py` runs: Panda shows in the viewer and a joint moves
- [x] Skill interface written: `skills/__init__.py` and `skills/interface.py` (stubs)
- [x] Joint names and actuator count saved into `NOTES.md` (via `sim/save_joint_names.py`)
- [x] `requirements.txt` committed (UTF-8)
- [x] First Git commit made

### Phase 2: Library scene (done)

- [x] `scene/layout.json`: room, window, 6 tables, shelf (2 x 4 compartments), 13 books, robot start
- [x] `scene/make_textures.py`: procedural floor, wood, shelf and plaster textures
- [x] `scene/generate_scene.py`: writes `library.xml` + `library_meta.json`, with `--check` and `--preview`
- [x] `sim/view_scene.py`: opens the scene in the viewer
- [x] Done-when check passed: `--check` shows all 13 books stable over 10 s (each settles 0.11 cm, the 1 mm spawn gap, then stays put)

### Phase 3: Robot assembly (next)

- [ ] Choose cart height so the Panda reaches the table tops (0.75 m) and both shelf rows (0.5 m, 0.9 m) before adding the optional lift
- [ ] Holonomic base (x, y, yaw joints) with the Panda mounted, inserted into the library scene

### Phases 4-10: not started

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
│   └── mujoco_menagerie/      # cloned, git-ignored
├── scene/
│   ├── layout.json            # single source of truth for the library
│   ├── generate_scene.py      # layout -> library.xml + library_meta.json
│   ├── make_textures.py       # procedural textures -> textures/
│   ├── library.xml            # generated MJCF
│   ├── library_meta.json      # generated positions (tables, shelf slots, books)
│   ├── textures/
│   └── preview_*.png
├── sim/
│   ├── test_panda.py          # Phase 1 viewer test
│   ├── save_joint_names.py    # writes NOTES.md
│   └── view_scene.py          # opens the library scene
├── skills/
│   ├── __init__.py
│   └── interface.py           # skill stubs + SKILLS dictionary
├── control/                   # ik.py, jacobian.py, rrt.py, astar.py (Phases 4-6)
├── planner/                   # baseline_planner.py, llm_planner.py, prompts/, schema.json
├── benchmark/                 # tasks.json, run_eval.py, logs/
├── requirements.txt
├── NOTES.md                   # joint names, misc notes
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
python sim\test_panda.py
python sim\view_scene.py
```

Expected: `test_panda.py` prints joint names and actuator count, and a viewer window opens with the Panda; its first joint swings slowly. `view_scene.py` opens the library.

Notes:
- Run commands from the project root with the venv active (prompt shows `(venv)`).
- In PowerShell a stuck `>>` prompt means a multi-line command is not finished; press Ctrl+C to cancel.
- When updating dependencies, `pip freeze > requirements.txt` in Windows PowerShell 5.1 writes UTF-16. Use `pip freeze | Out-File -Encoding utf8 requirements.txt` instead.

---

## 12. How to resume in a new chat

Paste this block, plus this README, into a new chat:

```
I am working on ShelfSage, a course project (Robot Manipulators): an LLM-guided mobile
librarian robot in MuJoCo with a Franka Panda (2-finger gripper) on a virtual holonomic
cart. The LLM plans, conventional controllers execute. Simulator is MuJoCo (not Isaac Sim,
because of an RTX 4060 with 8 GB). Please read the README, treat "Progress tracker" as the
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
- **Later ideas, not part of the 10 phases:** polished demo video, Blender-rendered visuals, 3-finger gripper variant as an extra experiment, RGB-D perception

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
