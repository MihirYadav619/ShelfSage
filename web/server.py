"""ShelfSage desk: a local web page to give the robot commands and watch it work.

    python web/server.py                 # then open http://localhost:8000
    python web/server.py --port 8080 --speed 3

One simulator thread owns the robot (search mode: it finds loose books with its camera, as in
run_demo.py) and renders the scene off-screen while it moves; the page shows those frames as an
MJPEG stream and gets the plan, the steps and the robot's messages as server-sent events. The
library keeps its state between commands; "Reset" starts from the original scene. Standard
library only (http.server), no extra packages. Local use only: it binds to 127.0.0.1.

The LLM planner needs GEMINI_API_KEY (or ANTHROPIC_API_KEY) as for run_demo.py (README, Phase 8).
"""
import argparse
import io
import json
import math
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
FONTS = ROOT / "assets/fonts"
sys.path.insert(0, str(ROOT))

FRAME_W, FRAME_H = 960, 600
FRAME_FPS = 12.0                     # rendered frames per wall-clock second while the robot moves
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".ttf": "font/ttf", ".svg": "image/svg+xml",
         ".pdf": "application/pdf"}


# ---------------------------------------------------------------------------- events
class Hub:
    """Fan-out of events (server-sent events) and of the latest camera frame."""

    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.frame_cv = threading.Condition()
        self.history = []                # events of the current command, for late joiners

    def emit(self, kind, **data):
        ev = {"type": kind, "t": time.time(), **data}
        with self.lock:
            if kind == "command":
                self.history = []
            self.history.append(ev)
            for q in list(self.clients):
                q.put(ev)

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            for ev in self.history:
                q.put(ev)
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def put_frame(self, jpeg):
        with self.frame_cv:
            self.frame, self.frame_id = jpeg, self.frame_id + 1
            self.frame_cv.notify_all()


HUB = Hub()


# ---------------------------------------------------------------------------- simulator
class FrameViewer:
    """Stands in for MuJoCo's passive viewer (Robot.viewer): the robot calls sync() at up to
    60 Hz of simulated time and paces itself to real time; this renders at most FRAME_FPS
    frames per wall second into JPEGs for the page."""

    def __init__(self, sim):
        self.sim = sim
        self.last = 0.0

    def is_running(self):
        return True

    def sync(self):
        now = time.perf_counter()
        if now - self.last >= 1.0 / FRAME_FPS:
            self.last = now
            self.sim.render()


class Sim(threading.Thread):
    def __init__(self, speed, perception=False):
        super().__init__(daemon=True)
        self.speed, self.perception = speed, perception
        self.jobs = queue.Queue()
        self.busy = False
        self.camera = "follow"
        self.robot = None
        self.renderer = None
        self.cam = None
        self.status = "starting"
        self.ready = threading.Event()

    # -------------------------------------------------------------- set-up
    def _new_robot(self):
        from sim.run_command import make_robot
        self.robot = make_robot(truth=False, perception=self.perception, search=True)
        self.robot.viewer, self.robot.speed = FrameViewer(self), self.speed
        import mujoco
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = mujoco.Renderer(self.robot.m, FRAME_H, FRAME_W)
        self.cam = mujoco.MjvCamera()
        self._aim(snap=True)
        self.robot.wait(0.3)
        self.render(snap=True)

    def _aim(self, snap=False):
        """Camera: 'overview' (the whole room from the open south side) or 'follow' (behind and
        above the robot, eased so it does not shake)."""
        c = self.cam
        if self.camera == "overview":
            target = (0.0, -0.6, 0.6, 7.6, 90.0, -42.0)
        else:
            x, y, yaw = self.robot.true_pose()
            ahead = 0.8
            target = (x + ahead * math.cos(yaw), y + ahead * math.sin(yaw), 0.75, 3.4,
                      math.degrees(yaw) + 25.0, -28.0)
        cur = (c.lookat[0], c.lookat[1], c.lookat[2], c.distance, c.azimuth, c.elevation)
        k = 1.0 if snap else 0.18
        az = cur[4] + ((target[4] - cur[4] + 180.0) % 360.0 - 180.0) * k
        c.lookat[:] = [cur[i] + (target[i] - cur[i]) * k for i in range(3)]
        c.distance = cur[3] + (target[3] - cur[3]) * k
        c.azimuth, c.elevation = az, cur[5] + (target[5] - cur[5]) * k

    def render(self, snap=False):
        from PIL import Image
        self._aim(snap)
        self.renderer.update_scene(self.robot.d, self.cam)
        img = self.renderer.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=82)
        HUB.put_frame(buf.getvalue())

    # -------------------------------------------------------------- state for the page
    def state(self):
        from planner import world as W
        from skills import search
        r = self.robot
        truth = W.snapshot(r, truth=True)
        memory = getattr(r, "memory", {}) or {}
        books = []
        for bid, info in W.META["books"].items():
            m = memory.get(bid)
            known = "held" if r.holding == bid else (search.target_of(m) if m else None)
            books.append({"id": bid, "title": info["title"], "home": info["home"],
                          "known": known, "at": truth["books"][bid]["at"]})
        x, y, _ = r.get_pose()
        return {"books": sorted(books, key=lambda b: b["title"]),
                "robot": {"at": truth["robot"]["at"], "holding": W.META["books"][r.holding]["title"] if r.holding else None,
                          "pose": [round(x, 2), round(y, 2)]},
                "sim_time": round(r.d.time, 1), "busy": self.busy, "camera": self.camera,
                "status": self.status}

    # -------------------------------------------------------------- jobs
    def run(self):
        try:
            self._new_robot()
            self.status = "idle"
        except Exception as e:
            self.status = f"could not start the simulator: {e}"
            traceback.print_exc()
        self.ready.set()
        HUB.emit("state", state=self.state() if self.robot else {})
        while True:
            job = self.jobs.get()
            self.busy = True
            try:
                if job["kind"] == "command":
                    self._command(job["command"], job["planner"])
                elif job["kind"] == "reset":
                    self.status = "resetting"
                    HUB.emit("reset")
                    self._new_robot()
            except Exception as e:
                traceback.print_exc()
                HUB.emit("error", message=f"{type(e).__name__}: {e}")
            finally:
                self.busy = False
                self.status = "idle"
                self.render(snap=job["kind"] == "noop")
                HUB.emit("state", state=self.state())

    def _command(self, command, planner_name):
        from benchmark.run_eval import PLANNERS
        from planner.replan import run_with_replanning
        self.status = "planning"
        HUB.emit("command", command=command, planner=planner_name)
        attempt = {"n": 0}

        def planner(cmd, world, context):
            attempt["n"] += 1
            self.status = "planning" if context is None else "replanning"
            HUB.emit("planning", attempt=attempt["n"], replan=context is not None)
            t0 = time.time()
            try:
                plan, msg, _ = PLANNERS[planner_name](cmd, world, context)
            except Exception as e:                   # API error, daily quota, ...
                plan, msg = None, f"{type(e).__name__}: {e}"
            HUB.emit("plan", attempt=attempt["n"], replan=context is not None, reason=msg,
                     steps=[str(a) for a in plan] if plan else [], seconds=round(time.time() - t0, 1))
            self.status = "working" if plan else "idle"
            return plan, msg

        def on_step(i, action):
            HUB.emit("step", attempt=attempt["n"], index=i, action=str(action), sim_time=round(self.robot.d.time, 1))
            HUB.emit("state", state=self.state())

        last = {"t": 0.0}

        def say(text):
            text = text.strip()
            if not text or text.startswith("["):
                return
            now = time.time()
            HUB.emit("log", text=text, sim_time=round(self.robot.d.time, 1))
            if now - last["t"] > 2.0:
                last["t"] = now
                HUB.emit("state", state=self.state())

        rec = run_with_replanning(command, planner, self.robot, max_replans=2, on_step=on_step, say=say)
        HUB.emit("result", success=rec.success, message=rec.message, sim_time=rec.sim_time,
                 attempts=[{"plan": a["plan"], "steps": a["steps"], "error": a["error"]} for a in rec.attempts])


# ---------------------------------------------------------------------------- http
SIM = None


class Handler(BaseHTTPRequestHandler):
    server_version = "ShelfSage"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not path.is_file():
            return self._send(404, {"error": "not found"})
        self._send(200, path.read_bytes(), TYPES.get(path.suffix, "application/octet-stream"))

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if p.startswith("/static/"):
            f = (STATIC / p[len("/static/"):]).resolve()
            return self._file(f) if STATIC in f.parents else self._send(404, {})
        if p == "/fonts/bebas.ttf":
            return self._file(FONTS / "BebasNeue-Regular.ttf")
        if p == "/catalog.pdf":
            return self._file(ROOT / "Book_Catalog.pdf")
        if p == "/api/state":
            SIM.ready.wait(120)
            return self._send(200, SIM.state() if SIM.robot else {"status": SIM.status})
        if p == "/api/history":
            with HUB.lock:
                return self._send(200, list(HUB.history))
        if p == "/frame.jpg":
            return self._send(200, HUB.frame or b"", "image/jpeg")
        if p == "/stream.mjpg":
            return self._stream()
        if p == "/events":
            return self._events()
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad JSON"})
        if self.path == "/api/command":
            cmd = str(body.get("command", "")).strip()[:400]
            planner = body.get("planner", "llm")
            if not cmd:
                return self._send(400, {"error": "empty command"})
            if planner not in ("llm", "baseline"):
                return self._send(400, {"error": "planner must be llm or baseline"})
            if SIM.busy or not SIM.jobs.empty():
                return self._send(409, {"error": "the robot is still busy with the last request"})
            SIM.busy = True
            SIM.jobs.put({"kind": "command", "command": cmd, "planner": planner})
            return self._send(202, {"ok": True})
        if self.path == "/api/reset":
            if SIM.busy:
                return self._send(409, {"error": "wait until the robot has finished"})
            SIM.busy = True
            SIM.jobs.put({"kind": "reset"})
            return self._send(202, {"ok": True})
        if self.path == "/api/camera":
            mode = body.get("mode")
            if mode not in ("follow", "overview"):
                return self._send(400, {"error": "mode must be follow or overview"})
            SIM.camera = mode
            if not SIM.busy:                       # re-render now (the sim thread is idle)
                SIM.jobs.put({"kind": "noop"})
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seen = -1
        try:
            while True:
                with HUB.frame_cv:
                    HUB.frame_cv.wait_for(lambda: HUB.frame_id != seen and HUB.frame is not None, timeout=10)
                    frame, seen = HUB.frame, HUB.frame_id
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = HUB.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            HUB.unsubscribe(q)


def main():
    global SIM
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speed", type=float, default=2.0, help="real-time factor of the simulation")
    ap.add_argument("--perception", action="store_true", help="grasp poses from the wrist camera too")
    a = ap.parse_args()
    from planner.llm import load_env
    load_env()
    SIM = Sim(a.speed, a.perception)
    SIM.start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"ShelfSage desk: http://localhost:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
