"""Phase 8 done-when test: the LLM planner.

1. Offline (no API key needed, a scripted stand-in for the Claude client): the world view is
   deterministic and holds the pre-computed facts; the prompt and the output schema list exactly
   the five skills; a valid plan is accepted; an invalid plan gets the validator's errors back
   and the corrected plan is accepted; a refusal is passed on; a model that never fixes its plan
   gives up after the retries; an API error is reported, not raised.
2. Live (GEMINI_API_KEY or ANTHROPIC_API_KEY in the environment or in .env; skipped without):
   a. plan level on the whole benchmark (benchmark/tasks.json, failure tasks excluded): the LLM
      must pass every task the baseline passes (the same commands work) and at least 2/3 of the
      tasks the baseline fails (new phrasings, spatial relations, several steps, questions).
   b. exec level: a few of the newly phrased tasks run end to end in the simulator.
   Results: benchmark/results/llm_<effort>_plan.json and sim/output/phase8/summary.json.

Usage (project root, venv active):
    python sim/test_phase8.py --offline           # part 1 only
    python sim/test_phase8.py                     # both parts
    python sim/test_phase8.py --effort high
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "sim/output/phase8"
EXEC_TASKS = ["P01", "P06", "X01", "X03", "M04", "Q04"]


# ---------------------------------------------------------------------------- 1. offline
class FakeClient:
    """Returns scripted replies (JSON strings or exceptions) and records the requests."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.messages = NS(create=self._create)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return NS(stop_reason="end_turn", content=[NS(type="text", text=r)],
                  usage=NS(input_tokens=100, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0))


def offline_checks(world):
    from planner import plan as P
    from planner import world as W
    from planner.llm import SCHEMA, SYSTEM, LLMPlanner
    checks = []
    v1, v2 = json.dumps(W.llm_view(world)), json.dumps(W.llm_view(world))
    view = W.llm_view(world)
    empty = {c: s["distance_to_window_m"] for c, s in view["compartments"].items() if not s["books_left_to_right"]}
    checks.append((f"world view deterministic, {len(v1)} chars", v1 == v2))
    checks.append(("world view: empty compartment nearest the window is S3-A4", min(empty, key=empty.get) == "S3-A4"))
    checks.append(("world view: Kinematics is Physics' right neighbour",
                   view["books"]["Physics"]["right_neighbour"] == "Kinematics"))
    acts = SCHEMA["properties"]["plan"]["items"]["properties"]["action"]["enum"]
    checks.append(("schema and prompt list exactly the five skills",
                   set(acts) == set(P.SIGNATURES) and all(f"- {a}(" in SYSTEM for a in acts)))

    good = json.dumps({"status": "plan", "reason": "ok", "plan": [
        {"action": "FIND_BOOK", "args": ["Machine Learning"]}, {"action": "NAVIGATE", "args": ["Table B"]},
        {"action": "GRASP", "args": ["Machine Learning"]}, {"action": "NAVIGATE", "args": ["S2-A3"]},
        {"action": "PLACE", "args": ["Machine Learning", "S2-A3"]}, {"action": "VERIFY", "args": ["Machine Learning", "S2-A3"]}]})
    bad = json.dumps({"status": "plan", "reason": "forgot to drive", "plan": [
        {"action": "GRASP", "args": ["Machine Learning"]}]})
    refuse = json.dumps({"status": "refuse", "reason": "no such book", "plan": []})

    fc = FakeClient([good])
    r = LLMPlanner(client=fc).plan("cmd", world)
    req = fc.requests[0]
    checks.append(("valid plan accepted in 1 attempt", r.status == "plan" and r.attempts == 1 and len(r.plan) == 6))
    checks.append(("request: model claude-opus-5-5, structured output, cached system prompt, world + command",
                   req["model"] == "claude-opus-5-5" and req["output_config"]["format"]["type"] == "json_schema"
                   and req["system"][0]["cache_control"]["type"] == "ephemeral"
                   and req["messages"][0]["content"][1]["text"] == "Command: cmd"))
    fc = FakeClient([bad, good])
    r = LLMPlanner(client=fc).plan("cmd", world)
    fb = fc.requests[1]["messages"]
    checks.append(("invalid plan -> validator errors sent back -> corrected plan accepted (2 attempts)",
                   r.status == "plan" and r.attempts == 2 and fb[-2]["role"] == "assistant"
                   and "NAVIGATE first" in fb[-1]["content"]))
    r = LLMPlanner(client=FakeClient([refuse])).plan("cmd", world)
    checks.append(("refusal passed on (no plan, reason kept)", r.status == "refuse" and r.plan is None and r.message == "no such book"))
    r = LLMPlanner(client=FakeClient([bad] * 3), retries=2).plan("cmd", world)
    checks.append(("never fixed -> gives up after 1 + 2 retries", r.status == "invalid" and r.plan is None and r.attempts == 3))
    r = LLMPlanner(client=FakeClient([RuntimeError("connection reset")])).plan("cmd", world)
    checks.append(("API error reported, not raised", r.status == "error" and "connection reset" in r.message))

    # Gemini backend, with a scripted HTTP post
    from planner.llm import GeminiBackend

    class Post:
        def __init__(self, replies):
            self.replies, self.calls = list(replies), []

        def __call__(self, url, json=None, timeout=None, headers=None):
            self.calls.append((url, json, headers))
            code, body = self.replies.pop(0)
            return NS(status_code=code, text=body if isinstance(body, str) else "", headers={},
                      json=lambda: body)

    def gem(text):
        return (200, {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
                      "usageMetadata": {"promptTokenCount": 4000, "candidatesTokenCount": 200, "thoughtsTokenCount": 300}})

    post = Post([gem(bad), gem(good)])
    r = LLMPlanner(backend=GeminiBackend(post=post, model="gemini-test")).plan("cmd", world)
    url, body, hdr = post.calls[0]
    checks.append(("gemini: request = generateContent, key in header, system prompt, JSON schema output",
                   url.endswith("/models/gemini-test:generateContent") and "x-goog-api-key" in hdr
                   and body["systemInstruction"]["parts"][0]["text"] == SYSTEM
                   and body["generationConfig"]["responseJsonSchema"] == SCHEMA))
    hist = post.calls[1][1]["contents"]
    checks.append(("gemini: invalid plan -> feedback as a 'user' turn after the 'model' turn -> corrected",
                   r.status == "plan" and r.attempts == 2 and hist[-2]["role"] == "model" and hist[-1]["role"] == "user"
                   and r.usage["output"] == 1000))
    post = Post([(400, "Invalid argument: thinking_level is not supported"), gem(good)])
    r = LLMPlanner(backend=GeminiBackend(post=post, model="gemini-old")).plan("cmd", world)
    checks.append(("gemini: a model without thinkingLevel is retried without it",
                   r.status == "plan" and "thinkingConfig" not in post.calls[1][1]["generationConfig"]))
    r = LLMPlanner(backend=GeminiBackend(post=Post([(403, "API key not valid")]), model="g")).plan("cmd", world)
    checks.append(("gemini: HTTP error reported, not raised", r.status == "error" and "403" in r.message))
    return checks


# ---------------------------------------------------------------------------- 2. live
def live(world, effort):
    from benchmark.run_eval import llm_planner, baseline_planner, run_exec, score_plan
    os.environ["SHELFSAGE_EFFORT"] = effort
    spec = json.loads((ROOT / "benchmark/tasks.json").read_text())
    tasks = [t for t in spec["tasks"] if t["category"] != "failure"]
    rows = []
    from planner.llm import QuotaExhausted
    for t in tasks:
        b = score_plan(t, *baseline_planner(t["command"], world)[:2], world)
        try:
            plan, msg, extra = llm_planner(t["command"], world)
        except QuotaExhausted as e:
            print(f"\n  STOPPED at {t['id']}: {e}. {len(rows)}/{len(tasks)} tasks done; the answers are "
                  "cached (sim/output/llm_cache), run again when the quota resets to continue.")
            return [(f"benchmark incomplete ({len(rows)}/{len(tasks)}): daily quota", False)]
        lres = score_plan(t, plan, msg, world)
        lres.update(extra)
        rows.append((t, b, lres))
        print(f"  {t['id']}  baseline {'ok' if b['success'] else '--'}  llm {'ok' if lres['success'] else 'XX'}"
              f"  {extra['latency']:5.1f} s  {extra['attempts']}x{' (cached)' if extra.get('cached') else ''}"
              f"  {t['command'][:60]}", flush=True)
        if not lres["success"]:
            print(f"        {lres['reason'][:150]}")
            print(f"        plan: {' -> '.join(lres['plan'])[:200]}")
    base_ok = [r for r in rows if r[1]["success"]]
    base_fail = [r for r in rows if not r[1]["success"]]
    same = sum(1 for _, _, l in base_ok if l["success"])
    new = sum(1 for _, _, l in base_fail if l["success"])

    print(f"\n  executing {len(EXEC_TASKS)} new-phrasing tasks in the simulator ...", flush=True)
    sel = [t for t in tasks if t["id"] in EXEC_TASKS]
    with ProcessPoolExecutor(max_workers=3) as ex:
        execs = list(ex.map(run_exec, sel, ["llm"] * len(sel), [False] * len(sel), [False] * len(sel)))
    for t, e in zip(sel, execs):
        print(f"  {'PASS' if e['success'] else 'FAIL'}  {t['id']}  {t['command'][:60]}  {e['reason'][:80]}"
              f"  ({e.get('sim_time', 0):.0f} s sim)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps({"effort": effort, "plan_level": [
        {"id": t["id"], "command": t["command"], "baseline": b["success"], "llm": l["success"], "llm_reason": l["reason"],
         "llm_plan": l["plan"], "latency": l["latency"], "attempts": l["attempts"], "usage": l["usage"]} for t, b, l in rows],
        "exec": [{"id": t["id"], **e} for t, e in zip(sel, execs)]}, indent=1, default=str))
    lat = sorted(l["latency"] for _, _, l in rows)
    return [(f"same commands: LLM passes every task the baseline passes ({same}/{len(base_ok)})", same == len(base_ok)),
            (f"new commands: LLM passes >= 2/3 of the tasks the baseline fails ({new}/{len(base_fail)})",
             new >= 2 * len(base_fail) / 3),
            (f"newly phrased tasks executed in the simulator ({sum(e['success'] for e in execs)}/{len(execs)})",
             all(e["success"] for e in execs)),
            (f"planning latency: median {lat[len(lat) // 2]:.1f} s, max {lat[-1]:.1f} s (logged)", True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    import mujoco
    from planner import world as W
    from planner.llm import load_env
    from sim.robot import Robot
    m = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    world = W.snapshot(Robot(m, mujoco.MjData(m), pose_source="true"))
    final = []
    print("1. offline checks")
    for name, ok in offline_checks(world):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        final.append((name, ok))
    if not a.offline:
        load_env()
        from planner.llm import default_backend
        backend = default_backend()
        key = "GEMINI_API_KEY" if backend == "gemini" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key):
            print(f"\n2. live checks skipped: no {key} (environment or .env)")
        else:
            model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite") if backend == "gemini" else "claude-opus-5-5"
            print(f"\n2. live checks ({backend}: {model}, effort {a.effort})")
            t0 = time.time()
            res = live(world, a.effort)
            print(f"  ({time.time() - t0:.0f} s)")
            for name, ok in res:
                print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            final += res
    print("\nRESULT:", "all passed" if all(ok for _, ok in final) else "FAILED")


if __name__ == "__main__":
    main()
