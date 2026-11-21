"""LLM planner (Phase 8): an LLM turns a command + the world state into a JSON plan.

    planner = LLMPlanner()                       # backend from .env / environment, see below
    out = planner.plan("Put away Linear Algebra", world)
    out.plan          # [Action] or None (refused / gave up)
    out.message       # the model's reason, or why it failed
    out.attempts, out.latency, out.usage

Two backends, same prompt, schema and retry loop:
    gemini   Google Gemini API, free tier (key from aistudio.google.com): GEMINI_API_KEY,
             model GEMINI_MODEL (default gemini-3.5-flash-lite). REST generateContent with
             responseMimeType application/json + responseJsonSchema. Free-tier requests may be
             used by Google to improve its products (only library commands are sent).
    claude   Anthropic Claude API (paid): ANTHROPIC_API_KEY, model claude-opus-5-5, structured
             outputs, cached system prompt + world state, server-side fallback on refusals.
LLM_BACKEND chooses (default: gemini if GEMINI_API_KEY is set, else claude). Keys come from
the environment or from a git-ignored .env file in the project root.

One request per command, made before anything moves (the LLM is not in the control loop):
system prompt (planner/prompts/system.md) + the world state as JSON (planner/world.llm_view:
names, contents, pre-computed distances / rows / neighbours) + the command. The reply must match
planner/schema.json. The plan then goes through the same validator as the baseline's; if it is
rejected, the validator's errors go back to the model in the same conversation and it answers
again (up to `retries` times). A refusal is a valid answer (for impossible commands).
"""
import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from planner import plan as P
from planner import world as W

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SYSTEM = (HERE / "prompts/system.md").read_text(encoding="utf-8")
SCHEMA = json.loads((HERE / "schema.json").read_text())
CLAUDE_MODEL = "claude-opus-5-5"
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_THINKING = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "XHIGH", "max": "XHIGH"}


def load_env():
    """Read KEY=value lines from ROOT/.env into the environment (if not already set). .env is
    git-ignored."""
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def default_backend():
    load_env()
    b = os.environ.get("LLM_BACKEND")
    if b:
        return b
    return "gemini" if os.environ.get("GEMINI_API_KEY") else "claude"


@dataclass
class LLMResult:
    plan: list                      # [Action] or None
    message: str
    status: str = ""                # "plan", "refuse", "invalid" (gave up), "error"
    attempts: int = 0
    latency: float = 0.0            # s, all requests
    usage: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)   # what the model answered each attempt
    backend: str = ""
    model: str = ""


def command_text(command, context=None):
    """The command, plus what happened so far when planning again after a failure (Phase 9)."""
    if not context:
        return "Command: " + command
    return f"Command: {command}\n\nReplanning after a failure:\n{context}"


# ---------------------------------------------------------------------------- backends
class QuotaExhausted(RuntimeError):
    """The free tier's daily request limit for this model is used up (resets after a day)."""


class Reply:
    def __init__(self, text, usage, refused=False, history_item=None):
        self.text, self.usage, self.refused, self.history_item = text, usage, refused, history_item


class ClaudeBackend:
    name = "claude"

    def __init__(self, effort="medium", client=None, fallbacks=True, model=CLAUDE_MODEL):
        self.model, self.effort, self.fallbacks = model, effort, fallbacks
        if client is None:
            import anthropic
            load_env()
            client = anthropic.Anthropic()
        self.client = client

    def start(self, world_text, command, context=None):
        return [{"role": "user", "content": [
            {"type": "text", "text": "World state:\n" + world_text, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": command_text(command, context)}]}]

    def ask(self, messages):
        kw = dict(model=self.model, max_tokens=16000,
                  system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                  output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                  messages=messages)
        if self.fallbacks:     # on a safety-classifier refusal the API retries on another model
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default", **kw)
        else:
            resp = self.client.messages.create(**kw)
        u = resp.usage
        usage = {"input": u.input_tokens or 0, "output": u.output_tokens or 0,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0}
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return Reply(text, usage, resp.stop_reason == "refusal",
                     {"role": "assistant", "content": resp.content})     # unchanged, for the retry

    def feedback(self, messages, reply, text):
        messages.append(reply.history_item)
        messages.append({"role": "user", "content": text})


class GeminiBackend:
    name = "gemini"

    def __init__(self, effort="medium", model=None, post=None):
        load_env()
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
        self.effort = effort
        self.thinking = True
        if post is None:
            import requests
            post = requests.post
        self.post = post

    def start(self, world_text, command, context=None):
        return [{"role": "user", "parts": [{"text": "World state:\n" + world_text},
                                           {"text": command_text(command, context)}]}]

    def ask(self, contents):
        body = {"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": contents,
                "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": SCHEMA}}
        if self.thinking:
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": GEMINI_THINKING.get(self.effort, "MEDIUM")}
        for _ in range(4):                       # free tier: back off on 429 / 503
            r = self.post(GEMINI_URL.format(model=self.model), json=body, timeout=180,
                          headers={"x-goog-api-key": self.key, "Content-Type": "application/json"})
            if r.status_code == 400 and self.thinking and "thinking" in r.text.lower():
                self.thinking = False            # an older model without thinkingLevel
                body["generationConfig"].pop("thinkingConfig", None)
                continue
            if r.status_code == 429 and "PerDay" in r.text:
                raise QuotaExhausted(f"{self.model}: free-tier requests per day used up")
            if r.status_code in (429, 500, 503):
                time.sleep(float(r.headers.get("retry-after", 20)))
                continue
            break
        if r.status_code != 200:
            raise RuntimeError(f"Gemini API {r.status_code}: {r.text[:300]}")
        data = r.json()
        um = data.get("usageMetadata", {})
        usage = {"input": um.get("promptTokenCount", 0) - um.get("cachedContentTokenCount", 0),
                 "output": um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0),
                 "cache_read": um.get("cachedContentTokenCount", 0), "cache_write": 0}
        cands = data.get("candidates") or []
        if not cands:
            return Reply("", usage, True)        # blocked (promptFeedback)
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return Reply(text, usage, cands[0].get("finishReason") == "SAFETY",
                     {"role": "model", "parts": copy.deepcopy(parts)})

    def feedback(self, contents, reply, text):
        contents.append(reply.history_item)
        contents.append({"role": "user", "parts": [{"text": text}]})


# ---------------------------------------------------------------------------- planner
class LLMPlanner:
    def __init__(self, backend=None, effort="medium", retries=2, client=None):
        if backend is None:
            backend = "claude" if client is not None else default_backend()
        if isinstance(backend, str):
            backend = ClaudeBackend(effort, client) if backend == "claude" else GeminiBackend(effort)
        self.backend, self.retries = backend, retries

    def plan(self, command, world, context=None):
        """context: what happened to earlier plans for this command (planner/replan.py)."""
        b = self.backend
        conv = b.start(json.dumps(W.llm_view(world), separators=(",", ":")), command, context)
        out = LLMResult(None, "", usage={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                        backend=b.name, model=b.model)
        t0 = time.time()
        for attempt in range(1, self.retries + 2):
            out.attempts = attempt
            try:
                reply = b.ask(conv)
            except QuotaExhausted:
                raise                                    # the caller decides (stop the run)
            except Exception as e:                       # API / network error: no plan
                out.status, out.message = "error", f"{type(e).__name__}: {e}"
                break
            for k in out.usage:
                out.usage[k] += reply.usage.get(k, 0)
            if reply.refused:
                out.status, out.message = "error", "the model declined the request (safety refusal)"
                break
            out.transcript.append(reply.text)
            try:
                data = json.loads(reply.text)
                if data.get("status") not in ("plan", "refuse"):
                    raise ValueError("status must be 'plan' or 'refuse'")
                plan = P.from_json(data)
            except (ValueError, KeyError, AttributeError) as e:
                fb = f"Your answer could not be read: {e}. Answer again with JSON matching the schema."
            else:
                if data["status"] == "refuse":
                    out.status, out.message = "refuse", data.get("reason", "")
                    break
                v = P.validate(plan, world)
                if v.ok:
                    out.plan, out.status, out.message = plan, "plan", data.get("reason", "")
                    break
                fb = ("The plan validator rejected this plan:\n- " + "\n- ".join(v.errors)
                      + "\nFix the plan (or refuse if the command cannot be done) and answer again.")
            out.status, out.message = "invalid", fb
            b.feedback(conv, reply, fb)
        out.latency = round(time.time() - t0, 2)
        return out


def main():
    """python -m planner.llm --models   : list the Gemini models this key can use
       python -m planner.llm "<command>" : plan one command (no robot moves)"""
    import sys
    load_env()
    if sys.argv[1:] == ["--models"]:
        import requests
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", timeout=60,
                         headers={"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "")})
        if r.status_code != 200:
            sys.exit(f"Gemini API {r.status_code}: {r.text[:300]}")
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(m["name"].split("/")[-1], "-", m.get("displayName", ""))
        return
    import mujoco
    from sim.robot import Robot
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene/library.xml"))
    world = W.snapshot(Robot(model, mujoco.MjData(model), pose_source="true"))
    out = LLMPlanner().plan(" ".join(sys.argv[1:]), world)
    print(f"{out.backend} {out.model}: {out.status} after {out.attempts} attempt(s), {out.latency} s, {out.usage}")
    print("reason:", out.message)
    print("plan  :", P.to_text(out.plan) if out.plan else "-")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
