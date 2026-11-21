"""LLM planner (Phase 8): Claude turns a command + the world state into a JSON plan.

    planner = LLMPlanner()                       # needs ANTHROPIC_API_KEY (env or .env file)
    out = planner.plan("Put away Linear Algebra", world)
    out.plan          # [Action] or None (refused / gave up)
    out.message       # the model's reason, or why it failed
    out.attempts, out.latency, out.usage

One request per command, made before anything moves (the LLM is not in the control loop):
system prompt (planner/prompts/system.md, fixed, cached) + the world state as JSON
(planner/world.llm_view: names, contents, pre-computed distances / rows / neighbours) + the
command. The reply is constrained to planner/schema.json (structured outputs), so it always
parses. The plan then goes through the same validator as the baseline's; if it is rejected,
the validator's errors go back to the model in the same conversation and it answers again
(up to `retries` times). A refusal is a valid answer (for impossible commands).
"""
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
MODEL = "claude-opus-5-5"


def load_env():
    """Read ANTHROPIC_API_KEY (and other KEY=value lines) from ROOT/.env if not already set.
    .env is git-ignored."""
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class LLMResult:
    plan: list                      # [Action] or None
    message: str
    status: str = ""                # "plan", "refuse", "invalid" (gave up), "error"
    attempts: int = 0
    latency: float = 0.0            # s, all requests
    usage: dict = field(default_factory=dict)
    transcript: list = field(default_factory=list)   # what the model answered each attempt


class LLMPlanner:
    def __init__(self, model=MODEL, effort="medium", retries=2, client=None, fallbacks=True):
        self.model, self.effort, self.retries, self.fallbacks = model, effort, retries, fallbacks
        if client is None:
            import anthropic
            load_env()
            client = anthropic.Anthropic()
        self.client = client

    def _request(self, messages):
        kw = dict(model=self.model, max_tokens=16000,
                  system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                  output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
                  messages=messages)
        if self.fallbacks:     # on a safety-classifier refusal the API retries on another model
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default", **kw)
        return self.client.messages.create(**kw)

    def plan(self, command, world):
        view = json.dumps(W.llm_view(world), separators=(",", ":"))
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "World state:\n" + view, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Command: " + command}]}]
        out = LLMResult(None, "", usage={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
        t0 = time.time()
        for attempt in range(1, self.retries + 2):
            out.attempts = attempt
            try:
                resp = self._request(messages)
            except Exception as e:                       # API / network error: no plan
                out.status, out.message = "error", f"{type(e).__name__}: {e}"
                break
            u = resp.usage
            out.usage["input"] += u.input_tokens or 0
            out.usage["output"] += u.output_tokens or 0
            out.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            out.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            if resp.stop_reason == "refusal":
                out.status, out.message = "error", "the model declined the request (safety refusal)"
                break
            text = next((b.text for b in resp.content if b.type == "text"), "")
            out.transcript.append(text)
            try:
                reply = json.loads(text)
                plan = P.from_json(reply)
            except (ValueError, KeyError) as e:          # should not happen with structured outputs
                feedback = f"Your answer could not be read: {e}. Answer again with JSON matching the schema."
            else:
                if reply["status"] == "refuse":
                    out.status, out.message = "refuse", reply["reason"]
                    break
                v = P.validate(plan, world)
                if v.ok:
                    out.plan, out.status, out.message = plan, "plan", reply["reason"]
                    break
                feedback = ("The plan validator rejected this plan:\n- " + "\n- ".join(v.errors)
                            + "\nFix the plan (or refuse if the command cannot be done) and answer again.")
            out.status, out.message = "invalid", feedback
            messages.append({"role": "assistant", "content": resp.content})   # unchanged
            messages.append({"role": "user", "content": feedback})
        out.latency = round(time.time() - t0, 2)
        return out
