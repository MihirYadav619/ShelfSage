// ShelfSage desk: talks to web/server.py (POST /api/*, server-sent events on /events).
(() => {
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  const cmd = $("cmd"), fileBtn = $("file"), msg = $("slip-msg");
  let busy = false;
  let attempts = {};           // attempt number -> {box, rows: [tr], reason}
  let current = null;          // current attempt number

  // ------------------------------------------------------------------ slip
  let slipNo = Number(localStorage.getItem("shelfsage-slip") || 1);
  const showSlipNo = () => { $("slip-no").textContent = String(slipNo).padStart(4, "0"); };
  showSlipNo();

  async function post(path, body) {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify(body || {}) });
    let data = {};
    try { data = await r.json(); } catch (_) { /* empty */ }
    return { ok: r.ok, data };
  }

  $("slip").addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = cmd.value.trim();
    if (!text) { msg.textContent = "Write a request first."; cmd.focus(); return; }
    const planner = document.querySelector('input[name="planner"]:checked').value;
    msg.textContent = "";
    setBusy(true);
    const r = await post("/api/command", { command: text, planner });
    if (!r.ok) { msg.textContent = r.data.error || "The desk did not accept the request."; setBusy(false); return; }
    slipNo += 1;
    localStorage.setItem("shelfsage-slip", slipNo);
  });

  cmd.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("slip").requestSubmit();
  });

  document.querySelectorAll(".try").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    cmd.value = a.textContent;
    cmd.focus();
  }));

  $("reset").addEventListener("click", async () => {
    if (busy) { msg.textContent = "Wait until the robot has finished."; return; }
    msg.textContent = "Putting every book back where it started...";
    setBusy(true);
    const r = await post("/api/reset");
    if (!r.ok) { msg.textContent = r.data.error || "Could not reset."; setBusy(false); }
  });

  function setBusy(b) {
    busy = b;
    fileBtn.disabled = b;
    fileBtn.textContent = b ? "Robot at work" : "File request";
  }

  // ------------------------------------------------------------------ camera
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", async () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("is-on", x === t));
    await post("/api/camera", { mode: t.dataset.cam });
  }));

  // ------------------------------------------------------------------ ledger
  function clearLedger() {
    attempts = {}; current = null;
    $("attempts").replaceChildren();
    $("verdict").hidden = true;
    $("ledger-empty").hidden = false;
  }

  function attemptBox(n, replan) {
    if (attempts[n]) return attempts[n];
    $("ledger-empty").hidden = true;
    const box = el("div", "attempt");
    const head = el("div", "attempt-head");
    head.append(el("span", "", replan ? `Replan ${n - 1}` : "Plan"), el("span", "", ""));
    const reason = el("p", "reason is-thinking", replan ? "Reading what went wrong and planning again" : "Consulting the planner");
    const table = el("table", "steps");
    box.append(head, reason, table);
    $("attempts").append(box);
    attempts[n] = { box, head, reason, table, rows: [] };
    return attempts[n];
  }

  function stamp(td, kind, text) {
    td.replaceChildren(el("span", `stamp ${kind} new`, text));
  }

  function setRow(a, i, state, note) {
    const tr = a.rows[i];
    if (!tr) return;
    const st = tr.querySelector(".st");
    tr.classList.remove("is-pending");
    if (state === "work") stamp(st, "work", "At work");
    else if (state === "done") { if (!st.querySelector(".done")) stamp(st, "done", "Done"); }
    else if (state === "fail") stamp(st, "fail", "Failed");
    else if (state === "skip") { st.replaceChildren(el("span", "", "-")); tr.classList.add("is-pending"); }
    if (note) {
      const act = tr.querySelector(".act");
      act.querySelector("small")?.remove();
      act.append(el("small", "", note));
    }
  }

  function renderPlan(ev) {
    const a = attemptBox(ev.attempt, ev.replan);
    current = ev.attempt;
    a.reason.classList.remove("is-thinking");
    a.reason.textContent = ev.reason || (ev.steps.length ? "" : "No reason given.");
    a.head.lastChild.textContent = `${ev.steps.length} step${ev.steps.length === 1 ? "" : "s"} · ${ev.seconds}s to plan`;
    a.table.replaceChildren();
    a.rows = ev.steps.map((s, i) => {
      const tr = el("tr", "is-pending");
      tr.append(el("td", "n", String(i + 1).padStart(2, "0")), el("td", "act", s), el("td", "st", ""));
      a.table.append(tr);
      return tr;
    });
    if (!ev.steps.length) a.reason.classList.add("refused");
  }

  function onStep(ev) {
    const a = attempts[ev.attempt];
    if (!a) return;
    for (let i = 0; i < ev.index - 1; i++) setRow(a, i, "done");
    setRow(a, ev.index - 1, "work");
    $("view-step").textContent = `Step ${ev.index}: ${ev.action}`;
  }

  function onResult(ev) {
    ev.attempts.forEach((at, k) => {
      const a = attempts[k + 1];
      if (!a) return;
      at.plan.forEach((_, i) => {
        const s = at.steps[i];
        if (!s) setRow(a, i, "skip");
        else setRow(a, i, s.ok ? "done" : "fail", s.ok ? null : s.msg);
      });
    });
    const v = $("verdict");
    v.hidden = false;
    v.className = "verdict " + (ev.success ? "ok" : "bad");
    v.replaceChildren(
      el("span", "big", ev.success ? "Completed" : "Not completed"),
      el("p", "", `${ev.message[0].toUpperCase()}${ev.message.slice(1)}. ${Math.round(ev.sim_time)} s of robot time.`));
    $("view-step").textContent = "";
  }

  // ------------------------------------------------------------------ notebook
  function log(text, simTime, cls) {
    const li = el("li", cls || "");
    li.append(el("time", "", simTime !== undefined ? `${simTime.toFixed(1)}s` : ""), el("span", "", text));
    const ol = $("log");
    ol.append(li);
    while (ol.children.length > 400) ol.firstChild.remove();
    ol.scrollTop = ol.scrollHeight;
  }

  // ------------------------------------------------------------------ state + catalogue
  const fmtClock = (s) => {
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };

  function whereNote(b) {
    if (b.known === "held") return ["in the gripper", "held"];
    if (!b.known) return ["not seen yet", "away"];
    if (b.known === b.home) return ["on its shelf", ""];
    return [b.known.startsWith("Table") ? `on ${b.known}` : `in ${b.known}`, "away"];
  }

  function renderState(s) {
    if (!s || !s.books) return;
    $("st-clock").textContent = fmtClock(s.sim_time || 0);
    const at = s.robot.at ? ` · ${s.robot.at}` : "";
    $("st-robot").textContent = `${s.status}${at}`;
    $("st-hands").textContent = s.robot.holding || "empty";
    if (!s.busy && busy && s.status === "idle") setBusy(false);
    if (s.busy && !busy) setBusy(true);
    $("frame-note").textContent = s.status === "idle" ? "" : s.status.charAt(0).toUpperCase() + s.status.slice(1) + "...";

    const shelves = {};
    s.books.forEach((b) => {
      const [shelf, comp] = [b.home.split("-")[0], b.home];
      (shelves[shelf] ||= {});
      (shelves[shelf][comp] ||= []).push(b);
    });
    const names = { S1: "west wall", S2: "middle", S3: "by the window" };
    const drawers = $("drawers");
    drawers.replaceChildren();
    ["S1", "S2", "S3"].forEach((sh) => {
      const d = el("div", "drawer");
      const h = el("h3", "", `Shelf ${sh}`);
      h.append(el("span", "", names[sh]));
      d.append(h);
      ["A", "B"].forEach((row) => [1, 2, 3, 4].forEach((col) => {
        const id = `${sh}-${row}${col}`;
        const c = el("div", "comp");
        c.append(el("span", "comp-id", id));
        const ul = el("ul");
        const books = (shelves[sh] && shelves[sh][id]) || [];
        if (!books.length) ul.append(el("li", "none", "empty"));
        books.forEach((b) => {
          const li = el("li");
          const btn = el("button", "", b.title);
          btn.type = "button";
          btn.addEventListener("click", () => {
            cmd.value = `Put ${b.title} back on its shelf`;
            cmd.focus();
          });
          const [note, cls] = whereNote(b);
          li.append(btn, el("span", `where ${cls}`, note));
          ul.append(li);
        });
        c.append(ul);
        d.append(c);
      }));
      drawers.append(d);
    });
  }

  fetch("/api/state").then((r) => r.json()).then(renderState).catch(() => {});

  // ------------------------------------------------------------------ events
  const still = new URLSearchParams(location.search).has("still");   // one frame, no live updates
  $("live").src = still ? "/frame.jpg" : "/stream.mjpg";
  function handle(ev) {
    switch (ev.type) {
      case "state": renderState(ev.state); break;
      case "command":
        clearLedger();
        setBusy(true);
        log(`Request: "${ev.command}" (${ev.planner === "llm" ? "language model" : "rule-based"})`, undefined, "is-head");
        break;
      case "planning":
        attemptBox(ev.attempt, ev.replan);
        $("frame-note").textContent = ev.replan ? "Replanning..." : "Planning...";
        break;
      case "plan":
        renderPlan(ev);
        if (!ev.steps.length) log(`No plan: ${ev.reason}`, undefined, "is-bad");
        break;
      case "step": onStep(ev); log(`Step ${ev.index}: ${ev.action}`, ev.sim_time, "is-head"); break;
      case "log": log(ev.text, ev.sim_time); break;
      case "result":
        onResult(ev);
        log(ev.success ? "Done." : `Stopped: ${ev.message}`, undefined, ev.success ? "is-head" : "is-bad");
        break;
      case "reset": clearLedger(); $("log").replaceChildren(); msg.textContent = ""; break;
      case "error": log(`Error: ${ev.message}`, undefined, "is-bad"); msg.textContent = ev.message; break;
    }
  }

  if (still) {                 // replay what has happened so far, no live connection
    fetch("/api/history").then((r) => r.json()).then((evs) => evs.forEach(handle)).catch(() => {});
    return;
  }
  const es = new EventSource("/events");
  es.onmessage = (m) => handle(JSON.parse(m.data));
  es.onerror = () => { $("frame-note").textContent = "Lost the connection to the desk. Is web/server.py still running?"; };
  $("live").addEventListener("error", () => { $("frame-note").textContent = "No picture from the simulator."; });
})();
