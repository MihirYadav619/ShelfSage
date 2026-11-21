"""Final evaluation (Phase 10): read the benchmark results, compute the metrics and statistics,
draw the plots, write the summary.

Inputs (benchmark/results/, written by run_eval.py):
    baseline_plan.json, baseline_exec.json                 the rule-based baseline (deterministic)
    llm_<backend>_<effort>_plan_r<k>.json                  LLM, plan level, repeat k
    llm_<backend>_<effort>_exec_r<k>.json                  LLM, executed in the simulator, repeat k

Metrics
    task success      per category and overall. Baseline: one run (its plans never change).
                      LLM: every repeat counted (n = tasks x repeats), mean and the range over
                      repeats; 95 % Wilson intervals on the pooled rate
    consistency       LLM tasks whose outcome differs between repeats
    paired test       exact McNemar test (two-sided binomial on the discordant tasks), baseline vs
                      the LLM's majority outcome per task, executed level
    planning          LLM latency (median, 90th percentile, max), validator retries, tokens
    execution         simulated time per successful task
    recovery          failure tasks: recovered runs per failure type

Outputs: benchmark/results/summary.md, summary.json, plots/*.png

Usage (project root, venv active):
    python benchmark/analyze.py
"""
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmark/results"
PLOTS = RES / "plots"
CATS = ["single", "multi", "phrasing", "spatial", "query", "infeasible", "failure"]

# reference palette (dataviz skill, light mode): categorical slots 1-2, surface and text tokens
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(pattern):
    out = {}
    for f in sorted(glob.glob(str(RES / pattern))):
        m = re.search(r"_r(\d+)\.json$", f)
        out[int(m.group(1)) if m else 1] = json.loads(Path(f).read_text())
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(b_only, l_only):
    """Exact two-sided McNemar p-value from the discordant counts."""
    from scipy.stats import binomtest
    n = b_only + l_only
    return 1.0 if n == 0 else binomtest(min(b_only, l_only), n, 0.5).pvalue


def by_id(run):
    return {r["id"]: r for r in run["results"]}


def rates(runs, cats):
    """{cat: (successes, counted, per-repeat rates)} over the repeats of one planner."""
    out = {}
    for c in cats:
        k = n = 0
        per = []
        for run in runs.values():
            rs = [r for r in run["results"] if (c == "all" or r["category"] == c) and r.get("success") is not None]
            s = sum(1 for r in rs if r["success"])
            k, n = k + s, n + len(rs)
            if rs:
                per.append(s / len(rs))
        out[c] = (k, n, per)
    return out


def setup_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=10)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)


def plot_categories(base, llm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = [c for c in CATS if base.get(c, (0, 0, []))[1] or llm.get(c, (0, 0, []))[1]]
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160, facecolor=SURFACE)
    setup_axes(ax)
    h = 0.34
    for i, c in enumerate(cats):
        y = len(cats) - 1 - i
        for dy, (data, color, name) in ((h / 2 + 0.02, (base, BLUE, "baseline")), (-h / 2 - 0.02, (llm, ORANGE, "LLM"))):
            k, n, per = data.get(c, (0, 0, []))
            if not n:
                continue
            v = 100 * k / n
            ax.barh(y + dy, v, height=h, color=color, label=name if i == 0 else None)
            end = v
            if len(per) > 1 and max(per) > min(per):  # range over the repeats
                ax.plot([100 * min(per), 100 * max(per)], [y + dy, y + dy], color=TEXT, lw=1.2)
                end = max(v, 100 * max(per))
            ax.text(end + 1.5, y + dy, f"{v:.0f} %", va="center", ha="left", fontsize=9, color=TEXT)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([f"{c}" for c in reversed(cats)], color=TEXT)
    ax.set_xlim(0, 112)
    ax.set_xlabel("tasks succeeded (%); thin line: range over the LLM's repeats")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=24)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_recovery(base_exec, llm_exec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = {t["id"]: t for t in json.loads((ROOT / "benchmark/tasks.json").read_text())["tasks"]}
    ids = [t for t in spec if spec[t]["category"] == "failure"]
    labels = [f"{t} {spec[t]['inject']['type'].replace('_', ' ')}" for t in ids]
    fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=160, facecolor=SURFACE)
    setup_axes(ax)
    h = 0.34
    for i, t in enumerate(ids):
        y = len(ids) - 1 - i
        for dy, runs, color, name in ((h / 2 + 0.02, base_exec, BLUE, "baseline"), (-h / 2 - 0.02, llm_exec, ORANGE, "LLM")):
            rs = [by_id(r).get(t) for r in runs.values()]
            rs = [r for r in rs if r and r.get("success") is not None]
            if not rs:
                continue
            k = sum(r["success"] for r in rs)
            ax.barh(y + dy, 100 * k / len(rs), height=h, color=color, label=name if i == 0 else None)
            ax.text(100 * k / len(rs) + 1.5, y + dy, f"{k}/{len(rs)}", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(list(reversed(labels)), color=TEXT)
    ax.set_xlim(0, 115)
    ax.set_xlabel("runs recovered (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Recovery from injected failures (executed in the simulator)", loc="left", color=TEXT, fontsize=12, pad=24)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    for tx in leg.get_texts():
        tx.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_latency(lat, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    lat = np.sort(np.array(lat))
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=160, facecolor=SURFACE)
    setup_axes(ax)
    ax.plot(lat, 100 * np.arange(1, len(lat) + 1) / len(lat), color=BLUE, lw=2, drawstyle="steps-post")
    med = float(np.median(lat))
    ax.axvline(med, color=TEXT2, lw=1, ls="--")
    ax.text(med * 1.08, 12, f"median {med:.1f} s", color=TEXT, fontsize=9)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([t for t in (1, 2, 3, 5, 10, 20, 50) if lat[0] * 0.8 <= t <= lat[-1] * 1.25]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("planning time per command (s, log scale)")
    ax.set_ylabel("commands (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"LLM planning latency, {len(lat)} requests (cumulative)", loc="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    base_plan = load("baseline_plan.json")
    base_exec = load("baseline_exec.json")
    llm_plan = load("llm_*_plan_r*.json")
    llm_exec = load("llm_*_exec_r*.json")
    if not (base_exec and llm_exec):
        sys.exit("need baseline_exec.json and llm_*_exec_r*.json (run benchmark/run_eval.py)")
    cats = CATS + ["all"]
    R = {"baseline_plan": rates(base_plan, cats), "baseline_exec": rates(base_exec, cats),
         "llm_plan": rates(llm_plan, cats), "llm_exec": rates(llm_exec, cats)}

    # consistency + McNemar (executed level, LLM majority per task)
    ids = [r["id"] for r in base_exec[1]["results"] if r.get("success") is not None]
    flips, b_only, l_only, both, none = [], 0, 0, 0, 0
    for t in ids:
        outs = [by_id(run)[t]["success"] for run in llm_exec.values() if by_id(run).get(t, {}).get("success") is not None]
        if not outs:
            continue
        if len(set(outs)) > 1:
            flips.append((t, outs))
        lm = sum(outs) * 2 > len(outs)
        bs = bool(by_id(base_exec[1])[t]["success"])
        both += bs and lm
        none += (not bs) and (not lm)
        b_only += bs and not lm
        l_only += lm and not bs
    p = mcnemar(b_only, l_only)

    # planning cost (plan level runs hold every LLM request of the non-failure tasks)
    lat, att, tok_in, tok_out = [], [], 0, 0
    for run in llm_plan.values():
        for r in run["results"]:
            if r.get("latency") is not None and not r.get("cached"):
                lat.append(r["latency"])
                att.append(r["attempts"])
                tok_in += r["usage"]["input"] + r["usage"]["cache_read"]
                tok_out += r["usage"]["output"]
    sim = {name: [r["sim_time"] for run in runs.values() for r in run["results"]
                  if r.get("success") and r.get("sim_time")] for name, runs in (("baseline", base_exec), ("llm", llm_exec))}

    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_categories(R["baseline_exec"], R["llm_exec"], PLOTS / "success_by_category.png",
                    f"Task success by category, executed in the simulator (LLM: {len(llm_exec)} repeats)")
    plot_recovery(base_exec, llm_exec, PLOTS / "failure_recovery.png")
    if lat:
        plot_latency(lat, PLOTS / "llm_latency.png")

    def row(name, d):
        k, n, per = d
        lo, hi = wilson(k, n)
        rng = f" ({100 * min(per):.0f}-{100 * max(per):.0f})" if len(per) > 1 else ""
        return f"{100 * k / n:.0f} %{rng} [{100 * lo:.0f}-{100 * hi:.0f}], {k}/{n}" if n else "-"

    lines = ["# ShelfSage benchmark: final results", "",
             f"Tasks: benchmark/tasks.json (42). Baseline: 1 run (deterministic). LLM: {len(llm_exec)} repeats "
             f"executed, {len(llm_plan)} at plan level. Cells: success rate (range over repeats) "
             "[95 % Wilson interval], successes / counted runs.", "",
             "| category | baseline plan | baseline executed | LLM plan | LLM executed |", "|---|---|---|---|---|"]
    for c in cats:
        lines.append(f"| {c} | {row(c, R['baseline_plan'][c])} | {row(c, R['baseline_exec'][c])} | "
                     f"{row(c, R['llm_plan'][c])} | {row(c, R['llm_exec'][c])} |")
    lines += ["", f"Paired per task, executed (LLM majority of its repeats): both succeed {both}, "
              f"only the baseline {b_only}, only the LLM {l_only}, neither {none}; exact McNemar p = {p:.2g}.",
              f"LLM tasks whose outcome changed between repeats: {len(flips)}"
              + (": " + ", ".join(f"{t} {''.join('o' if x else 'x' for x in o)}" for t, o in flips) if flips else ""),
              ""]
    if lat:
        s = sorted(lat)
        lines.append(f"LLM planning: {len(lat)} requests, latency median {s[len(s) // 2]:.1f} s, 90th percentile "
                     f"{s[int(0.9 * (len(s) - 1))]:.1f} s, max {s[-1]:.1f} s; validator retries in "
                     f"{sum(a > 1 for a in att)} of {len(att)}; ~{tok_in // len(lat)} input / {tok_out // len(lat)} "
                     "output tokens per command.")
    for name, v in sim.items():
        if v:
            v = sorted(v)
            lines.append(f"Simulated time per successful task, {name}: median {v[len(v) // 2]:.0f} s, "
                         f"range {v[0]:.0f}-{v[-1]:.0f} s (n = {len(v)}).")
    lines += ["", "Plots: plots/success_by_category.png, plots/failure_recovery.png, plots/llm_latency.png"]
    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps({
        "rates": {k: {c: {"successes": v[0], "runs": v[1], "per_repeat": v[2]} for c, v in d.items()} for k, d in R.items()},
        "mcnemar": {"both": both, "baseline_only": b_only, "llm_only": l_only, "neither": none, "p": p},
        "flips": flips, "latency": lat, "sim_time": sim}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
