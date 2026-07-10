"""Assemble the 'Dreaming to Dodge' showdown figures from synced result JSONs.

Runs LOCALLY with the plotting venv (no torch needed):
    ./.plotvenv/bin/python make_doom_figures.py

Reads results/{eval_doom,eval_modelfree_doom_mf,deliverables_doom}.json (whatever
is present) and writes figures/doom/*.png:
  * survival_showdown.png  — bar chart of mean survival (steps + tics) for
    random / model-free DQN / IRIS (imagination) / heuristic oracle, with the
    World Models "solved" line (750 tics = ~188 steps) marked.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES, OUT = "results", "figures/doom"
FS = 4                      # frame_skip: tics = steps * FS
SOLVED_TICS = 750           # World Models "solved" bar for DoomTakeCover


def _load(name):
    p = os.path.join(RES, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def showdown():
    ev = _load("eval_doom.json")
    mf = _load("eval_modelfree_doom_mf.json")
    if not ev:
        print("no eval_doom.json yet — skipping showdown"); return
    bars = []  # (label, mean_steps, std_steps, color)
    bars.append(("random", ev["random_baseline"]["return_mean"],
                 ev["random_baseline"].get("return_std", 0), "#B4B2A9"))
    if mf:
        bars.append(("model-free DQN\n(200k real frames)", mf["dqn_survival_steps"]["mean"],
                     mf["dqn_survival_steps"].get("std", 0), "#378ADD"))
    bars.append(("IRIS\n(trained in imagination)", ev["policy"]["return_mean"],
                 ev["policy"].get("return_std", 0), "#1D9E75"))
    if "oracle_baseline" in ev:
        bars.append(("heuristic oracle\n(sees the game)", ev["oracle_baseline"]["return_mean"],
                     ev["oracle_baseline"].get("return_std", 0), "#EF9F27"))

    labels = [b[0] for b in bars]
    means = [b[1] for b in bars]
    stds = [b[2] for b in bars]
    colors = [b[3] for b in bars]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bars))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, width=0.62, edgecolor="white")
    ax.axhline(SOLVED_TICS / FS, ls="--", color="#A32D2D", lw=1.5)
    ax.text(len(bars) - 0.5, SOLVED_TICS / FS + 2,
            f"World Models 'solved' = {SOLVED_TICS} tics ({SOLVED_TICS // FS} steps)",
            ha="right", va="bottom", fontsize=9, color="#A32D2D")
    for xi, m in zip(x, means):
        ax.text(xi, m + 2, f"{m:.0f} steps\n{m*FS:.0f} tics", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean survival (agent steps)")
    ax.set_title("Dreaming to Dodge — real-env survival on VizDoom take_cover")
    ax.set_ylim(0, max(max(means) * 1.25, SOLVED_TICS / FS * 1.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/survival_showdown.png", dpi=140); plt.close(fig)
    print(f"wrote {OUT}/survival_showdown.png")
    print("  survival (steps):", {l.split(chr(10))[0]: round(m, 1) for l, m in zip(labels, means)})


if __name__ == "__main__":
    showdown()
