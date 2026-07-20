#!/usr/bin/env python3
"""Plot a clean learning curve (grasp success rate vs training) from a LeRobot RL train.log.
Handles the append-multiple-runs log by segmenting on step resets and picking the run that
converges highest. Styled to match the 'Grasping in the Dark' site (teal, clean).

  python3 figures/plot_reward_curve.py <train.log> [out.png]
"""
import re, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

TEAL = "#0f8f68"; INK = "#161a20"; MUT = "#5c626f"; AMBER = "#c9871a"; LINE = "#e8eaef"

def parse(path):
    pat = re.compile(r"Global step (\d+): Episode reward: ([-\d.]+)")
    pts = [(int(s), float(r)) for s, r in pat.findall(pathlib.Path(path).read_text())]
    # segment on step resets (new run restarts the counter)
    segs, cur = [], []
    last = -1
    for s, r in pts:
        if s < last - 50:      # counter reset => new run
            if cur: segs.append(cur)
            cur = []
        cur.append((s, r)); last = s
    if cur: segs.append(cur)
    # pick the segment with the highest rolling-success peak (the converged run)
    def peak(seg):
        r = np.array([x[1] for x in seg])
        if len(r) < 5: return 0
        w = min(15, len(r)); return np.convolve(r, np.ones(w)/w, "valid").max()
    return max(segs, key=peak) if segs else []

def main():
    log = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "reward_curve.png"
    seg = parse(log)
    if len(seg) < 5:
        sys.exit("not enough data")
    steps = np.array([s for s, _ in seg]); rew = np.array([r for _, r in seg])
    w = min(15, max(3, len(rew)//8))
    roll = np.convolve(rew, np.ones(w)/w, "valid")
    rsteps = steps[w-1:]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.edgecolor": MUT, "axes.linewidth": 1.0})
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=160)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    # raw per-episode successes as faint ticks
    ax.scatter(steps, rew, s=10, color=TEAL, alpha=0.14, zorder=1)
    # rolling success line + fill
    ax.plot(rsteps, roll, color=TEAL, lw=3, zorder=3, solid_capstyle="round")
    ax.fill_between(rsteps, 0, roll, color=TEAL, alpha=0.08, zorder=1)

    # first-success marker
    fs = next((s for s, r in seg if r > 0.5), None)
    if fs is not None:
        ax.axvline(fs, color=AMBER, ls=(0, (4, 3)), lw=1.3, alpha=.8, zorder=2)
        ax.annotate("first successful grasp", xy=(fs, 0.06), xytext=(fs + (steps.max()-steps.min())*0.03, 0.20),
                    color=AMBER, fontsize=11, fontweight="bold")
    # converged marker
    ax.annotate(f"≈{int(round(roll[-1]*100))}% — converged", xy=(rsteps[-1], roll[-1]),
                xytext=(rsteps[-1]-(steps.max()-steps.min())*0.30, min(roll[-1], 0.9)-0.14),
                color=TEAL, fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))

    ax.set_xlim(steps.min(), steps.max()); ax.set_ylim(-0.02, 1.06)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("training — environment steps", color=INK, fontsize=12.5)
    ax.set_ylabel("grasp success rate (rolling)", color=INK, fontsize=12.5)
    ax.set_title("Grasping in the Dark — learning from a reward it almost never sees",
                 color=INK, fontsize=13.5, fontweight="bold", pad=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, lw=0.9)
    ax.tick_params(colors=MUT)
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print(f"wrote {out}  (segment: {len(seg)} episodes, peak {roll.max():.0%}, final {roll[-1]:.0%})")

if __name__ == "__main__":
    main()
