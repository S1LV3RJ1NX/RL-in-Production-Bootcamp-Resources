"""Publication-quality, WHITE-BACKGROUND analysis graphs from results/analysis_data.json.

Produces four figures (paper + website + slides):
  fig_reactivity.png   — the agent moves AWAY from the fireball (reactive dodging proof)
  fig_survival.png     — survival-step distributions: agent ~ oracle >> random
  fig_fidelity.png     — dream accuracy vs rollout horizon (why short imagined rollouts work)
  fig_summary_bars.png — mean survival with 95% CI, three policies
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- house palette (matches the academic paper figures) ---
INK   = "#1f2733"
GREY  = "#8a94a6"
BLUE  = "#2f6fd0"   # agent / mechanism
TEAL  = "#0f8f68"   # oracle / numbers
AMBER = "#c9871a"   # fireball / warning
LIGHT = "#e7ebf1"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 13, "axes.edgecolor": "#c7ced9", "axes.linewidth": 1.1,
    "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 1.0,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False,
})

def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results/analysis_data.json")))
OUT = os.path.join(HERE, "figures/doom"); os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. reactivity
react = np.array(D["reactivity"], dtype=float)      # [col_norm(-1 left .. +1 right), action]
col, act = react[:, 0], react[:, 1].astype(int)
# action mapping for take_cover: 0=stay, 1=move LEFT, 2=move RIGHT
nA = int(act.max()) + 1
edges = np.linspace(-1, 1, 10)
mids = 0.5 * (edges[:-1] + edges[1:])
# "rightward tendency" = P(move right) - P(move left) per fireball-position bin
tend, n_per = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (col >= lo) & (col < hi)
    n = m.sum(); n_per.append(n)
    if n == 0:
        tend.append(np.nan); continue
    a = act[m]
    p_right = np.mean(a == 2) if nA > 2 else 0.0
    p_left = np.mean(a == 1)
    tend.append(p_right - p_left)
tend = np.array(tend)

fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.axhline(0, color=GREY, lw=1.2, ls=(0, (4, 4)))
ax.axvline(0, color=LIGHT, lw=8, zorder=0)
ax.plot(mids, tend, "-", color=BLUE, lw=3, zorder=3)
ax.scatter(mids, tend, s=[30 + 220 * (nn / max(n_per)) for nn in n_per],
           color=BLUE, edgecolor="white", lw=1.5, zorder=4)
ax.fill_between(mids, tend, 0, where=~np.isnan(tend), color=BLUE, alpha=0.08, zorder=1)
ax.set_xlabel("fireball horizontal position  (−1 = far left   →   +1 = far right)")
ax.set_ylabel("move-right minus move-left  (action tendency)")
ax.set_title("The agent steps AWAY from the fireball")
ax.set_xlim(-1, 1); ax.set_ylim(-0.75, 0.75)
ax.annotate("fireball on the left →\nagent moves RIGHT", (-0.72, 0.42), color=BLUE,
            fontsize=11.5, ha="left", fontweight="600")
ax.annotate("fireball on the right →\nagent moves LEFT", (0.72, -0.42), color=BLUE,
            fontsize=11.5, ha="right", fontweight="600")
ax.annotate("dot size = how often\nthe fireball was there", (0.0, -0.66), color=GREY,
            fontsize=10, ha="center")
despine(ax); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_reactivity.png"), dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 2. survival distributions
surv = D["survival"]
ctrl = np.array(surv["controller"]); rnd = np.array(surv["random"]); orc = np.array(surv["oracle"])
fig, ax = plt.subplots(figsize=(8.6, 5.0))
bins = np.linspace(0, max(ctrl.max(), rnd.max(), orc.max()) + 10, 26)
for data, c, lab, mean in [(rnd, GREY, "random", rnd.mean()),
                           (ctrl, BLUE, "our agent (dreamed policy)", ctrl.mean()),
                           (orc, TEAL, "oracle (hand-coded)", orc.mean())]:
    ax.hist(data, bins=bins, color=c, alpha=0.32, edgecolor=c, linewidth=1.4, label=lab)
    ax.axvline(mean, color=c, lw=2.4, ls=(0, (2, 2)))
ax.set_xlabel("steps survived per episode  (held-out seeds)")
ax.set_ylabel("number of episodes")
ax.set_title("Trained only in the dream — survives like the oracle")
ax.legend(loc="upper left", fontsize=11.5)
ax.annotate(f"agent  {ctrl.mean():.1f}", (ctrl.mean(), ax.get_ylim()[1]*0.72), color=BLUE,
            fontsize=11, ha="center", fontweight="700")
ax.annotate(f"random  {rnd.mean():.1f}", (rnd.mean(), ax.get_ylim()[1]*0.9), color=GREY,
            fontsize=11, ha="center", fontweight="700")
despine(ax); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_survival.png"), dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 3. dream fidelity vs horizon
fid = np.array(D["dream_fidelity_mse_per_step"])
steps = np.arange(1, len(fid) + 1)
fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.plot(steps, fid, "-", color=AMBER, lw=3, zorder=3)
ax.scatter(steps, fid, s=34, color=AMBER, edgecolor="white", lw=1.4, zorder=4)
ax.fill_between(steps, fid, fid.min(), color=AMBER, alpha=0.10)
ax.set_xlabel("imagined step into the future (rollout horizon)")
ax.set_ylabel("pixel error vs the real game  (MSE)")
ax.set_title("The dream is sharp up close, blurs far out")
ax.annotate("short rollouts stay faithful →\nwe train the policy here", (2.2, fid[:6].mean()),
            color=TEAL, fontsize=11.5, ha="left", fontweight="600",
            xytext=(6.5, fid.min() + 0.32*(fid.max()-fid.min())),
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.8))
despine(ax); ax.set_xlim(0.5, len(fid) + 0.5); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_fidelity.png"), dpi=170); plt.close(fig)

# ---------------------------------------------------------------- 4. summary bars w/ 95% CI
labels = ["random", "our agent", "oracle"]
data = [rnd, ctrl, orc]; colors = [GREY, BLUE, TEAL]
means = [d.mean() for d in data]
cis = [1.96 * d.std(ddof=1) / np.sqrt(len(d)) for d in data]
fig, ax = plt.subplots(figsize=(7.4, 5.0))
xs = np.arange(3)
bars = ax.bar(xs, means, width=0.62, color=colors, alpha=0.85,
              yerr=cis, capsize=8, ecolor=INK, error_kw=dict(lw=1.6))
for x, m in zip(xs, means):
    ax.annotate(f"{m:.1f}", (x, m), ha="center", va="bottom", xytext=(0, 8),
                textcoords="offset points", fontsize=15, fontweight="700", color=INK)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel("mean steps survived  (±95% CI)")
ax.set_title("Held-out survival — 40 unseen seeds")
ax.set_ylim(0, max(means) * 1.22)
despine(ax); ax.grid(axis="x", visible=False); fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_summary_bars.png"), dpi=170); plt.close(fig)

print("wrote:", [f for f in ("fig_reactivity", "fig_survival", "fig_fidelity", "fig_summary_bars")])
print(f"n_react={len(react)}  ctrl={ctrl.mean():.1f}  rand={rnd.mean():.1f}  orc={orc.mean():.1f}  nA={nA}")
