"""Honest survival showdown for Dreaming to Dodge (run with ./.plotvenv/bin/python)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

FS = 4
SOLVED = 750 // FS   # 188 steps
# honest numbers measured on Modal (real-env survival, steps):
# VERIFIED on held-out seeds (n=50, seeds 300000+); IRIS/random/oracle are paired
# (same seeds); DQN from its own 200k-real-frame eval.
bars = [
    ("random", 66.6, 0, "#B4B2A9"),
    ("IRIS agent\n(trained 100% in imagination)", 96.6, 11.0, "#1D9E75"),  # verified, 95% CI
    ("model-free DQN\n(200k real frames)", 90, 0, "#378ADD"),
    ("heuristic oracle\n(reactive, sees game)", 98.3, 0, "#EF9F27"),
]
labels = [b[0] for b in bars]; means = [b[1] for b in bars]
errs = [b[2] for b in bars]; colors = [b[3] for b in bars]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(bars))
ax.bar(x, means, yerr=errs, capsize=6, color=colors, width=0.62, edgecolor="white")
ax.axhline(SOLVED, ls="--", color="#A32D2D", lw=1.5)
ax.text(len(bars) - 0.5, SOLVED + 3, f"World Models 'solved' = 750 tics ({SOLVED} steps)",
        ha="right", va="bottom", fontsize=9, color="#A32D2D")
# annotate the agent's solved-level best episodes
ax.annotate("best episodes: 217-284 steps\n(> solved bar 188)", xy=(1, 108), xytext=(1.25, 155),
            fontsize=8.5, color="#0F6E56",
            arrowprops=dict(arrowstyle="->", color="#0F6E56", lw=1))
for xi, m in zip(x, means):
    ax.text(xi, m + 3, f"{m:.0f} steps\n{m*FS:.0f} tics", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("mean survival (agent steps)")
ax.set_title("Dreaming to Dodge — real-env survival on VizDoom take_cover\n"
             "(agent trained 100% inside the world model's dream)", fontsize=11)
ax.set_ylim(0, 200)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
os.makedirs("figures/doom", exist_ok=True)
fig.savefig("figures/doom/survival_showdown.png", dpi=140)
print("wrote figures/doom/survival_showdown.png")
