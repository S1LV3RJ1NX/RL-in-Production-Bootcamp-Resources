"""Publication vector figures for Dreaming to Dodge (run with ./.plotvenv/bin/python).

Writes figures/doom/: fig_pipeline.png, fig_journey.png, fig_dream_rewards.png,
fig_recall.png  (plus survival_showdown.png from make_doom_showdown.py).
Clean, flat, journal-ready matplotlib.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "figures/doom"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
TEAL, BLUE, AMBER, CORAL, GRAY, RED = "#1d9e75", "#378add", "#ef9f27", "#d85a30", "#b4b2a9", "#a32d2d"


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=color, ec="none", alpha=0.16, mutation_aspect=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="none", ec=color, lw=1.6, mutation_aspect=1))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, weight="bold", color="#222")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.2, color="#555")


def arrow(ax, x1, y1, x2, y2, color="#444", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=1.6, color=color, linestyle=ls, shrinkA=2, shrinkB=2))


# ---- Figure 1: method pipeline -------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
    y, h, w = 1.5, 1.3, 2.0
    box(ax, 0.2, y, w, h, "VizDoom take_cover", "64×64×3 frames\n3 actions, +1/step", GRAY)
    box(ax, 2.6, y, w, h, "Tokenizer (VQ-VAE)", "frame → 64 tokens\nwarmth-weighted", TEAL)
    box(ax, 5.0, y, w, h, "World model (GPT)", "next tokens + reward\n+ done | KV-cached", BLUE)
    box(ax, 7.4, y, w, h, "Controller (MLP)", "~900 params\nCMA-ES in dream", AMBER)
    box(ax, 9.55, y + 0.15, 1.25, h - 0.3, "Real env", "held-out\neval", CORAL)
    for x1, x2 in [(2.2, 2.6), (4.6, 5.0), (7.0, 7.4), (9.4, 9.55)]:
        arrow(ax, x1, y + h / 2, x2, y + h / 2)
    # dream loop (WM <-> controller)
    ax.text(6.4, y + h + 0.42, "imagination loop  (no real env)", ha="center", fontsize=9, style="italic", color=BLUE)
    arrow(ax, 8.0, y + h, 6.4, y + h + 0.28, color=BLUE, ls="--", style="-|>")
    arrow(ax, 6.4, y + h + 0.28, 5.6, y + h, color=BLUE, ls="--", style="-|>")
    ax.text(5.5, 0.55, "trained 100% inside the dream — the policy never touches the real game during learning",
            ha="center", fontsize=9, color="#333")
    fig.savefig(f"{OUT}/fig_pipeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote fig_pipeline.png")


# ---- Figure 2: the method progression (the levers) -----------------------
def journey():
    labels = ["random", "LSTM+REINFORCE\n(exploits WM)", "linear controller\n(reactive)",
              "MLP controller", "best-of-N + MLP\n(held-out)", "oracle"]
    vals = [67, 44, 64, 76, 96.6, 98.3]
    cols = [GRAY, RED, "#9fd6bd", "#5cc39e", TEAL, AMBER]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=cols, width=0.64, edgecolor="white")
    ax.axhline(90, ls=":", color=BLUE, lw=1.4); ax.text(0.05, 92, "model-free DQN (90)", color=BLUE, fontsize=8.5)
    ax.axhline(188, ls="--", color=RED, lw=1.2); ax.text(5.4, 190, "solved 188", color=RED, fontsize=8.5, ha="right")
    for x, v in zip(xs, vals):
        ax.text(x, v + 2.5, f"{v:g}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("real-env survival (agent steps)"); ax.set_ylim(0, 205)
    ax.set_title("From world-model exploitation to a reactive agent that matches the oracle", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_journey.png", dpi=200); plt.close(fig)
    print("wrote fig_journey.png")


# ---- Figure 3: the decisive diagnostic (dream rewards dodging) ------------
def dream_rewards():
    labels = ["always\nno-op", "always\nleft", "always\nright", "reactive\noracle"]
    dream = [31, 30.6, 24, 46.3]; real = [44.4, 45.3, 45.8, 104.5]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(x - w / 2, dream, w, label="survival IN THE DREAM", color=BLUE)
    ax.bar(x + w / 2, real, w, label="survival in REALITY", color=AMBER)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("survival (agent steps)")
    ax.set_title("The dream correctly rewards reactive dodging\n(reactive oracle wins in-dream — so the signal exists)", fontsize=10.5)
    ax.legend(fontsize=9, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_dream_rewards.png", dpi=200); plt.close(fig)
    print("wrote fig_dream_rewards.png")


# ---- Figure 4: fireball recall (tokenizer fix) ---------------------------
def recall():
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    bars = ["uniform L1", "luminance\n(fg)", "warmth-weighted\n(ours)"]
    vals = [0.53, 0.63, 0.95]
    ax.bar(bars, vals, color=[GRAY, "#9fd6bd", TEAL], width=0.6, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("fireball-recall (brightness retained)"); ax.set_ylim(0, 1.05)
    ax.set_title("Keeping the fireball through quantization", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_recall.png", dpi=200); plt.close(fig)
    print("wrote fig_recall.png")


if __name__ == "__main__":
    pipeline(); journey(); dream_rewards(); recall()
