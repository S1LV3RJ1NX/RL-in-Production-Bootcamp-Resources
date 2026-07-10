"""Build the 7 student-facing figures for Project 2 (IRIS-from-scratch).

Reads ONLY the real synced results JSON in results/ (no synthetic numbers) and
writes the following into figures/ :

  reconstruction_grid.png   real (top) vs tokenizer-decoded (bottom)
  dream_filmstrip.png       one imagined rollout, annotated with actions/rewards
  wm_vs_real.png            real vs dreamed frames + the mse_per_step curve
  curves_tokenizer.png      recon loss + codebook_usage vs step (the collapse)
  curves_world_model.png    token/reward/done accuracy vs step (the success)
  curves_policy.png         imagined_return + entropy vs update (it flatlines)
  return_vs_baseline.png    policy vs random vs oracle (catch_rate + return)

Warm cream (#F7F2E8) background.  Run with:  ./.plotvenv/bin/python make_student_figures.py
"""
import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CREAM = "#F7F2E8"
INK = "#2b2722"
ACCENT = "#8a5a2b"
GOOD = "#1f7a4d"
BAD = "#b23b3b"
BLUE = "#2f6fb0"
GOLD = "#c8892b"

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
    "axes.edgecolor": "#c9bfa8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 11,
    "axes.grid": True, "grid.color": "#e3d9c4", "grid.linewidth": 0.7,
    "font.family": "DejaVu Sans",
})

ACTION_LABEL = {0: "left", 1: "stay", 2: "right"}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def b64_to_img(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")) / 255.0


# ---------------------------------------------------------------------------
# 1. reconstruction_grid.png
# ---------------------------------------------------------------------------
def fig_reconstruction_grid():
    d = load("dreams_catch.json")
    rec = d["reconstructions"][:8]
    n = len(rec)
    fig, axes = plt.subplots(2, n, figsize=(1.55 * n, 3.9))
    for j, r in enumerate(rec):
        axes[0, j].imshow(b64_to_img(r["real"]))
        axes[1, j].imshow(b64_to_img(r["recon"]))
        for row in (0, 1):
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
            for s in axes[row, j].spines.values():
                s.set_edgecolor("#c9bfa8")
    axes[0, 0].set_ylabel("real\nframe", fontsize=11, rotation=0, ha="right", va="center", labelpad=18)
    axes[1, 0].set_ylabel("decoded\n(16 tokens)", fontsize=11, rotation=0, ha="right", va="center", labelpad=18)
    fig.suptitle("Tokenizer reconstructions  —  recon MSE 0.0165  (good pixels, but codebook usage only 1.2%)",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    out = os.path.join(FIG, "reconstruction_grid.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] reconstruction_grid.png")


# ---------------------------------------------------------------------------
# 2. dream_filmstrip.png  (one imagined rollout, left -> right)
# ---------------------------------------------------------------------------
def fig_dream_filmstrip():
    d = load("dreams_catch.json")
    # pick the dream with the most action variety for a lively strip
    dreams = d["dreams"]
    dr = max(dreams, key=lambda x: len(set(x["actions"])))
    frames = dr["frames"]
    actions = dr["actions"]
    rewards = dr["rewards"]
    dones = dr["dones"]
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n, 2.5))
    for t in range(n):
        ax = axes[t]
        ax.imshow(b64_to_img(frames[t]))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#c9bfa8")
        if t == 0:
            ax.set_title("t=0\n(real seed)", fontsize=9)
        else:
            a = actions[t - 1]
            r = rewards[t - 1]
            done = dones[t - 1]
            lbl = f"t={t}\n{ACTION_LABEL.get(a, a)}"
            if r != 0:
                lbl += f"  r={r:+.0f}"
            if done:
                lbl += "  [done]"
            ax.set_title(lbl, fontsize=9, color=ACCENT if t > 0 else INK)
    fig.suptitle("One imagined rollout  —  every frame after t=0 is hallucinated by the world model "
                 "(no real env). Arrow of paddle follows the action.",
                 fontsize=11.5, y=1.06)
    fig.tight_layout()
    out = os.path.join(FIG, "dream_filmstrip.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] dream_filmstrip.png")


# ---------------------------------------------------------------------------
# 3. wm_vs_real.png  (real vs dreamed per step + mse_per_step curve)
# ---------------------------------------------------------------------------
def fig_wm_vs_real():
    d = load("dreams_catch.json")
    cmp = d["wm_vs_real"][0]
    real, dream = cmp["real"], cmp["dream"]
    mse = cmp["mse_per_step"]
    actions = cmp["actions"]
    n = min(len(real), len(dream))

    fig = plt.figure(figsize=(1.5 * n, 6.2))
    gs = fig.add_gridspec(3, n, height_ratios=[1.0, 1.0, 1.15], hspace=0.35, wspace=0.15)

    for t in range(n):
        axr = fig.add_subplot(gs[0, t])
        axr.imshow(b64_to_img(real[t])); axr.set_xticks([]); axr.set_yticks([])
        ttl = f"t={t}"
        if t < len(actions):
            ttl += f"\n{ACTION_LABEL.get(actions[t], actions[t])}"
        axr.set_title(ttl, fontsize=9)
        for s in axr.spines.values():
            s.set_edgecolor("#c9bfa8")
        if t == 0:
            axr.set_ylabel("real", fontsize=10, rotation=0, ha="right", va="center", labelpad=10)

        axd = fig.add_subplot(gs[1, t])
        axd.imshow(b64_to_img(dream[t])); axd.set_xticks([]); axd.set_yticks([])
        for s in axd.spines.values():
            s.set_edgecolor("#c9bfa8")
        if t == 0:
            axd.set_ylabel("dream", fontsize=10, rotation=0, ha="right", va="center", labelpad=10)

    # mse curve spanning the full width
    axm = fig.add_subplot(gs[2, :])
    steps = list(range(len(mse)))
    axm.plot(steps, mse, "-o", color=BAD, ms=5, lw=1.8)
    axm.axhline(0.0165, color=GOOD, ls="--", lw=1.1, alpha=0.8, label="tokenizer recon floor (0.0165)")
    axm.set_xlabel("imagination step")
    axm.set_ylabel("per-step MSE\n(dream vs real)")
    axm.set_xlim(-0.3, len(mse) - 0.7)
    axm.set_ylim(0, max(0.022, max(mse) * 1.15))
    axm.set_title("Dream fidelity: coherent early, drifting late (mean MSE ≈ 0.02)", fontsize=11)
    axm.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig.suptitle("World model vs. real environment under identical actions", fontsize=13, y=0.98)
    out = os.path.join(FIG, "wm_vs_real.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] wm_vs_real.png")


# ---------------------------------------------------------------------------
# 4. curves_tokenizer.png  (recon loss + codebook usage — the collapse)
# ---------------------------------------------------------------------------
def fig_curves_tokenizer():
    d = load("tokenizer_catch_r0.json")
    log = d["log"]
    step = [e["step"] for e in log]
    recon = [e["recon"] for e in log]
    usage = [e["codebook_usage"] for e in log]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(step, recon, "-o", color=ACCENT, ms=5, lw=1.9, label="reconstruction MSE")
    ax.set_xlabel("tokenizer training step")
    ax.set_ylabel("reconstruction MSE", color=ACCENT)
    ax.tick_params(axis="y", colors=ACCENT)
    ax.annotate(f"final recon MSE {recon[-1]:.4f}",
                xy=(step[-1], recon[-1]), xytext=(step[-1] * 0.42, recon[-1] + 0.11),
                fontsize=10, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))

    ax2 = ax.twinx()
    ax2.plot(step, [u * 100 for u in usage], "--s", color=BAD, ms=4, lw=1.6, label="codebook usage")
    ax2.set_ylabel("codebook usage (%)", color=BAD)
    ax2.tick_params(axis="y", colors=BAD)
    ax2.set_ylim(0, 62)
    ax2.grid(False)
    ax2.annotate("COLLAPSE\nto ~1.2% of 256 codes\n(≈ 3 codes used)",
                 xy=(step[2], usage[2] * 100), xytext=(step[3], 30),
                 fontsize=9.5, color=BAD, ha="left",
                 arrowprops=dict(arrowstyle="->", color=BAD, lw=1.1))

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", framealpha=0.92)
    ax.set_title("Tokenizer training: pixels get sharp, but the codebook collapses",
                 fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(FIG, "curves_tokenizer.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] curves_tokenizer.png")


# ---------------------------------------------------------------------------
# 5. curves_world_model.png  (token/reward/done accuracy — the success)
# ---------------------------------------------------------------------------
def fig_curves_world_model():
    d = load("world_model_catch_r0.json")
    log = d["log"]
    step = [e["step"] for e in log]
    tacc = [e["token_acc"] for e in log]
    racc = [e["reward_acc"] for e in log]
    dacc = [e["done_acc"] for e in log]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(step, tacc, "-o", color=GOOD, ms=5, lw=1.9, label=f"next-token acc  (→ {tacc[-1]*100:.1f}%)")
    ax.plot(step, racc, "-s", color=BLUE, ms=4, lw=1.6, label=f"reward acc  (→ {racc[-1]*100:.1f}%)")
    ax.plot(step, dacc, "-^", color=GOLD, ms=4, lw=1.6, label=f"done acc  (→ {dacc[-1]*100:.1f}%)")
    ax.set_xlabel("world-model training step")
    ax.set_ylabel("prediction accuracy")
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.set_title("World-model training: the easy part works\n"
                 "learns Catch dynamics to 98% next-token accuracy",
                 fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(FIG, "curves_world_model.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] curves_world_model.png")


# ---------------------------------------------------------------------------
# 6. curves_policy.png  (imagined return + entropy — it flatlines)
# ---------------------------------------------------------------------------
def fig_curves_policy():
    d = load("policy_catch_r0.json")
    e_eval = load("eval_catch.json")
    log = d["log"]
    upd = [e["update"] for e in log]
    ret = [e["imagined_return"] for e in log]
    ent = [e["entropy"] for e in log]
    rand_ret = e_eval["random_baseline"]["return_mean"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(upd, ret, "-o", color=BAD, ms=5, lw=1.9, label="imagined return")
    ax.axhline(rand_ret, color="#9a8f78", ls="--", lw=1.4,
               label=f"random-policy return ({rand_ret:+.2f})")
    ax.set_xlabel("actor-critic update")
    ax.set_ylabel("imagined return", color=BAD)
    ax.tick_params(axis="y", colors=BAD)
    ax.set_ylim(-1.0, 0.05)

    ax2 = ax.twinx()
    ax2.plot(upd, ent, "--", color=BLUE, lw=1.6, label="policy entropy")
    ax2.set_ylabel("policy entropy", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.set_ylim(0, 1.2)
    ax2.grid(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower left", framealpha=0.92, fontsize=9.5)
    ax.set_title("Policy training in imagination: the hard part\n"
                 "return stays ≈ random; entropy never collapses → no policy learned",
                 fontsize=12.5)
    fig.tight_layout()
    out = os.path.join(FIG, "curves_policy.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] curves_policy.png")


# ---------------------------------------------------------------------------
# 7. return_vs_baseline.png  (policy vs random vs oracle: return + catch rate)
# ---------------------------------------------------------------------------
def fig_return_vs_baseline():
    e = load("eval_catch.json")
    order = [("random_baseline", "random", "#9a8f78"),
             ("policy", "IRIS policy\n(imagination-trained)", BAD),
             ("oracle_baseline", "oracle\n(optimal)", BLUE)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    labels = [lab for k, lab, c in order]
    rets = [e[k]["return_mean"] for k, lab, c in order]
    errs = [e[k]["return_std"] for k, lab, c in order]
    cols = [c for k, lab, c in order]
    x = np.arange(len(order))

    ax1.bar(x, rets, yerr=errs, color=cols, capsize=5, edgecolor="#5a5040", width=0.62)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("mean episode return")
    ax1.axhline(0, color="#5a5040", lw=0.8)
    ax1.set_title(f"Mean return over {e['n_episodes']} eval episodes")
    for xi, v in zip(x, rets):
        ax1.text(xi, v + (0.05 if v >= 0 else -0.13), f"{v:+.2f}",
                 ha="center", fontsize=11, fontweight="bold")

    crs = [e[k]["catch_rate"] for k, lab, c in order]
    ax2.bar(x, crs, color=cols, edgecolor="#5a5040", width=0.62)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("catch rate")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Catch rate (fraction of balls caught)")
    for xi, v in zip(x, crs):
        ax2.text(xi, v + 0.03, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("The honest result: the imagination-trained policy ≈ random (0.11 vs 0.10 catch rate); "
                 "oracle = 1.0",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "return_vs_baseline.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] return_vs_baseline.png")


if __name__ == "__main__":
    print("building student figures from", RES)
    fig_reconstruction_grid()
    fig_dream_filmstrip()
    fig_wm_vs_real()
    fig_curves_tokenizer()
    fig_curves_world_model()
    fig_curves_policy()
    fig_return_vs_baseline()
    print("done ->", FIG)
