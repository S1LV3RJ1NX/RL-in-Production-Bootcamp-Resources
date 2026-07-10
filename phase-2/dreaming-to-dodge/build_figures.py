"""Build all paper/demo figures from the REAL synced results JSON.

Reads results/*.json (pulled off the Modal Volume by `evals.py::sync`) and the
dream/reconstruction base64 blobs in results/dreams_catch.json, and writes:

  figures/learning_curves.png       tokenizer recon, WM token-acc, policy return
  figures/reconstructions.png       real vs tokenizer-decoded frames
  figures/dream_filmstrip.png       imagined (dreamed) policy rollouts
  figures/wm_vs_real.png            world model vs real env under same actions
  figures/policy_vs_baseline.png    final return: policy / random / oracle
  figures/dreams/dream_*.gif        animated imagined rollouts (copied from vol)

Everything is REAL — no synthetic numbers.  Warm cream background (#F7F2E8).
Run with the plotting venv:  ./.plotvenv/bin/python build_figures.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(FIG, "dreams"), exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
    "axes.edgecolor": "#c9bfa8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 11,
    "axes.grid": True, "grid.color": "#e3d9c4", "grid.linewidth": 0.7,
})


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def b64_to_img(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")) / 255.0


def rounds_available(prefix):
    r = 0
    out = []
    while True:
        d = load(f"{prefix}_catch_r{r}.json")
        if d is None:
            break
        out.append(d)
        r += 1
    return out


# ---------------------------------------------------------------------------
# 1. Learning curves  (tokenizer recon / WM token-acc / policy imagined return)
# ---------------------------------------------------------------------------
def fig_learning_curves():
    toks = rounds_available("tokenizer")
    wms = rounds_available("world_model")
    pols = rounds_available("policy")
    if not (toks and wms and pols):
        print("  [skip] learning_curves: missing per-round logs")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # tokenizer reconstruction loss (concatenate rounds, global step)
    gstep, recon, usage = [], [], []
    off = 0
    for d in toks:
        for e in d["log"]:
            gstep.append(off + e["step"]); recon.append(e["recon"]); usage.append(e["codebook_usage"])
        off += d["log"][-1]["step"] + 1
    ax = axes[0]
    ax.plot(gstep, recon, "-o", color=ACCENT, ms=3, lw=1.6)
    ax.set_title("Tokenizer reconstruction loss")
    ax.set_xlabel("training step"); ax.set_ylabel("L1 recon loss")
    ax2 = ax.twinx()
    ax2.plot(gstep, usage, "--", color=BLUE, lw=1.2, alpha=0.8)
    ax2.set_ylabel("codebook usage", color=BLUE); ax2.tick_params(colors=BLUE)
    ax2.grid(False); ax2.set_ylim(0, 1)

    # world-model token accuracy
    gstep, tacc, racc, dacc = [], [], [], []
    off = 0
    for d in wms:
        for e in d["log"]:
            gstep.append(off + e["step"]); tacc.append(e["token_acc"])
            racc.append(e["reward_acc"]); dacc.append(e["done_acc"])
        off += d["log"][-1]["step"] + 1
    ax = axes[1]
    ax.plot(gstep, tacc, "-o", color=ACCENT, ms=3, lw=1.6, label="next-token acc")
    ax.plot(gstep, racc, "-s", color=GOOD, ms=3, lw=1.3, label="reward acc")
    ax.plot(gstep, dacc, "-^", color=BLUE, ms=3, lw=1.3, label="done acc")
    ax.set_title("World-model prediction accuracy")
    ax.set_xlabel("training step"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)

    # policy imagined return
    gstep, ret, ent = [], [], []
    off = 0
    for d in pols:
        for e in d["log"]:
            gstep.append(off + e["update"]); ret.append(e["imagined_return"]); ent.append(e["entropy"])
        off += d["log"][-1]["update"] + 1
    ax = axes[2]
    ax.plot(gstep, ret, "-o", color=GOOD, ms=3, lw=1.6)
    ax.set_title("Policy return in imagination")
    ax.set_xlabel("actor-critic update"); ax.set_ylabel("imagined return")
    ax3 = ax.twinx()
    ax3.plot(gstep, ent, "--", color="#b06f2b", lw=1.1, alpha=0.7)
    ax3.set_ylabel("policy entropy", color="#b06f2b"); ax3.tick_params(colors="#b06f2b")
    ax3.grid(False)

    fig.suptitle("IRIS-from-scratch learning curves (Catch) — real Modal run",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning_curves.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] learning_curves.png")


# ---------------------------------------------------------------------------
# 2. Reconstructions  (real vs tokenizer-decoded)
# ---------------------------------------------------------------------------
def fig_reconstructions():
    d = load("dreams_catch.json")
    if not d or not d.get("reconstructions"):
        print("  [skip] reconstructions: no dreams json")
        return
    rec = d["reconstructions"][:8]
    n = len(rec)
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 3.9))
    for j, r in enumerate(rec):
        axes[0, j].imshow(b64_to_img(r["real"])); axes[0, j].axis("off")
        axes[1, j].imshow(b64_to_img(r["recon"])); axes[1, j].axis("off")
    axes[0, 0].set_title("real", loc="left", fontsize=10)
    axes[1, 0].set_title("decoded", loc="left", fontsize=10)
    fig.suptitle("Tokenizer reconstructions — top: real frame, bottom: decoded from 16 discrete tokens",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "reconstructions.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] reconstructions.png")


# ---------------------------------------------------------------------------
# 3. Dream filmstrip  (imagined policy rollouts)
# ---------------------------------------------------------------------------
def fig_dream_filmstrip():
    d = load("dreams_catch.json")
    if not d or not d.get("dreams"):
        print("  [skip] dream_filmstrip: no dreams")
        return
    dreams = d["dreams"][:3]
    maxlen = max(len(dr["frames"]) for dr in dreams)
    fig, axes = plt.subplots(len(dreams), maxlen, figsize=(1.35 * maxlen, 1.6 * len(dreams)))
    if len(dreams) == 1:
        axes = axes[None, :]
    for i, dr in enumerate(dreams):
        frames = dr["frames"]
        rewards = dr.get("rewards", [])
        for t in range(maxlen):
            ax = axes[i, t]
            if t < len(frames):
                ax.imshow(b64_to_img(frames[t]))
                lbl = f"t={t}"
                if t > 0 and t - 1 < len(dr.get("actions", [])):
                    a = dr["actions"][t - 1]
                    lbl += f" a={['L','·','R'][a] if a in (0,1,2) else a}"
                if t - 1 < len(rewards) and t > 0 and rewards[t - 1] != 0:
                    lbl += f" r={rewards[t-1]:+.0f}"
                ax.set_title(lbl, fontsize=8)
            ax.axis("off")
        axes[i, 0].set_ylabel(f"dream {i}", fontsize=9)
    fig.suptitle("Imagined (dreamed) rollouts — every frame after t=0 is hallucinated by the world model",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dream_filmstrip.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] dream_filmstrip.png")


# ---------------------------------------------------------------------------
# 4. World model vs real  (same oracle actions)
# ---------------------------------------------------------------------------
def fig_wm_vs_real():
    d = load("dreams_catch.json")
    if not d or not d.get("wm_vs_real"):
        print("  [skip] wm_vs_real: none")
        return
    cmp = d["wm_vs_real"][0]
    real, dream = cmp["real"], cmp["dream"]
    mse = cmp.get("mse_per_step", [])
    n = min(len(real), len(dream))
    fig, axes = plt.subplots(2, n, figsize=(1.35 * n, 3.6))
    for t in range(n):
        axes[0, t].imshow(b64_to_img(real[t])); axes[0, t].axis("off")
        axes[1, t].imshow(b64_to_img(dream[t])); axes[1, t].axis("off")
        ttl = f"t={t}"
        if t < len(mse):
            ttl += f"\nmse={mse[t]:.3f}"
        axes[0, t].set_title(ttl, fontsize=8)
    axes[0, 0].set_ylabel("real env", fontsize=9)
    axes[1, 0].set_ylabel("world model", fontsize=9)
    fig.suptitle("World model vs. real environment under identical (oracle) actions",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "wm_vs_real.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] wm_vs_real.png")


# ---------------------------------------------------------------------------
# 5. Policy return vs baseline bar
# ---------------------------------------------------------------------------
def fig_policy_vs_baseline():
    e = load("eval_catch.json")
    if not e:
        print("  [skip] policy_vs_baseline: no eval json")
        return
    labels, vals, errs, colors = [], [], [], []
    order = [("random_baseline", "random", "#9a8f78"),
             ("policy", "IRIS policy\n(imagination-trained)", GOOD),
             ("oracle_baseline", "oracle\n(optimal)", BLUE)]
    for key, lab, col in order:
        if key in e:
            labels.append(lab); vals.append(e[key]["return_mean"])
            errs.append(e[key].get("return_std", 0)); colors.append(col)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(labels))
    ax1.bar(x, vals, yerr=errs, color=colors, capsize=5, edgecolor="#5a5040", width=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("mean episode return"); ax1.axhline(0, color="#5a5040", lw=0.8)
    ax1.set_title(f"Final return over {e.get('n_episodes','?')} eval episodes")
    for xi, v in zip(x, vals):
        ax1.text(xi, v + (0.03 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", fontsize=10, fontweight="bold")

    # catch-rate panel
    labels2, cr, colors2 = [], [], []
    for key, lab, col in order:
        if key in e and "catch_rate" in e[key]:
            labels2.append(lab); cr.append(e[key]["catch_rate"]); colors2.append(col)
    x2 = np.arange(len(labels2))
    ax2.bar(x2, cr, color=colors2, edgecolor="#5a5040", width=0.6)
    ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=10)
    ax2.set_ylabel("catch rate"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Catch rate (fraction of balls caught)")
    for xi, v in zip(x2, cr):
        ax2.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Does the imagination-trained policy beat the baselines?", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "policy_vs_baseline.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  [ok] policy_vs_baseline.png")


if __name__ == "__main__":
    print("building figures from", RES)
    fig_learning_curves()
    fig_reconstructions()
    fig_dream_filmstrip()
    fig_wm_vs_real()
    fig_policy_vs_baseline()
    print("done ->", FIG)
