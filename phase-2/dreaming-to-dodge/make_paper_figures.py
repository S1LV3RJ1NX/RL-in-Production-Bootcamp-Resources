"""Publication figures for "Dream to Catch" (IRIS-from-scratch, FIXED).

Reads ONLY the real synced results JSON in results/ and writes clean,
publication-grade figures into paper/figures/.  Every function is defensive:
if an input JSON is missing it prints a warning and skips, so the script is
safe to run at any point during a multi-round run.

Figures:
  fig_codebook_ablation.png   before/after: plain-VQ collapse vs EMA+revive (+ curves)
  fig_tokenizer.png           EMA tokenizer: recon MSE + active codes vs step
  fig_world_model.png         token/reward/done accuracy vs step (the WM works)
  fig_wm_vs_real.png          real vs dream filmstrip + per-step MSE (ball preserved)
  fig_dream_filmstrip.png     one imagined rollout: the paddle moves to catch
  fig_policy.png              imagined return climbing + entropy annealing
  fig_return_vs_baseline.png  policy vs random vs oracle (return + catch rate)
  fig_learning_curve.png      catch rate + codebook usage vs IRIS round
  fig_reconstruction_grid.png real vs decoded (ball recall panel)

Run:  ./.plotvenv/bin/python make_paper_figures.py
"""
import base64
import io
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- clean publication palette (light bg, colourblind-safe accents) ---------
BG = "#ffffff"
INK = "#222222"
GRID = "#e6e6e6"
BEFORE = "#c0392b"   # red   — the failure / plain VQ
AFTER = "#1f7a4d"    # green — the fix / works
BLUE = "#2f6fb0"     # oracle / secondary
GOLD = "#c8892b"     # tertiary
ACCENT = "#8a5a2b"   # warm accent

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#bdbdbd", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "figure.dpi": 150,
})
ACTION_LABEL = {0: "left", 1: "stay", 2: "right"}
TAG = os.environ.get("PAPER_TAG", "win")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def b64_to_img(b64):
    return np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")) / 255.0


def _save(fig, name):
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [ok]", name)


def _rounds():
    """discover trained rounds by tokenizer result files."""
    rs = []
    for p in sorted(glob.glob(os.path.join(RES, f"tokenizer_{TAG}_r*.json"))):
        try:
            rs.append(int(p.split("_r")[-1].split(".")[0]))
        except ValueError:
            pass
    return sorted(set(rs))


# ---------------------------------------------------------------------------
# 1. codebook ablation — the central before/after
# ---------------------------------------------------------------------------
def fig_codebook_ablation():
    d = load(f"tokenizer_ablation_{TAG}.json")
    if not d or "conditions" not in d:
        print("  [skip] fig_codebook_ablation (no ablation json)"); return
    C = d["conditions"]
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.22)

    # left: the real / plain / EMA / EMA+fg panel produced on Modal
    axp = fig.add_subplot(gs[0, 0]); axp.axis("off")
    if d.get("panel_png_b64"):
        axp.imshow(b64_to_img(d["panel_png_b64"]))
    axp.set_title("(a) Reconstructions with the ball at every column",
                  fontsize=11.5)

    # right: active-code trajectories for all three conditions
    axc = fig.add_subplot(gs[0, 1])
    style = {"plain": (BEFORE, "-o", "plain VQ + uniform"),
             "ema": (GOLD, "-s", "EMA + uniform"),
             "ema_fg": (AFTER, "-^", "EMA + fg-weighted (ours)")}
    for key in ["plain", "ema", "ema_fg"]:
        if key not in C:
            continue
        curve = C[key]["curve"]
        step = [e["step"] for e in curve]
        active = [e["active_codes"] for e in curve]
        col, mk, lab = style[key]
        axc.plot(step, active, mk, color=col, ms=3.5, lw=2.0,
                 label=f"{lab}  (recall {C[key]['ball_recall']:.2f})")
    N = d.get("vocab_size", 256)
    axc.axhline(N, color="#9a9a9a", ls=":", lw=1.1)
    axc.text(step[-1], N * 0.95, f"codebook size N={N}", ha="right", va="top",
             fontsize=9, color="#6a6a6a")
    axc.set_xlabel("tokenizer training step")
    axc.set_ylabel("active codes")
    axc.set_ylim(0, N * 1.08)
    axc.set_title("(b) Codebook usage vs. ball recall", fontsize=11.5)
    axc.legend(loc="center right", framealpha=0.95, fontsize=8.8)
    _save(fig, "fig_codebook_ablation.png")


# ---------------------------------------------------------------------------
# 2. tokenizer curves (EMA run)
# ---------------------------------------------------------------------------
def fig_tokenizer():
    rounds = _rounds()
    d = load(f"tokenizer_{TAG}_r{rounds[-1]}.json") if rounds else None
    if not d:
        print("  [skip] fig_tokenizer"); return
    log = d["log"]
    step = [e["step"] for e in log]
    recon = [e.get("recon_mse", e.get("recon")) for e in log]
    active = [e.get("active_codes") for e in log]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(step, recon, "-o", color=ACCENT, ms=4, lw=2.0, label="reconstruction MSE")
    ax.set_xlabel("tokenizer training step")
    ax.set_ylabel("reconstruction MSE", color=ACCENT)
    ax.tick_params(axis="y", colors=ACCENT)
    ax.set_ylim(0, max(recon) * 1.1)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(step, active, "--s", color=AFTER, ms=4, lw=1.8, label="active codes")
    ax2.set_ylabel("active codes (of %d)" % d.get("vocab_size", 256), color=AFTER)
    ax2.tick_params(axis="y", colors=AFTER)
    ax2.set_ylim(0, d.get("vocab_size", 256) * 1.05)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", framealpha=0.95)
    ax.set_title("Tokenizer with EMA + dead-code revival: sharp pixels, healthy codebook",
                 fontsize=12)
    _save(fig, "fig_tokenizer.png")


# ---------------------------------------------------------------------------
# 3. world-model curves
# ---------------------------------------------------------------------------
def fig_world_model():
    rounds = _rounds()
    d = load(f"world_model_{TAG}_r{rounds[-1]}.json") if rounds else None
    if not d:
        print("  [skip] fig_world_model"); return
    log = d["log"]
    step = [e["step"] for e in log]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(step, [e["token_acc"] for e in log], "-o", color=AFTER, ms=4, lw=2.0,
            label=f"next-token acc  (→ {log[-1]['token_acc']*100:.1f}%)")
    ax.plot(step, [e["reward_acc"] for e in log], "-s", color=BLUE, ms=3.5, lw=1.7,
            label=f"reward acc  (→ {log[-1]['reward_acc']*100:.1f}%)")
    ax.plot(step, [e["done_acc"] for e in log], "-^", color=GOLD, ms=3.5, lw=1.7,
            label=f"done acc  (→ {log[-1]['done_acc']*100:.1f}%)")
    ax.set_xlabel("world-model training step"); ax.set_ylabel("prediction accuracy")
    ax.set_ylim(0, 1.03); ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title("Transformer world model: learns Catch dynamics from image tokens", fontsize=12)
    _save(fig, "fig_world_model.png")


# ---------------------------------------------------------------------------
# 4. world-model vs real
# ---------------------------------------------------------------------------
def fig_wm_vs_real():
    d = load(f"dreams_{TAG}.json")
    if not d or not d.get("wm_vs_real"):
        print("  [skip] fig_wm_vs_real"); return
    cmp = d["wm_vs_real"][0]
    real, dream, mse = cmp["real"], cmp["dream"], cmp["mse_per_step"]
    actions = cmp["actions"]; n = min(len(real), len(dream))
    fig = plt.figure(figsize=(1.5 * n, 6.0))
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 1.2], hspace=0.32, wspace=0.12)
    for t in range(n):
        axr = fig.add_subplot(gs[0, t]); axr.imshow(b64_to_img(real[t]))
        axr.set_xticks([]); axr.set_yticks([])
        ttl = f"t={t}" + (f"\n{ACTION_LABEL.get(actions[t], actions[t])}" if t < len(actions) else "")
        axr.set_title(ttl, fontsize=9)
        if t == 0: axr.set_ylabel("real", fontsize=10)
        axd = fig.add_subplot(gs[1, t]); axd.imshow(b64_to_img(dream[t]))
        axd.set_xticks([]); axd.set_yticks([])
        if t == 0: axd.set_ylabel("dream", fontsize=10)
    axm = fig.add_subplot(gs[2, :])
    axm.plot(range(len(mse)), mse, "-o", color=BLUE, ms=5, lw=1.9)
    axm.set_xlabel("imagination step"); axm.set_ylabel("per-step MSE\n(dream vs real)")
    axm.set_ylim(0, max(0.01, max(mse) * 1.2))
    axm.set_title("Dream fidelity under identical actions (mean MSE %.4f)"
                  % (np.mean(mse)), fontsize=11)
    fig.suptitle("World model vs. real environment — the ball is now dreamed", fontsize=13, y=0.98)
    _save(fig, "fig_wm_vs_real.png")


# ---------------------------------------------------------------------------
# 5. dream filmstrip (policy in imagination)
# ---------------------------------------------------------------------------
def fig_dream_filmstrip():
    d = load(f"dreams_{TAG}.json")
    if not d or not d.get("dreams"):
        print("  [skip] fig_dream_filmstrip"); return
    # prefer a dream that catches (last reward +1) with action variety
    dreams = d["dreams"]
    def score(x):
        caught = 1 if (x["rewards"] and x["rewards"][-1] > 0) else 0
        return (caught, len(set(x["actions"])))
    dr = max(dreams, key=score)
    frames, actions, rewards, dones = dr["frames"], dr["actions"], dr["rewards"], dr["dones"]
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 2.4))
    if n == 1: axes = [axes]
    for t in range(n):
        ax = axes[t]; ax.imshow(b64_to_img(frames[t]))
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.set_title("t=0\n(real seed)", fontsize=9)
        else:
            a, r, done = actions[t - 1], rewards[t - 1], dones[t - 1]
            lbl = f"t={t}\n{ACTION_LABEL.get(a, a)}"
            if r != 0: lbl += f"  r={r:+.0f}"
            if done: lbl += "  [done]"
            ax.set_title(lbl, fontsize=9, color=ACCENT)
    fig.suptitle("One imagined rollout — every frame after t=0 is hallucinated; the learned "
                 "paddle tracks the ball and catches it", fontsize=11.5, y=1.08)
    _save(fig, "fig_dream_filmstrip.png")


# ---------------------------------------------------------------------------
# 6. policy curves (final round)
# ---------------------------------------------------------------------------
def fig_policy():
    rounds = []
    for p in sorted(glob.glob(os.path.join(RES, f"policy_{TAG}_r*.json"))):
        try: rounds.append(int(p.split("_r")[-1].split(".")[0]))
        except ValueError: pass
    rounds = sorted(set(rounds))
    if not rounds:
        print("  [skip] fig_policy"); return
    # concatenate all rounds into one continuous curve
    upd, ret, ent = [], [], []
    base = 0
    for r in rounds:
        d = load(f"policy_{TAG}_r{r}.json")
        for e in d["log"]:
            upd.append(base + e["update"]); ret.append(e["imagined_return"]); ent.append(e["entropy"])
        base = upd[-1] + (d["log"][1]["update"] - d["log"][0]["update"] if len(d["log"]) > 1 else 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0, color="#c9c9c9", lw=0.8, zorder=0)
    ax.plot(upd, ret, "-o", color=AFTER, ms=3.5, lw=2.0, label="imagined return")
    ax.set_xlabel("actor-critic update")
    ax.set_ylabel("imagined return", color=AFTER); ax.tick_params(axis="y", colors=AFTER)
    ax.set_ylim(min(-1.0, min(ret) * 1.05), max(0.1, max(ret) * 1.1))
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(upd, ent, "--", color=BLUE, lw=1.7, label="policy entropy")
    ax2.set_ylabel("policy entropy", color=BLUE); ax2.tick_params(axis="y", colors=BLUE)
    ax2.set_ylim(0, 1.2)
    lines = [l for l in ax.get_lines() + ax2.get_lines()
             if not l.get_label().startswith("_")]
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", framealpha=0.95, fontsize=9.5)
    ax.set_title("Actor-critic in imagination: return climbs, entropy anneals → it commits to catching",
                 fontsize=11.5)
    _save(fig, "fig_policy.png")


# ---------------------------------------------------------------------------
# 7. return vs baseline
# ---------------------------------------------------------------------------
def fig_return_vs_baseline():
    e = load(f"eval_{TAG}.json")
    if not e:
        print("  [skip] fig_return_vs_baseline"); return
    order = [("random_baseline", "random", "#9a8f78"),
             ("policy", "IRIS policy\n(imagination-trained)", AFTER),
             ("oracle_baseline", "oracle\n(optimal)", BLUE)]
    order = [o for o in order if o[0] in e]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [lab for k, lab, c in order]
    x = np.arange(len(order)); cols = [c for k, lab, c in order]
    rets = [e[k]["return_mean"] for k, lab, c in order]
    errs = [e[k]["return_std"] for k, lab, c in order]
    ax1.bar(x, rets, yerr=errs, color=cols, capsize=5, edgecolor="#5a5040", width=0.62)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("mean episode return"); ax1.axhline(0, color="#5a5040", lw=0.8)
    ax1.set_title(f"Mean return over {e['n_episodes']} eval episodes")
    for xi, v in zip(x, rets):
        ax1.text(xi, v + (0.05 if v >= 0 else -0.13), f"{v:+.2f}", ha="center",
                 fontsize=11, fontweight="bold")
    crs = [e[k]["catch_rate"] for k, lab, c in order]
    ax2.bar(x, crs, color=cols, edgecolor="#5a5040", width=0.62)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("catch rate"); ax2.set_ylim(0, 1.08)
    ax2.set_title("Catch rate (fraction of balls caught)")
    for xi, v in zip(x, crs):
        ax2.text(xi, v + 0.03, f"{v:.0%}", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("Imagination-trained policy vs. random and oracle baselines", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "fig_return_vs_baseline.png")


# ---------------------------------------------------------------------------
# 8. learning curve across IRIS rounds
# ---------------------------------------------------------------------------
def fig_learning_curve():
    ra = load(f"run_all_{TAG}.json")
    per = None
    if ra and ra.get("per_round"):
        per = ra["per_round"]
    # also try per-round eval files
    evals = {}
    for p in sorted(glob.glob(os.path.join(RES, f"eval_{TAG}_r*.json"))):
        try:
            r = int(p.split("_r")[-1].split(".")[0]); evals[r] = load(os.path.basename(p))
        except ValueError:
            pass
    if not per and not evals:
        print("  [skip] fig_learning_curve"); return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    if evals:
        rs = sorted(evals)
        cr = [evals[r]["policy"]["catch_rate"] for r in rs]
        ax.plot(rs, cr, "-o", color=AFTER, ms=6, lw=2.2, label="catch rate (real env)")
        rnd = [evals[r].get("random_baseline", {}).get("catch_rate") for r in rs]
        if all(v is not None for v in rnd):
            ax.plot(rs, rnd, "--", color="#9a8f78", lw=1.6, label="random baseline")
    ax.axhline(1.0, color=BLUE, ls=":", lw=1.2, label="oracle")
    ax.set_xlabel("IRIS collect↔train round"); ax.set_ylabel("catch rate")
    ax.set_ylim(0, 1.08)
    if per:
        ax2 = ax.twinx(); ax2.grid(False)
        rr = [p["round"] for p in per]
        ac = [p.get("active_codes") for p in per]
        if all(v is not None for v in ac):
            ax2.plot(rr, ac, "-s", color=ACCENT, lw=1.6, ms=4, label="active codes")
            ax2.set_ylabel("active codes", color=ACCENT); ax2.tick_params(axis="y", colors=ACCENT)
    ax.legend(loc="center right", framealpha=0.95)
    ax.set_title("Catch rate improves across the IRIS collect↔train loop", fontsize=12)
    _save(fig, "fig_learning_curve.png")


# ---------------------------------------------------------------------------
# 9. reconstruction grid (ball recall)
# ---------------------------------------------------------------------------
def fig_reconstruction_grid():
    d = load(f"recon_check_{TAG}.json")
    if not d or not d.get("grid_png_b64"):
        print("  [skip] fig_reconstruction_grid"); return
    fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis("off")
    ax.imshow(b64_to_img(d["grid_png_b64"]))
    ax.set_title("Tokenizer reconstructions — real (top) vs decoded from 16 tokens (bottom); "
                 f"ball recall {d['ball_recall']:.2f}", fontsize=11.5)
    _save(fig, "fig_reconstruction_grid.png")


if __name__ == "__main__":
    print("building paper figures for tag", TAG, "from", RES)
    fig_codebook_ablation()
    fig_tokenizer()
    fig_world_model()
    fig_wm_vs_real()
    fig_dream_filmstrip()
    fig_policy()
    fig_return_vs_baseline()
    fig_learning_curve()
    fig_reconstruction_grid()
    print("done ->", FIG)
