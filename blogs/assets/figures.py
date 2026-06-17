"""Reproducible figure generator for the RL blogs.

Every numeric/data figure in the blogs is produced here so the charts stay
consistent and can be regenerated from scratch. Output is SVG (vector, tiny,
renders on GitHub markdown and in the Astro portfolio with no JavaScript).

House style matches the portfolio palette:
    canvas  #FAFAFA   ink     #0A0A0A
    accent  #C8421A   muted   #525252   divider #E5E5E5

Run:
    uv run python blogs/assets/figures.py
    uv run python blogs/assets/figures.py --blog 01   # only blog 01's figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ----------------------------------------------------------------------------
# Palette + house style (shared across every figure)
# ----------------------------------------------------------------------------
INK = "#0A0A0A"
ACCENT = "#C8421A"
MUTED = "#525252"
DIVIDER = "#E5E5E5"
CANVAS = "#FAFAFA"
ACCENT_SOFT = "#E8A48F"

BLOGS_DIR = Path(__file__).resolve().parent.parent


def house_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": CANVAS,
            "axes.facecolor": CANVAS,
            "savefig.facecolor": CANVAS,
            "font.family": "sans-serif",
            "font.sans-serif": ["Geist", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.edgecolor": DIVIDER,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": DIVIDER,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "svg.fonttype": "none",  # keep text as text in the SVG
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, blog: str, name: str) -> None:
    out_dir = BLOGS_DIR / blog / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  wrote {path.relative_to(BLOGS_DIR)}")


# ----------------------------------------------------------------------------
# Blog 01 — RL intro + math prerequisites
# ----------------------------------------------------------------------------
BLOG01 = "01-rl-intro-and-prerequisites"


def fig_discounting() -> None:
    """How a single future reward shrinks with the discount factor gamma."""
    k = np.arange(0, 26)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for gamma, color in [(0.99, MUTED), (0.9, ACCENT), (0.7, ACCENT_SOFT)]:
        ax.plot(k, gamma**k, marker="o", ms=3.5, lw=2, color=color,
                label=rf"$\gamma = {gamma}$")
    ax.set_xlabel("steps into the future  $k$")
    ax.set_ylabel(r"weight on that reward  $\gamma^{\,k}$")
    ax.set_title("Discounting: how much a future reward is worth today")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False)
    save(fig, BLOG01, "fig-discounting")


def fig_lln() -> None:
    """Law of large numbers: the running sample mean of die rolls -> 3.5."""
    rng = np.random.default_rng(0)
    n = 2000
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for color, alpha in [(ACCENT_SOFT, 0.9), (MUTED, 0.7), (ACCENT, 1.0)]:
        rolls = rng.integers(1, 7, size=n)
        running = np.cumsum(rolls) / np.arange(1, n + 1)
        ax.plot(np.arange(1, n + 1), running, lw=1.6, color=color, alpha=alpha)
    ax.axhline(3.5, ls="--", lw=1.5, color=INK)
    ax.annotate("true mean  $\\mathbb{E}[X]=3.5$", xy=(n, 3.5),
                xytext=(n * 0.45, 3.95), color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("number of samples  (log scale)")
    ax.set_ylabel("running average of rolls")
    ax.set_title("Estimating an expectation by sampling (law of large numbers)")
    ax.set_ylim(1, 6)
    save(fig, BLOG01, "fig-lln")


def fig_entropy() -> None:
    """Entropy of a biased coin as a function of P(heads)."""
    p = np.linspace(1e-4, 1 - 1e-4, 500)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(p, h, lw=2.4, color=ACCENT)
    ax.scatter([0.5], [1.0], color=INK, zorder=5)
    ax.annotate("fair coin: max uncertainty\n$H = 1$ bit", xy=(0.5, 1.0),
                xytext=(0.5, 0.55), ha="center", color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.annotate("rigged coin:\nno uncertainty", xy=(0.97, 0.05),
                xytext=(0.72, 0.28), color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel(r"$P(\mathrm{heads})$")
    ax.set_ylabel("entropy  $H$  (bits)")
    ax.set_title("Entropy measures how spread out a distribution is")
    ax.set_ylim(0, 1.1)
    save(fig, BLOG01, "fig-entropy")


def fig_incremental_average() -> None:
    """Running/incremental average tracking a noisy reward stream."""
    rng = np.random.default_rng(3)
    true_mean = 5.0
    n = 60
    samples = true_mean + rng.normal(0, 3, size=n)
    alpha = 0.1
    est = np.zeros(n + 1)
    sample_mean = np.zeros(n)
    for t in range(n):
        est[t + 1] = est[t] + alpha * (samples[t] - est[t])
        sample_mean[t] = samples[: t + 1].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(np.arange(1, n + 1), samples, s=16, color=DIVIDER,
               edgecolor=MUTED, lw=0.5, label="observed reward")
    ax.plot(np.arange(1, n + 1), sample_mean, lw=1.6, color=MUTED,
            label="true running mean")
    ax.plot(np.arange(0, n + 1), est, lw=2.4, color=ACCENT,
            label=rf"incremental est. ($\alpha={alpha}$)")
    ax.axhline(true_mean, ls="--", lw=1.3, color=INK)
    ax.set_xlabel("update step  $t$")
    ax.set_ylabel("estimate of the mean")
    ax.set_title(r"The shape of every RL update:  $\mu \leftarrow \mu + \alpha\,(x - \mu)$")
    ax.legend(frameon=False, loc="lower right")
    save(fig, BLOG01, "fig-incremental-average")


def fig_surprise() -> None:
    """Surprise = -log2(p): rare events are surprising, certain events are not."""
    p = np.linspace(0.02, 1.0, 500)
    surprise = -np.log2(p)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(p, surprise, lw=2.4, color=ACCENT)
    for px, label in [(1.0, "certain: 0 bits"), (0.5, "1 in 2: 1 bit"),
                      (0.25, "1 in 4: 2 bits"), (0.125, "1 in 8: 3 bits")]:
        ax.scatter([px], [-np.log2(px)], color=INK, zorder=5, s=22)
        ax.annotate(label, xy=(px, -np.log2(px)),
                    xytext=(px + 0.04, -np.log2(px) + 0.25), color=MUTED, fontsize=10)
    ax.set_xlabel(r"probability of the outcome  $p$")
    ax.set_ylabel(r"surprise  $-\log_2 p$  (bits)")
    ax.set_title("Surprise: the rarer the event, the more it tells you")
    ax.set_ylim(0, 5.6)
    save(fig, BLOG01, "fig-surprise")


def fig_kl_policies() -> None:
    """Two policies over three actions, used in the worked KL example."""
    actions = ["left", "stay", "right"]
    pi1 = np.array([0.7, 0.2, 0.1])
    pi2 = np.array([0.4, 0.4, 0.2])
    x = np.arange(len(actions))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w / 2, pi1, width=w, color=ACCENT, label=r"$\pi_1$ (old policy)")
    ax.bar(x + w / 2, pi2, width=w, color=MUTED, label=r"$\pi_2$ (new policy)")
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.set_ylabel(r"$\pi(a \mid s)$")
    ax.set_title("How different are two policies? That gap is KL divergence")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False)
    save(fig, BLOG01, "fig-kl-policies")


def build_blog01() -> None:
    print(f"[{BLOG01}]")
    fig_discounting()
    fig_lln()
    fig_entropy()
    fig_incremental_average()
    fig_surprise()
    fig_kl_policies()


# ----------------------------------------------------------------------------
# Blog 02 — MDPs and the Bellman equation
# ----------------------------------------------------------------------------
BLOG02 = "02-mdps-and-bellman"


def fig_backup_diagram() -> None:
    """V/Q one-step lookahead tree: V branches over actions (π), Q branches over next states (p)."""
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.3, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")

    circle_kw = dict(lw=2, zorder=5)
    label_kw = dict(ha="center", va="center", fontsize=13, fontweight="bold", zorder=6)
    edge_kw = dict(lw=2, zorder=3)
    annot_kw = dict(fontsize=10, zorder=4)

    # V node (top)
    v_xy = (0, 3.0)
    ax.add_patch(plt.Circle(v_xy, 0.22, fc=CANVAS, ec=ACCENT, **circle_kw))
    ax.text(*v_xy, r"$V$", color=ACCENT, **label_kw)

    # Q nodes (middle row)
    q_positions = [(-1.0, 1.8), (0, 1.8), (1.0, 1.8)]
    q_labels = [r"$Q_1$", r"$Q_2$", r"$Q_3$"]
    for (qx, qy), ql in zip(q_positions, q_labels):
        ax.add_patch(plt.Circle((qx, qy), 0.22, fc=CANVAS, ec=INK, **circle_kw))
        ax.text(qx, qy, ql, color=INK, **label_kw)
        ax.plot([v_xy[0], qx], [v_xy[1] - 0.22, qy + 0.22], color=ACCENT, **edge_kw)

    # π labels on V->Q edges
    pi_labels = [r"$\pi(a_1)$", r"$\pi(a_2)$", r"$\pi(a_3)$"]
    pi_offsets = [(-0.75, 2.55), (0.18, 2.55), (0.75, 2.55)]
    for (px, py), pl in zip(pi_offsets, pi_labels):
        ax.text(px, py, pl, color=ACCENT, **annot_kw)

    # V' nodes (bottom row) — two per Q node (simplified)
    v_prime_x = [-1.3, -0.7, -0.3, 0.3, 0.7, 1.3]
    v_prime_y = [0.4] * 6
    q_parents = [0, 0, 1, 1, 2, 2]
    for i, (vx, vy) in enumerate(zip(v_prime_x, v_prime_y)):
        ax.add_patch(plt.Circle((vx, vy), 0.18, fc=CANVAS, ec=MUTED, **circle_kw))
        ax.text(vx, vy, r"$V'$", color=MUTED, fontsize=9, ha="center", va="center", zorder=6)
        qx, qy = q_positions[q_parents[i]]
        ax.plot([qx, vx], [qy - 0.22, vy + 0.18], color=MUTED, lw=1.5, zorder=3)

    # p labels
    ax.text(-1.0, 1.1, r"$p(s',r|s,a)$", color=MUTED, fontsize=10, ha="center")

    # Layer labels
    ax.text(-1.45, 3.0, "state $s$", color=ACCENT, fontsize=10, va="center")
    ax.text(-1.45, 1.8, "action $a$", color=INK, fontsize=10, va="center")
    ax.text(-1.45, 0.4, "next $s'$", color=MUTED, fontsize=10, va="center")

    ax.set_title("Backup diagram: V decomposes into Q, Q decomposes into V'", fontsize=12, pad=12)
    save(fig, BLOG02, "fig-backup-diagram")


def fig_v_vs_q() -> None:
    """Bar chart of Q(s,a) for one FrozenLake state with V(s) as horizontal line."""
    actions = ["Left (0)", "Down (1)", "Right (2)", "Up (3)"]
    q_vals = np.array([0.028, 0.018, 0.035, 0.022])
    v_val = q_vals.mean()  # uniform random policy: V = mean of Q

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = [MUTED, MUTED, ACCENT, MUTED]
    bars = ax.bar(actions, q_vals, color=colors, width=0.55, edgecolor=INK, lw=0.8)
    ax.axhline(v_val, ls="--", lw=2, color=ACCENT_SOFT,
               label=rf"$V^\pi(s) = \sum_a \pi \cdot Q = {v_val:.4f}$")
    ax.set_ylabel(r"$Q^\pi(s, a)$")
    ax.set_title(r"Value vs action-value: $V$ is the policy-weighted average of $Q$")
    ax.set_ylim(0, max(q_vals) * 1.25)
    ax.legend(frameon=False, fontsize=11)
    save(fig, BLOG02, "fig-v-vs-q")


def fig_grid_values() -> None:
    """FrozenLake 4x4 V* heatmap with greedy-policy arrows."""
    import gymnasium as gym

    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P
    nS, nA = env.observation_space.n, env.action_space.n
    gamma = 0.99

    # Value iteration to get V*
    V = np.zeros(nS)
    for _ in range(2000):
        V_new = np.zeros(nS)
        for s in range(nS):
            q_vals = []
            for a in range(nA):
                q = sum(p * (r + gamma * V[s2]) for p, s2, r, d in P[s][a])
                q_vals.append(q)
            V_new[s] = max(q_vals)
        if np.max(np.abs(V_new - V)) < 1e-10:
            break
        V = V_new
    env.close()

    # Greedy policy
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_vals = []
        for a in range(nA):
            q = sum(p * (r + gamma * V[s2]) for p, s2, r, d in P[s][a])
            q_vals.append(q)
        policy[s] = int(np.argmax(q_vals))

    grid_V = V.reshape(4, 4)

    # FrozenLake layout: S=start, F=frozen, H=hole, G=goal
    desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    holes = set()
    goal = set()
    for r_idx, row in enumerate(desc):
        for c_idx, ch in enumerate(row):
            if ch == "H":
                holes.add((r_idx, c_idx))
            elif ch == "G":
                goal.add((r_idx, c_idx))

    # dx, dy offsets for arrow direction (in data coords)
    arrow_dx = {0: -0.28, 1: 0.0,  2: 0.28, 3: 0.0}   # L, D, R, U
    arrow_dy = {0: 0.0,   1: 0.28, 2: 0.0,  3: -0.28}

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid_V, cmap="YlOrRd", origin="upper", vmin=0, vmax=1)
    for r_idx in range(4):
        for c_idx in range(4):
            s = r_idx * 4 + c_idx
            if (r_idx, c_idx) in holes:
                ax.text(c_idx, r_idx, "H", ha="center", va="center",
                        fontsize=20, fontweight="bold", color=INK)
            elif (r_idx, c_idx) in goal:
                ax.text(c_idx, r_idx, "G", ha="center", va="center",
                        fontsize=20, fontweight="bold", color=INK)
            else:
                ax.text(c_idx, r_idx - 0.22, f"{grid_V[r_idx, c_idx]:.2f}",
                        ha="center", va="center", fontsize=10, fontweight="bold",
                        color=INK)
                a = policy[s]
                ax.annotate("", xy=(c_idx + arrow_dx[a], r_idx + 0.12 + arrow_dy[a]),
                            xytext=(c_idx, r_idx + 0.12),
                            arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                            lw=2.5, mutation_scale=16))

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(r"$V^*$ heatmap + greedy policy $\pi^*$ on FrozenLake 4x4", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label=r"$V^*(s)$")
    save(fig, BLOG02, "fig-grid-values")


def build_blog02() -> None:
    print(f"[{BLOG02}]")
    fig_backup_diagram()
    fig_v_vs_q()
    fig_grid_values()


# ----------------------------------------------------------------------------
# Blog 03 — DP, Monte Carlo, and TD
# ----------------------------------------------------------------------------
BLOG03 = "03-dp-mc-td"

# Mars Rover constants (same as in the notebook)
_GRID = 5
_GOAL = (4, 4)
_CRATERS = {(2, 2), (1, 3)}
_GAMMA = 0.95
_MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
_PERP = {0: [3, 1], 1: [0, 2], 2: [1, 3], 3: [2, 0]}


def _rc(s: int):
    return divmod(s, _GRID)


def _idx(r: int, c: int) -> int:
    return r * _GRID + c


def _build_rover_model(slip: float = 0.1):
    """Build the Mars Rover transition model P[s][a] = [(prob, s', r, done), ...]"""
    goal_idx = _idx(*_GOAL)
    crater_idxs = {_idx(*c) for c in _CRATERS}
    terminals = crater_idxs | {goal_idx}

    def move(s, a):
        r, c = _rc(s)
        dr, dc = _MOVES[a]
        nr = min(max(r + dr, 0), _GRID - 1)
        nc = min(max(c + dc, 0), _GRID - 1)
        return _idx(nr, nc)

    def reward(s_next):
        if s_next == goal_idx:
            return 10.0
        elif s_next in crater_idxs:
            return -10.0
        return -1.0

    P = {s: {a: [] for a in range(4)} for s in range(_GRID * _GRID)}
    for s in range(_GRID * _GRID):
        for a in range(4):
            if s in terminals:
                P[s][a] = [(1.0, s, 0.0, True)]
                continue
            outcomes = {}
            intended = move(s, a)
            outcomes[intended] = outcomes.get(intended, 0) + (1 - 2 * slip)
            for slip_a in _PERP[a]:
                slipped = move(s, slip_a)
                outcomes[slipped] = outcomes.get(slipped, 0) + slip
            P[s][a] = [(p, sn, reward(sn), sn in terminals) for sn, p in outcomes.items()]
    return P, terminals


def _value_iteration(P, gamma=_GAMMA, theta=1e-6):
    """Value iteration on a model dict P. Returns V, policy arrays."""
    nS = _GRID * _GRID
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v_old = V[s]
            q_vals = []
            for a in range(4):
                q = sum(p * (r + gamma * V[sn] * (1 - done)) for p, sn, r, done in P[s][a])
                q_vals.append(q)
            V[s] = max(q_vals)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_vals = [sum(p * (r + gamma * V[sn] * (1 - done)) for p, sn, r, done in P[s][a]) for a in range(4)]
        policy[s] = int(np.argmax(q_vals))
    return V, policy


def _mc_prediction(P, policy, terminals, episodes=20000, gamma=_GAMMA, alpha=0.02, max_steps=100):
    """MC prediction using model for sampling."""
    rng = np.random.default_rng(42)
    nS = _GRID * _GRID
    V = np.zeros(nS)
    trace = []
    for ep in range(1, episodes + 1):
        s = 0  # always start at (0,0)
        episode = []
        for _ in range(max_steps):
            a = policy[s]
            transitions = P[s][a]
            probs = [t[0] for t in transitions]
            i = rng.choice(len(transitions), p=probs)
            _, s_next, reward, done = transitions[i]
            episode.append((s, reward))
            s = s_next
            if done:
                break
        G = 0.0
        for st, rew in reversed(episode):
            G = rew + gamma * G
            V[st] = V[st] + alpha * (G - V[st])
        if ep % 100 == 0:
            trace.append(V[0])
    return V, trace


def _td_prediction(P, policy, terminals, episodes=20000, gamma=_GAMMA, alpha=0.05, max_steps=100):
    """TD(0) prediction using model for sampling."""
    rng = np.random.default_rng(42)
    nS = _GRID * _GRID
    V = np.zeros(nS)
    trace = []
    for ep in range(1, episodes + 1):
        s = 0
        for _ in range(max_steps):
            a = policy[s]
            transitions = P[s][a]
            probs = [t[0] for t in transitions]
            i = rng.choice(len(transitions), p=probs)
            _, s_next, reward, done = transitions[i]
            bootstrap = V[s_next] if not done else 0.0
            V[s] = V[s] + alpha * (reward + gamma * bootstrap - V[s])
            s = s_next
            if done:
                break
        if ep % 100 == 0:
            trace.append(V[0])
    return V, trace


def fig_rover_grid() -> None:
    """Annotated 5x5 Mars Rover map with start, goal, craters, and slip arrows."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, _GRID - 0.5)
    ax.set_ylim(-0.5, _GRID - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Draw grid cells
    for r in range(_GRID):
        for c in range(_GRID):
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True,
                                 fc=CANVAS, ec=DIVIDER, lw=1.5)
            ax.add_patch(rect)

    # Color special cells
    # Goal
    gr, gc = _GOAL
    ax.add_patch(plt.Rectangle((gc - 0.5, gr - 0.5), 1, 1, fc="#D4EDDA", ec=DIVIDER, lw=1.5))
    ax.text(gc, gr, "GOAL\n+10", ha="center", va="center", fontsize=11, fontweight="bold", color="#155724")

    # Craters
    for cr, cc in _CRATERS:
        ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1, fc="#F8D7DA", ec=DIVIDER, lw=1.5))
        ax.text(cc, cr, "CRATER\n-10", ha="center", va="center", fontsize=9, fontweight="bold", color="#721C24")

    # Start
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fc="#CCE5FF", ec=DIVIDER, lw=1.5))
    ax.text(0, 0, "START", ha="center", va="center", fontsize=10, fontweight="bold", color="#004085")

    # Regular cells: show coordinate
    special = {(0, 0), _GOAL} | _CRATERS
    for r in range(_GRID):
        for c in range(_GRID):
            if (r, c) not in special:
                ax.text(c, r, f"({r},{c})", ha="center", va="center", fontsize=8, color=MUTED)

    # Slip arrows on cell (1,1): show 0.8 right, 0.1 up, 0.1 down
    demo_r, demo_c = 3, 2
    ax.annotate("", xy=(demo_c + 0.4, demo_r), xytext=(demo_c, demo_r),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.5, mutation_scale=14))
    ax.text(demo_c + 0.22, demo_r - 0.18, "0.8", fontsize=8, color=ACCENT, fontweight="bold")
    ax.annotate("", xy=(demo_c, demo_r - 0.35), xytext=(demo_c, demo_r),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5, mutation_scale=10))
    ax.text(demo_c + 0.08, demo_r - 0.3, "0.1", fontsize=7, color=MUTED)
    ax.annotate("", xy=(demo_c, demo_r + 0.35), xytext=(demo_c, demo_r),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5, mutation_scale=10))
    ax.text(demo_c + 0.08, demo_r + 0.28, "0.1", fontsize=7, color=MUTED)

    # Labels
    ax.set_xticks(range(_GRID))
    ax.set_yticks(range(_GRID))
    ax.set_xticklabels([f"col {c}" for c in range(_GRID)], fontsize=9)
    ax.set_yticklabels([f"row {r}" for r in range(_GRID)], fontsize=9)
    ax.set_title("Mars Rover: 5x5 Gridworld", pad=12)
    ax.text(_GRID - 0.5, _GRID + 0.1, "Step cost: -1 everywhere\nSlip: 0.1 per perpendicular",
            ha="right", va="top", fontsize=9, color=MUTED)
    save(fig, BLOG03, "fig-rover-grid")


def fig_contraction() -> None:
    """Error shrinking by gamma each sweep (DP convergence visualization)."""
    sweeps = np.arange(0, 30)
    error = _GAMMA ** sweeps
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(sweeps, error, lw=2.5, color=ACCENT, marker="o", ms=4)
    ax.axhline(0, ls="-", lw=0.8, color=DIVIDER)
    ax.fill_between(sweeps, 0, error, alpha=0.08, color=ACCENT)
    ax.set_xlabel("sweep number $k$")
    ax.set_ylabel(r"$\|V_k - V^*\|_\infty$ (upper bound)")
    ax.set_title(r"Contraction: error shrinks by $\gamma$ each sweep")
    ax.annotate(rf"$\gamma = {_GAMMA}$: halves every {int(np.ceil(np.log(0.5)/np.log(_GAMMA)))} sweeps",
                xy=(14, _GAMMA**14), xytext=(18, 0.6), fontsize=10, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    save(fig, BLOG03, "fig-contraction")


def fig_rover_values() -> None:
    """Mars Rover V* heatmap with greedy policy arrows."""
    P, terminals = _build_rover_model(slip=0.1)
    V, policy = _value_iteration(P)
    grid_V = V.reshape(_GRID, _GRID)

    arrow_dx = {0: 0.0, 1: 0.28, 2: 0.0, 3: -0.28}  # up, right, down, left
    arrow_dy = {0: -0.28, 1: 0.0, 2: 0.28, 3: 0.0}

    fig, ax = plt.subplots(figsize=(7, 6.5))
    im = ax.imshow(grid_V, cmap="YlOrRd", origin="upper", vmin=grid_V.min(), vmax=grid_V.max())
    for r in range(_GRID):
        for c in range(_GRID):
            s = _idx(r, c)
            if s in terminals:
                label = "G" if (r, c) == _GOAL else "C"
                ax.text(c, r, label, ha="center", va="center", fontsize=18,
                        fontweight="bold", color=INK)
            else:
                ax.text(c, r - 0.2, f"{grid_V[r, c]:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=INK)
                a = policy[s]
                ax.annotate("", xy=(c + arrow_dx[a], r + 0.12 + arrow_dy[a]),
                            xytext=(c, r + 0.12),
                            arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2, mutation_scale=14))
    ax.set_xticks(range(_GRID))
    ax.set_yticks(range(_GRID))
    ax.set_xticklabels([str(c) for c in range(_GRID)])
    ax.set_yticklabels([str(r) for r in range(_GRID)])
    ax.set_title(r"Mars Rover $V^*$ + greedy policy $\pi^*$ (slip=0.1)", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label=r"$V^*(s)$")
    save(fig, BLOG03, "fig-rover-values")


def fig_rover_slip_comparison() -> None:
    """Side-by-side V* heatmaps: SLIP=0.1 vs SLIP=0.0."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, slip, title in zip(axes, [0.1, 0.0], ["Slip = 0.1 (stochastic)", "Slip = 0.0 (deterministic)"]):
        P, terminals = _build_rover_model(slip=slip)
        V, _ = _value_iteration(P)
        grid_V = V.reshape(_GRID, _GRID)
        im = ax.imshow(grid_V, cmap="YlOrRd", origin="upper", vmin=-2, vmax=10)
        for r in range(_GRID):
            for c in range(_GRID):
                s = _idx(r, c)
                if s in terminals:
                    label = "G" if (r, c) == _GOAL else "C"
                    ax.text(c, r, label, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
                else:
                    ax.text(c, r, f"{grid_V[r, c]:.1f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=INK)
        ax.set_xticks(range(_GRID))
        ax.set_yticks(range(_GRID))
        ax.set_xticklabels([str(c) for c in range(_GRID)])
        ax.set_yticklabels([str(r) for r in range(_GRID)])
        ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=axes, shrink=0.7, label=r"$V^*(s)$")
    fig.suptitle("Stochasticity makes craters dangerous to neighbors", fontsize=13, fontweight="bold", y=0.98)
    save(fig, BLOG03, "fig-rover-slip-comparison")


def fig_convergence() -> None:
    """V(start) vs episodes for MC and TD, with DP value as reference."""
    P, terminals = _build_rover_model(slip=0.1)
    V_dp, policy = _value_iteration(P)
    dp_value = V_dp[0]

    _, mc_trace = _mc_prediction(P, policy, terminals, episodes=20000)
    _, td_trace = _td_prediction(P, policy, terminals, episodes=20000)

    episodes = list(range(100, 20001, 100))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(episodes, mc_trace, lw=1.5, color=ACCENT_SOFT, alpha=0.9, label=r"MC ($\alpha$=0.02)")
    ax.plot(episodes, td_trace, lw=1.5, color=ACCENT, alpha=0.9, label=r"TD(0) ($\alpha$=0.05)")
    ax.axhline(dp_value, ls="--", lw=2, color=INK, label=f"DP exact = {dp_value:.2f}")
    ax.set_xlabel("episodes")
    ax.set_ylabel("$V$(start state)")
    ax.set_title("Convergence: MC vs TD(0) on Mars Rover")
    ax.legend(frameon=False)
    save(fig, BLOG03, "fig-convergence")


def fig_mc_td_grids() -> None:
    """3-panel heatmap comparison: DP exact, MC 20k, TD 20k."""
    P, terminals = _build_rover_model(slip=0.1)
    V_dp, policy = _value_iteration(P)
    V_mc, _ = _mc_prediction(P, policy, terminals, episodes=20000)
    V_td, _ = _td_prediction(P, policy, terminals, episodes=20000)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    vmin, vmax = min(V_dp.min(), V_mc.min(), V_td.min()), max(V_dp.max(), V_mc.max(), V_td.max())
    for ax, V, title in zip(axes, [V_dp, V_mc, V_td],
                            ["DP (exact)", "MC (20k episodes)", "TD(0) (20k episodes)"]):
        grid_V = V.reshape(_GRID, _GRID)
        im = ax.imshow(grid_V, cmap="YlOrRd", origin="upper", vmin=vmin, vmax=vmax)
        for r in range(_GRID):
            for c in range(_GRID):
                s = _idx(r, c)
                if s in terminals:
                    label = "G" if (r, c) == _GOAL else "C"
                    ax.text(c, r, label, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
                else:
                    ax.text(c, r, f"{grid_V[r, c]:.1f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=INK)
        ax.set_xticks(range(_GRID))
        ax.set_yticks(range(_GRID))
        ax.set_xticklabels([str(c) for c in range(_GRID)])
        ax.set_yticklabels([str(r) for r in range(_GRID)])
        ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.75, label=r"$V(s)$")
    fig.suptitle("All three converge to the same values", fontsize=13, fontweight="bold", y=0.99)
    save(fig, BLOG03, "fig-mc-td-grids")


def build_blog03() -> None:
    print(f"[{BLOG03}]")
    fig_rover_grid()
    fig_contraction()
    fig_rover_values()
    fig_rover_slip_comparison()
    fig_convergence()
    fig_mc_td_grids()


# ----------------------------------------------------------------------------
# Blog 04 — SARSA, Q-learning, DQN
# ----------------------------------------------------------------------------
BLOG04 = "04-sarsa-qlearning-dqn"

# --- CliffWalking (deterministic 4x12), reimplemented in numpy for reproducibility ---
_CW_ROWS, _CW_COLS = 4, 12
_CW_START = 36          # (3, 0)
_CW_GOAL = 47           # (3, 11)
_CW_CLIFF = set(range(37, 47))  # (3, 1) .. (3, 10)
_CW_MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left


def _cw_step(s: int, a: int):
    """One deterministic CliffWalking transition -> (next_state, reward, done)."""
    r, c = divmod(s, _CW_COLS)
    dr, dc = _CW_MOVES[a]
    nr = min(max(r + dr, 0), _CW_ROWS - 1)
    nc = min(max(c + dc, 0), _CW_COLS - 1)
    ns = nr * _CW_COLS + nc
    if ns in _CW_CLIFF:
        return _CW_START, -100.0, False  # fall off -> back to start
    if ns == _CW_GOAL:
        return ns, -1.0, True
    return ns, -1.0, False


def _cw_train(kind: str, episodes=5000, alpha=0.1, gamma=0.99, eps=0.1, seed=0):
    """Train tabular Q-learning ('q') or SARSA ('sarsa') on CliffWalking."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((_CW_ROWS * _CW_COLS, 4))
    rewards = []

    def act(s):
        if rng.random() < eps:
            return int(rng.integers(4))
        return int(np.argmax(Q[s]))

    for _ in range(episodes):
        s = _CW_START
        a = act(s)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < 1000:
            ns, r, done = _cw_step(s, a)
            if kind == "q":  # off-policy: bootstrap off the BEST next action
                target = r + (0.0 if done else gamma * np.max(Q[ns]))
                Q[s, a] += alpha * (target - Q[s, a])
                s = ns
                a = act(s)
            else:            # SARSA, on-policy: bootstrap off the action actually taken
                na = act(ns) if not done else 0
                target = r + (0.0 if done else gamma * Q[ns, na])
                Q[s, a] += alpha * (target - Q[s, a])
                s, a = ns, na
            total += r
            steps += 1
        rewards.append(total)
    return Q, rewards


def _cw_greedy_path(Q, max_steps=60):
    s = _CW_START
    path = [s]
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        ns, _, done = _cw_step(s, a)
        path.append(ns)
        s = ns
        if done:
            break
    return path


def _draw_cliff_grid(ax, paths=None):
    """Draw the 4x12 cliff grid on ax. paths: list of (states, color, label, marker)."""
    grid = np.zeros((_CW_ROWS, _CW_COLS))
    for s in _CW_CLIFF:
        r, c = divmod(s, _CW_COLS)
        grid[r, c] = -1.0
    ax.imshow(grid, cmap="RdYlGn", vmin=-1.4, vmax=0.6, aspect="equal")
    for s in _CW_CLIFF:
        r, c = divmod(s, _CW_COLS)
        ax.text(c, r, "×", ha="center", va="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(0, 3, "S", ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(11, 3, "G", ha="center", va="center", fontsize=14, fontweight="bold", color=ACCENT)
    if paths:
        for states, color, label, marker in paths:
            rows = [s // _CW_COLS for s in states]
            cols = [s % _CW_COLS for s in states]
            ax.plot(cols, rows, marker + "-", color=color, ms=5, lw=2, alpha=0.85, label=label)
        ax.legend(loc="upper center", fontsize=9, frameon=False, ncol=2)
    ax.set_xticks(range(_CW_COLS))
    ax.set_yticks(range(_CW_ROWS))
    ax.set_xticklabels([str(c) for c in range(_CW_COLS)], fontsize=8)
    ax.set_yticklabels([str(r) for r in range(_CW_ROWS)], fontsize=8)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(3.5, -0.5)
    ax.grid(False)


def fig_q_vs_v() -> None:
    """A state with three action-values; V(s) is the max."""
    actions = ["a1\n(Up)", "a2\n(Right)", "a3\n(Down)"]
    qvals = [3, 7, 5]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = [MUTED if q != max(qvals) else ACCENT for q in qvals]
    bars = ax.bar(actions, qvals, color=colors, width=0.6)
    ax.axhline(max(qvals), ls="--", lw=1.5, color=ACCENT)
    ax.text(2.45, 7.15, r"$V(s)=\max_a Q(s,a)=7$", ha="right", color=ACCENT, fontsize=11)
    for b, q in zip(bars, qvals):
        ax.text(b.get_x() + b.get_width() / 2, q + 0.15, f"Q={q}", ha="center", fontsize=10, color=INK)
    ax.set_ylabel(r"action-value  $Q(s,a)$")
    ax.set_title("To act, compare action-values and take the biggest")
    ax.set_ylim(0, 8.2)
    ax.grid(axis="x", visible=False)
    save(fig, BLOG04, "fig-q-vs-v")


def fig_cliffwalking() -> None:
    """The CliffWalking environment with the two candidate routes drawn."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    # Optimal/risky route: hug the cliff along row 2.
    risky = [36, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 47]
    # Safe route: climb to row 1, cross, then drop.
    safe = [36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35, 47]
    _draw_cliff_grid(ax, paths=[
        (risky, ACCENT, "optimal / risky (−13)", "o"),
        (safe, MUTED, "safe (−17)", "s"),
    ])
    ax.set_title("CliffWalking: the short path hugs the cliff edge", pad=10)
    save(fig, BLOG04, "fig-cliffwalking")


def fig_sarsa_vs_q_curves() -> None:
    """Real CliffWalking results: training curves + learned greedy paths."""
    Q_q, rew_q = _cw_train("q", seed=0)
    Q_s, rew_s = _cw_train("sarsa", seed=0)
    path_q = _cw_greedy_path(Q_q)
    path_s = _cw_greedy_path(Q_s)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    w = 100
    qs = np.convolve(rew_q, np.ones(w) / w, mode="valid")
    ss = np.convolve(rew_s, np.ones(w) / w, mode="valid")
    axes[0].plot(qs, color=ACCENT, lw=1.8, label="Q-learning (off-policy)")
    axes[0].plot(ss, color=MUTED, lw=1.8, label="SARSA (on-policy)")
    axes[0].axhline(-13, ls="--", lw=1.4, color=INK, label="optimal (−13)")
    axes[0].set_title("Training reward (100-episode average)")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("total reward")
    axes[0].set_ylim(-100, 0)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")

    _draw_cliff_grid(axes[1], paths=[
        (path_q, ACCENT, "Q-learning (risky)", "o"),
        (path_s, MUTED, "SARSA (safe)", "s"),
    ])
    axes[1].set_title("Learned greedy paths")
    fig.suptitle("Same algorithm, one symbol apart: Q-learning takes the edge, SARSA plays it safe",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save(fig, BLOG04, "fig-sarsa-vs-q-curves")


def fig_max_propagation() -> None:
    """The +10 ringing backward: two Q-learning updates up the safe column."""
    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    cells = [("gem (+10)\nterminal", "", 3.0), ("tile (1)", "Q=5.0", 2.0), ("tile (2)", "Q=1.75", 1.0)]
    for label, qval, y in cells:
        face = ACCENT_SOFT if "gem" in label else CANVAS
        ax.add_patch(plt.Rectangle((0.3, y - 0.4), 1.4, 0.8, facecolor=face,
                                   edgecolor=INK, lw=1.6))
        ax.text(1.0, y + 0.12, label, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
        if qval:
            ax.text(1.0, y - 0.2, qval, ha="center", va="center", fontsize=11, color=ACCENT)
    # arrows (each tile steps UP)
    ax.annotate("", xy=(1.0, 2.55), xytext=(1.0, 2.4),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2, mutation_scale=16))
    ax.annotate("", xy=(1.0, 1.55), xytext=(1.0, 1.4),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2, mutation_scale=16))
    ax.text(1.95, 2.5, r"$Q \leftarrow 0 + 0.5(10+0-0)=5.0$", ha="left", fontsize=10, color=MUTED)
    ax.text(1.95, 1.5, r"$Q \leftarrow 0 + 0.5(-1+0.9\cdot5-0)=1.75$", ha="left", fontsize=10, color=MUTED)
    ax.text(1.0, 0.25, r"$\alpha=0.5,\ \gamma=0.9$", ha="center", fontsize=11, color=INK)
    ax.set_xlim(0, 5.4)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    ax.set_title("The reward rings backward, one update at a time", pad=8)
    save(fig, BLOG04, "fig-max-propagation")


def fig_deadly_triad() -> None:
    """Three ingredients that are dangerous together."""
    fig, ax = plt.subplots(figsize=(7, 6.2))
    circ = [
        ((0.40, 0.62), "Function\napproximation", ACCENT),
        ((0.60, 0.62), "Bootstrapping", MUTED),
        ((0.50, 0.42), "Off-policy\ndata", ACCENT_SOFT),
    ]
    for (x, y), label, color in circ:
        ax.add_patch(plt.Circle((x, y), 0.22, color=color, alpha=0.32, ec=color, lw=1.5))
    ax.text(0.26, 0.74, "Function\napproximation", ha="center", fontsize=10.5, color=INK)
    ax.text(0.74, 0.74, "Bootstrapping", ha="center", fontsize=10.5, color=INK)
    ax.text(0.50, 0.26, "Off-policy data", ha="center", fontsize=10.5, color=INK)
    ax.text(0.50, 0.52, "diverges", ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.95)
    ax.axis("off")
    ax.set_title("The deadly triad: any two are safe, all three can blow up", pad=8)
    save(fig, BLOG04, "fig-deadly-triad")


def fig_max_overestimation() -> None:
    """E[max of noisy estimates] > true max, growing with the number of actions."""
    rng = np.random.default_rng(0)
    n_actions = np.arange(1, 17)
    trials = 200_000
    bias = []
    for n in n_actions:
        est = rng.normal(0.0, 1.0, size=(trials, n))  # true value 0 for every action
        bias.append(est.max(axis=1).mean())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(n_actions, bias, marker="o", ms=4, lw=2, color=ACCENT)
    ax.axhline(0, ls="--", lw=1.4, color=INK, label="true value of every action = 0")
    ax.set_xlabel("number of actions")
    ax.set_ylabel(r"$\mathbb{E}[\max_a \hat{Q}(s,a)]$")
    ax.set_title("Taking the max of noisy estimates is biased upward")
    ax.legend(frameon=False, fontsize=9)
    save(fig, BLOG04, "fig-max-overestimation")


# Real avg100 logs captured from the Pong DQN run in assignments/lecture2.2.ipynb
_PONG_LOG = [
    (8765, -20.80), (17767, -20.55), (26769, -20.43), (36699, -20.25), (46053, -20.20),
    (55511, -20.18), (64771, -20.19), (73749, -20.23), (82486, -20.27), (91754, -20.30),
    (101253, -20.23), (110829, -20.22), (120037, -20.25), (130133, -20.23), (139884, -20.23),
    (148875, -20.27), (158705, -20.27), (169027, -20.18), (179366, -20.07), (189741, -19.97),
    (199896, -19.91), (210935, -19.81), (221075, -19.73), (231794, -19.74), (241909, -19.76),
    (253364, -19.57), (263830, -19.54), (275199, -19.58), (286346, -19.57), (297499, -19.59),
    (310692, -19.49), (323989, -19.38), (336911, -19.35), (349282, -19.28), (361806, -19.16),
    (375289, -19.17), (389075, -19.04), (402851, -18.94), (417009, -18.83), (431026, -18.71),
    (445182, -18.72), (459111, -18.79), (474602, -18.68), (489218, -18.65), (504376, -18.59),
    (519187, -18.53), (535213, -18.44), (550323, -18.29), (566371, -18.18), (582528, -18.01),
    (600253, -17.80), (618128, -17.42), (635224, -17.12), (654267, -16.87), (673371, -16.49),
    (691576, -16.27), (708791, -16.10), (727101, -15.82), (747896, -15.44), (769119, -15.03),
    (791110, -14.65), (814509, -14.35), (838564, -13.94), (865410, -13.21), (896044, -12.55),
    (925277, -11.49), (956377, -10.10), (991386, -8.45), (1018737, -6.03), (1046442, -3.73),
    (1074037, -1.47), (1100738, 0.94), (1126019, 3.28), (1154704, 5.18), (1179417, 7.32),
    (1204370, 9.30), (1229099, 10.87), (1252841, 12.09), (1278123, 12.21), (1301489, 12.60),
    (1323873, 13.13), (1347656, 13.29), (1369962, 13.59), (1393385, 14.08), (1417096, 14.19),
    (1439421, 14.26), (1463512, 14.29), (1487800, 14.35), (1510775, 14.60), (1532076, 14.75),
    (1553653, 14.93), (1575006, 15.17), (1599348, 15.01), (1623929, 14.93), (1647244, 14.97),
    (1669555, 14.94), (1695198, 14.74), (1719400, 14.55), (1741648, 14.68), (1764896, 14.70),
    (1787517, 14.49), (1810864, 14.37), (1834350, 14.50), (1857036, 14.60), (1879017, 14.67),
    (1903744, 14.70), (1927599, 14.97), (1954010, 14.96), (1980020, 14.60),
]


def fig_pong_training() -> None:
    """The real Pong DQN learning curve (avg of last 100 episodes vs frames)."""
    frames = np.array([f for f, _ in _PONG_LOG]) / 1e6
    avg100 = np.array([v for _, v in _PONG_LOG])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(frames, avg100, lw=2, color=ACCENT)
    ax.axhline(-21, ls="--", lw=1.4, color=MUTED, label="loses every point (−21)")
    ax.axhline(0, ls=":", lw=1.2, color=INK, label="break-even (0)")
    ax.set_xlabel("environment frames (millions)")
    ax.set_ylabel("reward (avg of last 100 episodes)")
    ax.set_title("DQN on Pong: from −21 to +15 in ~2M frames")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, BLOG04, "fig-pong-training")


def build_blog04() -> None:
    print(f"[{BLOG04}]")
    fig_q_vs_v()
    fig_cliffwalking()
    fig_sarsa_vs_q_curves()
    fig_max_propagation()
    fig_deadly_triad()
    fig_max_overestimation()
    fig_pong_training()


# ----------------------------------------------------------------------------
# Blog 05 — Policy Gradients
# ----------------------------------------------------------------------------
BLOG05 = "05-policy-gradients"


def fig_policy_fan() -> None:
    """Softmax policy over 9 angles BEFORE and AFTER one REINFORCE update."""
    angles = [f"a{i}" for i in range(1, 10)]
    before = np.array([0.042, 0.063, 0.094, 0.172, 0.256, 0.172, 0.094, 0.063, 0.042])
    after = np.array([0.039, 0.059, 0.089, 0.164, 0.246, 0.189, 0.089, 0.059, 0.039])

    x = np.arange(len(angles))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w / 2, before, width=w, color=ACCENT, label="before")
    ax.bar(x + w / 2, after, width=w, color=INK, label="after")
    ax.bar(5 + w / 2, after[5], width=w, color=INK, edgecolor=ACCENT, lw=2.2)
    ax.annotate("a₆ taken", xy=(5, after[5]), xytext=(6.4, 0.24),
                fontsize=10, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels(angles)
    ax.set_xlabel("angle")
    ax.set_ylabel("π(angle)")
    ax.set_title("One REINFORCE update shifts the fan toward a₆")
    ax.legend(frameon=False)
    save(fig, BLOG05, "fig-policy-fan")


def fig_logit_gradients() -> None:
    """Per-logit gradient pushes from one REINFORCE update."""
    angles = [f"a{i}" for i in range(1, 10)]
    grads = np.array([-0.038, -0.057, -0.085, -0.155, -0.230,
                       +0.745, -0.085, -0.057, -0.038])

    colors = [ACCENT if g > 0 else MUTED for g in grads]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(angles, grads, color=colors, width=0.6)
    ax.axhline(0, ls="-", lw=0.8, color=DIVIDER)
    ax.set_xlabel("angle")
    ax.set_ylabel("logit gradient")
    ax.set_title("Per-logit gradients: one up, eight down, sum = 0")
    save(fig, BLOG05, "fig-logit-gradients")


def fig_score_push() -> None:
    """Push = 1 − π(a) vs current probability π(a)."""
    pi = np.linspace(0.001, 0.999, 500)
    push = 1.0 - pi

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(pi, push, lw=2.4, color=ACCENT)
    ax.scatter([0.10], [0.90], color=INK, zorder=5, s=30)
    ax.annotate("surprising: big push", xy=(0.10, 0.90),
                xytext=(0.22, 0.85), fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.scatter([0.85], [0.15], color=INK, zorder=5, s=30)
    ax.annotate("confident: tiny push", xy=(0.85, 0.15),
                xytext=(0.55, 0.30), fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("current probability π(a)")
    ax.set_ylabel("push per unit advantage: 1 − π(a)")
    ax.set_title("How hard ∇log π pushes: surprising actions move more")
    save(fig, BLOG05, "fig-score-push")


def fig_credit_assignment() -> None:
    """Cumulative discounted return building across the Archer MDP trajectory."""
    gamma = 0.9
    G_shoot = 10.0
    G_10 = -0.2 + gamma * G_shoot
    G_20 = -0.2 + gamma * G_10
    G_30 = -0.2 + gamma * G_20
    G_40 = -0.2 + gamma * G_30

    states = ["step@40m", "step@30m", "step@20m", "step@10m", "shoot"]
    returns = [G_40, G_30, G_20, G_10, G_shoot]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(states, returns, color=ACCENT, width=0.6)
    ax.axhline(0, ls="-", lw=1.2, color=INK)
    for i, (s, g) in enumerate(zip(states, returns)):
        ax.text(i, g + 0.25, f"{g:.2f}", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("state")
    ax.set_ylabel("return $G_t$")
    ax.set_title("Credit assignment: the return at 40 m is +5.87, not −0.2")
    save(fig, BLOG05, "fig-credit-assignment")


def fig_variance_ladder() -> None:
    """Gradient variance for three methods on a log scale."""
    methods = ["REINFORCE", "+baseline", "Actor-Critic"]
    variances = [0.0437, 0.0006, 0.0004]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(methods, variances, color=ACCENT, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("gradient variance (log scale)")
    ax.set_title("The variance ladder: baseline cuts noise by ~73×, critic by ~110×")
    for i, v in enumerate(variances):
        ax.text(i, v * 1.5, f"{v:.4f}", ha="center", fontsize=10, color=INK, fontweight="bold")
    save(fig, BLOG05, "fig-variance-ladder")


def fig_greedy_returns() -> None:
    """Greedy return per method with a solved threshold line."""
    methods = ["REINFORCE", "+baseline", "Actor-Critic"]
    returns = [4.66, 9.60, 9.62]
    threshold = 9.0

    colors = [ACCENT if r >= threshold else MUTED for r in returns]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(methods, returns, color=colors, width=0.55)
    ax.axhline(threshold, ls="--", lw=1.8, color=INK)
    ax.text(2.35, threshold + 0.15, "solved threshold", fontsize=10, color=INK,
            ha="right", va="bottom")
    for i, r in enumerate(returns):
        ax.text(i, r + 0.15, f"{r:.2f}", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("greedy return")
    ax.set_title("Greedy evaluation: only variance-reduced methods solve the task")
    save(fig, BLOG05, "fig-greedy-returns")


def build_blog05() -> None:
    print(f"[{BLOG05}]")
    fig_policy_fan()
    fig_logit_gradients()
    fig_score_push()
    fig_credit_assignment()
    fig_variance_ladder()
    fig_greedy_returns()


# ----------------------------------------------------------------------------
# Registry + CLI
# ----------------------------------------------------------------------------
BUILDERS = {
    "01": build_blog01,
    "02": build_blog02,
    "03": build_blog03,
    "04": build_blog04,
    "05": build_blog05,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blog figures (SVG).")
    parser.add_argument("--blog", choices=sorted(BUILDERS), default=None,
                        help="only build one blog's figures (default: all)")
    args = parser.parse_args()

    house_style()
    targets = [args.blog] if args.blog else sorted(BUILDERS)
    for key in targets:
        BUILDERS[key]()
    print("done.")


if __name__ == "__main__":
    main()
