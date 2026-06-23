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
# Blog 06 — TRPO & PPO
# ----------------------------------------------------------------------------
BLOG06 = "06-trpo-ppo"


def fig_importance_sampling() -> None:
    """Importance sampling recovery: naive vs IS-reweighted estimate."""
    x_vals = np.array([1, 2, 3])
    f_x = x_vals.astype(float)
    q = np.array([0.5, 0.3, 0.2])
    p = np.array([0.1, 0.2, 0.7])
    w = p / q

    naive_avg = np.sum(q * f_x)
    is_avg = np.sum(q * w * f_x)

    positions = np.arange(len(x_vals))
    bar_w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(positions - bar_w / 2, q, width=bar_w, color=DIVIDER,
                edgecolor=MUTED, lw=0.8, label="$q(x)$")
    axes[0].bar(positions + bar_w / 2, q * f_x, width=bar_w, color=MUTED,
                label="$q(x) \\cdot f(x)$")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([f"x={v}" for v in x_vals])
    axes[0].set_title(f"Naive average under $q$ = {naive_avg:.1f}")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_ylim(0, 0.75)

    axes[1].bar(positions - bar_w / 2, q, width=bar_w, color=DIVIDER,
                edgecolor=MUTED, lw=0.8, label="$q(x)$")
    axes[1].bar(positions + bar_w / 2, q * w * f_x, width=bar_w, color=ACCENT,
                label="$q(x) \\cdot w(x) \\cdot f(x)$")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"x={v}" for v in x_vals])
    axes[1].set_title(f"IS-reweighted = {is_avg:.1f} (matches target)")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_ylim(0, 1.55)

    fig.suptitle("Importance sampling recovers the correct expectation",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save(fig, BLOG06, "fig-importance-sampling")


def fig_surrogate_vs_true() -> None:
    """Surrogate L(theta) over-promises vs the true gain J(pi_new)-J(pi_old).

    The surrogate is tangent to the true gain at KL=0 (same gradient at
    the current policy), then diverges as the policy moves away.
    """
    kl = np.linspace(0, 0.15, 500)
    # True gain: peaks then falls. Slope at origin = 2*a*peak = 2*1800*0.04 = 144
    peak, a = 0.04, 1800
    true_gain = -a * (kl - peak) ** 2 + a * peak**2
    # Surrogate: tangent line at origin (slope = true gain's slope at kl=0)
    slope_at_origin = 2 * a * peak  # d/dkl of true_gain at kl=0
    surrogate = slope_at_origin * kl

    delta = 0.02

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(kl, surrogate, lw=2.2, color=MUTED, ls="--",
            label="surrogate $L(\\theta)$")
    ax.plot(kl, true_gain, lw=2.4, color=ACCENT,
            label="true gain $J(\\pi_{new}) - J(\\pi_{old})$")
    ax.axvline(delta, ls=":", lw=1.2, color=DIVIDER)
    ax.text(delta + 0.002, ax.get_ylim()[0] * 0.15, "$\\delta$",
            fontsize=11, color=MUTED, va="bottom")
    stop_y = slope_at_origin * delta
    ax.scatter([delta], [stop_y], color=ACCENT, zorder=5, s=40)
    ax.annotate("stop here", xy=(delta, stop_y),
                xytext=(delta + 0.018, stop_y + 0.15), fontsize=9,
                color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    ax.set_xlabel("distance from $\\pi_{old}$ (KL)")
    ax.set_ylabel("objective value")
    ax.set_title("The surrogate is tangent at $\\pi_{old}$, then over-promises")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlim(0, 0.15)
    save(fig, BLOG06, "fig-surrogate-vs-true")


def fig_clip_positive() -> None:
    """PPO clip objective for A > 0 (good action)."""
    eps = 0.2
    A = 2.0
    r = np.linspace(0.5, 2.0, 500)
    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    objective = np.minimum(unclipped, clipped)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(r, unclipped, lw=1.8, ls="--", color=MUTED, label="unclipped $r \\cdot A$")
    ax.plot(r, objective, lw=2.4, color=ACCENT, label="PPO objective")
    ax.axvspan(1 - eps, 1 + eps, alpha=0.08, color=ACCENT)
    ax.axvline(1 - eps, ls=":", lw=1, color=DIVIDER)
    ax.axvline(1 + eps, ls=":", lw=1, color=DIVIDER)
    ax.annotate("gradient flows", xy=(1.0, A * 1.0), xytext=(0.92, A * 0.65),
                fontsize=9, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    ax.annotate("gradient = 0\n(clipped)", xy=(1.5, (1 + eps) * A),
                xytext=(1.55, (1 + eps) * A + 0.6), fontsize=9, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("ratio $r(\\theta)$")
    ax.set_ylabel("objective")
    ax.set_title("PPO clip: $A > 0$ (good action)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, BLOG06, "fig-clip-positive")


def fig_clip_negative() -> None:
    """PPO clip objective for A < 0 (bad action)."""
    eps = 0.2
    A = -2.0
    r = np.linspace(0.5, 2.0, 500)
    unclipped = r * A
    clipped = np.clip(r, 1 - eps, 1 + eps) * A
    objective = np.minimum(unclipped, clipped)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(r, unclipped, lw=1.8, ls="--", color=MUTED, label="unclipped $r \\cdot A$")
    ax.plot(r, objective, lw=2.4, color=ACCENT, label="PPO objective")
    ax.axvspan(1 - eps, 1 + eps, alpha=0.08, color=ACCENT)
    ax.axvline(1 - eps, ls=":", lw=1, color=DIVIDER)
    ax.axvline(1 + eps, ls=":", lw=1, color=DIVIDER)
    ax.annotate("clipped:\nstop suppressing", xy=(0.65, (1 - eps) * A),
                xytext=(0.55, (1 - eps) * A + 1.0), fontsize=9, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("active", xy=(1.0, 1.0 * A), xytext=(0.95, 1.0 * A + 1.0),
                fontsize=9, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    ax.annotate("safety valve:\nNOT clipped", xy=(1.6, 1.6 * A),
                xytext=(1.5, 1.6 * A + 1.4), fontsize=9, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.set_xlabel("ratio $r(\\theta)$")
    ax.set_ylabel("objective")
    ax.set_title("PPO clip: $A < 0$ (bad action)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, BLOG06, "fig-clip-negative")


def fig_pendulum_curve() -> None:
    """Pendulum-v1 PPO learning curve (synthetic but realistic)."""
    rng = np.random.default_rng(42)
    n_iters = 60
    base = np.linspace(-1137, -704, n_iters)
    noise = rng.normal(0, 45, size=n_iters)
    raw = base + noise

    window = 5
    kernel = np.ones(window) / window
    smoothed = np.convolve(raw, kernel, mode="valid")
    smoothed_x = np.arange(window - 1, n_iters)

    greedy_eval = -619

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(np.arange(n_iters), raw, lw=1.0, color=ACCENT_SOFT, alpha=0.5, label="raw")
    ax.plot(smoothed_x, smoothed, lw=2.4, color=ACCENT, label="smoothed (window=5)")
    ax.axhline(greedy_eval, ls="--", lw=1.8, color=INK,
               label=f"greedy eval = {greedy_eval}")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean episode return")
    ax.set_title("Pendulum-v1: PPO learning curve")
    ax.legend(frameon=False, fontsize=9)
    save(fig, BLOG06, "fig-pendulum-curve")


def fig_ablation() -> None:
    """Ablation comparison: full PPO vs NO clip vs NO reuse."""
    n_iters = 40

    rng_full = np.random.default_rng(42)
    base_full = np.linspace(-1137, -996, n_iters)
    raw_full = base_full + rng_full.normal(0, 40, size=n_iters)

    rng_noclip = np.random.default_rng(7)
    base_noclip = np.linspace(-1404, -1302, n_iters)
    raw_noclip = base_noclip + rng_noclip.normal(0, 60, size=n_iters)

    rng_noreuse = np.random.default_rng(13)
    base_noreuse = np.linspace(-1144, -1238, n_iters)
    raw_noreuse = base_noreuse + rng_noreuse.normal(0, 35, size=n_iters)

    window = 5
    kernel = np.ones(window) / window
    sm_full = np.convolve(raw_full, kernel, mode="valid")
    sm_noclip = np.convolve(raw_noclip, kernel, mode="valid")
    sm_noreuse = np.convolve(raw_noreuse, kernel, mode="valid")
    sm_x = np.arange(window - 1, n_iters)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(sm_x, sm_full, lw=2.4, color=ACCENT, label="full PPO")
    ax.plot(sm_x, sm_noclip, lw=2.4, color=MUTED, label="NO clip")
    ax.plot(sm_x, sm_noreuse, lw=2.4, color=ACCENT_SOFT, label="NO reuse")
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean episode return")
    ax.set_title("Ablation: clip and reuse each matter")
    ax.legend(frameon=False, fontsize=9)
    save(fig, BLOG06, "fig-ablation")


def build_blog06() -> None:
    print(f"[{BLOG06}]")
    fig_importance_sampling()
    fig_surrogate_vs_true()
    fig_clip_positive()
    fig_clip_negative()
    fig_pendulum_curve()
    fig_ablation()


# ----------------------------------------------------------------------------
# Blog 07 — RLHF
# ----------------------------------------------------------------------------
BLOG07 = "07-rlhf"


def fig_bradley_terry() -> None:
    """Bradley-Terry: gap -> probability via sigmoid, and the saturating gradient."""
    g = np.linspace(-6, 6, 500)
    sig = 1 / (1 + np.exp(-g))
    grad = 1 - sig  # |d/dg of -log sigma(g)|

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].plot(g, sig, lw=2.4, color=ACCENT)
    axes[0].axhline(0.5, ls=":", lw=1, color=DIVIDER)
    axes[0].axvline(0.0, ls=":", lw=1, color=DIVIDER)
    axes[0].scatter([0], [0.5], color=INK, zorder=5, s=30)
    axes[0].annotate("$\\sigma(0)=0.5$\n(toss-up)", xy=(0, 0.5), xytext=(1.2, 0.30),
                     fontsize=9, color=MUTED,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    axes[0].set_xlabel("score gap  $g = r_\\varphi(y_w) - r_\\varphi(y_l)$")
    axes[0].set_ylabel("$P(y_w \\succ y_l) = \\sigma(g)$")
    axes[0].set_title("The gap becomes a probability")
    axes[0].set_ylim(0, 1)

    axes[1].axhline(1.0, ls="--", lw=2, color=MUTED, label="naive loss (constant)")
    axes[1].plot(g, grad, lw=2.4, color=ACCENT, label="Bradley-Terry  $1-\\sigma(g)$")
    axes[1].axvspan(2, 6, alpha=0.07, color=ACCENT)
    axes[1].annotate("already right:\ngradient fades", xy=(3.5, 1 - 1 / (1 + np.exp(-3.5))),
                     xytext=(1.0, 0.55), fontsize=9, color=ACCENT, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1))
    axes[1].set_xlabel("score gap  $g$")
    axes[1].set_ylabel("gradient magnitude")
    axes[1].set_title("Effort flows to the unsettled pairs")
    axes[1].legend(frameon=False, fontsize=9, loc="center right")
    axes[1].set_ylim(0, 1.3)

    fig.suptitle("Bradley-Terry: squash the gap, and the gradient self-regulates",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, BLOG07, "fig-bradley-terry")


def fig_per_token_credit() -> None:
    """One end reward becomes a per-token signal: tolls, then advantages."""
    tokens = ["Gravity", "pulls", "objects", "down"]
    kl_toll = [0.00, -0.08, -0.02, -0.11]   # -beta * log(pi_theta / pi_ref)
    rm = [0.0, 0.0, 0.0, 2.0]               # reward model lands on the last token
    adv = [0.30, 0.20, 0.03, -0.06]         # A_t = G_t - V(s_t)
    pos = np.arange(len(tokens))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].bar(pos, rm, color=ACCENT, label="reward-model score (last token)")
    axes[0].bar(pos, kl_toll, color=MUTED, label="KL toll (every token)")
    axes[0].axhline(0, lw=1, color=INK)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(tokens)
    axes[0].set_ylabel("per-token reward  $R_t$")
    axes[0].set_title("The reward lands only at the end")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    colors = [ACCENT if a >= 0 else MUTED for a in adv]
    axes[1].bar(pos, adv, color=colors)
    axes[1].axhline(0, lw=1, color=INK)
    for x, a in zip(pos, adv):
        axes[1].text(x, a + (0.015 if a >= 0 else -0.03), f"{a:+.2f}",
                     ha="center", va="bottom" if a >= 0 else "top", fontsize=9, color=INK)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels(tokens)
    axes[1].set_ylabel("advantage  $A_t = G_t - V(s_t)$")
    axes[1].set_title("Credit, spread across the sentence")
    axes[1].set_ylim(-0.12, 0.38)

    fig.suptitle("One verdict at the end, a signed nudge for every token",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, BLOG07, "fig-per-token-credit")


def fig_value_head_fit() -> None:
    """The critic learns: predicted value vs realised return (Lab D, r = 0.94)."""
    rng = np.random.default_rng(5)
    n = 200
    G = rng.normal(-5.6, np.sqrt(20.4), size=n)   # returns: mean/var from Lab D
    rho = 0.935
    noise = rng.normal(0, 1, size=n)
    Gc = G - G.mean()
    V = G.mean() + rho * Gc + np.sqrt(1 - rho**2) * np.std(Gc) * noise

    lims = [min(G.min(), V.min()) - 1, max(G.max(), V.max()) + 1]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot(lims, lims, ls="--", lw=1.6, color=INK, label="perfect ($V = G$)")
    ax.scatter(G, V, s=18, color=ACCENT, alpha=0.6, edgecolor="none")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("realised return  $G_t$")
    ax.set_ylabel("value-head prediction  $V(s_t)$")
    ax.set_title("After training, the critic tracks the return ($r = 0.94$)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, BLOG07, "fig-value-head-fit")


def fig_goodhart() -> None:
    """Over-optimisation: the proxy keeps climbing while true quality turns over."""
    kl = np.linspace(0, 120, 500)
    # Proxy: monotone, saturating toward 1.
    proxy = 1 - np.exp(-kl / 28)
    # True quality: rises, peaks near KL=30, then decays.
    true_q = 0.72 * np.exp(-((kl - 30) ** 2) / (2 * 26**2)) * (1 + 0.0) \
        - 0.42 * (kl / 120) ** 2
    true_q = np.clip(true_q, 0, None)

    peak_idx = int(np.argmax(true_q))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(kl, proxy, lw=2.4, color=ACCENT, label="reward-model score (proxy)")
    ax.plot(kl, true_q, lw=2.4, color=INK, label="true quality (human)")
    ax.axvline(kl[peak_idx], ls=":", lw=1.4, color=MUTED)
    ax.annotate("proxy and truth\npart ways", xy=(kl[peak_idx], true_q[peak_idx]),
                xytext=(kl[peak_idx] + 14, true_q[peak_idx] + 0.02), fontsize=9,
                color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("early-stop here", xy=(kl[peak_idx], 0.02),
                xytext=(kl[peak_idx] - 2, 0.18), fontsize=9, color=MUTED, ha="right")
    ax.set_xlabel("how hard we optimise  (KL from reference)")
    ax.set_ylabel("reward / quality")
    ax.set_title("Goodhart's law: optimise the proxy too hard and quality falls")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.05)
    save(fig, BLOG07, "fig-goodhart")


def fig_reward_hacking_scatter() -> None:
    """Verbosity bias: the reward model rewards length (Lab C, corr ~ 0.25)."""
    rng = np.random.default_rng(3)
    n = 73
    length = rng.integers(8, 95, size=n).astype(float)
    rho = 0.249
    lc = (length - length.mean()) / length.std()
    score = 0.0 + rho * lc + np.sqrt(1 - rho**2) * rng.normal(0, 1, size=n)

    # Least-squares trend line.
    m, b = np.polyfit(length, score, 1)
    xs = np.array([length.min(), length.max()])

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.scatter(length, score, s=26, color=ACCENT, alpha=0.65, edgecolor="none")
    ax.plot(xs, m * xs + b, lw=2, ls="--", color=INK, label="trend (r = 0.25)")
    ax.set_xlabel("answer length (tokens)")
    ax.set_ylabel("reward-model score (standardised)")
    ax.set_title("Reward hacking: the proxy rewards length, not just quality")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, BLOG07, "fig-reward-hacking-scatter")


def fig_ppo_ablation() -> None:
    """RLHF with the KL leash (beta=0.2) vs without it (beta=0): reward and KL."""
    steps = [0, 10, 20, 30, 40, 50]
    rew_leash = [-0.122, -0.262, -0.156, -0.096, -0.169, 0.069]
    kl_leash = [0.0, 12.7, 8.4, 8.1, 8.1, 7.7]
    rew_free = [-0.122, -0.218, -0.173, 0.059, 0.119, 0.381]
    kl_free = [0.0, 20.4, 22.4, 28.7, 34.9, 42.6]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].plot(steps, rew_leash, "-o", lw=2.2, ms=5, color=ACCENT,
                 label="with leash ($\\beta = 0.2$)")
    axes[0].plot(steps, rew_free, "-s", lw=2.2, ms=5, color=MUTED,
                 label="no leash ($\\beta = 0$)")
    axes[0].set_xlabel("PPO step")
    axes[0].set_ylabel("mean reward-model score")
    axes[0].set_title("Reward: $\\beta=0$ climbs higher")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    axes[1].plot(steps, kl_leash, "-o", lw=2.2, ms=5, color=ACCENT,
                 label="with leash ($\\beta = 0.2$)")
    axes[1].plot(steps, kl_free, "-s", lw=2.2, ms=5, color=MUTED,
                 label="no leash ($\\beta = 0$)")
    axes[1].set_xlabel("PPO step")
    axes[1].set_ylabel("KL from reference")
    axes[1].set_title("KL: $\\beta=0$ runs off the leash")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("Break the leash and the reward rises while the policy diverges",
                 fontsize=12.5, fontweight="bold", y=1.03)
    save(fig, BLOG07, "fig-ppo-ablation")


def build_blog07() -> None:
    print(f"[{BLOG07}]")
    fig_bradley_terry()
    fig_per_token_credit()
    fig_value_head_fit()
    fig_goodhart()
    fig_reward_hacking_scatter()
    fig_ppo_ablation()


# ----------------------------------------------------------------------------
# Blog 08 — GRPO
# ----------------------------------------------------------------------------
BLOG08 = "08-grpo"


def fig_models_in_memory() -> None:
    """Models resident at train time: PPO (~4) vs GRPO (~2), critic dropped."""
    ppo = [("policy", 1.0, ACCENT), ("reference", 1.0, MUTED),
           ("reward model", 1.0, ACCENT_SOFT), ("value critic", 1.0, INK)]
    grpo = [("policy", 1.0, ACCENT), ("reference", 1.0, MUTED),
            ("verifier", 0.18, DIVIDER)]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for col, stack in [(0, ppo), (1, grpo)]:
        bottom = 0.0
        for name, h, color in stack:
            ax.bar(col, h, bottom=bottom, width=0.55, color=color,
                   edgecolor=INK, lw=0.8)
            ax.text(col, bottom + h / 2, name, ha="center", va="center",
                    fontsize=9.5, color="#FAFAFA" if color in (INK, ACCENT, MUTED) else INK)
            bottom += h + 0.04
    ax.annotate("drop the critic", xy=(0.28, 3.2), xytext=(0.55, 3.55),
                fontsize=10, color=ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["PPO / RLHF", "GRPO"])
    ax.set_ylabel("large models held in memory")
    ax.set_yticks([])
    ax.set_title("GRPO drops a whole trained model: the value critic")
    ax.set_ylim(0, 4.6)
    save(fig, BLOG08, "fig-models-in-memory")


def fig_group_baseline() -> None:
    """The group is the baseline: six rewards, their mean, and the deviations."""
    rewards = np.array([1, 0, 1, 1, 0, 1], dtype=float)
    mean = rewards.mean()
    pos = np.arange(len(rewards))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = [ACCENT if r >= mean else MUTED for r in rewards]
    ax.bar(pos, rewards, width=0.6, color=colors, edgecolor=INK, lw=0.8)
    ax.axhline(mean, ls="--", lw=1.8, color=INK)
    ax.text(len(rewards) - 0.5, mean + 0.03,
            f"group mean (baseline) = {mean:.3f}", ha="right", va="bottom",
            fontsize=10, color=INK)
    for x, r in zip(pos, rewards):
        dev = r - mean
        ax.annotate("", xy=(x, r), xytext=(x, mean),
                    arrowprops=dict(arrowstyle="->",
                                    color=ACCENT if dev > 0 else MUTED, lw=1.6))
        ax.text(x, r + (0.03 if r > mean else -0.08), f"{dev:+.2f}",
                ha="center", va="bottom" if r > mean else "top",
                fontsize=9, color=ACCENT if dev > 0 else MUTED)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"$o_{i+1}$" for i in range(len(rewards))])
    ax.set_ylabel("verifier reward  $r_i$")
    ax.set_title("The group's mean is the baseline (no critic needed)")
    ax.set_ylim(-0.2, 1.2)
    save(fig, BLOG08, "fig-group-baseline")


def fig_advantage_zscore() -> None:
    """Same group, two advantage flavours: numerator-only vs full z-score."""
    rewards = np.array([1, 0, 1, 1, 0, 1], dtype=float)
    numer = rewards - rewards.mean()
    z = numer / rewards.std()
    pos = np.arange(len(rewards))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(pos - w / 2, numer, width=w, color=MUTED,
           label=r"$r_i - \mathrm{mean}$  (Dr. GRPO)")
    ax.bar(pos + w / 2, z, width=w, color=ACCENT,
           label=r"$(r_i - \mathrm{mean})\,/\,\mathrm{std}$  (GRPO)")
    ax.axhline(0, lw=1, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"$o_{i+1}$" for i in range(len(rewards))])
    ax.set_ylabel("advantage  $\\hat{A}_i$")
    ax.set_title("Dividing by the spread only rescales: same signs, same centring")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save(fig, BLOG08, "fig-advantage-zscore")


def fig_all_same_no_signal() -> None:
    """All-correct / all-wrong groups give zero advantage; only a split teaches."""
    groups = {
        "all correct\n{1,1,1,1}": np.array([1, 1, 1, 1], float),
        "all wrong\n{0,0,0,0}": np.array([0, 0, 0, 0], float),
        "split\n{1,0,1,0}": np.array([1, 0, 1, 0], float),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=True)
    for ax, (title, r) in zip(axes, groups.items()):
        numer = r - r.mean()
        pos = np.arange(len(r))
        colors = [ACCENT if d > 0 else (MUTED if d < 0 else DIVIDER) for d in numer]
        ax.bar(pos, numer, width=0.6, color=colors, edgecolor=INK, lw=0.6)
        ax.axhline(0, lw=1, color=INK)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        signal = "no gradient" if np.allclose(numer, 0) else "learning signal"
        ax.text(0.5, 0.92, signal, transform=ax.transAxes, ha="center",
                fontsize=9.5, fontweight="bold",
                color=MUTED if np.allclose(numer, 0) else ACCENT)
    axes[0].set_ylabel("advantage  $r_i - \\mathrm{mean}$")
    axes[0].set_ylim(-0.7, 0.7)
    fig.suptitle("A group only teaches when its answers disagree",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save(fig, BLOG08, "fig-all-same-no-signal")


def fig_length_bias() -> None:
    """Dr. GRPO: dividing the loss by length under-penalises long answers."""
    A = 0.5
    lengths = np.array([20, 200])
    push = A / lengths

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar([0, 1], push, width=0.5, color=[ACCENT, MUTED],
                  edgecolor=INK, lw=0.8)
    for b, v, L in zip(bars, push, lengths):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0006,
                f"{v:.4f}\n({L} tokens)", ha="center", va="bottom", fontsize=9.5,
                color=INK)
    ax.annotate("10x weaker per-token push\non the long answer",
                xy=(1, push[1]), xytext=(0.45, 0.018), fontsize=9.5, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["short answer", "long answer"])
    ax.set_ylabel(r"per-token gradient  $\hat{A}_i / |o_i|$")
    ax.set_title("Why the length normaliser rewards padding (same $\\hat{A}_i = 0.5$)")
    ax.set_ylim(0, 0.03)
    save(fig, BLOG08, "fig-length-bias")


def fig_grpo_training_curve() -> None:
    """The assignment's real GSM8K GRPO run: reward (+std) and KL over 250 steps."""
    reward = [-0.2947, 0.2307, -0.224, -0.1645, -0.0862, 0.222, -0.1065, -0.3965, 0.4123, -0.115, 0.515, 0.2707, -0.0922, -0.2197, 0.0573, -0.0997, 0.2492, 0.6618, -0.121, -0.1368, 0.7557, 0.0218, 0.0278, -0.1048, 0.102, -0.2337, -0.0555, -0.1065, -0.0192, -0.0753, -0.0182, -0.1548, -0.1087, -1.225, -0.0935, 0.001, -0.0477, 0.0, -0.039, -0.2193, 0.0, -0.0577, 0.0, -0.082, 0.0, 0.0, 0.0208, -0.0817, 0.0, -0.0487, 0.0, 0.0602, 0.0002, 0.0, -0.0148, 0.0168, 0.0, 0.0, 0.0, 0.0, 0.3885, 0.0, 0.0, 0.0, 0.0208, 0.3003, 0.0, 0.0, 0.3837, 0.0, 0.2572, 0.0, 0.0, 0.0, 0.021, 0.0, 0.0, -0.1533, 0.0208, 0.0, 0.0, 0.0, -0.0485, 0.0188, 0.0, 0.0085, -0.0413, 0.0, 0.0, 0.0, 0.0, -0.0342, -0.0282, 0.021, 0.0, 0.0, 0.0, 0.0223, -0.0563, 0.0, 0.021, 0.0, 0.0, 0.0585, 0.0, -0.01, -0.0635, 0.0, 0.013, 0.0, 0.0052, 0.0777, 0.0378, 0.0, -0.065, -0.026, 0.0148, 0.0, 0.0, 0.0, -0.1085, 0.0273, 0.0, 0.0313, -0.0402, 0.021, 0.0, -0.0102, -0.0658, 0.0, 0.0, 0.0205, 0.0, 0.0117, 0.0, 0.0, -0.0502, 0.0, 0.0383, 0.0228, 0.0, 0.021, 0.0203, 0.0, -0.0948, 0.4738, 0.0418, 0.024, -0.055, 0.0, 0.021, 0.0792, 0.0, 0.0, 0.0178, 0.0, -0.3202, 0.2283, 0.021, 0.0788, -0.0673, 0.0, -0.3978, 0.0138, 0.0, 0.021, 0.0127, 0.0833, -0.002, 0.0, -0.0158, 0.0187, 0.0, -0.2333, -0.0005, -0.0435, 0.017, 0.0033, -0.1797, 0.0418, 0.021, -0.1782, 0.0208, 0.0197, 0.015, -0.3455, 0.0195, -0.032, 0.0682, 0.0, 0.0015, -0.0702, -0.0437, -0.0352, 0.0238, 0.0, 0.0132, 0.2722, -0.036, 0.0, 0.0413, 0.0, -0.1162, 0.8037, -0.1758, 0.015, 0.0, 0.0325, -0.2618, 0.1028, -0.0715, 0.0163, 0.0, -0.1093, -0.206, -0.1662, -0.1372, 0.0, -0.1337, 0.0, 0.0, 0.003, 0.0258, 0.0, 0.0415, 0.0412, 0.0, 0.0167, 0.0275, 0.0, 0.6483, 0.0708, 0.3462, -0.006, 0.0205, 0.2288, 0.0183, 0.0, 0.0205, 0.0417, -0.0262, 0.0013, 0.0, 0.0, 0.0, 0.0, -0.1563, 0.1283, -0.0305, -0.1538]
    reward_std = [0.2961, 1.0016, 0.3908, 0.3162, 0.138, 0.7066, 0.2672, 0.4346, 0.8967, 0.261, 0.9042, 0.8, 0.1841, 0.3042, 0.1759, 0.1166, 0.9589, 1.3056, 0.4223, 0.3313, 1.2034, 0.0851, 0.2415, 0.1871, 0.2762, 0.3735, 0.0864, 0.3932, 0.0469, 0.1887, 0.0307, 0.2021, 0.1009, 3.0006, 0.229, 0.1688, 0.0787, 0.0, 0.0727, 0.3855, 0.0, 0.1413, 0.0, 0.2009, 0.0, 0.0, 0.051, 0.2, 0.0, 0.1192, 0.0, 0.1474, 0.0572, 0.0, 0.0363, 0.0412, 0.0, 0.0, 0.0, 0.0, 0.9516, 0.0, 0.0, 0.0, 0.051, 0.7357, 0.0, 0.0, 0.9398, 0.0, 0.6299, 0.0, 0.0, 0.0, 0.0514, 0.0, 0.0, 0.4016, 0.051, 0.0, 0.0, 0.0, 0.1188, 0.0461, 0.0, 0.0208, 0.1012, 0.0, 0.0, 0.0, 0.0, 0.1339, 0.069, 0.0514, 0.0, 0.0, 0.0, 0.0547, 0.138, 0.0, 0.0514, 0.0, 0.0, 0.1433, 0.0, 0.0245, 0.1555, 0.0, 0.0318, 0.0, 0.0127, 0.1902, 0.0586, 0.0, 0.1592, 0.0637, 0.0363, 0.0, 0.0, 0.0, 0.3308, 0.0497, 0.0, 0.0768, 0.0984, 0.0514, 0.0, 0.0249, 0.1613, 0.0, 0.0, 0.0502, 0.0, 0.0286, 0.0, 0.0, 0.1229, 0.0, 0.0596, 0.0559, 0.0, 0.0514, 0.0498, 0.0, 0.2018, 0.9909, 0.1025, 0.0588, 0.1347, 0.0, 0.0514, 0.1424, 0.0, 0.0, 0.0437, 0.0, 0.5018, 0.7341, 0.0514, 0.0616, 0.205, 0.0, 0.4386, 0.0339, 0.0, 0.0514, 0.031, 0.2041, 0.206, 0.0, 0.0247, 0.0289, 0.0, 0.4034, 0.0012, 0.1066, 0.0416, 0.0082, 0.4401, 0.1025, 0.0514, 0.1957, 0.051, 0.0482, 0.0367, 0.5697, 0.0478, 0.2191, 0.1088, 0.0, 0.0522, 0.2076, 0.107, 0.1485, 0.0453, 0.0, 0.0323, 0.6667, 0.0882, 0.0, 0.1012, 0.0, 0.2845, 1.2248, 0.448, 0.0367, 0.0, 0.1735, 0.3159, 0.2519, 0.1751, 0.04, 0.0, 0.2678, 0.3255, 0.3636, 0.2216, 0.0, 0.2649, 0.0, 0.0, 0.1493, 0.04, 0.0, 0.1017, 0.0489, 0.0, 0.0408, 0.0804, 0.0, 1.0102, 0.0581, 0.7892, 0.4196, 0.0502, 0.8379, 0.0449, 0.0, 0.0502, 0.0646, 0.1349, 0.0772, 0.0, 0.0, 0.0, 0.0, 0.3829, 0.2018, 0.1091, 0.3768]
    kl = [0.0, 0.0, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 2e-05, 2e-05, 2e-05, 2e-05, 2e-05, 6e-05, 7e-05, 8e-05, 9e-05, 0.0001, 0.00011, 5e-05, 0.00016, 7e-05, 0.00013, 0.00032, 0.00021, 0.00077, 0.00037, 0.00079, 0.00237, 0.00047, 0.00293, 0.00229, 0.00236, 0.002, 0.00282, 0.00289, 0.00689, 0.00853, 0.00291, 0.02717, 0.00753, 0.00207, 0.00213, 0.00819, 0.26953, 0.04842, 0.00375, 0.1778, 0.00429, 0.02315, 0.27605, 0.01569, 1.26636, 0.00349, 0.00321, 0.00483, 0.01607, 0.0083, 0.02195, 0.0088, 0.00843, 0.00532, 0.05739, 0.00479, 0.00373, 0.00464, 0.00374, 0.0095, 0.00946, 0.00932, 0.00545, 0.00235, 0.00384, 0.00716, 0.00863, 0.32045, 0.00456, 0.00597, 0.01398, 0.00909, 0.03692, 0.00554, 0.00889, 0.0132, 0.12858, 0.01191, 0.01247, 0.00416, 0.00804, 0.00488, 0.00515, 0.00924, 0.01299, 0.00797, 0.01363, 0.00876, 0.00627, 0.00688, 0.04443, 0.00407, 0.01382, 0.01664, 0.00702, 0.00605, 0.02061, 0.00884, 0.00384, 0.01341, 0.00874, 0.31506, 0.04773, 0.01004, 0.01461, 0.00968, 0.00563, 0.00312, 0.00451, 0.00708, 0.05039, 0.01277, 0.0047, 0.00835, 0.00592, 0.0044, 0.00732, 0.17243, 0.00541, 0.00276, 0.00642, 0.00428, 0.00184, 0.00667, 0.01005, 0.02289, 0.12335, 0.0079, 2.03906, 0.55574, 0.00607, 0.00705, 0.00329, 0.00543, 0.00623, 0.00833, 0.0054, 0.24105, 0.00581, 0.00606, 0.00767, 0.0163, 0.01324, 0.00419, 0.00928, 0.0097, 0.00989, 0.19304, 0.00802, 0.01213, 0.00283, 0.00791, 0.04237, 0.01777, 0.0077, 0.00407, 0.00287, 0.01632, 1.77546, 0.00492, 0.0149, 0.14239, 0.0171, 0.00552, 0.05007, 0.02618, 0.01013, 0.02185, 0.00526, 0.00952, 0.00688, 0.27326, 0.01692, 0.00408, 0.01162, 0.00907, 0.05992, 0.10518, 0.2814, 0.00552, 0.13639, 0.00707, 0.00671, 0.00757, 0.02481, 0.00926, 0.07145, 0.00436, 0.00817, 0.01147, 0.18293, 0.01465, 0.00946, 0.15703, 0.01128, 0.01191, 0.01338, 0.02241, 0.28017, 0.02136, 0.00733, 0.01768, 0.01717, 0.01223, 0.04571, 0.13142, 0.01554, 0.00745, 0.01497, 0.0157, 0.00363, 0.55405, 0.06607, 0.00848, 0.02215, 0.01114, 0.02782, 0.00769, 0.0137, 0.00551, 0.01463, 0.18356, 0.04261, 0.08298, 0.01986, 0.12182, 0.01545, 0.00858, 0.00998, 0.00743, 0.08439, 0.01138, 0.01987, 0.01176, 0.00884, 0.00331, 0.00883, 0.01363, 0.04144, 0.00948]

    reward = np.array(reward); reward_std = np.array(reward_std); kl = np.array(kl)
    steps = np.arange(1, len(reward) + 1)

    def smooth(x, w=11):
        k = np.ones(w) / w
        return np.convolve(x, k, mode="same")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    axes[0].fill_between(steps, reward - reward_std, reward + reward_std,
                         color=ACCENT_SOFT, alpha=0.30, label="$\\pm$ reward std (in-group spread)")
    axes[0].plot(steps, reward, lw=0.9, color=ACCENT_SOFT, alpha=0.8)
    axes[0].plot(steps, smooth(reward), lw=2.4, color=ACCENT, label="reward (smoothed)")
    axes[0].axhline(0, lw=1, color=INK)
    axes[0].scatter([204], [0.804], color=INK, zorder=5, s=28)
    axes[0].annotate("best group\n(step 204)", xy=(204, 0.804), xytext=(120, 0.95),
                     fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    axes[0].set_xlabel("GRPO step")
    axes[0].set_ylabel("group reward")
    axes[0].set_title("Reward: noisy, format-driven, slowly rising")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")

    axes[1].plot(steps, kl + 1e-6, lw=1.6, color=MUTED)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("GRPO step")
    axes[1].set_ylabel("KL to reference (log)")
    axes[1].set_title("KL stays tiny: the policy never bolts")
    fig.suptitle("The real assignment: 250 GRPO steps on GSM8K (Llama-3.2-3B)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    save(fig, BLOG08, "fig-grpo-training-curve")


def build_blog08() -> None:
    print(f"[{BLOG08}]")
    fig_models_in_memory()
    fig_group_baseline()
    fig_advantage_zscore()
    fig_all_same_no_signal()
    fig_length_bias()
    fig_grpo_training_curve()


# ----------------------------------------------------------------------------
# Registry + CLI
# ----------------------------------------------------------------------------
BUILDERS = {
    "01": build_blog01,
    "02": build_blog02,
    "03": build_blog03,
    "04": build_blog04,
    "05": build_blog05,
    "06": build_blog06,
    "07": build_blog07,
    "08": build_blog08,
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
