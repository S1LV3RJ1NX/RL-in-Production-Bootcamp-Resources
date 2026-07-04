"""
plot_rewards.py — overlay GRPO reward (or solved%) curves from one or more training logs.

Each train_log*.txt is treated as one run and drawn on the SAME axes, so you can watch
improvement across experiments. Re-run this after every new training run to refresh the plot.

The training loop prints lines like:
    step  992 | loss +0.0498 | reward 0.138 | solved   6.2%
We parse those, smooth reward with a moving average (raw reward is noisy), and overlay runs.

Usage (from lectures/capstone/):
    # auto-discover all train_log*.txt in the current dir:
    uv run python plot_rewards.py

    # or name specific logs and set the smoothing window:
    uv run python plot_rewards.py train_log_run1_lr1e6.txt train_log_lr3e6.txt --window 25

    # plot solved% instead of mean reward:
    uv run python plot_rewards.py --metric solved
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless (SSH box, no display)
import matplotlib.pyplot as plt
import numpy as np

# step  992 | loss +0.0498 | reward 0.138 | solved   6.2%
LINE_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*loss\s+([+-]?[\d.]+)\s*\|\s*reward\s+([\d.]+)\s*\|\s*solved\s+([\d.]+)%"
)


def parse_log(path: str):
    steps, loss, reward, solved = [], [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue  # skip headers / warnings / checkpoint lines
            steps.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            reward.append(float(m.group(3)))
            solved.append(float(m.group(4)))
    return np.array(steps), np.array(loss), np.array(reward), np.array(solved)


def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def run_label(path: str) -> str:
    name = os.path.basename(path)
    name = re.sub(r"^train_log_?", "", name)  # strip the common prefix
    name = re.sub(r"\.txt$", "", name)
    return name or os.path.basename(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="train_log*.txt files (default: glob train_log*.txt)")
    ap.add_argument("--metric", choices=["reward", "solved"], default="reward")
    ap.add_argument("--window", type=int, default=25, help="moving-average window (smoothing)")
    ap.add_argument("--out", default="reward_curve.png")
    args = ap.parse_args()

    logs = args.logs or sorted(glob.glob("train_log*.txt"))
    if not logs:
        raise SystemExit("No logs found (looked for train_log*.txt). Pass filenames explicitly.")

    col = {"reward": 2, "solved": 3}[args.metric]
    ylabel = {"reward": "mean shaped reward", "solved": "solved % (per batch)"}[args.metric]

    plt.figure(figsize=(9, 5))
    for path in logs:
        parsed = parse_log(path)
        steps, y = parsed[0], parsed[col]
        if len(steps) == 0:
            print(f"  (skip) no parsable step lines in {path}")
            continue
        label = run_label(path)
        line, = plt.plot(steps, y, alpha=0.15, linewidth=0.8)          # faint raw
        sm = moving_avg(y, args.window)
        sm_steps = steps[len(steps) - len(sm):]
        plt.plot(sm_steps, sm, linewidth=2.0, color=line.get_color(),  # bold smoothed
                 label=f"{label}  (final~{sm[-1]:.3f})")
        print(f"  {label}: {len(steps)} steps, final smoothed {args.metric} = {sm[-1]:.3f}")

    plt.xlabel("training step")
    plt.ylabel(ylabel)
    plt.title(f"GRPO {args.metric} vs step  (moving avg, window={args.window})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
