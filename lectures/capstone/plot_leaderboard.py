"""
plot_leaderboard.py — bar chart of leaderboard.csv so you can watch runs improve over time.

Reads the CSV that `evaluate.py --append leaderboard.csv` builds and draws grouped bars
(accuracy = primary, accuracy_hard = tie-break 1, format_rate = diagnostic) per run, in the
order they were appended (chronological), so the climb over experiments is visible at a glance.

Usage (from lectures/capstone/):
    uv run python plot_leaderboard.py
    uv run python plot_leaderboard.py --csv leaderboard.csv --out leaderboard.png
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")  # headless (SSH box, no display)
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="leaderboard.csv")
    ap.add_argument("--out", default="leaderboard.png")
    args = ap.parse_args()

    names, acc, hard, fmt = [], [], [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            acc.append(float(row["accuracy"]) * 100)
            hard.append(float(row["accuracy_hard"]) * 100)
            fmt.append(float(row["format_rate"]) * 100)

    if not names:
        raise SystemExit(f"No rows in {args.csv}")

    x = np.arange(len(names))
    w = 0.27
    plt.figure(figsize=(max(7, 1.6 * len(names)), 5))
    b1 = plt.bar(x - w, acc, w, label="accuracy (primary)", color="#2b8cbe")
    b2 = plt.bar(x, hard, w, label="accuracy_hard (tie-break 1)", color="#e6550d")
    b3 = plt.bar(x + w, fmt, w, label="format_rate (diagnostic)", color="#31a354")

    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2, h + 0.3, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=8)

    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("percent")
    plt.title("Countdown GRPO leaderboard (dev_public)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}  ({len(names)} runs)")


if __name__ == "__main__":
    main()
