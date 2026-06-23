"""
evaluate.py — the LEADERBOARD scorer. Deterministic. This is the only judge.

Given a split file (the private test set) and a predictions file (one completion per
puzzle), it computes accuracy and emits a leaderboard row. The instructor runs THIS;
the student's self-reported number is never trusted.

    predictions.jsonl : {"id": "test-00007", "completion": "<think>..</think><answer>..</answer>", "num_tokens": 123}
    test file         : output of make_dataset.py (has id, numbers, target, difficulty)

Scoring (see countdown.is_correct — ground-truth, no partial credit):
    accuracy         = fraction solved on the WHOLE private test set         <- primary rank
    accuracy_hard    = fraction solved on the 'hard' split                   <- tie-break 1
    avg_tokens       = mean completion length among CORRECT answers          <- tie-break 2 (shorter wins)
    format_rate      = fraction with a well-formed <think>/<answer>          <- diagnostic

Usage:
    python3 evaluate.py --test data/test_private.jsonl \
                        --predictions submissions/alice/preds.jsonl \
                        --name alice --append leaderboard.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

from countdown import Puzzle, is_correct, has_format


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score(test_path: str, pred_path: str) -> dict:
    test = load_jsonl(test_path)
    preds = {r["id"]: r for r in load_jsonl(pred_path)}

    by_diff_total: dict[str, int] = defaultdict(int)
    by_diff_correct: dict[str, int] = defaultdict(int)
    n_correct = n_format = 0
    correct_token_counts: list[int] = []
    missing = 0

    for row in test:
        diff = row.get("difficulty", "?")
        by_diff_total[diff] += 1
        pred = preds.get(row["id"])
        if pred is None:
            missing += 1
            continue
        completion = pred.get("completion", "")
        if has_format(completion):
            n_format += 1
        puzzle = Puzzle.from_dict(row)
        if is_correct(completion, puzzle):
            n_correct += 1
            by_diff_correct[diff] += 1
            if "num_tokens" in pred:
                correct_token_counts.append(int(pred["num_tokens"]))

    n = len(test)
    avg_tokens = (sum(correct_token_counts) / len(correct_token_counts)) if correct_token_counts else None
    return {
        "n": n,
        "missing_predictions": missing,
        "accuracy": n_correct / n if n else 0.0,
        "accuracy_by_difficulty": {
            d: by_diff_correct[d] / by_diff_total[d] for d in sorted(by_diff_total)
        },
        "accuracy_hard": (by_diff_correct["hard"] / by_diff_total["hard"]) if by_diff_total.get("hard") else 0.0,
        "format_rate": n_format / n if n else 0.0,
        "avg_tokens_when_correct": avg_tokens,
        "n_correct": n_correct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test_private.jsonl")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--name", required=True, help="submission / student name for the board")
    ap.add_argument("--append", default=None, help="optional leaderboard.csv to append to")
    args = ap.parse_args()

    r = score(args.test, args.predictions)

    print(f"\n  Submission: {args.name}")
    print(f"  {'-'*46}")
    print(f"  accuracy (PRIMARY) : {r['accuracy']*100:6.2f}%   ({r['n_correct']}/{r['n']})")
    for d, a in r["accuracy_by_difficulty"].items():
        print(f"    {d:8s}         : {a*100:6.2f}%")
    print(f"  format_rate        : {r['format_rate']*100:6.2f}%")
    if r["avg_tokens_when_correct"] is not None:
        print(f"  avg_tokens(correct): {r['avg_tokens_when_correct']:6.1f}")
    if r["missing_predictions"]:
        print(f"  ⚠ missing predictions for {r['missing_predictions']} puzzles (scored as wrong)")

    if args.append:
        new = not os.path.exists(args.append)
        with open(args.append, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["name", "accuracy", "accuracy_hard", "format_rate", "avg_tokens_correct", "n"])
            w.writerow([
                args.name, f"{r['accuracy']:.4f}", f"{r['accuracy_hard']:.4f}",
                f"{r['format_rate']:.4f}",
                f"{r['avg_tokens_when_correct']:.1f}" if r["avg_tokens_when_correct"] else "",
                r["n"],
            ])
        print(f"  -> appended to {args.append}")

    # also dump the full json next to the predictions
    print("\n  full result:")
    print("  " + json.dumps(r))


if __name__ == "__main__":
    main()
