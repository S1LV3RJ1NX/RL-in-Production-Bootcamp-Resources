"""
countdown.py — the verifiable task engine for the RL Capstone ("RLVR Arena").

This is the SINGLE SOURCE OF TRUTH for the Countdown ("reach the target") task:

  * puzzle generation .......... guaranteed-solvable, difficulty-stratified
  * the canonical prompt ....... identical for everyone (fairness)
  * answer extraction .......... pull the final expression out of a completion
  * the VERIFIER ............... safe, exact-arithmetic checker  -> used by the LEADERBOARD
  * the shaped RLVR reward ..... dense signal                    -> used by GRPO TRAINING

Design rules:
  - No third-party deps. Pure standard library.
  - Exact arithmetic via fractions.Fraction (no float rounding bugs).
  - NEVER call eval()/exec() on model output. A whitelisted-AST evaluator is used,
    so a model that emits  __import__('os').system('rm -rf ~')  inside <answer>
    is simply scored "invalid", never executed. (Teachable moment for students.)

The leaderboard uses ONLY `is_correct()` (ground-truth 0/1). The shaped `reward()`
is for training and must never leak into scoring.
"""
from __future__ import annotations

import ast
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional


# --------------------------------------------------------------------------- #
# 1. Puzzle representation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Puzzle:
    numbers: tuple[int, ...]
    target: int

    def key(self) -> tuple:
        """Canonical identity, used to keep train/dev/test disjoint."""
        return (tuple(sorted(self.numbers)), self.target)

    def to_dict(self) -> dict:
        return {"numbers": list(self.numbers), "target": self.target}

    @staticmethod
    def from_dict(d: dict) -> "Puzzle":
        return Puzzle(tuple(int(x) for x in d["numbers"]), int(d["target"]))


# --------------------------------------------------------------------------- #
# 2. The canonical prompt (every submission is evaluated with THIS, verbatim)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a careful reasoner who solves Countdown number puzzles.\n"
    "You are given a list of numbers and a target. Combine ALL the numbers, using "
    "each one exactly once, with + - * / and parentheses, to make an arithmetic "
    "expression that equals the target.\n"
    "First think step by step inside <think> </think>. Then output ONLY the final "
    "expression inside <answer> </answer>. For example: "
    "<answer>(3 + 7) * 8 / 2 - 11</answer>."
)


def format_user_prompt(puzzle: Puzzle) -> str:
    nums = ", ".join(str(n) for n in puzzle.numbers)
    return f"Numbers: {nums}\nTarget: {puzzle.target}"


def chat_messages(puzzle: Puzzle) -> list[dict]:
    """The exact message list to feed `tokenizer.apply_chat_template(...)`."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_prompt(puzzle)},
    ]


# --------------------------------------------------------------------------- #
# 3. Answer extraction
# --------------------------------------------------------------------------- #
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def extract_answer(completion: str) -> Optional[str]:
    """Return the contents of the LAST <answer>...</answer> block, or None."""
    matches = _ANSWER_RE.findall(completion or "")
    if not matches:
        return None
    return matches[-1].strip()


def has_format(completion: str) -> bool:
    """True if the completion has both a <think> block and an <answer> block."""
    return bool(_THINK_RE.search(completion or "")) and bool(_ANSWER_RE.search(completion or ""))


# --------------------------------------------------------------------------- #
# 4. The VERIFIER  (safe, exact). This is what the leaderboard trusts.
# --------------------------------------------------------------------------- #
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_SANITIZE_RE = re.compile(r"^[0-9+\-*/()\s]+$")
_MAX_EXPR_LEN = 256


@dataclass(frozen=True)
class VerifyResult:
    correct: bool          # uses each number once AND evaluates to target
    parsable: bool         # was a safe arithmetic expression at all
    numbers_ok: bool       # used exactly the given multiset of numbers
    value: Optional[Fraction]
    reason: str


def _collect_and_eval(node: ast.AST, leaves: list[int]) -> Fraction:
    """Walk a whitelisted AST: evaluate exactly AND record integer leaves used."""
    if isinstance(node, ast.Expression):
        return _collect_and_eval(node.body, leaves)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _collect_and_eval(node.left, leaves)
        right = _collect_and_eval(node.right, leaves)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("division by zero")
        return left / right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _collect_and_eval(node.operand, leaves)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        leaves.append(node.value)
        return Fraction(node.value)
    raise ValueError(f"disallowed token: {type(node).__name__}")


def verify(puzzle: Puzzle, expression: Optional[str]) -> VerifyResult:
    """Check a single arithmetic expression against a puzzle. Never executes code."""
    if not expression:
        return VerifyResult(False, False, False, None, "empty answer")
    expr = expression.strip()
    if len(expr) > _MAX_EXPR_LEN or not _SANITIZE_RE.match(expr):
        return VerifyResult(False, False, False, None, "illegal characters")
    try:
        tree = ast.parse(expr, mode="eval")
        leaves: list[int] = []
        value = _collect_and_eval(tree, leaves)
    except Exception as e:  # noqa: BLE001 — any parse/eval failure = invalid, not fatal
        return VerifyResult(False, False, False, None, f"unparsable: {e}")

    numbers_ok = Counter(leaves) == Counter(puzzle.numbers)
    is_target = value == puzzle.target
    correct = numbers_ok and is_target
    if correct:
        reason = "correct"
    elif not numbers_ok:
        reason = f"wrong numbers used: {sorted(leaves)} vs {sorted(puzzle.numbers)}"
    else:
        reason = f"evaluates to {value} != {puzzle.target}"
    return VerifyResult(correct, True, numbers_ok, value, reason)


# --------------------------------------------------------------------------- #
# 5. Scoring  (leaderboard) and Reward  (training)
# --------------------------------------------------------------------------- #
def is_correct(completion: str, puzzle: Puzzle) -> bool:
    """Ground-truth 0/1 used by the LEADERBOARD. No partial credit, no shaping."""
    return verify(puzzle, extract_answer(completion)).correct


def reward(completion: str, puzzle: Puzzle) -> float:
    """
    Shaped RLVR reward for GRPO TRAINING (dense -> easier to bootstrap):
        1.00  correct
        0.10  right numbers, wrong value (real attempt)
        0.05  a parsable expression at all
        0.00  no answer / garbage
    Do NOT use this for the leaderboard — use is_correct().
    """
    ans = extract_answer(completion)
    if ans is None:
        return 0.0
    res = verify(puzzle, ans)
    if res.correct:
        return 1.0
    if res.numbers_ok:
        return 0.10
    if res.parsable:
        return 0.05
    return 0.0


# --------------------------------------------------------------------------- #
# 6. Puzzle generation (guaranteed solvable)
# --------------------------------------------------------------------------- #
DIFFICULTY = {
    # 3 numbers = easiest tier; widened range so the puzzle space is large enough
    # to draw thousands of UNIQUE puzzles without dedup collisions.
    "easy":   dict(n=3, num_lo=1, num_hi=20, tgt_lo=10, tgt_hi=150),
    "medium": dict(n=4, num_lo=1, num_hi=12, tgt_lo=20, tgt_hi=100),
    "hard":   dict(n=5, num_lo=1, num_hi=15, tgt_lo=50, tgt_hi=300),
}


def _random_reachable_value(numbers: tuple[int, ...], rng: random.Random) -> Fraction:
    """Combine the numbers pairwise with random ops -> one Fraction (Countdown-style)."""
    vals = [Fraction(n) for n in numbers]
    rng.shuffle(vals)
    while len(vals) > 1:
        a = vals.pop()
        b = vals.pop()
        op = rng.choice("+-*/")
        if op == "+":
            vals.append(a + b)
        elif op == "-":
            vals.append(a - b)
        elif op == "*":
            vals.append(a * b)
        else:
            vals.append(a / b if b != 0 else a + b)
    return vals[0]


def gen_solvable(difficulty: str, rng: random.Random, tries: int = 600) -> Puzzle:
    cfg = DIFFICULTY[difficulty]
    for _ in range(tries):
        numbers = tuple(rng.randint(cfg["num_lo"], cfg["num_hi"]) for _ in range(cfg["n"]))
        for _ in range(80):
            v = _random_reachable_value(numbers, rng)
            if v.denominator == 1:
                t = int(v)
                if cfg["tgt_lo"] <= t <= cfg["tgt_hi"] and t not in numbers:
                    return Puzzle(numbers, t)
    raise RuntimeError(f"could not generate a solvable '{difficulty}' puzzle; loosen ranges")


def generate_set(
    n_per_difficulty: dict[str, int],
    seed: int,
    exclude: Optional[set] = None,
) -> list[tuple[Puzzle, str]]:
    """Generate a deduped list of (Puzzle, difficulty), disjoint from `exclude` keys."""
    rng = random.Random(seed)
    exclude = set(exclude or set())
    out: list[tuple[Puzzle, str]] = []
    seen: set = set()
    for difficulty, count in n_per_difficulty.items():
        made = 0
        guard = 0
        while made < count:
            guard += 1
            if guard > count * 200 + 1000:
                raise RuntimeError(f"stuck generating '{difficulty}' — too many collisions")
            p = gen_solvable(difficulty, rng)
            k = p.key()
            if k in seen or k in exclude:
                continue
            seen.add(k)
            out.append((p, difficulty))
            made += 1
    return out


if __name__ == "__main__":
    # tiny self-demo
    rng = random.Random(0)
    for diff in ("easy", "medium", "hard"):
        p = gen_solvable(diff, rng)
        print(f"{diff:6s}  numbers={list(p.numbers)}  target={p.target}")
