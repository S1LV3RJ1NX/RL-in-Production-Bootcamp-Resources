# RL Capstone — The RLVR Arena 🏆

**Train a small language model with GRPO to solve Countdown number puzzles.**
You submit a checkpoint. I re-run it on puzzles you've never seen and score it. The number goes on a leaderboard.

This is the finale of the whole bootcamp. The throughline — **V(s) = r + γ·V(s′)** — collapses here to its purest form: a *single sparse, verifiable terminal reward* (did the answer hit the target?), no critic, group-relative baseline. It is L5 (RL for LLMs) + L6 (GRPO) made real, on the same stack as [OpenClaw-RL].

---

## 1. The task: Countdown

Given a list of numbers and a target, combine **all** the numbers — each used **exactly once** — with `+ − × ÷` and parentheses to reach the target.

```
Numbers: 3, 7, 8, 2
Target:  24
Answer:  (8 - 2) * (7 - 3)        ✓  = 24, uses 3,7,8,2 once each
```

Why this task is the right capstone:
- **Verifiable by construction.** Correctness is exact arithmetic — no human, no LLM-judge, no fuzzy matching. This is *RLVR* (RL from Verifiable Rewards), the modern paradigm behind your Text-to-SQL and reasoning-model work.
- **Unmemorizable.** Puzzles are generated combinatorially; the test set is held out. You can't win by looking anything up.
- **The "aha moment."** A 0.5B base model goes from ~persistent failure to real reasoning under GRPO — the famous TinyZero effect. You will *watch the reward curve climb.*
- **No gold solutions are provided** — only puzzles + a checker. So plain supervised fine-tuning isn't available to you; you have to learn from the *reward*. That's the point.

---

## 2. Fixed rules (this is what makes the leaderboard fair)

| Rule | Value | Why |
|---|---|---|
| **Base model** | `Qwen2.5-0.5B-Instruct` (frozen starting point) | Everyone starts equal — we measure *your RL*, not who has the bigger GPU. |
| **Method** | GRPO (or any policy-gradient RL you implement). **No SFT on solutions, no distillation.** | It's an RL capstone. The dataset has no solution traces anyway. |
| **No tools at inference** | The model reasons in tokens only. No calculator/code/API calls. | We're training reasoning, not plumbing. (Mirrors the OpenClaw "no API keys" rule.) |
| **Eval decoding** | greedy, `max_new_tokens = 640`, the canonical prompt in `countdown.py` | Deterministic ⇒ reproducible ⇒ fair. I run the *exact* same config for everyone. |
| **Output format** | reasoning in `<think>…</think>`, final expression in `<answer>…</answer>` | The verifier reads the last `<answer>`. |

> Want a stretch? There's an **Open Track** (any base model ≤ 3B) ranked on a separate board — see §7.

---

## 3. What you're given

```
countdown.py          the task engine + the EXACT verifier/reward I will score with
data/train.jsonl      4000 puzzles to train on
data/dev_public.jsonl 300 puzzles to self-score (scored identically to the leaderboard)
train_grpo.py         a GRPO starter with 📝 TODOs (the advantage + loss are yours to write)
run_model.py          turns a checkpoint into predictions.jsonl (the same script I run)
evaluate.py           the leaderboard scorer (run it on dev to see your number)
```

You can self-grade anytime: `python run_model.py --model <ckpt> --test data/dev_public.jsonl --out dev_preds.jsonl` then `python evaluate.py --test data/dev_public.jsonl --predictions dev_preds.jsonl --name me`.

---

## 4. What you submit (the contract)

A single folder / repo containing:

1. **`model/`** — your fine-tuned checkpoint (full weights *or* a LoRA adapter + base id). Must load with `AutoModelForCausalLM.from_pretrained`.
2. **`dev_preds.jsonl`** — your predictions on `dev_public.jsonl` (sanity / reproducibility check).
3. **`train_grpo.py`** — your training code (the TODOs filled in). This is read.
4. **`reward_curve.png`** + a **≤1-page report** — what you tried, the reward curve, one ablation, one failure mode.

> I do **not** trust your reported accuracy. I load your `model/`, run `run_model.py` on the **private** test set myself, and `evaluate.py` produces the official number. Your `dev_preds.jsonl` only has to *reproduce* when I re-run your checkpoint — a mismatch flags tampering.

---

## 5. How it's verified (the harness you can't game)

```
your model/  ──run_model.py──▶  test_preds.jsonl  ──evaluate.py──▶  leaderboard row
                (greedy, fixed)   (private test set)   (exact verifier)
```

- The **private test set** (`data/test_private.jsonl`) is generated with a different seed, is **provably disjoint** from train/dev (checked in `make_dataset.py`), and never leaves my machine.
- The verifier (`countdown.verify`) uses **exact rational arithmetic** and a **whitelisted-AST evaluator** — it never `eval()`s your model's text, so a model that emits `__import__('os')...` scores *invalid*, not *pwned*.
- Everything is deterministic. Re-running gives byte-identical scores.

---

## 6. The leaderboard

| Rank key | Metric | Notes |
|---|---|---|
| **Primary** | accuracy on the full private test set | the headline number |
| Tie-break 1 | accuracy on the **hard** split (5 numbers) | rewards real reasoning over easy-puzzle grinding |
| Tie-break 2 | avg tokens among correct answers (**fewer wins**) | efficiency — no rambling |
| Diagnostic | format-compliance rate | did it learn the protocol? |

`evaluate.py` appends a row to `leaderboard.csv`; I render it as a ranked page (same Vercel style as the decks). Reference points seeded on the board: **base model (no RL)** as the floor, and **my reference GRPO run** as a target to beat.

---

## 7. Tiers

- **Core (everyone):** beat the base model by a clear margin on the fixed `Qwen2.5-0.5B-Instruct`. Everyone who submits a working checkpoint + report passes.
- **Competitive:** climb the 0.5B leaderboard (reward shaping, KL tuning, group size, curriculum easy→hard, length control).
- **Open Track (stretch):** any base model ≤ 3B, separate board. Show GRPO still drives the gain (your `reward_curve.png` must show it, not just a strong base).

---

## 8. Grading (100 pts)

| | pts |
|---|---|
| Working checkpoint that loads + beats the base floor | 40 |
| GRPO implemented correctly (the advantage + loss TODOs, read in your code) | 25 |
| Report: reward curve + one ablation + one failure mode | 20 |
| Reproducibility (your dev_preds match my re-run) | 10 |
| Leaderboard placement (curved) | 5 |

---

## 9. Compute

- **Floor:** free Colab **T4** — `Qwen2.5-0.5B-Instruct` GRPO fits (LoRA, small group, short generations). A few hours gets a visible climb.
- **Recommended:** **Modal** (`teamvizuara` profile) — one A10/A100, full run in ~1–2 h. Mirrors the [OpenClaw-RL] SGLang-generation + train loop you saw in lecture.
- Keep generations short early (`max_new_tokens` 256→640) — generation dominates wall-clock.

---

## 10. Timeline (suggested, 1 week)

| Day | Milestone |
|---|---|
| 1 | Run the base model through `evaluate.py` on dev — see the floor. Read `countdown.py`. |
| 2–3 | Fill the GRPO TODOs in `train_grpo.py`; get reward moving on **easy** only. |
| 4–5 | Full difficulty mix; tune (group size, KL, reward shaping, length). |
| 6 | Lock a checkpoint; write the report + reward curve. |
| 7 | Submit. Leaderboard goes live. |

---

### Files in this folder
`countdown.py` · `make_dataset.py` · `evaluate.py` · `run_model.py` · `train_grpo.py` · `selftest.py` (verifier proof) · `data/`

[OpenClaw-RL]: ../../OpenClaw-RL-Tutorial/
