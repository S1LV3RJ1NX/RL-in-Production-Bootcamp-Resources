# RL Capstone — RLVR Arena

Train a small language model with GRPO to solve Countdown number puzzles. Submit your checkpoint. It gets re-run on a held-out test set. The number goes on a leaderboard.

---

## The task

Given a list of numbers and a target, combine **all** the numbers — each used exactly once — with `+ − × ÷` and parentheses to reach the target.

```
Numbers: 3, 7, 8, 2    Target: 24
Answer:  (8 - 2) * (7 - 3)   ✓
```

Your model must reason inside `<think>…</think>` and give its final expression inside `<answer>…</answer>`. That's the entire output contract.

---

## What you implement

Open `train_grpo.py`. Two functions are marked `📝 TODO`:

1. **`group_advantages`** — the one idea that defines GRPO: compute how much better each answer was relative to the other answers sampled from the same puzzle. Review the L6 slides.

2. **`grpo_loss`** — the loss that combines what you learned in L4 (PPO clip) and L5 (KL leash to the reference model). No critic.

Everything else — model loading, the rollout, per-token log-probs, the training loop, checkpointing — is provided.

---

## Files

| File | Purpose |
|---|---|
| `countdown.py` | The task engine. The exact verifier used to score your submission. |
| `train_grpo.py` | GRPO training starter — fill the two TODOs. |
| `run_model.py` | Turn your checkpoint into a `predictions.jsonl` for self-scoring. |
| `evaluate.py` | Score a predictions file → accuracy breakdown. |
| `data/train.jsonl` | 4000 puzzles to train on (easy / medium / hard). |
| `data/dev_public.jsonl` | 300 puzzles to self-score. Scored identically to the leaderboard. |
| `CAPSTONE.md` | Full spec: base model, submission rules, leaderboard, rubric, timeline. |
| `requirements.txt` | Python dependencies. |

---

## Quick start

```bash
pip install -r requirements.txt

# Smoke-test: does the plumbing run? (CPU, tiny model, 2 steps)
python train_grpo.py --model sshleifer/tiny-gpt2 --steps 2 --group 4 --bsz 2 --smoke

# Real training (T4 GPU, ~2-3 hours):
# Note: T4 has 16GB. group*bsz sequences all pass through the model in one step.
# Keep group*bsz ≤ 16 and add gradient_checkpointing_enable() if you hit OOM.
python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 400 --group 4 --bsz 4

# Self-score on the public dev set:
python run_model.py --model ./model --test data/dev_public.jsonl --out dev_preds.jsonl
python evaluate.py --test data/dev_public.jsonl --predictions dev_preds.jsonl --name yourname
```

The dev accuracy you see is the same metric used for the leaderboard. Your self-reported number is **never** trusted — the instructor re-runs your checkpoint.

---

## How to submit

**Email [rajatdandekar@vizuara.com](mailto:rajatdandekar@vizuara.com)** with the subject line:
`[RLVR Capstone] <Your Name> — <self-reported dev accuracy>%`

For example: `[RLVR Capstone] Alice Sharma — 14.3%`

Include in the email:

1. **Your self-reported dev accuracy** — run `evaluate.py` on `dev_public.jsonl` and paste the output.
2. **HuggingFace Hub link to your model** — upload your `model/` folder to a free HF repo (`huggingface-cli upload yourname/grpo-countdown ./model`) and share the URL. The checkpoint must load with `AutoModelForCausalLM.from_pretrained`.
3. **`dev_preds.jsonl`** — attach the file directly (it's small, ~300 lines).
4. **`train_grpo.py`** — your filled-in training code (the two TODOs implemented).
5. **`reward_curve.png`** — a plot of mean reward vs. training step from your training logs.
6. **≤ 1-page report** — what you tried, one ablation, one failure mode. PDF or plain text is fine.

> Your self-reported accuracy is **never** the official score. Once your email arrives, your checkpoint gets re-run on a **private held-out test set** using the exact same `run_model.py` and `evaluate.py`. That number goes on the leaderboard — not what you report.

See `CAPSTONE.md` for the full rubric and grading breakdown.

---

## Compute

**Colab T4 (free):** the 0.5B model fits at float16. Use `--group 4 --bsz 4`. Checkpoint to Drive to survive disconnects.

**Modal (if you have an account):** free account at modal.com, run `modal token new` once, then use `A10G` with `--group 8 --bsz 2`. A 1,000-step run takes ~2 hours and costs ~$1–2.

---

## Self-check before submitting

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("./model")   # must not error
```

```bash
python evaluate.py --test data/dev_public.jsonl \
    --predictions dev_preds.jsonl --name me
# your reported accuracy must match what you get when you re-run run_model.py
```
