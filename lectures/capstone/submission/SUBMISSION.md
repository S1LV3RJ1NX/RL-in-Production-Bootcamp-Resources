# RLVR Arena — Submission (Countdown, GRPO)

**Author:** s1lv3rj1nx
**Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Method:** from-scratch GRPO (group-relative advantage + PPO clipped surrogate + k3 KL leash)

## Headline result (dev_public, 300 puzzles, greedy, exact verifier)

| model | accuracy | hard (5-num) | format_rate | avg_tokens (correct) |
|---|---|---|---|---|
| base floor (no RL) | 0.33% | 0.00% | 0.00% | 20.0 |
| **submitted checkpoint (GRPO, lr 3e-6, 1500 steps)** | **12.00%** | **1.67%** | 0.00% | 16.9 |

36x over the base floor (1/300 -> 36/300).

## The `model/` (weights live on the Hugging Face Hub)

The checkpoint is **not** committed to git (full-weight, ~1 GB). It is published at:

    https://huggingface.co/s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-lr3e6

Load it exactly as the harness expects:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-lr3e6"
model = AutoModelForCausalLM.from_pretrained(repo)
tok   = AutoTokenizer.from_pretrained(repo)
```

## Contents of this bundle

| file | what it is |
|---|---|
| `train_grpo.py` | training code — TODO 1 `group_advantages` + TODO 2 `grpo_loss` filled in |
| `dev_preds.jsonl` | predictions on `dev_public.jsonl` (reproduces from the checkpoint below) |
| `reward_curve.png` | reward vs step (Run 1 lr1e6 vs best lr3e6, moving avg) |
| `REPORT.md` | 1-page report: method, results, reward curve, LR ablation, failure mode |

## Reproduce the numbers

```bash
# 1) predictions from the published checkpoint
uv run python run_model.py \
  --model s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-lr3e6 \
  --test data/dev_public.jsonl --out repro.jsonl

# 2) official score (should match dev_preds.jsonl -> 12.00%)
uv run python evaluate.py --test data/dev_public.jsonl --predictions repro.jsonl
```

Decoding is greedy and deterministic, so `repro.jsonl` reproduces `dev_preds.jsonl` byte-for-byte.
