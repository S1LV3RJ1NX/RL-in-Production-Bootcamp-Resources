# RLVR Arena — Submission (Countdown, GRPO)

**Author:** s1lv3rj1nx
**Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
**Method:** from-scratch GRPO (group-relative advantage + PPO clipped surrogate + k3 KL leash),
trained with a **dense closeness-shaped reward**.

## Headline result (dev_public, 300 puzzles, greedy, exact verifier)

| model | accuracy | easy | medium | hard | avg_tokens (correct) |
|---|---|---|---|---|---|
| base floor (no RL) | 0.33% | — | — | 0.00% | 20.0 |
| GRPO, original step reward | 12.00% | 25.83% | 3.33% | 1.67% | 16.9 |
| **submitted (GRPO + dense reward)** | **14.67%** | 29.17% | 7.50% | 0.00% | 17.1 |

44x over the base floor (1/300 -> 44/300).

## The `model/` (weights live on the Hugging Face Hub)

The checkpoint is **not** committed to git (full-weight, ~1 GB). Published at:

    https://huggingface.co/s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-dense

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-dense"
model = AutoModelForCausalLM.from_pretrained(repo)
tok   = AutoTokenizer.from_pretrained(repo)
```

(Earlier 12.00% checkpoint: `s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-lr3e6`.)

## Contents of this bundle

| file | what it is |
|---|---|
| `train_grpo.py` | training code — TODO 1 `group_advantages` + TODO 2 `grpo_loss` filled in, `--reward` flag |
| `dev_preds.jsonl` | predictions on `dev_public.jsonl` from the submitted checkpoint (reproduces) |
| `reward_curve.png` | reward / solved-% vs step across runs |
| `REPORT.md` | report: method, LR ablation, dense-reward improvement, failure mode |

## Reproduce the numbers

```bash
uv run python run_model.py \
  --model s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-dense \
  --test data/dev_public.jsonl --out repro.jsonl
uv run python evaluate.py --test data/dev_public.jsonl --predictions repro.jsonl --name repro
```

Decoding is greedy and deterministic, so `repro.jsonl` reproduces `dev_preds.jsonl` -> 14.67%.
