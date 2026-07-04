# RLVR Arena — Capstone Report

**Task.** Train `Qwen2.5-0.5B-Instruct` with GRPO to solve Countdown puzzles (combine all given
numbers exactly once with `+ - * /` to hit a target), reasoning in `<think>` and answering in
`<answer>`. Reward is verifiable (exact arithmetic), so this is RLVR — no reward model, no critic.

## Method (what I implemented)
From-scratch GRPO in `train_grpo.py`; I wrote the two core functions:
- **`group_advantages`** — group-relative advantage `A_i = (r_i - mean)/(std + 1e-8)`, standardized
  within each puzzle's group of sampled answers. The group mean is the baseline (no value critic).
- **`grpo_loss`** — PPO clipped surrogate `min(ρA, clip(ρ,1±ε)A)` with `ρ = exp(logp - logp_old)`,
  plus a k3 KL leash to the frozen reference `exp(Δ) - Δ - 1` (Δ = logp_ref - logp), averaged over
  completion tokens only (masked mean), returned negative.

## Setup / compute
- Hardware: 1× NVIDIA A10 (24 GB). Full fine-tune in bf16 (policy + frozen reference).
- Config (Run 1): `--group 8 --bsz 2 --max-new-tokens 128 --lr 1e-6 --steps 1000`.
- Memory lesson: OOM at bsz4/256-tok and bsz2/256-tok — the killer is the fp32 vocab-logits tensor
  `[bsz*group, seq_len, ~152k]`, not the 0.5B weights. Cutting `max-new-tokens` to 128 (+ expandable
  segments) fit comfortably; gradient checkpointing was not needed.

## Results (dev_public, 300 puzzles, greedy, exact verifier)
| run | accuracy | easy | hard | format_rate | avg_tokens(correct) |
|---|---|---|---|---|---|
| base (floor, no RL) | 0.33% | 0.00% | 0.00% | 0.00% | 20.0 |
| GRPO Run 1 (lr 1e-6, 1000 steps) | **1.00%** | 1.67% | 0.00% | 0.00% | 16.7 |

GRPO tripled accuracy over the floor (1/300 → 3/300) and taught the model to reliably emit a
parsable `<answer>` (the base model never did). See `reward_curve.png`.

## Reward curve (see reward_curve.png)
Three phases: (1) fast rise 0.03→0.05 in the first ~50 steps = learning the format gate;
(2) plateau at ~0.05 to step ~450 = the bare-guess local optimum; (3) slow climb to ~0.10 with
growing spikes after ~450 = the first correct answers appearing. The curve is still rising at
step 1000 (not plateaued), indicating headroom.

## Ablation: learning rate (1e-6 vs 3e-6)   [TO COMPLETE after the lr-3e-6 run]
Hypothesis: Run 1's reward was still climbing at step 1000, so lr 1e-6 is too gentle. Re-ran at
lr 3e-6 (single changed variable), 1500 steps. Result: <fill in dev accuracy + curve comparison
from the overlaid reward_curve.png>.

## Failure mode: reasoning collapse (format_rate = 0%)
Despite the system prompt asking for `<think>`, the trained model emits **bare** answers with **no
reasoning** (e.g. puzzle `5,5,10` → `<answer>5 * (5 + 5)</answer>`, ~15 tokens, and it even used the
wrong numbers). Root cause: the shaped reward (`countdown.reward`) scores **only** the `<answer>`
content and gives **zero credit for `<think>`**. GRPO therefore found the cheapest reward — skip
reasoning, emit any parsable answer (0.05) — a local optimum it never escaped in 1000 steps. This is
the classic "the model optimizes the reward you gave, not the reward you meant." A `<think>`/format
reward term (as real recipes use, GRPO blog §2.1) would likely pull reasoning back and lift accuracy.
