# RLVR Arena — Capstone Report

**Submitted checkpoint:** [`s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-lr3e6`](https://huggingface.co/s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-lr3e6) — loads with `AutoModelForCausalLM.from_pretrained`.

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
| run | accuracy | hard | format_rate | avg_tokens(correct) |
|---|---|---|---|---|
| base (floor, no RL) | 0.33% | 0.00% | 0.00% | 20.0 |
| GRPO Run 1 (lr 1e-6, 1000 steps) | 1.00% | 0.00% | 0.00% | 16.7 |
| **GRPO best (lr 3e-6, group 8, 1500 steps)** | **12.00%** | **1.67%** | 0.00% | 16.9 |

Best GRPO run reached **12.00%** dev accuracy — **36× the base floor** (1/300 → 36/300) — and began
solving hard 5-number puzzles (1.67%). See `reward_curve.png` and the ablation section below for how
the learning rate got us there. (Full per-run comparison in `leaderboard.csv` / `leaderboard.png`.)

## Reward curve (see reward_curve.png)
Three phases: (1) fast rise 0.03→0.05 in the first ~50 steps = learning the format gate;
(2) plateau at ~0.05 to step ~450 = the bare-guess local optimum; (3) slow climb to ~0.10 with
growing spikes after ~450 = the first correct answers appearing. The curve is still rising at
step 1000 (not plateaued), indicating headroom.

## Ablation: learning rate (1e-6 vs 3e-6)
Hypothesis: Run 1's reward was still rising at step 1000, so lr 1e-6 is too gentle. Re-ran at
lr 3e-6, 1500 steps.

| run | lr | steps | dev accuracy | accuracy_hard | final smoothed reward |
|---|---|---|---|---|---|
| Run 1 | 1e-6 | 1000 | 1.00% | 0.00% | ~0.101 |
| Ablation A (best) | 3e-6 | 1500 | **12.00%** | 1.67% | ~0.191 |
| Ablation B | 3e-6 | 3000 | 10.67% | 1.67% | ~0.159 |

LR ablation (A vs Run 1): a 12x jump in dev accuracy, and the hard (5-number) split went from 0 to
non-zero. The overlaid reward curve controls for step count -- the 3e-6 curve dominates Run 1 at
*every shared step* (~0.15 vs ~0.10 at step 1000), so the gain is from the learning rate.

Step ablation (B vs A): doubling steps to 3000 did NOT help -- reward plateaus right after ~1500
(brief spike then flat ~0.13-0.16), and dev accuracy is flat-to-down (10.67% vs 12.00%, within the
~1.8% noise band at n=300). Takeaway: this recipe saturates near step 1500; more steps is wasted
compute. Completions also stay short (~17 tokens even when correct), so max_new_tokens is not the
bottleneck.

Group-size ablation (C): tried `--group 16` (vs 8) to lower advantage variance and raise the odds of
a correct answer per group. At 1200 steps it reached 10.33% (hard 0.00%) -- no improvement, same
plateau band, so it was not run to completion (it also OOM'd at step 1379 from allocator
fragmentation on the ref forward; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the fix, but
the result did not justify a rerun).

**Synthesis.** Three levers, one conclusion: only the learning rate moved the needle (1% -> 12%);
neither more steps nor a larger group helped. The recipe saturates at ~12% and the binding
constraint is the reward design (reasoning collapse, `format_rate = 0` in every run), not the
hyperparameters. Best checkpoint kept for submission: **Ablation A (lr 3e-6, group 8, 1500 steps,
12.00%)**.

## Failure mode: reasoning collapse (format_rate = 0%)
Despite the system prompt asking for `<think>`, the trained model emits **bare** answers with **no
reasoning** (e.g. puzzle `5,5,10` → `<answer>5 * (5 + 5)</answer>`, ~15 tokens, and it even used the
wrong numbers). Root cause: the shaped reward (`countdown.reward`) scores **only** the `<answer>`
content and gives **zero credit for `<think>`**. GRPO therefore found the cheapest reward — skip
reasoning, emit any parsable answer (0.05) — a local optimum it never escaped in 1000 steps. This is
the classic "the model optimizes the reward you gave, not the reward you meant." A `<think>`/format
reward term (as real recipes use, GRPO blog §2.1) would likely pull reasoning back and lift accuracy.
