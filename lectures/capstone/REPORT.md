# RLVR Arena — Capstone Report

**Submitted checkpoint:** [`s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-dense`](https://huggingface.co/s1lv3rj1nx/countdown-qwen2.5-0.5b-grpo-dense) (14.67% dev) — loads with `AutoModelForCausalLM.from_pretrained`. (Earlier 12.00% checkpoint: `…-grpo-lr3e6`.)

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
| run | accuracy | easy | medium | hard | format_rate | avg_tokens(correct) |
|---|---|---|---|---|---|---|
| base (floor, no RL) | 0.33% | — | — | 0.00% | 0.00% | 20.0 |
| GRPO Run 1 (lr 1e-6, 1000 steps) | 1.00% | — | — | 0.00% | 0.00% | 16.7 |
| GRPO (lr 3e-6, group 8, 1500 steps) | 12.00% | 25.83% | 3.33% | 1.67% | 0.00% | 16.9 |
| **GRPO + dense reward (lr 3e-6, 1500 steps)** | **14.67%** | 29.17% | 7.50% | 0.00% | 0.00% | 17.1 |

Best run reached **14.67%** dev accuracy — **44× the base floor** (1/300 → 44/300). The learning rate
took us from 1% → 12% (ablation below); a **dense closeness reward** then took 12% → 14.67% (reward
section below). See `reward_curve.png` and `leaderboard.png` for the full comparison.

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

**Synthesis of hyperparameter ablations.** Only the learning rate moved the needle (1% -> 12%);
neither more steps nor a larger group helped. The recipe saturated at ~12% — and the diagnosis was
that the binding constraint is the **reward design**, not the hyperparameters. That motivated the
next section.

## Improvement: dense (closeness-shaped) reward — 12.00% -> 14.67%
The original `reward()` gives the SAME 0.10 whether an expression evaluates 1 away or 5000 away from
the target. Inside a GRPO group, near-misses then all tie -> zero std -> zero advantage -> no
gradient. So I added `dense_reward()` (in `countdown.py`, selected via `--reward dense`; the
leaderboard scorer is untouched — still `is_correct()`): keep 1.0 for exact, but for the
right-numbers/wrong-value case return `0.10 + 0.85 * exp(-|value - target| / 10)`, a smooth slope
that pays more the closer you land. Ceiling ~0.95 stays below the exact reward of 1.0, so exact is
still strictly best.

| reward | dev accuracy | easy | medium | hard |
|---|---|---|---|---|
| original step-function | 12.00% | 25.83% | 3.33% | 1.67% |
| **dense closeness** | **14.67%** | 29.17% | 7.50% | 0.00% |

+8 correct puzzles (36 -> 44). The gain is concentrated in **medium (3.3% -> 7.5%, doubled)** and easy
— exactly where the model often lands *near* the target, which is what the dense signal rewards. Hard
(5-number) fell to 0 (near-misses are rarer there, and it's a small-n split), a minor tie-break-1
tradeoff. Net: converting the reward from a cliff into a slope broke the ~12% plateau. Submitted
checkpoint: **dense reward, lr 3e-6, group 8, 1500 steps (14.67%)**.

## Failure mode: reasoning collapse (format_rate = 0%)
Despite the system prompt asking for `<think>`, the trained model emits **bare** answers with **no
reasoning** (e.g. puzzle `5,5,10` → `<answer>5 * (5 + 5)</answer>`, ~15 tokens, and it even used the
wrong numbers). Root cause: the shaped reward (`countdown.reward`) scores **only** the `<answer>`
content and gives **zero credit for `<think>`**. GRPO therefore found the cheapest reward — skip
reasoning, emit any parsable answer (0.05) — a local optimum it never escaped in 1000 steps. This is
the classic "the model optimizes the reward you gave, not the reward you meant."

I tested the obvious remedy: a `dense_format` reward (dense + a +0.10 bonus for a non-trivial
`<think>` block) plus an entropy bonus for exploration. It *failed* — accuracy fell to 9.67% and
`format_rate` stayed at 0. Root cause is cold-start: the policy almost never *samples* a `<think>`
block, so the bonus is essentially never triggered and provides no gradient toward format; and the
closeness term (≤0.95) dwarfs the 0.10 bonus, so bare near-misses stay optimal. Reward-only induction
of a rare structured behavior is very hard at 0.5B; the clean fix (a short SFT warm-up on `<think>`
traces) is disallowed by the rules, so the dense-reward checkpoint (14.67%) stands as the best result.
