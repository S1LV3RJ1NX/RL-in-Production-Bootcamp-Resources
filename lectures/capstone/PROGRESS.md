# RLVR Arena Capstone — Progress Log

A running journal of what we've done, the commands run, and the outputs seen.
Newest step at the bottom. Resume from the "CURRENT STEP" marker.

Environment: NVIDIA A10-24Q (24 GB), driver CUDA 12.9, over SSH. Package manager: `uv`.
Always run project code with `uv run python ...` (uses the synced `.venv`, Python 3.13),
NOT the conda `jupyter-base` interpreter.

Plan (baseline FIRST, then the two TODOs, then train):
1. [DONE] Environment setup + confirm CUDA.
2. [DONE] Baseline floor: untrained Qwen through run_model.py + evaluate.py on dev.
3. [IN PROGRESS] TODO 1 `group_advantages` (GRPO blog 2.2).
4. [TODO] TODO 2 `grpo_loss` (blog 2.3 clip + 2.4 KL).
5. [TODO] Smoke test with tiny-gpt2.
6. [TODO] Real GRPO training on the A10 (Qwen2.5-0.5B-Instruct).
7. [TODO] Re-evaluate vs floor + reward_curve.png + report.

---

## Step 1 — Environment setup + CUDA (DONE)

Problem chain we hit and fixed (all in root `pyproject.toml`):

1. `uv run` refused: lockfile only supported `sys_platform == 'darwin'` (macOS).
   Fix: changed `[tool.uv] environments` to include Linux:
   `environments = ["sys_platform == 'linux'", "sys_platform == 'darwin'"]`
2. Default PyPI torch installed as `2.12.1+cu130` (CUDA 13.0) → driver 12.9 too old →
   `cuda available: False`. Fix: added an explicit PyTorch CUDA index + platform-scoped source.
3. cu128 index only had torch <= 2.11.0 (our pin is >=2.12.0). Fix: switched to the
   **cu129** index (matches the 12.9 driver exactly).

Final `pyproject.toml` additions:
```toml
[tool.uv]
environments = ["sys_platform == 'linux'", "sys_platform == 'darwin'"]

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu129", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cu129", marker = "sys_platform == 'linux'" }]
```

Commands:
```bash
uv lock && uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```
Output (GREEN):
```
torch 2.12.1+cu129
cuda available: True
device: NVIDIA A10-24Q
bf16 supported: True
```

---

## Step 2 — Baseline floor (DONE)

Run from inside `lectures/capstone/` (scripts import `countdown` and use `data/` relative paths).
Uses only the PROVIDED harness — no training code.

Commands (in tmux):
```bash
cd lectures/capstone
uv run python run_model.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --test data/dev_public.jsonl --out baseline_dev_preds.jsonl
uv run python evaluate.py --test data/dev_public.jsonl \
    --predictions baseline_dev_preds.jsonl --name base-qwen-floor
```

Output — THE FLOOR (must beat this by a clear margin for Core / 40 pts):
```
accuracy (PRIMARY) :   0.33%   (1/300)
    easy   : 0.00%   medium : 0.83%   hard : 0.00%
format_rate        :   0.00%
avg_tokens(correct): 20.0
```
Reading:
- 0.33% accuracy = the "persistent failure" TinyZero starting point (blog 3).
- format_rate 0% = the base model never emits both <think> and <answer>, despite the prompt.
- The 1 correct answer had an <answer> but no <think> (so correct, but not format-compliant).
- Key link to training: countdown.reward = 0 unless there's an <answer> tag, so GRPO is
  forced to learn the format because it's the gate to ALL positive reward.

Artifacts: `lectures/capstone/baseline_dev_preds.jsonl`

---

## Step 3 — TODO 1 `group_advantages` (DONE)

Implemented in `train_grpo.py`. Final body:
```python
rewards_reshaped_by_group = rewards.reshape(-1, group)          # [bsz, group]
group_mean, std = rewards_reshaped_by_group.mean(dim=1), rewards_reshaped_by_group.std(dim=1)  # [bsz]
advantages = (rewards_reshaped_by_group - group_mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-8)  # [bsz, group]
return advantages.reshape(-1)                                  # [bsz*group] = [B]
```
Key lessons:
- Reduce over dim=1 (per-puzzle, across the group), NOT the whole tensor.
- Must return FLAT [bsz*group] so it aligns with seqs[i]/metas[i] and broadcasts
  against comp_mask [B, T-1] in the loss. (Bug we caught: returned [bsz, group].)
- +1e-8 makes all-same groups -> 0 advantage (no signal), not nan.
- Also added detailed learning comments to the provided `rollout` fn.

---

## Step 4 — TODO 2 `grpo_loss` (DONE)

Implemented + commented in `train_grpo.py`. Final body:
```python
policy_ratio = torch.exp(logp - logp_old)
policy_ratio_clipped = torch.clamp(policy_ratio, 1 - clip_eps, 1 + clip_eps)
A = advantages.unsqueeze(-1)
surrogate = torch.min(policy_ratio_clipped * A, policy_ratio * A)
KL = torch.exp(logp_ref - logp) - (logp_ref - logp) - 1
per_token_objective = surrogate - kl_beta * KL
loss = -(per_token_objective * comp_mask).sum() / comp_mask.sum().clamp(min=1)
return loss
```
Key lessons:
- ratio = exp(logp - logp_old) (PPO importance weight, blog 2.3).
- min(clipped*A, ratio*A) = PPO pessimistic clip; A = advantages.unsqueeze(-1) broadcasts [B]->[B,1].
- k3 KL = exp(ref-pol) - (ref-pol) - 1, always >=0 (blog 2.4).
- BUG we caught: used .mean() -> must be masked mean sum()/comp_mask.sum() so padding/prompt
  tokens and answer length don't rescale the gradient (blog 4 length bias).
- Return NEGATIVE (optimizer minimizes; we maximize the objective).

---

## Step 5 — Smoke test (DONE)

Ran: `uv run python train_grpo.py --model sshleifer/tiny-gpt2 --steps 2 --group 4 --bsz 2 --smoke`
Output:
```
step 0 | loss -0.0000 | reward 0.000 | solved 0.0%
step 1 | loss -0.0000 | reward 0.000 | solved 0.0%
saved -> model
```
- First hit: tiny-gpt2 has NO chat_template -> apply_chat_template crashed. Fixed by adding a
  guarded fallback chat_template in main() (inert for Qwen which has its own).
- loss ~0 is CORRECT: all rewards equal -> advantages 0 (blog 2.2 all-same); step-0 policy==ref -> KL 0.
- Both TODOs validated end-to-end (no shape error, finite loss).

---

## CURRENT STEP → Step 6 — Real GRPO training on the A10 (Qwen2.5-0.5B-Instruct)

Model: Qwen/Qwen2.5-0.5B-Instruct (the fixed capstone base). Run in tmux.
Start config: --group 8 --bsz 4 --max-new-tokens 256  (=32 seqs/step). Grow only if headroom.
Watch nvidia-smi; if OOM -> drop --bsz first.

MEMORY LESSON (two OOMs before it fit):
- OOM #1 at --bsz 4 --group 8 --max-new-tokens 256: the killer is the fp32 vocab logits
  [bsz*group, T, ~152k] + log_softmax copy, NOT the 0.5B weights. bsz*group*seq_len*vocab
  is what fills VRAM in LLM RL.
- OOM #2 at --bsz 2 --group 8 --max-new-tokens 256: only ~70 MB short.
- FIT at --bsz 2 --group 8 --max-new-tokens 128 (+ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True).
- Deck says to call gradient_checkpointing_enable() (halves activation mem) but we did NOT
  need it once max-new-tokens dropped to 128. Kept it out for simplicity.
- Infra edit added: rollout now generates in eval() + use_cache=True (fast KV cache), restores train().

Probe (5 steps) OUTPUT — plumbing + signal confirmed:
```
step 0 | loss +0.3986 | reward 0.028 | solved 0.0%
step 4 | loss +0.3860 | reward 0.025 | solved 0.0%
```
reward non-zero (Qwen emits some parsable answers) vs tiny-gpt2's 0.000. Memory fits.

FULL RUN command (in tmux, tee log for the reward curve). 1000 steps because the deck says
the aha moment lands between step 600-1000; 400 would stop in the flat format-learning phase.
Added infra: --save-every (periodic checkpoint, default 200) so a late crash doesn't lose the run.
```bash
tmux new -s train
cd lectures/capstone
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --steps 1000 --group 8 --bsz 2 --max-new-tokens 128 --out model \
    --save-every 200 \
    2>&1 | tee train_log.txt
```
Watch: steps 0-~500 reward rises ~0.03->~0.1 (learning FORMAT, solved ~0% = NORMAL);
steps ~600-1000 solved should lift off 0% (aha moment). loss stays noisy - ignore it.

RUN 1 RESULT (lr 1e-6, 1000 steps, group 8, bsz 2, max-new-tokens 128) — COMPLETED:
- Training: reward 0.03 -> ~0.05-0.19 band; solved on batches 0% -> occasional 6-12% near step ~1000.
- Dev eval vs FLOOR:
    metric        floor(base)   run1(grpo)
    accuracy       0.33%          1.00%   (1->3 / 300)   = 3x over floor (Core pass, modest)
    format_rate    0.00%          0.00%
    easy           0.00%          1.67%
    avg_tokens     20.0           16.7

FAILURE MODE (the key finding, great for the report):
- Model dropped <think> ENTIRELY and emits bare short <answer>expr</answer> (~14-21 tok),
  often with WRONG numbers (e.g. puzzle 5,5,10 -> "5*(5+5)" uses 5,5,5).
- WHY: countdown.reward scores ONLY extract_answer (the <answer> content), gives 0 credit for
  <think>. GRPO found the cheap local optimum: skip reasoning, emit any parsable answer (0.05).
  It never climbed to reason->correct (1.0). "Model optimizes the reward you gave, not meant."
- Ties to blog 2.1 (real recipes add a FORMAT reward) and blog 3 (reasoning emerges only if it
  raises reward). Can't modify countdown.py, so improvement = push harder (lr/steps/group/tokens)
  so correct-answer reward gets discovered.

ARTIFACT NAMING CONVENTION (so runs don't clash):
- Run 1 (this one): model_run1_lr1e6/, train_log_run1_lr1e6.txt, dev_preds_run1_lr1e6.jsonl
- Ablation (next):   model_lr3e6/,      train_log_lr3e6.txt,      dev_preds_lr3e6.jsonl
- FINAL submission:  copy the BEST run's checkpoint -> model/ and its preds -> dev_preds.jsonl

LEADERBOARD:
- No local live server. The real leaderboard is instructor-side: submit via email (see README)
  with model (HF Hub link), dev_preds.jsonl, train_grpo.py, reward_curve.png, report.
- Locally we track our own runs by appending to leaderboard.csv:
    uv run python evaluate.py --test data/dev_public.jsonl --predictions <preds> --name <n> --append leaderboard.csv
  Reference points the instructor seeds: base model (floor) and their reference GRPO run (target).

---

## CURRENT STEP → Step 7 (finish Run 1 set): reward_curve.png + report, THEN ablation

Remaining for a complete Run 1 set:
1. [DONE] Renamed artifacts to run1 convention + built leaderboard.csv:
     name,accuracy,accuracy_hard,format_rate,avg_tokens_correct,n
     base-floor,0.0033,0.0000,0.0000,20.0,300
     grpo-run1-lr1e6,0.0100,0.0000,0.0000,16.7,300   (3x floor)
2. [DONE] reward_curve.png via plot_rewards.py. Run 1 curve: fast rise 0.03->0.05 (format gate),
   plateau ~0.05 to step ~450 (bare-guess local optimum), slow climb to ~0.10 + growing spikes
   after ~450 (first correct answers). Final smoothed ~0.101, STILL RISING at step 1000 -> headroom.
3. [DONE-draft] REPORT.md written (method, setup, floor->run1 table, curve reading, failure mode).
   Ablation section stubbed, to fill after the lr-3e-6 run.

>>> RUN 1 COMPLETE SET LOCKED. Next: IMPROVE via ablation.

## ABLATION RESULT — lr 3e-6, 1500 steps (BIG WIN)
leaderboard.csv:
    base-floor       accuracy 0.33%   hard 0.00%   format 0.00%
    grpo-run1-lr1e6  accuracy 1.00%   hard 0.00%   format 0.00%
    grpo-lr3e6       accuracy 12.00%  hard 1.67%   format 0.00%   <-- 12x floor, 12x run1
- Reward: final smoothed ~0.191 (vs run1 ~0.101); batch-solved up to ~44% near step 1500.
- Overlaid curve shows lr3e6 > run1 at EVERY shared step -> gain is from LR, not the extra 500 steps.
- Still rising at 1500 -> more steps should help.
- KEY INSIGHT: completions are only ~17 tokens even when correct -> max_new_tokens is NOT the
  bottleneck. Do NOT bother raising it to 256 / gradient checkpointing. Levers that matter: LR + steps.
- format_rate still 0 (no <think>) - the reasoning-collapse failure mode persists (reward doesn't pay
  for <think>); accuracy still climbs because bare correct answers score 1.0.
- REPORT.md ablation section filled. Added plot_leaderboard.py (bar chart of leaderboard.csv).
- Artifacts: model_lr3e6/, train_log_lr3e6.txt, dev_preds_lr3e6.jsonl.

## STEP ABLATION — lr 3e-6, 3000 steps (PLATEAU, no gain)
    grpo-lr3e6      (1500 steps) accuracy 12.00%  hard 1.67%
    grpo-lr3e6-3k   (3000 steps) accuracy 10.67%  hard 1.67%   <-- flat-to-down (within ~1.8% noise)
- Reward plateaus right after step 1500: brief spike to ~0.24 then flat ~0.13-0.16 to step 3000.
  Final smoothed 0.159 (LOWER than the 1500 run's 0.191).
- CONCLUSION: recipe saturates ~step 1500. More steps = wasted compute. Best = grpo-lr3e6 (1500).
- To break the plateau, change a DIFFERENT lever: larger --group (lower-variance advantage) or
  higher lr (5e-6). NOT more steps, NOT more tokens.
- Report ablation table updated (Run1 / A=best / B=plateau). Leaderboard.png regenerated (4 runs).

## GROUP ABLATION — group 16 (NO GAIN) + tuning DONE
    grpo-g16-1200  (group 16, 1200 steps)  accuracy 10.33%  hard 0.00%  <- no advantage, same plateau
- group-16 run OOM'd at step 1379 (allocator fragmentation on ref forward). Fix for future:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. But result didn't justify a rerun.
- FINAL VERDICT (3 ablations): only LR moved the needle (1%->12%). Steps & group = no gain.
  Recipe saturates ~12%; binding constraint is reward design (reasoning collapse, format_rate=0),
  NOT hyperparameters. Tuning phase CLOSED.
- SUBMISSION PICK: grpo-lr3e6  (model_lr3e6/ + dev_preds_lr3e6.jsonl)  = 12.00% dev accuracy.
- Report headline results table + synthesis paragraph updated. Next: package submission.

## DENSE REWARD — NEW BEST (12.00% -> 14.67%)
Added dense_reward() in countdown.py + --reward {shaped,dense} flag in train_grpo.py.
Dense = 0.10 + 0.85*exp(-|value-target|/10) for right-numbers/wrong-value (slope, not cliff).
Run: --reward dense --lr 3e-6 --group 8 --bsz 2 --max-new-tokens 128 --steps 1500 -> model_dense/
    grpo-dense  accuracy 14.67% (44/300)  easy 29.17%  medium 7.50%  hard 0.00%  format 0
- +8 puzzles over grpo-lr3e6 (12.00%). Gain in easy/medium (medium doubled); hard -> 0 (small-n).
- Why it worked: near-misses no longer tie inside a group -> non-zero std -> non-zero advantage.
- Leaderboard scorer untouched (is_correct exact); only training signal shaped.
- NEW SUBMISSION PICK: model_dense -> HF repo countdown-qwen2.5-0.5b-grpo-dense (14.67%).
- Report updated (dense section + headline). Model card written. Follow-up email to Rajat planned.

## FORMAT REWARD + ENTROPY — FAILED (9.67%, format still 0)
Added dense_format_reward (dense + 0.10 bonus for non-trivial <think>) + --entropy-coef.
Run: --reward dense_format --entropy-coef 0.01 --lr 3e-6 --group 8 --bsz 2 --steps 1500 -> model_dfmt
    grpo-dfmt  accuracy 9.67% (29/300)  easy 21.67%  medium 2.50%  hard 0.00%  format 0.00%
- WORSE than dense (14.67%) and format_rate STILL 0. Negative result.
- Why: cold-start. Model almost never SAMPLES a <think> block, so the +0.10 bonus is almost never
  triggered -> ~no gradient toward format. Closeness reward (<=0.95) dwarfs the 0.10 bonus, so bare
  near-misses stay optimal. entropy_coef 0.01 added noise, didn't help discover the structure.
- Fixed an OOM on the way: entropy was computed on the ref pass + full sequence (2.4GB); now
  policy-only + completion-slice (need_entropy flag in token_logprobs).
- CONCLUSION: reward-only format induction fails at 0.5B without SFT (which is banned). Keeping
  grpo-dense (14.67%) as the submission. Tuning effectively closed.

---

## CURRENT STEP → Ablation run (lr 3e-6, 1500 steps) to beat Run 1

Rationale: Run 1 reward still rising at step 1000 -> lr 1e-6 too gentle. Change ONE var (lr).
```bash
tmux new -s train2
cd lectures/capstone
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --steps 1500 --group 8 --bsz 2 --max-new-tokens 128 --lr 3e-6 --out model_lr3e6 \
    --save-every 300 \
    2>&1 | tee train_log_lr3e6.txt
```
After it finishes:
- uv run python plot_rewards.py        # overlays run1 vs lr3e6 automatically
- uv run python run_model.py --model ./model_lr3e6 --test data/dev_public.jsonl --out dev_preds_lr3e6.jsonl
- uv run python evaluate.py --test data/dev_public.jsonl --predictions dev_preds_lr3e6.jsonl --name grpo-lr3e6 --append leaderboard.csv
- Fill REPORT.md ablation section; pick BEST run -> copy to model/ + dev_preds.jsonl for submission.

TOOL: plot_rewards.py — overlays reward (or --metric solved) vs step for ALL train_log*.txt on
one graph (faint raw + bold moving avg). Re-run after each new run to compare. Saves reward_curve.png.
    uv run python plot_rewards.py                 # all runs, reward
    uv run python plot_rewards.py --metric solved # solved% instead

(Update this file as we finish each step.)
