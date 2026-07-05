"""
train_grpo.py — GRPO from scratch for Countdown.  STARTER (fill the 📝 TODOs).

This is the heart of the capstone and the heart of L6: no critic, a group of sampled
answers per prompt, advantage = how much better than the group mean, a clipped
policy-gradient surrogate with a KL leash to the frozen reference.

  ✅ PROVIDED : model/ref loading, the rollout (sampling a group), reward (countdown.reward),
                per-token log-probs, the training loop, logging, checkpointing.
  📝 TODO 1   : the group-relative advantage          (L6's one idea)
  📝 TODO 2   : the GRPO clipped surrogate + KL loss   (L4 clip + L5 KL, no critic)

Smoke-test the plumbing on CPU with a tiny model:
    python train_grpo.py --model sshleifer/tiny-gpt2 --steps 2 --group 4 --bsz 2 --smoke
Real run (Colab T4 / Modal):
    python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 400 --group 8 --bsz 8
"""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown import Puzzle, chat_messages, reward as countdown_reward, is_correct, dense_reward


# --------------------------------------------------------------------------- #
# ✅ PROVIDED — per-token log-probs of `completion` tokens under a model
# --------------------------------------------------------------------------- #
def token_logprobs(model, input_ids, attn_mask, prompt_len):
    """log p(token_t | token_<t) for every position; caller masks the prompt."""
    out = model(input_ids=input_ids, attention_mask=attn_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(out, dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # mask: only completion tokens (positions >= prompt_len-1 in the shifted frame)
    idx = torch.arange(tok_logp.shape[1], device=tok_logp.device).unsqueeze(0)
    comp_mask = (idx >= (prompt_len - 1)) & (attn_mask[:, 1:] == 1)
    return tok_logp, comp_mask.float()


# --------------------------------------------------------------------------- #
# ✅ PROVIDED — rollout: for each puzzle, sample a GROUP of completions + reward
# --------------------------------------------------------------------------- #
@torch.no_grad()  # just sampling answers here, so no gradients
def rollout(model, tok, puzzles, group, max_new_tokens, device, reward_fn):
    # For each puzzle, ask the model for `group` answers. Total = bsz * group.

    # Repeat each puzzle's prompt `group` times, back to back. Keeping them
    # grouped like [P0,P0,P0,P0, P1,P1,P1,P1] is what lets us later do reshape(-1, group).
    prompts, metas = [], []
    for p in puzzles:
        text = tok.apply_chat_template(chat_messages(p), tokenize=False, add_generation_prompt=True)
        for _ in range(group):
            prompts.append(text)   # same question, asked `group` times
            metas.append(p)        # remember which puzzle this row is for (needed to score it)

    # Turn the prompts into padded token ids. Shapes: [bsz*group, prompt_len].
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)

    # Sampling (not greedy), so the `group` identical prompts give DIFFERENT answers.
    # That variety is what creates a learning signal. Shape: [bsz*group, prompt_len + new].
    # Generate in eval mode with the KV cache on for speed, then restore train mode.
    was_training = model.training
    model.eval()
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id,
                         use_cache=True)
    model.train(was_training)
    prompt_len = enc["input_ids"].shape[1]

    # Keep only the newly generated part and turn it back into text.
    completions = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)

    # Score every answer with the chosen reward (shaped step-fn or dense closeness).
    # Same grouped order as above, so rewards is [bsz*group] -> exactly what TODO 1 wants.
    rewards = torch.tensor([reward_fn(c, p) for c, p in zip(completions, metas)],
                           dtype=torch.float32)

    # Fraction actually correct (exact verifier). For logging only, not training.
    solved = sum(is_correct(c, p) for c, p in zip(completions, metas)) / len(metas)

    return gen, enc["attention_mask"], prompt_len, rewards, solved, metas


# --------------------------------------------------------------------------- #
# 📝 TODO 1 — group-relative advantage
# --------------------------------------------------------------------------- #
def group_advantages(rewards: torch.Tensor, group: int) -> torch.Tensor:
    """
    rewards: shape [num_prompts * group], grouped consecutively per prompt.

    For each group of `group` completions sampled from the SAME puzzle, compute
    how much better (or worse) each answer was compared to the rest of the group.
    That deviation — not an absolute score — is the advantage.

    Hint: this is the single idea that distinguishes GRPO from PPO (no critic needed).
    Review the L6 slides before implementing.
    """
    # "How much better than my group's average was I?" That deviation is the advantage.

    # Put each puzzle's answers on their own row: [bsz*group] -> [bsz, group].
    rewards_reshaped_by_group = rewards.reshape(-1, group)

    # Average + spread of each puzzle's own answers (the group is its own baseline).
    group_mean, std = rewards_reshaped_by_group.mean(dim=1), rewards_reshaped_by_group.std(dim=1)

    # Above the group average -> positive (push up); below -> negative (push down).
    # +1e-8 so an all-equal group gives 0 (no signal) instead of dividing by zero.
    advantages = (rewards_reshaped_by_group - group_mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-8)

    # Back to a flat list, one advantage per answer, same order as the completions.
    return advantages.reshape(-1)

# --------------------------------------------------------------------------- #
# 📝 TODO 2 — the GRPO loss (clipped surrogate + KL to reference)
# --------------------------------------------------------------------------- #
def grpo_loss(logp, logp_old, logp_ref, comp_mask, advantages, clip_eps=0.2, kl_beta=0.04):
    """
    logp, logp_old, logp_ref : [B, T-1] per-token log-probs (policy / behavior / reference)
    comp_mask                : [B, T-1] 1.0 on completion tokens, else 0
    advantages               : [B] sequence-level advantage (broadcast to tokens)

    Two ideas from lecture combine here:
      • From L4 (PPO): the clipped importance-sampling surrogate prevents large updates.
        Compute per-token probability ratios and clip them.
      • From L5 (RLHF / GRPO): a KL penalty to the frozen reference keeps the model
        from drifting too far from its starting point.

    Average the per-token objective over actual completion tokens (use comp_mask).
    Return NEGATIVE (we minimise, so gradient ascent on the objective).
    KL estimator: the k3 form  exp(log_ref − log_policy) − (log_ref − log_policy) − 1
    is standard and numerically stable — you'll find it in the GRPO paper.
    """
    # How much more likely is each token now vs. when we sampled it? (PPO ratio)
    policy_ratio = torch.exp(logp - logp_old)
    # Don't trust a huge jump: cap the ratio to a small trust band around 1.
    policy_ratio_clipped = torch.clamp(policy_ratio, 1 - clip_eps, 1 + clip_eps)

    # One advantage per answer -> give it a token axis so it applies to every token.
    A = advantages.unsqueeze(-1)
    # Take the more pessimistic of clipped/unclipped: rewards good moves, but the
    # clip stops any single step from pushing a token too hard.
    surrogate = torch.min(policy_ratio_clipped * A, policy_ratio * A)

    # Leash: how far each token drifted from the frozen reference model (k3 KL, always >= 0).
    KL = torch.exp(logp_ref - logp) - (logp_ref - logp) - 1

    # Reward beating the group, minus a small fee for drifting from the reference.
    per_token_objective = surrogate - kl_beta * KL

    # Average over REAL completion tokens only (mask hides prompt/padding), then
    # flip sign because the optimizer minimizes but we want to maximize the objective.
    loss = -(per_token_objective * comp_mask).sum() / comp_mask.sum().clamp(min=1)
    return loss


# --------------------------------------------------------------------------- #
# ✅ PROVIDED — training loop
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="model")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--group", type=int, default=8, help="completions sampled per puzzle")
    ap.add_argument("--bsz", type=int, default=8, help="puzzles per step")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--smoke", action="store_true", help="tiny run to test plumbing")
    ap.add_argument("--save-every", type=int, default=200,
                    help="checkpoint every N steps (0 disables intermediate saves)")
    ap.add_argument("--reward", choices=["shaped", "dense"], default="shaped",
                    help="training reward: 'shaped' = original step-function (countdown.reward), "
                         "'dense' = closeness-scaled near-miss (countdown.dense_reward)")
    args = ap.parse_args()

    # Pick the training reward once, up front. Scoring still uses is_correct() regardless.
    reward_fn = dense_reward if args.reward == "dense" else countdown_reward

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # Fallback for base models (e.g. tiny-gpt2 smoke test) that ship NO chat template.
    # Real instruct models like Qwen2.5-Instruct already have one, so this is inert for them.
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )

    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    pool = [Puzzle.from_dict(json.loads(l)) for l in open(args.train) if l.strip()]
    rng = random.Random(0)
    if args.smoke:
        pool, args.steps, args.group, args.bsz = pool[:8], 2, 4, 2

    print(f"config | reward={args.reward} | steps={args.steps} | group={args.group} | "
          f"bsz={args.bsz} | lr={args.lr} | max_new_tokens={args.max_new_tokens} | out={args.out}")

    for step in range(args.steps):
        puzzles = rng.sample(pool, args.bsz)
        seqs, attn, prompt_len, rewards, solved, _ = rollout(
            policy, tok, puzzles, args.group, args.max_new_tokens, device, reward_fn)
        rewards = rewards.to(device)

        advantages = group_advantages(rewards, args.group)            # 📝 TODO 1
        attn_full = (seqs != tok.pad_token_id).long()
        logp, comp_mask = token_logprobs(policy, seqs, attn_full, prompt_len)
        with torch.no_grad():
            logp_old = logp.detach()
            logp_ref, _ = token_logprobs(ref, seqs, attn_full, prompt_len)

        loss = grpo_loss(logp, logp_old, logp_ref, comp_mask, advantages)  # 📝 TODO 2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        print(f"step {step:4d} | loss {loss.item():+.4f} | "
              f"reward {rewards.mean().item():.3f} | solved {solved*100:5.1f}%")

        # Periodic checkpoint so a crash/OOM late in a long run doesn't lose everything,
        # and so we have intermediate checkpoints to evaluate/compare.
        if args.save_every and (step + 1) % args.save_every == 0:
            policy.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            print(f"  checkpoint saved -> {args.out} (step {step + 1})")

    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
