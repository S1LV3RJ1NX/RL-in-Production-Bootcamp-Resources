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

from countdown import Puzzle, chat_messages, reward as countdown_reward, is_correct


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
@torch.no_grad()
def rollout(model, tok, puzzles, group, max_new_tokens, device):
    prompts, metas = [], []
    for p in puzzles:
        text = tok.apply_chat_template(chat_messages(p), tokenize=False, add_generation_prompt=True)
        for _ in range(group):
            prompts.append(text)
            metas.append(p)
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
    prompt_len = enc["input_ids"].shape[1]
    completions = tok.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
    rewards = torch.tensor([countdown_reward(c, p) for c, p in zip(completions, metas)],
                           dtype=torch.float32)
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
    raise NotImplementedError("📝 TODO 1: group-relative advantage")


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
    raise NotImplementedError("📝 TODO 2: GRPO clipped surrogate + KL loss")


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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

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

    for step in range(args.steps):
        puzzles = rng.sample(pool, args.bsz)
        seqs, attn, prompt_len, rewards, solved, _ = rollout(
            policy, tok, puzzles, args.group, args.max_new_tokens, device)
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

    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
