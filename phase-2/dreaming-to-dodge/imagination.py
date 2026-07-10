"""The imagination rollout loop — where the policy learns entirely in dreams.

This is IRIS Algorithm 1's `update_behavior` inner loop (and the dream-rendering
routine).  Given a real starting frame, we:

    1. encode it to tokens with the tokenizer         z_0 = E(x_0)
    2. decode back to the reconstructed frame          x_hat_0 = D(z_0)
    3. the policy picks an action                       a_0 ~ pi(a | x_hat_0)
    4. the world model imagines the next frame's tokens + reward + done
                                                        z_1, r_0, d_0 ~ G(...)
    5. decode x_hat_1 = D(z_1), and repeat for H steps.

The policy NEVER sees the real environment here — every frame after the first is
hallucinated by the Transformer.  The rewards and dones used to train the
actor-critic are the world model's *predictions*, exactly as in IRIS.

`imagine_rollout` returns the tensors the actor-critic loss needs.
`render_dream` returns a filmstrip (list of decoded frames) for the demo, and
`compare_to_real` rolls the real env and the dream under the SAME action
sequence to measure world-model fidelity.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# reward class index {0,1,2} <-> reward value {-1, 0, +1}
REWARD_VALUES = torch.tensor([-1.0, 0.0, 1.0])


def reward_class_to_value(cls, device):
    return REWARD_VALUES.to(device)[cls]


def reward_value_to_class(r):
    """+1 -> 2, 0 -> 1, -1/anything<0 -> 0."""
    return (torch.sign(r).long() + 1)


# ---------------------------------------------------------------------------
# Training rollout: imagine H steps from a batch of real starting frames
# ---------------------------------------------------------------------------
def imagine_rollout(tokenizer, world_model, actor_critic, start_frames,
                    horizon, device, temperature=1.0):
    """
    start_frames : (B, C, H, W)  real frames sampled from the replay buffer
    Returns a dict of stacked tensors for actor_critic_loss:
      logits (H,B,A), actions (H,B), values (H+1,B), rewards (H,B), dones (H,B)
    plus the list of decoded frames (for optional logging).
    """
    B = start_frames.shape[0]
    K = tokenizer.cfg.num_tokens

    # 1) encode the real start frame -> tokens ; keep a running token/action
    #    history for the world model (last L steps).
    tokens_hist = tokenizer.encode(start_frames).unsqueeze(1)   # (B,1,K)
    actions_hist = torch.zeros(B, 1, dtype=torch.long, device=device)

    ac_state = actor_critic.initial_state(B, device)
    logits_l, actions_l, values_l, rewards_l, dones_l = [], [], [], [], []
    alive = torch.ones(B, device=device)                        # 1 until done

    # decode the first reconstructed frame the policy will see
    cur_frame = tokenizer.decode(tokens_hist[:, -1])            # (B,C,H,W)

    max_ctx = world_model.cfg.max_timesteps
    for t in range(horizon):
        logits, value, ac_state = actor_critic(cur_frame, ac_state)
        dist = torch.distributions.Categorical(logits=logits / max(temperature, 1e-6))
        action = dist.sample()

        # write the chosen action into the history's latest step
        actions_hist[:, -1] = action

        # world model imagines next tokens + reward class + done (KV-cached fast
        # path; verified token-identical to generate_step, ~10x faster at K=64)
        nxt_tokens, rew_cls, done = world_model.generate_step_fast(
            tokens_hist, actions_hist, temperature=temperature, sample=True)
        reward = reward_class_to_value(rew_cls, device)

        logits_l.append(logits)
        actions_l.append(action)
        values_l.append(value)
        rewards_l.append(reward * alive)          # zero out reward after death
        dones_l.append(done.float())

        # advance: append the new step to history, trim to context window
        tokens_hist = torch.cat([tokens_hist, nxt_tokens.unsqueeze(1)], dim=1)
        actions_hist = torch.cat(
            [actions_hist, torch.zeros(B, 1, dtype=torch.long, device=device)], dim=1)
        if tokens_hist.shape[1] > max_ctx:
            tokens_hist = tokens_hist[:, -max_ctx:]
            actions_hist = actions_hist[:, -max_ctx:]

        cur_frame = tokenizer.decode(tokens_hist[:, -1])
        alive = alive * (1.0 - done.float())

    # bootstrap value V(x_H)
    with torch.no_grad():
        _, last_value, _ = actor_critic(cur_frame, ac_state)

    values_l.append(last_value)
    return {
        "logits": torch.stack(logits_l),        # (H,B,A)
        "actions": torch.stack(actions_l),      # (H,B)
        "values": torch.stack(values_l),        # (H+1,B)
        "rewards": torch.stack(rewards_l),      # (H,B)
        "dones": torch.stack(dones_l),          # (H,B)
    }


# ---------------------------------------------------------------------------
# Dream rendering (for the demo filmstrips)
# ---------------------------------------------------------------------------
@torch.no_grad()
def render_dream(tokenizer, world_model, actor_critic, start_frame, horizon,
                 device, temperature=1.0, use_policy=True, action_seq=None):
    """Roll one imagined trajectory and return the decoded frames + predictions.

    start_frame : (C,H,W)  single real frame
    If use_policy, the policy chooses actions; else follow `action_seq`.
    Returns dict with frames (list of HxWxC np arrays), actions, rewards, dones.
    """
    x0 = start_frame.unsqueeze(0).to(device)
    tokens_hist = tokenizer.encode(x0).unsqueeze(1)       # (1,1,K)
    actions_hist = torch.zeros(1, 1, dtype=torch.long, device=device)
    ac_state = actor_critic.initial_state(1, device)
    max_ctx = world_model.cfg.max_timesteps

    frames = [_to_np(tokenizer.decode(tokens_hist[:, -1])[0])]
    acts, rews, dns = [], [], []
    cur = tokenizer.decode(tokens_hist[:, -1])
    for t in range(horizon):
        if use_policy:
            action, _, ac_state = actor_critic.act(cur, ac_state, sample=True,
                                                   temperature=temperature)
            a = int(action.item())
        else:
            a = int(action_seq[t]); action = torch.tensor([a], device=device)
        actions_hist[:, -1] = action
        nxt, rew_cls, done = world_model.generate_step_fast(
            tokens_hist, actions_hist, temperature=temperature, sample=True)
        tokens_hist = torch.cat([tokens_hist, nxt.unsqueeze(1)], dim=1)
        actions_hist = torch.cat(
            [actions_hist, torch.zeros(1, 1, dtype=torch.long, device=device)], dim=1)
        if tokens_hist.shape[1] > max_ctx:
            tokens_hist = tokens_hist[:, -max_ctx:]
            actions_hist = actions_hist[:, -max_ctx:]
        cur = tokenizer.decode(tokens_hist[:, -1])
        frames.append(_to_np(cur[0]))
        acts.append(a)
        rews.append(float(REWARD_VALUES[rew_cls.item()]))
        dns.append(int(done.item()))
        if done.item():
            break
    return {"frames": frames, "actions": acts, "rewards": rews, "dones": dns}


@torch.no_grad()
def compare_to_real(env, tokenizer, world_model, device, horizon, seed=0):
    """Roll the REAL env and the DREAM under the SAME actions; measure fidelity.

    Uses the oracle policy in the real env for a deterministic, meaningful
    trajectory, then feeds those exact actions to the world model starting from
    the same first frame.  Returns paired frames + per-step MSE.
    """
    obs = env.reset(seed)
    real_frames = [obs.copy()]
    actions = []
    # roll real env with oracle actions
    for t in range(horizon):
        a = env.oracle_action() if hasattr(env, "oracle_action") else \
            int(np.random.randint(env.num_actions))
        obs, r, done = env.step(a)
        real_frames.append(obs.copy())
        actions.append(a)
        if done:
            break

    # dream from the same first frame following the same actions
    x0 = torch.tensor(real_frames[0], dtype=torch.float32).permute(2, 0, 1)
    ac_dummy = None  # not used; we follow the fixed action_seq
    from types import SimpleNamespace

    class _Fixed:                       # minimal object with .act/.initial_state
        def initial_state(self, b, d): return None
        def act(self, *a, **k): raise RuntimeError
    dream = render_dream(tokenizer, world_model, _Fixed(), x0, len(actions),
                         device, use_policy=False, action_seq=actions)
    real = [f for f in real_frames[:len(dream["frames"])]]
    mses = [float(np.mean((np.array(r) - np.array(d)) ** 2))
            for r, d in zip(real, dream["frames"])]
    return {"real": real, "dream": dream["frames"], "mse_per_step": mses,
            "actions": actions, "mean_mse": float(np.mean(mses)) if mses else None}


def _to_np(t):
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
