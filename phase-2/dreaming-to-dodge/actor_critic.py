"""IRIS component 3/3 — the actor-critic policy, trained ONLY in imagination.

The policy never touches the real environment during learning.  It observes
*reconstructed* frames x_hat = D(z) produced by the world model and outputs an
action distribution (actor) and a state-value estimate (critic).  IRIS shares
the actor and critic trunk except for the last layer (App. A.3): a small
convolutional block followed by an LSTM cell.

We keep that architecture: conv stem -> LSTM -> {policy head, value head}.
The LSTM gives the policy memory across an imagined trajectory; before each
imagined rollout we reset the hidden state (a lightweight stand-in for IRIS's
"burn-in", which matters more in long partially-observed Atari games than in
Catch).

Learning objectives (IRIS App. B, following DreamerV2):
  * lambda-return targets (eq. 4)   balance bias/variance for the value target
  * value loss (eq. 5)              MSE to sg[lambda-return]
  * REINFORCE actor loss (eq. 6)    with the value as a baseline + entropy bonus
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActorCriticConfig:
    in_channels: int = 3
    resolution: int = 64
    num_actions: int = 3
    conv_ch: int = 32           # IRIS repeats a conv block; we use conv_ch width
    hidden_dim: int = 256       # LSTM hidden size (IRIS: 512)
    gamma: float = 0.995        # discount (IRIS)
    lam: float = 0.95           # lambda for lambda-returns (IRIS)
    entropy_coef: float = 0.01  # eta (IRIS: 1e-3; a bit higher helps tiny envs)
    value_coef: float = 1.0


class ConvStem(nn.Module):
    """IRIS App. A.3: the same block repeated -- 3x3 conv (stride 1, pad 1),
    ReLU, 2x2 max-pool (stride 2) -- applied until the spatial map is small."""

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        n_pool = 0
        r = cfg.resolution
        while r > 4:
            r //= 2
            n_pool += 1
        layers, c = [], cfg.in_channels
        for _ in range(n_pool):
            layers += [nn.Conv2d(c, cfg.conv_ch, 3, stride=1, padding=1),
                       nn.ReLU(), nn.MaxPool2d(2, 2)]
            c = cfg.conv_ch
        self.net = nn.Sequential(*layers)
        self.out_dim = cfg.conv_ch * r * r

    def forward(self, x):
        return self.net(x).flatten(1)


class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.stem = ConvStem(cfg)
        self.lstm = nn.LSTMCell(self.stem.out_dim, cfg.hidden_dim)
        self.policy_head = nn.Linear(cfg.hidden_dim, cfg.num_actions)
        self.value_head = nn.Linear(cfg.hidden_dim, 1)

    def initial_state(self, batch, device):
        h = torch.zeros(batch, self.cfg.hidden_dim, device=device)
        c = torch.zeros(batch, self.cfg.hidden_dim, device=device)
        return h, c

    def forward(self, frame, state):
        """One step. frame: (B, C, H, W); state: (h, c). Returns logits, value,
        new_state."""
        feat = self.stem(frame)
        h, c = self.lstm(feat, state)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value, (h, c)

    @torch.no_grad()
    def act(self, frame, state, sample=True, temperature=1.0):
        logits, value, state = self.forward(frame, state)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        if sample:
            action = torch.multinomial(probs, 1).squeeze(-1)
        else:
            action = probs.argmax(-1)
        return action, value, state

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# lambda-returns + losses over an IMAGINED trajectory
# ---------------------------------------------------------------------------
def lambda_returns(rewards, values, dones, gamma, lam):
    """IRIS eq. 4 (Dreamer lambda-return), computed backwards.

    rewards : (T, B)   imagined rewards r_0..r_{T-1}
    values  : (T+1, B) value estimates V(x_0)..V(x_T)
    dones   : (T, B)   predicted done flags in {0,1}
    Returns Lambda_t of shape (T, B).
    """
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    nxt = values[-1]
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        boot = (1 - lam) * values[t + 1] + lam * nxt
        nxt = rewards[t] + gamma * nonterminal * boot
        returns[t] = nxt
    return returns


def actor_critic_loss(logits, actions, values, rewards, dones, cfg: ActorCriticConfig,
                      entropy_coef: float | None = None):
    """Compute the combined actor+critic loss over an imagined batch.

    logits  : (T, B, A)
    actions : (T, B)
    values  : (T+1, B)   (includes bootstrap V(x_T))
    rewards : (T, B)
    dones   : (T, B)
    entropy_coef : optional per-update override (used to anneal exploration).
    """
    eta = cfg.entropy_coef if entropy_coef is None else entropy_coef
    returns = lambda_returns(rewards, values, dones, cfg.gamma, cfg.lam)  # (T,B)
    v_pred = values[:-1]                                                  # (T,B)

    # critic: MSE to stop-grad lambda-return (eq. 5)
    value_loss = F.mse_loss(v_pred, returns.detach())

    # actor: REINFORCE with value baseline + entropy (eq. 6)
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(actions)                                        # (T,B)
    advantage = (returns - v_pred).detach()
    policy_loss = -(logp * advantage).mean()
    entropy = dist.entropy().mean()

    loss = policy_loss + cfg.value_coef * value_loss - eta * entropy
    return {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss,
            "entropy": entropy, "entropy_coef": eta, "return_mean": returns.mean(),
            "value_mean": v_pred.mean()}
