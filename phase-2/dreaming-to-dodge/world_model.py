"""IRIS component 2/3 — the autoregressive Transformer world model.

This is the "imagination engine".  It reads a history of interleaved frame
tokens and actions and predicts, autoregressively, the *next* frame's tokens
plus the reward and the episode-done flag (IRIS, Micheli et al. 2023, Sec. 2.2).

    input sequence per step t :   z_t^1 ... z_t^K  a_t
    (K frame tokens from the tokenizer, then 1 action token)

    the Transformer G models three distributions:
      transition   :  z_{t+1}^k ~ p(. | z_{<=t}, a_{<=t}, z_{t+1}^{<k})   (eq.1)
      reward       :  r_t       ~ p(. | z_{<=t}, a_{<=t})                 (eq.2)
      termination  :  d_t       ~ p(. | z_{<=t}, a_{<=t})                 (eq.3)

We build a small GPT (minGPT-style causal Transformer, exactly as IRIS does)
over a vocabulary that is the union of {N frame tokens} and {A actions}; a
per-position type tells the model whether a slot is a frame token or an action.

Heads:
  * token head   -> logits over N   (cross-entropy)  on frame-token positions
  * reward head  -> logits over 3   {-1, 0, +1}       on action positions
  * done head    -> logits over 2   {continue, done}  on action positions

Reward is treated as a 3-way CLASSIFICATION (sign of the reward). Atari-style
rewards are essentially in {-1, 0, +1}; classification is what IRIS uses when
the reward function is discrete and it makes the sparse signal easy to learn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelConfig:
    num_tokens: int = 16        # K frame tokens per step
    vocab_size: int = 512       # N tokenizer codebook size
    num_actions: int = 6        # A discrete actions
    max_timesteps: int = 20     # L context length in *timesteps*
    embed_dim: int = 256        # D transformer width (IRIS: 256)
    num_layers: int = 6         # M blocks (IRIS: 10; 6 is enough & cheaper)
    num_heads: int = 4          # attention heads (IRIS: 4)
    dropout: float = 0.1
    num_reward_classes: int = 3  # sign(reward) in {-1, 0, +1}

    @property
    def tokens_per_step(self) -> int:
        return self.num_tokens + 1        # K frame tokens + 1 action token

    @property
    def block_size(self) -> int:
        return self.max_timesteps * self.tokens_per_step


# ---------------------------------------------------------------------------
# A single GPT-2-style causal Transformer block
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        assert cfg.embed_dim % cfg.num_heads == 0
        self.h = cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        # PyTorch fused causal attention (fast, memory-efficient)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def forward_cached(self, x, past):
        """Incremental attention for KV-cached autoregressive generation.

        x    : (B, Tnew, C)  embeddings of the NEW positions only
        past : None (prime over the whole history, causal) OR (past_k, past_v)
               each (B, h, Tpast, hd) -- then the new queries attend to ALL keys
               (past + new); we only ever call this with Tnew==1 in that case, so
               no causal mask is needed within the (single) new position.
        Returns (y, present) where present=(k, v) is the extended cache.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
        if past is not None:
            pk, pv = past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                           is_causal=(past is None))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), present


class Block(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim), nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_cached(self, x, past):
        a, present = self.attn.forward_cached(self.ln1(x), past)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


# ---------------------------------------------------------------------------
# The world model
# ---------------------------------------------------------------------------
class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        # separate embedding tables for frame tokens and actions (IRIS App. A.2)
        self.frame_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.action_emb = nn.Embedding(cfg.num_actions, cfg.embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.embed_dim))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)

        self.head_token = nn.Linear(cfg.embed_dim, cfg.vocab_size)   # next frame token
        self.head_reward = nn.Linear(cfg.embed_dim, cfg.num_reward_classes)
        self.head_done = nn.Linear(cfg.embed_dim, 2)

        # precompute, for each position in the sequence, whether it is a frame
        # token (0..K-1 within a step) or the action token (K).
        self.register_buffer(
            "is_action", self._position_types(), persistent=False)
        self.apply(self._init)
        nn.init.normal_(self.pos_emb, std=0.02)

    def _position_types(self):
        tps = self.cfg.tokens_per_step
        pos = torch.arange(self.cfg.block_size) % tps
        return (pos == self.cfg.num_tokens)   # True on action slots

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # -- embed an interleaved (tokens, actions) sequence into vectors ----------
    def _embed(self, tokens, actions):
        """tokens: (B, T, K) frame tokens; actions: (B, T) actions.
        Returns (B, T*(K+1), D) interleaved embeddings + the flat action mask.
        """
        B, T, K = tokens.shape
        tok_e = self.frame_emb(tokens)                     # (B, T, K, D)
        act_e = self.action_emb(actions).unsqueeze(2)      # (B, T, 1, D)
        seq = torch.cat([tok_e, act_e], dim=2)             # (B, T, K+1, D)
        seq = seq.reshape(B, T * (K + 1), -1)
        return seq

    def forward(self, tokens, actions):
        """Training forward over full L-step segments.

        tokens  : (B, T, K)  frame tokens for steps 0..T-1
        actions : (B, T)     action taken at each step
        Returns per-position logits for all three heads.
        """
        B, T, K = tokens.shape
        S = T * (K + 1)
        x = self._embed(tokens, actions) + self.pos_emb[:, :S]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        # token logits at *every* position (we only use frame-token slots)
        token_logits = self.head_token(x)                  # (B, S, N)
        # reward/done are read off the ACTION positions (they summarize the step)
        act_mask = self.is_action[:S]                      # (S,)
        act_hidden = x[:, act_mask]                         # (B, T, D)
        reward_logits = self.head_reward(act_hidden)       # (B, T, 3)
        done_logits = self.head_done(act_hidden)           # (B, T, 2)
        return {"token_logits": token_logits,
                "reward_logits": reward_logits,
                "done_logits": done_logits}

    # -- loss over a sampled segment ------------------------------------------
    def compute_loss(self, tokens, actions, rewards, dones, done_pos_weight=1.0):
        """
        tokens  : (B, T, K)   frame tokens (targets are shifted by one step)
        actions : (B, T)
        rewards : (B, T)  reward CLASS index in {0,1,2} == sign+1
        dones   : (B, T)  in {0,1}
        done_pos_weight : class weight on the rare done=1 (death) class. In Doom,
            death is ~1% of steps (89:1 imbalance); without upweighting, the done
            head collapses to always-"alive" and the dream never ends, so the
            policy gets no signal to survive. Default 1.0 leaves Catch unchanged.
        The transition target for step t is the frame tokens of step t+1, so we
        predict tokens for steps 0..T-2.
        """
        cfg = self.cfg
        out = self.forward(tokens, actions)
        B, T, K = tokens.shape

        # --- transition loss ---
        # For each step t we want the model, having read up to a_t, to predict
        # the K tokens of step t+1. The action slot of step t (last position of
        # the step block) plus the already-revealed tokens of step t+1 drive the
        # autoregressive token prediction. Concretely, the logits that predict
        # z_{t+1}^k live at the position immediately BEFORE z_{t+1}^k in the flat
        # sequence. We therefore align the flat token-logits with a
        # next-frame-token target built by dropping the first step.
        token_logits = out["token_logits"]                 # (B, S, N)
        S = T * (K + 1)
        # positions whose *next* element is a frame token of a future step:
        # the action slot of step t predicts z_{t+1}^0, and frame slot k of
        # step t+1 predicts z_{t+1}^{k+1}. Build the target sequence by shifting.
        flat_tokens = self._flat_token_targets(tokens, actions)  # (B, S) with -100 fill
        tr_loss = F.cross_entropy(
            token_logits.reshape(-1, cfg.vocab_size),
            flat_tokens.reshape(-1), ignore_index=-100)

        # --- reward + done loss (only steps that have a next step defined) ---
        rew_loss = F.cross_entropy(
            out["reward_logits"].reshape(-1, cfg.num_reward_classes),
            rewards.reshape(-1))
        done_flat = out["done_logits"].reshape(-1, 2)
        if done_pos_weight and done_pos_weight != 1.0:
            dw = torch.tensor([1.0, float(done_pos_weight)], device=dones.device)
            done_loss = F.cross_entropy(done_flat, dones.reshape(-1), weight=dw)
        else:
            done_loss = F.cross_entropy(done_flat, dones.reshape(-1))

        loss = tr_loss + rew_loss + done_loss
        with torch.no_grad():
            tok_acc = _masked_acc(token_logits, flat_tokens)
            rew_acc = (out["reward_logits"].argmax(-1) == rewards).float().mean()
            done_pred = out["done_logits"].argmax(-1)
            done_acc = (done_pred == dones).float().mean()
            # recall on the rare death class — the metric that actually matters
            pos = dones == 1
            done_recall = ((done_pred == 1) & pos).float().sum() / pos.float().sum().clamp(min=1)
        return {"loss": loss, "transition_loss": tr_loss, "reward_loss": rew_loss,
                "done_loss": done_loss, "token_acc": tok_acc,
                "reward_acc": rew_acc, "done_acc": done_acc, "done_recall": done_recall}

    def _flat_token_targets(self, tokens, actions):
        """Build the flat next-token target aligned to token_logits positions.

        Flat layout per step t: [z_t^0..z_t^{K-1}, a_t]. The logit at position p
        predicts the element at p+1. We only supervise predictions of FRAME
        tokens that belong to a *future* step (the transition task). So:
          * position = action slot of step t  -> target z_{t+1}^0
          * position = frame slot k of step t (k<K-1) -> would predict z_t^{k+1}
            of the SAME step; that is teacher-forced reconstruction within a
            frame, which we DO supervise (it teaches token-order coherence).
          * position = frame slot K-1 of step t -> predicts a_t (an action);
            not a frame token -> ignore (-100).
        Net: every position predicts the next element if that next element is a
        frame token, else -100.
        """
        B, T, K = tokens.shape
        tps = K + 1
        # flat element ids: frame tokens keep their value, action slots = -100
        flat = torch.full((B, T * tps), -100, dtype=torch.long, device=tokens.device)
        for k in range(K):
            flat[:, k::tps] = tokens[:, :, k]
        # target at position p = element at p+1 (shift left by one)
        target = torch.full_like(flat, -100)
        target[:, :-1] = flat[:, 1:]
        return target

    # -- autoregressive imagination step (used by imagination.py) --------------
    @torch.no_grad()
    def generate_step(self, tokens, actions, temperature=1.0, sample=True):
        """Given history tokens (B,T,K) and actions (B,T) predict the NEXT
        frame's K tokens plus reward class and done flag.

        Returns next_tokens (B, K), reward_class (B,), done (B,).
        """
        cfg = self.cfg
        device = tokens.device
        # We append ONE new step below, so the history must be <= L-1 steps to
        # stay within block_size = L*(K+1).  Trim to the most recent L-1 steps.
        if tokens.shape[1] > cfg.max_timesteps - 1:
            tokens = tokens[:, -(cfg.max_timesteps - 1):]
            actions = actions[:, -(cfg.max_timesteps - 1):]
        B, T, K = tokens.shape

        # 1) reward + done for the *current* last step come from its action slot
        out = self.forward(tokens, actions)
        reward_class = out["reward_logits"][:, -1].argmax(-1)
        done = out["done_logits"][:, -1].argmax(-1)

        # 2) autoregressively roll K next-frame tokens. We append a running
        # buffer and re-embed; sequence stays within block_size because we
        # trimmed the history to L-1 steps above.
        gen = torch.zeros(B, K, dtype=torch.long, device=device)
        # working copies with a placeholder next step appended
        work_tok = torch.cat([tokens, torch.zeros(B, 1, K, dtype=torch.long,
                                                  device=device)], dim=1)
        work_act = torch.cat([actions, actions[:, -1:]], dim=1)  # dummy action
        for k in range(K):
            o = self.forward(work_tok, work_act)
            # position that predicts z_{T}^{k}: it's the slot just before it.
            # In the appended step, frame slot k lives at flat index
            # T*(K+1)+k; its predictor is index (that - 1).
            flat_pred_idx = T * (K + 1) + k - 1
            logits = o["token_logits"][:, flat_pred_idx] / max(temperature, 1e-6)
            if sample:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            else:
                nxt = logits.argmax(-1)
            gen[:, k] = nxt
            work_tok[:, T, k] = nxt
        return gen, reward_class, done

    # -- KV-cached imagination step (drop-in fast path for generate_step) -------
    @torch.no_grad()
    def generate_step_fast(self, tokens, actions, temperature=1.0, sample=True):
        """Identical contract to generate_step, but O(1 full + (K-1) single-token)
        forwards instead of O(K) full forwards, via a per-layer KV cache.  This is
        the difference between hours and minutes of policy-in-imagination at K=64.

        Correctness is pinned to generate_step by an argmax-equivalence check
        (see modal_apps/wm_cache_check); the two must produce identical tokens.
        """
        cfg = self.cfg
        device = tokens.device
        if tokens.shape[1] > cfg.max_timesteps - 1:
            tokens = tokens[:, -(cfg.max_timesteps - 1):]
            actions = actions[:, -(cfg.max_timesteps - 1):]
        B, T, K = tokens.shape
        S = T * (K + 1)

        def tok_from(hidden):
            logits = self.head_token(hidden) / max(temperature, 1e-6)
            if sample:
                return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            return logits.argmax(-1)

        # 1) prime the cache with one full (causal) forward over the history
        x = self._embed(tokens, actions) + self.pos_emb[:, :S]
        caches = []
        for blk in self.blocks:
            x, present = blk.forward_cached(x, None)
            caches.append(present)
        x = self.ln_f(x)
        last = x[:, -1]                                   # hidden at position S-1
        reward_class = self.head_reward(last).argmax(-1)
        done = self.head_done(last).argmax(-1)

        gen = torch.zeros(B, K, dtype=torch.long, device=device)
        cur = tok_from(last)                              # z_{T}^0 (from history)
        gen[:, 0] = cur

        # 2) roll the remaining K-1 frame tokens one at a time against the cache
        pos = S
        for k in range(1, K):
            h = self.frame_emb(cur).unsqueeze(1) + self.pos_emb[:, pos:pos + 1]
            new_caches = []
            for i, blk in enumerate(self.blocks):
                h, present = blk.forward_cached(h, caches[i])
                new_caches.append(present)
            caches = new_caches
            hlast = self.ln_f(h[:, -1])
            cur = tok_from(hlast)
            gen[:, k] = cur
            pos += 1
        return gen, reward_class, done

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def _masked_acc(logits, targets):
    mask = targets != -100
    if mask.sum() == 0:
        return torch.tensor(0.0)
    pred = logits.argmax(-1)
    return (pred[mask] == targets[mask]).float().mean()
