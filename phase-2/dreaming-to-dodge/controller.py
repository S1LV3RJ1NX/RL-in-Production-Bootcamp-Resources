"""A TINY controller trained gradient-free (CMA-ES) — the World Models (2018) fix
for the model-exploitation failure.

A big gradient policy (our LSTM actor-critic) is powerful enough to exploit an
imperfect world model instead of learning to dodge — it kept collapsing to a
fixed action the dream wrongly thought was safe.  Ha & Schmidhuber sidestepped
this with a controller of only a few hundred parameters trained by CMA-ES: too
small to overfit the world model, and gradient-free evolution is robust to the
dream's stochasticity.  We reproduce that here on the IRIS token world model.

Feature (frozen, position-preserving): the tokenizer's quantized latent for the
current frame, averaged over rows and projected per-column, so the HORIZONTAL
position of the fireball survives (that's what you dodge on).  The controller
sees [feature_t, feature_{t-1}] so it can read fireball MOTION, and outputs 3
action logits.  ~hundreds of params total.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class Reducer:
    """Reconstructed FRAME -> compact pooled-image feature (B, 3*rows*cols).

    CRITICAL for dream->real transfer: the feature is computed from the DECODER
    OUTPUT (a reconstructed image), not from raw token embeddings.  WM-sampled
    dream tokens and real-encoded tokens have different index statistics, so a
    token-embedding feature does NOT transfer (a controller that dodged in the
    dream collapsed to a fixed action in reality).  But the DECODED images look
    the same in dream and real (fireball recall ~0.95) -- just like the pixel-
    reading oracle, which transfers fine.  So we always feed the controller a
    decoder output: in the dream, decode(WM tokens); in reality, decode(encode(obs)).

    adaptive-avg-pool the 64x64x3 frame to a rows x cols grid (preserves fireball
    COLUMN = where to dodge, and ROW = how imminent), flatten -> 3*rows*cols dims.
    """

    def __init__(self, rows: int = 3, cols: int = 6, device="cpu"):
        self.rows, self.cols = rows, cols
        self.fdim = 3 * rows * cols            # 3*3*6 = 54

    @torch.no_grad()
    def features(self, frame):
        pooled = F.adaptive_avg_pool2d(frame, (self.rows, self.cols))   # (B,3,rows,cols)
        return pooled.reshape(frame.shape[0], -1)                       # (B, 3*rows*cols)


class TinyController:
    """Controller on [feat_t, feat_{t-1}] (2*fdim) -> action logits (num_actions).

    hidden==0 -> a linear map; hidden>0 -> one tanh hidden layer, so the
    controller can express the NONLINEAR "find the threatening column and move
    away from it" logic that a linear map cannot (a linear controller learned to
    react but dodged badly, ~below random).  Still tiny (~1-2k params) so CMA-ES
    optimises it and it cannot overfit the world model.  Parameters live in a flat
    vector; a batched apply lets a whole CMA population act on its own dream
    streams at once (theta_pop: (P, n_params); feats: (N, 2*fdim), cand[i]=i//E)."""

    def __init__(self, fdim: int, num_actions: int, hidden: int = 0):
        self.fdim = fdim
        self.A = num_actions
        self.in_dim = 2 * fdim
        self.hidden = hidden
        if hidden > 0:
            self.n_params = self.in_dim * hidden + hidden + hidden * num_actions + num_actions
        else:
            self.n_params = self.in_dim * num_actions + num_actions

    def _unpack(self, theta):
        """theta (..., n_params) -> layer tensors (batched over leading dims)."""
        pre = theta.shape[:-1]
        if self.hidden > 0:
            H, A, I = self.hidden, self.A, self.in_dim
            i = 0
            W1 = theta[..., i:i + I * H].reshape(*pre, I, H); i += I * H
            b1 = theta[..., i:i + H]; i += H
            W2 = theta[..., i:i + H * A].reshape(*pre, H, A); i += H * A
            b2 = theta[..., i:i + A]
            return W1, b1, W2, b2
        wsize = self.in_dim * self.A
        W = theta[..., :wsize].reshape(*pre, self.in_dim, self.A)
        b = theta[..., wsize:]
        return W, b

    def logits_pop(self, feats, theta_pop, cand):
        """feats (N, in_dim); theta_pop (P, n_params); cand (N,) -> logits (N, A)."""
        if self.hidden > 0:
            W1, b1, W2, b2 = self._unpack(theta_pop)                 # per-candidate
            h = torch.tanh(torch.einsum("ni,nih->nh", feats, W1[cand]) + b1[cand])
            return torch.einsum("nh,nha->na", h, W2[cand]) + b2[cand]
        W, b = self._unpack(theta_pop)
        return torch.einsum("ni,nij->nj", feats, W[cand]) + b[cand]

    def logits_single(self, feats, theta):
        """feats (B, in_dim); theta (n_params,) -> logits (B, A)."""
        if self.hidden > 0:
            W1, b1, W2, b2 = self._unpack(theta)
            h = torch.tanh(feats @ W1 + b1)
            return h @ W2 + b2
        W, b = self._unpack(theta)
        return feats @ W + b
