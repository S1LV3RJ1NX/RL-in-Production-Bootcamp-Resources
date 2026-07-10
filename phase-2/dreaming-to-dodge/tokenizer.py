"""IRIS component 1/3 — the discrete autoencoder (VQ-VAE-style tokenizer).

The world model can only "speak" in a small vocabulary of discrete image
tokens. This file learns that vocabulary from scratch:

    frame  x  in  R^{C x H x W}
      --Encoder (CNN)-->  y in R^{K x d}     (K spatial cells, each a d-vector)
      --Quantize-------->  z in {1..N}^K     (K discrete token indices)
      --Decoder (CNN)-->  x_hat in R^{C x H x W}  (reconstruction)

This is exactly the map IRIS (Micheli et al., 2023, Sec. 2.1) uses to turn a
64x64 frame into K=16 tokens from a codebook of size N=512.  We keep the same
recipe -- a convolutional encoder, a learned codebook, nearest-neighbour
quantization with a straight-through estimator, and a convolutional decoder --
but expose every dimension as a config so the same code trains on a tiny 64x64
custom env cheaply.

Loss (IRIS eq. in Appendix A.1), an equally weighted sum of:
  * L1 reconstruction        ||x - D(z)||_1
  * codebook loss            ||sg[E(x)] - e||^2      (moves codes toward encoder)
  * commitment loss   beta * ||E(x) - sg[e]||^2      (moves encoder toward codes)
where sg[.] is stop-gradient.  We drop IRIS's heavy VGG perceptual term (it
needs a pretrained ImageNet net and buys little on small synthetic frames) and
say so honestly in the report -- everything else is faithful.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TokenizerConfig:
    in_channels: int = 3        # C  (RGB frames)
    resolution: int = 64        # H = W  (square frames)
    ch: int = 64                # base conv width (IRIS: 64)
    ch_mults: tuple = (1, 1, 2, 2)  # width multiplier per downsampling stage
    num_tokens: int = 16        # K  tokens per frame (must be a perfect square)
    vocab_size: int = 512       # N  codebook entries (IRIS: 512)
    embed_dim: int = 128        # d  code vector dim (IRIS: 512; smaller = cheaper)
    beta: float = 0.25          # commitment weight (Van den Oord et al., 2017)
    # foreground-weighted reconstruction: upweight bright (ball/paddle) pixels so
    # the tiny moving ball is worth reconstructing under a background-dominated
    # objective. 0 => uniform L1 (the base project's setting, which drops the
    # ball); >0 => a cheap stand-in for IRIS's perceptual reconstruction term.
    fg_weight: float = 0.0
    # warmth-weighted reconstruction: upweight ORANGE/RED pixels (R >> G,B), i.e.
    # the fireballs/explosions, which luminance weighting alone underserves (lit
    # walls are also bright). Targets the small lethal thing directly. 0 => off.
    warm_weight: float = 0.0
    # --- codebook-collapse controls (the fix; see paper §Method) --------------
    ema: bool = True            # EMA codebook updates (VQ-VAE-2) instead of the
                                # codebook-loss term -- the primary collapse fix
    ema_decay: float = 0.99     # EMA decay for cluster stats
    ema_eps: float = 1e-5       # Laplace smoothing on cluster sizes
    revive_dead: bool = True    # random-restart dead codes toward live encodings
    revive_every: int = 200     # cadence (in training steps) of dead-code revival
    dead_threshold: float = 1.0 # EMA cluster size below which a code is "dead"

    @property
    def tokens_side(self) -> int:
        """K tokens are laid out on a sqrt(K) x sqrt(K) spatial grid."""
        s = int(round(self.num_tokens ** 0.5))
        assert s * s == self.num_tokens, "num_tokens must be a perfect square"
        return s


# ---------------------------------------------------------------------------
# Conv building blocks (small residual blocks, GroupNorm + SiLU)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    """CNN: (C, H, W) -> (embed_dim, sqrt K, sqrt K).

    We downsample from `resolution` to `tokens_side` by halving with strided
    convs.  For 64x64 -> 4x4 (K=16) that is 4 downsampling stages.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        n_down = _log2_ratio(cfg.resolution, cfg.tokens_side)
        assert len(cfg.ch_mults) >= n_down, (
            f"need >= {n_down} ch_mults to go {cfg.resolution}->{cfg.tokens_side}")
        layers = [nn.Conv2d(cfg.in_channels, cfg.ch, 3, padding=1)]
        c = cfg.ch
        for i in range(n_down):
            c_out = cfg.ch * cfg.ch_mults[i]
            layers += [ResBlock(c, c_out), nn.Conv2d(c_out, c_out, 4, stride=2, padding=1)]
            c = c_out
        layers += [ResBlock(c, c), nn.GroupNorm(min(8, c), c), nn.SiLU(),
                   nn.Conv2d(c, cfg.embed_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """CNN: (embed_dim, sqrt K, sqrt K) -> (C, H, W). Mirror of the encoder."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        n_up = _log2_ratio(cfg.resolution, cfg.tokens_side)
        c = cfg.ch * cfg.ch_mults[n_up - 1]
        layers = [nn.Conv2d(cfg.embed_dim, c, 3, padding=1), ResBlock(c, c)]
        for i in reversed(range(n_up)):
            c_out = cfg.ch * (cfg.ch_mults[i - 1] if i > 0 else 1)
            layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                       nn.Conv2d(c, c, 3, padding=1), ResBlock(c, c_out)]
            c = c_out
        layers += [nn.GroupNorm(min(8, c), c), nn.SiLU(),
                   nn.Conv2d(c, cfg.in_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return torch.sigmoid(self.net(z))   # frames are in [0, 1]


def _log2_ratio(a: int, b: int) -> int:
    n, r = 0, a
    while r > b:
        r //= 2
        n += 1
    assert r == b, f"{a} is not a power-of-two multiple of {b}"
    return n


# ---------------------------------------------------------------------------
# Vector-quantizer codebook
# ---------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """Nearest-neighbour quantization with a straight-through estimator.

    Given encoder output z_e (B, d, h, w), assign each spatial cell to its
    closest codebook vector, returning the quantized tensor z_q, the integer
    token indices, and the VQ loss.

    Two selectable regimes:
      * ``ema=False`` -- the plain VQ-VAE of Van den Oord et al. (2017): the
        codebook is learned through the *codebook-loss* term ``||sg[z_e]-e||^2``.
        On a scene that is ~99% background this reliably **collapses** onto a
        handful of codes (the failure the base project ships).
      * ``ema=True``  -- the VQ-VAE-2 fix: the codebook is updated by an
        exponential moving average of the encoder vectors assigned to each code
        (no codebook-loss gradient), and any code whose EMA cluster size falls
        below ``dead_threshold`` is periodically **re-initialised** ("dead-code
        revival") toward a random live encoder vector. Together these keep the
        codebook diverse, so the small moving ball survives quantization.

    In both regimes the encoder is pulled toward the code it chose by the
    commitment loss ``beta * ||z_e - sg[e]||^2`` and gradients flow to the
    decoder through the straight-through estimator.
    """

    def __init__(self, vocab_size: int, embed_dim: int, beta: float,
                 ema: bool = True, ema_decay: float = 0.99, ema_eps: float = 1e-5,
                 revive_dead: bool = True, revive_every: int = 200,
                 dead_threshold: float = 1.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.beta = beta
        self.ema = ema
        self.decay = ema_decay
        self.eps = ema_eps
        self.revive_dead = revive_dead
        self.revive_every = revive_every
        self.dead_threshold = dead_threshold
        # EMA statistics (buffers so they persist in the checkpoint / to device).
        # With ema=False these stay unused; the embedding is trained by the loss.
        if ema:
            self.embedding.weight.requires_grad_(False)
        self.register_buffer("cluster_size", torch.zeros(vocab_size))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(self, z_e):
        b, d, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)             # (B*h*w, d)
        # squared euclidean distance to every code
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        idx = dist.argmin(1)                                      # (B*h*w,)
        z_q = self.embedding(idx).view(b, h, w, d).permute(0, 3, 1, 2)

        if self.ema:
            # codebook is not learned by gradient -> only the commitment term
            vq_loss = self.beta * F.mse_loss(z_q.detach(), z_e)
            if self.training:
                self._ema_update(flat, idx)
        else:
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commit_loss = F.mse_loss(z_q.detach(), z_e)
            vq_loss = codebook_loss + self.beta * commit_loss

        z_q = z_e + (z_q - z_e).detach()                          # straight-through
        tokens = idx.view(b, h * w)                               # (B, K)
        # fraction of the codebook actually used this batch (health signal)
        usage = idx.unique().numel() / self.embedding.num_embeddings
        return z_q, tokens, vq_loss, usage

    @torch.no_grad()
    def _ema_update(self, flat, idx):
        """VQ-VAE-2 EMA codebook update + periodic dead-code revival."""
        n_codes = self.embedding.num_embeddings
        onehot = F.one_hot(idx, n_codes).type(flat.dtype)         # (M, N)
        batch_count = onehot.sum(0)                               # (N,)
        batch_sum = onehot.t() @ flat                             # (N, d)
        # EMA of per-code counts and vector sums
        self.cluster_size.mul_(self.decay).add_(batch_count, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(batch_sum, alpha=1 - self.decay)
        # Laplace-smoothed normalisation -> new code vectors
        n = self.cluster_size.sum()
        cluster = (self.cluster_size + self.eps) / (n + n_codes * self.eps) * n
        self.embedding.weight.data.copy_(self.embed_avg / cluster.unsqueeze(1))

        self._step += 1
        if self.revive_dead and (self._step % self.revive_every == 0):
            dead = self.cluster_size < self.dead_threshold        # (N,)
            n_dead = int(dead.sum())
            if 0 < n_dead <= flat.shape[0]:
                # reseed dead codes with random encoder vectors from this batch
                pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
                seed = flat[pick]
                self.embedding.weight.data[dead] = seed
                self.embed_avg[dead] = seed
                self.cluster_size[dead] = 1.0

    @torch.no_grad()
    def num_active(self, threshold: float | None = None) -> int:
        """How many codes are currently 'live' by EMA cluster size."""
        thr = self.dead_threshold if threshold is None else threshold
        return int((self.cluster_size >= thr).sum())

    def embed_tokens(self, tokens, side):
        """(B, K) integer tokens -> (B, d, side, side) quantized feature map."""
        b, k = tokens.shape
        z_q = self.embedding(tokens)                              # (B, K, d)
        return z_q.view(b, side, side, -1).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# The full tokenizer
# ---------------------------------------------------------------------------
class Tokenizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.quantizer = VectorQuantizer(
            cfg.vocab_size, cfg.embed_dim, cfg.beta,
            ema=cfg.ema, ema_decay=cfg.ema_decay, ema_eps=cfg.ema_eps,
            revive_dead=cfg.revive_dead, revive_every=cfg.revive_every,
            dead_threshold=cfg.dead_threshold)
        self.decoder = Decoder(cfg)

    # -- training forward: reconstruct + losses --------------------------------
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, tokens, vq_loss, usage = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        if self.cfg.fg_weight > 0 or self.cfg.warm_weight > 0:
            # per-pixel weight: brightness (fg) + warmth (R above G,B => fireballs)
            lum = x.amax(dim=1, keepdim=True)                 # (B,1,H,W) in [0,1]
            warm = (x[:, :1] - 0.5 * (x[:, 1:2] + x[:, 2:3])).clamp(min=0.0)
            w = 1.0 + self.cfg.fg_weight * lum + self.cfg.warm_weight * warm
            err = (x_hat - x).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
            recon = (w * err).sum() / w.sum()                  # weighted mean L1
        else:
            recon = F.l1_loss(x_hat, x)             # IRIS uses an L1 recon loss
        recon_mse = F.mse_loss(x_hat, x)            # reported separately (true MSE)
        loss = recon + vq_loss
        active = self.quantizer.num_active() if self.cfg.ema else \
            int(round(usage * self.cfg.vocab_size))
        return {"x_hat": x_hat, "tokens": tokens, "loss": loss,
                "recon_loss": recon, "recon_mse": recon_mse, "vq_loss": vq_loss,
                "codebook_usage": usage, "active_codes": active}

    # -- inference helpers (used by the world model & imagination) -------------
    @torch.no_grad()
    def encode(self, x):
        """(B, C, H, W) -> (B, K) integer tokens."""
        _, tokens, _, _ = self.quantizer(self.encoder(x))
        return tokens

    @torch.no_grad()
    def decode(self, tokens):
        """(B, K) integer tokens -> (B, C, H, W) reconstructed frame."""
        z_q = self.quantizer.embed_tokens(tokens, self.cfg.tokens_side)
        return self.decoder(z_q)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
