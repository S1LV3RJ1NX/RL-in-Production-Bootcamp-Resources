"""Decisive test: does the DREAM rank reactive dodging above fixed actions?

Both the LSTM policy and the CMA controller collapsed to "always-left". Either
(a) the world model's dream genuinely rewards always-left most (a WM-fidelity
wall — no policy can learn to dodge from it), or (b) the dream rewards the
reactive oracle more but our policies failed to find it (a controller/feature
problem worth more work).

We settle it by rolling the dream under fixed actions AND under the reactive
oracle policy (computed on the DECODED dream frames), and comparing mean dream
survival + how the SAME actions fare in the REAL env.  If oracle >> always-left
in the dream, (b); if not, (a).

Run:  .venv/bin/modal run modal_apps/doom_dream_rank.py --tag doom
"""
import json
import modal
from common import (app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, load_buffer,
                    data_path, jsonify)


def _oracle_on_frame(f, size):
    """Replicate DoomTakeCover.oracle_action on an arbitrary (H,W,3) frame."""
    import numpy as np
    band = f[size // 3: 7 * size // 8, :, :]
    bright = band.max(axis=2)
    warm = np.clip(band[:, :, 0] - 0.4 * band[:, :, 2], 0, 1)
    threat = (bright * warm).sum(axis=0)
    if float(threat.max()) < 1e-3:
        return 0
    center = size // 2
    col = int(np.argmax(threat))
    if col < center - 2:
        return 2
    if col > center + 2:
        return 1
    return 2 if threat[:center].sum() >= threat[center:].sum() else 1


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 15)
def rank(cfg: dict, tag: str = "doom", n: int = 96, horizon: int = 80):
    import numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig

    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc = cfg["tokenizer"], cfg["world_model"]
    size = cfg["resolution"]; A = make_env(cfg, seed=0).num_actions; max_ctx = wc["max_timesteps"]
    tok = Tokenizer(TokenizerConfig(resolution=size, ch=tc["ch"], num_tokens=tc["num_tokens"],
                    vocab_size=tc["vocab_size"], embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    num_actions=A, max_timesteps=max_ctx, embed_dim=wc["embed_dim"],
                    num_layers=wc["num_layers"], num_heads=wc["num_heads"])).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))

    frames, _, _, _ = load_buffer(data_path(tag))
    frames = torch.tensor(frames).permute(0, 3, 1, 2)

    @torch.no_grad()
    def dream_survival(policy):
        idx = np.random.randint(0, len(frames), n)
        start = frames[idx].to(device)
        th = tok.encode(start).unsqueeze(1)
        ah = torch.zeros(n, 1, dtype=torch.long, device=device)
        alive = torch.ones(n, device=device); surv = torch.zeros(n, device=device)
        for t in range(horizon):
            if policy in (0, 1, 2):
                action = torch.full((n,), policy, dtype=torch.long, device=device)
            else:  # oracle on decoded dream frames
                dec = tok.decode(th[:, -1]).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
                action = torch.tensor([_oracle_on_frame(dec[i], size) for i in range(n)],
                                      device=device)
            ah[:, -1] = action
            nxt, _, done = wm.generate_step_fast(th, ah, temperature=1.0, sample=True)
            surv += alive; alive = alive * (1.0 - done.float())
            th = torch.cat([th, nxt.unsqueeze(1)], 1)[:, -max_ctx:]
            ah = torch.cat([ah, torch.zeros(n, 1, dtype=torch.long, device=device)], 1)[:, -max_ctx:]
            if alive.sum() == 0:
                break
        return float(surv.mean())

    dream = {"always_noop": dream_survival(0), "always_left": dream_survival(1),
             "always_right": dream_survival(2), "oracle_reactive": dream_survival("oracle")}

    # real-env survival of the same policies (ground truth)
    def real_survival(kind, m=20):
        survs = []
        for ep in range(m):
            e = make_env(cfg, seed=90000 + ep); obs = e.reset(90000 + ep); done, c = False, 0
            while not done and c < cfg.get("t_max", 525):
                a = kind if isinstance(kind, int) else e.oracle_action()
                obs, r, done = e.step(a); c += 1
            survs.append(c); e.close()
        return round(float(np.mean(survs)), 1)
    real = {"always_noop": real_survival(0), "always_left": real_survival(1),
            "always_right": real_survival(2), "oracle_reactive": real_survival("oracle")}

    verdict = ("dream rewards reactivity (controller/feature problem)"
               if dream["oracle_reactive"] > max(dream["always_left"], dream["always_right"]) + 3
               else "dream MIS-RANKS: fixed action >= reactive (world-model fidelity wall)")
    out = {"tag": tag, "horizon": horizon, "n": n,
           "dream_survival": {k: round(v, 1) for k, v in dream.items()},
           "real_survival": real, "verdict": verdict}
    return jsonify(out)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom"):
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    out = rank.remote(c, tag)
    print(json.dumps(out, indent=2))
