"""WOW artifacts for the CMA controller: the agent DODGING, in reality and in
its dream.

Loads the trained tokenizer + world model + tiny controller and renders:
  1. AGENT IN REALITY  — the controller playing real VizDoom take_cover, strafing
     to dodge fireballs (GIF; several clips + the best one).
  2. AGENT IN THE DREAM — the controller acting inside the world model's
     hallucination (GIF): no engine, the Transformer generates each frame.
  3. REAL vs DREAM under the controller's OWN actions, side by side.

The controller reads the SAME reconstructed-frame feature in both (decode(WM
tokens) in the dream; decode(encode(obs)) in reality) — the transfer trick.

Run:  .venv/bin/modal run modal_apps/doom_deliverables_ctrl.py --tag doom
"""
import json
import modal
from common import (app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, dreams_dir,
                    write_result, jsonify)


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 30)
def deliverables(cfg: dict, tag: str = "doom", n_clips: int = 6, max_len: int = 320,
                 hidden_override: int = -1, ctrl_run_id: str = ""):
    import os, io, base64, time, numpy as np, torch, imageio.v2 as imageio
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from controller import Reducer, TinyController

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc = cfg["tokenizer"], cfg["world_model"]
    env0 = make_env(cfg, seed=0); A = env0.num_actions; max_ctx = wc["max_timesteps"]

    tok = Tokenizer(TokenizerConfig(resolution=cfg["resolution"], ch=tc["ch"],
                    num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    num_actions=A, max_timesteps=max_ctx, embed_dim=wc["embed_dim"],
                    num_layers=wc["num_layers"], num_heads=wc["num_heads"])).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))

    cdata = torch.load(ckpt_path(tag, "controller" + ctrl_run_id), map_location=device)
    hidden = cdata.get("hidden", 0) if hidden_override < 0 else hidden_override
    reducer = Reducer(rows=cdata.get("rows", 3), cols=cdata.get("cols", 6), device=device)
    ctrl = TinyController(reducer.fdim, A, hidden=hidden)
    # ensemble (top-K) if the checkpoint used it, else the single best controller
    if cdata.get("use_ensemble") and cdata.get("thetas"):
        ens_thetas = [t.to(device) for t in cdata["thetas"]]
    else:
        ens_thetas = [cdata["theta"].to(device)]

    outdir = f"{dreams_dir()}/doom_ctrl"; os.makedirs(outdir, exist_ok=True)

    def _u8(a): return (np.clip(np.asarray(a), 0, 1) * 255).astype(np.uint8)

    def frame_t(obs):
        return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    @torch.no_grad()
    def recon_frame(obs):
        return tok.decode(tok.encode(frame_t(obs)))     # (1,3,H,W) reconstructed

    @torch.no_grad()
    def act(feat_cur, feat_prev):
        feats = torch.cat([feat_cur, feat_prev], dim=-1)
        logit_sum = sum(ctrl.logits_single(feats, t) for t in ens_thetas)   # ensemble vote
        return int(logit_sum.argmax(-1).item())

    # -- 1. AGENT IN REALITY --------------------------------------------------
    clips, survivals, best = [], [], None
    for c in range(n_clips):
        e = make_env(cfg, seed=6000 + c); obs = e.reset(6000 + c)
        rf = recon_frame(obs); feat_prev = reducer.features(rf)
        frames, acts, done, steps = [obs.copy()], [], False, 0
        while not done and steps < max_len:
            rf = recon_frame(obs); feat_cur = reducer.features(rf)
            a = act(feat_cur, feat_prev); feat_prev = feat_cur
            obs, r, done = e.step(a); frames.append(obs.copy()); acts.append(a); steps += 1
        e.close(); survivals.append(steps)
        imageio.mimsave(f"{outdir}/real_agent_{c}.gif", [_u8(f) for f in frames], duration=0.06, loop=0)
        if best is None or steps > best[0]:
            best = (steps, frames, acts, 6000 + c)

    # -- 2. AGENT IN THE DREAM (controller acts inside the world model) -------
    @torch.no_grad()
    def dream_clip(seed, H=120, temp=1.0):
        obs = make_env(cfg, seed=seed).reset(seed)
        th = tok.encode(frame_t(obs)); ah = torch.zeros(1, 1, dtype=torch.long, device=device)
        tokens_hist = th.unsqueeze(1)
        feat_prev = reducer.features(tok.decode(tokens_hist[:, -1]))
        frames = [_u8(tok.decode(tokens_hist[:, -1])[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())]
        for t in range(H):
            cur = tok.decode(tokens_hist[:, -1]); feat_cur = reducer.features(cur)
            a = act(feat_cur, feat_prev); feat_prev = feat_cur
            ah[:, -1] = a
            nxt, _, done = wm.generate_step_fast(tokens_hist, ah, temperature=temp, sample=True)
            tokens_hist = torch.cat([tokens_hist, nxt.unsqueeze(1)], 1)[:, -max_ctx:]
            ah = torch.cat([ah, torch.zeros(1, 1, dtype=torch.long, device=device)], 1)[:, -max_ctx:]
            frames.append(_u8(tok.decode(tokens_hist[:, -1])[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()))
            if int(done.item()):
                break
        return frames
    for c in range(min(3, n_clips)):
        df = dream_clip(7000 + c)
        imageio.mimsave(f"{outdir}/dream_agent_{c}.gif", df, duration=0.08, loop=0)

    # -- 3. REAL vs DREAM under the controller's own actions (best real clip) --
    _, bframes, bacts, bseed = best
    x0 = tok.encode(frame_t(bframes[0]))
    th = x0.unsqueeze(1); ah = torch.zeros(1, 1, dtype=torch.long, device=device)
    dream_frames = [tok.decode(th[:, -1])[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()]
    with torch.no_grad():
        for a in bacts:
            ah[:, -1] = a
            nxt, _, done = wm.generate_step_fast(th, ah, temperature=1.0, sample=True)
            th = torch.cat([th, nxt.unsqueeze(1)], 1)[:, -max_ctx:]
            ah = torch.cat([ah, torch.zeros(1, 1, dtype=torch.long, device=device)], 1)[:, -max_ctx:]
            dream_frames.append(tok.decode(th[:, -1])[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            if int(done.item()):
                break
    L = min(len(bframes), len(dream_frames))
    sbs = [np.concatenate([_u8(bframes[i]), 255 * np.ones((2, 64, 3), np.uint8),
                           _u8(dream_frames[i])], 0) for i in range(L)]
    imageio.mimsave(f"{outdir}/real_vs_dream_ctrl.gif", sbs, duration=0.08, loop=0)

    def b64gif(fr, dur=0.06):
        buf = io.BytesIO(); imageio.mimsave(buf, [_u8(f) for f in fr], format="gif", duration=dur, loop=0)
        return base64.b64encode(buf.getvalue()).decode()

    VOL.commit()
    fs = cfg.get("frame_skip", 4)
    result = {"tag": tag, "hidden": cdata.get("hidden", 0),
              "real_survival_steps": {"mean": round(float(np.mean(survivals)), 1),
                                      "max": int(np.max(survivals)), "all": survivals},
              "real_survival_tics_mean": round(float(np.mean(survivals)) * fs, 1),
              "best_clip_steps": best[0],
              "real_agent_gif_b64": b64gif(best[1][:max_len]),
              "figures": [f"doom_ctrl/real_agent_{c}.gif" for c in range(n_clips)]
                         + [f"doom_ctrl/dream_agent_{c}.gif" for c in range(min(3, n_clips))]
                         + ["doom_ctrl/real_vs_dream_ctrl.gif"],
              "runtime_s": round(time.time() - t0, 1)}
    write_result(f"deliverables_ctrl_{tag}", result); VOL.commit()
    return jsonify(result)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom", ctrl_run_id: str = ""):
    import yaml, base64, os
    with open(cfg) as f:
        c = yaml.safe_load(f)
    out = deliverables.remote(c, tag, 6, 320, -1, ctrl_run_id)
    print(json.dumps({k: v for k, v in out.items() if k != "real_agent_gif_b64"}, indent=2))
    os.makedirs("results/doom_ctrl", exist_ok=True)
    if out.get("real_agent_gif_b64"):
        with open("results/doom_ctrl/real_agent_best.gif", "wb") as f:
            f.write(base64.b64decode(out["real_agent_gif_b64"]))
        print("  wrote results/doom_ctrl/real_agent_best.gif")
