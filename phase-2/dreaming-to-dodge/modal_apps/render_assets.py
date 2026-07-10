"""Render high-res publication assets: the trained c6 agent dodging in real Doom,
acting inside its dream, and real-vs-dream — plus key screenshots.

The model ALWAYS sees the 64x64 input it was trained on (render at VizDoom's
160x120, downsample to 64); the DISPLAY frames are upscaled 4x with LANCZOS for
crisp figures/GIFs. Ephemeral (`modal run`), so it does not disturb deployed jobs.

Run:  .venv/bin/modal run modal_apps/render_assets.py --tag doom --run-id c6
"""
import json
import modal
from common import app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, dreams_dir, write_result, jsonify


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 25)
def render(cfg: dict, tag: str = "doom", run_id: str = "c6", up: int = 4, n_try: int = 6):
    import os, io, base64, numpy as np, torch, imageio.v2 as imageio
    from PIL import Image
    from envs import make_env, _ensure_display
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from controller import Reducer, TinyController

    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc = cfg["tokenizer"], cfg["world_model"]
    A = make_env(cfg, seed=0).num_actions
    max_ctx = wc["max_timesteps"]
    tok = Tokenizer(TokenizerConfig(resolution=cfg["resolution"], ch=tc["ch"],
                    num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    num_actions=A, max_timesteps=max_ctx, embed_dim=wc["embed_dim"],
                    num_layers=wc["num_layers"], num_heads=wc["num_heads"])).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))
    cdata = torch.load(ckpt_path(tag, "controller" + run_id), map_location=device)
    reducer = Reducer(rows=cdata.get("rows", 3), cols=cdata.get("cols", 6), device=device)
    ctrl = TinyController(reducer.fdim, A, hidden=cdata.get("hidden", 16))
    thetas = ([t.to(device) for t in cdata["thetas"]] if cdata.get("use_ensemble") and cdata.get("thetas")
              else [cdata["theta"].to(device)])

    outdir = f"{dreams_dir()}/assets"; os.makedirs(outdir, exist_ok=True)
    _ensure_display()
    import vizdoom as vzd

    def big(rgb64_or_frame, size=None):
        a = (np.clip(np.asarray(rgb64_or_frame), 0, 1) * 255).astype(np.uint8) if rgb64_or_frame.max() <= 1.0 else np.asarray(rgb64_or_frame).astype(np.uint8)
        im = Image.fromarray(a)
        s = size or (a.shape[1] * up, a.shape[0] * up)
        return np.asarray(im.resize(s, Image.LANCZOS))

    def to64(hwc_uint8):
        h, w = hwc_uint8.shape[:2]
        ys = np.linspace(0, h - 1, 64).astype(int); xs = np.linspace(0, w - 1, 64).astype(int)
        return (hwc_uint8[np.ix_(ys, xs)][:, :, :3].astype(np.float32) / 255.0)

    def frame_t(obs64):
        return torch.tensor(obs64, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    @torch.no_grad()
    def act(feat_cur, feat_prev):
        feats = torch.cat([feat_cur, feat_prev], -1)
        return int(sum(ctrl.logits_single(feats, t) for t in thetas).argmax(-1).item())

    # ---- 1. REAL gameplay of the agent: render at 160x120, pick the best episode
    def real_episode(seed, max_len=320):
        g = vzd.DoomGame(); g.load_config(os.path.join(vzd.scenarios_path, "take_cover.cfg"))
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120); g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_window_visible(False); g.set_render_hud(False); g.set_render_weapon(False)
        g.set_render_crosshair(False); g.set_render_messages(False)
        g.clear_available_buttons(); g.add_available_button(vzd.Button.MOVE_LEFT); g.add_available_button(vzd.Button.MOVE_RIGHT)
        g.set_seed(seed); g.init(); g.new_episode()
        btn = [[0, 0], [1, 0], [0, 1]]
        hires, done, steps = [], False, 0
        s = g.get_state(); hr = np.transpose(np.asarray(s.screen_buffer), (0, 1, 2)) if False else np.asarray(s.screen_buffer)
        if hr.ndim == 3 and hr.shape[0] == 3: hr = np.transpose(hr, (1, 2, 0))
        feat_prev = reducer.features(tok.decode(tok.encode(frame_t(to64(hr)))))
        while not done and steps < max_len:
            s = g.get_state()
            hr = np.asarray(s.screen_buffer)
            if hr.ndim == 3 and hr.shape[0] == 3: hr = np.transpose(hr, (1, 2, 0))
            hires.append(hr[:, :, :3].copy())
            fc = reducer.features(tok.decode(tok.encode(frame_t(to64(hr)))))
            a = act(fc, feat_prev); feat_prev = fc
            g.make_action(btn[a], cfg.get("frame_skip", 4)); done = g.is_episode_finished(); steps += 1
        g.close()
        return hires, steps

    best = None
    survs = []
    for i in range(n_try):
        hr, st = real_episode(6000 + i)
        survs.append(st)
        if best is None or st > best[1]:
            best = (hr, st, 6000 + i)
    hires, best_steps, best_seed = best

    # GIF of the best real episode (upscaled)
    imageio.mimsave(f"{outdir}/agent_real_hires.gif", [big(f) for f in hires], duration=0.07, loop=0)
    # key screenshots: evenly spaced frames of the best episode
    picks = np.linspace(0, len(hires) - 1, 6).astype(int)
    for j, p in enumerate(picks):
        imageio.imwrite(f"{outdir}/screenshot_{j}.png", big(hires[p]))
    # a filmstrip of 8 frames
    strip = np.concatenate([big(hires[p]) for p in np.linspace(0, len(hires) - 1, 8).astype(int)], axis=1)
    imageio.imwrite(f"{outdir}/agent_filmstrip.png", strip)

    # ---- 2. AGENT IN THE DREAM (native 64 upscaled) ------------------------
    @torch.no_grad()
    def dream_clip(seed, H=140):
        obs = make_env(cfg, seed=seed).reset(seed)
        th = tok.encode(frame_t(obs)); ah = torch.zeros(1, 1, dtype=torch.long, device=device)
        tokens_hist = th.unsqueeze(1)
        feat_prev = reducer.features(tok.decode(tokens_hist[:, -1]))
        frs = [tok.decode(tokens_hist[:, -1])[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()]
        for t in range(H):
            cur = tok.decode(tokens_hist[:, -1]); fc = reducer.features(cur)
            a = act(fc, feat_prev); feat_prev = fc; ah[:, -1] = a
            nxt, _, done = wm.generate_step_fast(tokens_hist, ah, temperature=1.0, sample=True)
            tokens_hist = torch.cat([tokens_hist, nxt.unsqueeze(1)], 1)[:, -max_ctx:]
            ah = torch.cat([ah, torch.zeros(1, 1, dtype=torch.long, device=device)], 1)[:, -max_ctx:]
            frs.append(tok.decode(tokens_hist[:, -1])[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            if int(done.item()): break
        return frs
    dfr = dream_clip(7007)
    imageio.mimsave(f"{outdir}/agent_dream_hires.gif", [big(f, (256, 256)) for f in dfr], duration=0.09, loop=0)

    def b64(png_frame, size=None):
        buf = io.BytesIO(); imageio.imwrite(buf, big(png_frame, size), format="png")
        return base64.b64encode(buf.getvalue()).decode()

    VOL.commit()
    out = {"tag": tag, "run_id": run_id, "best_real_steps": best_steps, "best_seed": best_seed,
           "real_survivals": survs, "dream_steps": len(dfr) - 1,
           "hero_screenshot_b64": b64(hires[int(len(hires) * 0.6)]),
           "figures": ["assets/agent_real_hires.gif", "assets/agent_dream_hires.gif",
                       "assets/agent_filmstrip.png"] + [f"assets/screenshot_{j}.png" for j in range(6)]}
    write_result("assets_manifest", out); VOL.commit()
    return jsonify({k: v for k, v in out.items() if k != "hero_screenshot_b64"} | {"hero_len": len(out["hero_screenshot_b64"])})


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom", run_id: str = "c6"):
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    out = render.remote(c, tag, run_id)
    print(json.dumps(out, indent=2))
