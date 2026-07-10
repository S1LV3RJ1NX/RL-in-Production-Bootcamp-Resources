"""STAGE-1 (Modal) — tokenizer + world-model SANITY/FIDELITY gate for the
VizDoom take_cover env ("Dreaming to Dodge"), BEFORE spending on the policy.

Same discipline as the Catch/epidemic stage-1 gates ("prove the tokenizer keeps
the small lethal thing, and the world model can dream, before training a policy
in that dream"):

  1. Collect take_cover frames with a mix of random + heuristic-dodger actions
     (so the buffer contains both fast deaths AND long-survival dodging).
  2. Tokenizer DIAGNOSIS — train the discrete autoencoder TWO ways and measure
     whether the rare bright FIREBALL survives quantization:
        (a) fg_weight = 0   (uniform L1 recon — the naive baseline)
        (b) fg_weight = F   (luminance-weighted recon — the dropped-ball fix)
     Report recon-MSE, active codes, codebook usage, and **fireball-recall**
     (fraction of true fireball brightness retained in the reconstruction) —
     the ball-recall / I-recall analogue.
  3. World-model sanity on the better tokenizer: next-token / reward / done acc.
     (done-accuracy is the one that matters here: it predicts *when you die*.)
  4. FIDELITY: roll the REAL env and the DREAM under the SAME dodger actions;
     report per-step frame MSE (does the dream stay coherent Doom?).
  5. Save reconstruction grids + a real-vs-dream filmstrip + sample frames to the
     Volume so we can SEE whether the dream looks like Doom.

Run:  .venv/bin/modal run modal_apps/stage1_doom.py
"""
import json
import modal

from common import app, IMAGE, VOLUMES, VOL, GPU_SMALL, results_dir, jsonify

CFG = {
    "env": "doom", "resolution": 64, "frame_skip": 4, "t_max": 525,
    "living_reward": 1.0, "seed": 0,
    "collect": {"steps": 12000, "oracle_frac": 0.5},
    "tokenizer": {"num_tokens": 64, "vocab_size": 512, "embed_dim": 128, "ch": 64,
                  "beta": 0.25, "ema": True, "revive_dead": True, "fg_weight": 7.0,
                  "steps": 1500, "batch_size": 128, "lr": 3e-4},
    "world_model": {"max_timesteps": 16, "embed_dim": 256, "num_layers": 6,
                    "num_heads": 4, "steps": 2500, "batch_size": 24, "lr": 3e-4},
    "calib": {"seeds": 12, "horizon": 16},
}


def _fireball_mask(frame):
    """(H,W,3)->(H,W) bool: bright, warm (red>g,b) pixels == fireballs/explosions."""
    import numpy as np
    f = np.asarray(frame)
    bright = f.max(2)
    warm = f[:, :, 0] - np.maximum(f[:, :, 1], f[:, :, 2])
    return (bright > 0.55) & (warm > 0.10)


def _fireball_recall(origs, recons):
    """Fraction of true-fireball brightness retained by the reconstruction."""
    import numpy as np
    num = den = 0.0
    for o, r in zip(origs, recons):
        m = _fireball_mask(o)
        if m.sum() == 0:
            continue
        num += float(np.clip(np.asarray(r), 0, 1).max(2)[m].sum())
        den += float(np.clip(np.asarray(o), 0, 1).max(2)[m].sum())
    return num / den if den else 0.0


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 30)
def stage1():
    import os, time, numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from imagination import reward_value_to_class, compare_to_real

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"device": device,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "config": CFG}

    # -- 1. collect take_cover frames (random + heuristic dodger) -------------
    env = make_env(CFG, seed=0)
    ofrac = CFG["collect"]["oracle_frac"]
    frames, actions, rewards, dones = [], [], [], []
    obs = env.reset(0)
    for i in range(CFG["collect"]["steps"]):
        a = env.oracle_action() if np.random.rand() < ofrac else int(np.random.randint(env.num_actions))
        nobs, r, done = env.step(a)
        frames.append(obs); actions.append(a); rewards.append(r); dones.append(int(done))
        obs = env.reset(i + 1) if done else nobs
    frames_np = np.asarray(frames, dtype=np.float32)
    frames_t = torch.tensor(frames_np).permute(0, 3, 1, 2)
    fb_frac = float(np.mean([_fireball_mask(f).sum() > 2 for f in frames_np[:3000]]))
    report["collected"] = len(frames_t)
    report["fireball_frame_frac"] = round(fb_frac, 3)
    report["episodes"] = int(np.sum(dones))
    report["mean_survival_steps"] = round(len(frames_t) / max(1, int(np.sum(dones))), 1)

    # probe = frames that actually contain a fireball (for the recall metric)
    fb_idx = [j for j in range(min(4000, len(frames_np)))
              if _fireball_mask(frames_np[j]).sum() > 2][:256]
    probe = frames_t[fb_idx].to(device)
    probe_np = [frames_np[j] for j in fb_idx]
    report["probe_frames"] = len(fb_idx)

    # -- 2. tokenizer diagnosis: fg_weight 0 vs F -----------------------------
    tc = CFG["tokenizer"]
    F = tc["fg_weight"]

    def train_tokenizer(fg_weight):
        tok = Tokenizer(TokenizerConfig(
            resolution=64, ch=tc["ch"], num_tokens=tc["num_tokens"],
            vocab_size=tc["vocab_size"], embed_dim=tc["embed_dim"],
            beta=tc["beta"], ema=tc["ema"], revive_dead=tc["revive_dead"],
            fg_weight=fg_weight)).to(device)
        opt = torch.optim.Adam(tok.parameters(), lr=tc["lr"])
        tok.train()
        first = last = None
        for s in range(tc["steps"]):
            x = frames_t[torch.randint(0, len(frames_t), (tc["batch_size"],))].to(device)
            out = tok(x); opt.zero_grad(); out["loss"].backward(); opt.step()
            if s == 0: first = float(out["recon_mse"])
            last = float(out["recon_mse"])
        tok.eval()
        with torch.no_grad():
            recon = tok.decode(tok.encode(probe))
        recon_np = [r for r in recon.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()]
        return tok, {
            "recon_mse_first": round(first, 6), "recon_mse_last": round(last, 6),
            "active_codes": int(out["active_codes"]),
            "codebook_usage": round(float(out["codebook_usage"]), 4),
            "fireball_recall": round(_fireball_recall(probe_np, recon_np), 4),
        }, recon_np

    tok0, m0, recon0 = train_tokenizer(0.0)
    tokF, mF, reconF = train_tokenizer(F)
    report["tokenizer"] = {"fg0_uniform": m0, f"fg{int(F)}_luminance": mF,
                           "fg_weight": F, "params": tokF.num_params()}
    best_tok = tokF if mF["fireball_recall"] >= m0["fireball_recall"] else tok0
    report["tokenizer"]["chosen"] = f"fg{int(F)}" if best_tok is tokF else "fg0"

    # -- 3. world-model sanity on the chosen tokenizer ------------------------
    with torch.no_grad():
        toks = []
        for i in range(0, len(frames_t), 512):
            toks.append(best_tok.encode(frames_t[i:i + 512].to(device)).cpu())
        toks = torch.cat(toks)
    wc = CFG["world_model"]; L = wc["max_timesteps"]
    wm = WorldModel(WorldModelConfig(
        num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
        num_actions=env.num_actions, max_timesteps=L,
        embed_dim=wc["embed_dim"], num_layers=wc["num_layers"],
        num_heads=wc["num_heads"])).to(device)
    wopt = torch.optim.AdamW(wm.parameters(), lr=wc["lr"], weight_decay=0.01)
    rew_cls = reward_value_to_class(torch.tensor(rewards)).long()
    acts_t = torch.tensor(actions).long(); dones_t = torch.tensor(dones).long()
    wm.train(); wm_first = wm_last = None
    for s in range(wc["steps"]):
        st = np.random.randint(0, len(frames_t) - L, wc["batch_size"])
        tks = torch.stack([toks[i:i + L] for i in st]).to(device)
        acs = torch.stack([acts_t[i:i + L] for i in st]).to(device)
        rws = torch.stack([rew_cls[i:i + L] for i in st]).to(device)
        dns = torch.stack([dones_t[i:i + L] for i in st]).to(device)
        o = wm.compute_loss(tks, acs, rws, dns)
        wopt.zero_grad(); o["loss"].backward(); wopt.step()
        if s == 0: wm_first = float(o["loss"])
        wm_last = float(o["loss"])
    report["world_model"] = {
        "params": wm.num_params(), "loss_first": round(wm_first, 4),
        "loss_last": round(wm_last, 4), "token_acc": round(float(o["token_acc"]), 4),
        "reward_acc": round(float(o["reward_acc"]), 4),
        "done_acc": round(float(o["done_acc"]), 4)}

    # -- 4. fidelity: real vs dream under the same dodger actions -------------
    wm.eval()
    cs = CFG["calib"]; H = cs["horizon"]
    mses = []
    for sd in range(cs["seeds"]):
        cenv = make_env(CFG, seed=3000 + sd)
        cmp = compare_to_real(cenv, best_tok, wm, device, H, seed=3000 + sd)
        if cmp["mean_mse"] is not None:
            mses.append(cmp["mean_mse"])
        cenv.close()
    report["fidelity"] = {"mean_frame_mse": round(float(np.mean(mses)), 5),
                          "n_seeds": len(mses)}

    # -- 5. save recon grids + real-vs-dream filmstrip + samples --------------
    import imageio.v2 as imageio
    outdir = f"{results_dir()}/doom_stage1"
    os.makedirs(outdir, exist_ok=True)

    def _u8(a): return (np.clip(np.asarray(a), 0, 1) * 255).astype(np.uint8)

    def _grid(origs, recons, n=8):
        rows = []
        for i in range(min(n, len(origs))):
            rows.append(np.concatenate([_u8(origs[i]), 255 * np.ones((64, 2, 3), np.uint8),
                                        _u8(recons[i])], axis=1))
        return np.concatenate(rows, axis=0)

    imageio.imwrite(f"{outdir}/recon_fg0.png", _grid(probe_np, recon0))
    imageio.imwrite(f"{outdir}/recon_fg{int(F)}.png", _grid(probe_np, reconF))

    cenv = make_env(CFG, seed=2024)
    cmp = compare_to_real(cenv, best_tok, wm, device, H, seed=2024)
    strip = []
    for real, dream in zip(cmp["real"], cmp["dream"]):
        col = np.concatenate([_u8(real), 255 * np.ones((2, 64, 3), np.uint8), _u8(dream)], axis=0)
        strip.append(col); strip.append(255 * np.ones((col.shape[0], 2, 3), np.uint8))
    imageio.imwrite(f"{outdir}/real_vs_dream.png", np.concatenate(strip, axis=1))
    imageio.mimsave(f"{outdir}/real_vs_dream.gif",
                    [np.concatenate([_u8(r), 255 * np.ones((2, 64, 3), np.uint8), _u8(d)], axis=0)
                     for r, d in zip(cmp["real"], cmp["dream"])], duration=0.2, loop=0)
    cenv.close()

    import base64, io
    b64 = []
    for f in probe_np[:4]:
        buf = io.BytesIO(); imageio.imwrite(buf, _u8(f), format="png")
        b64.append(base64.b64encode(buf.getvalue()).decode())
    report["sample_frames_b64"] = b64
    VOL.commit()
    report["figures"] = [f"doom_stage1/recon_fg0.png", f"doom_stage1/recon_fg{int(F)}.png",
                         "doom_stage1/real_vs_dream.png", "doom_stage1/real_vs_dream.gif"]
    report["runtime_s"] = round(time.time() - t0, 1)
    report["ok"] = True
    return jsonify(report)


@app.local_entrypoint()
def main():
    out = stage1.remote()
    light = {k: v for k, v in out.items() if k != "sample_frames_b64"}
    print(json.dumps(light, indent=2))
    tk = out["tokenizer"]
    F = int(out["config"]["tokenizer"]["fg_weight"])
    print("\n=== DOOM STAGE 1", "OK" if out.get("ok") else "FAILED",
          f"in {out.get('runtime_s')}s on {out.get('gpu')} ===")
    print(f"  fireball frames in buffer: {out.get('fireball_frame_frac')}  "
          f"probe={out.get('probe_frames')}  mean_survival={out.get('mean_survival_steps')} steps")
    print(f"  fireball-recall  fg0={tk['fg0_uniform']['fireball_recall']}  "
          f"fg{F}={tk[f'fg{F}_luminance']['fireball_recall']}  (chosen={tk['chosen']})")
    print(f"  recon MSE        fg0={tk['fg0_uniform']['recon_mse_last']}  "
          f"fg{F}={tk[f'fg{F}_luminance']['recon_mse_last']}")
    print(f"  world model      token_acc={out['world_model']['token_acc']}  "
          f"reward_acc={out['world_model']['reward_acc']}  done_acc={out['world_model']['done_acc']}")
    print(f"  fidelity         mean_frame_mse={out['fidelity']['mean_frame_mse']}")
    import base64, os
    os.makedirs("results/doom_stage1", exist_ok=True)
    for i, b in enumerate(out.get("sample_frames_b64", [])):
        with open(f"results/doom_stage1/sample_{i}.png", "wb") as f:
            f.write(base64.b64decode(b))
    print("  wrote results/doom_stage1/sample_*.png")
