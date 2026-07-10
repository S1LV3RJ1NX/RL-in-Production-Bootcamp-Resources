"""STAGE-1 (Modal) — tokenizer + world-model SANITY/CALIBRATION gate for the
SpatialSIR epidemic-containment env, BEFORE spending anything on the policy.

Mirrors the discipline of the Catch phase-2 run ("verify the tokenizer keeps the
ball before training the policy"), but adapted to the epidemic:

  1. Collect epidemic frames with a random policy (sees S/I/R/V + the cursor).
  2. Tokenizer DIAGNOSIS — train the discrete autoencoder TWO ways and measure
     whether the rare RED infection front survives quantization:
        (a) fg_weight = 0   (uniform L1 recon — the naive baseline)
        (b) fg_weight = 25  (luminance-weighted recon — Catch's fix)
     Report recon-MSE, active codes, codebook usage, and **I-recall** (fraction
     of truly-infected cells that reconstruct as infected) — the ball-recall
     analogue.  Hypothesis: because here the rare class is a class-IMBALANCE
     (few red among many green) rather than a bright-on-black dominance,
     luminance weighting helps only mildly and a rareness-aware loss is the real
     Stage-2 fix.  We MEASURE it rather than assume.
  3. World-model sanity on the better tokenizer: next-token / reward / done acc.
  4. FIDELITY + CALIBRATION (the new claim Catch could not give): roll the REAL
     env and the DREAM under the SAME ring-vaccination actions; report per-step
     frame MSE and the infected-count curve real-vs-dreamed (does the model learn
     the stochastic spread rate?).
  5. Save reconstruction grids + a real-vs-dream filmstrip to the Volume.

Run:  .venv/bin/modal run modal_apps/stage1_epidemic.py
"""
import json
import modal

from common import (app, IMAGE, VOLUMES, VOL, GPU_SMALL, results_dir, jsonify)


# scaled-for-cost but real-signal config (consistent with configs/epidemic.yaml)
CFG = {
    "env": "epidemic", "grid": 16, "resolution": 64, "seed": 0,
    "beta": 0.3, "gamma": 0.2, "init_infected": 2, "t_max": 32,
    "cursor_step": 2, "patch": 3,
    "collect": {"steps": 8000},
    "tokenizer": {"num_tokens": 64, "vocab_size": 512, "embed_dim": 128, "ch": 32,
                  "beta": 0.25, "ema": True, "revive_dead": True,
                  "steps": 1000, "batch_size": 128, "lr": 3e-4},
    "world_model": {"max_timesteps": 16, "embed_dim": 256, "num_layers": 4,
                    "num_heads": 4, "steps": 1500, "batch_size": 16, "lr": 3e-4},
    "calib": {"seeds": 24, "horizon": 20},
}

# state colours must match envs.SpatialSIR.COLORS (S,I,R,V)
_STATE_COLORS = [(0.15, 0.65, 0.25), (0.95, 0.10, 0.10),
                 (0.30, 0.30, 0.30), (0.20, 0.45, 0.95)]
I_IDX = 1


def _classify(frame, grid=16):
    """(H,W,3) float frame -> (grid,grid) state ids by nearest colour at the
    centre pixel of each cell (avoids the 1px yellow cursor outline on edges)."""
    import numpy as np
    cell = frame.shape[0] // grid
    off = cell // 2
    centres = frame[off::cell, off::cell][:grid, :grid]          # (grid,grid,3)
    cols = np.array(_STATE_COLORS)                                # (4,3)
    d = ((centres[:, :, None, :] - cols[None, None, :, :]) ** 2).sum(-1)  # (g,g,4)
    return d.argmin(-1)                                           # (grid,grid)


def _infected_cells(frame, grid=16):
    import numpy as np
    return int((_classify(frame, grid) == I_IDX).sum())


def _i_recall(orig_frames, recon_frames, grid=16):
    """Mean over frames of (#cells infected in BOTH orig & recon) / (#infected
    in orig).  Only frames with >0 infected cells count."""
    import numpy as np
    num, den = 0, 0
    for o, r in zip(orig_frames, recon_frames):
        om = _classify(o, grid) == I_IDX
        if om.sum() == 0:
            continue
        rm = _classify(r, grid) == I_IDX
        num += int((om & rm).sum())
        den += int(om.sum())
    return num / den if den else 0.0


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 25)
def stage1():
    import os, time, math, numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from imagination import reward_value_to_class, compare_to_real

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"device": device,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
              "config": CFG}

    # -- 1. collect epidemic frames with a random policy ----------------------
    env = make_env(CFG, seed=0)
    frames, actions, rewards, dones = [], [], [], []
    obs = env.reset(0)
    for i in range(CFG["collect"]["steps"]):
        a = int(np.random.randint(env.num_actions))
        nobs, r, done = env.step(a)
        frames.append(obs); actions.append(a); rewards.append(r); dones.append(int(done))
        obs = env.reset(i + 1) if done else nobs
    frames_np = np.asarray(frames, dtype=np.float32)             # (N,64,64,3)
    frames_t = torch.tensor(frames_np).permute(0, 3, 1, 2)       # (N,3,64,64)
    inf_frac = float(np.mean([_infected_cells(f) > 0 for f in frames_np[:2000]]))
    report["collected"] = len(frames_t)
    report["infected_frame_frac"] = round(inf_frac, 3)

    # frames that actually contain infected cells (for the I-recall probe)
    inf_idx = [j for j in range(min(3000, len(frames_np)))
               if _infected_cells(frames_np[j]) > 0][:256]
    probe = frames_t[inf_idx].to(device)
    probe_np = [frames_np[j] for j in inf_idx]

    # -- 2. tokenizer diagnosis: fg_weight 0 vs 25 ----------------------------
    tc = CFG["tokenizer"]

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
            recon = tok.decode(tok.encode(probe))                # (P,3,64,64)
        recon_np = [r for r in recon.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()]
        return tok, {
            "recon_mse_first": round(first, 6), "recon_mse_last": round(last, 6),
            "active_codes": int(out["active_codes"]),
            "codebook_usage": round(float(out["codebook_usage"]), 4),
            "I_recall": round(_i_recall(probe_np, recon_np), 4),
        }, recon_np

    tok0, m0, recon0 = train_tokenizer(0.0)
    tok25, m25, recon25 = train_tokenizer(25.0)
    report["tokenizer"] = {"fg0_uniform": m0, "fg25_luminance": m25,
                           "params": tok25.num_params()}

    # pick the tokenizer with the better I-recall for the world model
    best_tok = tok25 if m25["I_recall"] >= m0["I_recall"] else tok0
    report["tokenizer"]["chosen"] = "fg25" if best_tok is tok25 else "fg0"

    # -- 3. world-model sanity on the chosen tokenizer ------------------------
    with torch.no_grad():
        toks = best_tok.encode(frames_t.to(device)).cpu()
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

    # -- 4. fidelity + stochastic-spread calibration --------------------------
    wm.eval()
    cs = CFG["calib"]; H = cs["horizon"]
    mses, real_curves, dream_curves = [], [], []
    for sd in range(cs["seeds"]):
        cenv = make_env(CFG, seed=1000 + sd)
        cmp = compare_to_real(cenv, best_tok, wm, device, H, seed=1000 + sd)
        if cmp["mean_mse"] is not None:
            mses.append(cmp["mean_mse"])
        rc = [_infected_cells(np.asarray(f)) for f in cmp["real"]]
        dc = [_infected_cells(np.asarray(f)) for f in cmp["dream"]]
        real_curves.append(rc); dream_curves.append(dc)

    def _mean_curve(curves):
        m = min(len(c) for c in curves)
        return [round(float(np.mean([c[t] for c in curves])), 2) for t in range(m)]

    real_curve = _mean_curve(real_curves)
    dream_curve = _mean_curve(dream_curves)
    T = min(len(real_curve), len(dream_curve))
    calib_mae = float(np.mean([abs(real_curve[t] - dream_curve[t]) for t in range(T)]))
    report["fidelity"] = {"mean_frame_mse": round(float(np.mean(mses)), 5)}
    report["calibration"] = {
        "real_infected_curve": real_curve, "dream_infected_curve": dream_curve,
        "calib_mae_cells": round(calib_mae, 3),
        "calib_mae_frac": round(calib_mae / env.N, 4)}

    # -- 5. save figures to the Volume ----------------------------------------
    import imageio.v2 as imageio
    outdir = f"{results_dir()}/epidemic_stage1"
    os.makedirs(outdir, exist_ok=True)

    def _u8(a):
        return (np.clip(np.asarray(a), 0, 1) * 255).astype(np.uint8)

    def _grid(origs, recons, n=8):
        rows = []
        for i in range(min(n, len(origs))):
            rows.append(np.concatenate([_u8(origs[i]), 255 * np.ones((64, 2, 3), np.uint8),
                                        _u8(recons[i])], axis=1))
        return np.concatenate(rows, axis=0)

    imageio.imwrite(f"{outdir}/recon_fg0.png", _grid(probe_np, recon0))
    imageio.imwrite(f"{outdir}/recon_fg25.png", _grid(probe_np, recon25))
    # real-vs-dream filmstrip on one calibration seed
    cenv = make_env(CFG, seed=2024)
    cmp = compare_to_real(cenv, best_tok, wm, device, H, seed=2024)
    strip = []
    for real, dream in zip(cmp["real"], cmp["dream"]):
        col = np.concatenate([_u8(real), 255 * np.ones((2, 64, 3), np.uint8),
                              _u8(dream)], axis=0)
        strip.append(col)
        strip.append(255 * np.ones((col.shape[0], 2, 3), np.uint8))
    imageio.imwrite(f"{outdir}/real_vs_dream.png", np.concatenate(strip, axis=1))
    VOL.commit()
    report["figures"] = ["recon_fg0.png", "recon_fg25.png", "real_vs_dream.png"]

    report["runtime_s"] = round(time.time() - t0, 1)
    report["ok"] = True
    return jsonify(report)


@app.local_entrypoint()
def main():
    out = stage1.remote()
    print(json.dumps(out, indent=2))
    tk = out["tokenizer"]
    print("\n=== STAGE 1", "OK" if out.get("ok") else "FAILED",
          f"in {out.get('runtime_s')}s on {out.get('gpu')} ===")
    print(f"  tokenizer I-recall  fg0={tk['fg0_uniform']['I_recall']}  "
          f"fg25={tk['fg25_luminance']['I_recall']}  (chosen={tk['chosen']})")
    print(f"  world model  token_acc={out['world_model']['token_acc']}  "
          f"reward_acc={out['world_model']['reward_acc']}")
    print(f"  calibration  real={out['calibration']['real_infected_curve']}")
    print(f"               dream={out['calibration']['dream_infected_curve']}")
    print(f"               MAE={out['calibration']['calib_mae_cells']} cells")
