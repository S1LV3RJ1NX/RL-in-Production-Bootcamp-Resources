"""Policy-FREE dream preview for the trained (tokenizer + world model) doom stack.

Answers the one question that gates spending on the policy: after full training,
does the DREAM show fireballs, and does it respond to actions?  Renders, from the
round-`round` checkpoints and needing NO actor-critic:

  1. reconstruction grid (real | tokenizer-decoded) on fireball frames + recall.
  2. real-vs-dream filmstrip/GIF under the heuristic dodger actions (fidelity).
  3. three dreams from the SAME start frame under FIXED action sequences
     (all-left, all-right, alternate) -- if the dream pans differently per action,
     the world model learned action-conditioning (a controllable dream).

Run:  .venv/bin/modal run modal_apps/doom_dream_preview.py --tag doom --round 0
"""
import json
import modal
from common import (app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, load_buffer,
                    data_path, dreams_dir, write_result, jsonify)


def _fireball_mask(f):
    import numpy as np
    f = np.asarray(f)
    return (f.max(2) > 0.55) & ((f[:, :, 0] - np.maximum(f[:, :, 1], f[:, :, 2])) > 0.10)


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 20)
def preview(cfg: dict, tag: str = "doom", round_idx: int = 0, horizon: int = 32):
    import os, time, base64, io, numpy as np, torch, imageio.v2 as imageio
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from imagination import render_dream, compare_to_real

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc = cfg["tokenizer"], cfg["world_model"]
    env = make_env(cfg, seed=555)
    A = env.num_actions

    tok = Tokenizer(TokenizerConfig(resolution=cfg["resolution"], ch=tc["ch"],
                    num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    num_actions=A, max_timesteps=wc["max_timesteps"], embed_dim=wc["embed_dim"],
                    num_layers=wc["num_layers"], num_heads=wc["num_heads"])).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))

    outdir = f"{dreams_dir()}/doom_preview"
    os.makedirs(outdir, exist_ok=True)

    def _u8(a): return (np.clip(np.asarray(a), 0, 1) * 255).astype(np.uint8)

    class _Fixed:
        def initial_state(self, b, d): return None
        def act(self, *a, **k): raise RuntimeError

    # -- 1. reconstruction on fireball frames ---------------------------------
    frames, _, _, _ = load_buffer(data_path(tag))
    fb_idx = [j for j in range(min(6000, len(frames))) if _fireball_mask(frames[j]).sum() > 2][:8]
    xs = torch.tensor(frames[fb_idx]).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        rec = tok.decode(tok.encode(xs)).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    num = den = 0.0
    for o, r in zip(frames[fb_idx], rec):
        m = _fireball_mask(o)
        if m.sum():
            num += float(np.clip(r, 0, 1).max(2)[m].sum()); den += float(o.max(2)[m].sum())
    recon_grid = np.concatenate(
        [np.concatenate([_u8(frames[fb_idx[i]]), 255 * np.ones((64, 2, 3), np.uint8), _u8(rec[i])], 1)
         for i in range(len(fb_idx))], 0)
    imageio.imwrite(f"{outdir}/recon.png", recon_grid)

    # -- 2. real vs dream under the dodger actions ----------------------------
    cmp = compare_to_real(make_env(cfg, seed=2024), tok, wm, device, horizon, seed=2024)
    rvd = [np.concatenate([_u8(r), 255 * np.ones((2, 64, 3), np.uint8), _u8(d)], 0)
           for r, d in zip(cmp["real"], cmp["dream"])]
    imageio.imwrite(f"{outdir}/real_vs_dream.png", np.concatenate(
        [np.concatenate([c, 255 * np.ones((c.shape[0], 2, 3), np.uint8)], 1) for c in rvd], 1))
    imageio.mimsave(f"{outdir}/real_vs_dream.gif", rvd, duration=0.2, loop=0)

    # -- 3. controllability: same start, fixed action sequences ---------------
    obs = make_env(cfg, seed=777).reset(777)
    x0 = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
    seqs = {"all_left": [1] * horizon, "all_right": [2] * horizon,
            "alternate": [1 if i % 6 < 3 else 2 for i in range(horizon)]}
    ctrl = {}
    for name, seq in seqs.items():
        dr = render_dream(tok, wm, _Fixed(), x0, horizon, device, use_policy=False, action_seq=seq)
        imageio.mimsave(f"{outdir}/dream_{name}.gif", [_u8(f) for f in dr["frames"]], duration=0.15, loop=0)
        strip = np.concatenate([_u8(f) for f in dr["frames"][:24]], 1)
        imageio.imwrite(f"{outdir}/dream_{name}.png", strip)
        ctrl[name] = {"n_frames": len(dr["frames"]), "dreamed_dones": int(sum(dr["dones"]))}

    # echo a few base64 samples: real, decoded, and one dream frame mid-roll
    def b64(a):
        buf = io.BytesIO(); imageio.imwrite(buf, _u8(a), format="png"); return base64.b64encode(buf.getvalue()).decode()
    samples = {"real0": b64(frames[fb_idx[0]]), "recon0": b64(rec[0])}

    VOL.commit()
    result = {"tag": tag, "round": round_idx,
              "recon_fireball_recall": round(num / den if den else 0.0, 4),
              "fidelity_mean_mse": cmp["mean_mse"], "fidelity_steps": len(cmp["dream"]),
              "controllability": ctrl,
              "figures": ["doom_preview/recon.png", "doom_preview/real_vs_dream.png",
                          "doom_preview/real_vs_dream.gif", "doom_preview/dream_all_left.gif",
                          "doom_preview/dream_all_right.gif", "doom_preview/dream_alternate.gif"],
              "samples_b64": samples, "runtime_s": round(time.time() - t0, 1)}
    write_result(f"dream_preview_{tag}", result)
    VOL.commit()
    return jsonify(result)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom", round: int = 0):
    import yaml, base64, os
    with open(cfg) as f:
        c = yaml.safe_load(f)
    out = preview.remote(c, tag, round)
    light = {k: v for k, v in out.items() if k != "samples_b64"}
    print(json.dumps(light, indent=2))
    os.makedirs("results/doom_preview", exist_ok=True)
    for k, b in out.get("samples_b64", {}).items():
        with open(f"results/doom_preview/{k}.png", "wb") as f:
            f.write(base64.b64decode(b))
    print(f"\n  recon fireball-recall: {out['recon_fireball_recall']}  "
          f"fidelity_mse: {out['fidelity_mean_mse']}")
    print(f"  controllability: {out['controllability']}")
