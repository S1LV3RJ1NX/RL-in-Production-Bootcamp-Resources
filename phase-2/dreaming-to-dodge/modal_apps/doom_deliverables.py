"""The "Dreaming to Dodge" WOW artifacts, rendered from the trained stack.

Produces the three escalating beats for the demo, all saved to the Volume as
GIFs/PNGs (+ base64 echoes for a web page / notebook):

  1. AGENT IN REALITY  — the imagination-trained policy playing the REAL VizDoom
     take_cover env: it strafes to dodge fireballs and survives.  (proof it works)
  2. AGENT IN THE DREAM — the same policy acting inside the world model's
     hallucination: no game engine, the Transformer generates each next frame.
  3. REAL vs DREAM (same actions) — roll the real env under the policy, replay
     those exact actions in the dream from the same first frame, side by side:
     the dream tracks reality (the world model learned the game).

Run:  .venv/bin/modal run modal_apps/doom_deliverables.py --tag doom
"""
import json
import modal
from common import (app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, dreams_dir,
                    write_result, jsonify)


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 30)
def deliverables(cfg: dict, tag: str = "doom", n_clips: int = 4, max_len: int = 200):
    import os, io, base64, time, numpy as np, torch, imageio.v2 as imageio
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from actor_critic import ActorCritic, ActorCriticConfig
    from imagination import render_dream

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc, pc = cfg["tokenizer"], cfg["world_model"], cfg["policy"]
    env = make_env(cfg, seed=4321)
    A = env.num_actions

    tok = Tokenizer(TokenizerConfig(resolution=cfg["resolution"], ch=tc["ch"],
                    num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    num_actions=A, max_timesteps=wc["max_timesteps"], embed_dim=wc["embed_dim"],
                    num_layers=wc["num_layers"], num_heads=wc["num_heads"])).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))
    ac = ActorCritic(ActorCriticConfig(resolution=cfg["resolution"], num_actions=A,
                     conv_ch=pc["conv_ch"], hidden_dim=pc["hidden_dim"])).to(device).eval()
    ac.load_state_dict(torch.load(ckpt_path(tag, "actor_critic"), map_location=device))

    outdir = f"{dreams_dir()}/doom_deliverables"
    os.makedirs(outdir, exist_ok=True)

    def _u8(a): return (np.clip(np.asarray(a), 0, 1) * 255).astype(np.uint8)

    def b64gif_frames(frames, dur=0.1):
        buf = io.BytesIO()
        imageio.mimsave(buf, [_u8(f) for f in frames], format="gif", duration=dur, loop=0)
        return base64.b64encode(buf.getvalue()).decode()

    # -- 1. AGENT IN REALITY: greedy policy, record frames + actions ----------
    real_clips, survivals = [], []
    best = None
    for c in range(n_clips):
        e = make_env(cfg, seed=6000 + c)
        obs = e.reset(6000 + c)
        st = ac.initial_state(1, device)
        frames, acts, done, steps = [obs.copy()], [], False, 0
        while not done and steps < max_len:
            with torch.no_grad():
                f = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
                a, _, st = ac.act(f, st, sample=False)
                a = int(a.item())
            obs, r, done = e.step(a)
            frames.append(obs.copy()); acts.append(a); steps += 1
        e.close()
        survivals.append(steps)
        imageio.mimsave(f"{outdir}/real_agent_{c}.gif", [_u8(f) for f in frames], duration=0.08, loop=0)
        if best is None or steps > best[0]:
            best = (steps, frames, acts, 6000 + c)

    # -- 2. AGENT IN THE DREAM (policy acts inside the world model) ------------
    dream_clips = []
    for c in range(min(n_clips, 3)):
        obs = make_env(cfg, seed=7000 + c).reset(7000 + c)
        x0 = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        dr = render_dream(tok, wm, ac, x0, min(max_len, 80), device, temperature=1.0, use_policy=True)
        imageio.mimsave(f"{outdir}/dream_agent_{c}.gif", [_u8(f) for f in dr["frames"]], duration=0.1, loop=0)
        dream_clips.append({"steps": len(dr["frames"]) - 1, "dreamed_dones": int(sum(dr["dones"]))})

    # -- 3. REAL vs DREAM under the policy's OWN actions (best real clip) ------
    _, bframes, bacts, bseed = best
    x0 = torch.tensor(bframes[0], dtype=torch.float32).permute(2, 0, 1)

    class _Fixed:
        def initial_state(self, b, d): return None
        def act(self, *a, **k): raise RuntimeError
    dseq = render_dream(tok, wm, _Fixed(), x0, len(bacts), device, use_policy=False, action_seq=bacts)
    L = min(len(bframes), len(dseq["frames"]))
    sbs = [np.concatenate([_u8(bframes[i]), 255 * np.ones((2, 64, 3), np.uint8),
                           _u8(dseq["frames"][i])], 0) for i in range(L)]
    imageio.mimsave(f"{outdir}/real_vs_dream_policy.gif", sbs, duration=0.1, loop=0)
    strip = np.concatenate([np.concatenate([s, 255 * np.ones((s.shape[0], 2, 3), np.uint8)], 1)
                            for s in sbs[:24]], 1)
    imageio.imwrite(f"{outdir}/real_vs_dream_policy.png", strip)

    VOL.commit()
    fs = cfg.get("frame_skip", 4)
    result = {"tag": tag,
              "real_survival_steps": {"mean": round(float(np.mean(survivals)), 1),
                                      "max": int(np.max(survivals)), "all": survivals},
              "real_survival_tics_mean": round(float(np.mean(survivals)) * fs, 1),
              "best_clip_steps": best[0], "best_clip_seed": bseed,
              "dream_clips": dream_clips,
              "real_agent_gif_b64": b64gif_frames(best[1][:max_len], dur=0.08),
              "figures": [f"doom_deliverables/real_agent_{c}.gif" for c in range(n_clips)]
                         + [f"doom_deliverables/dream_agent_{c}.gif" for c in range(min(n_clips, 3))]
                         + ["doom_deliverables/real_vs_dream_policy.gif",
                            "doom_deliverables/real_vs_dream_policy.png"],
              "runtime_s": round(time.time() - t0, 1)}
    write_result(f"deliverables_{tag}", result)
    VOL.commit()
    return jsonify(result)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom"):
    import yaml, base64, os
    with open(cfg) as f:
        c = yaml.safe_load(f)
    out = deliverables.remote(c, tag)
    light = {k: v for k, v in out.items() if k != "real_agent_gif_b64"}
    print(json.dumps(light, indent=2))
    os.makedirs("results/doom_deliverables", exist_ok=True)
    if out.get("real_agent_gif_b64"):
        with open("results/doom_deliverables/real_agent_best.gif", "wb") as f:
            f.write(base64.b64decode(out["real_agent_gif_b64"]))
        print("  wrote results/doom_deliverables/real_agent_best.gif")
