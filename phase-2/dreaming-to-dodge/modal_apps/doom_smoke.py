"""DE-RISK GATE (Modal) for the "Dreaming to Dodge" VizDoom env.

Before spending anything on tokenizer / world-model / policy, prove the hard,
uncertain part works: VizDoom's `take_cover` runs HEADLESS inside the Modal
image and behaves through the Catch interface.  Checks:

  1. vizdoom imports; the built-in take_cover scenario is found.
  2. DoomTakeCover resets and returns clean 64x64x3 frames in [0,1].
  3. determinism: same seed -> identical first frame + identical rollout.
  4. honest baseline ordering: random survives < heuristic oracle dodger.
  5. saves a random filmstrip + an oracle filmstrip PNG (and a GIF) to the
     Volume, and echoes a couple of base64 frames so we can eyeball the scene.

Run:  .venv/bin/modal run modal_apps/doom_smoke.py
"""
import json
import modal

from common import app, IMAGE, VOLUMES, VOL, GPU_CHEAP, results_dir, jsonify

CFG = {"env": "doom", "resolution": 64, "frame_skip": 4, "t_max": 525,
       "living_reward": 1.0}


@app.function(image=IMAGE, gpu=GPU_CHEAP, volumes=VOLUMES, timeout=60 * 15)
def smoke():
    import os, time, numpy as np
    t0 = time.time()
    report = {}

    # -- 1. vizdoom imports + scenario present --------------------------------
    import vizdoom as vzd
    report["vizdoom_version"] = getattr(vzd, "__version__", "?")
    scfg = os.path.join(vzd.scenarios_path, "take_cover.cfg")
    swad = os.path.join(vzd.scenarios_path, "take_cover.wad")
    report["scenarios_path"] = vzd.scenarios_path
    report["take_cover_cfg_exists"] = os.path.exists(scfg)
    report["take_cover_wad_exists"] = os.path.exists(swad)

    from envs import DoomTakeCover, make_env

    # -- 2. env resets + frame shape/range ------------------------------------
    env = make_env(CFG, seed=0)
    obs = env.reset(0)
    report["num_actions"] = env.num_actions
    report["frame_shape"] = list(obs.shape)
    report["frame_min"] = round(float(obs.min()), 4)
    report["frame_max"] = round(float(obs.max()), 4)
    report["frame_mean"] = round(float(obs.mean()), 4)
    report["available_buttons"] = [str(b) for b in env.game.get_available_buttons()]

    # -- 3. determinism: same seed -> same first frame & rollout --------------
    a_env = make_env(CFG, seed=7); b_env = make_env(CFG, seed=7)
    fa0 = a_env.reset(7); fb0 = b_env.reset(7)
    same_first = bool(np.allclose(fa0, fb0))
    same_roll = True
    acts = [1, 2, 0, 2, 1, 1, 2, 0]
    for a in acts:
        fa, ra, da = a_env.step(a); fb, rb, db = b_env.step(a)
        if not (np.allclose(fa, fb) and ra == rb and da == db):
            same_roll = False
            break
    report["determinism_first_frame"] = same_first
    report["determinism_rollout"] = same_roll

    # -- 4. baseline survival: random vs heuristic oracle ---------------------
    def survival(kind, n=20):
        steps = []
        for ep in range(n):
            e = make_env(CFG, seed=1000 + ep)
            e.reset(1000 + ep)
            s, done = 0, False
            while not done:
                if kind == "random":
                    a = int(np.random.randint(e.num_actions))
                else:
                    a = e.oracle_action()
                _, r, done = e.step(a)
                s += 1
            steps.append(s)
            e.close()
        return float(np.mean(steps)), float(np.std(steps)), int(np.max(steps))

    r_mean, r_std, r_max = survival("random")
    o_mean, o_std, o_max = survival("oracle")
    report["random_survival_steps"] = {"mean": round(r_mean, 1), "std": round(r_std, 1), "max": r_max}
    report["oracle_survival_steps"] = {"mean": round(o_mean, 1), "std": round(o_std, 1), "max": o_max}
    report["frame_skip"] = CFG["frame_skip"]
    report["random_survival_tics"] = round(r_mean * CFG["frame_skip"], 1)
    report["oracle_survival_tics"] = round(o_mean * CFG["frame_skip"], 1)
    report["ordering_ok"] = bool(o_mean > r_mean)

    # -- 5. save filmstrips + GIFs + echo a couple base64 frames --------------
    import imageio.v2 as imageio
    import base64, io
    outdir = f"{results_dir()}/doom_smoke"
    os.makedirs(outdir, exist_ok=True)

    def _u8(a): return (np.clip(np.asarray(a), 0, 1) * 255).astype(np.uint8)

    def rollout(kind, seed, n=48):
        e = make_env(CFG, seed=seed); e.reset(seed)
        frames, done, i = [e.render().copy()], False, 0
        while not done and i < n:
            a = int(np.random.randint(e.num_actions)) if kind == "random" else e.oracle_action()
            f, r, done = e.step(a); frames.append(f.copy()); i += 1
        e.close()
        return frames

    for kind, seed in [("random", 4242), ("oracle", 4242)]:
        fr = rollout(kind, seed)
        strip = np.concatenate([_u8(f) for f in fr[:24]], axis=1)
        imageio.imwrite(f"{outdir}/filmstrip_{kind}.png", strip)
        imageio.mimsave(f"{outdir}/rollout_{kind}.gif", [_u8(f) for f in fr], duration=0.12, loop=0)

    b64 = []
    fr = rollout("oracle", 99)
    for f in fr[:3]:
        buf = io.BytesIO(); imageio.imwrite(buf, _u8(f), format="png")
        b64.append(base64.b64encode(buf.getvalue()).decode())
    report["sample_frames_b64"] = b64

    VOL.commit()
    report["runtime_s"] = round(time.time() - t0, 1)
    report["ok"] = bool(report["take_cover_cfg_exists"] and same_first
                        and report["ordering_ok"] and obs.shape == (64, 64, 3))
    return jsonify(report)


@app.local_entrypoint()
def main():
    out = smoke.remote()
    light = {k: v for k, v in out.items() if k != "sample_frames_b64"}
    print(json.dumps(light, indent=2))
    print("\n=== DOOM SMOKE", "OK" if out.get("ok") else "FAILED",
          f"in {out.get('runtime_s')}s ===")
    print(f"  vizdoom {out.get('vizdoom_version')}  frame {out.get('frame_shape')} "
          f"range[{out.get('frame_min')},{out.get('frame_max')}]")
    print(f"  determinism: first={out.get('determinism_first_frame')} "
          f"rollout={out.get('determinism_rollout')}")
    print(f"  survival steps  random={out.get('random_survival_steps')}  "
          f"oracle={out.get('oracle_survival_steps')}")
    # dump the sample frames locally so we can look at the scene
    import base64, os
    os.makedirs("results/doom_smoke", exist_ok=True)
    for i, b in enumerate(out.get("sample_frames_b64", [])):
        with open(f"results/doom_smoke/sample_{i}.png", "wb") as f:
            f.write(base64.b64decode(b))
    print("  wrote results/doom_smoke/sample_*.png")
