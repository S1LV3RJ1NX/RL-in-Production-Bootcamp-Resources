"""Evaluation + dream rendering on Modal.

  * evaluate       — run the trained policy in the REAL env and compare to the
                     random and oracle baselines (the honest "beat-a-baseline"
                     numbers).  Also measures world-model fidelity (dream-vs-real
                     next-frame MSE) under the oracle action sequence.
  * render_dreams  — produce the demo artifacts: frame reconstructions, an
                     imagined-policy dream filmstrip, and a world-model-vs-real
                     side-by-side, all saved as PNG/GIF on the Volume and echoed
                     into results JSON (as base64 thumbnails for the demo page).

Both return plain JSON.  Mirrors rlhf-lab/modal_apps/evals.py.
"""
import json
import modal

from common import (
    app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, load_buffer, data_path,
    dreams_dir, write_result, jsonify,
)


def _build_models(cfg, tag, device, num_actions):
    import torch
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from actor_critic import ActorCritic, ActorCriticConfig
    tc, wc, pc = cfg["tokenizer"], cfg["world_model"], cfg["policy"]
    tcfg = TokenizerConfig(in_channels=3, resolution=cfg["resolution"], ch=tc["ch"],
                           num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                           embed_dim=tc["embed_dim"])
    tok = Tokenizer(tcfg).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wcfg = WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                            num_actions=num_actions, max_timesteps=wc["max_timesteps"],
                            embed_dim=wc["embed_dim"], num_layers=wc["num_layers"],
                            num_heads=wc["num_heads"], dropout=wc["dropout"])
    wm = WorldModel(wcfg).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))
    pcfg = ActorCriticConfig(resolution=cfg["resolution"], num_actions=num_actions,
                             conv_ch=pc["conv_ch"], hidden_dim=pc["hidden_dim"])
    ac = ActorCritic(pcfg).to(device).eval()
    ac.load_state_dict(torch.load(ckpt_path(tag, "actor_critic"), map_location=device))
    return tok, wm, ac


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 20)
def evaluate(cfg: dict, tag: str, n_episodes: int = 500, round_idx: int = -1):
    import time, numpy as np, torch
    from envs import make_env

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(cfg, seed=12345)
    tok, wm, ac = _build_models(cfg, tag, device, env.num_actions)

    has_saved = hasattr(env, "saved_fraction")

    def run_policy(kind, n):
        rets, catches, saved = [], [], []
        for ep in range(n):
            obs = env.reset(90000 + ep)
            state = ac.initial_state(1, device)
            ep_ret, done = 0.0, False
            while not done:
                if kind == "random":
                    a = int(np.random.randint(env.num_actions))
                elif kind == "oracle":
                    a = env.oracle_action() if hasattr(env, "oracle_action") \
                        else int(np.random.randint(env.num_actions))
                else:  # trained policy (greedy)
                    with torch.no_grad():
                        f = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
                        act, _, state = ac.act(f, state, sample=False)
                        a = int(act.item())
                obs, r, done = env.step(a)
                ep_ret += r
            rets.append(ep_ret)
            catches.append(1.0 if ep_ret > 0 else 0.0)
            if has_saved:
                saved.append(env.saved_fraction())
        out = {"return_mean": float(np.mean(rets)),
               "return_std": float(np.std(rets)),
               "catch_rate": float(np.mean(catches))}
        if has_saved:      # the epidemic-containment headline metric
            out["saved_fraction"] = float(np.mean(saved))
            out["saved_fraction_std"] = float(np.std(saved))
        return out

    has_oracle = hasattr(env, "oracle_action")
    result = {"tag": tag, "n_episodes": n_episodes,
              "policy": run_policy("policy", n_episodes),
              "random_baseline": run_policy("random", n_episodes)}
    if has_oracle:
        result["oracle_baseline"] = run_policy("oracle", n_episodes)

    # world-model fidelity (dream vs real under oracle actions), if oracle exists
    if has_oracle:
        from imagination import compare_to_real
        mses = [compare_to_real(make_env(cfg, seed=700 + s), tok, wm, device,
                                horizon=cfg["policy"]["horizon"], seed=700 + s)["mean_mse"]
                for s in range(10)]
        result["wm_fidelity_mse"] = float(np.mean([m for m in mses if m is not None]))

    result["runtime_s"] = round(time.time() - t0, 1)
    write_result(f"eval_{tag}", result); VOL.commit()
    return jsonify(result)


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 15)
def recon_check(cfg: dict, tag: str):
    """Tokenizer-only sanity gate: does the codebook still contain the ball?

    Builds the tokenizer for `tag`, then generates Catch frames with KNOWN ball
    positions (every column, several rows), reconstructs them, and measures:
      * recon_mse         — overall reconstruction MSE
      * ball_region_mse   — MSE restricted to the ball's true cell
      * ball_recall       — mean reconstructed brightness inside the ball cell
                            (1.0 = ball fully preserved, 0.0 = ball dropped)
    Also saves a real-vs-decoded grid PNG to the Volume + returns it as base64,
    and reports codebook health (active codes).  Needs NO world model / policy.
    """
    import io, base64, time, numpy as np, torch, imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from envs import Catch
    from tokenizer import Tokenizer, TokenizerConfig

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = cfg["tokenizer"]
    tcfg = TokenizerConfig(in_channels=3, resolution=cfg["resolution"], ch=tc["ch"],
                           num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                           embed_dim=tc["embed_dim"],
                           ema=tc.get("ema", True), revive_dead=tc.get("revive_dead", True))
    tok = Tokenizer(tcfg).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))

    grid = cfg.get("grid", 8)
    size = cfg["resolution"]
    cell = size // grid
    env = Catch(grid=grid, size=size, seed=0)

    # build a panel of frames with the ball at every column and a few rows
    specs, frames = [], []
    for col in range(grid):
        for row in (1, grid // 2, grid - 2):
            env.reset(0)
            env.ball_col = col; env.ball_row = row; env.paddle = (col + 2) % grid
            frames.append(env.render()); specs.append((row, col, env.paddle))
    frames = np.asarray(frames, dtype=np.float32)
    x = torch.tensor(frames).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        xh = tok.decode(tok.encode(x)).permute(0, 2, 3, 1).cpu().numpy()

    recon_mse = float(np.mean((frames - xh) ** 2))
    ball_mses, ball_recalls = [], []
    for i, (row, col, pad) in enumerate(specs):
        y0, x0 = row * cell, col * cell
        real_cell = frames[i, y0:y0 + cell, x0:x0 + cell]
        rec_cell = xh[i, y0:y0 + cell, x0:x0 + cell]
        ball_mses.append(float(np.mean((real_cell - rec_cell) ** 2)))
        # ball is white (1,1,1); recall = mean reconstructed brightness in the cell
        ball_recalls.append(float(np.clip(rec_cell, 0, 1).mean()))

    # save a compact real-vs-decoded grid PNG (12 samples)
    sel = list(range(0, len(frames), max(1, len(frames) // 12)))[:12]
    fig, axes = plt.subplots(2, len(sel), figsize=(1.4 * len(sel), 3.1))
    for j, i in enumerate(sel):
        axes[0, j].imshow(np.clip(frames[i], 0, 1)); axes[0, j].axis("off")
        axes[1, j].imshow(np.clip(xh[i], 0, 1)); axes[1, j].axis("off")
    axes[0, 0].set_title("real", fontsize=9); axes[1, 0].set_title("decoded", fontsize=9)
    fig.suptitle(f"[{tag}] recon MSE {recon_mse:.4f} | ball recall "
                 f"{np.mean(ball_recalls):.2f} | active codes {tok.quantizer.num_active()}",
                 fontsize=11)
    fig.tight_layout()
    import os
    os.makedirs(dreams_dir(), exist_ok=True)
    png_path = f"{dreams_dir()}/recon_check_{tag}.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    with open(png_path, "rb") as f:
        grid_b64 = base64.b64encode(f.read()).decode()

    result = {"tag": tag, "recon_mse": recon_mse,
              "ball_region_mse": float(np.mean(ball_mses)),
              "ball_recall": float(np.mean(ball_recalls)),
              "active_codes": tok.quantizer.num_active(),
              "vocab_size": tc["vocab_size"], "grid_png_b64": grid_b64,
              "runtime_s": round(time.time() - t0, 1)}
    write_result(f"recon_check_{tag}", result); VOL.commit()
    return jsonify({k: v for k, v in result.items() if k != "grid_png_b64"})


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 20)
def render_dreams(cfg: dict, tag: str, n_dreams: int = 6):
    import os, io, base64, time, numpy as np, torch, imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from envs import make_env
    from imagination import render_dream, compare_to_real

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(cfg, seed=555)
    tok, wm, ac = _build_models(cfg, tag, device, env.num_actions)
    outdir = dreams_dir(); os.makedirs(outdir, exist_ok=True)
    H = cfg["policy"]["horizon"]

    def png_b64(arr):
        buf = io.BytesIO()
        imageio.imwrite(buf, (np.clip(arr, 0, 1) * 255).astype("uint8"), format="png")
        return base64.b64encode(buf.getvalue()).decode()

    # 1) reconstructions: real vs tokenizer decode
    frames, _, _, _ = load_buffer(data_path(tag))
    idx = np.random.choice(len(frames), 8, replace=False)
    recon = []
    with torch.no_grad():
        for i in idx:
            x = torch.tensor(frames[i]).permute(2, 0, 1)[None].to(device)
            xh = tok.decode(tok.encode(x))[0].permute(1, 2, 0).cpu().numpy()
            recon.append({"real": png_b64(frames[i]), "recon": png_b64(xh)})

    # 2) policy dream filmstrips (imagined rollouts)
    dreams = []
    for d in range(n_dreams):
        obs = env.reset(1000 + d)
        x0 = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        dr = render_dream(tok, wm, ac, x0, H, device, temperature=0.9, use_policy=True)
        dreams.append({"frames": [png_b64(f) for f in dr["frames"]],
                       "actions": dr["actions"], "rewards": dr["rewards"],
                       "dones": dr["dones"]})
        # also save a GIF
        gif = [(np.clip(f, 0, 1) * 255).astype("uint8") for f in dr["frames"]]
        imageio.mimsave(f"{outdir}/dream_{tag}_{d}.gif", gif, duration=0.25, loop=0)

    # 3) world-model vs real (same oracle actions)
    wm_vs_real = []
    if hasattr(env, "oracle_action"):
        for s in range(3):
            cmp = compare_to_real(make_env(cfg, seed=2000 + s), tok, wm, device, H, 2000 + s)
            wm_vs_real.append({"real": [png_b64(f) for f in cmp["real"]],
                               "dream": [png_b64(f) for f in cmp["dream"]],
                               "mse_per_step": cmp["mse_per_step"],
                               "actions": cmp["actions"]})

    # a static reconstruction figure for the paper
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    with torch.no_grad():
        for j, i in enumerate(idx):
            x = torch.tensor(frames[i]).permute(2, 0, 1)[None].to(device)
            xh = tok.decode(tok.encode(x))[0].permute(1, 2, 0).cpu().numpy()
            axes[0, j].imshow(np.clip(frames[i], 0, 1)); axes[0, j].axis("off")
            axes[1, j].imshow(np.clip(xh, 0, 1)); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("real", fontsize=12)
    axes[1, 0].set_ylabel("recon", fontsize=12)
    fig.suptitle("Tokenizer reconstructions (top: real, bottom: decoded tokens)")
    fig.tight_layout(); fig.savefig(f"{outdir}/reconstructions_{tag}.png", dpi=120)
    plt.close(fig)

    payload = {"tag": tag, "reconstructions": recon, "dreams": dreams,
               "wm_vs_real": wm_vs_real, "runtime_s": round(time.time() - t0, 1)}
    write_result(f"dreams_{tag}", payload); VOL.commit()
    # return a light summary (not the base64 blobs) to the client
    return jsonify({"tag": tag, "n_reconstructions": len(recon),
                    "n_dreams": len(dreams), "n_wm_vs_real": len(wm_vs_real),
                    "runtime_s": payload["runtime_s"],
                    "results_json": f"results/dreams_{tag}.json"})


# ===========================================================================
# pull every results JSON off the Volume so the client can build figures/paper
# ===========================================================================
@app.function(image=IMAGE, volumes=VOLUMES, timeout=60 * 10)
def collect_results():
    import os, json
    from common import results_dir
    VOL.reload()
    rd = results_dir()
    out = {}
    if os.path.isdir(rd):
        for fn in sorted(os.listdir(rd)):
            if fn.endswith(".json"):
                with open(os.path.join(rd, fn)) as f:
                    out[fn] = json.load(f)
    return out


@app.local_entrypoint()
def sync(outdir: str = "results"):
    """Download all results JSON from the Volume into local `outdir/`."""
    import os, json
    os.makedirs(outdir, exist_ok=True)
    blobs = collect_results.remote()
    for fn, obj in blobs.items():
        with open(os.path.join(outdir, fn), "w") as f:
            json.dump(obj, f, indent=2)
    print(f"synced {len(blobs)} result files -> {outdir}/")
    for fn in sorted(blobs):
        print("  ", fn)


@app.local_entrypoint()
def eval(cfg: str = "configs/catch.yaml", tag: str = "catch", n: int = 500):
    import yaml
    with open(cfg) as f: c = yaml.safe_load(f)
    print(json.dumps(evaluate.remote(c, tag, n), indent=2))


@app.local_entrypoint()
def dreams(cfg: str = "configs/catch.yaml", tag: str = "catch", n: int = 6):
    import yaml
    with open(cfg) as f: c = yaml.safe_load(f)
    print(json.dumps(render_dreams.remote(c, tag, n), indent=2))


@app.local_entrypoint()
def recon(cfg: str = "configs/catch_win.yaml", tag: str = "win", out: str = "results"):
    """Tokenizer-only ball-recall gate. Saves the real-vs-decoded grid PNG
    locally (recon_check_<tag>.png) so it can be inspected before training the
    policy."""
    import os, io, base64, yaml
    with open(cfg) as f: c = yaml.safe_load(f)
    summary = recon_check.remote(c, tag)
    print(json.dumps(summary, indent=2))
    # pull the full result (with the PNG) off the Volume and write the PNG local
    blobs = collect_results.remote()
    key = f"recon_check_{tag}.json"
    if key in blobs and blobs[key].get("grid_png_b64"):
        os.makedirs(out, exist_ok=True)
        png = os.path.join(out, f"recon_check_{tag}.png")
        with open(png, "wb") as f:
            f.write(base64.b64decode(blobs[key]["grid_png_b64"]))
        print("saved", png)
