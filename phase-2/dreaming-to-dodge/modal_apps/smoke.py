"""STAGE-1 smoke test — prove the WHOLE IRIS pipeline runs end-to-end on a
handful of steps, on a CHEAP GPU, in under ~5 minutes.  Fix bugs here before
spending real budget.

It runs, in ONE container: collect -> train_tokenizer -> train_world_model ->
imagine one policy update -> evaluate a few episodes -> render one dream.  It
asserts shapes are sane, losses are finite, and a dream decodes -- then returns
a plain-JSON report.  Mirrors rlhf-lab/modal_apps/smoke.py in spirit (but does
the full pipeline, not just a GPU probe).
"""
import json
import modal

from common import app, IMAGE, VOLUMES, VOL, GPU_CHEAP, jsonify


TINY = {
    "env": "catch", "grid": 8, "resolution": 64, "seed": 0,
    "collect": {"rounds": 1, "steps_per_round": 800, "eps_greedy": 0.1,
                "buffer_capacity": 5000},
    "tokenizer": {"num_tokens": 16, "vocab_size": 128, "embed_dim": 64, "ch": 32,
                  "beta": 0.25, "steps_per_round": 60, "batch_size": 32, "lr": 3e-4},
    "world_model": {"max_timesteps": 6, "embed_dim": 128, "num_layers": 2,
                    "num_heads": 4, "dropout": 0.1, "steps_per_round": 60,
                    "batch_size": 16, "lr": 3e-4},
    "policy": {"hidden_dim": 128, "conv_ch": 32, "gamma": 0.995, "lam": 0.95,
               "entropy_coef": 0.01, "horizon": 6, "updates_per_round": 15,
               "batch_size": 32, "lr": 3e-4, "temperature": 1.0},
}


@app.function(image=IMAGE, gpu=GPU_CHEAP, volumes=VOLUMES, timeout=60 * 10)
def smoke():
    import time, math, numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from actor_critic import ActorCritic, ActorCriticConfig, actor_critic_loss
    from imagination import imagine_rollout, render_dream, reward_value_to_class

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"device": device, "gpu": torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "cpu"}
    cfg = TINY

    # -- 1. collect a few episodes with a random policy -----------------------
    env = make_env(cfg, seed=0)
    frames, actions, rewards, dones = [], [], [], []
    obs = env.reset(0)
    for i in range(cfg["collect"]["steps_per_round"]):
        a = int(np.random.randint(env.num_actions))
        nobs, r, done = env.step(a)
        frames.append(obs); actions.append(a); rewards.append(r); dones.append(int(done))
        obs = env.reset(i + 1) if done else nobs
    frames = torch.tensor(np.asarray(frames, dtype=np.float32)).permute(0, 3, 1, 2)
    report["collected"] = len(frames)
    assert frames.shape[1:] == (3, 64, 64)

    # -- 2. tokenizer -----------------------------------------------------------
    tc = cfg["tokenizer"]
    tok = Tokenizer(TokenizerConfig(resolution=64, ch=tc["ch"], num_tokens=tc["num_tokens"],
                                    vocab_size=tc["vocab_size"], embed_dim=tc["embed_dim"])).to(device)
    opt = torch.optim.Adam(tok.parameters(), lr=tc["lr"])
    first_recon = last_recon = None
    tok.train()
    for s in range(tc["steps_per_round"]):
        x = frames[torch.randint(0, len(frames), (tc["batch_size"],))].to(device)
        out = tok(x); opt.zero_grad(); out["loss"].backward(); opt.step()
        if s == 0: first_recon = float(out["recon_loss"])
        last_recon = float(out["recon_loss"])
    assert math.isfinite(last_recon)
    report["tokenizer"] = {"params": tok.num_params(), "recon_first": first_recon,
                           "recon_last": last_recon,
                           "codebook_usage": float(out["codebook_usage"])}

    # pre-encode
    tok.eval()
    with torch.no_grad():
        toks = tok.encode(frames.to(device)).cpu()
    assert toks.shape == (len(frames), tc["num_tokens"])

    # -- 3. world model ---------------------------------------------------------
    wc = cfg["world_model"]; L = wc["max_timesteps"]
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                                     num_actions=env.num_actions, max_timesteps=L,
                                     embed_dim=wc["embed_dim"], num_layers=wc["num_layers"],
                                     num_heads=wc["num_heads"])).to(device)
    wopt = torch.optim.AdamW(wm.parameters(), lr=wc["lr"], weight_decay=0.01)
    rew_cls = reward_value_to_class(torch.tensor(rewards)).long()
    acts_t = torch.tensor(actions).long(); dones_t = torch.tensor(dones).long()
    wm.train(); wm_first = wm_last = None
    for s in range(wc["steps_per_round"]):
        st = np.random.randint(0, len(frames) - L, wc["batch_size"])
        tks = torch.stack([toks[i:i + L] for i in st]).to(device)
        acs = torch.stack([acts_t[i:i + L] for i in st]).to(device)
        rws = torch.stack([rew_cls[i:i + L] for i in st]).to(device)
        dns = torch.stack([dones_t[i:i + L] for i in st]).to(device)
        o = wm.compute_loss(tks, acs, rws, dns)
        wopt.zero_grad(); o["loss"].backward(); wopt.step()
        if s == 0: wm_first = float(o["loss"])
        wm_last = float(o["loss"])
    assert math.isfinite(wm_last)
    report["world_model"] = {"params": wm.num_params(), "loss_first": wm_first,
                             "loss_last": wm_last, "token_acc": float(o["token_acc"]),
                             "reward_acc": float(o["reward_acc"]),
                             "done_acc": float(o["done_acc"])}

    # -- 4. one imagination policy update --------------------------------------
    wm.eval()
    for p in tok.parameters(): p.requires_grad_(False)
    for p in wm.parameters(): p.requires_grad_(False)
    pc = cfg["policy"]
    pcfg = ActorCriticConfig(resolution=64, num_actions=env.num_actions,
                             conv_ch=pc["conv_ch"], hidden_dim=pc["hidden_dim"],
                             entropy_coef=pc["entropy_coef"])
    ac = ActorCritic(pcfg).to(device)
    aopt = torch.optim.Adam(ac.parameters(), lr=pc["lr"])
    ac.train(); ret_first = ret_last = None
    for u in range(pc["updates_per_round"]):
        start = frames[torch.randint(0, len(frames), (pc["batch_size"],))].to(device)
        roll = imagine_rollout(tok, wm, ac, start, pc["horizon"], device)
        lo = actor_critic_loss(roll["logits"], roll["actions"], roll["values"],
                               roll["rewards"], roll["dones"], pcfg)
        aopt.zero_grad(); lo["loss"].backward(); aopt.step()
        if u == 0: ret_first = float(lo["return_mean"])
        ret_last = float(lo["return_mean"])
    assert math.isfinite(ret_last)
    report["policy"] = {"params": ac.num_params(), "imagined_return_first": ret_first,
                        "imagined_return_last": ret_last, "entropy": float(lo["entropy"])}

    # -- 5. eval a few real episodes -------------------------------------------
    ac.eval(); rets = []
    for ep in range(8):
        obs = env.reset(500 + ep); state = ac.initial_state(1, device); done = False; R = 0.0
        while not done:
            with torch.no_grad():
                f = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
                a, _, state = ac.act(f, state, sample=False)
            obs, r, done = env.step(int(a.item())); R += r
        rets.append(R)
    report["eval"] = {"policy_return_mean": float(np.mean(rets))}

    # -- 6. render one dream (assert it decodes) -------------------------------
    obs = env.reset(999)
    x0 = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
    dr = render_dream(tok, wm, ac, x0, pc["horizon"], device, use_policy=True)
    assert len(dr["frames"]) >= 1 and dr["frames"][0].shape == (64, 64, 3)
    report["dream"] = {"n_frames": len(dr["frames"]), "rewards": dr["rewards"],
                       "dones": dr["dones"]}

    report["runtime_s"] = round(time.time() - t0, 1)
    report["ok"] = True
    return jsonify(report)


@app.local_entrypoint()
def main():
    out = smoke.remote()
    print(json.dumps(out, indent=2))
    print("\n=== SMOKE", "PASSED" if out.get("ok") else "FAILED",
          f"in {out.get('runtime_s')}s on {out.get('gpu')} ===")
