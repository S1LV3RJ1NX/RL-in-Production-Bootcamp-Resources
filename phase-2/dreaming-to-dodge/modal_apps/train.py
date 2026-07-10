"""The IRIS training pipeline on Modal — collect / tokenizer / world-model /
policy-in-imagination.  Each is a Modal function returning plain JSON and
committing the Volume.  Mirrors rlhf-lab/modal_apps/train.py.

Local entrypoints let you run one stage at a time, or `run_all` for a full
round-based loop.  Everything reads/writes the shared Volume under a `tag`
(e.g. "catch") so stages compose.
"""
import json
import modal

from common import (
    app, IMAGE, VOLUMES, VOL_PATH, VOL, GPU_SMALL, GPU_MED,
    data_path, ckpt_path, load_buffer, append_buffer, write_result, jsonify,
)


# ===========================================================================
# 1. collect_data  — roll the real env with the current policy (or random)
# ===========================================================================
@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 30)
def collect_data(cfg: dict, tag: str, round_idx: int = 0):
    import os, time, numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from actor_critic import ActorCritic, ActorCriticConfig

    t0 = time.time()
    VOL.reload()
    seed = cfg.get("seed", 0) * 1000 + round_idx
    env = make_env(cfg, seed=seed)
    n_steps = cfg["collect"]["steps_per_round"]
    eps = cfg["collect"]["eps_greedy"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if a policy checkpoint exists, act with it (epsilon-greedy); else random
    policy = None
    tok = None
    pol_ckpt = ckpt_path(tag, "actor_critic")
    tok_ckpt = ckpt_path(tag, "tokenizer")
    if round_idx > 0 and os.path.exists(pol_ckpt) and os.path.exists(tok_ckpt):
        tcfg = TokenizerConfig(in_channels=3, resolution=cfg["resolution"],
                               ch=cfg["tokenizer"]["ch"],
                               num_tokens=cfg["tokenizer"]["num_tokens"],
                               vocab_size=cfg["tokenizer"]["vocab_size"],
                               embed_dim=cfg["tokenizer"]["embed_dim"])
        tok = Tokenizer(tcfg).to(device).eval()
        tok.load_state_dict(torch.load(tok_ckpt, map_location=device))
        pcfg = ActorCriticConfig(resolution=cfg["resolution"],
                                 num_actions=env.num_actions,
                                 conv_ch=cfg["policy"]["conv_ch"],
                                 hidden_dim=cfg["policy"]["hidden_dim"])
        policy = ActorCritic(pcfg).to(device).eval()
        policy.load_state_dict(torch.load(pol_ckpt, map_location=device))

    # optional: seed exploration with the heuristic oracle (ring vaccination for
    # the epidemic env) so the world model actually OBSERVES contained outbreaks
    # -- otherwise a dense-but-always-negative reward is never seen at its 0
    # (fully-contained) value and the policy has no positive signal to learn from.
    oracle_frac = cfg["collect"].get("oracle_frac", 0.0)
    has_oracle = hasattr(env, "oracle_action")

    frames, actions, rewards, dones = [], [], [], []
    ep_returns, ep_ret = [], 0.0
    obs = env.reset(seed)
    state = policy.initial_state(1, device) if policy else None
    for i in range(n_steps):
        if policy is not None and np.random.rand() > eps:
            with torch.no_grad():
                f = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
                a, _, state = policy.act(f, state, sample=True)
                a = int(a.item())
        elif oracle_frac and has_oracle and np.random.rand() < oracle_frac:
            a = env.oracle_action()          # heuristic demo (ring vaccination)
        else:
            a = int(np.random.randint(env.num_actions))
        nobs, r, done = env.step(a)
        frames.append(obs); actions.append(a); rewards.append(r); dones.append(int(done))
        ep_ret += r
        obs = nobs
        if done:
            ep_returns.append(ep_ret); ep_ret = 0.0
            obs = env.reset(seed + i + 1)
            if policy:
                state = policy.initial_state(1, device)

    frames = np.asarray(frames, dtype=np.float32)
    total = append_buffer(data_path(tag), frames, np.asarray(actions),
                          np.asarray(rewards), np.asarray(dones),
                          cfg["collect"]["buffer_capacity"])
    VOL.commit()
    out = {"tag": tag, "round": round_idx, "collected_steps": n_steps,
           "buffer_size": total, "episodes": len(ep_returns),
           "collect_return_mean": float(np.mean(ep_returns)) if ep_returns else None,
           "used_policy": policy is not None, "runtime_s": round(time.time() - t0, 1)}
    return jsonify(out)


# ===========================================================================
# 2. train_tokenizer — the discrete autoencoder
# ===========================================================================
@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 40)
def train_tokenizer(cfg: dict, tag: str, round_idx: int = 0, data_tag: str = ""):
    import os, time, numpy as np, torch
    from tokenizer import Tokenizer, TokenizerConfig

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, _, _, _ = load_buffer(data_path(data_tag or tag))
    frames_t = torch.tensor(frames).permute(0, 3, 1, 2)         # (N,C,H,W)

    tc = cfg["tokenizer"]
    tcfg = TokenizerConfig(in_channels=3, resolution=cfg["resolution"],
                           ch=tc["ch"], num_tokens=tc["num_tokens"],
                           vocab_size=tc["vocab_size"], embed_dim=tc["embed_dim"],
                           beta=tc["beta"],
                           fg_weight=tc.get("fg_weight", 0.0),
                           warm_weight=tc.get("warm_weight", 0.0),
                           ema=tc.get("ema", True),
                           ema_decay=tc.get("ema_decay", 0.99),
                           revive_dead=tc.get("revive_dead", True),
                           revive_every=tc.get("revive_every", 200),
                           dead_threshold=tc.get("dead_threshold", 1.0))
    model = Tokenizer(tcfg).to(device)
    ckpt = ckpt_path(tag, "tokenizer")
    if os.path.exists(ckpt):                                    # warm-start
        model.load_state_dict(torch.load(ckpt, map_location=device))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])

    n, bs, steps = len(frames_t), tc["batch_size"], tc["steps_per_round"]
    log = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (bs,))
        x = frames_t[idx].to(device)
        out = model(x)
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            log.append({"step": step, "loss": float(out["loss"]),
                        "recon": float(out["recon_loss"]),
                        "recon_mse": float(out["recon_mse"]),
                        "vq": float(out["vq_loss"]),
                        "codebook_usage": float(out["codebook_usage"]),
                        "active_codes": int(out["active_codes"])})

    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    VOL.commit()
    out = {"tag": tag, "round": round_idx, "num_params": model.num_params(),
           "ema": tcfg.ema, "vocab_size": tcfg.vocab_size,
           "final_recon_loss": log[-1]["recon"],
           "final_recon_mse": log[-1]["recon_mse"], "final_vq_loss": log[-1]["vq"],
           "final_codebook_usage": log[-1]["codebook_usage"],
           "final_active_codes": log[-1]["active_codes"],
           "log": log, "runtime_s": round(time.time() - t0, 1)}
    write_result(f"tokenizer_{tag}_r{round_idx}", out)
    VOL.commit()
    return jsonify(out)


# ===========================================================================
# 3. train_world_model — the autoregressive token Transformer
# ===========================================================================
@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 50)
def train_world_model(cfg: dict, tag: str, round_idx: int = 0, data_tag: str = ""):
    import os, time, numpy as np, torch
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from imagination import reward_value_to_class
    from envs import make_env

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, actions, rewards, dones = load_buffer(data_path(data_tag or tag))
    num_actions = make_env(cfg, seed=0).num_actions

    # frozen tokenizer -> pre-encode all frames to tokens (once)
    tc = cfg["tokenizer"]
    tcfg = TokenizerConfig(in_channels=3, resolution=cfg["resolution"], ch=tc["ch"],
                           num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                           embed_dim=tc["embed_dim"])
    tok = Tokenizer(tcfg).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    frames_t = torch.tensor(frames).permute(0, 3, 1, 2)
    all_tokens = []
    with torch.no_grad():
        for i in range(0, len(frames_t), 512):
            all_tokens.append(tok.encode(frames_t[i:i + 512].to(device)).cpu())
    all_tokens = torch.cat(all_tokens)                          # (N,K)

    wc = cfg["world_model"]
    wcfg = WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                            num_actions=num_actions, max_timesteps=wc["max_timesteps"],
                            embed_dim=wc["embed_dim"], num_layers=wc["num_layers"],
                            num_heads=wc["num_heads"], dropout=wc["dropout"])
    wm = WorldModel(wcfg).to(device)
    ckpt = ckpt_path(tag, "world_model")
    if os.path.exists(ckpt):
        wm.load_state_dict(torch.load(ckpt, map_location=device))
    opt = torch.optim.AdamW(wm.parameters(), lr=wc["lr"], weight_decay=0.01)

    # build valid segment start indices: a segment of L steps must not cross an
    # episode boundary in a way that breaks the (frame->next frame) target.
    L = wc["max_timesteps"]
    N = len(all_tokens)
    ends = np.where(dones == 1)[0]
    is_end = np.zeros(N, dtype=bool); is_end[ends] = True
    valid_starts = []
    for s in range(0, N - L):
        # allow a done inside; we just require the window to stay in-buffer
        valid_starts.append(s)
    valid_starts = np.asarray(valid_starts)

    rew_cls_all = reward_value_to_class(torch.tensor(rewards)).long()   # (N,)
    dones_t = torch.tensor(dones).long()
    actions_t = torch.tensor(actions).long()

    bs, steps = wc["batch_size"], wc["steps_per_round"]
    log = []
    wm.train()
    for step in range(steps):
        pick = np.random.choice(valid_starts, bs)
        tok_seq, act_seq, rew_seq, done_seq = [], [], [], []
        for s in pick:
            tok_seq.append(all_tokens[s:s + L])
            act_seq.append(actions_t[s:s + L])
            rew_seq.append(rew_cls_all[s:s + L])
            done_seq.append(dones_t[s:s + L])
        tok_seq = torch.stack(tok_seq).to(device)               # (B,L,K)
        act_seq = torch.stack(act_seq).to(device)               # (B,L)
        rew_seq = torch.stack(rew_seq).to(device)
        done_seq = torch.stack(done_seq).to(device)
        out = wm.compute_loss(tok_seq, act_seq, rew_seq, done_seq,
                              done_pos_weight=wc.get("done_pos_weight", 1.0))
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 10.0)
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log.append({"step": step, "loss": float(out["loss"]),
                        "transition": float(out["transition_loss"]),
                        "reward": float(out["reward_loss"]),
                        "done": float(out["done_loss"]),
                        "token_acc": float(out["token_acc"]),
                        "reward_acc": float(out["reward_acc"]),
                        "done_acc": float(out["done_acc"]),
                        "done_recall": float(out["done_recall"])})

    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(wm.state_dict(), ckpt)
    VOL.commit()
    out = {"tag": tag, "round": round_idx, "num_params": wm.num_params(),
           "final_token_acc": log[-1]["token_acc"],
           "final_reward_acc": log[-1]["reward_acc"],
           "final_done_acc": log[-1]["done_acc"],
           "final_done_recall": log[-1]["done_recall"], "log": log,
           "runtime_s": round(time.time() - t0, 1)}
    write_result(f"world_model_{tag}_r{round_idx}", out)
    VOL.commit()
    return jsonify(out)


# ===========================================================================
# 4. train_policy_in_imagination — actor-critic trained purely in dreams
# ===========================================================================
@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 150)
def train_policy_in_imagination(cfg: dict, tag: str, round_idx: int = 0):
    import os, time, numpy as np, torch
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from actor_critic import ActorCritic, ActorCriticConfig, actor_critic_loss
    from imagination import imagine_rollout
    from envs import make_env

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, _, _, _ = load_buffer(data_path(tag))
    frames_t = torch.tensor(frames).permute(0, 3, 1, 2)
    num_actions = make_env(cfg, seed=0).num_actions

    tc, wc, pc = cfg["tokenizer"], cfg["world_model"], cfg["policy"]
    tcfg = TokenizerConfig(in_channels=3, resolution=cfg["resolution"], ch=tc["ch"],
                           num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                           embed_dim=tc["embed_dim"])
    tok = Tokenizer(tcfg).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    for p in tok.parameters(): p.requires_grad_(False)

    wcfg = WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                            num_actions=num_actions, max_timesteps=wc["max_timesteps"],
                            embed_dim=wc["embed_dim"], num_layers=wc["num_layers"],
                            num_heads=wc["num_heads"], dropout=wc["dropout"])
    wm = WorldModel(wcfg).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))
    for p in wm.parameters(): p.requires_grad_(False)

    pcfg = ActorCriticConfig(resolution=cfg["resolution"], num_actions=num_actions,
                             conv_ch=pc["conv_ch"], hidden_dim=pc["hidden_dim"],
                             gamma=pc["gamma"], lam=pc["lam"],
                             entropy_coef=pc["entropy_coef"])
    ac = ActorCritic(pcfg).to(device)
    ckpt = ckpt_path(tag, "actor_critic")
    if os.path.exists(ckpt):
        ac.load_state_dict(torch.load(ckpt, map_location=device))
    opt = torch.optim.Adam(ac.parameters(), lr=pc["lr"])

    H, bs, updates = pc["horizon"], pc["batch_size"], pc["updates_per_round"]
    # entropy schedule: explore early, commit late (linear anneal start -> end).
    ent_start = pc.get("entropy_start", pc.get("entropy_coef", 0.01))
    ent_end = pc.get("entropy_end", pc.get("entropy_coef", 0.01))
    N = len(frames_t)
    log = []
    ac.train()
    for u in range(updates):
        frac = u / max(1, updates - 1)
        ent_coef = ent_start + (ent_end - ent_start) * frac
        idx = torch.randint(0, N, (bs,))
        start = frames_t[idx].to(device)
        roll = imagine_rollout(tok, wm, ac, start, H, device,
                               temperature=pc["temperature"])
        loss_out = actor_critic_loss(roll["logits"], roll["actions"], roll["values"],
                                     roll["rewards"], roll["dones"], pcfg,
                                     entropy_coef=ent_coef)
        opt.zero_grad(); loss_out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(ac.parameters(), 10.0)
        opt.step()
        if u % max(1, updates // 20) == 0 or u == updates - 1:
            log.append({"update": u, "loss": float(loss_out["loss"]),
                        "policy_loss": float(loss_out["policy_loss"]),
                        "value_loss": float(loss_out["value_loss"]),
                        "entropy": float(loss_out["entropy"]),
                        "entropy_coef": float(ent_coef),
                        "imagined_return": float(loss_out["return_mean"])})
        # periodic checkpoint so a timeout/kill on a long run never loses progress
        # (the actor-critic is small; committing every ~100 updates is cheap).
        if u > 0 and u % 100 == 0:
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save(ac.state_dict(), ckpt)
            VOL.commit()

    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(ac.state_dict(), ckpt)
    VOL.commit()
    out = {"tag": tag, "round": round_idx, "num_params": ac.num_params(),
           "final_imagined_return": log[-1]["imagined_return"],
           "final_entropy": log[-1]["entropy"], "log": log,
           "runtime_s": round(time.time() - t0, 1)}
    write_result(f"policy_{tag}_r{round_idx}", out)
    VOL.commit()
    return jsonify(out)


# ===========================================================================
# Local entrypoints
# ===========================================================================
def _load(cfg_path):
    import yaml
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@app.local_entrypoint()
def collect(cfg: str = "configs/catch.yaml", tag: str = "catch", round: int = 0):
    print(json.dumps(collect_data.remote(_load(cfg), tag, round), indent=2))


@app.local_entrypoint()
def tokenizer(cfg: str = "configs/catch.yaml", tag: str = "catch", round: int = 0):
    out = train_tokenizer.remote(_load(cfg), tag, round)
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=2))


@app.local_entrypoint()
def world_model(cfg: str = "configs/catch.yaml", tag: str = "catch", round: int = 0):
    out = train_world_model.remote(_load(cfg), tag, round)
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=2))


@app.local_entrypoint()
def policy(cfg: str = "configs/catch.yaml", tag: str = "catch", round: int = 0):
    out = train_policy_in_imagination.remote(_load(cfg), tag, round)
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=2))


# ---------------------------------------------------------------------------
# run_all — the full IRIS collect<->train loop, bounded, in one command.
#
# Each round: collect (real env) -> train tokenizer -> train world model ->
# train policy PURELY in imagination.  Every stage is its own Modal function
# call (isolated failures, Volume persists between them), so if a round dies
# the completed checkpoints survive and you can resume.  `rounds` overrides the
# config so you can cap cost from the CLI.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_all(cfg: str = "configs/catch.yaml", tag: str = "catch", rounds: int = 0):
    import time
    c = _load(cfg)
    R = rounds if rounds > 0 else c["collect"]["rounds"]
    summary = {"tag": tag, "rounds": R, "per_round": []}
    t0 = time.time()
    for r in range(R):
        rt0 = time.time()
        print(f"\n########## ROUND {r}/{R-1}  (tag={tag}) ##########")
        col = collect_data.remote(c, tag, r)
        print(f"  [collect]  buffer={col['buffer_size']} "
              f"collect_return={col.get('collect_return_mean')} "
              f"used_policy={col['used_policy']} ({col['runtime_s']}s)")
        tk = train_tokenizer.remote(c, tag, r)
        print(f"  [tokenizer] recon={tk['final_recon_loss']:.4f} "
              f"usage={tk['final_codebook_usage']:.3f} "
              f"active={tk.get('final_active_codes','?')} ({tk['runtime_s']}s)")
        wm = train_world_model.remote(c, tag, r)
        print(f"  [world_model] tok_acc={wm['final_token_acc']:.3f} "
              f"rew_acc={wm['final_reward_acc']:.3f} "
              f"done_acc={wm['final_done_acc']:.3f} ({wm['runtime_s']}s)")
        po = train_policy_in_imagination.remote(c, tag, r)
        print(f"  [policy] imagined_return={po['final_imagined_return']:.4f} "
              f"entropy={po['final_entropy']:.3f} ({po['runtime_s']}s)")
        summary["per_round"].append({
            "round": r, "buffer": col["buffer_size"],
            "collect_return": col.get("collect_return_mean"),
            "recon": tk["final_recon_loss"], "codebook_usage": tk["final_codebook_usage"],
            "active_codes": tk.get("final_active_codes"),
            "wm_token_acc": wm["final_token_acc"], "wm_reward_acc": wm["final_reward_acc"],
            "wm_done_acc": wm["final_done_acc"],
            "imagined_return": po["final_imagined_return"],
            "round_runtime_s": round(time.time() - rt0, 1)})
    summary["total_runtime_s"] = round(time.time() - t0, 1)
    # persist the round-by-round learning curve to the Volume for the figures
    _write_summary.remote(tag, summary)
    print("\n=== run_all DONE ===")
    print(json.dumps(summary, indent=2))


@app.function(image=IMAGE, volumes=VOLUMES, timeout=60 * 5)
def _write_summary(tag: str, summary: dict):
    write_result(f"run_all_{tag}", summary)
    VOL.commit()
    return {"ok": True}


# ===========================================================================
# iris_loop — the full IRIS collect<->train loop as ONE server-side function.
#
# Runs rounds start_round..start_round+rounds-1 IN-PROCESS (via .local()) so a
# single `.spawn()` runs the whole loop on Modal, robust to any local-session
# teardown.  Each round: collect with the current policy (+oracle/random) ->
# re-ground the world model on the NEW data (this is the fix for the policy
# exploiting the WM's blind spots) -> retrain the policy in the updated dream.
# The tokenizer is env-level (fireballs/walls don't change with the policy) so
# we DON'T retrain it each round -- the WM re-encodes with the existing one.
# Writes a per-round result to the Volume as it goes; long timeout; every stage
# commits so progress survives interruption.
# ===========================================================================
@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 60 * 6)
def iris_loop(cfg: dict, tag: str, rounds: int = 2, start_round: int = 1):
    import time
    summary = {"tag": tag, "rounds": rounds, "start_round": start_round, "per_round": []}
    for r in range(start_round, start_round + rounds):
        rt0 = time.time()
        col = collect_data.local(cfg, tag, r)
        wm = train_world_model.local(cfg, tag, r)
        po = train_policy_in_imagination.local(cfg, tag, r)
        row = {"round": r, "buffer": col["buffer_size"],
               "collect_return": col.get("collect_return_mean"),
               "wm_token_acc": wm["final_token_acc"], "wm_done_recall": wm.get("final_done_recall"),
               "imagined_return": po["final_imagined_return"], "policy_entropy": po["final_entropy"],
               "round_runtime_s": round(time.time() - rt0, 1)}
        summary["per_round"].append(row)
        write_result(f"iris_loop_{tag}_r{r}", row)
        VOL.commit()
    write_result(f"iris_loop_{tag}", summary)
    VOL.commit()
    return jsonify(summary)


# ===========================================================================
# k256_build — Track A: build the K=256 finer tokenizer + world model on the
# EXISTING frame buffer (data_tag), server-side in one spawn.
# ===========================================================================
@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 150)
def k256_build(cfg: dict, tag: str = "doom256", data_tag: str = "doom"):
    tk = train_tokenizer.local(cfg, tag, 0, data_tag)
    wm = train_world_model.local(cfg, tag, 0, data_tag)
    out = {"tag": tag, "tok_recon_mse": tk["final_recon_mse"],
           "tok_active_codes": tk["final_active_codes"],
           "wm_token_acc": wm["final_token_acc"],
           "wm_done_recall": wm.get("final_done_recall")}
    write_result(f"k256_build_{tag}", out); VOL.commit()
    return jsonify(out)


# ===========================================================================
# Codebook-collapse ABLATION — train an EMA tokenizer and a plain (no-EMA)
# tokenizer on the SAME buffer, same steps, and compare usage + ball recall.
# This is the paper's central before/after evidence.  Writes its own result
# JSON and does NOT touch the main per-tag checkpoints.
# ===========================================================================
@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 40)
def tokenizer_ablation(cfg: dict, tag: str, steps: int = 2500, seed: int = 0):
    import io, base64, time, numpy as np, torch, imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tokenizer import Tokenizer, TokenizerConfig
    from envs import Catch

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, _, _, _ = load_buffer(data_path(tag))
    frames_t = torch.tensor(frames).permute(0, 3, 1, 2)
    tc = cfg["tokenizer"]

    fgw = tc.get("fg_weight", 25.0) or 25.0

    def make_cfg(ema, fg):
        return TokenizerConfig(in_channels=3, resolution=cfg["resolution"], ch=tc["ch"],
                               num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                               embed_dim=tc["embed_dim"], beta=tc["beta"], ema=ema,
                               fg_weight=fg, revive_dead=tc.get("revive_dead", True),
                               revive_every=tc.get("revive_every", 150))

    def train_one(ema, fg, seed):
        torch.manual_seed(seed)
        model = Tokenizer(make_cfg(ema, fg)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
        n, bs = len(frames_t), tc["batch_size"]
        curve = []
        model.train()
        for step in range(steps):
            idx = torch.randint(0, n, (bs,))
            out = model(frames_t[idx].to(device))
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
            if step % max(1, steps // 25) == 0 or step == steps - 1:
                curve.append({"step": step, "recon_mse": float(out["recon_mse"]),
                              "codebook_usage": float(out["codebook_usage"]),
                              "active_codes": int(out["active_codes"])})
        return model.eval(), curve

    # three conditions isolate the two independent pathologies
    CONDS = [("plain", False, 0.0, "plain VQ + uniform loss"),
             ("ema", True, 0.0, "EMA+revive + uniform loss"),
             ("ema_fg", True, fgw, "EMA+revive + fg-weighted loss")]
    models, curves = {}, {}
    for key, ema, fg, _ in CONDS:
        models[key], curves[key] = train_one(ema, fg, seed)

    # ball-recall panel on frames with KNOWN ball positions
    grid, size = cfg.get("grid", 8), cfg["resolution"]
    cell = size // grid
    env = Catch(grid=grid, size=size, seed=0)
    specs, panel = [], []
    for col in range(grid):
        env.reset(0); env.ball_col = col; env.ball_row = grid // 2; env.paddle = (col + 3) % grid
        panel.append(env.render()); specs.append((grid // 2, col))
    panel = np.asarray(panel, dtype=np.float32)
    x = torch.tensor(panel).permute(0, 3, 1, 2).to(device)

    def recall(model):
        with torch.no_grad():
            xh = model.decode(model.encode(x)).permute(0, 2, 3, 1).cpu().numpy()
        rr = []
        for i, (row, col) in enumerate(specs):
            y0, x0 = row * cell, col * cell
            rr.append(float(np.clip(xh[i, y0:y0 + cell, x0:x0 + cell], 0, 1).mean()))
        return xh, float(np.mean(rr))

    decoded, recalls = {}, {}
    for key, *_ in CONDS:
        decoded[key], recalls[key] = recall(models[key])

    # (1+len) row panel: real + one row per condition. Short horizontal row
    # labels and NO suptitle (the paper figure adds its own caption/title).
    SHORT = {"plain": "plain VQ", "ema": "EMA", "ema_fg": "EMA + fg"}
    ncol = min(8, len(panel))
    sel = list(range(0, len(panel), max(1, len(panel) // ncol)))[:ncol]
    rows = [("real", None)] + [(SHORT[key], key) for key, ema, fg, lab in CONDS]
    fig, axes = plt.subplots(len(rows), len(sel), figsize=(1.5 * len(sel), 1.35 * len(rows)))
    for r, (lab, key) in enumerate(rows):
        rr = "" if key is None else f"  (recall {recalls[key]:.2f})"
        for j, i in enumerate(sel):
            img = panel[i] if key is None else decoded[key][i]
            axes[r, j].imshow(np.clip(img, 0, 1)); axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[r, 0].set_ylabel(lab + rr, fontsize=10, rotation=0, ha="right",
                              va="center", labelpad=8)
    fig.subplots_adjust(left=0.16, wspace=0.05, hspace=0.08)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    panel_b64 = base64.b64encode(buf.getvalue()).decode()

    N = tc["vocab_size"]
    def active_count(key):
        # EMA models report cluster-based active codes; plain VQ has no EMA
        # stats, so fall back to the per-batch unique-code usage.
        n = models[key].quantizer.num_active()
        return n if n > 0 else int(round(curves[key][-1]["codebook_usage"] * N))
    result = {"tag": tag, "steps": steps, "seed": seed, "vocab_size": N, "fg_weight": fgw,
              "conditions": {key: {"label": lab, "ema": ema, "fg_weight": fg,
                                   "final": curves[key][-1], "ball_recall": recalls[key],
                                   "active_codes": active_count(key),
                                   "curve": curves[key]}
                             for key, ema, fg, lab in CONDS},
              "panel_png_b64": panel_b64, "runtime_s": round(time.time() - t0, 1)}
    write_result(f"tokenizer_ablation_{tag}_s{seed}", result)
    write_result(f"tokenizer_ablation_{tag}", result)  # default name for the figure panel
    VOL.commit()
    return jsonify({"seed": seed, **{key: {"recall": round(recalls[key], 4),
                          "active": active_count(key),
                          "usage": round(curves[key][-1]["codebook_usage"], 3)}
                    for key, *_ in CONDS}})


@app.local_entrypoint()
def ablate(cfg: str = "configs/catch_win.yaml", tag: str = "win", steps: int = 2500, seed: int = 0):
    out = tokenizer_ablation.remote(_load(cfg), tag, steps, seed)
    print(json.dumps(out, indent=2))
