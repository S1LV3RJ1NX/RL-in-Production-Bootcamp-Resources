"""Diagnose why the imagination-trained policy underperforms random.

Hypothesis: the world model's done-head does not fire during AUTOREGRESSIVE
imagination (even though its recall on teacher-forced real frames is high), so
every dreamed trajectory "survives" the horizon -> the return is ~constant across
actions -> no gradient to dodge -> the policy collapses to a degenerate action.

Measures, on the round-`round` checkpoints:
  1. imagination death rate: over a batch of dreamed rollouts, what fraction of
     trajectories hit done within the horizon? mean dreamed length?
  2. imagined return spread across the 3 fixed policies (always-noop/left/right):
     if the dream can't tell them apart, there is nothing to learn.
  3. the trained policy's greedy action histogram in the REAL env (what did it
     collapse to?) and its real survival.

Run:  .venv/bin/modal run modal_apps/doom_policy_diag.py --tag doom --round 0
"""
import json
import modal
from common import (app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, load_buffer,
                    data_path, jsonify)


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 15)
def diag(cfg: dict, tag: str = "doom", round_idx: int = 0):
    import numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from actor_critic import ActorCritic, ActorCriticConfig
    from imagination import imagine_rollout, REWARD_VALUES

    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc, pc = cfg["tokenizer"], cfg["world_model"], cfg["policy"]
    env = make_env(cfg, seed=1)
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

    rep = {"tag": tag, "round": round_idx, "horizon": pc["horizon"]}

    # -- 1. imagination death rate under the trained policy -------------------
    frames, _, _, _ = load_buffer(data_path(tag))
    idx = np.random.choice(len(frames), 64, replace=False)
    start = torch.tensor(frames[idx]).permute(0, 3, 1, 2).to(device)
    H = pc["horizon"]
    roll = imagine_rollout(tok, wm, ac, start, H, device, temperature=pc.get("temperature", 1.0))
    dones = roll["dones"]                        # (H, B)
    died = (dones.sum(0) > 0).float().mean().item()
    mean_dones = dones.sum(0).mean().item()
    rep["imagination_policy"] = {
        "frac_trajectories_that_die": round(died, 3),
        "mean_dones_per_traj": round(mean_dones, 3),
        "imagined_return_mean": round(roll["values"][:-1].mean().item(), 2)}

    # -- 2. can the dream tell fixed policies apart? --------------------------
    class _Const:
        def __init__(self, a): self.a = a
        def initial_state(self, b, d): return None
        def __call__(self, frame, state):
            B = frame.shape[0]
            logits = torch.full((B, A), -10.0, device=device); logits[:, self.a] = 10.0
            return logits, torch.zeros(B, device=device), None
        def act(self, frame, state, sample=True, temperature=1.0):
            B = frame.shape[0]
            return torch.full((B,), self.a, device=device, dtype=torch.long), torch.zeros(B, device=device), None
    fixed = {}
    for a in range(A):
        r = imagine_rollout(tok, wm, _Const(a), start, H, device, temperature=pc.get("temperature", 1.0))
        rew = r["rewards"].sum(0).mean().item()          # mean dreamed survived-reward
        d = (r["dones"].sum(0) > 0).float().mean().item()
        fixed[f"always_{a}"] = {"dream_reward_sum": round(rew, 2), "die_frac": round(d, 3)}
    rep["imagination_fixed_actions"] = fixed

    # -- 3. trained policy greedy action histogram + survival in REAL env -----
    hist = np.zeros(A, dtype=int); survs = []
    for ep in range(10):
        e = make_env(cfg, seed=50000 + ep); obs = e.reset(50000 + ep)
        st = ac.initial_state(1, device); done, c = False, 0
        while not done and c < 400:
            with torch.no_grad():
                f = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
                a, _, st = ac.act(f, st, sample=False); a = int(a.item())
            hist[a] += 1; obs, _, done = e.step(a); c += 1
        survs.append(c); e.close()
    rep["real_policy"] = {"action_histogram": hist.tolist(),
                          "action_fractions": [round(float(x / hist.sum()), 3) for x in hist],
                          "survival_steps_mean": round(float(np.mean(survs)), 1)}
    return jsonify(rep)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom", round: int = 0):
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    print(json.dumps(diag.remote(c, tag, round), indent=2))
