"""The MODEL-FREE baseline for the "Dreaming to Dodge" showdown.

The headline claim of a world model is SAMPLE EFFICIENCY: learn more from the
same amount of real experience.  To make that claim honestly we need a strong,
conventional model-free agent trained on the SAME number of REAL env frames the
IRIS agent's world model was built from -- then compare real-env survival.

This is a faithful Double-DQN on pixels (Nature-CNN, grayscale 4-frame stack,
replay buffer, target network, epsilon anneal, Huber loss) -- no shortcuts, it
simply doesn't get to hallucinate extra experience.  If IRIS (trained purely in
imagination) beats this at equal real frames, the world model earned its keep.

Run (defaults match configs/doom.yaml's 5x40k = 200k real-frame budget):
  .venv/bin/modal run modal_apps/baseline_modelfree.py --steps 200000
"""
import json
import modal

from common import app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, write_result, jsonify


class _FrameStack:
    """Grayscale + last-k frame stack around a DoomTakeCover env (motion cues)."""
    def __init__(self, env, k=4):
        import numpy as np
        self.env, self.k, self.np = env, k, np
        self.num_actions = env.num_actions
        self.buf = None

    def _gray(self, obs):
        return (obs.mean(2) * 255).astype(self.np.uint8)     # (64,64) uint8

    def reset(self, seed=None):
        g = self._gray(self.env.reset(seed))
        self.buf = self.np.stack([g] * self.k, 0)            # (k,64,64)
        return self.buf.copy()

    def step(self, a):
        obs, r, done = self.env.step(a)
        g = self._gray(obs)
        self.buf = self.np.concatenate([self.buf[1:], g[None]], 0)
        return self.buf.copy(), r, done


def _build_qnet(k, num_actions):
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(k, 32, 8, stride=4, padding=2), nn.ReLU(),   # 64->16
        nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 16->8
        nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),  # 8->8
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
        nn.Linear(512, num_actions),
    )


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 150, memory=16384)
def train_dqn(cfg: dict, tag: str = "doom_mf", total_steps: int = 200000,
              eval_every: int = 40000, n_eval: int = 30):
    import os, time, numpy as np, torch, torch.nn.functional as Fnn
    from envs import make_env

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = 4
    env = _FrameStack(make_env(cfg, seed=0), k=K)
    A = env.num_actions

    q = _build_qnet(K, A).to(device)
    tgt = _build_qnet(K, A).to(device)
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-4)

    # replay buffer (uint8 stacks)
    cap = 80000
    S = np.zeros((cap, K, 64, 64), dtype=np.uint8)
    S2 = np.zeros((cap, K, 64, 64), dtype=np.uint8)
    Aa = np.zeros(cap, dtype=np.int64)
    Rr = np.zeros(cap, dtype=np.float32)
    Dd = np.zeros(cap, dtype=np.float32)
    ptr, full = 0, False

    gamma, batch = 0.99, 32
    learn_start, train_every, target_update = 5000, 4, 2000
    eps_start, eps_end, eps_decay = 1.0, 0.05, int(0.5 * total_steps)

    def epsilon(t):
        return max(eps_end, eps_start - (eps_start - eps_end) * t / eps_decay)

    @torch.no_grad()
    def greedy(state):
        x = torch.tensor(state[None], dtype=torch.float32, device=device) / 255.0
        return int(q(x).argmax(1).item())

    def evaluate(n):
        # IMPORTANT: use a SEPARATE env so evaluation never leaves the training
        # env in a finished state (that corrupts the online rollout).
        ev = _FrameStack(make_env(cfg, seed=90000), k=K)
        steps = []
        for ep in range(n):
            s = ev.reset(90000 + ep); done, c = False, 0
            while not done:
                a = greedy(s)
                s, r, done = ev.step(a); c += 1
            steps.append(c)
        ev.env.close()
        return float(np.mean(steps)), float(np.std(steps)), int(np.max(steps))

    log, eval_curve = [], []
    s = env.reset(0)
    ep_len, ep_lens = 0, []
    for t in range(total_steps):
        a = np.random.randint(A) if (t < learn_start or np.random.rand() < epsilon(t)) else greedy(s)
        s2, r, done = env.step(a)
        S[ptr], S2[ptr], Aa[ptr], Rr[ptr], Dd[ptr] = s, s2, a, r, float(done)
        ptr = (ptr + 1) % cap; full = full or ptr == 0
        s = s2; ep_len += 1
        if done:
            ep_lens.append(ep_len); ep_len = 0
            s = env.reset(t + 1)

        if t >= learn_start and t % train_every == 0:
            n = cap if full else ptr
            idx = np.random.randint(0, n, batch)
            bs = torch.tensor(S[idx], dtype=torch.float32, device=device) / 255.0
            bs2 = torch.tensor(S2[idx], dtype=torch.float32, device=device) / 255.0
            ba = torch.tensor(Aa[idx], device=device)
            br = torch.tensor(Rr[idx], device=device)
            bd = torch.tensor(Dd[idx], device=device)
            qv = q(bs).gather(1, ba[:, None]).squeeze(1)
            with torch.no_grad():
                a2 = q(bs2).argmax(1)                          # double DQN
                q2 = tgt(bs2).gather(1, a2[:, None]).squeeze(1)
                target = br + gamma * (1 - bd) * q2
            loss = Fnn.smooth_l1_loss(qv, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
        if t % target_update == 0:
            tgt.load_state_dict(q.state_dict())

        if (t + 1) % eval_every == 0:
            m, sd, mx = evaluate(max(10, n_eval // 2))
            recent = float(np.mean(ep_lens[-50:])) if ep_lens else 0.0
            eval_curve.append({"step": t + 1, "eval_survival_mean": round(m, 1),
                               "eval_survival_max": mx, "train_ep_len_recent": round(recent, 1),
                               "epsilon": round(epsilon(t), 3)})
            print(f"  [dqn t={t+1}] eval_survival={m:.1f} (max {mx})  "
                  f"train_recent={recent:.1f}  eps={epsilon(t):.2f}")

    # final eval + random reference under identical conditions
    fm, fsd, fmx = evaluate(n_eval)
    rnd = []
    for ep in range(n_eval):
        e = make_env(cfg, seed=90000 + ep); e.reset(90000 + ep); d, c = False, 0
        while not d:
            _, _, d = e.step(int(np.random.randint(A))); c += 1
        rnd.append(c); e.close()

    os.makedirs(os.path.dirname(ckpt_path(tag, "dqn")), exist_ok=True)
    torch.save(q.state_dict(), ckpt_path(tag, "dqn"))
    fs = cfg.get("frame_skip", 4)
    result = {"tag": tag, "algo": "double_dqn_pixels", "real_frames": total_steps,
              "n_eval": n_eval, "frame_skip": fs,
              "dqn_survival_steps": {"mean": round(fm, 1), "std": round(fsd, 1), "max": fmx},
              "dqn_survival_tics": round(fm * fs, 1),
              "random_survival_steps": {"mean": round(float(np.mean(rnd)), 1),
                                        "std": round(float(np.std(rnd)), 1),
                                        "max": int(np.max(rnd))},
              "random_survival_tics": round(float(np.mean(rnd)) * fs, 1),
              "eval_curve": eval_curve, "runtime_s": round(time.time() - t0, 1)}
    write_result(f"eval_modelfree_{tag}", result); VOL.commit()
    return jsonify(result)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom_mf", steps: int = 200000):
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    out = train_dqn.remote(c, tag, steps)
    print(json.dumps({k: v for k, v in out.items() if k != "eval_curve"}, indent=2))
    print(f"\n=== MODEL-FREE DQN baseline ({out['real_frames']} real frames) ===")
    print(f"  DQN survival:    {out['dqn_survival_steps']} steps "
          f"({out['dqn_survival_tics']} tics)")
    print(f"  random survival: {out['random_survival_steps']} steps "
          f"({out['random_survival_tics']} tics)")
