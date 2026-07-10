"""Extra analysis data for the paper's insightful graphs (ephemeral `modal run`).

Produces JSON with: (1) reactivity pairs (fireball horizontal position vs chosen
action) for the trained controller; (2) per-episode survival distributions for
controller / random / oracle; (3) dream-vs-real fidelity (MSE per rollout step).

Run:  .venv/bin/modal run modal_apps/analyze.py --tag doom --run-id c6
"""
import json
import modal
from common import app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, write_result, jsonify


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 30)
def analyze(cfg: dict, tag: str = "doom", run_id: str = "c6", n_eps: int = 40):
    import numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from controller import Reducer, TinyController
    from imagination import compare_to_real

    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc = cfg["tokenizer"], cfg["world_model"]; size = cfg["resolution"]
    A = make_env(cfg, seed=0).num_actions
    tok = Tokenizer(TokenizerConfig(resolution=size, ch=tc["ch"], num_tokens=tc["num_tokens"],
                    vocab_size=tc["vocab_size"], embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"], num_actions=A,
                    max_timesteps=wc["max_timesteps"], embed_dim=wc["embed_dim"], num_layers=wc["num_layers"],
                    num_heads=wc["num_heads"])).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))
    cd = torch.load(ckpt_path(tag, "controller" + run_id), map_location=device)
    reducer = Reducer(rows=cd.get("rows", 3), cols=cd.get("cols", 6), device=device)
    ctrl = TinyController(reducer.fdim, A, hidden=cd.get("hidden", 16))
    thetas = ([t.to(device) for t in cd["thetas"]] if cd.get("use_ensemble") and cd.get("thetas")
              else [cd["theta"].to(device)])

    def ft(obs): return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    def fireball_col(f):
        band = f[size // 3: 7 * size // 8, :, :]
        warm = np.clip(band[:, :, 0] - 0.4 * band[:, :, 2], 0, 1) * band.max(2)
        col = warm.sum(0)
        if col.max() < 1e-3: return None
        return (int(np.argmax(col)) / (size - 1)) * 2 - 1  # normalized [-1(left)..+1(right)]

    # (1) reactivity + (2) controller survival
    react, csurv = [], []
    for ep in range(n_eps):
        e = make_env(cfg, seed=200000 + ep); obs = e.reset(200000 + ep)
        fp = reducer.features(tok.decode(tok.encode(ft(obs)))); done, c = False, 0
        while not done and c < cfg.get("t_max", 525):
            fc = reducer.features(tok.decode(tok.encode(ft(obs))))
            feats = torch.cat([fc, fp], -1)
            a = int(sum(ctrl.logits_single(feats, t) for t in thetas).argmax(-1).item()); fp = fc
            col = fireball_col(obs)
            if col is not None: react.append([round(col, 3), a])
            obs, r, done = e.step(a); c += 1
        csurv.append(c); e.close()

    def baseline(kind):
        s = []
        for ep in range(n_eps):
            e = make_env(cfg, seed=200000 + ep); e.reset(200000 + ep); done, c = False, 0
            while not done and c < cfg.get("t_max", 525):
                a = int(np.random.randint(A)) if kind == "random" else e.oracle_action()
                _, r, done = e.step(a); c += 1
            s.append(c); e.close()
        return s
    rsurv, osurv = baseline("random"), baseline("oracle")

    # (3) dream fidelity vs horizon
    per = []
    for sd in range(16):
        cmp = compare_to_real(make_env(cfg, seed=5000 + sd), tok, wm, device, 24, seed=5000 + sd)
        per.append(cmp["mse_per_step"])
    m = min(len(p) for p in per)
    fidelity = [round(float(np.mean([p[t] for p in per])), 5) for t in range(m)]

    out = {"tag": tag, "run_id": run_id, "n_eps": n_eps,
           "reactivity": react,                       # [ [fireball_col_norm, action], ... ]
           "survival": {"controller": csurv, "random": rsurv, "oracle": osurv},
           "dream_fidelity_mse_per_step": fidelity}
    write_result("analysis_data", out); VOL.commit()
    return jsonify({"n_react": len(react), "ctrl_mean": round(float(np.mean(csurv)), 1),
                    "rand_mean": round(float(np.mean(rsurv)), 1), "oracle_mean": round(float(np.mean(osurv)), 1),
                    "fidelity_len": len(fidelity)})


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom", run_id: str = "c6"):
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    print(json.dumps(analyze.remote(c, tag, run_id), indent=2))
