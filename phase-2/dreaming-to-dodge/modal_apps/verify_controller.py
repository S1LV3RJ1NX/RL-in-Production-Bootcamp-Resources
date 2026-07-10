"""Adversarial verification of a trained controller on a THIRD, fresh seed set.

The controller was CMA-trained on the dream and model-selected on seeds 90000+,
then reported on held-out seeds 200000+.  To rule out any residual seed luck
before a publishable claim, re-evaluate on a completely fresh set (300000+),
n=50, alongside the random + oracle baselines on the SAME seeds, and report the
action histogram (must stay reactive/varied).

Run (ephemeral, does NOT disturb deployed jobs):
  .venv/bin/modal run modal_apps/verify_controller.py --tag doom --run-id c6 --hidden 16
"""
import json
import modal
from common import app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, jsonify


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 40)
def verify(cfg: dict, tag: str = "doom", run_id: str = "c6", hidden: int = 16,
           seed_base: int = 300000, n: int = 50):
    import numpy as np, torch
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from controller import Reducer, TinyController

    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = cfg["tokenizer"]
    A = make_env(cfg, seed=0).num_actions
    tok = Tokenizer(TokenizerConfig(resolution=cfg["resolution"], ch=tc["ch"],
                    num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))

    cdata = torch.load(ckpt_path(tag, "controller" + run_id), map_location=device)
    hid = cdata.get("hidden", hidden)
    reducer = Reducer(rows=cdata.get("rows", 3), cols=cdata.get("cols", 6), device=device)
    ctrl = TinyController(reducer.fdim, A, hidden=hid)
    thetas = ([t.to(device) for t in cdata["thetas"]] if cdata.get("use_ensemble") and cdata.get("thetas")
              else [cdata["theta"].to(device)])

    def frame_t(obs):
        return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    @torch.no_grad()
    def run(kind):
        survs, hist = [], np.zeros(A, dtype=int)
        for ep in range(n):
            e = make_env(cfg, seed=seed_base + ep); obs = e.reset(seed_base + ep)
            fp = reducer.features(tok.decode(tok.encode(frame_t(obs)))); done, c = False, 0
            while not done and c < cfg.get("t_max", 525):
                if kind == "random":
                    a = int(np.random.randint(A))
                elif kind == "oracle":
                    a = e.oracle_action()
                else:
                    fc = reducer.features(tok.decode(tok.encode(frame_t(obs))))
                    feats = torch.cat([fc, fp], dim=-1)
                    a = int(sum(ctrl.logits_single(feats, t) for t in thetas).argmax(-1).item())
                    fp = fc
                obs, r, done = e.step(a); hist[a] += 1; c += 1
            survs.append(c); e.close()
        return np.array(survs), hist

    cs, ch = run("controller"); rs, _ = run("random"); os_, _ = run("oracle")
    fs = cfg.get("frame_skip", 4)
    ci = 1.96 * float(cs.std()) / np.sqrt(n)   # 95% CI half-width
    out = {"tag": tag, "run_id": run_id, "hidden": hid, "n_ensemble": len(thetas),
           "seed_base": seed_base, "n": n,
           "controller": {"mean": round(float(cs.mean()), 1), "std": round(float(cs.std()), 1),
                          "ci95": round(ci, 1), "max": int(cs.max()), "min": int(cs.min()),
                          "tics_mean": round(float(cs.mean()) * fs, 1)},
           "random": {"mean": round(float(rs.mean()), 1), "max": int(rs.max())},
           "oracle": {"mean": round(float(os_.mean()), 1), "max": int(os_.max())},
           "action_fractions": [round(float(x / ch.sum()), 3) for x in ch],
           "beats_random": bool(cs.mean() > rs.mean()),
           "frac_episodes_over_random_mean": round(float((cs > rs.mean()).mean()), 3),
           "frac_episodes_solved_188": round(float((cs >= 188).mean()), 3)}
    return jsonify(out)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom", run_id: str = "c6", hidden: int = 16):
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    print(json.dumps(verify.remote(c, tag, run_id, hidden), indent=2))
