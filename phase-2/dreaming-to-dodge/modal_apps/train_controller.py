"""Train the tiny controller by CMA-ES, purely in imagination (World Models fix).

Loads the trained (fireball-preserving, grounded) tokenizer + world model for
`tag`, then evolves a ~few-hundred-parameter linear controller to maximise
survival INSIDE THE DREAM.  The whole CMA population is evaluated in ONE batched
dream rollout (N = popsize * episodes streams), so a generation is cheap.  Being
tiny + gradient-free, the controller cannot exploit the world model's flaws the
way the LSTM+REINFORCE policy did — it is forced to learn robust reactive dodging.

Saves the best controller params to ckpt/<tag>/controller.pt (periodically, so a
timeout never loses progress) and evaluates it in the REAL env vs random/oracle,
reporting the action histogram (varied == reactive dodging, the breakthrough).

Deploy + spawn (server-side, robust):
    modal deploy modal_apps/train_controller.py
    then FunctionCall via modal.Function.from_name('iris-wm','train_controller').spawn(cfg, tag, ...)
"""
import json
import modal
from common import (app, IMAGE, VOLUMES, VOL, GPU_SMALL, ckpt_path, load_buffer,
                    data_path, write_result, jsonify)


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 240, memory=16384)
def train_controller(cfg: dict, tag: str = "doom", popsize: int = 24, episodes: int = 6,
                     horizon: int = 64, generations: int = 50, dream_temp: float = 1.15,
                     n_eval: int = 30, hidden: int = 0, ensemble_k: int = 6,
                     run_id: str = "", cma_seed: int = 1, data_tag: str = ""):
    import os, time, numpy as np, torch, cma
    from envs import make_env
    from tokenizer import Tokenizer, TokenizerConfig
    from world_model import WorldModel, WorldModelConfig
    from controller import Reducer, TinyController

    t0 = time.time()
    VOL.reload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc, wc = cfg["tokenizer"], cfg["world_model"]
    env0 = make_env(cfg, seed=0)
    A = env0.num_actions
    max_ctx = wc["max_timesteps"]

    tok = Tokenizer(TokenizerConfig(resolution=cfg["resolution"], ch=tc["ch"],
                    num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    embed_dim=tc["embed_dim"])).to(device).eval()
    tok.load_state_dict(torch.load(ckpt_path(tag, "tokenizer"), map_location=device))
    wm = WorldModel(WorldModelConfig(num_tokens=tc["num_tokens"], vocab_size=tc["vocab_size"],
                    num_actions=A, max_timesteps=max_ctx, embed_dim=wc["embed_dim"],
                    num_layers=wc["num_layers"], num_heads=wc["num_heads"])).to(device).eval()
    wm.load_state_dict(torch.load(ckpt_path(tag, "world_model"), map_location=device))

    reducer = Reducer(rows=3, cols=6, device=device)   # pooled reconstructed-image feature
    ctrl = TinyController(reducer.fdim, A, hidden=hidden)
    D = ctrl.n_params
    print(f"  controller: fdim={reducer.fdim} hidden={hidden} n_params={D}")

    frames, _, _, _ = load_buffer(data_path(data_tag or tag))
    frames = torch.tensor(frames).permute(0, 3, 1, 2)         # (M,3,64,64)

    @torch.no_grad()
    def eval_pop(theta_pop):
        P = theta_pop.shape[0]; E = episodes; N = P * E
        cand = torch.arange(P, device=device).repeat_interleave(E)
        idx = np.random.randint(0, len(frames), N)
        start = frames[idx].to(device)
        tokens_hist = tok.encode(start).unsqueeze(1)          # (N,1,K)
        actions_hist = torch.zeros(N, 1, dtype=torch.long, device=device)
        feat_prev = reducer.features(tok.decode(tokens_hist[:, -1]))
        alive = torch.ones(N, device=device); survival = torch.zeros(N, device=device)
        for t in range(horizon):
            feat_cur = reducer.features(tok.decode(tokens_hist[:, -1]))   # decoder output
            feats = torch.cat([feat_cur, feat_prev], dim=-1)
            action = ctrl.logits_pop(feats, theta_pop, cand).argmax(-1)
            actions_hist[:, -1] = action
            nxt, _, done = wm.generate_step_fast(tokens_hist, actions_hist,
                                                 temperature=dream_temp, sample=True)
            survival += alive
            alive = alive * (1.0 - done.float())
            tokens_hist = torch.cat([tokens_hist, nxt.unsqueeze(1)], dim=1)[:, -max_ctx:]
            actions_hist = torch.cat(
                [actions_hist, torch.zeros(N, 1, dtype=torch.long, device=device)], dim=1)[:, -max_ctx:]
            feat_prev = feat_cur
            if alive.sum() == 0:
                break
        return survival.reshape(P, E).mean(1).cpu().numpy()

    # ---- CMA-ES ----
    def frame_t(obs):
        return torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    @torch.no_grad()
    def recon_feat(obs):   # SAME reconstructed-frame view the dream feeds -> transfers
        return reducer.features(tok.decode(tok.encode(frame_t(obs))))

    @torch.no_grad()
    def eval_real(theta, kind="controller", n=10, seed_base=90000):
        """Roll greedily in the REAL env. Returns (survivals, action_hist).
        seed_base 90000 = selection seeds; 200000 = HELD-OUT seeds for the honest
        final number (must differ from selection to avoid seed-optimistic bias)."""
        survs, hist = [], np.zeros(A, dtype=int)
        th = None if theta is None else theta.to(device)
        for ep in range(n):
            e = make_env(cfg, seed=seed_base + ep); obs = e.reset(seed_base + ep)
            feat_prev = recon_feat(obs); done, c = False, 0
            while not done and c < cfg.get("t_max", 525):
                if kind == "random":
                    a = int(np.random.randint(A))
                elif kind == "oracle":
                    a = e.oracle_action()
                else:
                    feat_cur = recon_feat(obs)
                    feats = torch.cat([feat_cur, feat_prev], dim=-1)
                    a = int(ctrl.logits_single(feats, th).argmax(-1).item())
                    feat_prev = feat_cur
                obs, r, done = e.step(a); hist[a] += 1; c += 1
            survs.append(c); e.close()
        return survs, hist

    # ---- CMA-ES in imagination, collecting a DIVERSE candidate pool ----------
    es = cma.CMAEvolutionStrategy(D * [0.0], 0.5, {"popsize": popsize, "seed": cma_seed, "verbose": -9})
    best_dream, curve, pool = -1.0, [], []
    for gen in range(generations):
        sols = es.ask()
        theta_pop = torch.tensor(np.array(sols), dtype=torch.float32, device=device)
        surv = eval_pop(theta_pop)
        es.tell(sols, [float(-s) for s in surv])
        # pool: the gen's best-dream candidate + 2 random members (search diversity)
        order = np.argsort(-surv)
        keep = [order[0]] + list(np.random.choice(len(sols), 2, replace=False))
        for k in keep:
            pool.append((float(surv[k]), theta_pop[k].clone().cpu()))
        best_dream = max(best_dream, float(surv.max()))
        curve.append({"gen": gen, "dream_best": round(float(surv.max()), 2),
                      "dream_mean": round(float(surv.mean()), 2)})
        if gen % 5 == 0 or gen == generations - 1:
            print(f"  [cma gen {gen}] dream_best={surv.max():.1f} dream_mean={surv.mean():.1f}")

    # ---- REAL-ENV SELECTION: among all dream-trained candidates, keep the one
    #      that actually TRANSFERS (this is model selection on a validation set;
    #      the policy is still 100% dream-trained). Reactive controllers (dream
    #      ~oracle) beat the non-transferring exploiters (high dream, low real). --
    print(f"  two-stage real-selection over {len(pool)} dream-trained candidates ...")
    # stage 1: cheap pass (few episodes) to filter; stage 2: reliable pass on the top
    stage1 = []
    for ds, th in pool:
        s, _ = eval_real(th, "controller", 6)            # selection seeds (90000+)
        stage1.append((float(np.mean(s)), ds, th))
    stage1.sort(key=lambda x: -x[0])
    top = stage1[:20]
    print(f"  stage-1 top real survival {round(top[0][0],1)}; re-evaluating top {len(top)} reliably ...")
    scored = []
    for s1, ds, th in top:
        s, h = eval_real(th, "controller", 24)           # reliable pass, selection seeds
        scored.append((float(np.mean(s)), ds, th, h))
    scored.sort(key=lambda x: -x[0])
    best_real, best_dream_of_sel, best_theta, _ = scored[0]
    n_beat_random = int(sum(1 for s, *_ in scored if s > 74))

    # ---- ENSEMBLE the top-K reactive controllers (average their action votes).
    #      Each single controller dodges well on some fireball patterns and badly
    #      on others (high variance); averaging their votes cancels the idiosyncratic
    #      mistakes and dodges more consistently -> higher, steadier survival. ----
    topK = [th for _, _, th, _ in scored[:ensemble_k]]
    ens = [t.to(device) for t in topK]

    @torch.no_grad()
    def eval_ensemble(thetas, n, seed_base=90000):
        survs, hist = [], np.zeros(A, dtype=int)
        for ep in range(n):
            e = make_env(cfg, seed=seed_base + ep); obs = e.reset(seed_base + ep)
            feat_prev = recon_feat(obs); done, c = False, 0
            while not done and c < cfg.get("t_max", 525):
                feat_cur = recon_feat(obs)
                feats = torch.cat([feat_cur, feat_prev], dim=-1)
                logit_sum = sum(ctrl.logits_single(feats, t) for t in thetas)
                a = int(logit_sum.argmax(-1).item()); feat_prev = feat_cur
                obs, r, done = e.step(a); hist[a] += 1; c += 1
            survs.append(c); e.close()
        return survs, hist

    # decide single-vs-ensemble on SELECTION seeds, then report on HELD-OUT seeds
    use_ens = float(np.mean(eval_ensemble(ens, 16)[0])) >= float(np.mean(eval_real(best_theta, "controller", 16)[0]))
    HELD = 200000
    if use_ens:
        csurv, chist = eval_ensemble(ens, n_eval, seed_base=HELD)
        ssurv, shist = eval_real(best_theta, "controller", n_eval, seed_base=HELD)
        esurv, ehist = csurv, chist
    else:
        ssurv, shist = eval_real(best_theta, "controller", n_eval, seed_base=HELD)
        esurv, ehist = eval_ensemble(ens, n_eval, seed_base=HELD)
        csurv, chist = ssurv, shist

    ckpt = ckpt_path(tag, "controller" + run_id)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save({"theta": best_theta, "thetas": topK, "use_ensemble": use_ens,
                "fdim": reducer.fdim, "A": A, "hidden": hidden,
                "rows": reducer.rows, "cols": reducer.cols}, ckpt); VOL.commit()

    rsurv, _ = eval_real(None, "random", n_eval, seed_base=HELD)
    osurv, _ = eval_real(None, "oracle", min(n_eval, 20), seed_base=HELD)
    fs = cfg.get("frame_skip", 4)
    result = {"tag": tag, "n_params": D, "popsize": popsize, "episodes": episodes,
              "horizon": horizon, "generations": generations, "dream_temp": dream_temp,
              "pool_size": len(pool), "n_candidates_beating_random": n_beat_random,
              "best_dream_survival": round(best_dream, 2),
              "selected_val_survival": round(best_real, 1),
              "selected_dream_survival": round(best_dream_of_sel, 1),
              "used_ensemble": bool(use_ens), "ensemble_k": len(topK),
              "single_survival_mean": round(float(np.mean(ssurv)), 1),
              "ensemble_survival_mean": round(float(np.mean(esurv)), 1),
              "controller_survival_steps": {"mean": round(float(np.mean(csurv)), 1),
                                            "std": round(float(np.std(csurv)), 1),
                                            "max": int(np.max(csurv))},
              "controller_survival_tics": round(float(np.mean(csurv)) * fs, 1),
              "controller_action_fractions": [round(float(x / chist.sum()), 3) for x in chist],
              "random_survival_steps": {"mean": round(float(np.mean(rsurv)), 1),
                                        "max": int(np.max(rsurv))},
              "oracle_survival_steps": {"mean": round(float(np.mean(osurv)), 1),
                                        "max": int(np.max(osurv))},
              "curve": curve, "runtime_s": round(time.time() - t0, 1)}
    write_result(f"controller_{tag}{run_id}", result); VOL.commit()
    return jsonify(result)


@app.local_entrypoint()
def main(cfg: str = "configs/doom.yaml", tag: str = "doom", gens: int = 50):
    import yaml
    with open(cfg) as f:
        c = yaml.safe_load(f)
    out = train_controller.remote(c, tag, generations=gens)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=2))
