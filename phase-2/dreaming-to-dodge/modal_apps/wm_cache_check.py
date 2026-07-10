"""Correctness + speed check for the KV-cached WorldModel.generate_step_fast.

Equivalence is architectural (independent of trained weights), so we test a
freshly-initialised world model at the doom config's dimensions: with sample=False
(argmax) the cached path must produce IDENTICAL next-frame tokens / reward / done
as the reference generate_step at every position.  Also times both to confirm the
speedup that makes K=64 policy-in-imagination tractable.

Run:  .venv/bin/modal run modal_apps/wm_cache_check.py
"""
import json
import modal
from common import app, IMAGE, VOLUMES, GPU_SMALL, jsonify


@app.function(image=IMAGE, gpu=GPU_SMALL, volumes=VOLUMES, timeout=60 * 10)
def check():
    import time, torch
    from world_model import WorldModel, WorldModelConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = WorldModelConfig(num_tokens=64, vocab_size=512, num_actions=3,
                           max_timesteps=16, embed_dim=256, num_layers=8, num_heads=4)
    wm = WorldModel(cfg).to(device).eval()

    rep = {"device": device, "cases": []}
    all_match = True
    for (B, T) in [(4, 15), (8, 5), (16, 1), (64, 15)]:
        tokens = torch.randint(0, cfg.vocab_size, (B, T, cfg.num_tokens), device=device)
        actions = torch.randint(0, cfg.num_actions, (B, T), device=device)
        g_ref, r_ref, d_ref = wm.generate_step(tokens, actions, sample=False)
        g_fast, r_fast, d_fast = wm.generate_step_fast(tokens, actions, sample=False)
        match = bool(torch.equal(g_ref, g_fast) and torch.equal(r_ref, r_fast)
                     and torch.equal(d_ref, d_fast))
        mism = int((g_ref != g_fast).sum())
        all_match = all_match and match
        rep["cases"].append({"B": B, "T": T, "tokens_match": match,
                             "n_token_mismatch": mism})

    # timing at the imagination working point (B=64, full history)
    B, T = 64, 15
    tokens = torch.randint(0, cfg.vocab_size, (B, T, cfg.num_tokens), device=device)
    actions = torch.randint(0, cfg.num_actions, (B, T), device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    n = 20
    t0 = time.time()
    for _ in range(n):
        wm.generate_step(tokens, actions, sample=True)
    if device == "cuda":
        torch.cuda.synchronize()
    slow = (time.time() - t0) / n
    t0 = time.time()
    for _ in range(n):
        wm.generate_step_fast(tokens, actions, sample=True)
    if device == "cuda":
        torch.cuda.synchronize()
    fast = (time.time() - t0) / n
    rep["ref_step_s"] = round(slow, 4)
    rep["fast_step_s"] = round(fast, 4)
    rep["speedup"] = round(slow / max(fast, 1e-9), 2)
    rep["all_match"] = all_match
    rep["ok"] = all_match
    return jsonify(rep)


@app.local_entrypoint()
def main():
    out = check.remote()
    print(json.dumps(out, indent=2))
    print("\n=== WM CACHE CHECK", "OK" if out["ok"] else "FAILED", "===")
    print(f"  all cases identical: {out['all_match']}")
    print(f"  per generate_step: ref={out['ref_step_s']}s  fast={out['fast_step_s']}s  "
          f"speedup={out['speedup']}x")
