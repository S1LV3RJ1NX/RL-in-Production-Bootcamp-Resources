"""Stage-0 LOCAL smoke test for the SpatialSIR epidemic-containment env.

Zero Modal cost, no torch: this only exercises the pure-numpy environment and
renders it, so we can eyeball the outbreak dynamics + confirm the honest
baseline ordering BEFORE spending anything on GPUs.

It checks / produces:
  1. Env sanity: shapes, reward sign, deterministic-given-seed, episode ends.
  2. Baseline ordering across N seeds (the honest bar):
        no-intervention  <  random  <  ring-vaccination (oracle)
     printed as mean saved-fraction (population never infected) +/- std.
  3. Figures under figures/epidemic_smoke/:
        filmstrip_no_intervention.png / filmstrip_ring.png  (10 frames each)
        outbreak_ring.gif / outbreak_none.gif               (full episodes)

Run with the repo's plotting venv (numpy + imageio present):
    .plotvenv/bin/python smoke_epidemic_local.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs import SpatialSIR

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "epidemic_smoke")
os.makedirs(OUT, exist_ok=True)


def rollout(seed: int, policy: str):
    """Run one episode under a policy; return (saved_fraction, list-of-frames)."""
    env = SpatialSIR(seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    frames = [env.render().copy()]
    total_r = 0.0
    done = False
    while not done:
        if policy == "none":
            a = 1                       # move only, never vaccinate = let it burn
        elif policy == "random":
            a = int(rng.integers(0, env.num_actions))
        elif policy == "ring":
            a = env.oracle_action()     # reactive ring vaccination
        else:
            raise ValueError(policy)
        _, r, done = env.step(a)
        total_r += r
        frames.append(env.render().copy())
    return env.saved_fraction(), total_r, frames


def eval_policy(policy: str, n: int = 200):
    saved, ret = [], []
    for s in range(n):
        sf, tr, _ = rollout(s, policy)
        saved.append(sf)
        ret.append(tr)
    return np.array(saved), np.array(ret)


def upscale(frame, k=4):
    return np.repeat(np.repeat(frame, k, axis=0), k, axis=1)


def filmstrip(frames, n=10, k=3):
    """Evenly sample n frames, upscale, lay side by side with white gaps."""
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    tiles = [upscale(frames[i], k) for i in idx]
    gap = np.ones((tiles[0].shape[0], 3, 3), dtype=np.float32)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [gap, t]
    return np.concatenate(row, axis=1)


def save_png(path, img):
    import imageio.v2 as imageio
    imageio.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


def save_gif(path, frames, k=4, fps=4):
    import imageio.v2 as imageio
    imgs = [(np.clip(upscale(f, k), 0, 1) * 255).astype(np.uint8) for f in frames]
    imageio.mimsave(path, imgs, duration=1.0 / fps, loop=0)


def main():
    print("=" * 68)
    print("Stage-0 smoke — SpatialSIR epidemic-containment env")
    print("=" * 68)

    # --- 1. sanity -------------------------------------------------------
    env = SpatialSIR(seed=0)
    obs = env.reset(0)
    assert obs.shape == (64, 64, 3) and obs.dtype == np.float32, obs.shape
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    o2, r, d = env.step(4)
    assert isinstance(r, float) and isinstance(d, bool)
    # determinism given the seed
    a = SpatialSIR(seed=7); b = SpatialSIR(seed=7)
    for _ in range(5):
        fa, ra, da = a.step(2); fb, rb, db = b.step(2)
        assert np.allclose(fa, fb) and ra == rb and da == db
    # episodes terminate
    e = SpatialSIR(seed=1); steps = 0
    while not e.done and steps < 1000:
        e.step(1); steps += 1
    assert e.done and steps <= e.t_max, steps
    print(f"[ok] shapes/reward/determinism/termination  (episode <= {e.t_max} steps)")

    # --- 2. honest baseline ordering ------------------------------------
    print("\nSaved-fraction (population never infected) over 200 seeds:")
    stats = {}
    for pol in ("none", "random", "ring"):
        saved, ret = eval_policy(pol, n=200)
        stats[pol] = saved.mean()
        print(f"  {pol:>6}:  saved = {saved.mean():.3f} +/- {saved.std():.3f}"
              f"   |  return = {ret.mean():+.3f}")
    ok = stats["none"] < stats["random"] < stats["ring"]
    print(f"\n[{'ok' if ok else 'WARN'}] ordering  none < random < ring  ->  {ok}")
    print(f"      ring vaccination saves {stats['ring'] - stats['none']:+.2f} "
          f"of the population vs letting it burn (the bar to beat from imagination)")

    # --- 3. figures ------------------------------------------------------
    # pick a seed where 'none' clearly burns, for a legible contrast
    seed = 3
    _, _, f_none = rollout(seed, "none")
    _, _, f_ring = rollout(seed, "ring")
    save_png(os.path.join(OUT, "filmstrip_no_intervention.png"), filmstrip(f_none))
    save_png(os.path.join(OUT, "filmstrip_ring.png"), filmstrip(f_ring))
    save_gif(os.path.join(OUT, "outbreak_none.gif"), f_none)
    save_gif(os.path.join(OUT, "outbreak_ring.gif"), f_ring)
    print(f"\n[ok] wrote filmstrips + GIFs to {OUT}")
    print("      (green=susceptible red=infected grey=recovered blue=vaccinated "
          "yellow=responder)")


if __name__ == "__main__":
    main()
