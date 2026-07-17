"""
Record a REAL successful rollout in gym-hil PandaPickCube to disk (mp4 + gif),
front + wrist cameras side-by-side.

This is a *scripted* reach-grasp-lift controller — i.e. exactly the kind of
DEMONSTRATION that seeds HIL-SERL's offline replay buffer. It is NOT the trained
RL policy (that checkpoint comes from the Modal run); it is honest proof that the
task, the cameras, and the success/reward machinery work end-to-end, and it gives
the deck/notebook a genuine (non-mockup) clip to show.

Run:  .venv/bin/python smoke/record_scripted_grasp.py
Out:  assets/scripted_grasp.mp4, assets/scripted_grasp.gif
"""
import os
import numpy as np
import gymnasium as gym
import imageio
import gym_hil  # noqa: F401

HERE = os.path.dirname(__file__)
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
os.makedirs(ASSETS, exist_ok=True)

ENV_ID = "gym_hil/PandaPickCubeBase-v0"


def obs_parts(obs, env):
    ap = np.asarray(obs["agent_pos"]).reshape(-1)
    tcp = ap[-3:]
    if "environment_state" in obs:                      # state mode exposes cube xyz
        cube = np.asarray(obs["environment_state"]).reshape(-1)[:3]
    else:                                               # image mode: read cube ("block") from the sim
        u = env.unwrapped
        data = getattr(u, "data", None) or getattr(u, "_data")
        cube = np.asarray(data.body("block").xpos).reshape(-1)[:3]
    return cube, tcp


def scripted_action(obs, env):
    cube, tcp = obs_parts(obs, env)
    act_shape = env.action_space.shape
    d = cube - tcp
    a = np.zeros(act_shape, dtype=np.float32)
    if np.linalg.norm(d[:2]) > 0.03:          # 1) center above cube
        a[:2] = np.clip(d[:2] * 4.0, -0.05, 0.05)
        a[2] = np.clip(d[2] * 2.0, -0.02, 0.03)
        a[-1] = -1.0
    elif d[2] < -0.01:                         # 2) descend
        a[2] = np.clip(d[2] * 4.0, -0.05, 0.0)
        a[-1] = -1.0
    else:                                      # 3) close + lift
        a[-1] = 1.0
        a[2] = 0.04
    return a


def side_by_side(px):
    f = np.asarray(px["front"]).astype(np.uint8)
    w = np.asarray(px["wrist"]).astype(np.uint8)
    return np.concatenate([f, w], axis=1)  # (128, 256, 3)


def rollout(seed):
    env = gym.make(ENV_ID, image_obs=True)
    obs, info = env.reset(seed=seed)
    frames, success = [], False
    for _ in range(env.spec.max_episode_steps or 100):
        frames.append(side_by_side(obs["pixels"]))
        act = scripted_action(obs, env)
        obs, rew, term, trunc, info = env.step(act)
        if info.get("succeed") or info.get("is_success") or info.get("success"):
            success = True
        if term or trunc:
            frames.append(side_by_side(obs["pixels"]))
            break
    env.close()
    return frames, success


def hold_last(frames, n=6):
    return frames + [frames[-1]] * n if frames else frames


def main():
    N_EPISODES = 4
    all_frames, first_success = [], None
    seed = 100
    got = 0
    while got < N_EPISODES and seed < 140:
        frames, success = rollout(seed)
        print(f"seed {seed}: {len(frames)} frames, success={success}")
        seed += 1
        if success:
            if first_success is None:
                first_success = frames
            all_frames += hold_last(frames, 6)  # pause on the lifted cube between episodes
            got += 1
    if not all_frames:
        raise SystemExit("no successful rollout recorded")

    mp4 = os.path.join(ASSETS, "scripted_grasp.mp4")
    gif = os.path.join(ASSETS, "scripted_grasp.gif")
    imageio.mimsave(mp4, all_frames, fps=12, quality=8)
    imageio.mimsave(gif, all_frames, fps=10)

    # contact sheet of ONE successful episode so we can eyeball the reach->grasp->lift
    strip = np.concatenate(first_success, axis=1)  # (128, 256*T, 3)
    imageio.imwrite(os.path.join(ASSETS, "scripted_grasp_contactsheet.png"), strip)

    print(f"\nwrote {mp4} ({os.path.getsize(mp4)//1024} KB), {gif} ({os.path.getsize(gif)//1024} KB)")
    print(f"episodes={got}  total_frames={len(all_frames)}  one_episode={len(first_success)} frames  panel=(128,256,3) front|wrist")


if __name__ == "__main__":
    main()
