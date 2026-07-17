"""Generate the Colab teaching+eval notebook: notebook/hil_serl_grasp.ipynb.

Every code cell here reuses code paths already VERIFIED in this repo:
  - env facts / rendering      -> smoke/smoke_env.py (9 PASS/1 WARN/0 FAIL locally)
  - scripted demonstration     -> smoke/record_scripted_grasp.py (real grasp, reward=1.0)
  - load checkpoint + eval     -> modal/train_hilserl.py::evaluate (ran on Modal, produced video)

Run:  python notebook/build_notebook.py   ->   notebook/hil_serl_grasp.ipynb
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hil_serl_grasp.ipynb")

# The published checkpoint repo (set after `modal run ... --hf-repo <this>` publishes it).
CKPT_REPO = "REPLACE_ME/hilserl-panda-pickcube-sac"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines)
    # nbformat wants a list of lines each ending in \n (except the last)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = [
    md(
        "# The Grasp — watch RL teach a robot arm 🦾",
        "### HIL-SERL (SAC + RLPD) in simulation, on the LeRobot `gym-hil` Franka Panda",
        "",
        "This is the hands-on notebook for **Session 3 (Robotics)** of the *RL in Production* workshop.",
        "We watch **reinforcement learning** teach an arm to **pick up and lift a cube** — learning purely",
        "from a **reward signal**, no human demonstrations of the final skill required.",
        "",
        "**The RL ideas you'll see in action:**",
        "1. **Reward from pixels** — on a real robot HIL-SERL learns a *success classifier* from camera images;",
        "   in **simulation the environment itself supplies the reward**, so training is fully autonomous.",
        "2. **Sample-efficient off-policy RL** — **SAC + RLPD** (50/50 online/offline sampling, LayerNorm",
        "   critics) converges in ~1–2 h instead of days.",
        "3. **Async actor–learner** — the systems architecture that makes real-robot RL practical.",
        "",
        "> The same stack trains a real **SO-101** ($200 arm) from scratch — this sim is one config file away.",
        "",
        "**Runtime:** Colab with a **GPU** runtime (Runtime → Change runtime type → T4 GPU).",
    ),

    md("## 0 · Setup — GPU + headless MuJoCo\n\nMuJoCo needs a GL backend. On Colab's Linux GPU we use **EGL** (set *before* importing mujoco)."),
    code(
        "!nvidia-smi -L || echo 'No GPU — set Runtime → Change runtime type → T4 GPU'",
    ),
    code(
        "import os, subprocess",
        "# Headless EGL for MuJoCo offscreen rendering on Colab's NVIDIA GPU.",
        "subprocess.run('apt-get -qq install -y libgl1-mesa-glx libglfw3 libglew2.2 libegl1-mesa-dev libgles2-mesa-dev ffmpeg', shell=True)",
        "p = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'",
        "os.makedirs(os.path.dirname(p), exist_ok=True)",
        "if not os.path.exists(p):",
        "    open(p, 'w').write('{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libEGL_nvidia.so.0\"}}')",
        "os.environ['MUJOCO_GL'] = 'egl'          # MUST be set before importing mujoco/gym_hil",
        "os.environ['PYOPENGL_PLATFORM'] = 'egl'",
        "print('EGL configured')",
    ),
    code(
        "# Install gym-hil (the sim) + LeRobot with the HIL-SERL extra (SAC/RLPD stack).",
        "# NOTE: heavy deps (mujoco, grpcio, torch) — if Colab asks to RESTART the runtime after this,",
        "# do it, then run every cell from the top EXCEPT skip re-installing.",
        "!pip -q install gym-hil imageio imageio-ffmpeg",
        "!git clone -q --depth 1 https://github.com/huggingface/lerobot /content/lerobot",
        "!pip -q install -e '/content/lerobot[hilserl]'",
        "print('installed — restart the runtime if prompted, then continue from Section 1')",
    ),

    md(
        "## 1 · The task — `PandaPickCube`\n",
        "A 7-DoF Franka Panda must lift a cube by >10 cm. The policy sees two **128×128** camera images",
        "(front + wrist) and an **18-D** proprioceptive state; it outputs a **3-D end-effector delta** plus a",
        "discrete gripper command. Reward is **sparse**: `1.0` on a successful lift, else `0.0`.",
    ),
    code(
        "import os",
        "os.environ.setdefault('MUJOCO_GL', 'egl')",
        "import numpy as np, gymnasium as gym, gym_hil  # gym_hil registers gym_hil/* ids",
        "",
        "env = gym.make('gym_hil/PandaPickCubeBase-v0', image_obs=True)",
        "obs, info = env.reset(seed=0)",
        "print('observation keys :', list(obs.keys()))",
        "print('agent_pos (state):', np.asarray(obs['agent_pos']).shape,   '(qpos7 + qvel7 + gripper1 + tcp3 = 18)')",
        "print('front / wrist img :', obs['pixels']['front'].shape, '/', obs['pixels']['wrist'].shape)",
        "print('action space     :', env.action_space)",
        "assert np.asarray(obs['agent_pos']).shape == (18,)",
        "assert obs['pixels']['front'].shape == (128, 128, 3)",
    ),
    code(
        "# Peek at what the robot sees (front | wrist).",
        "import matplotlib.pyplot as plt",
        "panel = np.concatenate([obs['pixels']['front'], obs['pixels']['wrist']], axis=1)",
        "plt.figure(figsize=(6, 3)); plt.imshow(panel); plt.axis('off'); plt.title('front  |  wrist'); plt.show()",
    ),

    md(
        "## 2 · A demonstration — the kind that *seeds* RL\n",
        "Before showing the learned policy, here's a **scripted** reach-grasp-lift. In HIL-SERL a few such",
        "demonstrations seed the *offline* half of RLPD's 50/50 replay buffer. (This is a hand-coded",
        "controller, **not** the RL policy — that's next.)",
    ),
    code(
        "from IPython.display import HTML",
        "import imageio, base64",
        "",
        "def cube_and_tcp(env, obs):",
        "    tcp = np.asarray(obs['agent_pos']).reshape(-1)[-3:]",
        "    data = getattr(env.unwrapped, 'data', None) or getattr(env.unwrapped, '_data')",
        "    cube = np.asarray(data.body('block').xpos).reshape(-1)[:3]",
        "    return cube, tcp",
        "",
        "def scripted_action(env, obs):",
        "    cube, tcp = cube_and_tcp(env, obs); d = cube - tcp",
        "    a = np.zeros(env.action_space.shape, dtype=np.float32)",
        "    if np.linalg.norm(d[:2]) > 0.03:      a[:2] = np.clip(d[:2]*4, -.05, .05); a[2] = np.clip(d[2]*2, -.02, .03); a[-1] = -1.0",
        "    elif d[2] < -0.01:                    a[2] = np.clip(d[2]*4, -.05, 0.0); a[-1] = -1.0",
        "    else:                                 a[-1] = 1.0; a[2] = 0.04",
        "    return a",
        "",
        "def rollout_frames(policy_fn, seed=100, max_steps=100):",
        "    o, _ = env.reset(seed=seed); frames=[]; succ=False",
        "    for _ in range(max_steps):",
        "        frames.append(np.concatenate([o['pixels']['front'], o['pixels']['wrist']], axis=1))",
        "        o, r, term, trunc, info = env.step(policy_fn(env, o))",
        "        succ = succ or bool(info.get('succeed'))",
        "        if term or trunc: break",
        "    return frames, succ",
        "",
        "def show_video(frames, fps=12):",
        "    imageio.mimsave('/tmp/clip.mp4', frames, fps=fps, quality=8)",
        "    b64 = base64.b64encode(open('/tmp/clip.mp4','rb').read()).decode()",
        "    return HTML(f'<video autoplay loop controls width=420 src=\"data:video/mp4;base64,{b64}\">')",
        "",
        "frames, ok = rollout_frames(scripted_action, seed=100)",
        "print('scripted grasp success:', ok, '| frames:', len(frames))",
        "show_video(frames)",
    ),

    md(
        "## 3 · The trained HIL-SERL policy — load it from the Hub\n",
        "This policy was trained with **SAC + RLPD** on Modal (a single L4 GPU, ~1–2 h) — actor and learner",
        "running asynchronously over gRPC, learning from the environment's sparse reward. We load the",
        "published checkpoint and evaluate it. *(Eval uses the actor's processor pipeline — the shipped",
        "`eval_policy.py` mishandles gym-hil observations.)*",
    ),
    code(
        f"CKPT_REPO = '{CKPT_REPO}'   # <- the published HIL-SERL checkpoint (a pretrained_model dir)",
        "",
        "import torch, draccus",
        "from huggingface_hub import snapshot_download",
        "from lerobot.rl.train_rl import TrainRLServerPipelineConfig",
        "from lerobot.rl import gym_manipulator as gm",
        "from lerobot.processor import TransitionKey",
        "from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy",
        "",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "local = snapshot_download(CKPT_REPO)",
        "cfg = draccus.parse(TrainRLServerPipelineConfig, args=['--config_path', f'{local}/train_config.json'])",
        "cfg.env.task = 'PandaPickCube-v0'          # autonomous, headless",
        "policy = GaussianActorPolicy.from_pretrained(local).to(device).eval()",
        "eval_env, teleop = gm.make_robot_env(cfg.env)",
        "env_proc, act_proc = gm.make_processors(eval_env, teleop, cfg.env, device)",
        "input_keys = list(cfg.policy.input_features.keys())",
        "print('loaded policy on', device, '| inputs:', input_keys)",
    ),
    code(
        "def eval_policy(n_episodes=20, record_first=True):",
        "    successes, frames = [], []",
        "    for ep in range(n_episodes):",
        "        tr = gm.reset_and_build_transition(eval_env, env_proc, act_proc)",
        "        while True:",
        "            obs = {k: v for k, v in tr[TransitionKey.OBSERVATION].items() if k in input_keys}",
        "            with torch.no_grad():",
        "                action = policy.select_action(batch=obs)",
        "            tr = gm.step_env_and_process_transition(env=eval_env, transition=tr, action=action,",
        "                                                    env_processor=env_proc, action_processor=act_proc)",
        "            if record_first and ep == 0:",
        "                r = eval_env.render(); r = r[0] if isinstance(r, (list, tuple)) else r",
        "                frames.append(np.asarray(r).astype(np.uint8))",
        "            reward = float(tr[TransitionKey.REWARD])",
        "            if bool(tr[TransitionKey.DONE]) or bool(tr[TransitionKey.TRUNCATED]):",
        "                successes.append(1.0 if reward > 0 else 0.0); break",
        "    print(f'success rate over {n_episodes} episodes: {np.mean(successes):.0%}')",
        "    return successes, frames",
        "",
        "_, frames = eval_policy(n_episodes=20)",
        "show_video(frames) if frames else None",
    ),

    md(
        "## 4 · How it learned — the reward curve\n",
        "Training logs the episode reward as the actor steps the environment. A healthy run shows reward",
        "climbing from ~0 toward 1.0 as the policy discovers the grasp. (For an autonomous sim run the",
        "human-intervention rate is trivially 0 — the reward alone drives learning.)",
    ),
    code(
        "# The training log is published alongside the checkpoint. Parse 'Global step N: Episode reward: R'.",
        "import re, glob",
        "logs = glob.glob(f'{local}/**/train.log', recursive=True) + glob.glob(f'{local}/train.log')",
        "if logs:",
        "    txt = open(logs[0]).read()",
        "    pts = [(int(s), float(r)) for s, r in re.findall(r'Global step (\\d+): Episode reward: ([-\\d.]+)', txt)]",
        "    if pts:",
        "        xs, ys = zip(*pts)",
        "        # simple moving average to see the trend",
        "        w = max(1, len(ys)//25); sm = np.convolve(ys, np.ones(w)/w, 'valid')",
        "        plt.figure(figsize=(7,3)); plt.plot(xs, ys, alpha=.25, label='episode reward')",
        "        plt.plot(xs[w-1:], sm, lw=2, label=f'moving avg ({w})')",
        "        plt.xlabel('actor env step'); plt.ylabel('reward'); plt.legend(); plt.title('HIL-SERL learning curve'); plt.show()",
        "else:",
        "    print('No train.log in the checkpoint repo — publish it alongside the model to see the curve.')",
    ),

    md(
        "## 5 · Human-in-the-loop — the mechanic\n",
        "On a **real robot**, HIL-SERL lets a human *intervene* with a gamepad when the policy is about to fail;",
        "those corrections enter the replay buffer (HG-DAgger style) and the intervention rate **falls to zero**",
        "as the policy improves. That interface needs a physical controller, so it can't run in a headless",
        "notebook — here's a recording of it on real hardware.",
        "",
        "*(Embed a teleop/intervention clip here — see the HIL-SERL project page: https://hil-serl.github.io )*",
    ),

    md(
        "## 6 · From sim to the real **SO-101** ($200 arm)\n",
        "Everything above — the async actor–learner, SAC+RLPD, the policy — is **reused byte-for-byte** on a",
        "physical SO-101. What changes is only the **env/robot config** (`so101_follower` + cameras + a leader",
        "arm for teleop + EE-space IK bounds). What must be **re-collected** on the real arm: ~15–25 teleop",
        "demonstrations and a **reward-classifier** dataset of labeled camera frames (in sim the env gives the",
        "reward for free). People have trained a real SO-101 grasp from scratch in ~1–3 h this way.",
        "",
        "**Reading:** HIL-SERL paper (arXiv:2410.21845) · RLPD (arXiv:2302.02948) · LeRobot HIL-SERL docs ·",
        "ggando.com/blog/so101-hil-serl · indraneelpatil.github.io/blog/2026/hil-serl",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
