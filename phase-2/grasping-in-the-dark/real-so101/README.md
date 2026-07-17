# HIL-SERL on a Real SO-101: Whiteboard Wiping with Leader-Arm Interventions

Human-in-the-Loop Sample-Efficient Reinforcement Learning ([HIL-SERL](https://arxiv.org/abs/2410.21845)) on a **single real SO-101 arm**, on **Apple Silicon (MPS, no CUDA)**, using the **leader arm** for human interventions — none of which stock [LeRobot](https://github.com/huggingface/lerobot) supports out of the box.

This repo documents the full pipeline **and** every wall we hit making it work, so you don't have to rediscover them. If you're doing HIL-SERL on a leader/follower arm without a gamepad or an NVIDIA GPU, start here.

> **Task:** pick up a duster and wipe a marker off a whiteboard.
> **Approach:** joint-space SAC, seeded by leader-teleoperated demos, corrected by live leader-arm interventions.

---

## TL;DR — what's non-obvious

1. **Stock LeRobot HIL-SERL does not support the leader arm** as a controller — only gamepad/keyboard. The `control_mode` config field is a dead option; the intervention step is hardcoded to end-effector deltas; the leader lacks `get_teleop_events()`. True in the release **and** on `main`. *(The docs describe leader support that isn't implemented.)*
2. **The SAC actor never normalizes actions.** In EE space that's fine (the `[-1,1]` output is a delta scaled later). In joint space it means the policy output goes straight to the motors unscaled — the arm drifts to ~0 and freezes.
3. **`lerobot-record` datasets have no `next.reward`** — the SAC offline buffer requires it.
4. **All fixes live in launcher shims + one small teleop subclass.** LeRobot's source is never edited.

Full detail in [`CHALLENGES.md`](CHALLENGES.md).

---

## Pipeline overview

```
1. Record demos (leader teleop, stock lerobot-record)   → so101_whiteboard_wipe
2. Crop/resize images to 128x128 (stock crop_dataset_roi) → *_cropped_resized
3. Inject sparse reward (add_reward.py)                   → next.reward column
4. Normalize offline actions to [-1,1] (normalize_offline_actions.py)
5. Train: learner + actor (run_learner.py / run_actor.py) with leader interventions
```

Everything downstream of step 1 keeps actions in normalized `[-1,1]`; they're converted to real joint degrees only at the robot boundary (inside `RobotEnv.step`).

---

## Files

| File | Purpose |
|---|---|
| `record_leader_demos.sh` | Record demos with the leader via stock `lerobot-record` (10 fps, 2 cameras, joint actions). |
| `resume_leader_demos.sh` | Append more demos to an existing dataset (survives USB disconnects). |
| `capture_reset_pose.py` | Read the follower's current joints (calibrated degrees) for `fixed_reset_joint_positions`. |
| `add_reward.py` | Add a sparse `next.reward` (1.0 on each episode's last frame) to the cropped dataset. |
| `normalize_offline_actions.py` | Normalize the offline `action` column from joint degrees → `[-1,1]`. |
| `leader_intervention_teleop.py` | **Custom teleop**: leader outputs normalized joint actions + keyboard events (SPACE/`s`/`r`/`q`). Registered as `so101_leader_intervention`. |
| `run_learner.py` | Launcher for the stock SAC learner + a one-line import so the config parses. |
| `run_actor.py` | Launcher for the stock SAC actor + shims: flatten `[1,6]→[6]`, unnormalize `[-1,1]→degrees` at the robot, and a fail-safe front-cam recorder. |
| `train_config.json` | The full joint-space SAC + real-robot env config. |
| `article_assets/` | Front-cam recordings and stills for the write-up. |
| `substack_article.md` | The narrative write-up. |

---

## Setup

```bash
# From your lerobot checkout, with its venv active:
python -m pip install -e ".[hilserl]"      # grpcio, gym-hil, placo, transformers

# placo on Apple Silicon: fix the urdfdom version mismatch (see CHALLENGES.md)
cd .venv/lib/python3.12/site-packages/cmeel.prefix/lib
for lib in model world sensor; do ln -sf liburdfdom_${lib}.6.0.0.dylib liburdfdom_${lib}.4.0.dylib; done

# URDF for kinematics (only needed for EE-space; joint-space path doesn't use it):
#   TheRobotStudio/SO-ARM100 → Simulation/SO101/so101_new_calib.urdf + assets/*.stl
```

Set your ports/IDs in `train_config.json` and the `.sh` scripts to match your arms
(`lerobot-find-port`, and calibrate with `lerobot-calibrate`).

---

## Running it

```bash
# 1. Record ~10-20 demos with the leader
./record_leader_demos.sh                       # or ./resume_leader_demos.sh to append

# 2. Crop + resize to 128x128 (draw an ROI per camera, press 'c')
python -m lerobot.rl.crop_dataset_roi --repo-id <user>/so101_whiteboard_wipe

# 3. Prep the offline dataset
python add_reward.py
python normalize_offline_actions.py

# 4. Train — two terminals, same config
export PYTHONPATH=$(pwd)
export HF_TOKEN=...                             # only if pushing checkpoints
python run_learner.py --config_path train_config.json   # wait for "Starting learner thread"
python run_actor.py   --config_path train_config.json   # in a second terminal
```

**During the run:**
- The policy drives the follower autonomously.
- **SPACE** = toggle intervention (then move the **leader** to guide the follower).
- **`s`** = mark success (reward 1, ends episode) · **`q`** = fail · **`r`** = redo.
- Watch your **intervention rate** drop over time — that's learning.

macOS: grant your terminal **Accessibility** and **Input Monitoring** permissions, or the keyboard events won't register.

---

## Honest results

The full HIL-SERL loop runs live on real hardware with leader interventions. A fully
autonomous wipe did **not** converge in a single session — MPS-speed training, 10 seed
demos, and joint-space (vs. the paper's end-effector space) all work against convergence.
The value here is a **working, inspectable, interactive RL loop on a real robot**, plus a
complete map of the pitfalls. For faster convergence: use a gamepad + end-effector space +
CUDA + more demos (see [`CHALLENGES.md`](CHALLENGES.md) → "What we'd do differently").

---

## Artifacts

- Demos: [`RajatDandekar/so101_whiteboard_wipe`](https://huggingface.co/datasets/RajatDandekar/so101_whiteboard_wipe)
- Policy: [`RajatDandekar/so101_whiteboard_wipe_hilserl`](https://huggingface.co/RajatDandekar/so101_whiteboard_wipe_hilserl)

## Credits

Built on [LeRobot](https://github.com/huggingface/lerobot) and [SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100).
Method: Luo et al., [*Precise and Dexterous Robotic Manipulation via HIL-RL*](https://arxiv.org/abs/2410.21845), 2024.
