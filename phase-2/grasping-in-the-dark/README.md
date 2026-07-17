# Grasping in the Dark — HIL-SERL (SAC + RLPD) in simulation

> **Phase 2 · Project 04 · Robotics.** Watch reinforcement learning teach a robot arm to grasp —
> from a reward it almost never sees, on a single GPU, in ~40 minutes.

**Project site:** https://grasping-in-the-dark.vercel.app · **Checkpoint:** [`RajatDandekar/hilserl-panda-pickcube-sac`](https://huggingface.co/RajatDandekar/hilserl-panda-pickcube-sac)

We reproduce the learning core of **HIL-SERL** (Luo et al., 2024) — **Soft Actor-Critic** + **RLPD**
(Reinforcement Learning with Prior Data) — entirely in the LeRobot [`gym-hil`](https://github.com/huggingface/gym-hil)
`PandaPickCube` simulation, **fully autonomously** (no human, no reward classifier: the MuJoCo environment supplies a
sparse reward), on a **single L4 GPU**. A Franka Panda goes from **0% to ~100% grasp success in ≈5,000 gradient steps
(≈40 minutes)** — its first successful lift appears around step 2,900, then RLPD's prior-data blend compounds the climb.

The same stack, one config file away, trains a real **$200 SO-101** arm from scratch.

## The expert demonstrations

RLPD samples every training batch **50% online / 50% offline**. The offline half is seeded with **~30 teleoperated
demonstrations from the public dataset [`lilkm/pick_cube_franka_panda_30`](https://huggingface.co/datasets/lilkm/pick_cube_franka_panda_30)** —
the standard seed set shipped with the LeRobot `gym-hil` HIL-SERL example. We did not collect these ourselves; on a real
robot you would record ~15–25 teleop demos and a reward-classifier dataset instead.

## Layout

```
modal/       Modal GPU training + eval app (SAC+RLPD, actor+learner in one container over localhost gRPC)
configs/     the headless-autonomous train config (+ the vendored upstream examples)
notebook/    Colab teaching + eval notebook (load the checkpoint, roll it out, plot the curve)
smoke/       local env-fact smoke test + scripted-grasp recorder (CPU, ~1 GB)
figures/     figure generators — reward curve, analysis bar charts, and the hand-drawn (Gemini) diagrams
website/     the project website (static, deployed to Vercel)
paper/       the write-up (paper.md)
real-so101/  the REAL-hardware experiment — HIL-SERL on a physical SO-101 (whiteboard wiping)
FACTS.md     pinned, source-verified implementation facts · RESOURCES.md  annotated reading list
```

## The real-robot experiment (`real-so101/`)

The sim study above has a physical counterpart: the **same HIL-SERL (SAC + RLPD) method on a real
$200 SO-101 arm**, on Apple Silicon (MPS, no CUDA), using the **leader arm** for human interventions —
the task is to pick up a duster and wipe a marker off a whiteboard. `real-so101/` is the full pipeline
and the complete autopsy of every wall we hit (leader-arm control, joint-space actions, action
normalization, the missing reward column) — each fixed in launcher shims with LeRobot's source
untouched. See `real-so101/README.md` and `real-so101/CHALLENGES.md`. Honest result: the loop runs live
with leader interventions, but a fully autonomous wipe did not converge in one session — the value is
the loop, not (yet) the trophy. Dataset [`RajatDandekar/so101_whiteboard_wipe`](https://huggingface.co/datasets/RajatDandekar/so101_whiteboard_wipe),
policy [`RajatDandekar/so101_whiteboard_wipe_hilserl`](https://huggingface.co/RajatDandekar/so101_whiteboard_wipe_hilserl).

## Reproduce

```bash
pip install modal && modal token new
# 1. train (detached, single L4). save_freq small because cloud GPUs get preempted near convergence.
modal run --detach modal/train_hilserl.py::main \
  --config configs/train_gym_hil.json --save-freq 500 --gpu L4 \
  --hf-repo <you>/hilserl-panda-pickcube-sac
# 2. evaluate a checkpoint (success rate + before/after video)
modal run modal/train_hilserl.py::evaluate --job-name grasp --n-episodes 50
# 3. teach + eval interactively: open notebook/hil_serl_grasp.ipynb in Colab (GPU runtime)
```

**Reading:** HIL-SERL ([arXiv:2410.21845](https://arxiv.org/abs/2410.21845)) · RLPD ([arXiv:2302.02948](https://arxiv.org/abs/2302.02948)) ·
SERL ([arXiv:2401.16013](https://arxiv.org/abs/2401.16013)) · SAC ([arXiv:1801.01290](https://arxiv.org/abs/1801.01290)) ·
[LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl_sim).

Vizuara AI Labs · *RL in Production* — Session 3 (Robotics).
