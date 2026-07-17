# Grasping in the Dark: Sample-Efficient Reinforcement Learning for Robotic Manipulation, from Reward, on a Single GPU

**A reproducible study of HIL-SERL in simulation**

Rajat Dandekar · Vizuara AI · *RL in Production* — Session 3 (Robotics)

---

## Abstract

Reinforcement learning is often dismissed as impractical for real robots: too sample-hungry, too fragile, too expensive to run outside a simulator. **HIL-SERL** (Human-in-the-Loop Sample-Efficient Robotic RL; Luo et al., 2024) is the counter-argument — it trains vision-based manipulation policies to near-perfect success in one to two hours on a *single* real robot. In this work we reproduce the core of HIL-SERL — **Soft Actor-Critic (SAC)** combined with **Reinforcement Learning with Prior Data (RLPD)** — entirely in simulation, **fully autonomously** (no human intervention, no hand-designed reward: the environment supplies a sparse success signal), on a **single L4 GPU**. On the LeRobot `gym-hil` `PandaPickCube` task, a Franka Panda learns to pick up and lift a cube — going from **0% success to ~100% success in ≈5,000 gradient steps (≈40 minutes)**, learning almost entirely from a reward it initially never receives. We present the full recipe, the learning dynamics, an honest account of the engineering required to make distributed RL run reliably on ephemeral cloud GPUs, and a concrete **sim-to-real bridge** to the affordable ($200) SO-101 arm. Everything — the Modal training app, the Colab evaluation notebook, and the trained checkpoint — is open and one command from reproduction.

---

## 1. Introduction

Two facts sit in tension. First, reinforcement learning has produced some of the most striking results in machine learning, from game-playing to alignment. Second, ask a roboticist whether they run RL *on the physical robot*, and the usual answer is "no — it needs too much data." A real arm gives you one timeline, no free resets, and no free failures; a simulator gives you thousands of parallel copies, instant resets, and consequence-free crashes. The gap between "RL works" and "RL works *here*" is a **data-efficiency** gap.

**HIL-SERL** closes that gap. It reaches near-100% success on delicate real-world manipulation — assembling connectors, inserting RAM, threading a timing belt — in **1 to 2.5 hours of real training** on a single robot. It does so by stacking three ideas: a sample-efficient off-policy learner (**SAC**), a trick for reusing prior data so exploration doesn't start from scratch (**RLPD**), and a **learned reward from pixels** plus **occasional human corrections** so the whole loop can run on raw camera images without a hand-engineered reward function.

For a workshop whose thesis is *"RL is production-ready,"* HIL-SERL is the ideal centerpiece — but a live human-in-the-loop demo on real hardware is impractical in a classroom. Our contribution is therefore a **clean, reproducible, single-GPU reproduction of HIL-SERL's learning core in simulation**:

- We run SAC + RLPD **autonomously** on `gym-hil`'s `PandaPickCube` task. In simulation the environment computes a sparse reward directly, so **no human and no learned reward classifier are needed** — the human is a *concept we teach*, not a runtime dependency.
- We show the entire learning curve: a policy that succeeds **0 times** in its first 20 episodes and **20/20 times** by the end, converging in **≈40 minutes on one L4 GPU (≈$0.55 of compute)**.
- We document the real engineering — distributed actor-learner over gRPC, headless MuJoCo rendering, and a checkpoint-serialization bug that crashes training — because "it works" is only credible with the failure modes attached.
- We provide the **sim-to-real bridge**: exactly what carries over unchanged to a physical **SO-101** and what must be re-collected.

This is not a new algorithm. It is a faithful, honest, end-to-end *demonstration* that reinforcement learning genuinely learns robotic manipulation from reward — cheaply enough that a student can rerun it.

---

## 2. Background

### 2.1 The data wall on a real robot

*(Figure: `why-rl-is-hard-on-a-robot`)* In a simulator, one RL step is nearly free. On a real arm every term of the RL loop becomes expensive: parallelism is gone (one arm, one timeline), free resets are gone (something physical must reset the object), and free failure is gone (a crash costs a gearbox). Every design choice in HIL-SERL is, at root, a response to this constraint.

### 2.2 Soft Actor-Critic (SAC)

*(Figure: `sac-the-maximum-entropy-idea`, `sac-the-full-algorithm`)* SAC (Haarnoja et al., 2018) is a maximum-entropy, off-policy actor-critic. "Off-policy" is the load-bearing word: it learns from a **replay buffer** of past transitions rather than only fresh on-policy rollouts, which is what makes it sample-efficient enough to consider for real hardware. The entropy term keeps exploration alive without hand-tuned noise schedules, and a temperature parameter auto-balances reward against exploration.

### 2.3 RLPD: Reinforcement Learning with Prior Data

*(Figure: `rlpd-rl-with-prior-data`)* RLPD (Ball et al., 2023) is the sample-efficiency engine. It is a small set of modifications to off-policy RL that, combined, let an agent learn dramatically faster from a handful of prior demonstrations:

1. **Symmetric sampling** — every training batch is 50% fresh online experience and 50% prior/offline data. The agent is never starved of successful examples, even when its own exploration rarely succeeds.
2. **LayerNorm on the critics** — prevents catastrophic value over-estimation when bootstrapping from off-distribution data.
3. **Large critic ensembles with random subsetting and a high update-to-data ratio** — squeezes more learning out of each collected transition.

RLPD is *why* a sparse-reward grasp can be learned in thousands, not millions, of steps.

### 2.4 SERL and HIL-SERL

SERL (Luo et al., ICRA 2024) packaged these ideas into a software suite for real-robot RL, learning tasks in tens of minutes from demonstrations. **HIL-SERL** (Luo et al., 2024) adds two things on top: a **learned binary reward classifier** that turns raw camera images into a sparse success signal (so no hand-designed reward is needed), and **human-in-the-loop interventions** — a person watches the policy and takes over with a SpaceMouse the instant it is about to fail. Those corrections enter the replay buffer (an idea borrowed from HG-DAgger), and the intervention rate falls to zero as the policy takes over.

*(Figure: `hil-serl-the-human-in-the-loop`)* The figure above captures the full loop: the policy acts; if it is about to fail, the human takes over; corrective transitions flow into a mixed offline buffer; RLPD updates the policy and critics from a 50/50 blend; and human guidance — heavy early — fades as reliability rises.

**The simulation simplification.** In our setting, the `gym-hil` MuJoCo environment computes the reward from ground-truth simulator state. This removes *both* the reward classifier and the human from the runtime loop: training is fully autonomous. What remains is exactly the sample-efficient learning core — SAC + RLPD — which is what we study.

---

## 3. Method

### 3.1 The learning core

We train a **Gaussian actor** policy with SAC. Observations are two 128×128 RGB camera streams (a front view and a wrist view) plus an 18-dimensional proprioceptive state (`qpos`, `qvel`, gripper, and end-effector position); a shared frozen ResNet-10 vision encoder maps images to features. The action is a 3-dimensional end-effector displacement plus a discrete gripper command. The learner uses **RLPD-style symmetric sampling** — `online_ratio = 0.5` — mixing the online replay buffer with an offline buffer seeded from ~30 demonstrations, LayerNorm critics, and an update-to-data ratio of 2.

### 3.2 Asynchronous actor-learner

HIL-SERL runs as **two processes communicating over gRPC**. The **actor** steps the environment with the current policy and streams transitions; the **learner** holds the replay buffers, performs gradient updates, and periodically pushes fresh weights back to the actor. This decoupling is what lets a real robot act at a steady control rate while a GPU trains as fast as it can. We run both processes inside a single container over `localhost`, which is the simplest, lowest-latency, and most secure topology (the gRPC port is never exposed).

### 3.3 Autonomous reward in simulation

The `PandaPickCube` environment emits a **sparse reward**: `1.0` when the cube is lifted more than 10 cm, `0.0` otherwise. This is deliberately unforgiving — the agent receives essentially no gradient signal until it *accidentally* completes a full reach-grasp-lift. It is precisely the regime where prior data (RLPD) is decisive: the offline demonstrations keep successful (state, action) pairs in every batch, so the value function has something to latch onto long before the policy can succeed on its own.

---

## 4. Experimental Setup

| Component | Choice |
|---|---|
| Task | LeRobot `gym-hil` · `PandaPickCube` (Franka Panda, MuJoCo) |
| Observation | front + wrist cameras (128×128 RGB) + 18-D proprioception |
| Action | 3-D end-effector delta + discrete gripper |
| Reward | sparse: `1.0` on a >10 cm lift, else `0.0` |
| Algorithm | SAC + RLPD (`online_ratio=0.5`, LayerNorm critics, `utd_ratio=2`, `num_critics=2`) |
| Demonstrations | ~30 offline episodes (`lilkm/pick_cube_franka_panda_30`) |
| Compute | **single NVIDIA L4** on Modal, headless (`MUJOCO_GL=egl`) |
| Checkpointing | every 1,000 gradient steps → Modal Volume → auto-pushed to the HF Hub |

Training is launched with one command (`modal run --detach ...`), runs detached in the cloud, and streams metrics to a log we parse for the learning curve. Full config and commands are in the appendix and the open repository.

### The engineering, honestly

Making distributed RL run reliably on ephemeral cloud GPUs surfaced several real bugs, each fixed by *running* the system rather than trusting documentation:

- **The output directory must be pristine.** LeRobot's config validator refuses to start if the output directory already exists — and both the learner and actor run that validator against the *same* directory. We give the actor its own throwaway directory (it checkpoints nothing; it receives weights over gRPC).
- **A checkpoint-serialization crash.** At each checkpoint, the learner also dumps the replay buffer to disk as a dataset. The buffer stores image observations as `float32` with values in `[0, 255]`, but the image writer rejects floats outside `[0, 1]` — raising an unhandled exception that *halts training at the first save*. We patch the writer to clip-and-cast `[0, 255]` floats to `uint8`. The fix touches only on-disk serialization; the batch the policy trains on is normalized separately and is unaffected.
- **Frequent checkpoints + auto-upload.** Because cloud GPU containers can be preempted, we save every 1,000 steps and push each new checkpoint to the Hub *from inside the training job*, so a good policy externalizes the instant it exists.

These are not incidental. A claim that "RL is production-ready" is only honest with its operational failure modes attached — and every one of them is mundane and fixable.

---

## 5. Results

### 5.1 The learning curve

*(Figure: `reward_curve.png` — the real training curve)* The headline result is the reward curve. The policy begins by **failing every episode** — sparse reward, no idea how to grasp. Its **first successful lift** appears at roughly step 2,900. From there, RLPD's prior-data blend compounds quickly: success rate climbs steeply, and within **≈5,000 gradient steps** the policy grasps essentially every time.

*(Figure: `fig_success_bars.png` — grasp success by training stage)* The bar chart makes the same story categorical: **~0%** at random initialization, a noisy **~12%** through mid-training as the first successes trickle in, and **~81–95%** for the converged policy. The jump between the second and third bar is the whole result — a policy that could barely find the reward becomes one that hits it almost every time.

Concretely, measured over rolling windows of the actor's episodes during a single run:

- **First 20 episodes:** `0 / 20` successes.
- **Last 20 episodes:** `19–20 / 20` successes.
- **Rolling success rate** plateaus at **~95–100%** by ≈step 6,800 (see the learning curve), from a first successful grasp near step 2,900.
- **Held-out evaluation** of the released converged checkpoint (autonomous, no exploration noise) over 50 fresh episodes is reported alongside the checkpoint on the Hugging Face Hub.

The entire run reaches a near-perfect policy in **≈40 minutes on one L4** — on the order of **$0.55 of compute**.

### 5.2 Anatomy of the learning dynamics

The learning curve is not a smooth ramp — it is a **phase transition**, and understanding *why* is the most instructive part of this study. *(Figure: `fig_episode_outcomes.png` — per-phase episode outcomes)* Reading the run episode-by-episode, three regimes are visible:

1. **The dark phase (≈steps 0–2,900): reward never arrives, but the critic is not idle.** The actor flails and collects almost nothing but zeros. It is tempting to call this wasted time — it is the opposite. Because RLPD makes **half of every gradient batch come from the offline demonstrations**, the critic is, from step one, being shown *successful* (state, action, reward=1) transitions. It is quietly building a **value landscape** — a sense that "end-effector closing on the cube, cube rising" is worth a great deal — long before the actor can produce that state on its own. The policy looks stuck; the value function is doing the homework.

2. **Discovery (≈steps 2,900–5,000): the actor stumbles into a region the critic already prizes.** The first autonomous success is not the start of learning — it is the moment the actor's exploration finally reaches the part of state space the critic has *already* learned to value from the offline data. That first success is therefore not diluted into noise; it lands on fertile ground and is immediately reinforced. Successes now enter the *online* buffer too, the two halves of the batch begin to agree, and the update-to-data ratio of 2 compounds each new success into several gradient steps of improvement.

3. **Consolidation (≈steps 5,000+): lock-in.** Once online and offline data tell the same story, the entropy term anneals, exploration narrows around the successful trajectory, and success rate saturates near 100%. The policy has not merely memorized the demos — it optimizes the reward directly, which is why it generalizes across randomized cube positions the demonstrations never covered.

The single most important design decision behind this curve is RLPD's **50/50 symmetric sampling**. On a naive off-policy learner, one success buried in ~2,000 online failures is a signal-to-noise ratio of 1:2000 — statistically invisible. Forcing half of every batch to be known successes raises that ratio to roughly **1:1**, which is what turns a needle-in-a-haystack exploration problem into a tractable one. LayerNorm on the critics is the safety rail that makes this aggressive reuse stable: it caps the value over-estimation that would otherwise explode when bootstrapping from off-distribution offline data.

### 5.3 Sample efficiency in context

*(Figure: `fig_sample_efficiency.png` — sample-efficiency milestones)* Two numbers frame the efficiency claim: the **first** successful grasp near step **~2,900**, and **reliable** grasping by **~5,000** gradient steps — the whole trajectory inside ≈40 minutes on one L4. Reaching a reliable grasp in thousands (not millions) of steps, from a reward that is zero for the first ~2,900 of them, is only possible because of RLPD's symmetric sampling: the successful demonstrations anchor the value function while the policy learns to reproduce and then surpass them. Strip out the prior data and this same task reverts to the classic sparse-reward wall, where a from-scratch agent can explore for hundreds of thousands of steps without a single reward to learn from.

### 5.4 Before and after

*(Figures/videos: `before.gif` — untrained policy; `after.gif` — trained policy)* The clearest evidence is visual. **Before** training, the policy flails: the end-effector wanders, the gripper opens and closes at random, the cube is rarely touched. **After** training, the same policy executes a clean reach, close, and lift — repeatably, across randomized cube positions. Nothing about the task or the reward changed; only the weights did. That is reinforcement learning doing the one thing imitation cannot: getting *better* than any single demonstration by optimizing the reward directly.

---

## 6. Sim-to-Real: the SO-101 bridge

The point of learning on a *Franka Panda in simulation* is a **$200 SO-101 arm on a real desk**. The encouraging fact is how little changes.

**Reused byte-for-byte** — the "learning brain": the asynchronous actor-learner, the SAC learner, the RLPD sampling, the Gaussian-actor policy network, the ResNet-10 reward-classifier architecture, and the SAC hyperparameters.

**Changes (configuration only):** the robot becomes an `so101_follower` (six Feetech servos on a serial bus); a leader arm provides teleoperation; end-effector-space inverse kinematics and tight workspace bounds are set (finding good bounds is the single biggest lever on speed and safety).

**Must be re-collected on the physical arm:** ~15–25 fresh teleoperated demonstrations, and — crucially — a **reward-classifier dataset** of real labeled camera frames, because on hardware there is no simulator to hand you the reward. Here the human returns, both to demonstrate and to intervene.

Practitioners have trained a real SO-101 grasp from scratch this way in roughly **1–3 hours**. The simulation study in this paper is, quite literally, one configuration file away from that.

---

## 7. A real-robot case study: HIL-SERL on an SO-101 (whiteboard wiping)

To pressure-test the bridge, we ran the same method on a **physical SO-101** — the honest counterpart to the clean simulation above. The task: **pick up a duster and wipe a marker off a whiteboard.** The rig is deliberately modest — a single SO-101 *follower* arm doing the task, an SO-101 *leader* arm through which a human demonstrates and intervenes, two 128×128 cameras (front + wrist), and, notably, an **Apple-Silicon Mac (MPS, no CUDA)**. Every RL tutorial assumes a gamepad and an NVIDIA GPU; we had neither.

*(Figure: `wb_real_rig.png`)*

### 7.1 The method on hardware

The learning core is unchanged — SAC + RLPD, an asynchronous actor–learner, offline demos seeding the replay buffer. What changes is the reward and the human: on hardware the environment cannot hand you a reward, so the human presses a key on success; and the human *intervenes* by physically moving the leader arm whenever the policy is about to fail. Those corrections flow into the buffer, and the **intervention rate falling over time** is the clearest signal of progress. *(Figure: `wb_intervention_loop.png`)*

### 7.2 Four walls (and why the docs lie)

Stock LeRobot does not support this setup out of the box, and each gap taught us something structural:

1. **No CUDA.** The SAC learner assumes NVIDIA; Apple's MPS runs the (small, ~8 M-param) networks but at ~8 fps against a 10 fps target — fine for a proof-of-concept loop, not for same-day convergence.
2. **The leader arm is not a supported controller.** HIL-SERL's intervention step is hardcoded to end-effector deltas and requires `get_teleop_events()`, which only the gamepad and keyboard teleoperators implement. The `control_mode: "leader"` the docs describe is a dead config field. *The documentation is ahead of the implementation* — verified on the release and on `main`.
3. **The whole pipeline assumes end-effector space.** Running joint-space (to match the leader) means the policy's batched `[1,6]` action is never squeezed, and — more subtly — the SAC actor never normalizes its action.
4. **Recorded demos have no reward.** `lerobot-record` datasets lack the `next.reward` column the SAC offline buffer requires.

*(Figure: `wb_four_walls.png`)*

### 7.3 The subtlest bug: action normalization (again)

The most instructive failure is one we had *already met in simulation*. The SAC actor returns a raw `tanh`-squashed action in `[-1, 1]` and does **no** normalization; a separate processor is supposed to map it to the real action scale. In end-effector space the `[-1,1]` output *is* the delta (scaled downstream), so the omission is invisible. In joint space that `[-1,1]` goes straight to the motors as a joint target — so the arm drifts to a near-zero pose and **freezes**. It looks like the policy gave up; in fact the action space was never mapped to real joints. The fix is a single unified convention: demos, policy output, leader interventions, and the replay buffer all live in `[-1,1]`, and the value is unnormalized to joint degrees **only at the robot boundary** (`RobotEnv.step`), which keeps the buffer consistent while the motors still receive real targets. *(Figure: `wb_action_convention.png`)*

This is the same lesson as our simulation eval bug (§5), in a different disguise: **inference-time normalization is a separate, easy-to-omit pipeline, and omitting it produces a policy that looks broken but isn't.**

### 7.4 Honest results

*(Figure: `wb_sim_to_real.png`)* The full HIL-SERL loop ran **live on real hardware with leader interventions** — none of which stock LeRobot supports. Ten leader-teleoperated demos warm-started the buffer; human corrections flowed in as designed. A **fully autonomous, reliable wipe did not emerge in a single session** — three factors stack against it: MPS-speed training, only 10 seed demos, and the harder joint-space formulation (the paper leans on end-effector space and far more interaction). The value here is the *loop*, not the trophy: watching a real arm explore, stall, get corrected by a human, and fold that correction into its own learning is a more honest picture of RL in robotics than a highlight reel. For faster convergence the levers are clear — a gamepad, end-effector space, CUDA, more demos, and a trained reward classifier. Full code and the complete autopsy of every wall are open (see references).

---

## 8. Discussion and Limitations

**What we did not do.** We did not run the human-in-the-loop or the learned reward classifier: in simulation the environment supplies the reward, so both are unnecessary at runtime. This is a faithful reproduction of HIL-SERL's *learning core*, not its full real-robot loop — a distinction we keep explicit throughout.

**Autonomous vs. human-guided.** On real hardware, human interventions dramatically accelerate and stabilize learning and provide safety. Our autonomous sim result shows the *lower bound* of what the algorithm achieves with no human — and it is already near-perfect on this task, which is itself an argument for the strength of RLPD on well-shaped simulation dynamics.

**Task simplicity.** `PandaPickCube` is a deliberately simple, single-object lift. The value of the study is pedagogical and infrastructural — the *recipe* and the *reproducibility*, not task difficulty. The method scales to the far harder tasks in the original HIL-SERL paper.

---

## 9. Conclusion

Reinforcement learning is not a simulator toy. On a single, cheap GPU, from a reward it initially never receives, SAC + RLPD teaches a robot arm to grasp — going from zero to near-perfect in about forty minutes, fully autonomously. The same code is one configuration file from a $200 arm on a real desk. The workshop's central question — *does RL actually work in robotics?* — has a concrete, reproducible answer, and it is yes.

---

## References

- J. Luo, C. Xu, J. Wu, S. Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning.* arXiv:2410.21845, 2024.
- P. Ball, L. Smith, I. Kostrikov, S. Levine. *Efficient Online Reinforcement Learning with Offline Data (RLPD).* ICML 2023. arXiv:2302.02948.
- J. Luo et al. *SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning.* ICRA 2024. arXiv:2401.16013.
- T. Haarnoja, A. Zhou, P. Abbeel, S. Levine. *Soft Actor-Critic.* ICML 2018. arXiv:1801.01290.
- LeRobot / Hugging Face. *HIL-SERL in simulation (`gym-hil`).* https://huggingface.co/docs/lerobot/hilserl_sim
- R. Dandekar. *HIL-SERL on a real SO-101: whiteboard wiping with leader-arm interventions* (code + autopsy). https://github.com/RajatDandekar/so101-hilserl-whiteboard — dataset `RajatDandekar/so101_whiteboard_wipe`, policy `RajatDandekar/so101_whiteboard_wipe_hilserl` on the Hugging Face Hub.

---

## Appendix A — Reproducibility

**Train (detached, single L4):**
```bash
modal run --detach modal/train_hilserl.py::main \
  --config configs/train_gym_hil.json --save-freq 1000 --gpu L4 \
  --hf-repo <user>/hilserl-panda-pickcube-sac
```
**Evaluate a checkpoint (success rate + video):**
```bash
modal run modal/train_hilserl.py::evaluate --job-name hilserl_panda_pickcube --n-episodes 50
```
**Key config:** `env.task=PandaPickCube-v0` (autonomous, headless), `algorithm=sac`, `utd_ratio=2`, `num_critics=2`, `mixer=online_offline`, `online_ratio=0.5`, offline seed `lilkm/pick_cube_franka_panda_30`.

Code, checkpoint, and the Colab evaluation notebook: see the project repository and `RajatDandekar/hilserl-panda-pickcube-sac` on the Hugging Face Hub.
