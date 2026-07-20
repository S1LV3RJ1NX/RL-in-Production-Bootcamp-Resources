# Teaching a Robot to Wipe a Whiteboard with Human-in-the-Loop RL

### What it actually takes to run HIL-SERL on a real SO-101 arm — the demos, the dead-ends, and the human corrections that make reinforcement learning work

---

Imitation learning has a ceiling. You show a robot 10, 50, 200 demonstrations, it learns to mimic them — and then it fails the moment the world drifts a few centimeters from what it saw. The robot never learns to *recover*, because your demonstrations never contained failure.

Reinforcement learning promises to break that ceiling: let the robot try, fail, and improve from its own experience. But "let a robot explore" on real hardware is terrifying and slow. The breakthrough idea in **HIL-SERL** (Human-in-the-Loop Sample-Efficient RL, Luo et al. 2024) is deceptively simple: *let a human sit in the loop and take over whenever the policy is about to do something dumb.* The robot explores; you correct; those corrections become training data; the robot needs you less and less.

I wanted to see this work on my own hardware — a single [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm — on a task any teacher does a hundred times a day: **pick up a duster and wipe a marker off a whiteboard.**

This is the honest story of getting there. Not the tidy tutorial version — the real one, with the walls I hit and how I got over them. If you're planning to run HIL-SERL on a leader-follower arm without a fancy GPU or a gamepad, this will save you a *lot* of time.

---

## The Setup: One Arm, Two Cameras, One Duster

The hardware is intentionally modest:

- **SO-101 leader + follower arms.** The follower does the task; the leader is how a human demonstrates and intervenes. Both are 6-DOF (5 joints + gripper), driven by Feetech STS3215 servos.
- **Two cameras.** A fixed "front" webcam watching the whiteboard and duster, and a "wrist" camera on the follower's forearm.
- **A Mac mini (Apple Silicon).** No NVIDIA GPU. This matters more than you'd think — every RL tutorial assumes CUDA.
- **[LeRobot](https://github.com/huggingface/lerobot)** as the software stack.

The task: the duster sits on the table, a scribble sits on the whiteboard, and the robot must grasp the duster and wipe the mark. Short-horizon on paper; contact-rich and finicky in practice.

> **Video — the task via teleoperation.** *(front camera, from the recorded dataset)*
> `article_assets/demo_front_clip.mp4`

---

## How HIL-SERL Works

HIL-SERL combines three ingredients, and it's worth understanding each before we watch it break and get fixed.

**1. Offline demonstrations seed the policy.** You record a handful of successful teleoperated episodes. These don't train a policy directly — they fill an *offline replay buffer* that the RL learner samples from alongside fresh online experience. Think of it as a warm start so the robot isn't flailing from pure randomness.

**2. A distributed actor-learner loop.** This is the clever architecture. Two processes run at once:
- The **actor** lives on the robot. It runs the current policy, collects transitions (state → action → reward → next state), and streams them to the learner.
- The **learner** lives wherever your compute is. It runs a **Soft Actor-Critic (SAC)** algorithm — updating a policy network and twin Q-value critics — and periodically pushes fresh weights back to the actor over gRPC.

Decoupling them means the robot never stalls waiting for a gradient step, and the learner never stalls waiting for the robot.

**3. Human interventions.** During the online loop, when the policy heads somewhere unproductive or unsafe, the human takes over — on our rig, by physically moving the **leader arm** — and demonstrates the right thing. Those intervention transitions are gold: they're exactly the corrective, on-distribution data the policy is missing. As the policy improves, you intervene less. Your **intervention rate** dropping over time is the single best signal that learning is working.

The reward can come from a trained vision classifier, or — simplest — from you pressing a key when the task succeeds. We used the manual route.

---

## The Plan Met Reality: Four Walls

Here's where the tutorial ends and the real work begins. I went in expecting to follow LeRobot's HIL-SERL guide step by step. Instead I hit four walls, each of which taught me something about how the system actually fits together.

### Wall 1: No CUDA

LeRobot's SAC learner is written for NVIDIA GPUs. On Apple Silicon you get **MPS**, which *works* but is unvalidated for this code path and noticeably slower. Policy inference on the actor ran at ~8 fps against a 10 fps target. Not fatal — but it sets the tone: everything here is off the beaten path.

**Lesson:** MPS can run the SAC networks (they're small — an 8M-parameter policy with a ResNet-10 vision encoder), but budget for slow training and don't expect same-day convergence.

### Wall 2: The Leader Arm Isn't a Supported Controller

This was the big one. LeRobot's HIL-SERL docs describe using the SO-101 leader arm for teleoperation and intervention. But when I actually wired it up, the environment crashed at startup.

Digging into the code, the reason is structural:

- The intervention step (`InterventionActionProcessor`) is **hardcoded to read end-effector deltas** — `delta_x`, `delta_y`, `delta_z` — from the control device.
- The leader arm outputs **joint positions**, not EE deltas.
- The control device must also implement `get_teleop_events()` (for the success/intervene signals). Only the **gamepad** and **keyboard** teleoperators do. The leader doesn't.
- The `control_mode` config field the docs reference? It isn't wired to anything in this build — it's a dead option.

I confirmed this is true in the released code *and* on the main branch. **The docs are ahead of the implementation.** Stock LeRobot HIL-SERL supports a gamepad or a keyboard — nothing else.

I had no gamepad, and keyboard-driving a 3D arm through a pick-and-place is miserable. So the choice was: buy a gamepad, or make the leader work.

### Wall 3: Everything Is Built for End-Effector Space

The natural fix — "use the leader in joint space, since it outputs joints anyway" — runs into the fact that the *entire* HIL-SERL pipeline assumes **end-effector action space**. Actions are `[dx, dy, dz, gripper]`, converted to joint commands by an inverse-kinematics layer (using a URDF and the `placo` solver).

Running in joint space to match the leader means bypassing that IK layer — and then discovering all the places the pipeline quietly relied on it:

- The policy outputs a **batched** `[1, 6]` action; in EE mode the IK pipeline squeezes it, but in joint space nothing does — so `env.step` crashes indexing it.
- More subtly: **the SAC actor never applies action normalization at all.** In EE mode that's fine, because the `[-1, 1]` policy output *is* the delta, scaled later by a small step size. In joint space, that `[-1, 1]` gets sent straight to the motors as joint targets — so the arm just drifts toward a near-zero pose and freezes. It looks like the policy "gave up." It hadn't; the action space was simply never mapped to real joints.

### Wall 4: Recorded Demos Aren't RL Demos

I recorded the offline demonstrations with LeRobot's stock `lerobot-record` and the leader arm — the workflow I already trusted. But feeding that dataset to the SAC learner failed: the offline replay buffer **requires a `next.reward` column**, and `lerobot-record` doesn't write rewards. (It *can* infer episode boundaries for `done`, but not reward.)

---

## Making It Work: The Minimal Custom Layer

The theme of the fixes: **keep LeRobot's source untouched, add the smallest possible glue in launcher scripts**, and make the action space consistent from end to end. Here's what it took.

**A unified action convention.** Everything — offline demos, the policy's online output, and the leader's interventions — lives in normalized `[-1, 1]`. The normalized value is converted to real joint degrees *only* at the very last moment, inside `RobotEnv.step`, right before the motors move. This keeps the learner's replay buffer internally consistent while the robot still receives real joint targets.

```
normalized ∈ [-1,1]   ── unnormalize ──▶   joint degrees   ──▶   motors
   (buffer, policy,                     (only at robot
    leader, offline)                      boundary)
```

**A custom leader-intervention teleoperator.** A ~120-line subclass of the stock leader that:
- returns the leader's 6 joint positions **normalized to `[-1, 1]` as a NumPy array** — which happens to be exactly the format the stock intervention step accepts as a raw action (bypassing the EE-delta assumption entirely);
- adds a keyboard listener for events: **SPACE** to toggle taking control, **`s`** for success, **`q`**/**`r`** to end/redo an episode;
- reuses the arm's existing calibration by pointing at the right calibration directory.

**Reward injection.** A one-time script adds a sparse `next.reward` to the demo dataset — `1.0` on the last frame of each (successful) episode, `0.0` elsewhere — matching the online reward the human gives with `s`.

**A safety cap.** `max_relative_target` limits how far any joint can move per control step, so an exploring-from-scratch policy can't slam the arm into the table. (Set it too low and human teleop feels sluggish; ~15° per step was the sweet spot on our rig.)

With those pieces, the loop finally ran end to end: the policy explored across a real joint range, and pressing SPACE + moving the leader let me correct it in real time.

> **Video — human intervention during training.** *(front camera, recorded live from the actor)*
> `article_assets/intervention_session_20260717_152733.mp4`

---

## Results: An Honest Accounting

Let me be direct about what this configuration can and can't do.

**What worked:**
- The full HIL-SERL actor-learner loop ran live on a real SO-101, on a Mac, with the leader arm as the intervention device — none of which stock LeRobot supports out of the box.
- Offline demos (10 leader-teleoperated episodes) loaded into the replay buffer and warm-started the policy.
- Human interventions flowed into training exactly as designed: take over with the leader, guide the wipe, hand back control.

**What didn't:**
- A fully autonomous, reliable wipe did **not** emerge in a single session. Three factors stacked against convergence: MPS-speed training, only 10 seed demos, and the inherently harder joint-space action formulation. HIL-SERL's own paper leans heavily on end-effector space and far more interaction.

So is it a failure? No — and this is the important framing for anyone doing this as a teaching demo or a first RL-on-hardware project. **The value is the loop, not (yet) the trophy.** Watching a real arm explore, stall, get corrected by a human, and fold that correction into its own learning — live — is a far more honest picture of "RL in robotics" than a polished highlight reel. The trophy comes with more compute, more demos, and end-effector control; the *understanding* comes from the loop.

---

## What I'd Do Differently

- **Get a gamepad.** A $15 controller is the one thing that makes stock HIL-SERL work end to end, in the well-trodden end-effector-space path, with no custom code. Every hour I spent adapting the leader is an hour a gamepad would have saved. (But I'd have learned far less about the internals.)
- **Use end-effector space** if at all possible. It's what the whole pipeline is tuned for, it learns faster, and its workspace bounds are a real safety net that joint space lacks.
- **Record more demos** — 20-30, with varied duster and marker positions. Ten is thin.
- **Run on CUDA.** MPS is a proof of concept, not a training platform for this.
- **Train a reward classifier** early, so the reward signal doesn't depend on a human's finger on the `s` key.

---

## Key Takeaways

1. **HIL-SERL's real insight is the human, not the algorithm.** Interventions are the lever; SAC is just the engine underneath.
2. **Read the code, not just the docs.** The single biggest time sink was a documented feature (leader control) that isn't implemented. The code is the ground truth.
3. **Action-space consistency is everything.** The subtlest bug — an arm drifting to zero and freezing — was pure normalization mismatch, not a broken policy.
4. **Keep custom code at the edges.** Every adaptation lived in launcher shims and one small teleop subclass; LeRobot's source stayed pristine, which kept the whole thing debuggable.

---

## Resources and Reproducibility

- **Dataset (10 leader-teleop demos):** [`RajatDandekar/so101_whiteboard_wipe`](https://huggingface.co/datasets/RajatDandekar/so101_whiteboard_wipe)
- **Trained policy checkpoint:** [`RajatDandekar/so101_whiteboard_wipe_hilserl`](https://huggingface.co/RajatDandekar/so101_whiteboard_wipe_hilserl)
- **Code + full write-up of the fixes:** [https://github.com/RajatDandekar/so101-hilserl-whiteboard](https://github.com/RajatDandekar/so101-hilserl-whiteboard)
- **Framework:** [LeRobot](https://github.com/huggingface/lerobot) · **Hardware:** [SO-ARM100/101](https://github.com/TheRobotStudio/SO-ARM100)
- **Paper:** Luo et al., *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*, 2024 ([arXiv:2410.21845](https://arxiv.org/abs/2410.21845))

---

*If you're teaching robotics: the most valuable thing you can show a class isn't a robot that already works — it's a robot learning, in real time, with a human in the loop. Even when it's messy. Especially when it's messy.*
