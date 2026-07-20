# X Article (Long-Form) — Blog 12: Grasping in the Dark (Robot RL)

## Schedule

- **Date:** Tuesday, September 8, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**Grasping in the Dark: A Robot Arm Learns from a Reward It Almost Never Sees**

---

## Article Body (~1,250 words)

A robot arm learned to pick up a cube for 55 cents.

The reward was brutal: 1.0 if the cube rises more than 10 cm, 0.0 otherwise. For the first 2,900 steps of training the agent never received it. Not once. Its first 20 episodes were 20 failures. Its last 20 were 19 or 20 successes, and the whole run took about 40 minutes on a single rented L4 GPU.

This is a reproduction of the learning core of HIL-SERL, the method that trains real robots to assemble connectors and insert RAM sticks with reinforcement learning in one to 2.5 hours. I ran it in the LeRobot gym-hil simulation, fully autonomously, then took the same code to a physical $200 SO-101 arm and hit four walls the documentation does not mention.

---

### Why nobody runs RL on a real robot

Every RL success story quietly assumes three luxuries: parallel copies of the environment, free resets, and free failure. A physical robot revokes all three. One arm means one timeline. A reset means something physical puts the cube back. A failure can be a stripped gearbox, so "explore randomly" is a repair bill.

That is why most robot learning is imitation: record demonstrations, copy them. Safe, cheap, and capped. A policy trained to copy a demonstrator can at best tie the demonstrator. Nothing in its objective rewards better, only same.

The recipe here is different. SAC, an off-policy learner, so every expensive transition gets reused across thousands of gradient steps. Plus RLPD, a sampling trick that turns 30 old demonstrations into a permanent source of signal.

---

### The trick: half of every batch is someone else's success

Early in training, the online replay buffer holds about 2,000 transitions and exactly one of them carries reward. Sample a 256-batch uniformly and you get 0.13 rewarded transitions on average. I simulated 10,000 draws: 87.5% of batches contain zero reward signal. The learning gradient is statistically invisible.

RLPD's fix is one line of configuration. Keep a second buffer holding ~30 teleoperated demos, and force every batch to be half fresh experience, half demo data:

```python
# RLPD symmetric sampling: 128 online + 128 offline in every batch
n_online = int(batch_size * online_ratio)
batch = sample(online_buffer, n_online) + sample(offline_buffer, 128)
```

Same simulation with the 50/50 mix: only 6.9% of batches carry zero signal, and every batch contains 128 transitions from successful trajectories. The signal-to-noise ratio goes from 1:2000 to roughly 1:1 by construction. A factor of a thousand, from a sampler.

[EMBED IMAGE HERE: fig-snr.png — 0.13 vs 128 successful-episode transitions per batch, log scale]

The consequence shows up in the learning curve as a phase transition, not a ramp. For 2,900 steps the policy looks stuck. It is not. Half of every gradient step has been teaching the critic what success looks like since batch one, so the value function is quietly mapping out which states matter. When exploration finally stumbles into that region, the first success lands on prepared ground and gets compounded immediately.

[EMBED IMAGE HERE: fig-reward-curve.png — flat at 0% for ~2,900 steps, then a steep climb to ~100% by ~6,800]

---

### The bugs that were actually hard

The algorithm is standard. The production story is where it got interesting.

LeRobot runs training as two processes, an actor stepping the sim and a learner doing gradient updates, talking over gRPC. Both validate the same config, and the validator refuses to start if the output directory exists. So whichever process starts second always crashes. The actor got its own throwaway directory.

Worse: at every checkpoint the learner dumps its replay buffer to disk. The buffer stores camera frames as float32 in [0, 255], and the image writer raises on float values above 1.0. Training dies at the first save, every time. The fix is a small patch that clips and casts to uint8, applied at container build time.

And because cloud GPUs get preempted, every new checkpoint gets pushed to the Hugging Face Hub from inside the training job. A good policy externalizes the instant it exists.

None of this is deep. That is the point. The gap between a paper and a working system is made of directory checks and float ranges, not concepts.

---

### Then we put it on a real $200 arm

The sim study has a physical counterpart: the same method on a real SO-101, task: pick up a duster and wipe a marker off a whiteboard. On an Apple Silicon Mac with no CUDA, using a second SO-101 leader arm for human interventions instead of the gamepad every tutorial assumes.

Four walls, in order: MPS runs SAC at 8 fps against a 10 fps control target. The leader arm is not a supported controller, and the config field the docs describe for it is never read in the code. The whole pipeline assumes end-effector space. And recorded demos have no reward column, which the offline buffer requires.

The best bug was the subtlest. The loop ran, the policy emitted actions, and the arm drifted to one pose and froze, like it had given up. The cause: the SAC actor outputs tanh-squashed actions in [-1, 1] and never rescales them. In end-effector space that is invisible, because [-1, 1] is a sensible delta. In joint space it goes to the motors as the joint target in degrees. Every joint was being commanded to within one degree of zero. The policy was fine. Its action space was never translated.

Honest result: the full loop ran live on real hardware with leader-arm interventions, and a fully autonomous wipe did not converge in one session. Ten demos, MPS speed, and joint space all stack against it, and each lever for fixing it is known. The value is the loop, not the trophy.

---

### Who this is for

If you have filed RL under "works in simulators, impractical on hardware," this post is the counterexample with receipts: a sparse-reward manipulation skill learned in 40 minutes for 55 cents, an explanation of exactly why 30 demonstrations change everything, and an unfiltered log of what breaks between the paper and the desk.

The one sentence to keep: sparse rewards are not the enemy, signal starvation is, and keeping known successes in half of every batch is the cheapest cure anyone has found.

Full post with the soft Bellman target derived, the worked examples, the training curve, and the complete SO-101 autopsy: https://prathameshsaraf.com/blogs/12-robot-rl/

The paper, videos, and trained checkpoint: https://grasping-in-the-dark.vercel.app

Learning RL for LLMs through the @VizuraAI bootcamp. The full series, from the Bellman equation to this, is on the same site.

---

## Header Image

- Use **`blog12-x-banner.png`** (this folder) for the article header. Series template: dark navy with a faint grid, neon nodes (SPARSE REWARD, SAC, RLPD 50/50) converging into a glowing "GRASP ~100%" node, title and subtitle on the right.
- Embed `fig-snr.png` in the symmetric-sampling section.
- Embed `fig-reward-curve.png` right after the phase-transition paragraph. This is the headline figure.
- Optional: `fig-success-bars.png` (0% / 12% / 81%) near the results, `fig-sample-efficiency.png` as a compact timeline.
- Fallback header: `ai-hero.png` from `blogs/12-robot-rl/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. The policy failed its first 20 episodes and aced its last 20, because half of every training batch was someone else's success. The critic learned the value of states the policy had never been rewarded in."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "Early in training, 87.5% of batches contain zero reward signal. One config line (sample half of every batch from 30 old demos) drops that to 6.9%. That line is why the robot learns in thousands of steps instead of millions." (recommended: quantified proof plus open loop)
2. "A robot arm went from 0/20 grasps to 19/20 in about 40 minutes on one L4. Total compute cost: about 55 cents."
3. "The arm drifted to one pose and froze, like it had given up. The policy was fine. Its [-1,1] actions were being sent to the motors as joint degrees. The subtlest bugs in robot RL are unit conversions."
4. "The reward was zero for the first 2,900 steps of training. The critic was learning the whole time, from successes that weren't its own. That is the entire trick behind HIL-SERL."

Then reply to anyone who engages, same as the first hour of the original.
