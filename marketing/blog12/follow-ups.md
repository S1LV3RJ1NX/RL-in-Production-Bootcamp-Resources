# Daily Follow-ups — Blog 12: Grasping in the Dark (Robot RL)

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Sep 9, then one per day through Follow-up 6 = Mon Sep 14.

Blog link: https://prathameshsaraf.com/blogs/12-robot-rl/
Project link: https://grasping-in-the-dark.vercel.app
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #Robotics #AI #LearningInPublic

---

## Follow-up 1 (Wed Sep 9) — the dark phase (attach fig-reward-curve)

My robot's learning curve was dead flat at 0% for almost 3,000 steps. That flat line was the most productive part of the run.

The reward is sparse: 1.0 if the cube rises 10 cm, nothing otherwise. A random policy essentially never triggers it, so for 2,900 steps the actor collected nothing but zeros. It looks like wasted compute.

It wasn't, because of one sampling rule. Half of every training batch comes from 30 old teleoperated demos, all successful. So the critic had been studying success since batch one, quietly building a value map in which "gripper closing over the cube, cube rising" scores high. The policy looked stuck; the value function was doing the homework. When exploration finally hit that region near step 2,900, the first success landed on prepared ground and compounded. By step 6,800 the arm grasped essentially every time.

The curve is a phase transition, not a ramp, and the flat part is where the learning happened.

What does your training curve hide in its flat sections?

(Attach: fig-reward-curve.png)

Link in comments.

---

## Follow-up 2 (Thu Sep 10) — the 1:2000 problem (attach fig-snr)

Early in this run, one transition out of 2,000 carried a reward. Sample a 256-batch uniformly and you get 0.13 rewarded transitions per batch on average. I simulated 10,000 draws: 87.5% of batches contain zero reward signal.

That is the sparse-reward wall, stated as arithmetic. The gradient is not small, it is absent from seven batches out of eight.

RLPD's answer is symmetric sampling: keep a second buffer with ~30 demonstrations and force every batch to be 128 fresh transitions plus 128 demo transitions. Rerun the simulation: 6.9% of batches are empty of signal, and every batch carries 128 transitions from successful trajectories. Signal-to-noise goes from 1:2000 to about 1:1, guaranteed by the sampler rather than hoped for from exploration.

One config line: online_ratio: 0.5. It is the difference between learning a grasp in thousands of steps and never learning it at all.

Where in your training pipeline are you hoping for signal that you could be guaranteeing instead?

(Attach: fig-snr.png)

Link in comments.

---

## Follow-up 3 (Fri Sep 11) — the bug that kills training at the first save

My robot RL run kept dying at exactly the same step, and the step number was the checkpoint frequency.

At every checkpoint, LeRobot's learner dumps its replay buffer to disk as a dataset. The buffer stores camera frames as float32 with values in [0, 255]. The image writer validates float images and raises if any value exceeds 1.0. So training runs perfectly until the first save, then an unhandled exception kills the whole job. On a rented GPU, that is money evaporating on schedule.

The fix is a ten-line patch applied when the container builds: floats already in [0, 255] get clipped and cast to uint8, true [0, 1] floats keep the original scaling. It touches only on-disk serialization; the batches the policy trains on take a different path entirely.

I found it by running the system, not by reading the docs. The gap between a paper and a working pipeline is made of float ranges and directory checks, and none of them appear in the abstract.

What is the most mundane bug that ever cost you a training run?

Link in comments.

---

## Follow-up 4 (Sat Sep 12) — the arm that gave up

The strangest failure of this project: the robot arm drifted to one pose, settled, and froze. Live loop, policy emitting actions, arm doing nothing. It looked exactly like a policy that had given up.

The logs showed joint targets like -0.46 and 0.63. Degrees. The SAC actor outputs tanh-squashed actions in [-1, 1] and never rescales them; a separate processor is supposed to do that, and the actor loop never builds it. In end-effector mode you never notice, because [-1, 1] is a perfectly good displacement after downstream scaling. In joint space, those same numbers go to the motors as absolute joint targets, so every joint was commanded to within one degree of zero.

Unnormalized, -0.46 becomes -50.6 degrees. The policy was never broken. Its action space was never translated.

The fix: one action convention end to end. Demos, policy outputs, human interventions, and the replay buffer all live in [-1, 1], converted to degrees at exactly one place, the robot boundary. Anything else quietly poisons the buffer.

What convention mismatch has made a working system of yours look broken?

Link in comments.

---

## Follow-up 5 (Sun Sep 13) — the docs said "leader," the code said nothing

I tried to use a leader arm for human interventions in HIL-SERL, the way the LeRobot docs describe. The config field the docs mention, control_mode: "leader", is never read anywhere in the pipeline. I grepped the release and main. It is a dead field. Stock HIL-SERL supports a gamepad or a keyboard, nothing else.

The workaround turned out to be small: a teleop subclass that reads the leader's six joints and returns them normalized as a NumPy array, because one existing code branch accepts a raw array directly as the action. About a hundred lines, LeRobot source untouched.

But the lesson is bigger than the fix. Documentation describes intent; code describes behavior. On the affordable edge of robotics (a $200 arm, a Mac with no CUDA, no gamepad), you will be off the documented path within an hour, and the only reliable manual is the source.

Honest status: the full intervention loop ran live on real hardware. A fully autonomous wipe did not converge in one session, and the post says so, with the reasons.

When did reading the source save you from a documented feature that didn't exist?

Link in comments.

---

## Follow-up 6 (Mon Sep 14) — 55 cents, and where the series stands

Final numbers from blog 12, one week later.

A Franka Panda in simulation learned to grasp from a reward it received zero times in its first 2,900 training steps: 0/20 successes in the first 20 episodes, 19-20/20 in the last 20, about 40 minutes on one L4 GPU, roughly 55 cents of compute. The trained checkpoint is public, and the same code is one configuration file away from a physical arm.

The mechanism, in one line: SAC learns from a replay buffer, RLPD forces half of every batch to come from 30 old demonstrations, so the critic learns what success is worth long before the policy can produce one.

That makes twelve posts: the Bellman equation, value iteration, TD, DQN, policy gradients, PPO, RLHF, GRPO, DPO, a production alignment study, an agent trained inside its own dream, and now a robot arm. The throughline never changed: the value of where I am is the reward I just got plus a discounted value of where I'll land next. This post's addendum: when the reward almost never arrives, borrow the value from someone who already found it.

Which frontier should the series tackle next: multi-task robot policies, offline RL, or RL for agentic LLMs in production?

Link in comments.

---

## Notes

- Vary the opening line when you reuse an angle; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep numbers readable (for example "0.13 vs 128 per batch" and "0/20 to 19/20").
- If a robot-learning paper or a LeRobot/pi0/OpenVLA-style release trends this week, quote-post with "Here's what it takes to run the RL version of this on a $200 arm" plus your link.
