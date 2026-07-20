# LinkedIn Post — Blog 12: Grasping in the Dark (Robot RL)

## Schedule

- **Date:** Tuesday, September 8, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

A robot arm learned to pick up a cube for 55 cents.

The reward: 1.0 if the cube rises 10 cm, 0.0 otherwise. The agent went 2,900 training steps without receiving it once. Then it went from 0/20 successful episodes to 19/20, in about 40 minutes on one rented L4 GPU.

Blog 12 is live: "Grasping in the Dark." It reproduces the learning core of HIL-SERL (the method behind real robots learning RAM insertion and connector assembly) in simulation, then takes the same code to a physical $200 SO-101 arm.

The whole result hangs on one sampling trick. Early in training, 87.5% of batches contain zero reward signal, so the gradient is statistically invisible. RLPD forces half of every batch to come from 30 old demonstrations. Same simulation afterwards: 6.9%. The critic learns the value of success from data that isn't its own, thousands of steps before the policy ever succeeds.

The real-robot half is the honest part. Four undocumented walls, a leader arm the docs claim is supported but the code never reads, and an arm that froze because its [-1,1] actions were sent to the motors as joint degrees.

Link in comments.

Written while working through the @VizuraAI RL-for-LLMs bootcamp.

#ReinforcementLearning #MachineLearning #Robotics #AI #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/12-robot-rl/

The paper, before/after videos, and the trained checkpoint: https://grasping-in-the-dark.vercel.app

Inside: the soft Bellman target derived and computed by hand, the maximum-entropy objective with a three-line temperature demo, the 50/50 sampling simulation, the checkpoint-serialization bug that kills training at the first save, and the complete challenge-by-challenge autopsy of HIL-SERL on a real SO-101 (no CUDA, no gamepad, leader-arm interventions).

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN
5. Policy Gradients
6. TRPO and PPO
7. RLHF
8. GRPO
9. DPO and Agentic RL
10. Socratic Alignment
11. Dreaming to Dodge (World Models)
12. Grasping in the Dark (this one)

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog12/blog12-social-cover.png` — series-style diagram cover: dark navy with a faint grid, neon nodes (SPARSE REWARD, SAC, RLPD 50/50, 30 DEMOS) converging into a glowing "GRASP ~100%" node, title and subtitle below (recommended hero)
2. **Learning curve**: `marketing/blog12/fig-reward-curve.png` — flat at 0% for ~2,900 steps, first grasp, then a steep climb to ~100% by ~6,800 (the headline figure)
3. **Signal-to-noise**: `marketing/blog12/fig-snr.png` — 0.13 vs 128 successful-episode transitions per 256-batch, log scale: why the grasp is learnable
4. **Success bars**: `marketing/blog12/fig-success-bars.png` — 0% random, 12% mid-training, 81% converged
5. **Sample efficiency**: `marketing/blog12/fig-sample-efficiency.png` — first grasp ~2,926 steps, reliable by ~7,000, all in ~40 minutes on one L4
6. **Blog hero (fallback)**: `blogs/12-robot-rl/images/ai-hero.png` — a robot arm reaching out of darkness toward a glowing cube

Recommended: lead with `blog12-social-cover.png`, or `fig-reward-curve.png` if you want the phase transition front and center. A carousel works well: slide 1 the "55 cents" hook, slide 2 the invisible-reward problem, slide 3 the 50/50 sampling fix, slide 4 the learning curve, final slide the real SO-101 walls plus link.
