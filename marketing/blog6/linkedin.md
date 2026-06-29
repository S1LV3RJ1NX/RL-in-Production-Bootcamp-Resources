# LinkedIn Post — Blog 6: TRPO & PPO

## Schedule

- **Date:** Tuesday, July 28, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

In supervised learning a bad training step is cheap. The data is fixed, so the next batch pulls you back. In RL a bad step can spiral: the policy picks its own future data, so one reckless update wrecks the policy, the wrecked policy collects wrecked data, and there may be no way back.

Blog 6 is live: "TRPO and PPO: The Largest Safe Step a Policy Can Take."

The gradient hands you a direction. Nobody hands you the step size. That gap is the whole problem this post solves.

What made it click for me:

- Importance sampling lets you reuse a batch instead of throwing it away. One number, the ratio r = pi_new / pi_old, reweights the old data to score a new policy.
- TRPO builds a hard fence, keep the KL between old and new policy under a small budget, and pays for it with heavy second-order optimization re-solved every update.
- PPO throws the fence out and clips that ratio to [0.8, 1.2] instead. Wander past the band and the objective goes flat, zero gradient. A trust region baked into the loss, in a few lines of plain SGD.

Then it runs. PPO swings a pendulum from hanging down to upright, the smoothed return climbing from about -1137 to -704. Turn the clip off and it is the worst, most unstable run on the board.

This is the same algorithm that taught ChatGPT to follow instructions.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #PPO #RLHF #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/06-trpo-ppo/

It builds the step-size fix from scratch: importance sampling to reuse a batch, the surrogate objective you actually climb, TRPO's KL fence, then PPO's clip, and finishes with a runnable PPO that swings up a pendulum plus an ablation that shows which piece does the work.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN
5. Policy Gradients
6. TRPO and PPO (this one)
7. Coming next: RLHF

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog6/blog6-social-cover.png` — series-style diagram cover: dark navy with a faint grid, three neon nodes (SURROGATE, KL FENCE, CLIP) converging into a glowing "SAFE STEP" node, title and subtitle below (recommended hero)
2. **The clip, good action**: `blogs/06-trpo-ppo/images/fig-clip-positive.svg` — the objective rises then goes flat at r = 1 + epsilon, the trust region drawn by the loss itself (signature visual; export to PNG for social)
3. **The clip, bad action**: `blogs/06-trpo-ppo/images/fig-clip-negative.svg` — the safety valve that never stops pulling a bad action back
4. **Pendulum learning curve**: `blogs/06-trpo-ppo/images/fig-pendulum-curve.svg` — PPO climbs from about -1137 to -704 and holds the pendulum upright (the payoff image)
5. **Ablation**: `blogs/06-trpo-ppo/images/fig-ablation.svg` — full PPO is the only curve that climbs; remove the clip and it is the worst and most unstable
6. **Blog hero (fallback)**: `blogs/06-trpo-ppo/images/ai-hero.png`

Recommended: lead with `blog6-social-cover.png`, or use `fig-clip-positive.svg` (exported to PNG) if you want the clip mechanism front and center. A carousel works well here: slide 1 the "a bad RL step can spiral" hook, slides 2-4 the ratio, the KL fence, and the clip, slide 5 the pendulum payoff, final slide the RLHF bridge plus link.
