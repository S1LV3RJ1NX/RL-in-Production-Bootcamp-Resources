# LinkedIn Post — Blog 5: Policy Gradients

## Schedule

- **Date:** Tuesday, July 21, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

Last week I learned a value for every action and picked moves with an argmax. This week I deleted that middleman and trained the policy directly. That one pivot is the foundation under RLHF, PPO, and GRPO.

Blog 5 is live: "Policy Gradients: Learning the Policy Directly, from a Bandit to Actor-Critic."

Value methods learn how good each action is, then act greedily. Policy gradients skip the detour: you parameterize the policy itself and climb expected reward by gradient ascent. The whole thing rests on one trick, the score function, which turns "the gradient of an average I cannot differentiate" into "an average of gradients I can sample."

The part that made it click for me:

- One reward, sampled once, updates all nine of the archer's aiming angles at the same time. The angle it tried goes up, the other eight go down, and the pushes sum to exactly zero.
- Raw policy gradients are painfully noisy. Subtract a baseline (the critic's value of the state) and the variance drops by orders of magnitude. In my runs plain REINFORCE averaged a 4.66 return, while the same method with a baseline reached 9.60.

It ends on Actor-Critic: the actor proposes moves, the critic judges them, and that judgement is the advantage. That is the template every modern method reuses.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #RLHF #LLMs #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/05-policy-gradients/

It builds the whole thing from scratch: a one-state Archer bandit for the pure mechanism, then states and credit assignment, then a runnable Actor-Critic that solves the Archer MDP.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN
5. Policy Gradients (this one)
6. Coming next: TRPO and PPO

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog5/blog5-social-cover.png` — series-style diagram cover: dark navy with a faint grid, three neon nodes (REINFORCE, baseline, Actor-Critic) converging into a glowing optimal policy node, title and subtitle below (recommended hero, still to generate from the series template)
2. **The variance ladder**: `blogs/05-policy-gradients/images/fig-variance-ladder.svg` — gradient variance on a log scale dropping across the three methods (strongest proof image; export to PNG for social)
3. **Greedy returns**: `blogs/05-policy-gradients/images/fig-greedy-returns.svg` — only the variance-reduced methods reach the near-optimal score
4. **One update, nine logits**: `blogs/05-policy-gradients/images/fig-logit-gradients.svg` — one positive bar for the taken angle, eight negative bars that sum to zero
5. **Blog hero (fallback)**: `blogs/05-policy-gradients/images/ai-hero.png`

Recommended: lead with `blog5-social-cover.png` once generated, or use `fig-variance-ladder.svg` (exported to PNG) if you want the variance story front and center. A carousel works well here: slide 1 the "delete the middleman" hook, slides 2-4 the score-function trick, one update touching every logit, and the baseline cutting variance, final slide the Actor-Critic loop plus link.
