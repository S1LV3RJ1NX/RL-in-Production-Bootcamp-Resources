# LinkedIn Post — Blog 9: DPO and Agentic RL

## Schedule

- **Date:** Tuesday, August 18, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

RLHF trains a reward model on human preference pairs, then runs PPO against it. DPO's authors looked at that pipeline and asked an uncomfortable question: the reward model was trained only on the pairs, so what does it know that the pairs don't?

Nothing. That answer deletes half the stack.

Blog 9 is live, the series finale: "DPO and Agentic RL: Align Without a Reward Model, Then Step Into the World."

What made it click for me:

- The policy is its own reward model. Score an answer by beta * log(pi/pi_ref), how much more likely the policy makes it than the frozen reference did. That quantity drops straight out of the RLHF objective; the awkward normalizer cancels because preferences always compare two answers to the same prompt.
- The gradient weight is the stability. DPO pushes hardest on pairs it currently mis-ranks and eases off once a pair is confidently correct. Delete that weight and you get runaway drift.
- SimPO drops the reference model, KTO drops the paired data. One loss, three flavors.
- Then the expansion: point the same gradient at a multi-step world and you get agents. A 40-line toy loop learns to call a search tool from reward alone, no demonstration anywhere. P(search) goes from 0.17 to 0.998.

The bridge between the two halves is one sentence: DPO is offline, so the moment a reward must be earned by acting, you need rollouts again.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #LLMs #RLHF #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/09-dpo-and-agentic-rl/

It derives the DPO loss from the RLHF objective in three moves, works one real preference pair through every number (margin +0.12, loss 0.635), builds SimPO and KTO from DPO's leftover problems, then switches gears: the agentic MDP, DeepEyes learning to zoom with pure RL, the observation-masking rule that catches the most common agentic-RL bug, and a runnable multi-turn agent that discovers tool use from a sparse reward.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN
5. Policy Gradients
6. TRPO and PPO
7. RLHF
8. GRPO
9. DPO and Agentic RL (this one, the finale)

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog9/blog9-social-cover.png` — series-style diagram cover: dark navy with a faint grid, neon nodes (PREFERENCES, ONE LOSS, TOOL CALLS) converging into a glowing "ALIGNED AGENT" node, title and subtitle below (recommended hero)
2. **DPO margin panels**: `marketing/blog9/fig-dpo-margin.png` — loss falls as the margin grows, gradient weight fades once a pair is confidently correct (the core mechanic)
3. **SimPO length fix**: `marketing/blog9/fig-simpo-length.png` — by sum the long answer wins, by average the short confident one does
4. **KTO value curve**: `marketing/blog9/fig-kto-value-curve.png` — the prospect-theory S-curve, losses steeper than gains
5. **Credit assignment**: `marketing/blog9/fig-credit-assignment.png` — one terminal reward credited back over a six-step trajectory
6. **Environment scaling**: `marketing/blog9/fig-env-scaling.png` — RL environment counts growing from ~20 to millions across model generations
7. **Blog hero (fallback)**: `blogs/09-dpo-and-agentic-rl/images/ai-hero.png` — a fork in a path: one branch folds three model shapes into one, the other opens into a world of tools

Recommended: lead with `blog9-social-cover.png`. A carousel works well: slide 1 the "what does the reward model know that the pairs don't" hook, slide 2 the implicit reward, slide 3 the gradient weight, slide 4 the agent loop with the emergent search numbers, final slide the series recap plus link.
