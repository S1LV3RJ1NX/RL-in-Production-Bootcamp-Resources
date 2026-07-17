# LinkedIn Post — Blog 11: World Models

## Schedule

- **Date:** Tuesday, September 1, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

I trained an RL agent that never played the real game. It practiced entirely inside a neural network's dream, then caught the ball 500 times out of 500 in reality.

The first version played at random, and the bug was nowhere near the agent.

Blog 11 is live: "World Models: Training an Agent Entirely Inside Its Own Dream." It rebuilds IRIS from scratch: a VQ-VAE tokenizer, a GPT-style Transformer that dreams the game forward, and an actor-critic trained purely in imagination.

What the honest run taught me:

- The world model hit 98.2% accuracy and the agent still learned nothing. It was a faithful model of the wrong world.
- The tokenizer's loss had quietly deleted the ball. It is 0.5% of the pixels, so erasing it barely moves an average pixel error.
- Codebook collapse looked like the culprit. I cured it across 11 seeds and the ball came back in 2. One changed loss line (a ball pixel gets 26 votes) made it 11 of 11.
- Same skeleton on VizDoom: survival 96.6 vs 67 random and 90 for a DQN trained on 200k real frames.

The lesson I keep reusing: a world model can only teach a policy what its tokenizer chooses to preserve.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #DeepLearning #AI #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/11-world-models/

Everything is built from scratch and runs for a couple of dollars on Modal: the straight-through estimator, EMA codebook updates with dead-code revival, the foreground-weighted reconstruction loss, the 170-position token Transformer, lambda-returns worked by hand, the full diagnosis of why the agent played at random, and the Doom capstone where the policy learned to exploit the dream and had to be caught.

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
11. World Models (this one, the series finale)

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog11/blog11-social-cover.png` — series-style diagram cover: dark navy with a faint grid, neon nodes (TOKENIZER, WORLD MODEL, ACTOR-CRITIC) looping into a glowing "DREAM" node, title and subtitle below (recommended hero)
2. **Catch showdown**: `marketing/blog11/fig-catch-showdown.png` — four bars: the as-shipped policy inside the random band, then 1.00 matching the oracle after one upstream fix (the headline figure)
3. **Ball-recall ablation**: `marketing/blog11/fig-ball-recall.png` — the 11-seed experiment: fixing the codebook did nothing (2/11), changing the loss fixed everything (11/11)
4. **Doom survival**: `marketing/blog11/fig-doom-survival.png` — random 67, DQN 90, the dream-trained agent 96.6, oracle 98.3
5. **Policy curves**: `marketing/blog11/fig-policy-curves.png` — the flatlined broken run against the fixed run crossing zero
6. **Blog hero (fallback)**: `blogs/11-world-models/images/ai-hero.png` — a sleeper dreaming a pixel game of Catch

Recommended: lead with `blog11-social-cover.png`, or use `fig-catch-showdown.png` if you want the noise-to-oracle result front and center. A carousel works well: slide 1 the "never played the real game" hook, slide 2 the ball as 0.5% of the pixels, slide 3 the 98%-accurate model of the wrong world, slide 4 the showdown bars, final slide the tokenizer lesson plus link.
