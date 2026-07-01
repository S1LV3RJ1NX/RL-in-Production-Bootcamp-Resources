# LinkedIn Post — Blog 7: RLHF

## Schedule

- **Date:** Tuesday, August 4, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

You cannot write a Python function that scores "be helpful and honest." There is no reward(text) that captures it. So how did anyone train ChatGPT to be either?

Blog 7 is live: "RLHF: Teaching a Language Model What 'Good' Means."

The move is to stop writing the reward and start learning it. People are unreliable when they score an answer 0 to 100, but reliable when they pick which of two answers is better. RLHF turns that pile of comparisons into a signal PPO can climb.

What made it click for me:

- A reward model is just a transformer with the word-predicting head swapped for a single number. Train it on "winner beats loser" with one line, -log sigma(r_w - r_l), and preference accuracy on GPT-2 climbs from 0.51 to 0.59.
- The reward arrives once, at the last token. A value head plus reward-to-go spreads that single score into a signed nudge for every token.
- RLHF says "KL" twice. The clip caps the size of each step. The leash caps how far the policy drifts from where it started. Different jobs, often confused.

Then the warning. Turn the leash off and the reward-model score climbs higher, +0.381 against +0.069, while the answers collapse into repeated tokens. Higher score, worse text. That is Goodhart's law, measured on a real run.

This is the exact PPO from blog 6, now pointed at language.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #RLHF #LLMs #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/07-rlhf/

It builds the whole pipeline from scratch: text generation as an MDP, the Bradley-Terry reward model learned from human comparisons, a value head that turns one end-of-answer score into per-token advantages, the two KLs, and a runnable RLHF loop on GPT-2. It finishes with the beta=0 ablation that shows reward hacking happen live.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN
5. Policy Gradients
6. TRPO and PPO
7. RLHF (this one)
8. Coming next: GRPO

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog7/blog7-social-cover.png` — series-style diagram cover: dark navy with a faint grid, three neon nodes (REWARD MODEL, VALUE HEAD, KL LEASH) converging into a glowing "ALIGNED LLM" node, title and subtitle below (recommended hero)
2. **Bradley-Terry**: `blogs/07-rlhf/images/fig-bradley-terry.svg` — the sigmoid turns a score gap into a win probability, and the gradient flattens once a pair is comfortably correct (the signature "the loss says enough" visual; export to PNG for social)
3. **Per-token credit**: `blogs/07-rlhf/images/fig-per-token-credit.svg` — one reward-model score at the last token becomes a spread of signed per-token advantages (the core RLHF idea in one picture)
4. **The leash ablation**: `blogs/07-rlhf/images/fig-ppo-ablation.svg` — beta=0.2 keeps KL in single digits, beta=0 lets reward climb higher while KL runs off past 40 (the reward-hacking payoff image)
5. **Goodhart curve**: `blogs/07-rlhf/images/fig-goodhart.svg` — the measured proxy rises forever while true quality peaks then falls, with the early-stop point marked
6. **Blog hero (fallback)**: `blogs/07-rlhf/images/ai-hero.png` — the frozen reference holding a leash on an eager dog straining toward a reward coin

Recommended: lead with `blog7-social-cover.png`, or use `fig-per-token-credit.svg` (exported to PNG) if you want the "one score becomes many" idea front and center. A carousel works well: slide 1 the "you can't write reward(text)" hook, slides 2-4 the reward model, the per-token credit, and the two KLs, slide 5 the reward-hacking ablation, final slide the GRPO bridge plus link.
