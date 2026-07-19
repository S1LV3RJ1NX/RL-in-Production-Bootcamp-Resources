# LinkedIn Post — Blog 11: Dreaming to Dodge (World Models)

## Schedule

- **Date:** Tuesday, September 1, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

I trained a Doom agent that never played the real game. It practiced dodging fireballs inside a neural network's dream, then matched a hand-coded oracle in reality.

The hard part was not the dream. It was everything the dream got subtly wrong.

Blog 11 is live: "Dreaming to Dodge: Training a Doom Agent Entirely Inside Its Own Dream." The IRIS recipe from scratch on VizDoom take_cover: a VQ-VAE tokenizer, a GPT that predicts the game forward, and a 1,795-parameter controller evolved inside the hallucination.

Every classic model-based failure showed up:

- The tokenizer erased the fireball. Brightness weighting stalled at 0.63 recall (the walls are bright too); weighting warm pixels took it to 0.95.
- The model refused to dream death (1% of steps) until a class weight made laziness expensive.
- The policy cheated: it held one direction, beat the oracle inside the dream, and collapsed in reality. Reward hacking, one abstraction down.

Final score, zero real-environment gradients: 96.6 steps vs 67 random, 90 for a DQN given the same 200k real frames, 98.3 oracle.

You can drive the dream yourself in the browser. Link in comments.

Written while working through the @VizuraAI RL-for-LLMs bootcamp.

#ReinforcementLearning #MachineLearning #DeepLearning #AI #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/11-world-models/

Play the neural simulator (no game engine, the Transformer renders every frame): https://dreaming-to-dodge.vercel.app

Everything is built from scratch and runs on a single A10G via Modal: the straight-through estimator, EMA codebook updates with dead-code revival, the warmth-weighted reconstruction loss, the 1040-position token Transformer with a KV cache, CMA-ES on a tiny controller, and held-out real-episode selection so the dream never grades its own homework.

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
11. Dreaming to Dodge (this one, the series finale)

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog11/blog11-social-cover.png` — series-style diagram cover: dark navy with a faint grid, neon nodes (FRAME, TOKENIZER, WORLD MODEL, CONTROLLER) looping into a glowing "DREAM" node, title and subtitle below (recommended hero)
2. **Survival showdown**: `marketing/blog11/fig-survival-showdown.png` — random 67, DQN 90, the dream-trained agent 96.6, oracle 98.3, with the 188-step "solved" bar (the headline figure)
3. **Exploitation gap**: `marketing/blog11/fig-exploitation-gap.png` — the cheat scores 55 in-dream vs the oracle's 46, then the ranking inverts in reality (45 vs 98.3)
4. **Fireball recall**: `marketing/blog11/fig-fireball-recall.png` — uniform 0.53, luminance 0.63, warmth 0.95: the loss decides what the dream contains
5. **Dream rank**: `marketing/blog11/fig-dream-rank.png` — inside the dream, the reactive oracle (46) out-survives any fixed strafe (~30): the dream rewards dodging
6. **Blog hero (fallback)**: `blogs/11-world-models/images/ai-hero.png` — a sleeper dreaming a dungeon corridor with an incoming fireball

Recommended: lead with `blog11-social-cover.png`, or use `fig-survival-showdown.png` if you want the result front and center. A carousel works well: slide 1 the "never played the real game" hook, slide 2 the vanishing fireball, slide 3 the policy that cheated the dream, slide 4 the showdown bars, final slide the playable simulator plus link.
