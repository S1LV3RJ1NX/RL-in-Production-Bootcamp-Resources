# LinkedIn Post — Blog 8: GRPO

## Schedule

- **Date:** Tuesday, August 11, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

To align a model with PPO you hold four large models in memory at once: the policy you are training, a frozen reference, a learned reward model, and a value-head critic. DeepSeek threw two of them away and got R1, a reasoning model trained by RL alone.

Blog 8 is live: "GRPO: Teaching a Model to Reason by Comparing It to Itself."

GRPO is PPO with two deletions.

What made it click for me:

- Drop the critic. Its only job was to give a baseline. So sample a group of G answers to the same prompt and use their average reward as the baseline instead. No value network to train, and the advantage is just (r_i - mean) / std.
- Drop the reward model. For math or code there is a right answer, so the reward becomes "is it correct?", checked by a few lines of Python. No hackable network, no preference labels.
- Everything else is the PPO from blog 6, unchanged. Same ratio pi_new/pi_old, same clip. The only new thing is where the advantage comes from.

The signal only appears when a group disagrees. All-correct or all-wrong groups give zero gradient, so learning lands exactly at the boundary of what the model can already solve.

On a toy task, a uniform policy climbs from 20% to 96% correct with nothing but a group baseline and a clip. Scale that up and you get DeepSeek-R1, which learned to check its own work with nobody ever scoring the reasoning.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #GRPO #LLMs #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/08-grpo/

It builds GRPO straight out of RLHF: why the critic was only ever a baseline, how a group of sampled answers replaces it for free, why a verifiable check beats a learned reward for math and code, how GRPO wraps that group-relative advantage in PPO's clip, and the DeepSeek-R1 story. It ends with the DAPO and Dr. GRPO fixes, a 30-line GRPO from scratch you can run on a CPU, and the real GSM8K run on Llama-3.2-3B.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN
5. Policy Gradients
6. TRPO and PPO
7. RLHF
8. GRPO (this one)
9. Coming next: DPO and Agentic RL

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Social cover**: `marketing/blog8/blog8-social-cover.png` — series-style diagram cover: dark navy with a faint grid, three neon nodes (GROUP BASELINE, VERIFIABLE REWARD, PPO CLIP) converging into a glowing "DEEPSEEK-R1" node, title and subtitle below (recommended hero)
2. **Models in memory**: `blogs/08-grpo/images/fig-models-in-memory.svg` — PPO holds policy, reference, reward model, and value critic; GRPO holds only policy, reference, and a tiny verifier (the "drop the critic" visual)
3. **Group baseline**: `blogs/08-grpo/images/fig-group-baseline.svg` — six reward bars with a dashed group-mean line, each answer's advantage measured as its deviation from the group (the core idea)
4. **All-same, no signal**: `blogs/08-grpo/images/fig-all-same-no-signal.svg` — all-correct and all-wrong groups are flat at zero, only a split group produces a gradient
5. **Real GSM8K run**: `blogs/08-grpo/images/fig-grpo-training-curve.svg` — a noisy reward curve rising with a wide in-group spread band, KL to reference staying tiny (the honest payoff image)
6. **Blog hero (fallback)**: `blogs/08-grpo/images/ai-hero.png` — a calm robot below five thought bubbles with a dashed line at their average height

Recommended: lead with `blog8-social-cover.png`, or use `fig-models-in-memory.svg` (exported to PNG) if you want the "four models become two" idea front and center. A carousel works well: slide 1 the "four models in memory" hook, slides 2-4 the group baseline, the verifiable reward, and the borrowed clip, slide 5 the DeepSeek-R1 aha moment, final slide the DPO and Agentic RL bridge plus link.
