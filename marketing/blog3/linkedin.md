# LinkedIn Post — Blog 3: DP, Monte Carlo & TD

## Schedule

- **Date:** Tuesday, July 7, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

Last week I wrote the Bellman equation. This week I solved it three different ways and watched all three land on the same answer.

Blog 3 is live: "DP, Monte Carlo, and TD: Three Ways to Solve the Bellman Equation."

Here is the part that took me a while to really get. Dynamic Programming, Monte Carlo, and Temporal Difference are not three rival theories. They are three answers to one practical question: what data do you have, and when can you afford to update?

- DP needs the full model of the world and sweeps every state. Exact, but you almost never have the model.
- Monte Carlo needs nothing but complete episodes. It waits for the game to finish, then averages what actually happened.
- TD updates after a single step by bootstrapping off its own estimate of the next state. No model, no waiting.

I built one Mars Rover gridworld and ran all three on it. Same environment, same target values. DP computes them exactly, MC and TD crawl toward them from sampled experience, and the convergence plot shows them meeting at the same numbers.

The detail worth saving: the TD update is the same running-average rule from Blog 1, V(s) ← V(s) + α[target − V(s)]. The only thing that changes between Monte Carlo and TD is what plays the role of the target. That one substitution is the whole difference, and it is the seed of Q-learning, DQN, and everything after.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #TemporalDifference #DeepLearning #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/03-dp-mc-td/

There's also a hands-on assignment with it: build the Mars Rover env, solve it with Value Iteration, then estimate the same values with MC and TD(0) and watch them converge.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD (this one)
4. Coming next: SARSA, Q-learning, and DQN

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **AI-generated cover**: `marketing/blog3/blog3-social-cover.png` — the three methods (DP, MC, TD) converging to one V\* (recommended hero)
2. **Convergence plot**: `marketing/blog3/fig-convergence.png` — V(start) over episodes for MC and TD against the DP exact line (strongest proof image)
3. **Three-panel heatmaps**: `marketing/blog3/fig-mc-td-grids.png` — DP vs MC vs TD grids side by side
4. **Hero from the blog**: `blogs/03-dp-mc-td/images/ai-three-paths.png` — the rover at a crossroads with three paths

Recommended: lead with `blog3-social-cover.png`, or use `fig-convergence.png` if you want the "they all meet" proof front and center.
