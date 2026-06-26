# LinkedIn Post — Blog 4: SARSA, Q-learning & DQN

## Schedule

- **Date:** Tuesday, July 14, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

Last week I solved the Bellman equation three ways to score a fixed policy. This week I added one symbol, a max, and the same equation started finding the best policy on its own.

Blog 4 is live: "SARSA, Q-learning, and DQN: From a Table to a Network That Plays Atari."

Prediction asks how good a given policy is. Control asks the harder question: what is the best policy, when nobody hands you one? The jump between them is one line of code. Take last week's TD update, apply it to action-values Q(s,a) instead of V(s), and take a max over the next action. That is Q-learning.

The detail that stuck with me: SARSA and Q-learning differ by a single symbol in the target, and on one cliff-walking grid they grow visibly different personalities.

- Q-learning takes the max, assuming perfect future play, so it learns the short path that hugs the cliff edge.
- SARSA backs up the action it actually took, exploration and all, so it decides the edge is risky and routes a row higher.

Same grid, same hyperparameters, one symbol of difference. In training SARSA scored -19.9 and Q-learning -48.2, yet Q-learning still learned the shorter greedy route. Neither is wrong. SARSA is optimal if you keep exploring; Q-learning is optimal if you will eventually act greedily.

Then we swap the lookup table for a neural network, and the exact same loop learns Pong from raw pixels.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #MachineLearning #DeepLearning #LLMs #LearningInPublic

---

## Comment (post immediately after)

Read the full post: https://prathameshsaraf.com/blogs/04-sarsa-qlearning-dqn/

It ends with two capstones: a runnable CliffWalking program where SARSA and Q-learning differ by one line, and a DQN that learns Pong from pixels using a replay buffer and a target network.

Series so far:

1. RL from First Principles
2. MDPs and Bellman Equations
3. DP, Monte Carlo, and TD
4. SARSA, Q-learning, and DQN (this one)
5. Coming next: Policy Gradients

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Hero from the blog**: `blogs/04-sarsa-qlearning-dqn/images/ai-hero.png` — a hiker at a fork, one branch hugging a cliff edge, an arcade screen faint in the background (recommended hero, it maps directly to the cliff story)
2. **The cliff paradox**: `marketing/blog4/fig-sarsa-vs-q-curves.png` — training curves plus the two learned paths, Q-learning on the edge and SARSA higher up (strongest proof image)
3. **CliffWalking grid**: `marketing/blog4/fig-cliffwalking.png` — the 4x12 grid with the risky and safe routes drawn
4. **Pong in motion**: `blogs/04-sarsa-qlearning-dqn/images/pong.gif` — the trained DQN agent playing (great for the "from a table to Atari" payoff)

Recommended: lead with `ai-hero.png`, or use `fig-sarsa-vs-q-curves.png` if you want the cliff paradox front and center. A carousel works well here: slide 1 the one-symbol hook, slides 2-4 max vs argmax, the cliff paradox, table to network, final slide the Pong gif plus link.
