# X Article (Long-Form) — Blog 4: SARSA, Q-learning & DQN

## Schedule

- **Date:** Wednesday, July 15, 2026
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**SARSA, Q-learning & DQN: How One max Turns Evaluation Into Control**

---

## Article Body (~1,200 words)

Last week I solved the Bellman equation three ways to score a fixed policy. This week I added a single symbol, a max, and the same equation started finding the best policy on its own. That one symbol is the entire jump from prediction to control, and it is the seed of Q-learning, DQN, and everything after.

Prediction and control get taught as if they were different subjects. They are not. Control is prediction plus a max. Once you see that, SARSA, Q-learning, and DQN stop being three algorithms to memorize and become one update with three small edits.

---

### Having a value is not the same as knowing what to do

Last week we estimated V(s), the value of a state. The problem is that you cannot act on V alone. To pick a move from V you would need the model's transition probabilities p(s' | s, a) to know where each action lands, which is exactly what we worked so hard to avoid.

So we switch from state-values to action-values Q(s, a): the value of taking action a in state s, then continuing. Today's one equation is the Bellman optimality equation, now written for action-values:

$$Q^*(s,a) = r + \gamma \max_{a'} Q^*(s', a')$$

Read it plainly: the best you can do from state s after action a is the reward you collect now, plus the discounted value of playing best from the next state.

The payoff of switching to Q is that acting needs no model at all. With a number for every action, the value of a state is just the max, and the move to make is just the arg max:

$$V(s) = \max_a Q(s,a), \qquad \pi(s) = \arg\max_a Q(s,a)$$

The max gives you the score, the arg max gives you the move. That max is the policy-improvement step from the previous post, now hidden inside a single number.

---

### One symbol separates SARSA from Q-learning

Both algorithms run the same online loop: act, observe one transition, nudge Q toward a target. The only thing that differs is which next-state value the target bootstraps from.

```python
# The single line that separates the two algorithms:

# Q-learning (off-policy): bootstrap off the BEST next action
target = r + gamma * np.max(Q[s2])      # the max over a'

# SARSA (on-policy): bootstrap off the action actually TAKEN
target = r + gamma * Q[s2, a2]          # a2 ~ epsilon-greedy
```

That max is the whole story. Q-learning ignores what the agent does next and always points at the greedy action, so it learns Q*, the value of acting optimally, even while it behaves randomly. SARSA backs up the action epsilon-greedy actually took, so it evaluates the policy it is really running, exploration included.

This off-policy property is the licence we cash in later. Because Q-learning's target depends only on (s, a, r, s') and a max, never on which policy produced the data, a transition collected by an older, more exploratory network is still valid to learn from. That single fact is what makes experience replay, and therefore DQN, possible.

---

### The cliff that makes the difference visible

I ran both on CliffWalking: a 4x12 grid where the bottom row between start and goal is a cliff. Every step costs -1, stepping onto the cliff costs -100 and teleports you back to start. The shortest path hugs the cliff edge, one random misstep away from disaster. A safer path climbs a row higher and pays a few extra steps.

Same hyperparameters, same fixed epsilon of 0.1, the only difference is the target line above.

[EMBED IMAGE HERE: fig-sarsa-vs-q-curves.png — left, training-reward curves for both; right, the learned greedy paths, Q-learning on the edge and SARSA a row higher]

```text
Q-Learning  avg last 500 reward: -48.2
SARSA       avg last 500 reward: -19.9
Q-Learning greedy path length: 13 steps
SARSA      greedy path length: 15 steps
```

Here is the paradox. SARSA's training reward (-19.9) is far better than Q-learning's (-48.2), yet Q-learning learned the shorter greedy path (13 steps vs 15). Both are correct. Q-learning's max assumes perfect future play, so it learns the optimal edge path, but while training with epsilon 0.1 it keeps falling off, dragging its scores down. SARSA is on-policy, so its values account for those random steps and it routes one row higher, away from the cliff.

SARSA is optimal given that you keep exploring. Q-learning is optimal assuming you will eventually act greedily. The cliff just makes that abstract difference impossible to miss.

---

### From a 48-cell table to a network that plays Atari

CliffWalking has 48 states, so Q fits in a table. Pong has more pixel configurations than atoms in the universe, so the table has to go. The fix changes only the container for Q, not the update rule. Replace the lookup table with a function Q(s, a; theta), a neural network with weights theta that takes a state in and emits one Q-value per action. A table memorizes; a network generalizes to states it has never seen.

The DQN loss is the same TD error from last week, squared:

$$\mathcal{L}(\theta) = \big( r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta) \big)^2$$

The catch is that this naive version diverges in practice. Two specific things break, and DQN adds one fix for each. Experience replay stores transitions in a buffer and samples random minibatches, so consecutive frames stop being correlated. The target network keeps a frozen copy of the weights to build the label, so the agent stops chasing a target that moves every step.

That is the entire jump from a wobbly idea to the network that learned 49 Atari games from raw pixels. The same loop that fills a 48-cell table, wrapped in those two fixes, learns Pong.

---

### Who this is for

If you are working toward RLHF, PPO, or GRPO and the value-based chapter felt like four unrelated algorithms, this reframes them as one update with three edits. Start with the Monte Carlo average, swap the target for a one-step bootstrap to get TD, move from V to Q to get SARSA, then add a max to get Q-learning. Bolt on a network and you have DQN. The squared TD error you meet here becomes the loss you will see again in every deep RL method that follows.

---

### What's next

Blog 5 is Policy Gradients: instead of learning values and acting greedily, we optimize the policy directly. That is the branch that leads to PPO and the RLHF stack.

Full post with typed Python, the CliffWalking build, the cliff paradox, and a runnable DQN: https://prathameshsaraf.com/blogs/04-sarsa-qlearning-dqn/

Learning RL for LLMs through the @VizuraAI bootcamp. Follow for the rest of the series.

---

## Header Image

- Use **`blog4-x-banner.png`** (this folder) for the article header. It matches the series template: dark navy with a faint grid, three neon nodes (SARSA, Q-learning, DQN) converging into a glowing Q\*, with the title and subtitle set on the right.
- Embed `fig-sarsa-vs-q-curves.png` inline in the cliff section (marked `[EMBED IMAGE HERE]`). X does not support SVG, so use this PNG (already exported in this folder with `rsvg-convert -z 2`).
- Optional inline images: `fig-cliffwalking.png` near the cliff setup, and `pong.gif` (from the blog's images folder) at the DQN section for the payoff.
- Fallback header: `ai-hero.png` from `blogs/04-sarsa-qlearning-dqn/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. Q-learning target = r + gamma * max_a' Q(s',a'). SARSA target = r + gamma * Q(s',a') for the a' you actually took. That one max is the difference between learning the optimal path and learning the safe one."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "SARSA and Q-learning differ by a single symbol. On a cliff, that one symbol is the difference between the brave path and the safe path." (recommended: pattern interrupt plus open loop)
2. "SARSA scored -19.9 in training, Q-learning scored -48.2, and yet Q-learning learned the better path. Both are right. Here is why."
3. "Control is just prediction plus a max. That is the entire jump from scoring a policy to finding the best one."
4. "The same update that fills a 48-cell table, wrapped in two fixes, learns Pong from raw pixels. Here is the full path from table to Atari."

Then reply to anyone who engages, same as the first hour of the original.
