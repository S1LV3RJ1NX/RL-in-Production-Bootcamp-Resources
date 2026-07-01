# Daily Follow-ups — Blog 4: SARSA, Q-learning & DQN

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Jul 15, then one per day through Follow-up 6 = Mon Jul 20. On Tuesday Jul 21 the series moves to blog 5.

Blog link: https://prathameshsaraf.com/blogs/04-sarsa-qlearning-dqn/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #DeepLearning #LLMs #LearningInPublic

---

## Follow-up 1 (Wed Jul 15) — control is prediction plus a max

Prediction asks how good a policy is. Control asks the harder question: what is the best policy, when nobody hands you one?

The jump between them is one symbol. Take last week's TD update, apply it to action-values, and add a max over the next action. That is Q-learning.

Control is prediction plus a max. What concept finally made RL "click" for you?

Link in comments.

---

## Follow-up 2 (Thu Jul 16) — max versus argmax

Two words that quietly run all of value-based RL, and they are easy to mix up.

- max over actions gives you a value: how good is this state?
- argmax over actions gives you a move: what should I actually do?

Same row of numbers, two different questions. The max scores the state, the argmax picks the action. No model needed for either.

Link in comments.

---

## Follow-up 3 (Fri Jul 17) — one symbol separates SARSA and Q-learning

SARSA and Q-learning differ by a single symbol in their target.

- Q-learning bootstraps off the best next action (a max). It learns the optimal path even while exploring.
- SARSA bootstraps off the action it actually took. It is honest about its own exploration.

That one symbol is the entire difference between off-policy and on-policy learning.

Link in comments.

---

## Follow-up 4 (Sat Jul 18) — the cliff paradox (attach fig-sarsa-vs-q-curves)

Same grid, same hyperparameters, one symbol of difference, two different personalities.

On CliffWalking, SARSA scored -19.9 in training and Q-learning scored -48.2. Yet Q-learning learned the shorter path.

Both are right. Q-learning takes the brave cliff-edge route assuming perfect play. SARSA routes higher because it knows it sometimes slips. Optimal depends on whether you keep exploring.

(Attach: fig-sarsa-vs-q-curves.png)

Link in comments.

---

## Follow-up 5 (Sun Jul 19) — from a table to a network

CliffWalking has 48 states, so the values fit in a table. Pong has more pixel screens than atoms in the universe, so the table has to go.

The fix changes only the container, not the rule. Replace the lookup table with a neural network that takes a state and emits one value per action. A table memorizes; a network generalizes to states it has never seen.

Same update, bigger container. That is DQN.

Link in comments.

---

## Follow-up 6 (Mon Jul 20) — the two tricks that make DQN work, and the week in recap

Naive deep Q-learning diverges. DQN adds two fixes, each aimed at one failure. Experience replay stores transitions in a buffer and samples random batches, so consecutive frames stop being correlated. A target network builds the label with a frozen copy of the weights, so the agent stops chasing a target that moves every step. Neither changes the objective; they just let gradient descent cope. That is the jump to the network that learned 49 Atari games.

So here is blog 4 in five lines:

- Control is prediction plus a max.
- max is a value, argmax is an action.
- One symbol separates SARSA (on-policy) from Q-learning (off-policy).
- On a cliff, that symbol changes the route the agent learns.
- Swap the table for a network and the same loop plays Pong from pixels.

Tomorrow the series moves to blog 5: policy gradients, where we skip values and optimize the policy directly, the branch that leads to PPO and RLHF. Have you hit the cliff paradox in your own training runs?

Link in comments.

---

## Notes

- Vary the opening line on reuse; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep equations readable (for example "target = r + gamma * max Q(s', a')").
- If a deep-RL or RLHF paper trends this week, quote-post with "the Q-learning foundation behind this" plus your link.
