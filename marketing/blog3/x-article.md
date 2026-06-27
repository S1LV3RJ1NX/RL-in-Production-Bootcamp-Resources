# X Article (Long-Form) — Blog 3: DP, Monte Carlo & TD

## Schedule

- **Date:** Tuesday, July 7, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**DP, Monte Carlo & TD: Three Ways to Solve the Same Equation**

---

## Article Body (~1,200 words)

Last week I wrote the Bellman equation. This week I solved it three different ways and watched all three land on the same numbers.

DP, Monte Carlo, and TD get taught as separate chapters, so they feel like three rival methods. They are not. They are three answers to one practical question: what data do you have, and when can you afford to update? Once you see them that way, the whole middle of every RL course collapses into a single idea.

---

### Having the equation is not the same as having a number

The Bellman equation tells you that the value of a state equals the reward you get plus the discounted value of where you land next. Beautiful, recursive, and completely silent on how you actually compute V(s).

Three families answer that, and they differ on exactly two axes: what they need, and when they update.

- **Dynamic Programming** needs the full model p(s', r | s, a) and sweeps every state. Exact. The catch is that you rarely have the model.
- **Monte Carlo** needs nothing but complete episodes. It waits for the episode to end, then averages the return that actually happened.
- **Temporal Difference** needs a single transition. It updates after every step by bootstrapping off its own estimate of the next state.

Same equation. Different assumptions about the world.

---

### One environment, three solvers

I built a 5x5 Mars Rover gridworld: start top-left, a science target worth +10, two craters worth -10, a step cost of -1, and a slip probability so the terrain is stochastic. One environment, so the comparison is honest.

DP gets to read the rover's internal model P directly. MC and TD only ever call reset() and step(), exactly like a real agent that was never handed a map. Then I ran all three and lined up the answers.

[EMBED IMAGE HERE: fig-convergence.png — V(start) over episodes for MC and TD, with the DP exact value as a dashed reference line]

DP computes the exact values. MC and TD crawl toward those same values from sampled experience, and the convergence plot shows them meeting at the dashed DP line. That picture is the whole post in one image: three methods, three assumptions, one answer.

---

### The update rule you already know

Here is the part that made it click for me. The TD(0) update is not a new idea. It is the running-average rule from the very first post, with the pieces renamed.

$$V(s) \leftarrow V(s) + \alpha \left[ R + \gamma V(s') - V(s) \right]$$

That bracket is the TD error: how much more, or less, I got than I expected. If my estimate were perfect, it would be zero. In code it is almost shorter than the math:

```python
def td_prediction(env, policy, episodes=20000, gamma=GAMMA, alpha=0.05):
    V = np.zeros(env.observation_space.n)
    for _ in range(episodes):
        s, _ = env.reset()
        while True:
            a = policy[s]
            s_next, reward, terminated, truncated, _ = env.step(a)
            # bootstrap: V(s') is our current estimate; 0 at a terminal state
            bootstrap = V[s_next] if not terminated else 0.0
            # TD error = (r + γ·V(s')) - V(s): the surprise
            td_error = reward + gamma * bootstrap - V[s]
            V[s] += alpha * td_error          # nudge a fraction α toward the target
            s = s_next
            if terminated or truncated:
                break
    return V
```

Now compare Monte Carlo. MC uses the same nudge, but its target is the full realized return G, so it has to wait for the episode to end before it can update at all.

The only thing that changes between the two methods is what plays the role of the target. MC uses G, the actual fully-observed return. TD uses R + γV(s'), one real reward plus its current guess of the rest. That single substitution is the entire difference between Monte Carlo and TD.

---

### Why TD usually wins on speed

MC and TD converge to the same values, so why does TD often get there in far fewer episodes? Because TD bootstraps. A reward's information propagates after one transition, instead of being buried inside a single noisy whole-episode return. MC throws that intermediate structure away and pays for it in variance.

MC earns its keep when episodes are short and cheap and you want zero bootstrap bias, or early in training when your value estimates are still too unreliable to bootstrap from safely. The tradeoff is real, not academic.

---

### The bridge to everything after

Everything above is prediction: someone hands you a policy, and you score it by estimating V. The real goal is control, finding the best policy without being handed one.

The fix is to learn Q(s, a) for every action instead of V(s), keep exploring so the alternatives actually get visited, then act greedily:

$$\pi'(s) = \arg\max_a Q(s,a)$$

Take the one-step TD update, apply it to Q-values, and add a max over the next action. That is Q-learning. Bolt a neural network onto it and you get DQN. Keep pulling the thread and you reach PPO, RLHF, and GRPO. The squared version of that same TD error becomes the DQN loss in the next post.

---

### Who this is for

If you are working toward RLHF or GRPO and the DP / MC / TD chapter felt like memorizing three unrelated algorithms, this reframes them as one idea seen from three angles. You will be able to look at a method and immediately ask the two questions that matter: do I have the model, and can I afford to wait for the episode to end?

---

### What's next

Blog 4 is SARSA, Q-learning, and DQN: turning prediction into control, then scaling the value table up to a neural network.

Full post with typed Python, the Mars Rover build, convergence plots, and a hands-on assignment: https://prathameshsaraf.com/blogs/03-dp-mc-td/

Learning RL for LLMs through the @VizuraAI bootcamp. Follow for the rest of the series.

---

## Header Image

- Use **`blog3-x-banner.png`** (5:2 aspect ratio, what X recommends for the article header)
- Embed `fig-convergence.png` inline in the "One environment, three solvers" section (marked `[EMBED IMAGE HERE]`). X does not support SVG, so use this PNG (already exported in this folder).
- Optional second inline image: `fig-mc-td-grids.png` (DP vs MC vs TD heatmaps) near the "Why TD usually wins" section.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The thing that unlocked it for me: TD(0) is just the running-average rule μ ← μ + α(x − μ) from blog 1. Swap the target x = G (Monte Carlo) for x = r + γV(s') (TD) and you've derived the entire difference between the two methods."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "DP, Monte Carlo, and TD aren't three algorithms. They're three answers to one question: what data do you have, and when can you update?" (recommended: pattern interrupt + open loop)
2. "Q-learning, DQN, PPO, GRPO all trace back to a one-line update you can write in a single afternoon. Here's where it comes from."
3. "MC waits for the episode to end. TD updates after one step. Same answer, far fewer episodes. Here's the one substitution that explains why."
4. "The TD update is the running-average rule you already know, with the pieces renamed. Built three solvers on one Mars Rover env to prove they all converge."

Then reply to anyone who engages, same as the first hour of the original.
