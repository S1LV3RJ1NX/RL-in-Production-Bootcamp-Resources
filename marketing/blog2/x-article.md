# X Article (Long-Form) — Blog 2: MDPs & Bellman Equations

## Schedule

- **Date:** Tuesday, June 30, 2026
- **Time:** 10:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, 3:00 PM IST (different hook)

## Article Title

**The Bellman Equation: One Recursive Idea That Powers Every RL Algorithm**

---

## Article Body (~1,200 words)

Last week I covered the five math foundations behind RL. Today, the single equation that ties them all together.

Every RL algorithm you've heard of, from Q-learning and SARSA to PPO, RLHF, and GRPO, is solving or approximating one equation. It's called the Bellman equation, and once you see it, you can't unsee it.

---

### What's an MDP?

Before the equation, you need the framework. A Markov Decision Process has four parts:

- **States (S)**: where you can be
- **Actions (A)**: what you can do
- **Dynamics p(s'|s,a)**: the probability of landing in s' after taking action a in state s
- **Rewards R(s,a,s')**: what you get for that transition

The Markov property says what happens next depends only on where you are *now*, not how you got here. That's what makes the recursion possible.

---

### Two ways to measure "how good"

**V(s)**, the state-value, answers "how good is it to *be* in state s?" It's the expected total return from s onward, following policy π.

**Q(s,a)**, the action-value, answers "how good is it to *do* action a in state s?" It's the expected total return after taking a, then following π.

The bridge from Q back to V:

$$V(s) = \sum_a \pi(a \mid s)\, Q(s,a)$$

Average the Q-values, weighted by how likely the policy is to pick each action.

And from V back to Q:

$$Q(s,a) = \sum_{s'} p(s' \mid s,a)\, \big[R + \gamma V(s')\big]$$

Average the outcomes, weighted by how likely each next state is. That sum is a single loop in code, and each line maps straight onto a term in the equation:

```python
def q_from_v(V: np.ndarray, s: int, a: int) -> float:
    q_value: float = 0.0
    # P[s][a] lists the possible outcomes of taking action a in state s
    for prob, next_state, reward, done in P[s][a]:
        # p(s'|s,a) · [ r + γ · V(s') ]
        q_value += prob * (reward + gamma * V[next_state])
    return q_value
```

---

### The Bellman equation

Combine both bridges and you get one self-referential equation:

$$V(s) = \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s,a)\, \big[R(s,a,s') + \gamma V(s')\big]$$

In words: the value of state s is, over all the actions I might take (weighted by my policy) and all the places I might land (weighted by the dynamics), the sum of the immediate reward plus the discounted value of the future.

It's recursive. V shows up on both sides. The value of *this* state depends on the value of the *next* states, which depend on the states after that, all the way down.

[EMBED IMAGE HERE: fig-backup-diagram.png — the V to Q to V' tree showing one step of the Bellman backup]

Every RL algorithm is a different way of solving that recursion. Dynamic programming iterates the backup with the full model. Monte Carlo samples complete episodes and averages them. TD learning updates after every step using a bootstrap.

---

### From "good" to "optimal"

Replace the policy average (the weighted sum over actions) with a max, and you get the Bellman *optimality* equation:

$$V^*(s) = \max_a \sum_{s'} p(s' \mid s,a)\, \big[R + \gamma V^*(s')\big]$$

Pick the best action instead of averaging over a policy.

The optimal policy then falls out for free:

$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

Once you have the optimal Q-values, acting optimally is just looking up the largest number.

---

### Why this matters for LLMs

If you're working with RLHF or GRPO, you're using algorithms that approximate value functions and optimize policies. The Bellman equation is the backbone underneath all of it. Once it clicks, you can read PPO papers without getting lost, debug a reward signal that isn't working, and see why the KL penalty is there in the first place: it bounds how far the policy is allowed to drift from the reference policy.

---

### What's next

Blog 3 is on DP, Monte Carlo, and TD: three ways to actually *solve* this equation, built from scratch on a custom environment.

Full post with typed Python, backup diagrams, and worked examples on FrozenLake: [YOUR_PORTFOLIO_URL/blogs/02-mdps-and-bellman]

Learning RL for LLMs through the @VizuraAI bootcamp. Follow for the rest of the series.

---

## Header Image

- Use **`blog2-x-banner.png`** (5:2 aspect ratio, what X recommends for the article header)
- Embed `fig-backup-diagram.png` inline in the Bellman equation section, right after the "it's recursive" paragraph (marked `[EMBED IMAGE HERE]` in the body). X does not support SVG, so use this PNG (already exported in this folder).

## First 30 Minutes Strategy

After publishing:
1. Self-reply with: "The moment it clicked for me: V*(s) = max_a Q*(s,a). The policy probabilities don't vanish, they collapse to a one-hot. A weighted average with a one-hot weight just is 'pick the max.'"
2. Quote-repost with a one-line hook 4-6 hours later (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "Q-learning, PPO, RLHF, GRPO: they're all solving the same equation. Once you see it, you can't unsee it." (recommended: open loop + names)
2. "One equation sits under every RL algorithm ever shipped. Most people never actually read it. Here it is, line by line."
3. "The Bellman equation looks terrifying. It says something a five-year-old understands: good now, plus good later."
4. "Wrote the MDP and Bellman explainer I wish I'd had: the math, how to read it symbol by symbol, and the typed Python behind it."

Then reply to anyone who engages, same as the first hour of the original.
