# LinkedIn Post — Blog 2: MDPs & Bellman Equations

## Schedule

- **Date:** Tuesday, June 30, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

Last week I shared the five math foundations behind RL.

This week, the equation the whole field spends its time solving.

Blog 2 is live: "MDPs and the Bellman Equation: One Recursive Idea, Unpacked Completely."

Here's the Bellman equation in plain English:

"The value of where I am is the weighted average, over my actions, of the reward I get plus the discounted value of where I land."

That's it. Value iteration, policy gradients, PPO, RLHF: each one is just a different way of solving or approximating that one recursion.

What the post covers:

- What an MDP actually is: states, actions, transitions, and rewards, with a dynamics table you can read line by line
- V versus Q, the two ways to measure "how good" (state-value and action-value)
- The bridge between them, and how to convert one into the other
- The Bellman expectation equation, derived step by step, with both the symbolic reading and the plain-English meaning
- Bellman optimality, and what changes when you pick the best action instead of averaging
- Typed Python that maps every term in the equation to a line of code

Every equation gets the same treatment: the math first, then how to read it symbol by symbol, then what it actually means. It's the format I kept wishing textbooks used.

I'm working through the @VizuraAI RL-for-LLMs bootcamp and writing these up for anyone on the same path.

Link in comments.

#ReinforcementLearning #BellmanEquation #MachineLearning #RLHF #LearningInPublic

---

## Comment (post immediately after)

Read the full post: [YOUR_PORTFOLIO_URL/blogs/02-mdps-and-bellman]

Series so far:
1. RL from First Principles
2. MDPs and Bellman Equations (this one)
3. Coming next: DP, Monte Carlo, and TD

Each post has typed Python, worked examples, and figures.

---

## Image Suggestions

1. **Backup diagram**: `blogs/02-mdps-and-bellman/images/fig-backup-diagram.svg` — shows the tree structure of V→Q→V'
2. **V vs Q bar chart**: `blogs/02-mdps-and-bellman/images/fig-v-vs-q.svg`
3. **Grid heatmap**: `blogs/02-mdps-and-bellman/images/fig-grid-values.svg` — shows optimal values on FrozenLake
4. **AI-generated cover** (generate separately if needed)
