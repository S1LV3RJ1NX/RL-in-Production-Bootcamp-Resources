# RL Blogs

Long-form, intuition-first writeups for the **RL in Production** course. Each post takes
you from intuition → math → a worked example → code (Gymnasium). The markdown here is the
**source of truth**; posts are synced to the portfolio for hosting (see
[`BLOG_RULES.md`](./BLOG_RULES.md) §7).

## Posts

| # | Title | Status |
|---|---|---|
| 01 | [Reinforcement Learning from First Principles (and the Math You Actually Need)](./01-rl-intro-and-prerequisites/) | ✅ Draft |
| 02 | [MDPs and the Bellman Equation: The Recursion Behind Every RL Algorithm](./02-mdps-and-bellman/) | ✅ Draft |
| 03 | [DP, Monte Carlo, and TD: Three Ways to Solve the Bellman Equation](./03-dp-mc-td/) | ✅ Draft |
| 04 | [SARSA, Q-learning, and DQN: From a Table to a Network That Plays Atari](./04-sarsa-qlearning-dqn/) | ✅ Draft |
| 05 | [Policy Gradients: Learning the Policy Directly, from a Bandit to Actor-Critic](./05-policy-gradients/) | ✅ Draft |
| 06 | [TRPO and PPO: The Largest Safe Step a Policy Can Take](./06-trpo-ppo/) | ✅ Draft |
| 07 | [RLHF: Teaching a Language Model What "Good" Means](./07-rlhf/) | ✅ Draft |
| 08 | [GRPO: Teaching a Model to Reason by Comparing It to Itself](./08-grpo/) | ✅ Draft |
| 09 | [DPO and Agentic RL: Align Without a Reward Model, Then Step Into the World](./09-dpo-and-agentic-rl/) | ✅ Draft |

## Writing a new post

Read [`BLOG_RULES.md`](./BLOG_RULES.md): it has the 4-part spine, math/image/code
conventions, the figure pipeline, publishing steps, and a copy-paste skeleton.

## Figures

All code-generated figures come from one reproducible script:

```bash
uv run python blogs/assets/figures.py            # all posts
uv run python blogs/assets/figures.py --blog 01  # one post
```

- `ai-*.png`: AI-generated conceptual illustrations (no precise labels).
- `fig-*.svg`: code-generated diagrams with numbers/axes/math (vector, no JS).
