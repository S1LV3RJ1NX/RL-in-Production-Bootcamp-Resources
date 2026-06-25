# LinkedIn Post — Blog 1: RL from First Principles

## Schedule

- **Date:** Thursday, June 25, 2026 (today)
- **Time:** ~12:45 PM IST (within the 10 AM–2 PM peak window)
- **Follow-up comment:** Post immediately after publishing
- **Be online:** stay available to reply for the first 60 minutes

## Post Text

Everyone's talking about RLHF, GRPO, and PPO.

When I sat down to read those papers, I hit a wall. I was missing the foundations nobody bothers to explain. So I started writing the series I wish I'd had: RL from scratch, with real math and real code, and none of the hand-waving.

Blog 1 is up: "Reinforcement Learning from First Principles (and the Math You Actually Need)."

The whole field rests on one sentence:

> The value of where I am is the reward I just got, plus a discounted value of where I'll land next.

That's it. Q-learning, DQN, PPO, RLHF, GRPO. Every one of them is a variation on that line.

The post walks through the five pieces of math everything else is built on:

- Expectation: why "how good" means "how good on average"
- The Markov property: why only the present matters
- Variance and the law of large numbers: why a single episode is mostly noise
- Running averages: how an agent learns a little at a time
- Discounting: why a dollar today beats a dollar tomorrow

Each one starts with the intuition, then one clean derivation, a worked example, and the Python to run it.

I'm learning this through the Vizura.ai RL-for-LLMs bootcamp, and writing the notes so the next person doesn't have to dig through the same gaps I did.

Link in comments.

---

Who this is for:

- ML engineers who skipped RL in school and now need it for LLM alignment
- Anyone who wants to read RLHF or GRPO papers without flinching at the math

Next up is Blog 2 on MDPs and the Bellman equation, the recursive structure that makes RL tractable.

#ReinforcementLearning #RLHF #MachineLearning #LLMs #LearningInPublic #AI

---

## Comment (post immediately after)

Read the full post: [YOUR_PORTFOLIO_URL/blogs/01-rl-intro-and-prerequisites]

Series so far:

1. RL from First Principles (this one)
2. Coming next week: MDPs and Bellman Equations

Built with Python, NumPy, Gymnasium, and Matplotlib.

---

## Image Suggestions

Use ONE of these as the post image (in order of preference):

1. **Hero image**: `blogs/01-rl-intro-and-prerequisites/images/ai-uncertain-world.png`
   - AI-generated illustration of agent facing uncertain futures
   - Eye-catching, works well as a LinkedIn thumbnail

2. **Law of Large Numbers figure**: `blogs/01-rl-intro-and-prerequisites/images/fig-lln.svg`
   - Shows die-roll average converging to 3.5
   - Great visual proof of "why single episodes are noise"

3. **AI-generated carousel** (optional): see `carousel-prompt.md` below for Canva/design prompts
