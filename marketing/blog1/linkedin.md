# LinkedIn Post — Blog 1: RL from First Principles

## Schedule

- **Date:** Tuesday, June 30, 2026
- **Time:** 10:00 AM IST
- **Follow-up comment:** Post immediately after publishing

## Post Text

Everyone's talking about RLHF, GRPO, and PPO.

But when I tried to read those papers, I realized I was missing the foundations. So I'm writing the blog series I wish existed: RL explained from scratch, with real math, real code, and zero hand-waving.

Blog 1 just dropped: "Reinforcement Learning from First Principles (and the Math You Actually Need)"

Here's the core idea in one sentence:

> The value of where I am = reward I just got + discounted value of where I'll land next.

That's it. Every RL algorithm — Q-learning, DQN, PPO, RLHF, GRPO — is a variation on that single line.

The post covers the 5 pieces of math the entire field is built on:
→ Expectation (why "how good" means "how good on average")
→ The Markov Property (why only the present matters)
→ Variance & the Law of Large Numbers (why single episodes are noise)
→ Running averages (how agents learn incrementally)
→ Discounting (why $1 today > $1 tomorrow)

Each one gets: intuition → one clean derivation → a worked example → Python code.

I'm learning this through the Vizura.ai RL-for-LLMs bootcamp, and writing these notes so the next person doesn't have to struggle through the same gaps I did.

🔗 Link in comments

---

Who this is for:

- ML engineers who skipped RL in school and now need it for LLM alignment
- Anyone who wants to read RLHF/GRPO papers without flinching at the math

Next week: Blog 2 — MDPs & Bellman Equations (the recursive structure that makes RL tractable).

#ReinforcementLearning #RLHF #MachineLearning #LLMs #LearningInPublic #AI

---

## Comment (post immediately after)

🔗 Read the full post: [YOUR_PORTFOLIO_URL/blogs/01-rl-intro-and-prerequisites]

Series so far:

1. ✅ RL from First Principles (this one)
2. Coming next week: MDPs & Bellman Equations

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
