# X Article (Long-Form) — Blog 1: RL from First Principles

## Schedule

- **Date:** Wednesday, July 1, 2026
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, 3:00 PM IST (different hook)

## Why Article instead of Thread?

With X Premium, you get:
- **2-4x visibility multiplier** over free accounts (documented in algorithm source)
- **Articles** are favored by the algorithm — they maximize dwell time (the #1 signal)
- Articles with code blocks, images, and sections outperform threads for tutorials
- Articles are searchable and bookmarkable long-term (threads decay faster)

Use Articles for deep-dive tutorials. Use threads for listicles and live commentary.

---

## Article Title

**RL from First Principles: The 5 Pieces of Math the Entire Field Is Built On**

---

## Article Body (~1,200 words target)

Everyone's shipping RLHF and GRPO. But when I tried to *read* those papers, I hit a wall. Not the algorithms — the foundations.

So I'm writing the blog series I wish existed: RL explained from scratch, with real math, real code, and zero hand-waving.

**The one sentence the whole field hangs on:**

> The value of where I am = reward I just got + discounted value of where I'll land next.

Every algorithm — Q-learning, DQN, PPO, RLHF, GRPO — is a variation on that single line. This post builds the intuition behind it.

---

### The 5 pieces of math you need

**1. Expectation — "how good" means "how good on average"**

A single trajectory is noise. Value is the *average* over all possible futures, weighted by probability. For a fair die: E[X] = 3.5. You never roll 3.5, but it's what the rolls average to.

This is exactly how an RL agent estimates value: collect episodes, average the returns, trust the mean.

**2. The Markov Property — only the present matters**

The probability of what happens next depends only on where you are *now*, not how you got here. This is what makes RL tractable — you don't need to store the entire history.

**3. Variance — why single episodes are noise**

In FrozenLake, V(start) ≈ 0.01 but std ≈ 0.10. That's 10x the mean. Any single episode returns either 0 or ~1 — pure noise. You need thousands of samples before the average stabilizes.

**4. Running averages — how agents learn incrementally**

You don't need to store all past rewards. The incremental update rule:

μ_new = μ_old + (1/n)(x_new - μ_old)

One number, one update, constant memory. This is the backbone of every TD algorithm.

**5. Discounting — why $1 today > $1 tomorrow**

Multiply future rewards by γ^k (0 < γ < 1). This makes the sum finite, encodes uncertainty about the future, and focuses the agent on near-term outcomes.

---

### Who this is for

→ ML engineers who skipped RL in school and now need it for LLM alignment
→ Anyone who reads "PPO maximizes the clipped surrogate objective" and goes "...what?"
→ People who learn best from code + math side by side

---

### What's next

Blog 2 (MDPs & Bellman Equations) drops next week — the recursive structure that makes all of this actually solvable.

I'm learning this through the @VizuraAI RL-for-LLMs bootcamp and writing these so the next person doesn't hit the same walls.

Full post with all code examples: [YOUR_PORTFOLIO_URL/blogs/01-rl-intro-and-prerequisites]

---

## Formatting Tips for X Articles

- Use **bold** and *italics* (Premium feature) for emphasis
- Keep paragraphs to 1-3 sentences
- Add generous line breaks between sections
- Embed the `fig-lln.svg` image inline (shows convergence visually)
- First 280 chars are what shows in feed — make them count

## First 30 Minutes Strategy

The algorithm scores heavily on early engagement. After publishing:
1. Reply to your own article with a question: "Which of these 5 concepts tripped you up when you first learned RL?"
2. DM 3-5 people in ML who might find it useful
3. Quote-repost with a one-line hook 4-6 hours later
