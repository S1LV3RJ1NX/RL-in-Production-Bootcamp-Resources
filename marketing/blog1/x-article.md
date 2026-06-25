# X Article (Long-Form) — Blog 1: RL from First Principles

## Schedule

- **Date:** Thursday, June 25, 2026 (today)
- **Time:** ~12:45 PM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30 PM IST (different hook)
- **Note:** posting LinkedIn + X together is fine — keep both tabs open for first-hour replies

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

Everyone's shipping RLHF and GRPO. When I tried to actually *read* those papers, I hit a wall. It wasn't the algorithms. It was the foundations underneath them.

So I started writing the series I wish I'd had: RL from scratch, with real math and real code, and none of the hand-waving.

**The one sentence the whole field hangs on:**

> The value of where I am is the reward I just got, plus a discounted value of where I'll land next.

Q-learning, DQN, PPO, RLHF, GRPO. Every one of them is a variation on that line. This post builds the intuition behind it.

---

### The five pieces of math you need

**1. Expectation: "how good" means "how good on average"**

A single trajectory is noise. Value is the *average* over all possible futures, weighted by probability. For a fair die, E[X] = 3.5. You never roll a 3.5, but it's what the rolls average to.

That's how an RL agent estimates value: collect episodes, average the returns, trust the mean.

**2. The Markov property: only the present matters**

What happens next depends only on where you are *now*, not how you got here. That is what keeps RL tractable. You don't have to carry the whole history around.

**3. Variance: why a single episode is noise**

In FrozenLake, V(start) is about 0.01 but the standard deviation is about 0.10, roughly 10x the mean. Any single episode returns either 0 or close to 1, which tells you almost nothing on its own. You need thousands of samples before the average settles down.

**4. Running averages: how agents learn a little at a time**

You don't need to store every past reward. The incremental update is:

μ_new = μ_old + (1/n)(x_new - μ_old)

One number, one update, constant memory. It's the backbone of every TD algorithm.

**5. Discounting: why a dollar today beats a dollar tomorrow**

Multiply future rewards by γ^k, with γ between 0 and 1. That keeps the sum finite, bakes in some uncertainty about the future, and pulls the agent's attention toward what's close.

---

### Who this is for

- ML engineers who skipped RL in school and now need it for LLM alignment
- Anyone who reads "PPO maximizes the clipped surrogate objective" and thinks "...what?"
- People who learn best with the code and the math side by side

---

### What's next

Blog 2 covers MDPs and the Bellman equation, the recursive structure that makes all of this solvable. It's out next week.

I'm learning this through the @VizuraAI RL-for-LLMs bootcamp and writing it down so the next person doesn't hit the same walls.

Full post with all the code: [YOUR_PORTFOLIO_URL/blogs/01-rl-intro-and-prerequisites]

---

## Header Image

- Use **`blog1-x-banner.png`** (5:2 aspect ratio, what X recommends for the article header)
- Embed `fig-lln.svg` inline in the body (shows convergence visually)

## Formatting Tips for X Articles

- Use **bold** and *italics* (Premium feature) for emphasis
- Keep paragraphs to 1-3 sentences
- Add generous line breaks between sections
- First 280 chars are what shows in feed — make them count

## First 30 Minutes Strategy

The algorithm scores heavily on early engagement. After publishing:
1. Reply to your own article with a question: "Which of these 5 concepts tripped you up when you first learned RL?"
2. DM 3-5 people in ML who might find it useful
3. Quote-repost with a one-line hook 4-6 hours later
