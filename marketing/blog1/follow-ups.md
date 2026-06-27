# Daily Follow-ups — Blog 1: RL from First Principles

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts (main LinkedIn post and the X article) live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text, put the blog link in the first comment (not the body), add the hashtags at the bottom.
- **X:** paste the text, drop the blog link in a self-reply right after posting. No hashtags.
- After posting, reply to your own post with the question at the end, and reply to every comment in the first hour. That hour matters more than the post itself.
- Blog 1 is already live, so start on any day. For a partial week (say Sat, Sun, Mon), just use Follow-up 1, 2, 3.

Blog link: https://prathameshsaraf.com/blogs/01-rl-intro-and-prerequisites/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #LLMs #LearningInPublic

---

## Follow-up 1 — the one sentence

Every RL algorithm you have heard of, PPO, DQN, RLHF, GRPO, is a variation on one sentence:

"The value of where I am is the reward I just got, plus a discounted value of where I will land next."

That is the whole field in a line. My first blog builds the five pieces of math you need to actually believe it.

What part of RL felt like it skipped a chapter for you?

---

## Follow-up 2 — expectation (attach fig-lln.png)

"How good is this state?" really means "how good on average."

A fair die never lands on 3.5, but that is the number every roll averages to. An RL agent learns value the same way: collect episodes, average the returns, trust the mean.

One image makes it obvious: the running average of die rolls settling onto 3.5 as samples pile up.

(Attach: fig-lln.png)

---

## Follow-up 3 — variance

Here is why one episode tells you almost nothing.

In FrozenLake the value of the start state is about 0.01, but the standard deviation is about 0.10, roughly ten times the mean. Any single run returns 0 or nearly 1. On its own that is pure noise.

This is the real reason RL needs thousands of samples before anything stabilizes. It is not slow learning, it is high variance.

---

## Follow-up 4 — the running average

You do not need to store every reward you have ever seen to learn from it.

The whole trick is one line:

new_mean = old_mean + (1/n) * (new_value - old_mean)

One number, one update, constant memory. This incremental average is the backbone of every TD method that comes later in the series. Learn it once here and Q-learning stops looking scary.

---

## Follow-up 5 — discounting

Why does a reward today beat the same reward ten steps from now?

You multiply future rewards by gamma to the power k, with gamma between 0 and 1. That one move does three jobs at once: it keeps an infinite sum finite, it bakes in uncertainty about the future, and it pulls the agent's attention toward what is close.

Gamma is a tiny knob with an outsized effect on how an agent behaves.

---

## Follow-up 6 — who this is actually for

You do not need a PhD to read RLHF papers. You need five ideas.

When I first tried to read alignment papers, the wall was never the algorithm. It was the foundations underneath that everyone assumed I already had: expectation, the Markov property, variance, running averages, discounting.

So I wrote the intro I wish I had before touching a single paper. If "PPO maximizes the clipped surrogate objective" reads like static, start here.

---

## Follow-up 7 — recap and the bridge

One week on blog 1, here is the whole thing in five lines:

- Value is an average, not a single outcome.
- Only the present state matters (Markov).
- One episode is noise; you need many.
- Agents learn with a running average, a little at a time.
- Discounting keeps the future finite and near-focused.

Next up is blog 2: MDPs and the Bellman equation, the recursive structure that ties all five together. Which of these five did you have to relearn as an adult?

---

## Notes

- Vary the opening line if you repost the same angle twice; the algorithm penalizes identical content on the same platform.
- If a paper or model drops this week, quote-post it with "here is the foundation behind this" and your blog link. A timely reply outperforms a scheduled post.
