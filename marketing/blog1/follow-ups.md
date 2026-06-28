# Daily Follow-ups — Blog 1: RL from First Principles

Copy-paste posts to keep one blog alive for a few days, one angle per day, on both LinkedIn and X. The big posts (main LinkedIn post and the X article) live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply to kick off engagement. Reply to every comment in the first hour. That hour matters more than the post itself.
- **This run is 3 days only** (blog 1 is already live): Follow-up 1 = Sat Jun 27, Follow-up 2 = Sun Jun 28, Follow-up 3 = Mon Jun 29. On Tuesday the series moves to blog 2.

Blog link: [https://prathameshsaraf.com/blogs/01-rl-intro-and-prerequisites/](https://prathameshsaraf.com/blogs/01-rl-intro-and-prerequisites/)
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #LLMs #LearningInPublic

---

## Follow-up 1 (Sat Jun 27) — the one sentence - DONE

Every RL algorithm you have heard of, PPO, DQN, RLHF, GRPO, is a variation on one sentence:

"The value of where I am is the reward I just got, plus a discounted value of where I will land next."

That is the whole field in a line. My first blog builds the five pieces of math you need to actually believe it.

What part of RL felt like it skipped a chapter for you?

Link in comments.

---

## Follow-up 2 (Sun Jun 28) — expectation (attach fig-lln.png) - DONE

"How good is this state?" really means "how good on average."

A fair die never lands on 3.5, but that is the number every roll averages to. An RL agent learns value the same way: collect episodes, average the returns, trust the mean.

One image makes it obvious: the running average of die rolls settling onto 3.5 as samples pile up.

Which of these five concepts tripped you up first when you learned RL?

(Attach: fig-lln.png)

Link in comments.

---

## Follow-up 3 (Mon Jun 29) — the running average

You do not need to store every reward you have ever seen to learn from it.

The whole trick is one line:

new_mean = old_mean + (1/n) * (new_value - old_mean)

One number, one update, constant memory. This incremental average is the backbone of every TD method later in the series. Learn it once here and Q-learning stops looking scary.

Tomorrow the series moves to blog 2: MDPs and the Bellman equation. What do you want unpacked next?

Link in comments.

---

## Spare angles (use if you extend the run or want variety)

**Variance.** Here is why one episode tells you almost nothing. In FrozenLake the value of the start state is about 0.01 but the standard deviation is about 0.10, ten times the mean. Any single run returns 0 or nearly 1. On its own that is pure noise, which is the real reason RL needs thousands of samples. Link in comments.

**Discounting.** Why does a reward today beat the same reward ten steps away? Multiply future rewards by gamma to the power k, with gamma between 0 and 1. That keeps an infinite sum finite, bakes in uncertainty, and pulls attention toward what is close. A tiny knob with an outsized effect. Link in comments.

**Who this is for.** You do not need a PhD to read RLHF papers. You need five ideas: expectation, the Markov property, variance, running averages, discounting. When I tried to read alignment papers, the wall was never the algorithm, it was these foundations everyone assumed I had. So I wrote the intro I wish I had. Link in comments.

---

## Notes

- Vary the opening line if you reuse an angle; identical reposts on the same platform get penalized.
- If a paper or model drops, quote-post it with "here is the foundation behind this" and your blog link. A timely reply outperforms a scheduled post.

