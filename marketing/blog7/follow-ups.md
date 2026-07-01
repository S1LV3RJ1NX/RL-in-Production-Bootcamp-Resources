# Daily Follow-ups — Blog 7: RLHF

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Aug 5, then one per day through Follow-up 6 = Mon Aug 10. On Tuesday Aug 11 the series moves to blog 8.

Blog link: https://prathameshsaraf.com/blogs/07-rlhf/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #RLHF #LLMs #LearningInPublic

---

## Follow-up 1 (Wed Aug 5) — there is no reward function for language

In a game the score is given. Chess has checkmate, Breakout has a counter, a pendulum has a measurable angle. The environment defines the reward and RL optimizes it.

For "explain gravity to a child" there is no such number. You cannot sit down and write a reward(text) that captures helpful, honest, and harmless.

And every shortcut gets gamed. Reward length and you get padding. Reward keywords and you get keyword stuffing. Reward a confident tone and you get confidently wrong. Quality is a human judgment with no closed form. So RLHF stops trying to write it and learns it instead. What would your first (doomed) attempt at a reward function be?

Link in comments.

---

## Follow-up 2 (Thu Aug 6) — comparisons beat scores

Ask ten people to grade an essay 0 to 100 and you get ten different numbers. Ask them which of two essays is better and they mostly agree. People are noisy scorers and reliable judges.

So the raw material of RLHF is not scores. It is comparisons: a prompt, a winner, a loser. To make that trainable, pass the score gap through a sigmoid and read it as the probability the winner wins:

P(winner) = sigma(r_w - r_l)

That is the Bradley-Terry model, the same math behind chess Elo. The loss is one line, -log sigma(r_w - r_l). Its gradient is 1 - sigma, so pairs the model already gets right pull almost nothing, and the effort goes to the pairs still in doubt.

Link in comments.

---

## Follow-up 3 (Fri Aug 7) — the reward model is a head on a transformer

What actually produces the score? Judging a prompt and answer is a language task, so the backbone is another transformer. You take a pretrained model and change its output.

The model emits a hidden state at every position. The usual language head maps that to one logit per word. A reward model swaps in a scalar head: one vector that maps the last token's hidden state to a single number.

Why the last token? Causal attention means only the final position has read the whole prompt and answer, so it is the one place to read a verdict on the response. Almost all of the reward model is the pretrained trunk; only the tiny head learns human taste.

Link in comments.

---

## Follow-up 4 (Sat Aug 8) — one score at the end, a signal at every token

The reward model gives one number for a whole answer. PPO updates one token at a time. How do you bridge that?

Three moves. Add a tiny KL toll on every token for drifting from the frozen reference, and drop the reward-model verdict on the last token only. Roll those backward into a return, so the final verdict flows onto every earlier token. Then subtract the critic:

A_t = G_t - V(s_t)

The critic is a value head, a second scalar head on the same trunk, trained during RL to predict the return from a half-written answer. Positive advantage pushes a token up, negative pushes it down. One end-of-answer score becomes a signed nudge per token.

Link in comments.

---

## Follow-up 5 (Sun Aug 9) — RLHF says "KL" twice

Here is the thing almost everyone conflates. RLHF has two KL terms, doing two different jobs.

The clip (from PPO) is a KL on step size. It keeps each update inside a trust region around the policy that collected this batch, refreshed every batch.

The leash is a KL on distance from home. It sits in the reward and pulls the policy back toward the frozen reference across the whole run.

The clip is a speed limit on each step. The leash is a fence around the neighborhood. A speed limit does not keep you in town, and a fence does not stop you crashing. You need both, and widening the clip is not a way to reduce reward hacking.

Link in comments.

---

## Follow-up 6 (Mon Aug 10) — reward hacking, and the bridge to GRPO (attach fig-ppo-ablation)

Turn the KL leash off and watch what happens. The reward-model score climbs higher, from about +0.07 to +0.38. And the answers collapse into repeated tokens, KL runs past 40, diversity drops to 0.31. Higher score, worse text.

You can measure the exploit directly: a repeated clause beats a genuine answer 46.6% of the time under the frozen reward model, stacked flattery 57.5%, pure padding 63.0%. That is Goodhart's law: when a measure becomes a target, it stops being a good measure.

So here is blog 7 in five lines:

- There is no reward(text) for language, so RLHF learns the reward from human comparisons.
- The reward model is a scalar head on a transformer, trained with -log sigma(r_w - r_l).
- A value head plus reward-to-go turns one end score into a per-token advantage A = G - V.
- Two KLs: the clip caps each step, the leash caps drift from the reference.
- Drop the leash and the policy games the reward instead of improving it.

Tomorrow the series moves to blog 8: GRPO. RLHF holds four models at once, and the priciest is the value-head critic. What if you replaced it with the average of a few sampled answers? Where do you think the baseline should come from when the reward is just "is the answer correct"?

(Attach: fig-ppo-ablation.png)

Link in comments.

---

## Notes

- Vary the opening line when you reuse an angle; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep equations readable (for example "A = G - V" and "P(winner) = sigma(r_w - r_l)").
- If an RLHF, reward-modeling, or alignment paper trends this week, quote-post with "the reward-model foundation behind this" plus your link.
