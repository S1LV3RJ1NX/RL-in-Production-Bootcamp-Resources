# Daily Follow-ups — Blog 3: DP, Monte Carlo & TD

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.

Blog link: https://prathameshsaraf.com/blogs/03-dp-mc-td/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #TemporalDifference #DeepLearning #LearningInPublic

---

## Follow-up 1 — three ways, one answer

DP, Monte Carlo, and TD get taught as three separate chapters, so they feel like rival methods. They are not.

They are three answers to one practical question: what data do you have, and when can you afford to update?

I built one Mars Rover gridworld and ran all three on it. Same environment, same target values, three different routes to the same numbers.

When you learned these, did they feel connected or like three random algorithms?

Link in comments.

---

## Follow-up 2 — DP needs the map

Dynamic Programming is the exact one. It sweeps every state and computes the true values.

The catch: it needs the full model of the world, the probability of every outcome of every action. In real problems you almost never have that map.

DP is the answer key. The rest of RL is about passing the test without it.

Link in comments.

---

## Follow-up 3 — Monte Carlo waits

Monte Carlo needs nothing but finished episodes. It plays the game to the end, then averages the returns that actually happened.

No model, no assumptions. The price is patience: it cannot learn anything until the episode is over, and a single whole-episode return is noisy.

Honest, simple, and slow. Sometimes that is exactly the right trade.

Link in comments.

---

## Follow-up 4 — TD updates after one step

Temporal Difference is the idea that powers modern RL.

It updates after a single step by bootstrapping: it nudges its guess of this state toward the reward plus its current guess of the next state. No model, no waiting for the episode to end.

And the update is the running-average rule from blog 1 with the pieces renamed. You already knew it.

Link in comments.

---

## Follow-up 5 — the proof (attach fig-convergence)

This is the whole post in one picture.

DP computes the exact values directly. Monte Carlo and TD crawl toward those same values from sampled experience. The plot shows both meeting the dashed DP line.

Three methods, three different assumptions about the world, one answer. That convergence is why you can treat them as one family.

(Attach: fig-convergence.png)

Link in comments.

---

## Follow-up 6 — why TD usually wins on speed

If MC and TD reach the same values, why does TD often get there in far fewer episodes?

Because TD bootstraps. A reward's information spreads after one transition instead of being buried inside one noisy whole-episode return. MC throws that structure away and pays for it in variance.

MC still earns its keep when episodes are short and you want zero bootstrap bias. The trade-off is real.

Link in comments.

---

## Follow-up 7 — recap and the bridge

Blog 3 in four lines:

- DP needs the model and is exact.
- Monte Carlo needs only finished episodes.
- TD needs one transition and bootstraps.
- All three solve the same Bellman equation and converge to the same values.

Everything so far is prediction: scoring a policy someone handed you. Next up is blog 4, where one extra symbol turns this into control, finding the best policy yourself. Which did you reach for first, MC or TD?

Link in comments.

---

## Notes

- Vary the opening line on reuse; identical reposts on one platform get penalized.
- If an RL paper trends this week, quote-post with "the prediction-vs-control foundation behind this" plus your link.
