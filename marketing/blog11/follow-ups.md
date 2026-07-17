# Daily Follow-ups — Blog 11: World Models

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Sep 2, then one per day through Follow-up 6 = Mon Sep 7.

Blog link: https://prathameshsaraf.com/blogs/11-world-models/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #DeepLearning #AI #LearningInPublic

---

## Follow-up 1 (Wed Sep 2) — the 98% accurate model of the wrong world

My world model reached 98.2% next-token accuracy. The agent trained inside it played at random. Both numbers are correct, and neither one is lying.

Accuracy measures agreement with the model's input, and its input was tokens from a tokenizer that had already deleted the ball. So the Transformer learned a perfect transition function for a game of Catch with nothing to catch, and dreamed it flawlessly. A faithful mirror of an amputated input reflects the amputation just as faithfully.

I now treat "the model is accurate" and "the model captures what the task needs" as two separate claims that need two separate checks. My favorite cheap probe: train a tiny classifier to predict the reward from the representation alone. If it can't beat the base rate, stop tuning the policy, the problem is upstream.

Which metric in your stack is accurate against the wrong target?

Link in comments.

---

## Follow-up 2 (Thu Sep 3) — the ball is 0.5% of the pixels

Here is the arithmetic that broke my agent.

A Catch frame is 4,096 pixels. The ball is about 20 of them, half a percent. My tokenizer's reconstruction loss was a uniform average over pixels, so erasing the ball entirely made 0.5% of the pixels wrong while thousands of correct background pixels kept the average looking excellent. The reported error: 0.0165. Looks great. The one object the task depends on was gone.

Keeping the ball means rendering a small bright object at a precise, always-moving position. Dropping it costs almost nothing. When two solutions land that close, seed noise picks the winner, and across 11 seeds the ball survived in 2.

The fix was one line: weight each pixel by brightness, so a ball pixel is worth 26 votes instead of 1. That lifts the ball from 0.5% of the objective to 13%. Eleven seeds, eleven survivals, zero variance.

A uniform loss defines "important" as "occupies many pixels." What small thing is your loss function rounding away?

Link in comments.

---

## Follow-up 3 (Fri Sep 4) — the diagnosis that survived its own fix (attach fig-ball-recall)

Everyone who saw my broken tokenizer said the same thing: codebook collapse. They were right. It was using 3 of its 256 codes, and dead codes get exactly zero gradient, so they never come back on their own.

So I fixed it. EMA updates plus dead-code revival took the codebook from 3 living codes to 254. Textbook cure, worked perfectly.

Then I ran the ablation. Healthy codebook, 11 seeds: the ball survived reconstruction in 2 of 11. The collapsed codebook had kept it in 4 of 11. The celebrated fix did nothing for the actual problem.

Changing only the loss function, with the codebook mechanism held fixed, went 11 for 11. Collapse was real, curable, and not the cause.

The general rule I took away: a diagnosis that survives its own fix is not the cause, and the only way to find that out is to change one variable at a time and count.

(Attach: fig-ball-recall.png)

Link in comments.

---

## Follow-up 4 (Sat Sep 5) — the policy that cheated the dream

On the Doom version of this project, my policy found a strategy that survived longer than the oracle inside the dream and collapsed in the real game. Its strategy: hold left forever.

The world model had blind spots, and the policy optimized straight into them. It stopped learning to dodge fireballs and started learning to exploit the simulator. If you've read about reward hacking in RLHF, this is the same event with the world model playing the gameable reward model.

What fixed it, in order of impact: longer dream horizons (short dreams ended before death could occur, so everything looked safe), more collect-train rounds so the model got grounded on the policy's own mistakes, and selecting the final controller on held-out real episodes, where an exploiter can't win.

Final score, trained with zero real-environment gradients: 96.6 steps survived, against 67 random and 90 for a DQN trained on 200k real frames.

Every simulator is wrong somewhere. What stops your policy from finding out where?

Link in comments.

---

## Follow-up 5 (Sun Sep 6) — the whole agent is smaller than you think

The Doom agent from this project, the one that beats a DQN trained on 200,000 real frames, is an MLP with about 1,800 parameters. Not 1.8 million. 1,800, evolved with CMA-ES instead of gradients.

It can be that small because the heavy lifting happens elsewhere. The tokenizer (0.8M params) compresses each frame to discrete tokens. The Transformer world model (3.3M) learns the dynamics and generates unlimited practice. By the time the controller sees the world, dodging fireballs is almost linearly separable.

That division of labor is the world-models thesis from Ha and Schmidhuber's 2018 paper, and it still holds up: learn the world once, then even a tiny policy can act well in it. The expensive part of RL was never the acting. It was understanding what you're looking at.

Where in your stack would a learned world model let you shrink everything downstream of it?

Link in comments.

---

## Follow-up 6 (Mon Sep 7) — the debugging chain, and a series recap

The most useful artifact from blog 11 is a seven-link chain:

codebook collapses, so there is no code for the ball, so reconstructions have no ball, so the dream has no ball, so reward never varies with the paddle, so there is no gradient, so the agent plays at random.

The symptom appeared at the last link. The cause lived at the first. Every hour I spent tuning the policy (reward shaping, longer horizons, more updates) was spent on the wrong end of the chain, because no policy work can recover a signal the representation destroyed. Debug coupled systems upstream, and verify each link with your eyes before spending compute on the next.

That closes the series: eleven posts from the Bellman equation through RLHF, GRPO, and DPO, to a production alignment study and an agent that learned inside its own dream. One throughline held the whole way: the value of where I am is the reward I just got plus a discounted value of where I'll land next.

Which link of your current pipeline have you actually verified, rather than assumed?

Link in comments.

---

## Notes

- Vary the opening line when you reuse an angle; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep numbers readable (for example "catch rate 0.11 to 1.00" and "survival 96.6 vs 90").
- If a world-models, JEPA, Genie, or Dreamer-style paper trends this week, quote-post with "I rebuilt the 2023 version of this from scratch and watched it fail in the most instructive way" plus your link.
