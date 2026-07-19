# Daily Follow-ups — Blog 11: Dreaming to Dodge (World Models)

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Sep 2, then one per day through Follow-up 6 = Mon Sep 7.

Blog link: https://prathameshsaraf.com/blogs/11-world-models/
Simulator link: https://dreaming-to-dodge.vercel.app
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #DeepLearning #AI #LearningInPublic

---

## Follow-up 1 (Wed Sep 2) — the vanishing fireball (attach fig-fireball-recall)

My world model's tokenizer had excellent reconstruction error and was missing the one object the task depends on.

A Doom frame is 4,096 pixels. The fireball, the thing that kills you, is a few dozen of them. Under a uniform pixel loss, erasing it entirely makes a fraction of a percent of the pixels wrong, so the network took the cheap path and smoothed it into the wall. Fireball recall: 0.53. The dream had half a threat.

The fix everyone reaches for, weight bright pixels more, stalled at 0.63. The walls in take_cover are bright too, and they absorbed the extra votes. What separates a fireball from a wall is not brightness. It is warmth: red above green and blue. One weight aimed at warmth took recall to 0.95, and the dream finally contained the thing worth dodging.

A uniform loss defines "important" as "occupies many pixels." What small thing is your objective rounding away?

(Attach: fig-fireball-recall.png)

Link in comments.

---

## Follow-up 2 (Thu Sep 3) — the model that refused to dream death

My world model dreamed beautiful, controllable Doom. One problem: nobody ever died in it.

Death lands on about 1% of steps in take_cover, an 89-to-1 class imbalance. The done head worked out that predicting "alive" every single time costs almost nothing under an unweighted loss, so it did. And in this task, that deletes the entire training signal: reward is +1 per surviving step, so return equals survival time, and the only thing separating a good action from a bad one is whether the episode ends. In a dream where nobody dies, every policy looks equally good and the gradient is zero.

The fix was a 55x class weight on the death class. Death recall went from 0 to 1.0, and the dream started ending when a fireball closed in, as it should.

Same failure as the fireball, one level up: rare events carry the task, and uniform objectives round them away.

Which rare event in your data is your model quietly ignoring?

Link in comments.

---

## Follow-up 3 (Fri Sep 4) — the policy that cheated the dream (attach fig-exploitation-gap)

My policy found a strategy that out-survived the oracle inside the dream and collapsed in the real game. Its strategy: hold one direction forever.

The world model had a soft spot that scored wall-hugging as safe, and an 0.8M-parameter gradient policy found it. Inside the dream it lasted 55 steps against the reactive oracle's 46. In reality it did 45 against the oracle's 98.3. If you've read about reward hacking in RLHF, this is the same event with the world model playing the gameable reward model.

The diagnostic that saved the project: roll the dream under canned policies. The oracle out-survived any fixed strafe 46 to about 30, so the dream itself rewards dodging. The signal existed. The failure was the policy's capacity to exploit, plus how I selected what to deploy.

What closed it: a controller too small to cheat (1,795 parameters, evolved by CMA-ES), a decoded-image feature that looks the same in dream and reality, and picking the final controller on held-out real episodes, where an exploiter can't win.

Every simulator is wrong somewhere. What stops your policy from finding out where?

(Attach: fig-exploitation-gap.png)

Link in comments.

---

## Follow-up 4 (Sat Sep 5) — the whole agent is 1,795 parameters

The Doom agent from this project, the one that beats a DQN trained on 200,000 real frames, is an MLP with 1,795 parameters. Not 1.8 million. 1,795, evolved with CMA-ES instead of gradients.

The small size is not a stunt; it is one of the anti-cheating disciplines. A big policy has enough capacity to memorize the world model's soft spots and exploit them. A tiny one can barely represent more than an honest reactive rule: find the threatening column, move away from it. (It does need one hidden layer. A purely linear controller reacted to fireballs and still dodged below random, because "move away from the threat" is not a linear function of the input.)

The heavy lifting happens upstream. The tokenizer compresses each frame to 64 tokens, the Transformer world model learns the dynamics and generates unlimited practice, and by the time the controller sees the world, dodging is almost easy. That was the thesis of Ha and Schmidhuber's 2018 World Models paper, and it held up here: learn the world once, then even a tiny policy can act well in it.

Where in your stack would a learned world model let you shrink everything downstream of it?

Link in comments.

---

## Follow-up 5 (Sun Sep 6) — the bug that had nothing to do with RL

The very first Doom run of this project produced frames that were uniform gray rectangles. No walls, no fireballs, just gray.

The obvious suspect was headless OpenGL rendering in a cloud container with no display, and I burned real time down that rabbit hole: Xvfb, software Mesa, environment variables. The actual culprit was numpy 2.x. VizDoom 1.2.3's screen-buffer readback silently breaks under numpy >= 2, a known upstream issue, and the fix was one pin in the container image: numpy==1.26.4.

The habit that would have caught it faster: cheap verification gates between expensive stages. Before spending GPU-days evolving a controller, render the dream and look at it. Does the fireball exist? Does strafing move the room? Those checks cost minutes and validate exactly the assumptions the expensive stage depends on. When a sensor returns implausibly uniform data, check the plumbing before the physics.

What's the cheapest gate you could add between two expensive stages of your pipeline?

Link in comments.

---

## Follow-up 6 (Mon Sep 7) — you can play the dream, and a series recap

The strangest artifact from blog 11 is a web page. It looks like Doom. It is not Doom. There is no game engine behind it; a Transformer predicts every frame from your strafes, 64 tokens at a time, and a "watch the AI" mode lets the trained agent play inside its own dream: dreaming-to-dodge.vercel.app

That agent never touched the real game during training. It saw its first real fireball at evaluation time and dodged it, because it had already dodged thousands of imagined ones. Final numbers on held-out episodes: 96.6 steps survived, against 67 random, 90 for a DQN trained on the same 200k real frames, and 98.3 for a hand-coded oracle that reads the real screen.

That closes the series: eleven posts from the Bellman equation through RLHF, GRPO, and DPO, to a production alignment study and an agent that learned inside its own dream. One throughline held the whole way: the value of where I am is the reward I just got plus a discounted value of where I'll land next. The finale's addendum: when a model, not the world, tells you where you'll land next, audit it, because a policy will optimize the model's promises rather than the world.

Which post in the series should I go deeper on next?

Link in comments.

---

## Notes

- Vary the opening line when you reuse an angle; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep numbers readable (for example "survival 96.6 vs 90" and "recall 0.53 to 0.95").
- If a world-models, JEPA, Genie, or Dreamer-style paper trends this week, quote-post with "I reproduced the 2018 version of this from scratch and the hard part surprised me" plus your link.
