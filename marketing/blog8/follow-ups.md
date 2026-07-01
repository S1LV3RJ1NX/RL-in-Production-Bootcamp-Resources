# Daily Follow-ups — Blog 8: GRPO

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Aug 12, then one per day through Follow-up 6 = Mon Aug 17. On Tuesday Aug 18 the series moves to blog 9.

Blog link: https://prathameshsaraf.com/blogs/08-grpo/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #GRPO #LLMs #LearningInPublic

---

## Follow-up 1 (Wed Aug 12) — four models is two too many

To align a model with PPO you hold four large models in memory at once: the policy you are training, a frozen reference for the KL leash, a learned reward model that scores answers, and a value-head critic that predicts a baseline.

Two of those are the expensive, fragile ones. The reward model is a whole transformer you can fool. The critic is a second network the size of the policy, and on a long reasoning chain, where the reward only lands at the last token, its early per-token guesses are mostly noise.

GRPO is the story of removing both, and still getting a reasoning model out. Which of the four would you try to cut first?

Link in comments.

---

## Follow-up 2 (Thu Aug 13) — drop the critic, the group is the baseline

The critic exists for one job: to give a baseline you subtract from the reward. But the survival fact of the whole series is that any estimate of expected reward that does not depend on the action is a valid baseline. The critic was just one choice.

GRPO picks the simplest one: the group's own average. Sample G answers to the same prompt, score each, and standardize:

A_i = (r_i - mean) / std

The mean is the baseline, measured for free instead of learned by a network. The std just rescales. It is the textbook Monte-Carlo estimate of the exact quantity the critic was trained to predict. No value network, no critic loss.

Link in comments.

---

## Follow-up 3 (Fri Aug 14) — drop the reward model, just check the answer

RLHF learned its reward because "a good answer" had no formula. But ask the model to compute 2 + 2 times 3, or pass a unit test, and "good" suddenly has a definition a short program can check.

So the reward stops being a network and becomes a rule:

r = 1 if answer == gold else 0

No neural net, no preference data, nothing to fit. A wrong answer scores zero no matter how fluent its reasoning, because style was never on the scoresheet. This is RLVR, and it is why a GRPO run can optimize hard without collapsing into the reward-hacked mush RLHF has to guard against.

Link in comments.

---

## Follow-up 4 (Sat Aug 15) — the clip is just PPO, and all-same means no signal

With a group advantage and a verifiable reward, the rest of GRPO is PPO, unchanged. Same importance ratio pi_new/pi_old, same clip into a band so no single step moves too far. The only new thing is that the advantage comes from a group, not a value network.

Here is the elegant part. If every answer in a group scores the same, the mean equals every reward, every advantage is zero, and the prompt teaches nothing.

So the gradient only appears when a group disagrees, which is exactly the boundary of what the model can currently solve, and where a step helps most. Verifiable rewards and group baselines are a natural match.

Link in comments.

---

## Follow-up 5 (Sun Aug 16) — DeepSeek-R1 and the aha moment

GRPO started as a quiet memory-saving trick. The revolution came when DeepSeek pointed it at a base model with a verifiable reward and let it run.

R1-Zero took a base LLM with no supervised reasoning traces at all: plus one for the correct final answer, a little for clean formatting, zero otherwise. Nothing ever judged the reasoning.

And it grew long chains of thought, self-checking, and backtracking on its own. The paper highlights an "aha moment" where the model writes "wait, let me re-evaluate that step" and fixes its own error. Nobody coded that. Re-checking raises the chance of a correct answer, which raises the reward, which GRPO reinforces. Capability was incentivized, not demonstrated.

Link in comments.

---

## Follow-up 6 (Mon Aug 17) — the fixes, a recap, and the finale (attach fig-grpo-training-curve)

Vanilla GRPO has weak spots, and the fixes are all small edits to the same gradient. DAPO raises the upper clip bound so rare good tokens keep growing, and drops all-same groups. Dr. GRPO removes two biases: dividing by answer length quietly rewards padding, dividing by the group std over-weights easy prompts.

So here is blog 8 in five lines:

- PPO for LLMs holds four models; GRPO deletes the critic and the reward model.
- Drop the critic: use a group's mean reward as the baseline, A_i = (r_i - mean) / std.
- Drop the reward model: for math and code, just check if the answer is correct.
- Everything else is PPO's clip, and a group that agrees gives zero gradient.
- Scaled up with a verifiable reward, this is the recipe behind DeepSeek-R1.

Tomorrow the series moves to blog 9, the finale: DPO and Agentic RL. First it simplifies, reaching the aligned model without a reward model or an RL loop at all. Then it expands, pointing the same gradient at a model that acts over many steps with tools. If you can skip the whole RL loop and still align a model, why run PPO at all?

(Attach: fig-grpo-training-curve.png)

Link in comments.

---

## Notes

- Vary the opening line when you reuse an angle; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep equations readable (for example "A_i = (r_i - mean) / std" and "r = 1 if answer == gold else 0").
- If a GRPO, DAPO, Dr. GRPO, or reasoning-model paper trends this week, quote-post with "the group-relative foundation behind this" plus your link.
