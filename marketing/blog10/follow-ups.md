# Daily Follow-ups — Blog 10: Socratic Alignment

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Aug 26, then one per day through Follow-up 6 = Mon Aug 31.

Blog link: https://prathameshsaraf.com/blogs/10-socratic-alignment/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #LLMs #AI #LearningInPublic

---

## Follow-up 1 (Wed Aug 26) — the one-sided constraint trap

"Be more helpful" and "never reveal the answer" look like the same kind of training target. They are not.

The first is two-sided: the only way to score well is to actually produce useful content. There is no shortcut that looks helpful while being empty.

The second is one-sided. It only says what not to do, so ask what the cheapest way to obey it is. Not the best way, the cheapest. It is to say nothing useful at all. Reply "Great question, what do you think?" to everything and your never-leaks record is flawless while you teach exactly nothing.

I spent a full study watching seven alignment recipes face that trap, and the ones that fell in were not the ones I expected. What "don't do X" constraint are you training against right now?

Link in comments.

---

## Follow-up 2 (Thu Aug 27) — the +1.8 tie

Here are two tutor replies to "how does average pooling differ from max pooling?"

Reply A: "Picture a small grid of numbers: what changes between keeping the biggest one and the typical one?"

Reply B: "Sure, I can help you with that! What do you think?"

I scored both with a rule-based reward: +1.0 for not leaking the answer, +0.5 for asking a question, +0.3 for reasonable length, -1.0 for near-empty. Reply A scores +1.8. Reply B scores +1.8.

A real guiding question and a hollow evasion, indistinguishable to the rule, because the difference between them is not lexical. It is pedagogical, and it lives in the effect on the student. A verifiable rule can enforce the letter of a behavior but never its spirit. What would your reward function miss?

Link in comments.

---

## Follow-up 3 (Fri Aug 28) — reward hacking without a reward model (attach fig-guide-evade)

DPO posted the lowest leakage in my whole study: 0.056, better than SFT. If leakage were the only scoreboard, DPO would be the champion.

Its judge score: 49.4, eight points below the untrained base model. The regression repeats on all four model sizes and survives a beta sweep. The best withholder in the study is the worst teacher.

The mechanism: DPO's loss is purely contrastive. It only wants the gap between chosen and rejected to be large, and there are two ways to widen a gap. Lift the good reply up, or crush the bad reply so hard that near-empty non-answers win by default. On a one-sided constraint the second road is wide open.

Note where this reward hacking surfaced: not in an RL loop with a learned reward model, but inside a supervised-style preference method with no reward model at all. The vulnerability was never in a network. It was in the objective.

(Attach: fig-guide-evade.png)

Link in comments.

---

## Follow-up 4 (Sat Aug 29) — why the RL heavyweights flatlined

The two most expensive runs in my study were GRPO and PPO. Both finished statistically indistinguishable from the untrained base: GRPO at leakage 0.36 and judge 52, PPO at 0.41 and 57.2.

GRPO optimized a clean verifiable rule. PPO optimized a learned reward model. They flatlined together, so the reward source was not the bottleneck. The regime was.

Three forces stacked. The signal is sparse: one nearly-binary scalar per reply, where SFT gets a target at every token. The base model already withholds 57% of the time by accident, so most groups of 4 rollouts look alike and the group-relative advantages are noise. And 60 online steps is far too few for a thin signal to reshape a policy.

The offline anchored methods won not because their reward was smarter but because their supervision was denser. When has dense-but-dumb beaten sparse-but-principled in your training runs?

Link in comments.

---

## Follow-up 5 (Sun Aug 30) — the recipe I would actually ship (attach fig-simpo-sizes)

After 71 GPU configurations, which alignment recipe do you reach for?

KTO posted the single best teaching score on the 0.5B model (76.2, up 18.8 from base). SFT cut leakage hardest. But my pick is SimPO, for one reason: consistency across sizes.

At the ragged edge, a 360M model where most recipes stall or regress (the next best managed +5.1), SimPO still delivered +13.6 on the judge. Top or near-top at every size I tested. Its bill is the study's highest alignment tax, 0.09 on the 1.5B, still under a tenth of the model's capability.

The surprise inside that result: SimPO has no reference model. The careful conservative leash everyone assumes protects a small model turned out, in several cases, to be the thing holding it back. When you only get one shot, do you pick the peak or the floor?

Link in comments.

---

## Follow-up 6 (Mon Aug 31) — the template, and a series recap

Strip the tutoring away from blog 10 and what remains is a template for aligning any behavior:

- An environment that puts the model where the behavior happens (a simulated classroom, here).
- A reward in three layers: a cheap auditable rule, an LLM judge for quality, an outcome metric the model cannot fake.
- A dataset that is both training material and held-out ruler.
- Every recipe under one fixed budget, so the objective is the only variable.
- A before-and-after anchored to the untrained base row.

And the finding worth carrying: the recipe matters more than the act of aligning. Objectives with an anchor toward the good behavior learn to guide. Purely contrastive ones learn to evade. Sparse online ones barely learn at all. My priors said GRPO would win and DPO would be strong. Both were wrong, which is exactly why you build the machine and trust the numbers it returns.

That closes the series: ten posts from the Bellman equation to a production alignment study, one gradient underneath all of it. Which behavior would you point this template at?

Link in comments.

---

## Notes

- Vary the opening line when you reuse an angle; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep numbers readable (for example "leakage 0.43 to 0.076" and "judge +18.8").
- If a DPO-failure-mode, reward-hacking, or LLM-judge paper trends this week, quote-post with "I watched exactly this happen across 71 configs" plus your link.
