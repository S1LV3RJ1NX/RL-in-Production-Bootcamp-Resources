# Daily Follow-ups — Blog 9: DPO and Agentic RL

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.
- **This run is 6 days** (Wed-Mon): Follow-up 1 = Wed Aug 19, then one per day through Follow-up 6 = Mon Aug 24. On Tuesday Aug 25 the series moves to blog 10.

Blog link: https://prathameshsaraf.com/blogs/09-dpo-and-agentic-rl/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #LLMs #RLHF #LearningInPublic

---

## Follow-up 1 (Wed Aug 19) — fire the critic and the manager

Picture RLHF as a restaurant. A chef (the model) learning to cook, a hired food critic (the reward model) scoring every plate, and a manager (the RL loop) coaching the chef toward higher scores.

DPO fires the critic and the manager, and still gets a better chef.

The reason it works: the critic was trained only on the diner cards (the human preference pairs). It repackages them into a score and adds no new information. If the only truth is the preferences, train the chef on them directly and skip the middleman.

One supervised loss. No reward model, no sampling, no RL loop. Which part of your pipeline would survive that question: what does this component know that its training data doesn't?

Link in comments.

---

## Follow-up 2 (Thu Aug 20) — the policy is secretly a reward model

DPO never trains a reward model, yet the reward is still there. An answer's reward is:

r = beta * log(pi_policy / pi_reference)

How much more likely the policy makes the answer than the frozen reference did. The policy's own probabilities are the reward model; you read the score off the model instead of training a second network.

And this is not a heuristic. Solve the RLHF objective, read the solution backwards, and the reward comes out as exactly this log-ratio plus a term that depends only on the prompt. Preferences always compare two answers to the same prompt, so that leftover term is identical on both sides and cancels.

The awkward normalizer everyone fears in these derivations never has to be computed. It subtracts itself away.

Link in comments.

---

## Follow-up 3 (Fri Aug 21) — the gradient weight is the stability (attach fig-dpo-margin)

The DPO gradient factors into a direction and a weight. The direction is always the same: lift the chosen answer, suppress the rejected one. The weight is the clever part:

weight = beta * sigmoid(-margin)

Mis-ranked pair (negative margin): big weight, hard push. Confidently correct pair (large positive margin): weight near zero, barely a nudge. In the worked example from the post, a pair with margin +0.12 gets weight 0.047, while a mis-ranked pair at -0.5 gets 0.062.

That data-dependent weight is what keeps DPO from degenerating. Remove it and you are pushing "raise the winner, lower the loser" forever, with no signal to stop, and the model drifts off fluent language. Where else does a single scalar quietly do all the stabilizing?

(Attach: fig-dpo-margin.png)

Link in comments.

---

## Follow-up 4 (Sat Aug 22) — SimPO drops the reference, KTO drops the pairs

DPO still asks for two things: a frozen reference model in memory the whole run, and paired data, a chosen and a rejected answer for every prompt. Each popular variation removes one.

SimPO scores an answer by its own average log-probability per token. No reference model. The averaging also fixes a real bias: plain DPO scores by the sum of token log-probs, which grows with length, so the model learns it can move the loss just by rambling.

KTO handles the data problem. Real feedback is rarely a matched A-vs-B pair; it is a lone thumbs-up or thumbs-down, and a comparison loss has nothing to subtract against. KTO borrows the Kahneman-Tversky value function: judge each verdict against a reference point, with losses weighted about 1.5x more than gains. The unpaired signal every product already collects becomes training data.

Which constraint bites harder in your setting, the memory or the labels?

Link in comments.

---

## Follow-up 5 (Sun Aug 23) — the masking rule

An agent's rollout interleaves two kinds of tokens: the ones the model generated (thoughts, tool calls, the answer) and the ones the environment inserted (search results, stack traces, an image patch). They sit side by side in one context.

The rule: compute the policy-gradient loss only on the tokens the model generated. Every observation token gets a loss mask of zero.

Get it wrong and you train the model to predict, and therefore hallucinate, tool outputs it cannot control. It learns to write plausible-looking search results instead of issuing a real search. This is the most common implementation bug in agentic RL, and it is invisible until you check the mask.

The memorable version: you only reinforce decisions, and the agent's only decisions are the tokens it wrote.

Link in comments.

---

## Follow-up 6 (Mon Aug 24) — tool use from reward alone, and a recap (attach fig-env-scaling)

The post ends with a 40-line agentic RL loop. A hidden digit, one search tool that reveals it, a sparse terminal reward, and the GRPO update from blog 8. At step 0 the agent searches 17% of the time. By step 400 it searches 99.8% of the time and names the revealed digit correctly 99.1% of the time. Nobody wrote "search first." The strategy emerged.

So here is blog 9 in five lines:

- The reward model only repackages the preference pairs, so DPO trains on the pairs directly.
- The policy is its own reward model: r = beta * log(pi/pi_ref), and the normalizer cancels.
- The gradient weight pushes hard on mis-ranked pairs and fades on correct ones. That is the stability.
- SimPO drops the reference, KTO drops the pairs. All of it is offline.
- When reward must be earned by acting, point the same gradient at a multi-step world: agents.

Tomorrow the series gets a coda: blog 10 takes everything from posts 7 through 9 and runs it on one production problem, teaching a small model to tutor without revealing answers. Seven alignment recipes enter, one leaves. Any guess which?

(Attach: fig-env-scaling.png)

Link in comments.

---

## Notes

- Vary the opening line when you reuse an angle; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep equations readable (for example "r = beta * log(pi/pi_ref)" and "weight = beta * sigmoid(-margin)").
- If a DPO, KTO, SimPO, or agents paper trends this week, quote-post with "the preference-optimization foundation behind this" plus your link.
