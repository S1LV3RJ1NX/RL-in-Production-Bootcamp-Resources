# X Article (Long-Form) — Blog 9: DPO and Agentic RL

## Schedule

- **Date:** Tuesday, August 18, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**DPO and Agentic RL: Align Without a Reward Model, Then Step Into the World**

---

## Article Body (~1,300 words)

The RLHF pipeline trains a reward model on human preference pairs, then runs PPO against it. DPO's authors asked one uncomfortable question about that pipeline: the reward model was trained only on the pairs, so what does it know that the pairs don't?

Nothing. It repackages the same preferences into a score and invents no new signal. That single observation deletes half the alignment stack, and it is where the final post of this series starts. The post makes two moves in opposite directions: first simplify, aligning a model with one supervised loss and no RL loop at all, then expand, pointing the same gradient at a multi-step world where the model calls tools and gets judged only at the end.

---

### The policy is its own reward model

The naive idea, "raise the chosen answer's probability, lower the rejected one's," drifts and degenerates because it has no anchor and never stops. DPO's fix is to score an answer by how much more likely the policy makes it than a frozen reference did:

$$\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

No separate network. The policy's own probabilities, read against the reference, are the reward. The loss is just the Bradley-Terry ranking loss with this implicit reward inside:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\big(\hat{r}(x, y_w) - \hat{r}(x, y_l)\big)$$

Here is one real pair worked through. The policy scores a prudent financial answer at log-prob -1.4 where the reference had -2.2, and a reckless one at -2.2 where the reference had -1.8. With beta 0.1 the implicit rewards are +0.08 and -0.04, the margin is +0.12, and the loss is 0.635, just under the coin-flip value of 0.69. The model ranks the pair correctly but not confidently, so it gets a small nudge.

```python
import math

def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta=0.1):
    r_w = beta * (logp_w - ref_w)   # implicit reward, chosen
    r_l = beta * (logp_l - ref_l)   # implicit reward, rejected
    margin = r_w - r_l
    return -math.log(1 / (1 + math.exp(-margin)))

print(dpo_loss(-1.4, -2.2, -2.2, -1.8))  # 0.635
```

[EMBED IMAGE HERE: fig-dpo-margin.png — the DPO loss falls as the margin grows; the gradient weight is high for mis-ranked pairs and fades once a pair is confidently correct]

The gradient is where DPO earns its stability. It factors into a direction (lift the winner, suppress the loser) times a weight, beta times sigma of the reversed margin. The weight is large when a pair is currently mis-ranked and shrinks toward zero once it is confidently correct. DPO spends its gradient on the pairs it gets wrong. Delete that weight and you are back to pushing forever with no signal to stop.

---

### Why this is RLHF in disguise

The implicit reward looks too convenient to be legal. It isn't a guess; it drops out of the RLHF objective in three moves. The KL-constrained objective has a known optimal policy, the reference tilted toward higher-reward answers. Read that backwards and the reward equals the implicit reward plus a term that depends only on the prompt. And preferences never use a reward alone, they always compare two answers to the same prompt, so the prompt-only term is identical on both sides and cancels.

What remains is exactly the DPO margin. So DPO optimizes the exact RLHF objective with a single classification loss. The paper's title says it straight: your language model is secretly a reward model.

---

### Two variations, two dropped requirements

DPO still asks for a frozen reference model in memory and paired data. Each variation removes one.

**SimPO** drops the reference. It scores an answer by its own average log-probability per token, which also fixes DPO's length bias (a sum over tokens grows with length, so plain DPO quietly rewards rambling) and matches the quantity decoding actually ranks by. The price: no reference means no implicit KL anchor, so a target margin and the normalization do the stabilizing.

**KTO** drops the pairs. Real feedback is mostly lone thumbs-up and thumbs-down verdicts, and a comparison loss has nothing to subtract against. KTO borrows the Kahneman-Tversky value function from behavioral economics: judge each verdict as a gain or loss against a reference point, with losses weighted more heavily than gains. A dislike avoided matters more than a like earned, and the abundant unpaired signal every product already collects becomes training data.

The whole family shares one ceiling though. It is offline. It learns from a fixed dataset and never samples the model. The moment a reward must be earned by acting, there are no fixed pairs to learn from, and that is the second half.

---

### From answering to acting

Everything so far aligns a model that answers once: prompt in, response out, scored a single time. An agent faces a genuinely bigger problem. At each turn it emits an action, text or a tool call. The environment returns an observation the agent did not choose. The reward usually arrives once, at the very end, as a single 0 or 1 for the whole episode.

DeepEyes is the cleanest published example. A vision-language model cannot read a distant road sign because the image encoding blurred it away. Give it one tool, zoom-in, and train with pure RL (no supervised demonstrations of tool use at all): accuracy reward for the right answer, a small bootstrap reward for actually using the tool early on, a format reward so the calls parse. Nobody ever showed it "first zoom, then answer." The strategy emerged because zooming led to correct answers, which earned reward.

[EMBED IMAGE HERE: fig-credit-assignment.png — one terminal reward, credited back over a six-step trajectory]

The same recipe recurs everywhere with the tool swapped out: Search-R1 (live search calls, +41% over RAG baselines), ReTool (run code mid-reasoning), UI-TARS (mouse and keyboard on real GUIs), SWE-RL (fix a GitHub issue, reward is whether the tests pass). Widen the action space, let the agent take many steps, reward the verifiable outcome, optimize with the GRPO machine from blog 8.

---

### The rule that catches the most common bug

A rollout interleaves tokens the model generated and tokens the environment inserted (search results, stack traces, an image patch). The policy-gradient loss is computed only on the model's own tokens. Every observation token gets a loss mask of zero.

Get this wrong and you train the model to predict, and therefore hallucinate, tool outputs it cannot control. It learns to write plausible-looking search results instead of issuing a real search. This is the most common implementation bug in agentic RL, and it is invisible until you check the mask. The memorable version: you only reinforce decisions, and the agent's only decisions are the tokens it wrote.

---

### Does it actually work?

The post ends with a 40-line agentic loop you can run on a laptop. A hidden digit, one search tool that reveals it, a sparse terminal reward, and the GRPO update. At step 0 the agent searches 17% of the time and guesses right 17% of the time even after seeing the digit. By step 400 it searches 99.8% of the time and names the seen digit 99.1% of the time. Nobody wrote "search first." The two-step strategy emerged from reward alone, the DeepEyes story in miniature.

One more number worth sitting with: RL environment counts across model generations grew from about 20 to millions, and the Qwen team attributes most of Qwen3.5's post-training gain to environment scale and diversity, not a bigger model. The model is no longer the scarce resource. The environments are.

[EMBED IMAGE HERE: fig-env-scaling.png — environment counts growing from ~20 to millions on a log axis]

---

### Who this is for

If DPO looks like a trick and agents look like a different field, this post shows both are the same gradient. It derives the DPO loss from the RLHF objective, works one preference pair through every number, builds SimPO and KTO from DPO's leftover problems, and ends with a runnable multi-turn agent that learns tool use from a sparse reward.

---

### The series ends here

Nine posts, one sentence underneath all of them: the value of where I am is the reward I just got, plus a discounted value of where I'll land next. DPO is the policy-gradient update with the advantage written as a preference margin. GRPO is it with a group baseline. Agentic RL is it with a tool call for an action and a sparse terminal reward propagated back over a trajectory. One gradient, nine posts.

Full post with the three-move derivation, every worked number, and the runnable agent: https://prathameshsaraf.com/blogs/09-dpo-and-agentic-rl/

Learning RL for LLMs through the @VizuraAI bootcamp. The full series is on the same site.

---

## Header Image

- Use **`blog9-x-banner.png`** (this folder) for the article header. It matches the series template: dark navy with a faint grid, neon nodes (PREFERENCES, ONE LOSS, TOOL CALLS) converging into a glowing "ALIGNED AGENT" node, with the title and subtitle set on the right.
- Embed `fig-dpo-margin.png` after the DPO loss section (already exported in this folder).
- Embed `fig-credit-assignment.png` in the "from answering to acting" section.
- Embed `fig-env-scaling.png` near the closing environment-scaling paragraph.
- Optional inline images: `fig-simpo-length.png` in the SimPO paragraph, `fig-kto-value-curve.png` in the KTO paragraph.
- Fallback header: `ai-hero.png` from `blogs/09-dpo-and-agentic-rl/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. The reward model only ever repackaged the preference pairs, so DPO trains on the pairs directly with one supervised loss, and the moment a reward must be earned by acting instead, you point the same gradient at a multi-step world and get an agent."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "The reward model in RLHF is trained only on the preference pairs. So what does it know that the pairs don't? Nothing. Here is the loss that follows from taking that seriously." (recommended: pattern interrupt plus open loop)
2. "DPO's stability lives in one scalar: the gradient weight is large on pairs the model mis-ranks and near zero on pairs it already gets right. Delete it and the model drifts."
3. "A 40-line RL loop, a hidden digit, and a search tool. Nobody says 'search first.' By step 400 the agent searches 99.8% of the time. Tool use emerges from reward alone."
4. "RL environments went from about 20 to millions in four model generations. The model stopped being the scarce resource."

Then reply to anyone who engages, same as the first hour of the original.
