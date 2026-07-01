# X Article (Long-Form) — Blog 8: GRPO

## Schedule

- **Date:** Tuesday, August 11, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**GRPO: Teaching a Model to Reason by Comparing It to Itself**

---

## Article Body (~1,200 words)

RLHF worked. But look at what it has to carry. To align a model with PPO you hold four large models in memory at once: the policy you are training, a frozen reference for the KL leash, a learned reward model that scores answers, and a value-head critic that predicts a per-token baseline. Two of those four are expensive and fragile, and GRPO is the story of removing both.

The payoff is not small. Drop those two passengers, point the result at a base model with a rule-based reward, and you get DeepSeek-R1, a model that learned to reason from reinforcement learning alone.

[EMBED IMAGE HERE: fig-models-in-memory.png — PPO holds four models; GRPO holds policy, reference, and a tiny verifier]

---

### Passenger 1: the critic, and why the group replaces it

The critic exists for one reason: to give a baseline, an estimate of "how well do I usually do on this prompt" that you subtract from the reward to get the advantage. It is a whole second network the size of the policy, and on a long reasoning chain, where the reward only arrives at the final token, its per-token guesses hundreds of tokens early are mostly noise.

Here is the survival fact from the whole series: any estimate of expected reward that does not depend on the action you are scoring is a valid baseline. The critic was just one choice. GRPO picks the simplest one imaginable, the group's own average. Sample a group of G answers to the same prompt, score each, and standardize:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$

The mean is the baseline, measured for free instead of learned by a network. The std just puts every prompt on a comparable scale. It is the textbook Monte-Carlo estimate of the same expectation the critic was trained to predict, computed directly.

```python
import numpy as np

def group_advantages(rewards):
    r = np.asarray(rewards, dtype=float)
    # A_i = (r_i - mean) / std
    return (r - r.mean()) / (r.std() + 1e-8)
```

One consequence matters. Every token of answer i shares the one advantage. It is an outcome-level signal, not a per-token one, which is exactly enough for a checkable answer at the end.

---

### Passenger 2: the reward model, and verifiable rewards

RLHF learned its reward because "a good answer" had no formula. Now change the task. Ask the model to compute 2 + 2 times 3, or solve an equation, or write a function that passes tests. Suddenly "good" has a definition a short program can check. The reward stops being a network and becomes a rule:

$$r(x, y) = \mathbb{1}\big[\text{answer}(y) = \text{gold}(x)\big] \in \{0, 1\}$$

No neural network, no preference data, nothing to fit. A wrong answer scores zero no matter how fluent its reasoning, because style was never on the scoresheet. This is RLVR, reinforcement learning from verifiable rewards, and it is why a GRPO run can optimize hard without collapsing into the reward-hacked nonsense the RLHF post warned about.

---

### The clip is just PPO

With a group advantage and a verifiable reward in hand, the rest of GRPO is PPO, unchanged. Take several gradient steps on the same batch, so correct for the drift with the same importance ratio pi_new/pi_old, and clip it into a band so no single step moves too far. The full objective is a double average of PPO's clipped surrogate, over the group and over each answer's tokens, minus an optional KL penalty:

$$J(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\big(\rho_{i,t}\hat{A}_i,\ \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\big) \;-\; \beta\,D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

If you understood why PPO clips, you already understand most of GRPO. Only two things changed: the advantage now comes from a group instead of a value network, and the KL moved out of the reward and into the loss. For pure-reasoning runs many recipes set beta to zero and cut the leash entirely.

[EMBED IMAGE HERE: fig-group-baseline.png — six reward bars with a dashed group-mean line, advantages measured as deviations]

There is a beautiful catch. If every answer in a group scores the same, the mean equals every reward, every advantage is zero, and the prompt teaches nothing. The signal is strongest exactly when the group is split, which is the boundary of what the model can currently solve, and where a gradient step helps most.

---

### The DeepSeek revolution

GRPO started as a quiet efficiency trick in the DeepSeekMath paper, described as a variant of PPO that saves memory. The revolution came when DeepSeek pointed it at a base model with a verifiable reward and let it run.

R1-Zero took a base LLM with no supervised reasoning traces at all and applied pure GRPO: plus one for the correct final answer, a little for clean formatting, zero otherwise. Nothing ever judged the reasoning. And the model spontaneously grew long chains of thought, self-checking, and backtracking. The paper calls out an "aha moment" where the model writes something like "wait, let me re-evaluate that step," and fixes its own error. Nobody coded that. Re-checking your work raises the chance of a correct answer, which raises the reward, which GRPO reinforces. Capability was incentivized, not demonstrated.

R1 added a small readability warm-up on top, matched OpenAI's o1 on reasoning, and shipped with open weights and a reproducible recipe. That is why GRPO exploded across the field: no human-preference dataset, no closed reward model, cheap to run.

---

### The rough edges get filed down

Vanilla GRPO has known weak spots, and the fixes are all small edits to the same gradient. DAPO decouples the clip and raises the upper bound so rare good tokens keep growing (fighting entropy collapse), and adds dynamic sampling that discards all-same groups. Dr. GRPO points out two biases hiding in the normalizers: dividing by answer length quietly rewards padding, and dividing by the group std over-weights easy prompts. Remove both and you get an unbiased estimator. Every variant tweaks the advantage, the clip, or the sampling, and nothing else.

---

### Does it actually work?

The post measures it rather than asserting it. A 30-line GRPO from scratch on a toy task, no critic and no reward model, climbs a uniform policy from 20% correct to 96% group reward and a 94% probability on the right answer, with nothing but a group baseline and a clip.

[EMBED IMAGE HERE: fig-grpo-training-curve.png — a noisy reward curve rising with a wide in-group spread band; KL to reference staying tiny]

Then the real run: GRPO on GSM8K with a Llama-3.2-3B model, only LoRA adapters training (1.49% of the weights), on a single 14.5 GB T4. No value head, no reward model, five plain Python reward functions. The reward is noisy and mostly driven by format early on, correctness is non-zero in only 19 of 250 steps on a small model in two hours, and the KL to reference stays around 0.01 the whole time. Honest and instructive: the machinery is exactly DeepSeek's, the scale is not.

---

### Who this is for

If GRPO looks like a new algorithm you have to learn from scratch, this shows it is PPO with the critic swapped for a group and the reward swapped for a checker. It builds both deletions from the baseline argument up, works the group advantage by hand, and runs the whole thing twice, once tiny on a CPU and once real on GSM8K.

---

### What's next

Blog 9 closes the series with DPO and Agentic RL. First it simplifies: Direct Preference Optimization reaches the same aligned model as RLHF while firing both the reward model and the RL loop. Then it expands: point the same group-relative gradient at a multi-step world where the model stops answering and starts acting, calling tools and earning a reward only at the end.

Full post with typed Python, the group-advantage build, the DeepSeek-R1 story, and a runnable GRPO on GSM8K: https://prathameshsaraf.com/blogs/08-grpo/

Learning RL for LLMs through the @VizuraAI bootcamp. Follow for the finale.

---

## Header Image

- Use **`blog8-x-banner.png`** (this folder) for the article header. It matches the series template: dark navy with a faint grid, three neon nodes (GROUP BASELINE, VERIFIABLE REWARD, PPO CLIP) converging into a glowing "DEEPSEEK-R1" node, with the title and subtitle set on the right.
- Embed `fig-models-in-memory.png` in the opening section. X does not support SVG, so export the PNG first: `rsvg-convert -z 2 blogs/08-grpo/images/fig-models-in-memory.svg -o marketing/blog8/fig-models-in-memory.png`.
- Embed `fig-group-baseline.png` in the "the clip is just PPO" section. Export with: `rsvg-convert -z 2 blogs/08-grpo/images/fig-group-baseline.svg -o marketing/blog8/fig-group-baseline.png`.
- Embed `fig-grpo-training-curve.png` in the "does it actually work" section. Export with: `rsvg-convert -z 2 blogs/08-grpo/images/fig-grpo-training-curve.svg -o marketing/blog8/fig-grpo-training-curve.png`.
- Optional inline images: `fig-all-same-no-signal.png` near the split-group paragraph, and `ai-aha-moment.png` near the DeepSeek section.
- Fallback header: `ai-hero.png` from `blogs/08-grpo/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. GRPO is PPO with two deletions: drop the critic and use a group's mean reward as the baseline, drop the reward model and just check if the answer is correct. Same clip, cheaper advantage, and the recipe behind DeepSeek-R1."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "PPO for LLMs holds four models in memory. DeepSeek deleted two of them and got a reasoning model. Here is exactly what GRPO removes and why it still works." (recommended: pattern interrupt plus open loop)
2. "GRPO's whole trick: your baseline is not a learned network, it is the average of a few answers you already sampled. The critic was overhead all along."
3. "R1-Zero was never taught to reason. It was rewarded for correct answers, and checking its own work raised that reward. The 'aha moment' emerged on its own."
4. "A GRPO group that all agrees teaches nothing. The gradient only shows up when the answers disagree, right at the edge of what the model can solve."

Then reply to anyone who engages, same as the first hour of the original.
