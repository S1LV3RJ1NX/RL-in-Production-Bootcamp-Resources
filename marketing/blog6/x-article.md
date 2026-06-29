# X Article (Long-Form) — Blog 6: TRPO & PPO

## Schedule

- **Date:** Tuesday, July 28, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**TRPO and PPO: The Largest Safe Step a Policy Can Take**

---

## Article Body (~1,200 words)

The gradient tells you which way to move the policy. It says nothing about how far. In supervised learning that gap barely matters, but in RL it is the whole game, and getting it wrong does not just slow you down, it can end the run.

Here is why RL is different. In supervised learning the dataset is fixed: overshoot on one batch and the next batch of the same data pulls you back. A bad step is reversible. In RL the policy chooses its own future data. A reckless update wrecks the policy, the wrecked policy collects wrecked data, and the next update is built on that wreckage. The feedback loop can spiral with no recovery.

So "just lower the learning rate" is a weak answer. A tiny step everywhere is safe but painfully slow. What we actually want is the largest step that is still safe, and that is exactly what TRPO and PPO deliver.

---

### Importance sampling: stop throwing batches away

Plain policy gradients use each batch once. The gradient is an average under the current policy, so the instant you update, the batch is stale and gets discarded. That is expensive.

Importance sampling is the one identity that buys the batch back. If you want the average of a function under distribution p but can only sample from q, reweight each sample by p over q:

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\!\left[\frac{p(x)}{q(x)} \cdot f(x)\right]$$

Apply it with p as the new policy and q as the old one, and the correction factor becomes a single ratio:

$$r = \frac{\pi_\text{new}(a \mid s)}{\pi_\text{old}(a \mid s)}$$

When r is 1 the new policy agrees with the old. Above 1 it likes the action more, below 1 it likes it less. That one ratio is the most important symbol in the post. It lets you score a brand new policy using data the old one already collected.

---

### The surrogate objective

Multiply the ratio by the advantage and average over the batch, and you get the objective TRPO and PPO actually optimize:

$$L(\theta) = \mathbb{E}\big[r_t(\theta)\, A_t\big]$$

Read it plainly: reweight each old sample by how much the new policy favors its action, scale by how good that action was, and average. Push advantageous actions up, push bad ones down, all measured on the batch you already have.

There is a catch hiding in it. The ratio is only trustworthy while the new policy stays close to the old one. Walk too far and the reweighting becomes a wild extrapolation, and the surrogate over-promises improvement that is not really there. So the objective needs a leash.

---

### TRPO: build a fence

TRPO enforces the leash with a hard constraint. It needs a ruler for how different two policies are, one that looks at the whole action distribution at each state, not just the sampled action. That ruler is KL divergence. The update is:

$$\max_\theta \; \mathbb{E}\big[r_t\, A_t\big] \quad \text{s.t.} \quad \mathbb{E}\big[D_\text{KL}(\pi_\text{old} \,\|\, \pi_\theta)\big] \le \delta$$

Maximize the surrogate, but keep the average KL under a small budget like 0.01. Two clean jobs: r times A picks the direction and size of improvement, the KL cap limits how far the policy may actually move.

TRPO worked. It made large neural policies stable on hard control tasks for the first time. But solving "maximize L subject to KL below delta" exactly needs second-order optimization re-run every update: natural gradients, conjugate gradient, a line search. It is heavy, fiddly, awkward with a shared actor-critic network, and it squeezes out only about one update per batch.

---

### PPO: clip instead of constrain

PPO asks a sharper question. What if the objective itself refused to reward a big step? Then there is no constraint left to solve. The trick is to clip the ratio:

$$L^\text{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\big(r_t A_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\, A_t\big)\right]$$

with epsilon around 0.2. Two pieces do the work. The clip pins the ratio inside [0.8, 1.2]; cross the band and the objective goes flat, so the gradient is zero. The min takes the more pessimistic of the clipped and unclipped terms, so a large ratio can never inflate the objective.

[EMBED IMAGE HERE: fig-clip-positive.png — for a good action the objective rises with r, then flattens at r = 1 + epsilon]

For a good action you want r above 1, and the objective climbs until r hits 1.2, then flattens. Once you have made the action 20% more likely, PPO gives you nothing for pushing further on this batch. The trust region is drawn by the shape of the loss, not by a separate solver.

The bad-action side is asymmetric on purpose:

```python
import torch

ratio = torch.tensor([0.95, 1.10, 1.50, 0.60])
advantage = torch.tensor([2.0, 2.0, -2.0, -2.0])
eps = 0.2

# L^CLIP = min(r*A, clip(r, 1-eps, 1+eps)*A)
unclipped = ratio * advantage
clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
objective = torch.min(unclipped, clipped)
```

When a bad action (A below 0) has grown much more likely (r = 1.5), the unclipped term is more negative, so min keeps it and the gradient keeps pulling that action back down. Clipping there would strand a mistake. But once a bad action is already cut about 20% (r = 0.6), the clip stops you from over-suppressing it on one noisy batch. The clip caps how hard you suppress, never how hard you undo an accidental increase.

Because every step is bounded, you can safely reuse the same batch for several epochs of cheap SGD instead of one expensive second-order step. That is PPO: TRPO's safety, with a fraction of the complexity.

---

### Does it actually work?

Yes, and the post measures it rather than asserts it. Running the full PPO program on Pendulum-v1, the policy goes from hanging the pole down to swinging it up and holding it near vertical, with the smoothed return climbing from about -1137 to -704 and a greedy evaluation at -619.

[EMBED IMAGE HERE: fig-ablation.png — full PPO climbs; remove the clip and it is the lowest and most jagged curve]

Then the ablation: turn off one ingredient at a time. Remove the clip and it is the worst and most unstable run, lurching up and down as unbounded updates undo each other. Remove batch reuse and the curve is smooth but flat, learning a tenth as fast for the same data. Clip matters most, reuse matters second. Full PPO is a trust region from clipping plus sample efficiency from reuse.

---

### Who this is for

If you are heading toward RLHF and PPO looks like a wall of ratios, KL terms, and clip functions, this rebuilds it from the step-size problem up. Derive importance sampling once, write the surrogate, see why TRPO fences it with KL, then watch the clip turn that fence into a few lines of SGD. The clipped objective you build here is the exact update that aligns large language models.

---

### What's next

Blog 7 is RLHF. On Pendulum the reward came from the environment. But how do you score "be helpful and honest" when no simulator can? The next post turns text generation into an MDP, learns a reward model from human comparisons, then runs exactly this clipped PPO update to optimize it, with a KL-to-reference leash keeping the model honest.

Full post with typed Python, the importance-sampling build, the clip figures, and a runnable PPO on Pendulum: https://prathameshsaraf.com/blogs/06-trpo-ppo/

Learning RL for LLMs through the @VizuraAI bootcamp. Follow for the rest of the series.

---

## Header Image

- Use **`blog6-x-banner.png`** (this folder) for the article header. It matches the series template: dark navy with a faint grid, three neon nodes (SURROGATE, KL FENCE, CLIP) converging into a glowing "SAFE STEP" node, with the title and subtitle set on the right.
- Embed `fig-clip-positive.png` inline in the "clip instead of constrain" section (marked `[EMBED IMAGE HERE]`). X does not support SVG, so export the PNG first: `rsvg-convert -z 2 blogs/06-trpo-ppo/images/fig-clip-positive.svg -o marketing/blog6/fig-clip-positive.png`.
- Embed `fig-ablation.png` in the "does it actually work" section. Export with: `rsvg-convert -z 2 blogs/06-trpo-ppo/images/fig-ablation.svg -o marketing/blog6/fig-ablation.png`.
- Optional inline images: `fig-clip-negative.png` near the safety-valve paragraph, and `fig-pendulum-curve.png` right before the ablation for the payoff.
- Fallback header: `ai-hero.png` from `blogs/06-trpo-ppo/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. The gradient gives a direction, not a step size, and in RL a step too big collapses the policy. PPO clips the ratio pi_new/pi_old into [0.8, 1.2]: past the band the gradient is zero, so SGD simply never wanders far. A trust region for free."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "In supervised learning a bad step is cheap. In RL it can end the run, because the policy picks its own next data. Here is the fix the whole field standardized on." (recommended: pattern interrupt plus open loop)
2. "PPO is three lines around one idea: clip the ratio pi_new/pi_old to [0.8, 1.2]. Past the band the gradient is zero, so the policy never wanders far. That is the trust region, for free."
3. "Turn off PPO's clip and it becomes the worst, most unstable run on the board. One min() is the difference between learning and collapse."
4. "This is the exact algorithm that taught ChatGPT to follow instructions, derived from the step-size problem up. No magic, one ratio and one clip."

Then reply to anyone who engages, same as the first hour of the original.
