# Daily Follow-ups — Blog 6: TRPO & PPO

Copy-paste posts to keep one blog alive for a whole week, one angle per day, on both LinkedIn and X. The big posts live in `linkedin.md` and `x-article.md`. This file is everything in between.

## How to use

- Post one item per day, around 10:30 AM IST. Both platforms can run the same angle.
- **LinkedIn:** paste the text (it ends with "Link in comments."), then put the blog link in the first comment, and add the hashtags at the bottom.
- **X:** delete the "Link in comments." line, paste the text, and drop the blog link in a self-reply. No hashtags.
- The closing question doubles as your first self-reply. Reply to every comment in the first hour.

Blog link: https://prathameshsaraf.com/blogs/06-trpo-ppo/
Hashtags (LinkedIn): #ReinforcementLearning #MachineLearning #PPO #RLHF #LearningInPublic

---

## Follow-up 1 — a bad RL step spirals

In supervised learning a bad training step is cheap. The data is fixed, so the next batch pulls you back. Mistakes are reversible.

RL has no such safety net. The policy chooses its own future data, so a reckless update wrecks the policy, the wrecked policy collects wrecked data, and the next update is built on that wreckage. The loop can spiral with no way back.

That is why "just lower the learning rate" is not enough: tiny steps everywhere are safe but painfully slow. We want the largest step that is still safe. Have you ever watched a training run collapse and never recover?

Link in comments.

---

## Follow-up 2 — reuse the batch with one ratio

Plain policy gradients use a batch once and throw it away, because the gradient is an average under the current policy. Update once and the data is stale.

Importance sampling buys it back. If you want an average under the policy you want but only have samples from the policy you had, reweight each sample by the ratio of the two:

r = pi_new(a|s) / pi_old(a|s)

r above 1 means the new policy likes that action more, below 1 means less. That single number lets you score a brand new policy on data the old one already collected. It is the most important symbol in the whole post.

Link in comments.

---

## Follow-up 3 — the surrogate you actually climb

Take that ratio, multiply by the advantage, average over the batch, and you have the objective TRPO and PPO optimize:

L = E[ r * A ]

Reweight each old sample by how much the new policy favors its action, scale by how good the action was, and average. Push good actions up, bad ones down, all on data you already have.

The catch: the ratio is only trustworthy while the new policy stays close to the old one. Step too far and the surrogate over-promises. So the objective needs a leash, and that leash is the whole story of TRPO and PPO.

Link in comments.

---

## Follow-up 4 — TRPO builds a fence

TRPO enforces the leash with a hard constraint. It measures policy change with KL divergence, which watches the whole action distribution at each state, not just the sampled action:

maximize E[ r * A ]  subject to  average KL(pi_old || pi_new) <= delta

Maximize the surrogate, keep the average KL under a small budget like 0.01. It worked: it made big neural policies stable on hard control tasks for the first time.

The cost is real though. Solving that exactly needs second-order optimization re-run every update, and it squeezes out about one step per batch. That price is exactly what PPO set out to avoid.

Link in comments.

---

## Follow-up 5 — PPO clips instead

PPO asks: what if the objective itself refused to reward a big step? Then there is no constraint to solve.

Clip the ratio into a band [0.8, 1.2] (epsilon = 0.2) and take the pessimistic side:

L_clip = E[ min( r * A , clip(r, 0.8, 1.2) * A ) ]

For a good action the objective climbs until r hits 1.2, then goes flat. Zero gradient. Once you have made the action 20% more likely on this batch, PPO gives you nothing for pushing further. The trust region is drawn by the shape of the loss, not a separate solver. A few lines of plain SGD.

Link in comments.

---

## Follow-up 6 — the safety valve

PPO's clip is deliberately asymmetric, and that asymmetry is the clever part.

For a bad action that has somehow grown much more likely (a large ratio), the objective keeps dropping instead of flattening, so the gradient keeps pulling it back down. Clipping there would strand a mistake.

But once a bad action is already cut about 20%, the clip stops you from suppressing it further on one noisy batch. So the clip caps how hard you suppress an action, but never how hard you undo an accidental increase. Pessimism, by design.

Link in comments.

---

## Follow-up 7 — does it work, and which piece matters (attach fig-ablation)

PPO swings a pendulum from hanging down to upright, smoothed return climbing from about -1137 to -704. But which ingredient does the work? Turn one off at a time.

- Full PPO: the only curve that steadily climbs.
- No clip: the worst and most unstable run, lurching up and down as unbounded updates undo each other.
- No reuse: smooth but flat, learning about a tenth as fast for the same data.

Clip matters most, reuse matters second. Full PPO is a trust region from clipping plus sample efficiency from reuse.

(Attach: fig-ablation.png)

Link in comments.

---

## Follow-up 8 — recap and the bridge to RLHF

Blog 6 in five lines:

- The gradient gives a direction, not a step size, and in RL a step too big can collapse the policy.
- Importance sampling reweights old data with the ratio r = pi_new / pi_old, so a batch can be reused.
- The surrogate L = E[r * A] scores a new policy on the old policy's data.
- TRPO fences it with a hard KL constraint, PPO clips the ratio so going far earns nothing.
- Measured on Pendulum, the clip is the piece that makes it stable.

Next up is blog 7: RLHF. On Pendulum the reward came from the environment. How do you score "be helpful and honest" when no simulator can? That is where this clipped PPO update meets a reward model learned from human comparisons. Where do you think the reward should come from for an LLM?

Link in comments.

---

## Notes

- Vary the opening line on reuse; identical reposts on one platform get penalized.
- Plain text only in these posts, no LaTeX. Keep equations readable (for example "L = E[r * A]" and "r = pi_new / pi_old").
- If a PPO, TRPO, or RLHF paper trends this week, quote-post with "the trust-region foundation behind this" plus your link.
