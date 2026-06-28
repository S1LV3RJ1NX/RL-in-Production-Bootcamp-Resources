# X Article (Long-Form) — Blog 5: Policy Gradients

## Schedule

- **Date:** Tuesday, July 21, 2026 (same day as the LinkedIn launch)
- **Time:** 9:00 AM IST
- **Self-reply:** Post within 5 minutes of publishing
- **Quote-repost:** Same day, ~5:30-6:00 PM (different hook)

## Article Title

**Policy Gradients: The One Trick That Powers RLHF, PPO, and GRPO**

---

## Article Body (~1,200 words)

Every value method we built so far learns how good each action is, then acts greedily. This week I threw out that middleman and trained the policy directly. That single move is the branch of RL that leads straight to RLHF, PPO, and GRPO, and the whole thing rests on one derivation.

Value-based RL has a quiet catch. To pick a move from a value you either take an argmax over action-values or you need the environment model. Policy gradients drop both. You parameterize the policy, then nudge its parameters to make good actions more likely. No argmax, no model, just gradient ascent on expected reward.

---

### The one derivation: the score-function trick

Start with the objective. The policy is a probability distribution over actions, and we want the parameters that maximize expected reward:

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[R(a)]$$

The problem is the gradient. We want to differentiate an average over actions, but the actions are sampled from the very distribution we are changing, so we cannot push the gradient inside the expectation directly. The score-function trick fixes this with one identity from calculus, that the gradient of a probability equals the probability times the gradient of its log. Apply it and the gradient turns back into an average we can actually sample:

$$\nabla_\theta J = \mathbb{E}\big[\,R(a)\, \nabla_\theta \log \pi_\theta(a)\,\big]$$

Read it plainly: sample an action, see its reward, and push the parameters to make that action more likely, scaled by how good the reward was. Good reward, big push toward the action. Bad reward, a push away. That expectation is REINFORCE, and it is the seed of every policy-gradient method that follows.

---

### One reward, and every logit moves

To see the mechanism with nothing else in the way, the post starts on a one-state, one-shot Archer bandit: a policy over nine aiming angles, one shot, one reward. No states, no discounting, no credit assignment, just the pure gradient.

The surprising part is what one update does. The policy is a softmax over nine logits, so a single reward on a single sampled angle moves all nine at once.

[EMBED IMAGE HERE: fig-logit-gradients.png — one positive bar for the angle that was tried, eight negative bars, all summing to zero]

The angle the archer tried gets its logit pushed up. The other eight get pushed down. The pushes sum to exactly zero, because softmax probabilities always add to one, so raising one must lower the rest. That is the entire REINFORCE update, visible in one picture.

---

### Variance is the enemy, and a baseline is the cure

Plain REINFORCE works, but it is loud. The gradient estimate from a single sampled trajectory swings wildly, so learning is slow and unstable. The fix is to stop scaling by the raw reward and start scaling by how much better the reward was than expected.

Subtract a baseline, the value of the state, and you get the advantage:

$$A_t = G_t - V(s_t)$$

This centers the signal. Without it, when rewards are mostly positive, every action gets pushed up and the policy drifts uphill without separating good from bad. With it, better-than-average actions get a positive push and worse-than-average actions get a negative one. Same expectation, far less noise.

The payoff shows up in numbers. Running the Archer MDP with three methods and measuring the variance of the gradient estimator:

```text
Method          Gradient variance     Greedy return
REINFORCE       0.0437                 4.66
+ baseline      0.0006                 9.60
Actor-Critic    0.0004                 9.62
```

The baseline cuts the variance by orders of magnitude, and only the variance-reduced methods actually solve the task. Lower variance is the difference between learning and not.

---

### Actor-Critic: the template everything else reuses

Give the baseline its own network and let it learn alongside the policy, and you have Actor-Critic. The actor is the policy, it chooses moves. The critic is the value, it judges them but never acts. Its judgement is the advantage, written as a one-step TD error:

$$A = r + \gamma V(s') - V(s)$$

In code it is three tight lines, with the advantage detached so the actor treats it as a fixed weight and cannot cheat by editing the critic:

```python
# advantage = TD target minus the critic's value, detached to a fixed weight
adv = (target - v).detach()                 # A = r + gamma*V(s') - V(s)

# actor loss: maximize A * log pi, negated because optimizers minimize
actor_loss = -(adv * logp).mean() - ent_coef * ent.mean()

# critic loss: regress V(s) toward the TD target
critic_loss = (target - v).pow(2).mean()
```

That is it. The actor climbs the advantage, the critic sharpens its estimate, and the two improve together. Once it is written this way, the advantage-weighted policy gradient becomes the single shape every modern method shares:

$$\nabla_\theta J = \mathbb{E}_{s,a}\big[\,A(s,a)\, \nabla_\theta \log \pi_\theta(a \mid s)\,\big]$$

They differ only in how they estimate A.

---

### Who this is for

If you are working toward RLHF, PPO, or GRPO and the policy-gradient chapter felt like a wall of expectations and log-probabilities, this rebuilds it from one bandit. Derive the score function once, watch one update move every logit, kill the variance with a baseline, add states for credit assignment, and finish on a runnable Actor-Critic. The advantage-weighted log-probability you meet here is the exact objective PPO clips.

---

### What's next

Blog 6 is TRPO and PPO. Actor-Critic gives a clean way to climb expected return, but each update can take a step of any size, and too large a step collapses the policy. Trust regions and clipping keep every update in a safe neighborhood of the old policy. That is the last piece before the RLHF stack.

Full post with typed Python, the bandit build, the variance ladder, and a runnable Actor-Critic: https://prathameshsaraf.com/blogs/05-policy-gradients/

Learning RL for LLMs through the @VizuraAI bootcamp. Follow for the rest of the series.

---

## Header Image

- Use **`blog5-x-banner.png`** (this folder) for the article header once generated. It matches the series template: dark navy with a faint grid, three neon nodes (REINFORCE, baseline, Actor-Critic) converging into a glowing optimal-policy node, with the title and subtitle set on the right.
- Embed `fig-logit-gradients.png` inline in the "every logit moves" section (marked `[EMBED IMAGE HERE]`). X does not support SVG, so export the PNG first: `rsvg-convert -z 2 blogs/05-policy-gradients/images/fig-logit-gradients.svg -o marketing/blog5/fig-logit-gradients.png`.
- Optional inline images: `fig-variance-ladder.png` near the variance section, and `fig-greedy-returns.png` right after it for the payoff.
- Fallback header: `ai-hero.png` from `blogs/05-policy-gradients/images/`.

## First 30 Minutes Strategy

After publishing:

1. Self-reply with: "The whole post in one line. Gradient of an average you cannot differentiate becomes an average of gradients you can sample: grad J = E[ R * grad log pi(a) ]. Sample an action, push it up in proportion to its reward. That is REINFORCE, and everything else is variance reduction on top."
2. Reply to every comment in the first hour.
3. Quote-repost with a one-line hook later the same day (options below).

## Quote-repost hooks (pick one, post ~5:30-6:00 PM the same day)

Hit repost on your own article, choose "Quote," and put one of these on top:

1. "Value methods learn how good each action is, then act greedily. Policy gradients delete the middleman and train the policy directly. That pivot is the whole foundation of RLHF." (recommended: pattern interrupt plus open loop)
2. "One reward, sampled once, moves all nine of the archer's aiming angles at the same time, and the pushes sum to exactly zero. Here is why."
3. "Plain REINFORCE averaged a 4.66 return. The same method with a baseline reached 9.60. One subtraction is the difference between learning and not."
4. "The advantage-weighted log-probability you derive on a toy bandit is the exact objective PPO clips. Here is the full path from one shot to Actor-Critic."

Then reply to anyone who engages, same as the first hour of the original.
