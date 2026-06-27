---
title: "Policy Gradients: Learning the Policy Directly, from a Bandit to Actor-Critic"
shortName: "Policy Gradients"
date: "2026-06-17"
summary: "Stop learning values and reading a policy off them: parameterize the policy and climb expected reward directly. We derive the score-function trick on a one-state Archer bandit, watch one softmax update touch all nine logits, kill the variance with a baseline, add states and credit assignment, and end on an Actor-Critic that solves the Archer MDP."
tags:
  [
    "reinforcement-learning",
    "policy-gradients",
    "reinforce",
    "actor-critic",
    "advantage",
    "gymnasium",
  ]
order: 5
---

# Policy Gradients: Learning the Policy Directly, from a Bandit to Actor-Critic

![An archer mid-draw aiming at a distant target, with a coach standing just behind observing, in a flat editorial illustration style with terracotta accents](./images/ai-hero.png)

> **The throughline:** _The value of where I am is the reward I just got, plus a discounted value of where I'll land next._
> The [SARSA, Q-learning & DQN](../04-sarsa-qlearning-dqn/README.md) post used that sentence to learn $Q(s,a)$ and read a policy off it by argmax. This post throws out the middleman: **learn the policy $\pi(a \mid s;\theta)$ directly** and climb expected reward by gradient ascent. The throughline becomes the critic's job, while the actor learns _what to do_.

## 1. The intuition

For two lectures we estimated a value and let it choose actions for us: learn $Q(s,a)$, then act by $\arg\max_a Q(s,a)$. That pipeline has a hidden assumption: the action set is small and discrete, so the argmax is just "compare a few numbers and pick the biggest."

**What if we skip the value and optimize the policy itself?**

There are two routes to a policy:

- **Value-based (DQN, last lecture).** Learn $Q(s,a)$, then read the policy off it: $\pi(s) = \arg\max_a Q(s,a)$. The policy is _implicit_, a by-product of the values.
- **Policy-based (this post).** Parameterize the policy itself, $\pi_\theta(a \mid s)$ (a network that outputs action probabilities), and adjust $\theta$ to make good actions more likely. The policy is the thing you learn.

Three concrete situations break the argmax:

**1. Continuous actions.** A robot arm's torque is a real number. You cannot take an argmax over infinitely many actions. A policy sidesteps this by outputting the mean and spread of a Gaussian, $a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$, and sampling. One forward pass gives you the move.

**2. Stochastic optimal policies.** In rock-paper-scissors, the only unbeatable strategy is uniform $\frac{1}{3}, \frac{1}{3}, \frac{1}{3}$. A deterministic argmax can never output "play each move a third of the time." A stochastic policy can.

**3. Smooth, direct optimization.** A tiny change in $Q$ can flip the argmax from one action to another, a discontinuous jump. Policy gradients move smoothly: a small step in $\theta$ nudges $\pi$, it never snaps it. And they climb expected return $J(\theta)$ directly, not a Bellman-error surrogate. (Recall the deadly triad from the [SARSA, Q-learning & DQN](../04-sarsa-qlearning-dqn/README.md) post: function approximation + bootstrapping + off-policy data can diverge. Policy gradients avoid off-policy data entirely.)

**The honest cost.** Policy gradients are _on-policy_ (each batch of experience is used once, then thrown away, so no replay buffer) and _high-variance_ (the gradient is estimated from noisy sampled returns). The rest of this post earns the method and then pays that cost down.

### The Archer: the simplest possible RL problem

For most of this post the world has exactly one state. The **Archer** stands at a fixed spot and picks one of 9 discrete release angles $a_1 \dots a_9$. After shooting, the episode is over: reward = how close the arrow lands to the target (+1 dead centre, falling off with distance).

A one-state, one-shot problem is called a **bandit**. Your action changes the reward, but never the next state, because there is no next state. That single property strips away returns, discounting, and credit assignment. What is left is the pure policy-gradient mechanism. We add states back in the later sections, once the mechanism is airtight.

```mermaid
flowchart LR
    subgraph actorCritic [Actor-Critic loop]
        S["state s"] --> Actor["Actor π(a|s;θ)"]
        S --> Critic["Critic V(s;w)"]
        Actor -->|"sample action a"| Env["Environment"]
        Env -->|"reward r, next state s'"| Adv["Advantage A = r + γV(s') − V(s)"]
        Adv -->|"weight on ∇log π"| Actor
        Adv -->|"TD target for V"| Critic
    end
```

This diagram is the destination. We start at the top-left (state in, action out) and build every arrow by the end of Section 2.

<details>
<summary><strong>Check:</strong> The best policy in rock-paper-scissors is randomized. In one line, why can a stochastic policy represent "play each move a third of the time" but an argmax-over-Q cannot?</summary>

**Answer.** An argmax always returns one fixed action, so it can only play one move, which an opponent then exploits. A stochastic policy outputs a probability for every action, so it can literally encode 1/3, 1/3, 1/3. Representing randomness needs a distribution, and only the policy gives you one.

</details>

<details>
<summary><strong>Check:</strong> With a softmax policy we said exploration is "built in." Where does that exploration actually come from?</summary>

**Answer.** From sampling. The policy is a probability distribution over actions and we draw the action from it, so we sometimes try non-top actions on our own. The randomness lives inside the policy; there is no separate epsilon to schedule by hand.

</details>

<details>
<summary><strong>Check:</strong> A robot arm's torque is a real number. Why does argmax-over-Q break here, and what does a policy output instead?</summary>

**Answer.** You cannot take an argmax over infinitely many actions, so value-based control has nothing to maximize over. A policy sidesteps that by outputting a distribution you can sample. For continuous actions, it outputs the mean and spread of a Gaussian.

</details>

<details>
<summary><strong>Check:</strong> For almost all of this post the world has just one state (a bandit). What does having "no next state" let us ignore?</summary>

**Answer.** Everything about the future: returns, discounting, and credit assignment. With no next state your action only changes this one reward, so the policy-gradient mechanism appears in its purest form. We add states (and those three things) back in Section 2.7.

</details>

---

## 2. The math you need

### 2.1 The objective: expected reward

The goal is a single number: the expected reward under the current policy.

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[R(a)] = \sum_a \pi_\theta(a) \cdot R(a)$$

Read it aloud, symbol by symbol: "$J$ of $\theta$ equals the expected value ($\mathbb{E}$) of the reward $R(a)$, when the action $a$ is drawn from the policy $\pi_\theta$ (that is what $a \sim \pi_\theta$ means); and that expected value is the same thing as the sum, over every action $a$, of $\pi_\theta(a)$ times $R(a)$." The two halves are equal because an expectation _is_ a probability-weighted average: you weight each action's reward by how often the policy plays it.

$J$ is a function of the policy parameters $\theta$: change $\theta$, the probabilities shift, the weighted sum changes, and $J$ goes up or down. It goes up when we put more probability on higher-reward actions. **The whole goal: gradient ascent on $J$.**

$$\theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)$$

We _maximize_ reward, so we move _along_ the gradient (ascent, the **+** sign), not against it. In code, optimizers minimize, so we descend on the loss $-J$: the minus sign you will see in every REINFORCE loop.

### 2.2 The obstacle: you cannot just differentiate the sum

We want $\nabla_\theta J$, the gradient of $J$ with respect to the parameters $\theta$. A "gradient" is just the collection of derivatives, one per parameter. It points in the direction that makes $J$ bigger, which is exactly the direction the update rule above wants to step. So we need to differentiate $J(\theta) = \sum_a \pi_\theta(a) \cdot R(a)$ with respect to $\theta$.

Two basic rules of derivatives are all we need:

1. **The derivative of a sum is the sum of the derivatives.** We are allowed to reach inside the $\sum_a$ and differentiate one term at a time, then add the results back up.
2. **A constant multiplier stays put.** Inside each term, $R(a)$ does not depend on $\theta$: the reward an action pays is fixed by the environment, while $\theta$ only changes how _likely_ we are to pick that action. So $R(a)$ is just a constant here, and a constant in front of a derivative comes along unchanged (the same way $\frac{d}{dx}(5x) = 5$). We differentiate only the $\theta$-dependent piece, $\pi_\theta(a)$.

Applying both, term by term:

$$\nabla_\theta J = \nabla_\theta \sum_a \pi_\theta(a) R(a) = \sum_a \nabla_\theta \big[\pi_\theta(a) R(a)\big] = \sum_a R(a) \cdot \nabla_\theta \pi_\theta(a)$$

Read the final result aloud, symbol by symbol: "the gradient with respect to $\theta$ of $J$ ($\nabla_\theta J$) equals the sum, over every action $a$, of $R(a)$ times the gradient with respect to $\theta$ of $\pi_\theta(a)$ ($\nabla_\theta \pi_\theta(a)$)." In plain English: to raise $J$, push on each action's probability in proportion to the reward that action earns, big rewards get a big push, small rewards a small one.

This is the naive gradient, and it has a fatal flaw. It needs $R(a)$ for _every_ action, including the ones we never tried, and the sum is huge or infinite for large or continuous action sets. **We cannot evaluate it. We need to turn it into something we can sample.**

### 2.3 The score-function trick (the one clean derivation)

**Why reach for the log at all?** Go back to what broke in Section 2.2. The naive gradient is a sum over every action, and we cannot evaluate it. The one thing we _can_ do in practice is sample actions from the policy and average what we see. A sum that we can estimate by sampling has a special name: an **expectation**. And an expectation under the policy always has the same shape, $\sum_a \pi_\theta(a) \cdot [\dots]$, with each term weighted by the probability of that action. That weight is what lets us replace "sum over all actions" with "average over the actions we actually drew."

Now look at our sum, $\sum_a R(a) \cdot \nabla_\theta \pi_\theta(a)$. It has no $\pi_\theta(a)$ out front, so it is not yet an expectation we can sample. The whole job is to manufacture that missing $\pi_\theta(a)$ factor. The trick is to multiply and divide each term by $\pi_\theta(a)$, and the quantity $\nabla_\theta \pi_\theta(a) / \pi_\theta(a)$ is exactly the derivative of $\log \pi_\theta(a)$. That is the only reason the log appears: it is the identity that pulls a $\pi_\theta(a)$ to the front and leaves behind a sum we can sample.

Here is that identity, from the chain rule applied to $\log$:

$$\nabla_\theta \log \pi_\theta(a) = \frac{\nabla_\theta \pi_\theta(a)}{\pi_\theta(a)}$$

Rearrange:

$$\nabla_\theta \pi_\theta(a) = \pi_\theta(a) \cdot \nabla_\theta \log \pi_\theta(a)$$

Now substitute into the intractable sum:

$$\nabla_\theta J = \sum_a R(a) \cdot \nabla_\theta \pi_\theta(a) = \sum_a \pi_\theta(a) \cdot R(a) \cdot \nabla_\theta \log \pi_\theta(a)$$

The leading $\pi_\theta(a)$ is exactly the probability of sampling $a$, so the sum becomes an expectation:

$$\boxed{\nabla_\theta J = \mathbb{E}_{a \sim \pi_\theta}\big[R(a) \cdot \nabla_\theta \log \pi_\theta(a)\big]}$$

<details>
<summary><strong>Check:</strong> The naive gradient needed the reward of every possible action. After the score-function trick, we only need the reward of the one action we sampled. Where did the other actions go?</summary>

**Answer.** They were absorbed into the sampling distribution. The trick rewrote the sum over all actions as an expectation under the policy. An expectation is estimated by sampling from the policy itself, so the actions we do not take are accounted for by how often we would sample them, not by an explicit sum. One sampled action gives one unbiased gradient estimate.

</details>

<details>
<summary><strong>Check:</strong> Why can't we actually compute the "naive" gradient sum_a R(a) * grad pi(a)?</summary>

**Answer.** It needs the reward R(a) of every action, including all the ones we never tried, and that sum is huge (or infinite) for large or continuous action sets. There is simply nothing to evaluate it from.

</details>

<details>
<summary><strong>Check:</strong> The score-function trick rewrites that sum as an expectation. Why is an expectation something we can handle when the sum was not?</summary>

**Answer.** An expectation is estimated by sampling from the policy. So instead of summing over all actions, we just take the one action we actually sampled and use its reward, which is exactly the data the agent already collects.

</details>

<details>
<summary><strong>Check:</strong> Why does "log pi" turn up in every policy-gradient method you will ever see?</summary>

**Answer.** Because of the single identity $\nabla \pi = \pi \cdot \nabla \log \pi$. Multiplying by $\pi$ is what converts the sum-over-actions into an expectation-under-$\pi$, and that step leaves a $\nabla \log \pi$ behind. Every method inherits it from this one move.

</details>

<details>
<summary><strong>Check:</strong> Write the log-derivative identity and prove it in one line. Why is it exactly the bridge from a sum to an expectation?</summary>

**Answer.** The identity is $\nabla_\theta \pi_\theta(a) = \pi_\theta(a) \cdot \nabla_\theta \log \pi_\theta(a)$. Proof in one line: $\nabla \log \pi = \frac{1}{\pi} \nabla \pi$ by the chain rule; multiply both sides by $\pi$. It is the bridge because:

$$\sum_a R(a)\,\nabla \pi(a) = \sum_a \pi(a)\,R(a)\,\nabla \log \pi(a) = \mathbb{E}_{a \sim \pi}\!\big[R(a)\,\nabla \log \pi(a)\big]$$

The leading $\pi(a)$ turns the sum over actions into an expectation under the policy, which we can estimate by sampling.

</details>

### 2.4 REINFORCE: sample, observe, push

The estimator from Section 2.3 is REINFORCE. Each episode: sample an action, see the reward, compute one gradient estimate, take a step.

$$\hat{g} = R(a) \cdot \nabla_\theta \log \pi_\theta(a), \quad a \sim \pi_\theta$$

Read it aloud, symbol by symbol: "the gradient estimate $\hat{g}$ equals the reward $R(a)$ times the gradient with respect to $\theta$ of $\log \pi_\theta(a)$, for an action $a$ sampled from the policy $\pi_\theta$ ($a \sim \pi_\theta$)."

$\nabla_\theta \log \pi_\theta(a)$ is a **vector**: the direction in parameter space that makes action $a$ more likely (the "nudge"). $R(a)$ is a single **number** sitting in front of it, so multiplying just stretches or shrinks that direction without rotating it. A big reward makes the nudge long (push hard toward $a$); a small reward makes it short; a negative reward flips the sign (push away from $a$). So in $\hat{g} = R(a) \cdot \nabla_\theta \log \pi_\theta(a)$, the "$\cdot$" _is_ the scaling and $R(a)$ is the scale factor. One sample gives one noisy estimate $\hat{g}$ of the true gradient.

In the bandit there is no future, so the weight on the chosen angle is simply its reward $r$. No returns, no credit assignment. One shot, one reward, one push.

**The Archer bandit, concretely.** Before the code, here is the exact problem the environment encodes.

- **State space.** There is only one state, so there is nothing to observe. We still have to feed the network a fixed-size input, so the observation is a constant dummy: the length-1 vector $[0.]$ (declared as a `Box` of one float in $[0, 1]$). It never changes and carries no information.
- **Action space.** Nine discrete release angles $a_1 \dots a_9$ (a `Discrete(9)`), indexed $0 \dots 8$ in code. Each episode the agent picks exactly one.
- **Reward.** How close the arrow lands to the bullseye, shaped as a Gaussian bell over the angle:

$$R(a) = \exp\!\left(-\frac{(a - a_\text{target})^2}{2\sigma^2}\right), \quad a_\text{target} = 4,\ \sigma = 1.5$$

Read it aloud: the reward of angle $a$ is $e$ raised to minus the squared distance from the target angle $a_\text{target}$, divided by $2\sigma^2$. In plain English: shoot exactly at the bullseye and the exponent is $0$, so $R = e^0 = 1$ (a perfect score). Miss by a little and the reward dips a little; miss by a lot and it falls toward $0$. With $a_\text{target} = 4$ (angle $a_5$) and $\sigma = 1.5$, angle $a_5$ scores $1.00$, its neighbors $a_4$ and $a_6$ score about $0.80$, and the far edges $a_1$/$a_9$ score about $0.03$. A uniformly random angle averages about $0.42$, which is the baseline the policy has to beat.

The episode ends after that single shot, so there is no next state and no future reward. That is exactly what the code below sets up: a `Box` observation, a `Discrete` action space, and this reward in `step`.

**One training knob: the entropy bonus.** The policy is a probability distribution over the 9 angles, and its _entropy_ $H(\pi) = -\sum_a \pi(a) \log \pi(a)$ measures how spread out it is: high when the nine probabilities are even (the policy is still exploring), near zero when one angle dominates (the policy has committed). Each call to `act` returns this entropy next to the log-probability, straight from the `Categorical` distribution. We add a small $+\,\texttt{ent\_coef} \cdot H$ term to the objective so the policy pays a penalty for collapsing onto one angle too soon, which keeps it trying all nine until it has actually found the bullseye. It appears in the loss as the `- ent_coef * ent` piece, and Section 2.12 ablates it.

Here is the full bandit training loop. The policy network maps the constant state to 9 logits, `Categorical` turns them into probabilities, and we sample one angle. Because the bandit always terminates after one shot, **each episode is a single step**, so there is no inner while-loop here (the MDP in Section 2.9 adds one):

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np, torch, torch.nn as nn

class ArcherBandit(gym.Env):
    """One-state bandit: 9 angles, reward = Gaussian centered on bullseye."""
    # 9 discrete angles (0-8), bullseye at index 4, spread 1.5
    N_ANGLES, TARGET, SIGMA = 9, 4, 1.5

    def __init__(self) -> None:
        super().__init__()
        # observation_space declares what a state looks like. spaces.Box is the
        # space for continuous values: Box(low, high, shape, dtype), so this is a
        # length-1 float32 array bounded to [0, 1]. The bandit has no real state,
        # so the observation is always the dummy [0.]
        self.observation_space = spaces.Box(0., 1., (1,), np.float32)
        # spaces.Discrete(N) is the action space: one integer choice in 0..N-1
        # (here the 9 release angles a1..a9)
        self.action_space = spaces.Discrete(self.N_ANGLES)

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # constant dummy state
        return np.array([0.], np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # reward = Gaussian bell curve exp(-(a - target)^2 / (2σ^2))
        # it peaks at 1.0 when a == TARGET (dead center) and falls
        # smoothly toward 0 as a moves away from the bullseye
        r = float(np.exp(-((action - self.TARGET)**2) / (2*self.SIGMA**2)))
        # terminated=True: bandit episodes are always one step
        return np.array([0.], np.float32), r, True, False, {}

# ── Policy network ──────────────────────────────────────────────
class Policy(nn.Module):
    def __init__(self, obs_dim: int, n_act: int, h: int = 64) -> None:
        super().__init__()
        # two-layer MLP: obs -> hidden (tanh) -> one logit per action
        self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, n_act))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # raw logits; softmax is applied by Categorical
        return self.net(x)

# ── Action sampling ─────────────────────────────────────────────
# state_np is the observation straight from the env: a NumPy array (here [0.]).
# the _np suffix is a reminder it is NumPy and must be cast to a tensor first
def act(policy: nn.Module, state_np: np.ndarray
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    st = torch.tensor(state_np, dtype=torch.float32)
    # Categorical takes raw logits and applies softmax internally
    dist = torch.distributions.Categorical(logits=policy(st))
    # stochastic draw: exploration is built into the policy
    action = dist.sample()
    # log_prob(a) = log π(a) feeds the REINFORCE gradient;
    # entropy H(π) measures how spread out the policy is
    return action, dist.log_prob(action), dist.entropy()

# ── Training loop ───────────────────────────────────────────────
def train_bandit(episodes: int = 1500, lr: float = 0.01, ent_coef: float = 0.1
                 ) -> tuple[Policy, list[float]]:
    env = ArcherBandit()
    pol = Policy(1, env.N_ANGLES)
    opt = torch.optim.Adam(pol.parameters(), lr)
    hist = []
    for ep in range(episodes):
        # one episode = one shot: reset, sample an angle, step once, then update
        s, _ = env.reset()
        a, logp, ent = act(pol, s)
        _, r, *_ = env.step(int(a))
        # REINFORCE loss -r·log π(a), negated because optimizers minimize
        # but we want to maximize expected reward; the entropy bonus
        # (+ent_coef·H) delays collapse to one angle before we explore
        loss = -(r * logp) - ent_coef * ent
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(r)
    return pol, hist

# seed everything so the run below is reproducible
torch.manual_seed(2)
pol, hist = train_bandit()
probs = torch.softmax(pol(torch.zeros(1)), -1).detach().numpy()
print(f"mean reward (last 200): {np.mean(hist[-200:]):.3f}  (uniform ≈ 0.42)")
print(f"final π: {[f'{p:.3f}' for p in probs]}")
print(f"argmax = a{np.argmax(probs)+1}  (bullseye = a{ArcherBandit.TARGET+1})")
```

```text title="Output"
mean reward (last 200): 1.000  (uniform ≈ 0.42)
final π: ['0.000', '0.000', '0.000', '0.000', '1.000', '0.000', '0.000', '0.000', '0.000']
argmax = a5  (bullseye = a5)
```

The fan sharpens onto the bullseye: all probability concentrates on angle 5 (the target, since TARGET=4 is zero-indexed). The loss line `-(r * logp) - ent_coef * ent` is the entire algorithm: weight the log-probability of the taken action by its reward (the REINFORCE estimator) and add an entropy bonus to keep exploring.

<details>
<summary><strong>Check:</strong> DQN happily trained on stale transitions from a replay buffer. Why can't REINFORCE do the same?</summary>

**Answer.** Because the gradient is an expectation under the _current_ policy. The samples must come from today's policy. The moment you update theta, the old shots were drawn from the wrong distribution, so reusing them would bias the estimate. That is what "on-policy" means, and it is why REINFORCE is sample-hungry: each batch of shots is used once and thrown away.

</details>

<details>
<summary><strong>Check:</strong> In the bandit, the gradient weight is just the reward r, not a return or a discount. What feature of the bandit makes that okay?</summary>

**Answer.** The bandit has no next state, so your action has no future to influence. There are no later rewards to add up. The "return" is just the single reward r. Returns and discounting only appear once actions change what happens next (Section 2.7).

</details>

<details>
<summary><strong>Check:</strong> One shot gives one gradient that's "correct on average." Why is a single shot still a poor guide?</summary>

**Answer.** It is one draw of a very noisy quantity; its direction can land almost anywhere. Only the average over many shots converges to the true gradient. That noise is exactly what Section 2.6 fixes.

</details>

<details>
<summary><strong>Check:</strong> REINFORCE is on-policy and DQN is off-policy; both are "model-free." In one sentence each, say what "on/off-policy" and "model-free" actually mean.</summary>

**Answer.** On/off-policy: whether you learn about the same policy that generated the data (on) or a different one (off). Model-free: you never learn or use the transition model P(s'|s,a); you learn purely from sampled experience. They are independent axes: REINFORCE is on-policy and model-free; DQN is off-policy and model-free.

</details>

### 2.5 The backward pass: all nine logits move

This section answers the question everyone asks: the policy gave nine probabilities, but we sampled only one angle. Whose gradient do we actually compute?

The loss uses only the sampled action $a_6$:

$$L = -r \cdot \log \pi(a_6)$$

A natural guess: "we only have a gradient for $a_6$." That guess is **wrong**. The softmax denominator ties all nine logits together:

$$\log \pi(a_6) = z_6 - \log \sum_k e^{z_k}$$

That sum $\sum_k$ contains _all nine_ logits. So the loss depends on every $z_k$, and the gradient touches all nine:

$$\frac{\partial L}{\partial z_k} = -r \cdot \big(\mathbb{1}[k = a_6] - \pi(a_k)\big)$$

For the **taken** angle $a_6$: gradient $\propto +(1 - \pi_6)$, push **up**.
For each **other** angle: gradient $\propto -\pi_k$, push **down**.
All nine pushes **sum to zero**: probability is conserved, only redistributed.

![The nine per-logit gradient pushes from one REINFORCE update, showing one positive bar for the taken action and eight negative bars that sum to zero](./images/fig-logit-gradients.svg)

The figure makes the formula concrete: one tall bar for the taken angle $a_6$ pushing it up, and eight short bars for the others pushing them down. The ups and downs are sized so they cancel exactly, which is the "sum to zero" we just derived. One sample, nine moves.

Here is a tiny snippet that reproduces the nine logit gradients with real numbers:

```python
import numpy as np

# hypothetical logits (raw scores) the policy network outputs for 9 angles;
# these are NOT probabilities yet, softmax converts them next
logits = np.array([0.2, 0.6, 1.0, 1.6, 2.0, 1.6, 1.0, 0.6, 0.2])

# softmax π(aₖ) = exp(zₖ) / Σ exp(zⱼ) turns logits into a valid
# probability distribution that sums to 1
probs = np.exp(logits) / np.exp(logits).sum()

# a6 (0-indexed): the action we actually sampled
taken = 5
# reward received for that shot
r = 0.9

# per-logit gradient of the REINFORCE loss L = -r · log π(a_taken):
#   ∂L/∂zₖ = -r · (𝟙[k = taken] - π(aₖ))
#   for the taken action: gradient = -r·(1 - π(a₆)), negative so the logit rises
#   for every other action: gradient = +r·π(aₖ), positive so the logit falls
# np.eye(9)[taken] is a one-hot vector: 1 at position 'taken', 0 elsewhere
grads = -r * (np.eye(9)[taken] - probs)

print("Per-logit gradients (r=0.9, taken=a6):")
for i, g in enumerate(grads):
    tag = " ← taken" if i == taken else ""
    print(f"  a{i+1}: {g:+.3f}{tag}")
# all nine gradients sum to exactly zero: probability is conserved,
# only redistributed among angles, never created or destroyed
print(f"  sum: {grads.sum():.6f}")
```

```text title="Output"
Per-logit gradients (r=0.9, taken=a6):
  a1: +0.038
  a2: +0.057
  a3: +0.085
  a4: +0.155
  a5: +0.231
  a6: -0.745 ← taken
  a7: +0.085
  a8: +0.057
  a9: +0.038
  sum: 0.000000
```

(Note: the signs are the gradient of the _loss_ $L = -r \log\pi(a_6)$; the optimizer subtracts this from the logits, so a6's logit rises and the others fall, exactly as expected.)

The nine logit-gradients form a single vector $\partial L / \partial \mathbf{z}$. We backprop it **once** (not nine times). Each weight's gradient is the sum of all nine logits' contributions via the chain rule:

$$\frac{\partial L}{\partial \theta_j} = \sum_{k=1}^{9} \frac{\partial L}{\partial z_k} \cdot \frac{\partial z_k}{\partial \theta_j}$$

Every logit $z_k$ is built from the same shared weights $\theta$, so the chain rule adds up all nine influences. The result is one gradient per weight and one Adam step, not nine.

**One update, by hand.** Take the same fan the snippet used, $\pi = [0.042, 0.063, 0.094, 0.172, 0.256, 0.172, 0.094, 0.063, 0.042]$ (peaked at $a_5$), and suppose we sample $a_6$ and score $r = 0.9$. The per-logit gradients are exactly the ones printed above: $-0.745$ on the taken angle $a_6$, small positives on the other eight, summing to zero. Now take one optimizer step at learning rate $\alpha = 0.1$. Each logit moves by $-\alpha \cdot \partial L / \partial z_k$, so $a_6$ rises and every other angle drops a little. The fan tilts toward the shot that scored well.

![Policy fan before and after one REINFORCE update, showing the shift toward the taken action](./images/fig-policy-fan.svg)

The figure shows the nine-angle fan before and after that single step: the bar on $a_6$ grows while the other eight shrink, and the total stays at 1. **A single sample moves all nine logits, not just the one you tried. Probability is redistributed, never created.**

<details>
<summary><strong>Check:</strong> The policy gave nine probabilities but we sampled only a6. Why does the update change all nine angles, not just a6?</summary>

**Answer.** Because the softmax ties them together: pi(a6) is e^{z6} divided by the sum of all nine exponentials. Touching any logit changes that shared denominator, so the gradient is nonzero for every angle. The taken angle is pushed up and the other eight are nudged down, all through the normalizer.

</details>

<details>
<summary><strong>Check:</strong> Those nine pushes added up to zero. Why must they always sum to zero?</summary>

**Answer.** Because the nine probabilities must always add to 1. You cannot create probability, only move it. So whatever you add to one angle has to come off the others. The update redistributes the fan; it does not grow it.

</details>

<details>
<summary><strong>Check:</strong> In the example the reward was 0.9 (a big push on a6). Redo it for a near-miss that scored only 0.1: which way does a6 move, and by how much?</summary>

**Answer.** The weight is just the reward, 0.1 > 0, so a6 still moves UP, only about a ninth as far ((1-.172)\*0.1 ≈ +0.083 on its logit, versus +0.745 before), with the other eight nudged down proportionally. The catch: with the reward alone, _every_ shot pushes the taken angle up. Only the size changes. That one-sidedness is wasteful, and it is exactly what the baseline in Section 2.6 fixes.

</details>

<details>
<summary><strong>Check:</strong> We visualize the gradient on the logits, but the actual parameters are the network weights theta. What single tool bridges the two, and why don't we update the logits directly?</summary>

**Answer.** Backpropagation (the chain rule) bridges them: the logit-gradient is the entry point, and backprop turns it into a gradient for every weight. We cannot update logits directly because they are not parameters; they are recomputed from theta on every forward pass. Only theta persists.

</details>

### 2.6 The variance problem and the baseline

Without a baseline, every return is positive (in the Archer, reward is always > 0), so every sampled action is pushed up. Good and mediocre alike. The signal is a small difference riding on a big positive offset, and sample noise swamps it.

**The fix.** Subtract a baseline $b$ that does not depend on the action:

$$\hat{g} = (R - b) \cdot \nabla_\theta \log \pi_\theta(a)$$

We call the centered weight the **advantage** $A = R - b$. Better than baseline pushes up ($A > 0$); worse pushes down ($A < 0$). The estimator now takes both signs and mostly cancels, dramatically cutting variance.

**Zero-bias proof.** We subtracted $b$, so we added a term $-b \cdot \nabla \log \pi(a)$. Does that bias the gradient? Take its expectation step by step:

$$\mathbb{E}_{a \sim \pi}[b \cdot \nabla \log \pi(a)] = b \cdot \sum_a \pi(a) \nabla \log \pi(a) = b \cdot \sum_a \nabla \pi(a) = b \cdot \nabla \underbrace{\sum_a \pi(a)}_{=1} = b \cdot \nabla(1) = 0$$

Walk through each `=` sign:

1. **Expectation becomes a weighted sum.** $\mathbb{E}_{a \sim \pi}[\cdot]$ means "sum over all actions, each weighted by its probability $\pi(a)$." The constant $b$ pulls out of the sum.
2. **Undo the log-derivative trick.** Recall $\pi(a) \cdot \nabla \log \pi(a) = \nabla \pi(a)$ (the identity from Section 2.3, just run in reverse). So $\pi(a) \nabla \log \pi(a)$ collapses back to $\nabla \pi(a)$.
3. **Swap sum and gradient.** $\sum_a \nabla \pi(a) = \nabla \sum_a \pi(a)$. The sum of all action probabilities is 1 by definition (it is a probability distribution).
4. **Gradient of a constant is zero.** $\nabla(1) = 0$. No matter how you change $\theta$, probabilities still sum to 1, so the derivative of that sum is always zero.

The punchline: the baseline term vanishes _in expectation_. Subtracting $b$ changes the _variance_ of the estimator but not its _mean_. We still climb the same $J$, just with far less noise.

**The push magnitude.** We know REINFORCE pushes good actions up and bad actions down. But _how hard_ does it push? That depends on how surprised the policy is by its own choice.

For a softmax policy, the gradient of $\log \pi(a)$ with respect to the logit $z_a$ of the chosen action is:

$$\frac{\partial \log \pi(a)}{\partial z_a} = 1 - \pi(a)$$

Read it as: "one minus the probability the policy already assigned to that action." This is the **score function** for the taken action, and it controls how big the parameter update is. Two concrete cases make the intuition click:

- **Surprising action pays off.** The archer tries angle 7, which the policy thought was unlikely ($\pi(a_7) = 0.05$). It scores well. The push magnitude is $1 - 0.05 = 0.95$: a large update. The policy had a lot to learn from this surprise.
- **Confident action pays off.** The archer tries angle 5, which the policy already favored ($\pi(a_5) = 0.80$). It scores well. The push magnitude is $1 - 0.80 = 0.20$: a small update. The policy already knew this was good; confirming it again does not warrant a big change.

The policy automatically spends its learning budget where it matters most: on actions it was _wrong_ about, not on ones it already had right. This is built into the math of $\nabla \log \pi$; you get it for free.

![The push magnitude 1 minus pi(a) plotted against current probability, showing that surprising actions get bigger updates](./images/fig-score-push.svg)

The curve is just $1 - \pi(a)$. On the left, an action the policy thought unlikely (small $\pi$) gets a near-1 push when it pays off. On the right, an action the policy was already sure of (large $\pi$) barely moves. The downward slope is the policy spending its learning budget on surprises, exactly the two cases above.

Back to the baseline itself: the near-optimal constant baseline is approximately the average return, $b^* \approx \mathbb{E}[G_t]$. An even better baseline is a _learned, per-state_ value $V(s)$, which we will see in Section 2.8.

<details>
<summary><strong>Check:</strong> We subtract a baseline b to cut variance. A skeptic worries: "you changed the gradient, now you're optimizing the wrong thing." Why is the skeptic wrong?</summary>

**Answer.** The baseline term has expectation exactly zero: $\mathbb{E}[b \cdot \nabla \log \pi(a)] = b \sum_a \pi(a) \nabla \log \pi(a) = b \cdot \nabla(1) = 0$. Subtracting $b$ changes the variance of the estimator but not its mean, so we still optimize $J$, just with less noise.

</details>

<details>
<summary><strong>Check:</strong> Without a baseline, for an all-positive reward every shot pushes its action's probability UP. Why does that one-sidedness make the noise so destructive?</summary>

**Answer.** Because every sample agrees in direction, they never cancel. You get one big, one-sided number whose swings drown out the small real differences between actions. Subtracting a baseline lets the weight take both signs, so the noise cancels.

</details>

<details>
<summary><strong>Check:</strong> After the baseline, A = R − b can be negative. What does a negative advantage do to the angle you took, and when does that happen?</summary>

**Answer.** A negative advantage ($A < 0$) pushes that angle's probability **down**. It happens whenever the shot scored below the baseline ($R < b$, worse than average), so a disappointing outcome makes the action you tried less likely. Better-than-average ($A > 0$) pushes up; worse-than-average ($A < 0$) pushes down. This two-sided signal is exactly why the baseline cuts variance: without it, every weight was positive and all actions got pushed up indiscriminately.

</details>

<details>
<summary><strong>Check:</strong> A student proposes a "baseline" equal to the return of the same trajectory, b = G_t, so the advantage is always 0. What goes wrong, and which rule does it violate?</summary>

**Answer.** If $b = G_t$, then $A = G_t - G_t = 0$ for every sample, so the gradient is always zero and the policy never updates. There is no learning signal at all. More fundamentally, $G_t$ depends on the action taken (different actions lead to different trajectories and different returns), so it violates the requirement that the baseline be **action-independent**. The zero-bias proof (Section 2.6) only works when $b$ can be pulled out of the expectation over actions, which requires $b$ not to depend on $a$. A legal baseline may depend on the state $s$ (that is $V(s)$), never on the action.

</details>

<details>
<summary><strong>Check:</strong> The estimator R(a) · ∇ log π(a) is unbiased. "Unbiased" means "correct on average." Why is that a weaker promise than "useful on any single sample"?</summary>

**Answer.** "Unbiased" means $\mathbb{E}[\hat{g}] = \nabla J$: if you averaged infinitely many one-sample estimates, you would get the true gradient. But on any _single_ episode the estimate $\hat{g} = R(a) \cdot \nabla \log \pi(a)$ can point almost anywhere, because the variance is huge (one lucky or unlucky reward swings the whole vector). So "correct on average" is a much weaker guarantee than "close to the truth on every draw." That gap is precisely why REINFORCE is so slow and why we need variance-reduction tricks (baselines, critics) to make each individual sample more informative.

</details>

<details>
<summary><strong>Check:</strong> grad log pi is called the "score function." For a softmax policy, show that the score for the chosen action is (1 - pi(a)) along its logit. Why does an already-confident action barely move?</summary>

**Answer.** For a softmax, $\log \pi(a) = z_a - \log \sum_k e^{z_k}$, so $\partial(\log \pi(a))/\partial z_a = 1 - \pi(a)$. If the action is already confident ($\pi(a)$ near 1), then $1 - \pi(a) \approx 0$ and the gradient barely moves it. You do not waste updates re-confirming what the policy already does.

</details>

<details>
<summary><strong>Check:</strong> The trick needs pi(a) > 0 for every action we might sample (we divide by it). What does that say about why a policy should never assign exactly zero probability during training?</summary>

**Answer.** Because the score function divides by $\pi(a)$: an action with $\pi(a) = 0$ makes $\nabla \log \pi$ blow up and can never be sampled or learned about. Policies should keep every probability strictly positive during training (e.g. via entropy regularization, or a softmax that never fully saturates) so every action stays explorable and the gradient stays well-defined.

</details>

<details>
<summary><strong>Check:</strong> An LLM's final layer is a softmax over 50,000 token logits, exactly like our Archer's 9-angle softmax. If a generated answer earns a positive reward, what does REINFORCE do to the token probabilities, and how is it the same push/pull we just derived?</summary>

**Answer.** Exactly the same mechanics: for each token the model produced, a positive advantage pushes that token's logit up ($\partial \log\pi / \partial z_a = 1 - \pi(a) > 0$) and pushes every other token's logit down ($-\pi(a_k)$). The only differences are scale (50K actions instead of 9) and the fact that the reward arrives at the end of a whole sequence, not one shot. This is the core of RLHF and PPO for LLMs, which the [TRPO & PPO](../06-trpo-ppo/README.md) post covers in full.

</details>

### 2.7 Adding states: from bandit to MDP

So far the Archer stood at one fixed spot and took one shot. Now we let it walk. The instant an action changes what happens next, the bandit becomes a full sequential problem.

Two things change in the mechanics:

**1. The policy reads the state.** $\pi_\theta(a) \to \pi_\theta(a \mid s)$. The Archer decides differently at 40 m than at 10 m. Same weights $\theta$, one distribution per distance.

**2. The weight becomes the return.** The single reward $r$ becomes $G_t$, the discounted sum of rewards that follows a step. A quiet "step closer" that costs $-0.2$ can now be credited for the +10 shot it set up. This is **credit assignment**.

The return $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$ was defined in the [MDPs & Bellman](../02-mdps-and-bellman/README.md) post. Discounting and returns were explored in the [DP, MC & TD](../03-dp-mc-td/README.md) post. REINFORCE waits for the full episode to compute $G_t$, exactly like Monte Carlo prediction from that same post.

The complete REINFORCE gradient with states:

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \Big[\sum_t G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Big]$$

Read it against the bandit: the bandit was the one-step case, a single term with $G = r$. Adding states just sums the same per-action push over the whole trajectory, each weighted by its return.

![Credit assignment across the Archer MDP trajectory, showing the return at each distance building up from the final reward](./images/fig-credit-assignment.svg)

The figure traces one trajectory from far (40 m) to the final shot. Each bar is that step's return $G_t$, and you can see the big +10 at the shot flowing backwards: the discount shrinks it a little at every earlier step, but even the first quiet move inherits most of it. That backward flow is credit assignment, and the snippet below computes those exact bars.

```python
import numpy as np

# a 5-step Archer MDP trajectory: 4 "step closer" moves (cost -0.2 each)
# followed by a final shot that earns +10
rewards = [-0.2, -0.2, -0.2, -0.2, 10.0]
states  = ["40m", "30m", "20m", "10m", "shoot"]
# discount factor: future rewards are worth 90% per step
gamma   = 0.9

# compute the discounted return G_t at every step, working BACKWARDS;
# G_t = r_t + γ·G_{t+1} is the recursive definition of the return, so
# iterating in reverse lets each step reuse the already-computed future
G, returns = 0.0, []
for r in reversed(rewards):
    # accumulate: this reward + discounted future
    G = r + gamma * G
    returns.append(G)
# flip back to chronological order
returns.reverse()

# even the earliest "step closer" at 40 m gets a large positive return
# because the discounted +10 final shot propagates back to it: this is
# CREDIT ASSIGNMENT, quiet moves credited for the future they set up,
# not just their own immediate -0.2
for s, Gt in zip(states, returns):
    print(f"  {s:>5s}: G = {Gt:+.3f}")
```

```text title="Output"
    40m: G = +5.873
    30m: G = +6.748
    20m: G = +7.720
    10m: G = +8.800
  shoot: G = +10.000
```

"Step closer at 40 m" has a return of +5.87, not -0.2, because the discounted future includes the +10 shot it set up. That is how an end-of-episode reward teaches the quiet moves that set it up.

<details>
<summary><strong>Check:</strong> REINFORCE waits for a full episode before updating (it needs G_t). Which earlier method does that remind you of, and what is the cost of waiting?</summary>

**Answer.** It is like Monte Carlo prediction from the [DP, MC & TD](../03-dp-mc-td/README.md) post: both wait for the full episode return before updating. The cost of waiting is high variance (the whole-episode return is noisy) and no online learning: you cannot update mid-episode.

</details>

<details>
<summary><strong>Check:</strong> An action at time t is weighted by G_t, the return that came after it. Why do we use the reward-to-go G_t rather than the whole-episode return for every step?</summary>

**Answer.** An action at time t cannot affect rewards earned before t, so including those earlier rewards only adds zero-mean noise. Reward-to-go drops them: same expectation (still unbiased), smaller variance.

</details>

### 2.8 The advantage: "better than typical _here_"

REINFORCE with states is painfully noisy because the return $G_t$ is a big, lurching number. The fix from Section 2.6 generalizes: subtract a _per-state_ baseline $V(s)$.

$$A_t = G_t - V(s_t)$$

The same return can mean very different things in different states. In the Archer MDP, $V(40\text{m}) \approx 6$ (far, hard) and $V(10\text{m}) \approx 9$ (close, easy). The same return $G = 7$ gives:

- at 40 m: $A = 7 - 6 = +1$ (above expectation, push up)
- at 10 m: $A = 7 - 9 = -2$ (below expectation, push down)

Same outcome, opposite lesson, because "good" is judged relative to what is typical at that distance. A single global baseline could never do this.

The advantage can also be written $A(s,a) = Q(s,a) - V(s)$: how much better is taking $a$ than acting typically from $s$? Its sign sets the direction; its magnitude sets how hard to push. That is all the policy needs.

The policy-gradient theorem becomes:

$$\nabla_\theta J(\theta) = \mathbb{E}\big[A(s,a) \cdot \nabla_\theta \log \pi_\theta(a \mid s)\big]$$

This is the form that every method in the rest of reinforcement learning shares. They differ only in how they estimate $A$.

<details>
<summary><strong>Check:</strong> A constant baseline subtracts the same number everywhere. Why is a state-dependent baseline V(s) strictly better?</summary>

**Answer.** Different states have very different typical returns. A constant can only be right "on average"; $V(s)$ subtracts the right amount per state, so $A = G - V(s)$ measures "better than expected in THIS state," removing more variance than any single constant could. That is the Actor-Critic idea.

</details>

<details>
<summary><strong>Check:</strong> At a state the critic predicts V(s) = 5. Two episodes pass through s: one returns G = 8, the other G = 3. Compute each advantage and say what happens to the probability of the action taken in each.</summary>

**Answer.** A = 8 - 5 = +3 (well above expectation, push that action's probability up, hard) and A = 3 - 5 = -2 (below expectation, push that action's probability down). Same state, same baseline: the advantage's sign and size do all the work.

</details>

### 2.9 Actor-Critic: two heads, one loop

Give the per-state baseline $V(s)$ its own network and let it learn alongside the policy. That single move is the foundation the rest of modern RL is built on.

**The Actor** is the policy $\pi_\theta(a \mid s)$. It acts: it chooses the move and takes the shot. **The Critic** is the value $V_w(s)$. It judges: it never acts. Having watched thousands of episodes, it knows how good each position is. Its verdict is the advantage, which is the only feedback the actor needs, and it is far steadier than a raw return.

The advantage is the TD error from the [DP, MC & TD](../03-dp-mc-td/README.md) post:

$$A = r + \gamma V_w(s') - V_w(s)$$

Read it as: "the reward I just got ($r$), plus what the critic thinks the next state is worth ($\gamma V_w(s')$), minus what the critic thought this state was worth ($V_w(s)$)." If $A > 0$, reality beat the critic's expectation, so push the action up; if $A < 0$, reality was worse, so push it down.

Two losses, one pass:

- **Critic:** regress $V(s)$ toward the TD target $r + \gamma V(s')$: minimize $(r + \gamma V(s') - V(s))^2$.
- **Actor:** push each action by its advantage: minimize $-(A \cdot \log \pi(a \mid s))$.

**Why `detach`?** The advantage is meant to be a fixed weight telling the actor how much to push. If the actor's gradient flowed into $V$, the actor could "cheat" by changing the critic to make its chosen actions look good (lowering $V(s)$) instead of improving the policy. `detach` keeps the two objectives clean.

Here is the complete, runnable Actor-Critic for the Archer MDP: the environment, both networks, the act helper, the training loop, and a greedy evaluation. It is the algorithm in full, so the rest of the post just analyzes what it does.

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np, torch, torch.nn as nn

# discount factor: how much we value future rewards vs. immediate
GAMMA = 0.99

# ── Environment: Archer MDP ─────────────────────────────────────
# unlike the bandit (one state, one shot), the MDP has a DISTANCE state;
# the archer can walk closer, step back, or shoot from the current distance
class ArcherMDP(gym.Env):
    """MDP: distance-to-target as state, 3 actions (closer/back/shoot)."""
    # three discrete actions
    CLOSER, BACK, SHOOT = 0, 1, 2
    # distance range in meters
    MIN_D, MAX_D = 10., 50.
    def __init__(self, max_steps=25):
        super().__init__()
        # observation = normalized distance d/50, so it lies in [0, 1]
        self.observation_space = spaces.Box(0., 1., (1,), np.float32)
        self.action_space = spaces.Discrete(3)
        self.max_steps = max_steps
    def _obs(self):
        return np.array([self.d / self.MAX_D], np.float32)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # start at a random distance from {10, 20, 30, 40, 50}
        self.d = float(self.np_random.choice([10.,20.,30.,40.,50.]))
        self.t = 0
        return self._obs(), {}
    def shoot_reward(self, d):
        # reward for shooting: 10 at 10 m (closest), drops linearly with distance;
        # optimal strategy is to walk to 10 m, then shoot for max reward
        return 10. - 0.28*(d - self.MIN_D)
    def step(self, a):
        self.t += 1
        if a == self.SHOOT:
            # shooting ends the episode (terminated=True) with a distance-based reward
            return self._obs(), float(self.shoot_reward(self.d)), True, False, {}
        # walking costs -0.2 per step (a small penalty to encourage efficiency)
        self.d = float(np.clip(
            self.d + (-10. if a == self.CLOSER else 10.), self.MIN_D, self.MAX_D))
        return self._obs(), -0.2, False, self.t >= self.max_steps, {}

# ── Actor (Policy network) ──────────────────────────────────────
# maps state -> one logit per action; softmax (inside Categorical) converts
# logits to π(a|s). This IS the policy we are optimizing
class Policy(nn.Module):
    def __init__(self, obs_dim, n_act, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, n_act))
    def forward(self, x): return self.net(x)

# ── Critic (Value network) ──────────────────────────────────────
# maps state -> single scalar V(s), the expected return from state s
# under the current policy; used as a baseline to compute the advantage
class Value(nn.Module):
    def __init__(self, obs_dim, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

# ── Action sampling ─────────────────────────────────────────────
def act(policy, s):
    st = torch.tensor(s, dtype=torch.float32)
    # Categorical applies softmax to logits -> π(a|s), then we sample
    dist = torch.distributions.Categorical(logits=policy(st))
    # stochastic action: exploration is built into the policy
    a = dist.sample()
    # log_prob = log π(a|s) is the "score" in the REINFORCE gradient;
    # entropy = H(π) measures exploration breadth
    return a, dist.log_prob(a), dist.entropy()

# ── Actor-Critic training loop ──────────────────────────────────
def actor_critic(env, episodes=2000, lr_a=0.004, lr_c=0.03, ent_coef=0.01, warmup=150):
    # actor: learns WHAT to do
    pol = Policy(1, env.action_space.n)
    # critic: learns HOW GOOD each state is
    val = Value(1)
    # actor optimizer (slower lr)
    popt = torch.optim.Adam(pol.parameters(), lr_a)
    # critic optimizer (faster lr)
    vopt = torch.optim.Adam(val.parameters(), lr_c)
    curve = []
    for ep in range(episodes):
        s, _ = env.reset()
        # collect one full episode: states, log-probs, entropies, rewards,
        # next-states, and done flags for every transition
        S, logps, ents, R, S2, Dn = [], [], [], [], [], []
        done = False
        while not done:
            a, logp, ent = act(pol, s)
            s2, r, term, trunc, _ = env.step(int(a))
            done = term or trunc
            S.append(s); logps.append(logp); ents.append(ent)
            R.append(r); S2.append(s2); Dn.append(float(term))
            s = s2

        # convert episode data to tensors for batch computation
        states = torch.tensor(np.array(S), dtype=torch.float32)
        snext  = torch.tensor(np.array(S2), dtype=torch.float32)
        rew    = torch.tensor(R, dtype=torch.float32)
        # 1.0 at the terminal step, 0.0 otherwise
        dn     = torch.tensor(Dn)

        # TD target r + γ·V(s')·(1 - done); the (1-dn) factor zeroes out the
        # bootstrap at terminal states, since there is no future after the
        # episode ends, so V(s') should not contribute there
        with torch.no_grad():
            target = rew + GAMMA * val(snext) * (1 - dn)

        v   = val(states)
        # advantage A = TD_target - V(s) = r + γV(s') - V(s):
        # positive means the action beat expectation, push probability UP;
        # negative means it was worse, push probability DOWN;
        # .detach() stops actor gradients from flowing into the critic
        adv = (target - v).detach()

        # ── Critic update: minimize (TD_target - V(s))² ────────────
        # trains V(s) to predict the discounted return from each state
        vopt.zero_grad()
        (target - v).pow(2).mean().backward()
        vopt.step()

        # ── Actor update (only after warmup) ────────────────────────
        # warmup lets the critic learn a reasonable V(s) before the actor
        # uses the advantage; early random advantages would just be noise
        if ep >= warmup:
            logp = torch.stack(logps)
            ent  = torch.stack(ents)
            popt.zero_grad()
            # actor loss = -𝔼[A · log π(a|s)] - ent_coef · H(π):
            # first term is REINFORCE weighted by advantage (negated for descent),
            # second term is the entropy bonus that keeps the policy exploring
            (-(adv * logp).mean() - ent_coef * ent.mean()).backward()
            popt.step()
        curve.append(sum(R))
    return pol, val, curve

# ── Greedy evaluation ───────────────────────────────────────────
# after training, test the policy deterministically (argmax, no sampling)
# to measure what it has actually learned, free from exploration noise
def greedy_eval(env, pol, n=20):
    total = 0.
    for _ in range(n):
        s, _ = env.reset(); done = False; ep_r = 0.
        while not done:
            with torch.no_grad():
                # greedy: pick the action with the highest logit (most confident)
                a = pol(torch.tensor(s, dtype=torch.float32)).argmax()
            s, r, term, trunc, _ = env.step(int(a))
            ep_r += r; done = term or trunc
        total += ep_r
    return total / n

# seed everything (torch init + sampling, and each env's RNG) so the
# greedy return below reproduces on a re-run
torch.manual_seed(0)
train_env = ArcherMDP(); train_env.reset(seed=0)
pol, val, curve = actor_critic(train_env)
eval_env = ArcherMDP(); eval_env.reset(seed=1)
print(f"Actor-Critic greedy return: {greedy_eval(eval_env, pol):.2f}")
```

```text title="Output"
Actor-Critic greedy return: 9.58
```

The Actor-Critic reaches a greedy return near the maximum possible score. The critic's warmup period lets $V$ settle before the actor starts moving; the detached advantage ensures the two heads stay clean.

<details>
<summary><strong>Check:</strong> The critic's job is to predict V(s). But the policy keeps changing, so the "correct" V(s) keeps moving. How can the critic ever learn a moving target, and why doesn't this stall the actor?</summary>

**Answer.** They co-adapt: the critic regresses toward the returns the current policy actually earns, so it tracks that policy's value as the policy drifts slowly (small learning rate). The actor does not need an absolute value, only the sign and rough size of the advantage (better or worse than this state's baseline). Even an imperfect critic gives a useful advantage, so both improve together. This co-adaptation is why a slow, stable step size matters, and why TRPO/PPO (next) are about controlling how far the actor moves per update.

</details>

<details>
<summary><strong>Check:</strong> Why is the policy called the "actor" and the value the "critic"? In the archer-and-coach picture, what does each do, and why is "judge but never act" actually useful rather than redundant?</summary>

**Answer.** The actor (policy pi) is the only part that acts: it takes the moves and shots. The critic (value V) never acts; it only judges how good a state is. The split is useful because the critic turns a noisy raw score into "better/worse than expected at THIS state" (a learned, per-state baseline), and because the two can specialize and improve together: the actor exploits the critic's sign, the critic sharpens its estimate.

</details>

<details>
<summary><strong>Check:</strong> "Step closer at 40 m" scored about 0 by itself, yet its probability goes up. Trace how a reward that only appears at the final shot reaches that early move.</summary>

**Answer.** Through the return $G_t$. The step's discounted future includes the +10 shot it set up, so its return is high, well above the critic's $V(40)$. The positive advantage $A = G_t - V(40)$ credits the quiet step for leading somewhere better, even though its own reward was just $-0.2$.

</details>

<details>
<summary><strong>Check:</strong> REINFORCE-with-a-baseline already subtracts a value to get A = G_t - V(s). What does calling that value a "critic" and giving it its own network actually add?</summary>

**Answer.** A learned, per-state network generalizes: it predicts the typical return for any state, including ones rarely visited, instead of a single global average. That sharper, state-specific baseline makes the advantage cleaner, and naming the two pieces "actor" and "critic" recognizes that the policy and its value can be separate networks that improve together.

</details>

<details>
<summary><strong>Check:</strong> Why do we detach the advantage when computing the actor's loss? What would break if the actor's gradient flowed into V?</summary>

**Answer.** The advantage is meant to be a fixed weight telling the actor how much to push. If the actor's gradient flowed into $V$, the actor could "cheat" by changing the critic to make its chosen actions look good (lowering $V(s)$) instead of improving the policy, corrupting both the baseline and the learning signal. `detach` keeps the two objectives clean.

</details>

<details>
<summary><strong>Check:</strong> Actor-Critic can update every step, but REINFORCE must wait for the episode to end. Which property of the advantage makes online updates possible?</summary>

**Answer.** Bootstrapping: the advantage uses $V(s')$, a one-step estimate of the future, instead of the full return $G_t$. You do not need the rest of the episode, just the next state's value, so you can update immediately after each transition.

</details>

<details>
<summary><strong>Check:</strong> Actor-Critic adds a critic network and reduces variance. Yet plain REINFORCE (no critic) is used in production for tasks like text-to-SQL. Why would the simpler algorithm ever be preferred?</summary>

**Answer.** When episodes are short, reward is sparse but clear (the SQL query either runs correctly or not), and the action space is small enough that variance is manageable, the extra complexity of training a critic is not worth the engineering cost. REINFORCE is dead simple to implement, and for tasks with a strong binary signal at episode end, its high variance is tolerable because the reward already tells you whether the output was right or wrong.

</details>

### 2.10 A note on continuous actions

Everything above used discrete actions (softmax over 9 angles). For continuous actions (robot torques, steering angles), the policy head outputs the parameters of a distribution:

```python
# for continuous actions (e.g. torque, steering angle), the policy outputs
# the PARAMETERS of a Gaussian distribution, not discrete logits;
# we sample a ~ N(μ(s), σ(s)) and log π(a|s) is the Gaussian log-density,
# so the same REINFORCE / Actor-Critic math still applies
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, h=64):
        super().__init__()
        # shared trunk: maps observation to hidden features
        self.trunk = nn.Sequential(nn.Linear(obs_dim, h), nn.Tanh())
        # mean head: one output per action dimension (state-dependent)
        self.mu_head  = nn.Linear(h, act_dim)
        # log standard deviation: a LEARNABLE parameter (not state-dependent),
        # stored as log(σ) so exp() guarantees σ > 0; starts at zeros, so
        # σ = 1.0 gives moderate initial exploration
        self.log_std  = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.trunk(x)
        # μ(s): center of the Gaussian
        mu = self.mu_head(h)
        # σ: spread (exploration width)
        std = self.log_std.exp()
        # returns a Normal distribution; call .sample() to get an action and
        # .log_prob(a) for the score ∇log π needed by REINFORCE
        return torch.distributions.Normal(mu, std)
```

The mean and standard deviation define a Gaussian; we sample the action from it. The same REINFORCE / Actor-Critic machinery applies: $\nabla \log \pi$ is just the log-density of the Gaussian, and the advantage still weights it. PPO (next lecture) uses this for MuJoCo continuous control.

<details>
<summary><strong>Check:</strong> For continuous actions we output a Gaussian's mean and standard deviation. Why is the standard deviation itself a learned parameter, and what would go wrong if we fixed it to a constant?</summary>

**Answer.** The right amount of exploration varies by state and over training, so a learned standard deviation lets the policy be uncertain where it should be and confident where it should not. Fix it too large and the policy never commits (it cannot exploit); too small and it stops exploring and gets stuck. A constant cannot adapt as learning progresses.

</details>

<details>
<summary><strong>Check:</strong> For each task, say whether you would reach for a value method or a policy method, and why: (a) a board game with 20 legal moves per turn; (b) steering angle and throttle for a self-driving car; (c) rock-paper-scissors against an adaptive opponent.</summary>

**Answer.** (a) Either works: discrete, modest action set, so a value argmax is cheap. (b) Policy: continuous controls; an argmax over a 2-D continuum every step is painful, a Gaussian policy is natural. (c) Policy: the optimal play is the stochastic uniform mix, which a deterministic argmax can never represent.

</details>

<details>
<summary><strong>Check:</strong> "Policy gradients optimize expected return directly; value methods optimize a Bellman error and hope." Name one situation where the value-method's indirection is actually an advantage.</summary>

**Answer.** When the policy is hard to represent but the value is not, e.g. large discrete action sets where the argmax over Q is cheap, or when off-policy data reuse matters: value methods can learn from a replay buffer and are far more sample-efficient, which an on-policy policy gradient cannot match.

</details>

<details>
<summary><strong>Check:</strong> A baseline reduces variance with zero added bias. Bootstrapping (using V(s') instead of the full return) reduces variance but ADDS bias. Why is one a free lunch and the other a trade?</summary>

**Answer.** A baseline sits inside a term that provably integrates to zero ($\mathbb{E}[b \cdot \nabla \log \pi] = 0$), so it cannot shift the mean: free variance reduction. Bootstrapping replaces the true return with an estimate $V(s')$ that is itself wrong, so it changes the target's mean: you trade some bias for the variance cut.

</details>

### 2.11 The variance ladder: REINFORCE vs baseline vs Actor-Critic

Every trick in this post was aimed at one enemy: the variance of the gradient estimate. It is worth seeing the payoff in numbers. We run 1500 episodes of the Archer MDP with three methods and measure the variance of the gradient estimator. (We use an `estimator_variance` helper, which records the per-episode gradient-estimator and returns `Var(estimator)`.)

Results (5-seed average):

| Method       | Gradient variance | Greedy return |
| ------------ | :---------------: | :-----------: |
| REINFORCE    |      0.0437       |     4.66      |
| + baseline   |      0.0006       |     9.60      |
| Actor-Critic |      0.0004       |     9.62      |

![Gradient variance on a log scale for the three methods, dropping by orders of magnitude with variance reduction](./images/fig-variance-ladder.svg)

The first figure plots the gradient variance on a log scale, so each step down the ladder is a full order of magnitude. Plain REINFORCE sits at the top; adding the baseline drops it sharply, and the Actor-Critic edges a little lower still. The bars are the table above, drawn to scale.

![Greedy returns for the three methods, showing only variance-reduced methods solve the task](./images/fig-greedy-returns.svg)

The second figure shows what that variance buys: the greedy return each method actually achieves. The two low-variance methods reach the near-optimal score while plain REINFORCE stalls well short. Lower variance is not a cosmetic win, it is the difference between learning the task and not.

**The baseline slashes variance by about 73x** (0.0437 / 0.0006). The Actor-Critic cuts it by another ~1.5x (0.0006 / 0.0004). But the performance gap is stark: plain REINFORCE barely learns, while both variance-reduced methods reach the near-optimal return of ~9.6.

Why does the baseline help so much? In the Archer MDP, rewards are mostly positive ($-0.2$ per step, $+10$ for shooting close). Without a baseline, the advantage $A = G_t$ is almost always positive, so every action gets pushed up. The signal is a tiny difference on top of a big positive offset, drowned in noise. The baseline centers it: good moves get pushed up ($A > 0$), bad moves pushed down ($A < 0$), so the gradient is sharper and more informative.

<details>
<summary><strong>Check:</strong> Actor-Critic uses a one-step TD error instead of the full return. That one-step estimate is biased (it uses an imperfect V(s')). How can a biased estimator produce better results than the unbiased REINFORCE?</summary>

**Answer.** Because bias is not the whole story: total error = bias^2 + variance. The one-step TD error has tiny bias (it is one sample of a recursion that converges) but massively lower variance than the full return. The net effect is a much better signal-to-noise ratio, so learning is faster and more stable despite the small bias.

</details>

<details>
<summary><strong>Check:</strong> In the Archer MDP there is always a walking cost of -0.2 per step. Without a baseline, the gradient estimator can never push an action DOWN (returns are positive). Why is that wasteful?</summary>

**Answer.** Because every action, good or bad, gets pushed UP by the positive return. The only difference is how much, and that difference is small compared to the offset. The policy drifts aimlessly uphill instead of quickly separating good from bad. With a baseline, worse-than-average actions get a negative advantage and are pushed down, giving a two-sided signal.

</details>

<details>
<summary><strong>Check:</strong> A student removes entropy regularization (ent_coef = 0) from Actor-Critic. What happens and why?</summary>

**Answer.** Without entropy regularization, the policy collapses early: one action dominates, exploration dies, and the agent gets stuck in a local optimum. The entropy bonus gives a "tax" on certainty, keeping probabilities spread so the agent tries all actions long enough to learn which is best before committing.

</details>

### 2.12 Entropy and the ablations

The entropy $H(\pi) = -\sum_a \pi(a) \log \pi(a)$ measures how spread out the policy is. Adding $+\texttt{ent\_coef} \cdot H$ to the objective discourages premature collapse: the policy pays a "tax" for becoming too certain too early.

Our ablation experiments:

**Ablation A (remove entropy, `ent_coef=0.0`):** Without entropy regularization, Actor-Critic's learning becomes much more unstable. Mean greedy return drops and the variance across seeds increases significantly. The policy commits to one action early and loses the ability to explore alternatives.

**Ablation B (remove critic warmup, `warmup=0`):** Starting actor updates from episode 0 degrades performance. Without warmup, the critic's early predictions are random, so the advantage signal is garbage, and the actor learns from noise. A short warmup lets the critic settle before the actor acts on its advice.

**Ablation C (large learning rate for bandit REINFORCE, `lr=0.2`):** A large learning rate makes the bandit's REINFORCE training faster (fewer episodes to converge) but noisier: the reward curve oscillates more, and unlucky seeds can converge to non-optimal angles. The right learning rate balances speed and stability.

<details>
<summary><strong>Check:</strong> Ablation B found that removing the critic warmup hurts performance. Why specifically does a "cold" critic hurt the ACTOR, not just the critic?</summary>

**Answer.** Because the actor's gradient uses the critic's advantage signal. A cold critic outputs near-random values, so the advantage $A = r + \gamma V(s') - V(s)$ is garbage: its sign and magnitude are noise. The actor follows this random signal and drifts aimlessly, potentially into a policy from which recovery is hard. The warmup gives the critic time to produce meaningful advantages before the actor starts trusting them.

</details>

---

## 3. Putting it all together

A quick recap of every concept in this post and how it maps to code:

| Concept                     | Math                                     | In code                                            |
| --------------------------- | ---------------------------------------- | -------------------------------------------------- |
| Objective                   | $J(\theta) = \mathbb{E}[R]$              | `loss = -(adv * logp).mean()` (negate for descent) |
| Score-function trick        | $\nabla J = \mathbb{E}[R \nabla\log\pi]$ | `logp = dist.log_prob(action)`                     |
| REINFORCE weight            | $G_t$ (return)                           | `G = r + gamma * G` (backward loop)                |
| Baseline / advantage        | $A_t = G_t - V(s_t)$                     | `adv = (target - v).detach()`                      |
| TD advantage (Actor-Critic) | $A = r + \gamma V(s') - V(s)$            | `target = rew + GAMMA * val(snext) * (1 - dn)`     |
| Entropy bonus               | $H(\pi) = -\sum \pi\log\pi$              | `- ent_coef * ent.mean()`                          |
| Critic loss                 | $(r + \gamma V(s') - V(s))^2$            | `(target - v).pow(2).mean()`                       |

Every concept above already appeared as a small snippet right next to its explanation, and §2.9 has the complete Actor-Critic loop that produced the 9.58 result. There is no separate capstone to paste here. The full end-to-end runnable program (the Archer bandit, the Archer MDP, and the whole climb from REINFORCE to Actor-Critic) lives in the assignment notebook:

> **[Assignment — Policy Gradients from Scratch (The Archer)](https://github.com/S1LV3RJ1NX/RL-in-Production-Bootcamp-Resources/blob/main/assignments/lecture3.ipynb)**

<details>
<summary><strong>Check:</strong> In the Actor-Critic loop, why do we use `(1 - dn)` in the TD target? What would go wrong without it?</summary>

**Answer.** The `dn` flag is 1 at terminal states. Without `(1 - dn)`, we would bootstrap through the terminal state, adding gamma \* V(s') after the episode is over. That injects a phantom future reward into the target, corrupting the critic's estimate. Setting the bootstrap to zero at terminal states keeps the value grounded in the actual final reward.

</details>

<details>
<summary><strong>Check:</strong> Why do we warm up the critic for 150 episodes before enabling the actor's updates?</summary>

**Answer.** A randomly initialized critic outputs random values, so the advantage is just noise. If the actor updates on that noise, it drifts into a bad policy that is hard to recover from. The warmup lets the critic settle into roughly correct $V(s)$ estimates first, so the advantage the actor sees is meaningful from the start.

</details>

---

## Practice: assignment

Build the Archer from scratch: a bandit, then an MDP agent, climbing the variance ladder from REINFORCE all the way to Actor-Critic.

> **[Assignment — Policy Gradients from Scratch (The Archer)](https://github.com/S1LV3RJ1NX/RL-in-Production-Bootcamp-Resources/blob/main/assignments/lecture3.ipynb)**

---

## Where this goes next

The Actor-Critic gives us a clean way to climb expected return, but each policy update can take a step of any size. If the step is too large, the policy overshoots and collapses. If it is too small, learning crawls. The next lecture adds **trust regions**: a hard constraint (TRPO) or a clipped surrogate (PPO) that keeps each update in a safe neighborhood of the old policy. The central equation becomes:

$$\max_\theta \; \mathbb{E}\left[\min\left(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_\text{old}}(a \mid s)} A, \; \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}}, 1-\epsilon, 1+\epsilon\right) A\right)\right]$$

Read it as: maximize the advantage-weighted probability ratio $\pi_\theta / \pi_{\theta_\text{old}}$ (how much more likely the new policy makes the action than the old one), but take the smaller of the raw ratio and a clipped version pinned to the band $[1-\epsilon, 1+\epsilon]$. In plain English, you still push good actions up and bad ones down, but you refuse to move the policy more than a fixed fraction per update. That probability ratio is the thread: it measures how far the new policy has drifted, and clipping it is the mechanism that makes large-scale policy optimization stable. We unpack all of it in the [TRPO & PPO](../06-trpo-ppo/README.md) post.
