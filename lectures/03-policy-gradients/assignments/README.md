# Lecture 3 — Coding Assignment (Policy Gradients)

One partially-filled Colab notebook. Read the `# PROVIDED` cells, then fill every
`# TODO`. Each TODO is one or two lines with the slide's equation right beside it.
PyTorch from scratch — **no Stable-Baselines3**. Includes written ablations and a
measured **variance ladder**.

## Build the Archer — REINFORCE → baseline → Actor-Critic
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VizuaraAI/RL-in-Production-Bootcamp-Resources/blob/main/lectures/03-policy-gradients/assignments/L3_Build_the_Archer.ipynb)

CPU only, trains in a few minutes. You build the lecture's own running example as two
custom `gymnasium` environments — a one-state **bandit** (nine firing angles) then an
**MDP** where the archer walks in before it shoots — and implement all three
policy-gradient methods by filling six short blanks:

1. **`act()`** — sample an action and read its `log_prob` + entropy.
2. **REINFORCE (bandit)** — the one-shot loss `-(r · logπ)`.
3. **`discounted_returns`** — `Gₜ = rₜ + γ·Gₜ₊₁`.
4. **REINFORCE (MDP)** — weight = the return `Gₜ`.
5. **REINFORCE + baseline** — advantage `Gₜ − V(s)` + the critic MSE.
6. **Actor-Critic** — the TD-error advantage `r + γV(s′) − V(s)`.

**What good looks like:** the bandit fan concentrates on the bullseye (mean reward
≈ 0.8 vs uniform 0.42); REINFORCE on the MDP *stalls* (greedy ≈ 4), while
**+baseline** and **Actor-Critic** both learn *walk-in-then-shoot* (greedy ≈ 9.6).
The deliverable that ties it together is the **variance ladder** — the gradient
estimator's variance measured on one fixed policy, dropping roughly
`0.15 → 0.007 → 0.001` as you add the baseline and then the critic.

> Tip: in Colab, **save a copy to your Drive** (File → Save a copy in Drive) before
> you start so your work persists across disconnects. The companion **assignment
> PDF** (on the lecture page) has the full spec, grading rubric, and a continuous-aim
> bonus.
