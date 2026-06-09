# Mars Rover — Lecture 01 assignment starter

Build a small **Gymnasium** environment and estimate its value function with **DP, Monte Carlo, and TD**.
Full spec + grading: [`../../lectures/01-fundamentals/practice/assignment-mars-rover.pdf`](../../lectures/01-fundamentals/practice/assignment-mars-rover.pdf).

## Why Gymnasium (not Gym)

The original `openai/gym` is **deprecated**. The living standard is **[Gymnasium](https://gymnasium.farama.org/)**,
maintained by the **Farama Foundation** — same API, drop-in successor. Note the modern API:

- `reset()` returns `(observation, info)`
- `step()` returns a **5-tuple** `(observation, reward, terminated, truncated, info)`
- tabular envs expose the model as `env.unwrapped.P[state][action] = [(prob, next_state, reward, done), ...]`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # or: uv venv
pip install gymnasium numpy matplotlib
python starter.py
```

## What to do

`starter.py` is a scaffold with `TODO`s. Fill them in:

1. **The environment** (`MarsRoverEnv`): complete `_reward` (TODO 1) and the transition model in `_build_model` (TODO 2).
2. **DP** — `value_iteration` (reads `env.P`).
3. **MC** and **TD(0)** — model-free; your code may call only `env.reset()` / `env.step()`.

Then compare all three (they should agree where the greedy policy actually drives) and, for Problem 4,
build your *own* small environment the same way.

> The full worked reference is intentionally **not** in this repo — it's the assignment. Ask on the issues
> tab or email `hello@vizuara.com` if you get stuck.
