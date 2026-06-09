# Lecture 01 — Fundamentals

**Duration:** 2 hours
**Date:** Cohort 2026 — opening lecture

## 🖥️ Slides

**Live deck:** https://rl-bootcamp-decks.vercel.app/lecture-01-fundamentals/
(arrow keys / click to advance; press `o` for the slide overview).

## What this lecture covers

The whole lecture runs on **one concrete, lived example — an evening commute home in Pune** (the drive from the office, with a stochastic stretch at the Wakad bridge). Every idea below lands on that one drive.

| Part | Topic | Time |
|---|---|---|
| 1 | A brief history of RL — Bellman → Sutton → DQN → AlphaGo → PPO → RLHF → GRPO → **Agentic RL → 2026 and beyond** | 22 min |
| 2 | Markov Decision Processes — the (S, A, P, R, γ) tuple, the Markov property, discounting | 15 min |
| 3 | Value functions — V(s) and Q(s, a), the return, why value depends on the policy | 18 min |
| 4 | The Bellman backup — derivation, *how a reward flows back* (chess), AlphaGo, bootstrapping | 25 min |
| 5 | DP, Monte Carlo, and TD **on the commute** — three recipes for the same Bellman recursion (plus policy iteration near DP) | 15 min |

## The single sentence

> **The value of where I am is the reward I just got, plus a discounted value of where I'll be one step from now.**

Every algorithm in the rest of the bootcamp — Q-learning, DQN, PPO, RLHF, GRPO, agentic RL — is a variation on that one sentence.

## Hands-on code

Run the comparison from the codebase:

```bash
cd code/tic-tac-toe
python compare.py
```

You should see all three algorithms (DP, MC, TD) converge to **V(empty board) ≈ +0.99** against a random opponent — three different paths to the same answer.

Read [`code/tic-tac-toe/README.md`](../../code/tic-tac-toe/README.md) for the full walkthrough.

## Mathematical companion (the proofs)

The lecture itself teaches everything *from an intuitive lens* and pushes the formal
arguments to homework. The companion PDF is that homework, written out properly:

📄 **[`lecture-01-proofs.pdf`](lecture-01-proofs.pdf)** — *Lecture 1 Mathematical Companion*

It contains, with short self-contained proofs:

1. The return is finite (bounded by `R_max / (1 − γ)`).
2. Derivation of the Bellman expectation equation from the definition of the return.
3. The Bellman operator is a **γ-contraction** in the sup-norm.
4. **Why value iteration converges** — the Banach fixed-point theorem and the `γ^k` rate.
5. Monte Carlo: the sample mean is unbiased and consistent; the incremental-average update.
6. **TD vs. MC**: the bias–variance trade-off, and tabular TD(0) convergence (Robbins–Monro).

Nothing here is needed to *run* the algorithms — it is what lets you *trust* them. Source: [`lecture-01-proofs.tex`](lecture-01-proofs.tex) (compile with `xelatex`, run twice for the table of contents).

## Practice materials

Three companion PDFs in [`practice/`](practice/), in increasing order of doing-it-yourself:

1. 📄 **[`practice-1-commute.pdf`](practice/practice-1-commute.pdf)** — DP vs. Monte Carlo vs. TD, worked end-to-end on the Pune commute. The same example as the lecture, on paper.
2. 📄 **[`practice-2-game.pdf`](practice/practice-2-game.pdf)** — the same three methods on a *second* example (clearing a mobile-game level, maximizing coins) so the pattern generalizes.
3. 📄 **[`assignment-mars-rover.pdf`](practice/assignment-mars-rover.pdf)** — the graded assignment: build a **Mars Rover** environment in **Gymnasium** and estimate its value function with DP, MC, and TD. Starter code in [`code/mars-rover/`](../../code/mars-rover/).

## Suggested reading

Pick the format that suits you best.

### Textbook

- **Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed)**
  - Ch. 3 — Finite Markov Decision Processes
  - Ch. 4 — Dynamic Programming
  - Ch. 5 — Monte Carlo Methods
  - Ch. 6 — Temporal-Difference Learning

### Papers (in chronological order, matching the §2 timeline)

- Bellman, *Dynamic Programming*, 1957 — read the preface and §1.
- Sutton, *Learning to predict by the methods of temporal differences*, 1988.
- Tesauro, *Temporal Difference Learning and TD-Gammon*, 1995.
- Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013 (DQN).
- Silver et al., *Mastering the game of Go with deep neural networks and tree search*, 2016 (AlphaGo).
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 (PPO).
- Christiano et al., *Deep Reinforcement Learning from Human Preferences*, 2017 (the RLHF foundation).
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2024 (GRPO).

### Talks (these cover *exactly* today's material)

- **David Silver, *DeepMind RL Course*** — Lectures 1–4 (Intro → MDPs → Planning by DP → Model-Free Prediction) line up almost exactly with today.
- **Stanford CS234 (Emma Brunskill)** — the opening lectures cover MDPs, DP, and model-free prediction (MC / TD).
- Andrej Karpathy, *Deep Reinforcement Learning: Pong from Pixels* (blog post) — a bridge toward Lecture 03.

## Before Lecture 02

The next lecture is on Q-learning and DQN. To prepare:

1. **Do the assignment** — [`practice/assignment-mars-rover.pdf`](practice/assignment-mars-rover.pdf). Build the Mars Rover Gymnasium env and value it with DP / MC / TD. Starter: [`code/mars-rover/`](../../code/mars-rover/).
2. Run `python code/tic-tac-toe/compare.py` and confirm DP, MC, and TD agree on a second small MDP.
3. (Optional) In `code/tic-tac-toe/td_agent.py`, swap the dictionary `V` for a small PyTorch MLP. That's the leap we make in Lecture 02.

## Errata, questions, discussion

Open an issue on this repo, or email `hello@vizuara.com`.
