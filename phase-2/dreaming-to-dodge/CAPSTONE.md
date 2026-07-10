# Capstone — Make the Dream Learn to Catch 🌙🏓

**Take the IRIS-from-scratch agent that currently plays at random, and make it beat the
baseline.** As shipped (one training round, `configs/catch.yaml`), the agent catches the ball
about **11%** of the time — statistically identical to random (**10–15%**). The oracle catches
**100%**. Your capstone is to close that gap.

This is the finale of Project 2. The throughline of the whole bootcamp — *learn a value/policy
from experience* — gets a model-based twist here: **the experience is imagined.** Your policy
never touches the real environment during training; it practises inside a world model you have
to make good enough to practise in.

> **The core lesson you're operating on:** *the world model is the easy part; the policy (and
> the codebook) are the hard part.* The world model already works (98.2% next-token accuracy).
> Your leverage is the **tokenizer** and the **policy** — and they're coupled.

---

## 1. The starting point (run this first)

Reproduce the honest baseline so you can see the floor you're beating:

```bash
modal run modal_apps/smoke.py                                   # ~5 min, cheap GPU
modal run modal_apps/train.py::run_all --cfg configs/catch.yaml # ~23 min, A10G
modal run modal_apps/evals.py::eval  --cfg configs/catch.yaml --n 500
modal run modal_apps/evals.py::dreams --cfg configs/catch.yaml
modal run modal_apps/evals.py::sync
./.plotvenv/bin/python make_student_figures.py
```

Look at `figures/reconstruction_grid.png`: **the ball is gone** in the decoded row. Look at
`figures/curves_tokenizer.png`: codebook usage collapsed to ~1.2%. Look at
`figures/curves_policy.png`: imagined return is flat at the random line. Those three pictures
*are* the problem statement.

---

## 2. Why it fails (read before you touch anything)

The two failures are **coupled**, and understanding the coupling is half the assignment:

1. **The tokenizer's codebook collapsed** to ~3 of 256 codes (classic VQ-VAE posterior
   collapse). With so few codes it captures the near-constant **paddle** and drops the small,
   fast **ball**.
2. **So the world model's dream has no ball in it.** The Transformer is 98% accurate — but only
   at reproducing what the tokenizer kept. A dream with no ball has nothing to catch.
3. **So the policy gets no signal.** The reward head is 90% accurate on tokens, but a catch vs.
   miss can't be grounded in anything the agent *sees*. The actor-critic correctly settles at
   random.

**Fixing the codebook is the first domino.** A policy fix alone, on a ball-less dream, will not
work — verify your fix actually puts the ball back in the reconstructions before you spend
compute on the policy.

---

## 3. The challenge list (pick your attack)

You may **edit the model code** for the capstone (unlike the base run). Ordered roughly by
leverage:

### A. Un-collapse the codebook (tokenizer) — *do this first*
- **EMA codebook updates** instead of the codebook-loss term (the standard VQ-VAE-2 fix).
- **Lower the commitment coefficient** `tokenizer.beta` (try 0.1 or 0.05).
- **Dead-code re-initialization / random restarts**: periodically reset unused codes toward
  recent encoder outputs.
- **Sanity gate:** re-render `reconstruction_grid.png` — *the ball must reappear* and codebook
  usage should climb well above 1.2%.

### B. Run the actual IRIS loop (data)
- Raise `collect.rounds` from `1` toward `5`. Each round re-collects with the improved policy
  and co-improves all three components. This is the single biggest lever and how IRIS is meant
  to be run. (Watch the budget — the policy stage dominates wall-clock.)

### C. Make the policy learn (actor-critic)
- **More / longer policy training** (`policy.updates_per_round`).
- **Reward shaping** — the terminal ±1 is sparse; a denser intermediate signal (e.g. paddle-to-
  ball horizontal distance) gives the actor-critic a gradient.
- **Larger imagination horizon** (`policy.horizon`) so the agent can plan the full fall.
- **Entropy schedule** — anneal `policy.entropy_coef` so the policy eventually commits.

### D. Verify the reward is learnable
- Probe: can a small classifier on the tokens predict the reward? If not, your tokenizer still
  isn't preserving what's needed — go back to (A).

---

## 4. What you submit (the contract)

A single folder / PR containing:

1. **Your changed code** (`tokenizer.py` / `actor_critic.py` / `configs/*.yaml` / …) — read.
2. **`results/eval_<tag>.json`** from your best run — the official number is the real-env
   **catch rate** over 500 episodes (from `modal run modal_apps/evals.py::eval`).
3. **The figures** for your run (`make_student_figures.py`): at minimum the new
   `reconstruction_grid.png` (ball back?), `curves_tokenizer.png` (usage up?), and
   `return_vs_baseline.png` (beat random?).
4. **A ≤1-page report** — what you changed, one before/after on the codebook, your best catch
   rate vs. the 0.11 baseline, and one remaining failure mode.

> Numbers are re-checked: I re-run `eval` on your synced checkpoint. A reported catch rate that
> doesn't reproduce flags tampering.

---

## 5. Grading (100 pts)

| | pts |
|---|---|
| Codebook un-collapsed (usage clearly ↑; ball visible in reconstructions) | 30 |
| Real-env **catch rate beats the random band** (> ~0.20, clear of 0.10–0.15) | 30 |
| Correct diagnosis in the report (the coupling, not just "trained longer") | 20 |
| Reproducibility (your `eval` re-runs to the reported number) | 15 |
| Leaderboard placement — highest catch rate (curved) | 5 |

**Tiers.**
- **Core:** get the ball back into reconstructions **and** beat the random band on catch rate.
- **Competitive:** climb toward the oracle — full IRIS loop (`rounds≥5`) + reward shaping +
  horizon tuning.
- **Stretch:** switch `configs/pong.yaml` and repeat — a genuinely harder environment.

---

## 6. Rules

- **Same base setup for everyone:** pixel Catch, `64×64×3`, IRIS's three components. We measure
  *your fix*, not a bigger environment. (The Stretch/Pong track is scored separately.)
- **No cheating the imagination:** the policy must still be trained **only** on world-model
  rollouts (that's the whole point of a world model). Don't train the actor-critic directly on
  real-env transitions.
- **No hard-coding the oracle.** The paddle policy must come from learning, not a rule that
  reads the ball position from the env.

---

## 7. Budget & timeline

- **Budget:** the shipped round is ~23 min / ≈ $0.4 of A10G. Full IRIS loop (`rounds=5`) is
  ≈ 1.5–2 h. Keep `smoke.py` in your loop — debug cheap, then spend.
- **Suggested week:** (1) reproduce the baseline, read the three diagnostic figures. (2–3) fix
  the codebook, confirm the ball is back. (4–5) run the IRIS loop + policy shaping. (6) lock
  your best `eval`, write the report. (7) submit.

*Make the dream contain a ball, and the agent can finally learn to catch it.*
