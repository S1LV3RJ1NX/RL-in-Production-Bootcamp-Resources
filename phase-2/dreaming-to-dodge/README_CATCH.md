# Dream to Catch — an IRIS-from-scratch world model 🌙🏓

**Project 2 of Vizuara's RL-in-Production bootcamp.** Build the three components of
[IRIS](https://arxiv.org/abs/2209.00588) *from scratch* — a VQ-VAE tokenizer, a Transformer
world model, and an actor-critic — then train an agent to play a pixel game **entirely inside
its own imagination**, on Modal, for a couple of dollars.

This is a *deliberately honest* project. When you run it as shipped (one training round,
~23 min on an A10G), **two of the four components work beautifully and two fail in ways that
teach you exactly where the hard problems in model-based RL live.** Your job in the capstone
(see [`CAPSTONE.md`](CAPSTONE.md)) is to fix them.

> **The one-line result:** *the world model is the easy part; the policy (and the codebook)
> are the hard part.*

Open [`demo/index.html`](demo/index.html) for the full illustrated story, or read
[`paper/main.pdf`](paper/main.pdf) for the short report.

---

## What a world model is (30-second version)

A **world model** is a neural net that *simulates an environment*: give it the current frame
and an action, it predicts the **next frame, the reward, and whether the episode ends**. Once
it's good enough, you can train a policy on *imagined* rollouts — the agent practises in a
learned dream instead of the real (slow/expensive) environment. IRIS makes the world model a
**next-token Transformer over image tokens**, i.e. a language model for pixels.

The three parts, and what happened in our run:

| Part | What it does | Params | Result |
|---|---|---|---|
| **1. Tokenizer** (VQ-VAE) | `64×64×3` frame → **16 discrete tokens** (256-code codebook) | 0.83M | recon MSE **0.0165** ✅ but codebook usage **1.2%** ❌ (collapse) |
| **2. World model** (Transformer) | predict next tokens + reward + done | 3.34M | **98.2%** next-token acc ✅ |
| **3. Actor-critic** | policy trained **only** in imagination (horizon 10) | 0.82M | catch rate **0.11 ≈ random** ❌ |

---

## Repo layout

```
tokenizer.py        VQ-VAE (encoder / codebook / decoder)     ← the from-scratch model code
world_model.py      token Transformer (transition/reward/done)   (do NOT edit for the base run)
actor_critic.py     the policy + value networks
imagination.py      rollout in the dream + dream/reconstruction rendering
envs.py             the pixel Catch environment

configs/catch.yaml       the primary config (the ~23-min A10G run)   ← your knobs live here
configs/catch_fast.yaml  a smaller/faster variant
configs/pong.yaml        a harder environment (stretch)

modal_apps/
  common.py         shared Modal App, image, Volume, helpers
  smoke.py          STAGE-1 smoke test: whole pipeline on a cheap GPU in ~5 min
  train.py          collect / tokenizer / world_model / policy / run_all entrypoints
  evals.py          eval / dreams / sync entrypoints

results/            synced result JSON (learning curves, eval, dreams) — REAL numbers
figures/            the 7 figures + figures/dreams/*.gif
make_student_figures.py   rebuilds every figure from results/*.json
demo/index.html     the student-facing story page
paper/main.tex, main.pdf  the short report
```

---

## Setup: Modal auth (one time)

Everything trains on [Modal](https://modal.com) — no local GPU needed.

```bash
pip install modal
modal token new          # opens a browser to authenticate
# (bootcamp cohort: use the teamvizuara profile if you were given one)
```

The Modal image (torch + gymnasium + numpy + einops + imageio) and a persistent Volume
(replay buffer, checkpoints, dreams, results) are declared in `modal_apps/common.py` — Modal
builds the image on first run. You do **not** `pip install` the training deps locally.

For rebuilding figures locally you only need matplotlib/numpy/pillow:

```bash
./.plotvenv/bin/python -c "import matplotlib, numpy, PIL; print('ok')"
# if missing:  ./.plotvenv/bin/pip install matplotlib pillow numpy
```

---

## Run it, step by step

Every command is a Modal `local_entrypoint`; results land on the Modal Volume and are pulled
down with `sync`.

```bash
# 1. SMOKE — prove the whole pipeline runs end-to-end on a CHEAP GPU (~5 min).
#    collect → tokenizer → world model → 1 imagined update → eval → 1 dream.
#    Fix bugs HERE before spending real budget.
modal run modal_apps/smoke.py

# 2. RUN_ALL — the full collect→train round (~23 min on an A10G).
#    Each round: collect (real env) → train tokenizer → train world model →
#    train policy PURELY in imagination. Writes per-round curves to the Volume.
modal run modal_apps/train.py::run_all --cfg configs/catch.yaml --tag catch

#    (individual stages, if you want to run them one at a time:)
#    modal run modal_apps/train.py::collect      --cfg configs/catch.yaml
#    modal run modal_apps/train.py::tokenizer    --cfg configs/catch.yaml
#    modal run modal_apps/train.py::world_model  --cfg configs/catch.yaml
#    modal run modal_apps/train.py::policy       --cfg configs/catch.yaml

# 3. EVAL — policy vs random vs oracle over N real-env episodes (default 500).
modal run modal_apps/evals.py::eval --cfg configs/catch.yaml --n 500

# 4. DREAMS — render imagined rollouts + reconstructions + world-model-vs-real.
modal run modal_apps/evals.py::dreams --cfg configs/catch.yaml --n 6

# 5. SYNC — download every result JSON from the Volume into results/.
modal run modal_apps/evals.py::sync --outdir results

# 6. FIGURES — rebuild all 7 figures from the synced JSON (no GPU).
./.plotvenv/bin/python make_student_figures.py
```

Then open `demo/index.html` and `paper/main.pdf`.

---

## The config knobs (`configs/catch.yaml`)

The whole experiment is one YAML file. The knobs you'll actually turn for the capstone:

```yaml
collect:
  rounds: 1              # ← THE big one. 1 = the shipped result. IRIS-style loop uses ~5.
  steps_per_round: 8000  # real env steps collected each round
  eps_greedy: 0.05       # exploration noise at collect time

tokenizer:
  num_tokens: 16         # K — tokens per frame (4×4 latent grid)
  vocab_size: 256        # N — codebook entries (only ~3 got used → collapse)
  beta: 0.25             # ← commitment weight; lower it to fight collapse
  steps_per_round: 1500

world_model:
  max_timesteps: 10      # L — context length (seq = 10×(16+1) = 170)
  num_layers: 4          # this component already works
  steps_per_round: 3000

policy:
  horizon: 10            # H — imagination horizon (≥ episode length 7)
  entropy_coef: 0.01
  updates_per_round: 800 # ← more updates / more rounds is the main policy lever
```

> ⚠️ For the base run, **don't edit the model code** (`tokenizer.py`, `world_model.py`, …) —
> tune behaviour through the config. The capstone explicitly *does* let you edit the code.

---

## The honest expected outcome

When you run `configs/catch.yaml` as shipped, expect **exactly** this (these are the real
numbers in `results/`):

- **Tokenizer:** recon MSE ≈ **0.0165** (good) but codebook usage ≈ **1.2%** — collapsed to
  ~3 of 256 codes. The moving ball disappears from reconstructions.
- **World model:** next-token acc ≈ **98.2%**, reward acc ≈ **90.3%**, done acc ≈ **91.3%**;
  dream-vs-real MSE ≈ **0.02**. This part works.
- **Policy:** imagined return ≈ **−0.78** (flat/declining over updates, entropy stays ~0.88);
  real-env catch rate ≈ **0.11** vs random **0.10–0.15**, oracle **1.0**. It does **not** learn
  to catch on one round.
- **Budget:** ~23 min for the round on A10G; whole project ≈ **1.5–2 GPU-hours ≈ $2–3** (est.).

That is not a bug in the code — it's the assignment. The failures are *coupled*: the tokenizer
threw away the ball, so the dream has nothing to catch, so the policy has no gradient. **Fixing
the codebook is the first domino.** See [`CAPSTONE.md`](CAPSTONE.md).

---

## Where to read more

- **`demo/index.html`** — the full illustrated walkthrough (world models → the 3 parts →
  results → the fixes).
- **`paper/main.pdf`** — the short report ("Dream to Catch").
- **`CAPSTONE.md`** — the assignment: *make it beat the baseline.*
- IRIS paper: *Transformers are Sample-Efficient World Models*, Micheli et al., 2023.
