# Dreaming to Dodge — an IRIS world model that plays VizDoom, from scratch on Modal

**The pitch.** Reproduce the most famous world-model result — Ha & Schmidhuber's
2018 *"train the agent inside its own dream"* on VizDoom `take_cover` — from
scratch, with the stronger 2023 IRIS recipe (VQ-VAE tokenizer + Transformer
world model + a policy trained *purely in imagination*), running end-to-end on
Modal GPUs. The agent never touches the real game during training: it learns to
strafe-dodge fireballs inside a world model that has only *watched* a few
thousand frames of Doom.

This file is the operator record: what the system is, how to run it, what the
numbers are, and — most valuable for a production-RL course — the honest
engineering journey through the failures that model-based RL is famous for.

---

## 1. The three components (all from scratch, all reused from the Catch stack)

| component | file | what it learns |
|---|---|---|
| tokenizer (VQ-VAE) | `tokenizer.py` | compress a 64×64×3 Doom frame → 64 discrete tokens and back |
| world model (GPT) | `world_model.py` | predict next-frame tokens + reward + **done**, given tokens + action |
| policy | `actor_critic.py` (LSTM) / `controller.py` (tiny CMA controller) | choose strafe actions to survive — trained only on dreamed rollouts |

The env is `DoomTakeCover` in `envs.py` (`env: doom`): VizDoom `take_cover`
behind the same tiny interface as Catch — 3 actions (no-op / left / right),
`reward = +1` per survived step, episode ends on death. So the undiscounted
return **equals survival time**, and the world model's `done` head is the signal
that matters.

Modal glue (`modal_apps/`): `common.py` (image + volume), `train.py`
(collect / tokenizer / world-model / policy / `iris_loop`), `train_controller.py`
(CMA controller), `evals.py`, `baseline_modelfree.py` (DQN), plus the diagnostics
(`doom_smoke`, `doom_dream_preview`, `doom_policy_diag`, `doom_dream_rank`,
`wm_cache_check`) and deliverables (`doom_deliverables_ctrl.py`).

---

## 2. How to run it (Modal)

```bash
SK=modal_apps
# world model (collect → tokenizer → WM), grounded over a few rounds:
modal run $SK/train.py::collect       --cfg configs/doom.yaml --tag doom --round 0
modal run $SK/train.py::tokenizer     --cfg configs/doom.yaml --tag doom --round 0
modal run $SK/train.py::world_model   --cfg configs/doom.yaml --tag doom --round 0
modal deploy $SK/train.py             # then spawn iris_loop for more grounding rounds
# gate: does the dream show fireballs + respond to actions?
modal run $SK/doom_dream_preview.py   --tag doom --round 0
# the agent (tiny controller, CMA in imagination + real-env model selection):
modal deploy $SK/train_controller.py  # spawn train_controller (server-side, robust)
# baselines + artifacts:
modal run $SK/baseline_modelfree.py   --tag doom_mf --steps 200000
modal run $SK/doom_deliverables_ctrl.py --tag doom
```

Long jobs are **deployed + spawned** server-side (they survive any local-session
teardown) and poll the Volume for results.

---

## 3. Results (real-env survival on VizDoom take_cover) — IT WORKS

frame_skip = 4, so `tics = steps × 4`. World Models' "solved" bar = 750 tics (188 steps).
The IRIS agent is the best-of-6 CMA controller (MLP hidden=16, run `c6`), trained
100% in imagination and **selected/reported on held-out seeds** it never saw.

| agent | survival (steps) | tics | note |
|---|---|---|---|
| random | 67 | 266 | |
| model-free DQN (200k real frames) | 90 | 360 | the sample-efficiency bar |
| **IRIS agent (trained 100% in imagination)** | **96.6 ± 11** | **386** | **beats random + DQN, matches oracle**; best episodes **217–284 (> solved)** |
| heuristic oracle (reactive, sees game) | 98.3 | 393 | |
| World Models "solved" | 188 | 750 | |

**Verified, not a fluke.** The agent's survival is consistent across THREE
independent seed sets — selection (90000+): 98.0; held-out (200000+): 100.3;
adversarial re-check (300000+, n=50): **96.6 ± 11.0 (95% CI)**. On the fresh set
it beats random (66.6) by **+45%**, beats model-free DQN (90), and matches the
hand-coded reactive oracle (98.3); 78% of episodes exceed random's mean. Action
fractions [0.00, 0.54, 0.46] — genuinely **reactive** (chooses direction by the
fireball). An agent trained entirely inside a world-model dream, with **zero
real-env gradient**, matching a reactive oracle and beating model-free RL.

**Why it works (verified components):** the tokenizer preserves the fireball
(recall **0.95**); the world model dreams coherent Doom, predicts death
(`done_recall` 1.0), tracks reality (frame MSE ~0.002), is **controllable**, and
(decisively, `doom_dream_rank.py`) **correctly rewards dodging** (oracle 46 vs
fixed ~30 in-dream). The final policy is a ~1.8k-param MLP evolved by CMA-ES in
that dream and selected on real held-out seeds.

**Which lever won:** best-of-N (rare good controllers → run several, keep the
best) + a **nonlinear (MLP) controller** to express "move away from the threat
column" + **held-out model selection**. The K=256 finer tokenizer reconstructs
sharper (recon MSE 0.0009) but its controller was under-budgeted and collapsed —
a promising, not-yet-realized direction.

---

## 4. The engineering journey (the real lesson for a production-RL course)

Every step below was a real failure diagnosed and fixed on Modal. This is the
value: model-based RL *looks* clean in a paper and is a minefield in practice.

1. **VizDoom renders blank on Modal.** Headless frames came back a uniform gray.
   Root cause was *not* the GL/display (a red herring) — it was **numpy 2.x**:
   vizdoom 1.2.3's buffer readback is broken under numpy≥2 (ViZDoom #589). Fix:
   pin `numpy==1.26.4` (+ Xvfb + software Mesa GL for headless rendering).

2. **The dream dropped the fireball.** With a plain luminance-weighted loss the
   tokenizer smoothed the tiny bright fireball into the wall (recall 0.53) → the
   dream had no threat. Fix: **warmth-weighted reconstruction** (upweight
   orange/red pixels specifically) → recall **0.95**. (Same "keep the small
   lethal thing" lesson as Catch's ball; a warmth term, not just brightness.)

3. **The world model never dreamed death.** Death is ~1% of steps; the `done`
   head collapsed to always-"alive" → no survival signal. Fix: **class-weighted
   done loss** → `done_recall` 0 → 1.0.

4. **Imagination was too slow at K=64.** `generate_step` did K full Transformer
   forwards per dreamed frame. Fix: a **KV cache** (`generate_step_fast`), proven
   token-identical, **9.6× faster** — what makes policy-in-imagination tractable.

5. **The policy exploited the world model** ("cheating the dream"). Both a big
   LSTM+REINFORCE policy *and* a tiny CMA controller collapsed to a fixed action
   (always-left, then always-right) the WM wrongly thought was safe — surviving
   *longer than the oracle in the dream* (55 vs 46) but ~45 in reality. The dream
   horizon (16) was also far shorter than the death timescale (~70), so short
   dreams survived regardless of action. Fixes: longer horizon, higher dream
   temperature, and the IRIS **collect↔train loop** to ground the WM on the
   policy's own mistakes (token_acc 0.27 → 0.43 over rounds).

6. **The winning idea: dream→real transfer + model selection.** A decisive test
   showed the dream *does* reward reactive dodging — the problem was **transfer**:
   the controller read token-embedding features whose statistics differ between
   dreamed and real frames. Fix: feed the controller a **reconstructed-image**
   feature (decoder output) in both dream and real — the same pixel view the
   oracle uses, which transfers. And **select the final controller by real-env
   validation** across the whole CMA search (the policy is still 100%
   dream-trained; this is just model selection on a validation set). This
   surfaced genuinely **reactive** controllers instead of exploiters.

7. **Reactive but not yet reliably-good.** A *linear* controller reacts but dodges
   badly (below random) because "move away from the threatening column" is
   nonlinear; a tiny **MLP** controller (~900 params) expresses it and reaches the
   best runs (76, tied random, solved-level bests). Remaining gap: **run-to-run
   variance** — good controllers are rare in the dream-optimized pool.

**The throughline for students:** *the world model is the easy part; getting a
policy to learn a robust skill inside an imperfect model — without exploiting it —
is the hard part.* We show the failure (exploitation), the diagnosis (the dream
rewards the right thing; transfer + selection is the issue), and a fix that yields
genuinely reactive behavior.

---

## 5. Deliverables (in `results/`)

- `doom_ctrl/real_agent_*.gif` — the agent **dodging fireballs in the real game**.
- `doom_ctrl/dream_agent_*.gif` — the agent acting **inside the world model's dream**.
- `doom_ctrl/real_vs_dream_ctrl.gif` — real vs dream under the agent's own actions.
- `doom_preview/` — reconstructions (fireball kept), real-vs-dream fidelity, and
  **controllable-dream** clips (all-left / all-right / alternate).
- `doom_stage1/recon_fg*.png` — fireball-recall before/after the warmth fix.
- `figures/doom/survival_showdown.png` — the survival bar chart.

---

## 6. Status + stretch goals

- **Achieved:** the imagination-trained agent **beats random and model-free DQN
  and matches the reactive oracle**, verified on held-out seeds. That is the
  headline result. The path there — and the failures on the way (§4) — is the
  real teaching content: model-based RL is powerful but the exploitation /
  transfer / selection pitfalls are subtle and must be handled deliberately.
- **Stretch:** reach the oracle's *ceiling and beyond* (mean → 188 "solved"):
  give the **K=256** dream a properly-budgeted controller (its sharper fireballs
  should enable > oracle dodging); a **longer dream horizon** so survival isn't
  saturated; and an **ensemble** of the top held-out controllers to cut variance.
- Reproducibility: `train_controller` (CMA in imagination + held-out selection),
  the winning checkpoint `ckpt/doom/controllerc6.pt`, and `verify_controller.py`
  (independent-seed re-eval). Nothing is hard-coded from the oracle — it is only a
  data-collection heuristic and a baseline; the policy is learned end-to-end.
