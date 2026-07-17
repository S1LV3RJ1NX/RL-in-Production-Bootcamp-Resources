# HIL-SERL — Annotated Reading/Watching List

All URLs WebFetch-verified to resolve on 2026-07-16 **except** the flagged entries. Two URLs from the original candidate list were dropped/flagged after verification: `huggingface.co/blog/hilserl` (**404 — does not exist**, removed) and the Science Robotics page (**403 to bots**, DOI real, kept with flag).

**HIL-SERL in one line:** SAC (off-policy RL) + RLPD (sample-efficiency trick) + a learned pixel reward-classifier (real-robot only) + occasional human interventions that fade to zero → near-100% real-robot manipulation in ~1–2.5 h.

## A. Fastest path to intuition (do first, in order)
1. **HIL-SERL project page** — https://hil-serl.github.io — 5-min orientation: reward classifier → demos → fading interventions; task videos. Start here.
2. **Intro/results video** — https://www.youtube.com/watch?v=GoJSW8e2qbI — the real robot learning; watch right after the project page.
3. **Indraneel Patil — "Learnings from deploying HIL-SERL on SO-101"** — https://indraneelpatil.github.io/blog/2026/hil-serl — honest 7-experiment practitioner write-up on the exact arm. (His single-setup "sim-pretrain didn't help" — don't over-generalize; tight workspace bounds mandatory.)

## B. Core papers (dependency order, bottom-up)
4. **SAC** (Haarnoja et al. 2018) — https://arxiv.org/abs/1801.01290 — the base max-entropy off-policy actor-critic. Read first.
5. **RLPD** (Ball, Smith, Kostrikov, Levine, ICML 2023) — https://arxiv.org/abs/2302.02948 — symmetric 50/50 sampling + LayerNorm critics + large ensemble (E=10 default) + high UTD (G=20 state-based default), clipped double-Q over Z=2. Code: https://github.com/ikostrikov/rlpd. **This is *why* HIL-SERL converges in an hour.** (Note: these E/Z/G are RLPD's defaults; LeRobot's sim SAC config uses num_critics=2/utd_ratio=2, not 10/20.)
6. **SERL** (Luo et al., ICRA 2024) — https://arxiv.org/abs/2401.16013 · project https://serl-robot.github.io — the demos-only predecessor. HIL-SERL = "SERL + human-in-the-loop."
7. **HIL-SERL** (main paper) — https://arxiv.org/abs/2410.21845 (HTML: https://arxiv.org/html/2410.21845v1).
8. **Science Robotics published version** — DOI 10.1126/scirobotics.ads5033 — https://www.science.org/doi/10.1126/scirobotics.ads5033 — **FLAGGED: 403 to automated fetchers; DOI is real, resolves via doi.org.** Cite in formal materials.

## C. Hands-on / code
9. **HIL-SERL official code (JAX)** — https://github.com/rail-berkeley/hil-serl — the reference implementation the paper was built on.
10. **Franka walkthrough doc** — https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md — **FLAGGED: transient GitHub render error on fetch; file confirmed to exist, content not directly read.** End-to-end pipeline.
11. **SERL code (deprecated)** — https://github.com/rail-berkeley/serl · docs https://rail-berkeley.github.io/serl/ — carries an explicit "being deprecated — use HIL-SERL" notice. Reference only.
12. **LeRobot sim doc (gym-hil)** — https://huggingface.co/docs/lerobot/hilserl_sim — **best starting point for a class**: full HIL-SERL stack in MuJoCo Franka Panda, zero hardware. *(This build is based on it.)*
13. **gym-hil repo** — https://github.com/huggingface/gym-hil — the Gymnasium HIL envs (Base/Gamepad/Keyboard).
14. **LeRobot real-robot HIL-SERL doc** — https://huggingface.co/docs/lerobot/hilserl — the richest operational manual: config schema, workspace bounds, ROI cropping, reward-classifier training (`helper2424/resnet10`), gRPC actor/learner commands, human-intervention guide.
15. **LeRobot v0.6.0 release blog** — https://huggingface.co/blog/lerobot-release-v060 — the unified `lerobot.rewards` API. Currency check.

## D. SO-101-specific
16. **ggando — "HIL-SERL for SO-101: Real-World Grasping from Scratch"** — https://ggando.com/blog/so101-hil-serl — most complete concrete SO-101 recipe (~70% success, 757 episodes, ~3 h). **Treat exact numbers as order-of-magnitude, not a benchmark.**
17. **Indraneel Patil blog** (also §A) — https://indraneelpatil.github.io/blog/2026/hil-serl.
18. **LeRobot SO-101 setup doc** — https://huggingface.co/docs/lerobot/so101 — assembly, motor id/baudrate, calibration.
19. **so101-nexus** — https://github.com/johnsutor/so101-nexus — record→clone→reinforce scaffolding + optional Warp GPU-parallel PPO baseline (**PPO/BC, not the SAC HIL-SERL learner**).

## Config artifacts (load-bearing for this build)
- Sim env config — https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json
- Sim train config — https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/train_config.json
- Semi-first-party checkpoint (unofficial, personal namespace, obs/action match unverified) — https://huggingface.co/aractingi/sac_gym_hil_pick_lift

> **If you read only three:** the video (2), then RLPD (5), then the HIL-SERL paper (7).
