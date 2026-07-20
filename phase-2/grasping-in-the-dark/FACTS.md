# FACTS — HIL-SERL Grasp (RL-in-Production, Session 3)

> **Pinned source of truth.** Every build decision in this repo is grounded here. Facts are tagged with their verification status from the `hilserl-build-grounding` workflow (run `wf_1cbe55ee-86d`, 22 agents, adversarially verified against LeRobot/gym-hil source, the HIL-SERL paper, and Modal/Colab docs — 2026-07-16). Do **not** assert anything here that isn't grounded; flag uncertain items inline.

## 0. What we are building (RL-only — DAgger dropped)

A **production-quality** (not toy) robotics-RL workshop deliverable set around **HIL-SERL** in **simulation** (gym-hil Franka Panda `PandaPickCube`):
1. **Colab notebook** — teaching + eval: loads a checkpoint, runs autonomous eval rollouts, renders inline video, plots the reward curve, optional short autonomous fine-tune.
2. **Lecture deck** (Slidev) — the RL story: SAC → RLPD → SERL → HIL-SERL.
3. **Checkpoint + demo dataset** published to the HF Hub, trained on **Modal** (GPU).

**HIL-SERL in one line:** SAC (off-policy RL) + RLPD (sample-efficiency trick) + a learned pixel reward-classifier (real-robot only) + occasional human interventions that fade to zero.

## 1. Architecture (locked)

| Surface | Role | Why |
|---|---|---|
| **Modal (GPU)** | Runs the real SAC+RLPD training → checkpoint. | Long, headless, needs stable GPU + EGL MuJoCo. |
| **Google Colab (T4)** | Teaching + eval: load checkpoint, autonomous eval rollouts, inline video, reward curve, optional short fine-tune. | Interactive, shareable, zero-hardware. |
| **HF Hub** | Publishes checkpoint + demo dataset. | Decouples Modal (writes) from Colab (reads). |

**Human-in-the-loop handling (the honest split).** HIL-SERL's live human intervention (SpaceMouse/gamepad/keyboard) **cannot run headless in cloud** — the interactive envs (`PandaPickCubeGamepad-v0`, `PandaPickCubeKeyboard-v0`) need a display + HID device. So the workshop uses a **3-part pattern (verified sound):**
1. **Live autonomous eval** — checkpoint in `PandaPickCubeBase-v0` (no teleop, no human) → inline video. The guaranteed wow.
2. **Optional short autonomous SAC fine-tune** — `learner`+`actor` as background processes on `127.0.0.1:50051`, `PandaPickCubeBase-v0`; **the MuJoCo env supplies reward itself** → reward rises with zero human. (LeRobot docs show this "no-intervention" curve as supported.)
3. **Pre-recorded human-intervention video** — conveys the HIL mechanic that can't run live.

**Key enabling fact:** in **sim, SAC learns with zero human** because `gym_hil` computes reward from MuJoCo state (`_compute_reward()`). The reward *classifier* is **real-robot-only** — the human is a *teaching topic*, not a runtime dependency.

## 2. Exact LeRobot sim pipeline

**Install** (Python **≥3.12** required by lerobot 0.6.0 — Colab's 3.12 satisfies it; **no** 3.10 conflict):
```bash
git clone https://github.com/huggingface/lerobot && cd lerobot
pip install -e ".[hilserl]"     # pulls gym-hil (>=0.1.14,<0.2.0) + mujoco + grpcio + SAC stack
```
Authoritative example configs (already vendored in `configs/*.upstream.json`):
- env: `https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/env_config.json`
- train: `.../rl/gym_hil/train_config.json`

**Step A — (optional) record demos.** Example reuses `lilkm/pick_cube_franka_panda_30` (30 eps) → can skip. To record: set `mode:"record"`, `dataset.repo_id`, `dataset.num_episodes_to_record`, then `python -m lerobot.rl.gym_manipulator --config_path configs/gym_hil_env.json`.

**Step B — reward classifier: NOT needed in sim.** Sim `train_config.json` has **no** `reward_classifier`/`reward_model` block; MuJoCo supplies reward. (Real robot: `lerobot-train` with `reward_model{type:"reward_classifier", model_name:"helper2424/resnet10", model_type:"cnn", num_cameras:2, num_classes:2}`, wired via `env.processor.reward_classifier{pretrained_path, success_threshold, success_reward}`.) Teach it; don't run it in sim.

**Step C — actor + learner SAC (two processes, LEARNER FIRST):**
```bash
python -m lerobot.rl.learner --config_path configs/train_gym_hil.json   # gRPC 127.0.0.1:50051, buffers, gradients
python -m lerobot.rl.actor  --config_path configs/train_gym_hil.json   # steps env, streams transitions
```

**Step D — eval:** `python -m lerobot.rl.eval_policy`, or roll out the checkpoint in `PandaPickCubeBase-v0` directly (Colab).

### `train_config.json` — verified field map (see `configs/train_gym_hil.upstream.json`)
- **Top level** (`TrainRLServerPipelineConfig`, `src/lerobot/rl/train_rl.py`): `output_dir`, `job_name`, `resume`, `seed:1000`, `num_workers:4`, `batch_size:256`, `steps:100000`, `log_freq:1`, `save_checkpoint:true`, `save_freq:2000000` → **lower to 5000–20000** for intermediate ckpts, `wandb{enable,project:"franka_sim",disable_artifact}`, `dataset{repo_id:"lilkm/pick_cube_franka_panda_30", use_imagenet_stats:false}`, **`mixer:"online_offline"`**, **`online_ratio:0.5`** (RLPD 50/50 — **CONFIRMED** consumed by `OnlineOfflineMixer` in `data_sources/data_mixer.py`: `n_online=int(batch_size*online_ratio)`), `algorithm{...}`, `policy{...}`, `env{...}`.
- **`algorithm`** (`SACAlgorithmConfig`, `type:"sac"`): `actor_lr/critic_lr/temperature_lr:3e-4`, `discount:0.97`, `critic_target_update_weight:0.005`, `num_critics:2`, `critic_network_kwargs.hidden_dims:[256,256]`, `temperature_init:0.01`, **`utd_ratio:2`**, `policy_update_freq:1`, `grad_clip_norm:10.0`, `use_torch_compile:true`.
  - **CAVEAT (CONFIRMED):** these are **not** RLPD's state-based defaults (E=10 / Z=2 / G=20). LayerNorm on critics *is* hard-coded. Teach RLPD's E=10/G=20 as the **paper's** recipe; do **not** claim the LeRobot sim config uses them (it uses `num_critics:2`, `utd_ratio:2`).
- **`policy`** (`GaussianActorConfig`, `type:"gaussian_actor"`) — **buffer/loop/gRPC fields live HERE, not in `algorithm`**: `online_steps:1000000`, `online_buffer_capacity:100000`, `offline_buffer_capacity:100000`, `online_step_before_learning:100`, `storage_device:"cpu"` (→ `"cuda"` for more updates/s), `device:"cuda"`, `vision_encoder_name:"lerobot/resnet10"`, `freeze_vision_encoder:true`, `shared_encoder:true`, `num_discrete_actions:3`, **`actor_learner_config{learner_host:"127.0.0.1", learner_port:50051, policy_parameters_push_frequency:50}`** (CONFIRMED path — defined in `configuration_gaussian_actor.py`, NOT `configuration_sac.py`), `concurrency{actor:"threads", learner:"threads"}`. Input features: images front+wrist `[3,128,128]` + state `[18]`; output action `[3]`.
- **`env`**: `type:"gym_manipulator"`, `name:"gym_hil"`, `task:"PandaPickCubeGamepad-v0"` → **switch to `"PandaPickCubeBase-v0"` for hands-off headless**, `fps:10`, `processor`, `features`.

### Checkpoint format (verified EXACT — zero drift, one skeptic REFUTED the drift risk)
```
<output_dir>/checkpoints/<step>/pretrained_model/   # safetensors + train_config.json
<output_dir>/checkpoints/<step>/training_state/     # optimizer/step
<output_dir>/checkpoints/<step>/algorithm/          # algorithm state
<output_dir>/checkpoints/last                        # symlink (LAST_CHECKPOINT_LINK)
<output_dir>/dataset/                                # replay buffer as a LeRobot dataset
```
Constants (`constants.py`): `CHECKPOINTS_DIR="checkpoints"`, `PRETRAINED_MODEL_DIR="pretrained_model"`, `TRAINING_STATE_DIR="training_state"`, `ALGORITHM_DIR="algorithm"`, `LAST_CHECKPOINT_LINK="last"`. Resume: top-level `resume:true`.

## 3. gym-hil env facts + headless rendering

**Tasks** (`gym_hil/__init__.py`):
- `PandaPickCubeBase-v0` → raw `PandaPickCubeGymEnv`, `max_episode_steps=100`. **Steppable with programmatic `np.ndarray` actions — no input device. Use for autonomous RL + eval.**
- `PandaPickCube-v0` → autonomous wrapped env (4-D EE action, `gripper_penalty=-0.05`, no human).
- `PandaPickCubeGamepad-v0` / `Keyboard-v0` → add `InputsControlWrapper`+`PassiveViewerWrapper` → **need display + input device; fail headless.**

**Observation** (`_compute_observation`):
- `image_obs=False`: `{"agent_pos":Box(18,), "environment_state":Box(3,)}`, `agent_pos = concat(qpos7, qvel7, gripper1, tcp_pos3)` (**agent_dim=18 CONFIRMED; assert empirically — computed at runtime from `get_robot_state().shape[0]`**), env_state = cube xyz.
- `image_obs=True`: `{"pixels":{"front":Box(128,128,3 uint8), "wrist":...}, "agent_pos":Box(18,)}`. MuJoCo cams `"front"` and `"handcam_rgb"`(wrist).

**Action:**
- Base env: **7-D** `Box(-1,1)` `[dx,dy,dz,rx,ry,rz,grasp]`; xyz are **UNSCALED mocap-target deltas (up to 1 m/step!)** → pass small deltas or use a wrapped env; rx,ry,rz unused.
- Wrapped (`EEActionWrapper`): **4-D** `[dx,dy,dz,gripper]`, xyz × `DEFAULT_EE_STEP_SIZE={0.025,0.025,0.025}` m.
- (The LeRobot policy emits `[3]` continuous + `num_discrete_actions:3` gripper — see the configs.)

**Reward/success** (`reward_type` default `"sparse"`): sparse `=1.0 if (block_z - z_init) > 0.1 else 0.0`; dense `=0.3*exp(-20*dist)+0.7*clip(lift_progress,0,1)`. Success `= dist(cube,TCP)<0.05 AND lift>0.1`. Terminates on success or cube out of `SAMPLING_BOUNDS±0.05`. `control_dt=0.1`, `physics_dt=0.002` → 50 substeps → ~10 fps.

**Headless rendering (MUJOCO_GL) — gym-hil never sets it; set BEFORE first `import mujoco`/`gym_hil`:**
- **Modal (Linux GPU):** `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`; apt `libgl1 libegl1 libglew-dev libglfw3 libosmesa6 patchelf ffmpeg`.
- **Colab (T4):** write NVIDIA EGL ICD then `os.environ['MUJOCO_GL']='egl'`:
  ```python
  import os
  p='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
  os.makedirs(os.path.dirname(p), exist_ok=True)
  if not os.path.exists(p):
      open(p,'w').write('{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}')
  os.environ['MUJOCO_GL']='egl'
  ```
  apt `ffmpeg libgl1-mesa-glx libglfw3 libglew2.2 libegl1-mesa-dev libgles2-mesa-dev`. CPU fallback `MUJOCO_GL=osmesa`.
- `glfw` needs a display → fails headless everywhere.
- **macOS (UNCERTAIN — flag):** EGL/OSMesa are Linux-only, but MuJoCo ≥2.3.4 does offscreen via CGL without a display, so `rgb_array` *may* work locally on Mac. Do all guaranteed-headless work on Linux/Colab.

## 4. Modal app design

**Image:** `debian_slim(python="3.12")` + apt `libgl1 libegl1 libglew-dev libglfw3 libosmesa6 patchelf ffmpeg git` + env `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_ENABLE_HF_TRANSFER=1` + `git clone lerobot` + `pip install -e "lerobot[hilserl]"` + `hf_transfer wandb`. **Pin a lerobot commit** for reproducibility (also mitigates gRPC pickle-RCE — see risks).

**Actor+learner = ONE container, TWO subprocesses on `127.0.0.1`** (localhost gRPC = simplest, lowest-latency, no port exposure). `output_dir` under a `modal.Volume`, `save_checkpoint:true`, low `save_freq`, `vol.commit()` on a timer to survive preemption.

**GPU: a single L4 (~$0.80/hr) or A10/A10G (~$1.10/hr)** — SAC here is **not** compute-bound (batch 256, small nets, throughput gated by env-step + gRPC + ~10 fps). A100/H100 buy nothing for one run. (Modal live prices CONFIRMED: L4 $0.000222/s, A10 $0.000306/s, A100-40 $0.000583/s, H100 $0.001097/s; region ×1.5–1.75, non-preempt ×3.)

**Where parallelism actually helps (honest):** one HIL-SERL run does **NOT** throughput-scale across GPUs. Use Modal fan-out (`.map()`/`.spawn()`) for **(a) multi-seed robustness, (b) hyperparameter sweeps, (c) reward-classifier training, (d) batched eval**. Parallelize the sweep/seeds/eval, **not one run**.

**Optional massively-parallel PPO baseline (the ONE place a big GPU pays off):** `so101-nexus[warp]` → `gym.make_vec("WarpTouch-v1", num_envs=4096, device="cuda")` + CleanRL-style PPO. Different algorithm & sim (on-policy, no human) → present as a **contrast** (throughput-first sim RL) vs sample-efficient HIL-SERL, not a speedup of it.

## 5. SO-101 sim-to-real bridge

**Reusable byte-identical (the "learning brain"):** async actor-learner over gRPC, SAC learner (`algorithm.type="sac"`), `gaussian_actor` policy, ResNet10 reward-classifier arch, the human-intervention protocol, SAC hyperparameters.

**Changes only in env/robot config** (`HILSerlRobotEnvConfig`, `lerobot/envs/configs.py`): `env.robot type=so101_follower` (6× Feetech STS3215, USB port); `env.teleop type=so101_leader`; `env.processor.inverse_kinematics{urdf_path, target_frame_name, end_effector_bounds, end_effector_step_sizes}` (learns in **EE space**; run `lerobot-find-joint-limits` first — **tight workspace bounds = biggest speed/safety lever**); `image_preprocessing.crop_params_dict`+`resize_size=[128,128]`; `reset.fixed_reset_joint_positions`; `fps≈10`.

**Must be re-collected on the physical arm (NOT config):** ~15–25 fresh teleop demos (sim demos don't transfer) + a reward-classifier dataset of thousands of real labeled frames (`terminate_on_success=false` while collecting).

**Real gotcha beyond config:** robot-specific **EE-space IK layer** (`SO101FollowerEndEffector`, damped-least-squares IK) + a 6→18-dim proprioception expansion. Budget **~1–3 h hands-on** for a short grasp once bounds/demos/classifier are set. (ggando: 757 eps / ~70% / 5.9% intervention — an order-of-magnitude guide, one hobbyist run, not a benchmark. `so101-nexus` ships PPO/BC scaffolding, not the SAC HIL-SERL learner. indraneelpatil "sim-pretrain didn't help" = single setup, don't over-generalize.)

## 6. Open risks — smoke-test before trusting (from the verification pass)

> **✅ LOCAL SMOKE TEST PASSED (2026-07-16)** — `smoke/smoke_env.py` on gym-hil + mujoco 3.8.1 / gymnasium 1.3.0 (uv Python 3.12), **9 PASS / 1 WARN / 0 FAIL**. Empirically confirmed: env id `gym_hil/PandaPickCubeBase-v0`; `agent_pos==(18,)`, `environment_state==(3,)`; action = `Box(-1,1,(7,))`; `step()` returns float reward; `info['succeed']` is the success key; **pixel obs `front`/`wrist` == `(128,128,3)` render headless ON MACOS** (CGL — resolves risk #3); a **scripted reach-grasp-lift reached `reward=1.0`** (success machinery fires — resolves risk #2 and de-risks the demo). Remaining risks below are the ones NOT yet exercised locally (they need lerobot/Modal/Colab).


1. **Colab install** not verified by a real run — Python 3.12 conflict is REFUTED, but heavy transitive deps (placo/grpcio/mujoco/transformers pins) may need a runtime restart. Run the install cells end-to-end once before the workshop.
2. ~~**agent_dim=18**~~ **RESOLVED** — smoke test confirms `obs['agent_pos'].shape==(18,)`.
3. ~~**macOS headless UNCERTAIN**~~ **RESOLVED** — `image_obs=True` renders `(128,128,3)` front/wrist frames headless on this Mac (mujoco 3.8.1, CGL). Colab/Modal still use `MUJOCO_GL=egl`.
4. **No official PandaPickCube SAC checkpoint exists** → must train + publish our own. `aractingi/sac_gym_hil_pick_lift` is personal-namespace `pick_lift`, no model card, obs/action match unverified — don't depend on it.
5. **RLPD defaults E=10/Z=2/G=20 are the PAPER's, not the sim config's** (num_critics=2, utd_ratio=2). Smoke-test that shipped hyperparams actually learn in a reasonable step budget on L4/T4.
6. **Base 7-D action = UNSCALED mocap deltas (≤1 m/step)** → pass small deltas or use a 4-D wrapped EE env, else training thrashes.
7. **`mixer:"online_offline"`+`online_ratio:0.5`** semantics confirmed, but confirm `dataset.repo_id` actually loads/seeds the offline buffer — a missing/private repo silently degrades to online-only.
8. **gRPC pickle-deserialization RCE** (reported CVE-2026-25874, single-source — confirm id + patched commit). Keeping actor+learner on `127.0.0.1` in one Modal container avoids network exposure; pin a patched commit.
9. **Modal spot preemption** can lose minutes → verify `vol.commit()` timer + `push_to_hub` persist a resumable checkpoint (kill+resume once).
10. **Modal prices** subject to change — reconfirm at modal.com/pricing before quoting cost.

## Provenance
Grounding workflow `hilserl-build-grounding` run `wf_1cbe55ee-86d` (2026-07-16). Local env-fact validation: `smoke/` on a uv Python-3.12 venv. See `README.md` for run order; `RESOURCES.md` for the reading list.
