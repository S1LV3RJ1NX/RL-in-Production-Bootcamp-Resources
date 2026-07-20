# Challenges & Fixes: HIL-SERL on a Real SO-101 (the full autopsy)

Everything that went wrong and how we fixed it, in the order we hit it. Every fix keeps
LeRobot's source untouched — adaptations live in launcher shims (`run_actor.py`,
`run_learner.py`), one teleop subclass (`leader_intervention_teleop.py`), and data-prep
scripts. Code references are to LeRobot **v0.5.1**.

---

## 0. Environment: Apple Silicon, no CUDA

- `torch.cuda.is_available()` → `False`; only **MPS**. LeRobot's SAC learner assumes CUDA.
- MPS runs the networks (8M-param policy, ResNet-10 encoder) but slowly: actor inference ~8 fps vs. 10 fps target.
- **Takeaway:** fine for a proof-of-concept loop; not a real training platform. Expect no same-day convergence.

---

## 1. `placo` won't import on macOS (urdfdom version mismatch)

The `hilserl` extra installs `placo` (IK solver) via `cmeel`, but the prebuilt `placo.so`
links `liburdfdom_*.4.0.dylib` while cmeel ships `*.6.0.0`:

```
ImportError: dlopen(... placo.so): Library not loaded: @rpath/liburdfdom_sensor.4.0.dylib
```

**Fix** — symlink the 6.x libs to the 4.0 names cmeel's `placo.so` expects:

```bash
cd .venv/lib/python3.12/site-packages/cmeel.prefix/lib
for lib in model world sensor; do
  ln -sf liburdfdom_${lib}.6.0.0.dylib liburdfdom_${lib}.4.0.dylib
done
```

After this, `import placo` and `lerobot.model.kinematics.RobotKinematics` work. (Only needed
for end-effector space; the joint-space path we ended up using doesn't call placo.)

Also: `pip` isn't on PATH in the uv-managed venv — use `python -m pip`.

---

## 2. The leader arm is NOT a supported HIL-SERL controller

**Symptom:** with `teleop.type=so101_leader`, `gym_manipulator` crashes at startup.

**Root causes (structural, not config):**
- `AddTeleopEventsAsInfoStep` calls `_check_teleop_with_events(teleop)`, which requires
  `get_teleop_events()`. Only `GamepadTeleop` and `KeyboardEndEffectorTeleop` implement it.
  `SOLeader` does **not** → `TypeError` at startup.
- `InterventionActionProcessorStep` extracts the human action as `delta_x/delta_y/delta_z`
  (end-effector deltas). The leader outputs **joint positions**. Mismatch.
- `processor.control_mode` (the `"leader"` value the docs mention) **is never read** in
  `gym_manipulator` — grep confirms the only `control_mode` usages are for the LIBERO sim.
- Verified identical on `main`: the actor just calls `make_teleoperator_from_config(cfg.teleop)`
  with no leader/keyboard event wiring.

**Conclusion:** stock HIL-SERL = gamepad or keyboard only. The docs are ahead of the code.

**Options:** (a) buy a gamepad [simplest, keeps you on the stock EE path]; (b) keyboard
[works, but driving a 3D arm by keyboard through a pick-and-place is miserable]; (c) build a
custom leader teleop [what we did].

---

## 3. Joint-space bites back: batched action shape

To use the leader (which speaks joints), we ran in **joint space** — no `inverse_kinematics`
in the config, so the IK action-processor steps are skipped. First crash:

```
IndexError: index 1 is out of bounds for dimension 0 with size 1
  joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(...motors)}
```

`policy.select_action` returns a **batched** `[1, 6]` tensor. In EE space,
`MapTensorToDeltaActionDictStep` squeezes it (`if action.dim() > 1: action.squeeze(0)`).
In joint space nothing does, so `env.step` indexes the batch dim.

**Fix** (`run_actor.py`): wrap `step_env_and_process_transition` to squeeze the policy action
to `[6]` before it enters the pipeline — so the stored transition shape is also consistent
with leader actions (which are `[6]`).

---

## 4. The arm drifts to zero and freezes: no action normalization

**Symptom:** the policy runs, but the arm creeps to one pose and stops. Logs show goal
positions like `-0.46`, `0.63` — i.e. values in `[-1, 1]`, not joint degrees.

**Root cause:** the SAC **actor never applies action (un)normalization.**
`select_action` returns the raw tanh-squashed output in `[-1, 1]`
(`TanhMultivariateNormalDiag` is built with no rescale). Normalization is supposed to happen
in a separate processor (`make_sac_pre_post_processors` with `dataset_stats`) — but the actor
loop never builds or applies it. In **EE space** this is invisible: the `[-1,1]` output *is*
the delta, scaled by a tiny `end_effector_step_size` downstream. In **joint space**, that
`[-1,1]` is sent to the motors as the joint target → the arm barely moves and settles near 0.

**Fix — one action convention end to end, unnormalize only at the robot:**
- Offline demos, policy output, and leader interventions all live in `[-1, 1]`.
- `run_actor.py` monkeypatches `RobotEnv.step` to unnormalize `[-1,1] → joint degrees`
  (per-joint affine using the dataset's action min/max) *at the robot boundary only*.
- Because the unnormalize happens **after** the intervention step, both policy actions and
  leader actions (also `[-1,1]`) get mapped identically — the replay buffer stays consistent.

```
buffer / policy / leader / offline   ── [-1,1] ──┐
                                                  ▼  (RobotEnv.step)
                                            unnormalize → joint degrees → motors
```

Constants (`ACTION_MIN/MAX`) live in `leader_intervention_teleop.py` and are reused by the
actor shim and the offline-normalization script, so all three action sources agree.

---

## 5. The custom leader-intervention teleop

`leader_intervention_teleop.py` — a subclass of `SOLeader` registered as
`so101_leader_intervention`:

- **`get_action()`** reads the leader's 6 joints and returns them **normalized to `[-1,1]`
  as a NumPy array.** This is the key trick: `InterventionActionProcessorStep` has an
  `isinstance(teleop_action, np.ndarray)` branch that uses the array *directly* as the action
  — bypassing the EE-delta assumption entirely.
- **`get_teleop_events()`** via a `pynput` listener: **SPACE** toggles intervention,
  **`s`** = success, **`r`** = redo, **`q`** = fail.
- **`name = "so_leader"`** (not the registry type) so it reuses the existing calibration at
  `calibration/teleoperators/so_leader/<id>.json` instead of prompting for a fresh one.

Both `run_actor.py` and `run_learner.py` import this module so the shared config parses
(the learner imports only `gamepad`/`so_leader` by default and can't otherwise resolve the
custom type).

---

## 6. Offline demos need a reward column

**Symptom:**
```
KeyError: 'next.reward'   (ReplayBuffer._lerobotdataset_to_transitions)
```

`lerobot-record` datasets have no reward. The SAC offline buffer *infers* `next.done` from
episode boundaries but **requires** `next.reward`.

**Fix** (`add_reward.py`): add a sparse success reward — `1.0` on each episode's last frame
(the demos all succeed), `0.0` elsewhere — and register `next.reward` in `meta/info.json`.
This matches the online reward the human gives by pressing `s`.

Then `normalize_offline_actions.py` converts the offline `action` column from degrees to
`[-1,1]` so the offline buffer matches the online action space (see #4).

---

## 7. Assorted smaller gotchas

- **`output_dir` collision:** learner and actor share the config; if `output_dir` is a fixed
  path, whichever starts second hits `FileExistsError`. Set `output_dir: null` → each process
  auto-timestamps its own folder. They communicate via gRPC, not the filesystem.
- **`policy.push_to_hub` requires `policy.repo_id`** or `validate()` raises.
- **USB flakiness:** the wrist camera rides on the moving follower arm; its cable flexes and
  drops frames (`OpenCVCamera(1) read failed`), same failure mode as the servo cables. Reseat,
  plug directly into the Mac, or drop to a single fixed camera.
- **macOS keyboard permissions:** `pynput` needs **Accessibility** + **Input Monitoring**
  granted to the terminal, or intervention keys are silently ignored
  (`This process is not trusted!`).
- **Camera access from a non-GUI shell** is blocked by macOS; test cameras from your normal
  terminal, not a background process.

---

## What we'd do differently

| Lever | Why |
|---|---|
| **Buy a gamepad (~$15)** | Unlocks stock HIL-SERL end to end, on the EE path, zero custom code. |
| **Use end-effector space** | What the pipeline is tuned for; faster learning; EE bounds give real safety. |
| **Run on CUDA** | MPS is a PoC, not a training platform for SAC here. |
| **20-30 demos, varied** | 10 is thin; vary duster/marker positions. |
| **Train a reward classifier** | Removes the human-finger-on-`s` dependency. |

The custom leader path taught us the internals, but if the goal is a *working policy* rather
than *understanding*, the gamepad + EE-space + CUDA route is dramatically less work.
