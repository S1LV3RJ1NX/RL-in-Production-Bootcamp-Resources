"""
Modal GPU training app for HIL-SERL (LeRobot gym-hil, Franka Panda PandaPickCube).

Runs the REAL SAC+RLPD training: the learner and actor as two subprocesses inside ONE
container talking over localhost gRPC (127.0.0.1:50051) — the simplest, lowest-latency,
never-network-exposed topology (see FACTS.md §4). Checkpoints stream to a modal.Volume
and (optionally) get pushed to the HF Hub. Training here is FULLY AUTONOMOUS — the MuJoCo
env supplies the reward, so no gamepad/keyboard/human is required.

Why one L4/A10 and not an H100: SAC here is not compute-bound (batch 256, small nets,
throughput gated by env-step + gRPC + ~10 fps). A big GPU buys nothing for one run.
Use Modal fan-out (.map) for seeds / sweeps / eval instead — NOT to speed up one run.

Prereqs:
    pip install modal && modal token new            # (already configured on this machine)
    modal secret create huggingface HF_TOKEN=hf_...  # optional: only for push_to_hub
    modal secret create wandb WANDB_API_KEY=...       # optional: online curves; else offline

Usage (two local entrypoints exist -> always name one with ::main / ::publish_main):
    # smoke test (few hundred steps, cheap, proves actor<->learner + checkpointing work):
    modal run modal/train_hilserl.py::main --config configs/train_gym_hil.json --steps 400 --job-name smoke --gpu L4
    # full run (detached -> survives your local session ending):
    modal run --detach modal/train_hilserl.py::main --config configs/train_gym_hil.json --save-freq 3000 --gpu L4
    # evaluate a checkpoint (success rate + video on the volume):
    modal run modal/train_hilserl.py::evaluate --job-name hilserl_panda_pickcube --n-episodes 20
    # publish to the HF Hub (needs a token):
    HF_TOKEN=hf_... modal run modal/train_hilserl.py::publish_main --hf-repo <user>/hilserl-panda-pickcube-sac
    # inspect the volume:
    modal run modal/train_hilserl.py::download --dest-run hilserl_panda_pickcube
"""
import json
import os

import modal

# Pin a lerobot commit for cohort reproducibility (also mitigates the reported gRPC
# pickle-deserialization issue — keep actor+learner on localhost, never expose the port).
LEROBOT_REF = os.environ.get("LEROBOT_REF", "main")  # TODO: pin to a specific SHA before the cohort
_HERE = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "ffmpeg", "patchelf",
        "libgl1", "libegl1", "libglew-dev", "libglfw3", "libosmesa6",
    )
    .env(
        {
            "MUJOCO_GL": "egl",           # headless GPU rendering on Linux
            "PYOPENGL_PLATFORM": "egl",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .run_commands(
        f"git clone --depth 1 --branch {LEROBOT_REF} https://github.com/huggingface/lerobot /root/lerobot"
        if LEROBOT_REF in ("main",)
        else "git clone https://github.com/huggingface/lerobot /root/lerobot "
        f"&& cd /root/lerobot && git checkout {LEROBOT_REF}",
        "pip install --no-cache-dir -e '/root/lerobot[hilserl]'",
    )
    .pip_install("hf_transfer", "wandb")
    # Patch LeRobot so RL replay-buffer serialization doesn't crash on float[0,255] images
    # (see modal/patch_image_writer.py). Editable install -> patching the source file works.
    .add_local_file(os.path.join(_HERE, "patch_image_writer.py"), "/root/patch_image_writer.py", copy=True)
    .run_commands("python /root/patch_image_writer.py")
)

app = modal.App("hilserl-grasp")
vol = modal.Volume.from_name("hilserl-grasp-ckpts", create_if_missing=True)

# NOTE on tokens: we deliberately do NOT attach named modal.Secrets to the function —
# Modal resolves them lazily and a missing 'huggingface'/'wandb' secret would break every
# run (even ones that don't push). Training-to-volume needs no token; the OPTIONAL HF-push
# and W&B keys are passed as function args from the local env (see main()), which travel
# over Modal's encrypted channel. To bake them instead, add secrets=[Secret.from_name(...)].


@app.function(image=image, gpu="L4", memory=32768, timeout=6 * 60 * 60, volumes={"/ckpts": vol})
def train(config: dict, steps: int | None = None, save_freq: int = 10_000, hf_repo: str | None = None,
          gpu: str = "L4", job_name: str = "hilserl_panda_pickcube",
          hf_token: str | None = None, wandb_key: str | None = None, resume: bool = False):
    """Run SAC+RLPD training (learner + actor) to completion. Returns the checkpoint path."""
    import shutil
    import subprocess
    import sys
    import threading
    import time

    # Optional tokens passed from the local env (not baked as secrets).
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    run_dir = f"/ckpts/{job_name}"
    actor_dir = f"/ckpts/{job_name}__actor"  # actor doesn't checkpoint; give it its OWN output_dir
    aux = f"/ckpts/{job_name}__aux"          # logs, wandb, used-config live OUTSIDE output_dir
    # LeRobot's cfg.validate() (run by BOTH learner and actor) refuses to start if output_dir
    # already exists and resume is False. Keep each pristine: let LeRobot create them, and for a
    # fresh run clear any leftover from a prior attempt.
    if not resume and os.path.exists(run_dir):
        print(f"[train] fresh run — removing existing {run_dir}")
        shutil.rmtree(run_dir)
    if os.path.exists(actor_dir):   # actor is stateless (gets policy over gRPC) -> always fresh
        shutil.rmtree(actor_dir)
    os.makedirs(aux, exist_ok=True)

    # ---- finalize the config for a headless autonomous Modal run --------------------
    cfg = dict(config)
    cfg["output_dir"] = run_dir
    cfg["job_name"] = job_name
    cfg["resume"] = resume
    cfg["save_checkpoint"] = True
    cfg["save_freq"] = int(save_freq)                 # low -> intermediate checkpoints
    if steps is not None:
        cfg["steps"] = int(steps)
    # W&B: online only if a key is present, else offline (curves still land on the volume).
    wandb_online = bool(os.environ.get("WANDB_API_KEY"))
    cfg.setdefault("wandb", {})
    if wandb_online:
        cfg["wandb"]["enable"] = True
        cfg["wandb"]["project"] = cfg["wandb"].get("project", "hilserl_grasp")
        os.environ["WANDB_DIR"] = aux
    else:
        # No W&B key -> disable it entirely (wandb 0.28 still tries to auth even in "offline").
        # Training metrics still stream to console -> {aux}/train.log, which we parse for the curve.
        cfg["wandb"]["enable"] = False
    # HF push (optional): only if a repo + token are both present.
    hf_ok = bool(hf_repo) and bool(os.environ.get("HF_TOKEN"))
    if "policy" in cfg:
        cfg["policy"]["push_to_hub"] = False   # pushed by our own uploader thread instead (below)

    cfg_path = f"{aux}/train_config.used.json"        # learner config; also read by eval
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    # Actor gets an identical config EXCEPT its own output_dir (avoids the shared-dir validate clash).
    actor_cfg = dict(cfg)
    actor_cfg["output_dir"] = actor_dir
    actor_cfg["resume"] = False   # actor never resumes from disk; keep its dir fresh
    cfg_path_actor = f"{aux}/train_config.actor.json"
    with open(cfg_path_actor, "w") as f:
        json.dump(actor_cfg, f, indent=2)
    print(f"[train] wrote configs -> {cfg_path} (+ actor)\n[train] steps={cfg['steps']} save_freq={cfg['save_freq']} "
          f"wandb={'online' if wandb_online else 'offline'} hf_push={hf_ok} gpu={gpu} resume={resume}")

    env = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    logf = open(f"{aux}/train.log", "a", buffering=1)

    def spawn(mod, path):
        p = subprocess.Popen(
            [sys.executable, "-u", "-m", mod, "--config_path", path],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        def pump():
            for line in p.stdout:
                logf.write(line)
                print(f"[{mod.split('.')[-1]}] {line}", end="")
        threading.Thread(target=pump, daemon=True).start()
        return p

    # Learner opens the gRPC server FIRST; give it a moment, then start the actor.
    learner = spawn("lerobot.rl.learner", cfg_path)
    time.sleep(20)
    actor = spawn("lerobot.rl.actor", cfg_path_actor)

    # Commit the volume periodically so a spot-preemption loses at most ~2 min.
    stop = threading.Event()
    def committer():
        while not stop.is_set():
            time.sleep(120)
            try:
                vol.commit()
            except Exception as e:
                print(f"[commit] {e}")
    threading.Thread(target=committer, daemon=True).start()

    # Push each NEW checkpoint to the HF Hub as it's saved — this runs on Modal, so a good
    # policy externalizes to the Hub the instant it exists, surviving any preemption/teardown.
    def uploader():
        if not (hf_repo and hf_token):
            return
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        try:
            api.create_repo(hf_repo, exist_ok=True, repo_type="model")
        except Exception as e:
            print("[upload] create_repo:", e)
        pushed = set()
        while not stop.is_set():
            time.sleep(30)
            ckroot = f"{run_dir}/checkpoints"
            dirs = sorted(d for d in os.listdir(ckroot)) if os.path.isdir(ckroot) else []
            dirs = [d for d in dirs if d.isdigit()]
            if not dirs or dirs[-1] in pushed:
                continue
            latest = dirs[-1]
            pm = f"{ckroot}/{latest}/pretrained_model"
            if not os.path.isdir(pm):
                continue
            try:
                with open("/tmp/README.md", "w") as f:
                    f.write(MODEL_CARD.format(repo=hf_repo))
                api.upload_folder(folder_path=pm, repo_id=hf_repo, repo_type="model")
                api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md", repo_id=hf_repo, repo_type="model")
                if os.path.exists(f"{aux}/train.log"):
                    api.upload_file(path_or_fileobj=f"{aux}/train.log", path_in_repo="train.log", repo_id=hf_repo, repo_type="model")
                pushed.add(latest)
                print(f"[upload] pushed checkpoint {latest} -> https://huggingface.co/{hf_repo}")
            except Exception as e:
                print("[upload] push failed:", e)
    threading.Thread(target=uploader, daemon=True).start()

    # Wait for the learner (it owns training length); then stop the actor.
    rc = learner.wait()
    print(f"[train] learner exited rc={rc}")
    for p in (actor,):
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()
    stop.set()
    vol.commit()

    last = f"{run_dir}/checkpoints/last"
    exists = os.path.exists(last)
    print(f"[train] done. checkpoint 'last' present: {exists} -> {os.path.realpath(last) if exists else 'MISSING'}")
    return {"run_dir": run_dir, "checkpoint_last": last, "ok": exists and rc == 0}


@app.function(image=image, volumes={"/ckpts": vol})
def download(dest_run: str = "hilserl_panda_pickcube") -> dict:
    """List what's on the volume for a run (call via `modal run ... ::download`)."""
    vol.reload()
    run_dir = f"/ckpts/{dest_run}"
    out = {"run_dir": run_dir, "exists": os.path.exists(run_dir), "checkpoints": []}
    ckdir = f"{run_dir}/checkpoints"
    if os.path.isdir(ckdir):
        out["checkpoints"] = sorted(os.listdir(ckdir))
    aux = f"/ckpts/{dest_run}__aux"
    if os.path.isdir(aux):
        out["aux"] = sorted(os.listdir(aux))
    print(json.dumps(out, indent=2))
    return out


@app.function(image=image)
def show_image_writer_src() -> str:
    """Print the exact installed source of the image writer's float-check, so we patch precisely."""
    import inspect
    from lerobot.datasets import image_writer as iw
    print("FILE:", iw.__file__)
    print("=====SRC=====")
    src = inspect.getsource(iw.image_array_to_pil_image)
    print(src)
    return src


@app.function(image=image, gpu="L4", timeout=60 * 60, volumes={"/ckpts": vol})
def evaluate(job_name: str = "smoke", n_episodes: int = 20, ckpt: str = "last") -> dict:
    """Load a trained checkpoint, roll it out AUTONOMOUSLY, and save success-rate + video.

    Mirrors the actor's processor pipeline (the shipped eval_policy.py is buggy for gym-hil).
    Self-diagnosing: on any LeRobot-API mismatch it prints the real signatures so the exact
    call can be fixed in a single iteration.
    """
    import inspect
    import numpy as np
    import torch
    import imageio

    os.environ.setdefault("MUJOCO_GL", "egl")
    vol.reload()
    aux = f"/ckpts/{job_name}__aux"
    cfg_json = f"{aux}/train_config.used.json"
    ckpt_dir = f"/ckpts/{job_name}/checkpoints/{ckpt}/pretrained_model"
    assert os.path.exists(cfg_json), f"missing config {cfg_json}"
    assert os.path.exists(ckpt_dir), f"missing checkpoint {ckpt_dir}"

    # --- load the training config object (same JSON lerobot trained from) ---------
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig
    import draccus
    cfg, errs = None, []
    for name, loader in (
        ("draccus.parse", lambda: draccus.parse(TrainRLServerPipelineConfig, args=["--config_path", cfg_json])),
        ("draccus.load", lambda: draccus.load(TrainRLServerPipelineConfig, open(cfg_json))),
    ):
        try:
            cfg = loader()
            print(f"[eval] loaded config via {name}")
            break
        except Exception as e:
            errs.append(f"{name}: {type(e).__name__}: {e}")
    if cfg is None:
        raise RuntimeError("could not load train config:\n" + "\n".join(errs))

    device = getattr(cfg.policy, "device", "cuda")
    cfg.env.task = "PandaPickCube-v0"          # autonomous, headless

    from lerobot.rl import gym_manipulator as gm
    from lerobot.processor import TransitionKey
    from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy

    def _diag(*fns):
        for f in fns:
            try:
                print(f"[eval][sig] {f.__module__}.{f.__name__}{inspect.signature(f)}")
            except Exception as e:
                print(f"[eval][sig] {getattr(f,'__name__',f)}: {e}")

    try:
        env, teleop = gm.make_robot_env(cfg.env)
        env_processor, action_processor = gm.make_processors(env, teleop, cfg.env, device)
        policy = GaussianActorPolicy.from_pretrained(ckpt_dir).to(device).eval()
        input_keys = list(cfg.policy.input_features.keys())
        print(f"[eval] input_features={input_keys} device={device}")
    except Exception:
        print("[eval] SETUP FAILED — real signatures:")
        _diag(gm.make_robot_env, gm.make_processors,
              gm.reset_and_build_transition, gm.step_env_and_process_transition,
              GaussianActorPolicy.from_pretrained)
        raise

    def render_front(e):
        r = e.render()
        r = r[0] if isinstance(r, (list, tuple)) else r
        return np.asarray(r).astype(np.uint8)

    successes, frames = [], []
    for ep in range(n_episodes):
        try:
            transition = gm.reset_and_build_transition(env, env_processor, action_processor)
            ep_succ = 0.0
            while True:
                obs = {k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in input_keys}
                with torch.no_grad():
                    action = policy.select_action(batch=obs)
                transition = gm.step_env_and_process_transition(
                    env=env, transition=transition, action=action,
                    env_processor=env_processor, action_processor=action_processor,
                )
                if ep == 0:
                    try:
                        frames.append(render_front(env))
                    except Exception:
                        pass
                reward = float(transition[TransitionKey.REWARD])
                if bool(transition[TransitionKey.DONE]) or bool(transition[TransitionKey.TRUNCATED]):
                    ep_succ = 1.0 if reward > 0.0 else 0.0
                    break
        except Exception:
            print("[eval] ROLLOUT FAILED — real signatures:")
            _diag(gm.reset_and_build_transition, gm.step_env_and_process_transition, policy.select_action)
            raise
        successes.append(ep_succ)
        print(f"[eval] ep {ep}: success={ep_succ}")

    sr = float(np.mean(successes)) if successes else 0.0
    metrics = {"job_name": job_name, "ckpt": ckpt, "n_episodes": n_episodes, "success_rate": sr}
    with open(f"{aux}/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if frames:
        imageio.mimsave(f"{aux}/eval.mp4", frames, fps=20, quality=8)
        metrics["video"] = f"{aux}/eval.mp4"
    vol.commit()
    print("[eval] RESULT:", json.dumps(metrics, indent=2))
    return metrics


@app.function(image=image, gpu="L4", timeout=30 * 60, volumes={"/ckpts": vol})
def render_demo(job_name: str = "hilserl_panda_pickcube", ckpt: str = "last", n_try: int = 25) -> dict:
    """Render before.mp4/gif (random policy, flailing) + after.mp4/gif (a successful TRAINED-policy
    grasp) to the run's __aux dir. Front|wrist side-by-side."""
    import numpy as np
    import torch
    import imageio
    import draccus
    import gymnasium as gym
    import gym_hil  # noqa: F401

    os.environ.setdefault("MUJOCO_GL", "egl")
    vol.reload()
    aux = f"/ckpts/{job_name}__aux"
    ckpt_dir = f"/ckpts/{job_name}/checkpoints/{ckpt}/pretrained_model"

    def panel(px):
        f = np.asarray(px["front"]).astype(np.uint8)
        w = np.asarray(px["wrist"]).astype(np.uint8)
        return np.concatenate([f, w], axis=1)

    # ---- BEFORE: a raw env with small random actions (untrained behaviour) ----
    raw = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
    o, _ = raw.reset(seed=3)
    before = []
    for _ in range(90):
        before.append(panel(o["pixels"]))
        a = raw.action_space.sample().astype(np.float32) * 0.35
        o, r, term, trunc, info = raw.step(a)
        if term or trunc:
            before.append(panel(o["pixels"])); break
    raw.close()

    # ---- AFTER: the trained policy, first successful episode ----
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig
    from lerobot.rl import gym_manipulator as gm
    from lerobot.processor import TransitionKey
    from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
    cfg = draccus.parse(TrainRLServerPipelineConfig, args=["--config_path", f"{aux}/train_config.used.json"])
    cfg.env.task = "PandaPickCube-v0"
    device = getattr(cfg.policy, "device", "cuda")
    env, teleop = gm.make_robot_env(cfg.env)
    env_p, act_p = gm.make_processors(env, teleop, cfg.env, device)
    policy = GaussianActorPolicy.from_pretrained(ckpt_dir).to(device).eval()
    keys = list(cfg.policy.input_features.keys())

    def frame(e):
        r = e.render()
        if isinstance(r, (list, tuple)):
            return np.concatenate([np.asarray(x).astype(np.uint8) for x in r], axis=1)
        return np.asarray(r).astype(np.uint8)

    after, got = None, False
    for attempt in range(n_try):
        tr = gm.reset_and_build_transition(env, env_p, act_p)
        frames, succ_at = [], -1
        for t in range(env.spec.max_episode_steps or 120):
            frames.append(frame(env))
            obs = {k: v for k, v in tr[TransitionKey.OBSERVATION].items() if k in keys}
            with torch.no_grad():
                action = policy.select_action(batch=obs)
            tr = gm.step_env_and_process_transition(env=env, transition=tr, action=action,
                                                    env_processor=env_p, action_processor=act_p)
            r = float(tr[TransitionKey.REWARD])
            if succ_at < 0 and r > 0:
                succ_at = t
            done = bool(tr[TransitionKey.DONE]) or bool(tr[TransitionKey.TRUNCATED])
            # end the clip a few frames AFTER the lift, so it closes on the held cube
            if (succ_at >= 0 and t >= succ_at + 10) or done:
                frames.append(frame(env)); break
        print(f"[render] after attempt {attempt}: success={succ_at >= 0} (step {succ_at}), frames={len(frames)}", flush=True)
        if succ_at >= 0:
            after, got = frames, True; break
        if after is None or len(frames) > len(after):
            after = frames
    print(f"[render] before={len(before)} frames, after={len(after)} frames, after_success={got}", flush=True)

    imageio.mimsave(f"{aux}/before.mp4", before, fps=15, quality=8)
    imageio.mimsave(f"{aux}/after.mp4", after, fps=15, quality=8)
    imageio.mimsave(f"{aux}/before.gif", before[::2], fps=8)
    imageio.mimsave(f"{aux}/after.gif", after[::2], fps=8)
    vol.commit()
    return {"before_frames": len(before), "after_frames": len(after), "after_success": got}


MODEL_CARD = """---
library_name: lerobot
tags: [reinforcement-learning, robotics, hil-serl, sac, rlpd, gym-hil, manipulation, vizuara]
license: apache-2.0
pipeline_tag: reinforcement-learning
---

# HIL-SERL (SAC + RLPD) — Franka Panda PickCube

A vision-based manipulation policy trained with **HIL-SERL** (SAC + RLPD) in the LeRobot
`gym-hil` simulation (`PandaPickCube`), fully autonomously on a single GPU — the MuJoCo
environment supplies the reward, so no human intervention or reward classifier is needed in sim.

Built for **Session 3 (Robotics)** of the Vizuara *RL in Production* workshop — the same stack
trains a real **SO-101** arm from scratch.

- **Observation:** front + wrist cameras (128×128) + 18-D proprioceptive state
- **Action:** 3-D end-effector delta + discrete gripper
- **Algorithm:** SAC, RLPD 50/50 online/offline mixing, LayerNorm critics
- **Demos (offline seed):** [`lilkm/pick_cube_franka_panda_30`](https://huggingface.co/datasets/lilkm/pick_cube_franka_panda_30)

## Load & evaluate

```python
import torch, draccus
from huggingface_hub import snapshot_download
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.rl import gym_manipulator as gm
from lerobot.processor import TransitionKey
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy

local = snapshot_download("{repo}")
cfg = draccus.parse(TrainRLServerPipelineConfig, args=["--config_path", f"{{local}}/train_config.json"])
cfg.env.task = "PandaPickCube-v0"
policy = GaussianActorPolicy.from_pretrained(local).to("cuda").eval()
env, teleop = gm.make_robot_env(cfg.env)
env_p, act_p = gm.make_processors(env, teleop, cfg.env, "cuda")
# roll out with policy.select_action(...) — see the workshop notebook.
```

**Reading:** HIL-SERL (arXiv:2410.21845) · RLPD (arXiv:2302.02948) · [LeRobot HIL-SERL docs](https://huggingface.co/docs/lerobot/hilserl_sim)
"""


@app.function(image=image, volumes={"/ckpts": vol})
def publish(hf_repo: str, hf_token: str, job_name: str = "hilserl_panda_pickcube", ckpt: str = "last") -> str:
    """Push the trained policy (+ config + train.log + model card) to the HF Hub."""
    from huggingface_hub import HfApi
    vol.reload()
    ckpt_dir = f"/ckpts/{job_name}/checkpoints/{ckpt}/pretrained_model"
    log = f"/ckpts/{job_name}__aux/train.log"
    assert os.path.exists(ckpt_dir), f"missing checkpoint {ckpt_dir}"
    with open("/tmp/README.md", "w") as f:
        f.write(MODEL_CARD.format(repo=hf_repo))
    api = HfApi(token=hf_token)
    api.create_repo(hf_repo, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=ckpt_dir, repo_id=hf_repo, repo_type="model")
    api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md", repo_id=hf_repo, repo_type="model")
    if os.path.exists(log):
        api.upload_file(path_or_fileobj=log, path_in_repo="train.log", repo_id=hf_repo, repo_type="model")
    url = f"https://huggingface.co/{hf_repo}"
    print("[publish] done ->", url)
    return url


@app.local_entrypoint()
def publish_main(hf_repo: str, job_name: str = "hilserl_panda_pickcube", ckpt: str = "last"):
    """Publish a trained checkpoint to the Hub. Set HF_TOKEN in your local env first.
       Usage: HF_TOKEN=hf_... modal run modal/train_hilserl.py::publish_main --hf-repo <user>/<name>"""
    tok = os.environ.get("HF_TOKEN")
    assert tok, "set HF_TOKEN in your local env (export HF_TOKEN=hf_...) before publishing"
    print("RESULT:", publish.remote(hf_repo, tok, job_name, ckpt))


@app.local_entrypoint()
def main(config: str, steps: int = 0, save_freq: int = 10_000, hf_repo: str = "", gpu: str = "L4",
         job_name: str = "hilserl_panda_pickcube", resume: bool = False):
    """Launch training. `config` is a local path to a train_config.json."""
    with open(config) as f:
        cfg = json.load(f)
    fn = train if gpu == "L4" else train.with_options(gpu=gpu)  # GPU is fixed at decoration; override here
    res = fn.remote(
        cfg,
        steps=(steps or None),
        save_freq=save_freq,
        hf_repo=(hf_repo or None),
        gpu=gpu,
        job_name=job_name,
        resume=resume,
        hf_token=(os.environ.get("HF_TOKEN") or None),      # optional: enables push_to_hub
        wandb_key=(os.environ.get("WANDB_API_KEY") or None),  # optional: online curves
    )
    print("RESULT:", json.dumps(res, indent=2))
