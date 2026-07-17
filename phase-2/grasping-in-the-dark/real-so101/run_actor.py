"""
Launcher for the stock LeRobot SAC actor (real-robot side), joint-space HIL-SERL
with LEADER-ARM interventions.

Registers two things lerobot's actor doesn't know about, then hands off to the stock
actor entry point. lerobot's own source stays untouched:

  1. keyboard + so101_leader_intervention teleop types (so the config parses).
  2. Two shims that make the off-stock JOINT-SPACE path work end to end:
     a) squeeze the policy action [1,6] -> [6] so the stored transition shape is
        consistent with leader actions (leaders return [6]).
     b) unnormalize the action [-1,1] -> joint degrees inside RobotEnv.step, i.e. at
        the robot boundary AFTER interventions are applied. The replay buffer therefore
        stays in the policy's native [-1,1] space (consistent offline+online), and the
        robot receives real joint targets. Without this, joint-space actions never get
        scaled (the stock EE pipeline is what normally scales them).

Usage (run AFTER the learner, in a second terminal):
    python hilserl_wipe/run_actor.py --config_path hilserl_wipe/train_config.json
"""

import lerobot.teleoperators.keyboard  # noqa: F401  # registers "keyboard"/"keyboard_ee"
import leader_intervention_teleop  # noqa: F401  # registers "so101_leader_intervention"
from leader_intervention_teleop import unnormalize_action

import atexit
import datetime
import os

import cv2
import numpy as np

import lerobot.rl.actor as actor_mod
from lerobot.rl.actor import actor_cli
from lerobot.rl.gym_manipulator import RobotEnv

# --- Shim (a): flatten policy action [1,6] -> [6] for consistent transition shape ---
_orig_step_fn = actor_mod.step_env_and_process_transition


def _step_env_and_process_transition(env, transition, action, *args, **kwargs):
    if hasattr(action, "ndim") and action.ndim > 1:
        action = action.squeeze(0)
    return _orig_step_fn(env, transition, action, *args, **kwargs)


actor_mod.step_env_and_process_transition = _step_env_and_process_transition

# --- Shim (b): unnormalize [-1,1] -> joint degrees at the robot boundary ---
_orig_env_step = RobotEnv.step


def _env_step(self, action):
    action = unnormalize_action(action)  # -> numpy array of 6 joint degrees
    return _orig_env_step(self, action)


RobotEnv.step = _env_step

# --- Front-cam recorder (fail-safe): saves the exact frames the actor sees to an mp4,
# capturing the whole session incl. interventions. No camera conflict (reuses the
# actor's frames). Wrapped so it can never break the run. ---
_REC_DIR = "/Users/rajatdandekar/Desktop/Robotics/hilserl_wipe/article_assets"
_rec = {"writer": None, "enabled": True, "path": None}
_orig_get_obs = RobotEnv._get_observation


def _release_writer():
    if _rec["writer"] is not None:
        _rec["writer"].release()
        print(f"[recorder] saved front-cam video -> {_rec['path']}")
        _rec["writer"] = None


atexit.register(_release_writer)


def _get_observation(self):
    obs = _orig_get_obs(self)
    if _rec["enabled"]:
        try:
            frame = obs.get("pixels", {}).get("front", None)
            if frame is not None:
                if hasattr(frame, "detach"):
                    frame = frame.detach().cpu().numpy()
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if _rec["writer"] is None:
                    os.makedirs(_REC_DIR, exist_ok=True)
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    _rec["path"] = os.path.join(_REC_DIR, f"intervention_session_{stamp}.mp4")
                    h, w = bgr.shape[:2]
                    _rec["writer"] = cv2.VideoWriter(
                        _rec["path"], cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h)
                    )
                    print(f"[recorder] recording front-cam -> {_rec['path']}")
                _rec["writer"].write(bgr)
        except Exception as e:
            print(f"[recorder] disabled ({e})")
            _rec["enabled"] = False
    return obs


RobotEnv._get_observation = _get_observation

if __name__ == "__main__":
    actor_cli()
