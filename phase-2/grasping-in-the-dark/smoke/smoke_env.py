"""
Env-fact smoke test for gym-hil (PandaPickCube).

Validates the load-bearing facts the whole build depends on, empirically, on THIS
machine's gym-hil/mujoco versions — BEFORE any policy/notebook/deck code assumes them:

  [1] the registered task ids (Base / autonomous / Gamepad / Keyboard)
  [2] observation space  : agent_pos == (18,), environment_state == (3,)
  [3] action space       : base env is 7-D Box(-1, 1)
  [4] reward path        : step() returns a float reward + a success flag in info
  [5] termination        : episode ends on success or out-of-bounds
  [6] pixel rendering     : image_obs=True yields front/wrist (128,128,3) uint8
  [7] a scripted reach-grasp-lift can drive reward -> 1.0 (WARN if it can't; the
      point is only to prove the success/reward machinery fires, not to be a policy)

Prints a PASS/WARN/FAIL line per check and exits non-zero if any hard check FAILs.
Run:  .venv/bin/python smoke/smoke_env.py
"""
import os
import sys
import traceback

import numpy as np

# On macOS, MuJoCo >=2.3.4 does offscreen rendering via CGL without a display; do not
# force egl/osmesa here (those are Linux-only). On Linux/Colab set MUJOCO_GL=egl BEFORE
# importing mujoco. Respect an externally-set value if present.
_GL = os.environ.get("MUJOCO_GL", "(default)")

import gymnasium as gym
import mujoco
import gym_hil  # noqa: F401  (registers the gym_hil/* env ids)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []


def record(check, status, detail=""):
    results.append((check, status, detail))
    tag = {"PASS": "\033[92mPASS\033[0m", "WARN": "\033[93mWARN\033[0m", "FAIL": "\033[91mFAIL\033[0m"}[status]
    print(f"  [{tag}] {check}" + (f" — {detail}" if detail else ""))


print(f"mujoco {mujoco.__version__} | gymnasium {gym.__version__} | MUJOCO_GL={_GL}")
print("-" * 72)

# ---------------------------------------------------------------- [1] task ids
panda_ids = sorted(k for k in gym.registry.keys() if "Panda" in k)
print("Registered Panda env ids:")
for k in panda_ids:
    print(f"    {k}")
base_id = next((k for k in panda_ids if "PandaPickCubeBase" in k), None)
if base_id:
    record("[1] PandaPickCubeBase-v0 is registered", PASS, base_id)
else:
    record("[1] PandaPickCubeBase-v0 is registered", FAIL, f"only found {panda_ids}")
    print("\nCannot continue without the Base env id.")
    sys.exit(1)


def make(image_obs):
    """Construct the raw Base env, tolerating constructor-signature differences across versions."""
    for kwargs in (
        {"image_obs": image_obs, "render_mode": "rgb_array"},
        {"image_obs": image_obs},
        {"render_mode": "rgb_array"},
        {},
    ):
        try:
            return gym.make(base_id, **kwargs), kwargs
        except TypeError:
            continue
    return gym.make(base_id), {}


# ---------------------------------------------------- [2][3] state obs + action
try:
    env, used = make(image_obs=False)
    obs, info = env.reset(seed=0)
    print(f"\nBase env constructed with kwargs={used}")
    print(f"reset() obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    print(f"reset() info keys: {list(info.keys())}")

    ap = np.asarray(obs["agent_pos"]).reshape(-1)
    record("[2] obs['agent_pos'] == (18,)", PASS if ap.shape == (18,) else FAIL, str(ap.shape))
    es = np.asarray(obs["environment_state"]).reshape(-1)
    record("[2] obs['environment_state'] == (3,)", PASS if es.shape == (3,) else FAIL, str(es.shape))

    a_space = env.action_space
    is7 = getattr(a_space, "shape", None) == (7,)
    record("[3] action_space is 7-D Box", PASS if is7 else WARN, f"{a_space}")
except Exception:
    record("[2/3] state obs + action space", FAIL, "exception (see trace)")
    traceback.print_exc()
    env = None

# --------------------------------------------------------- [4][5] reward + term
if env is not None:
    try:
        obs, info = env.reset(seed=0)
        a = env.action_space
        r_sample = None
        term_seen = False
        succ_key = next((k for k in ("succeed", "is_success", "success") if k in info), None)
        for _ in range(30):
            act = a.sample() * 0.0  # no-op-ish; we only test the plumbing here
            obs, rew, terminated, truncated, info = env.step(act)
            r_sample = rew
            succ_key = succ_key or next((k for k in ("succeed", "is_success", "success") if k in info), None)
            if terminated or truncated:
                term_seen = True
                break
        record("[4] step() returns float reward", PASS if isinstance(r_sample, (int, float, np.floating)) else FAIL, f"r={r_sample}")
        record("[4] info carries a success flag", PASS if succ_key else WARN, f"key={succ_key}, info_keys={list(info.keys())}")
        record("[5] episode terminates/truncates", PASS if term_seen else WARN, "hit terminal within 30 steps" if term_seen else "no terminal in 30 no-op steps (ok)")
    except Exception:
        record("[4/5] reward + termination", FAIL, "exception (see trace)")
        traceback.print_exc()

# ------------------------------------------------------------- [7] scripted grasp
def cube_xyz(obs):
    return np.asarray(obs["environment_state"]).reshape(-1)[:3]

def tcp_xyz(obs):
    return np.asarray(obs["agent_pos"]).reshape(-1)[-3:]

if env is not None:
    try:
        best_r = -1e9
        got_success = False
        for ep in range(6):
            obs, info = env.reset(seed=100 + ep)
            for t in range(env.spec.max_episode_steps or 100):
                cube, tcp = cube_xyz(obs), tcp_xyz(obs)
                d = cube - tcp
                act = np.zeros(env.action_space.shape, dtype=np.float32)
                horiz = np.linalg.norm(d[:2])
                if horiz > 0.03:                      # 1) center above cube
                    act[:2] = np.clip(d[:2] * 4.0, -0.05, 0.05)
                    act[2] = np.clip(d[2] * 2.0, -0.02, 0.03)
                    act[-1] = -1.0                    # keep open
                elif d[2] < -0.01:                    # 2) descend
                    act[2] = np.clip(d[2] * 4.0, -0.05, 0.0)
                    act[-1] = -1.0
                else:                                 # 3) close + lift
                    act[-1] = 1.0
                    act[2] = 0.04
                obs, rew, terminated, truncated, info = env.step(act)
                best_r = max(best_r, float(rew))
                if info.get("succeed") or info.get("is_success") or info.get("success"):
                    got_success = True
                if terminated or truncated:
                    break
            if got_success:
                break
        if got_success:
            record("[7] scripted reach-grasp-lift reaches success (reward fires)", PASS, f"best_reward={best_r:.3f}")
        else:
            record("[7] scripted grasp reached success", WARN, f"no grasp in 6 eps (best_reward={best_r:.3f}); reward machinery still exercised")
    except Exception:
        record("[7] scripted grasp", WARN, "exception (see trace)")
        traceback.print_exc()
    finally:
        env.close()

# ------------------------------------------------------------- [6] pixel render
try:
    penv, used = make(image_obs=True)
    obs, info = penv.reset(seed=0)
    ok = False
    if isinstance(obs, dict) and "pixels" in obs:
        px = obs["pixels"]
        shapes = {k: np.asarray(v).shape for k, v in px.items()}
        ok = any(tuple(s[-3:]) == (128, 128, 3) for s in shapes.values())
        record("[6] pixel obs front/wrist (128,128,3)", PASS if ok else WARN, str(shapes))
        # prove we can turn frames into a file (what the Colab video cell does)
        try:
            import imageio
            frame = np.asarray(next(iter(px.values())))
            out = os.path.join(os.path.dirname(__file__), "smoke_frame.png")
            imageio.imwrite(out, frame.astype(np.uint8))
            record("[6] wrote a rendered frame to disk", PASS, out)
        except Exception as e:
            record("[6] wrote a rendered frame to disk", WARN, f"imageio: {e}")
    else:
        record("[6] pixel obs present", WARN, f"obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    penv.close()
except Exception as e:
    # macOS offscreen GL is the documented-uncertain path; Colab/Modal use egl.
    record("[6] pixel rendering (headless GL)", WARN, f"{type(e).__name__}: {e} — expected-uncertain on macOS; Colab/Modal use MUJOCO_GL=egl")

# ------------------------------------------------------------------- summary
print("-" * 72)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
n_warn = sum(1 for _, s, _ in results if s == WARN)
n_pass = sum(1 for _, s, _ in results if s == PASS)
print(f"SUMMARY: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL")
sys.exit(1 if n_fail else 0)
